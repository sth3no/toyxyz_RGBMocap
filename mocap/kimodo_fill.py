"""Automatic fill worker: generate a hidden range with NVIDIA Kimodo between captured SMPL-X frames.

`plan` and `finish` are numpy only; torch and Kimodo are imported by `main`, which runs in the ExMR runtime with the
Kimodo overlay. The request and response arrays are the automatic-fill protocol of `take_join._auto`.
"""
import argparse
import os
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np

from . import soma_convert as sc

STAGE = 'take_join'
VERSION, CONTEXT, MAX_FRAMES, MAX_SIDE, FPS, STEPS, SEEDS = 3, 5, 290, 45, 30., 100, 1000
MODEL_KEY = 'kimodo-soma-rp-v1.2'
# Kimodo's guidance weights (text, constraints) with a description, as measured in step-5-gate.md; without one Kimodo's
# default guidance is used, as in version 2.
TEXT_WEIGHT, CONSTRAINT_WEIGHT = 4.0, 2.0
FEATURE = 4096  # numbers of Kimodo's text feature
SMOOTH, SMOOTH_MIN = 30, 5  # frames next to the range the smooth root is read from; fewest frames Kimodo's smoother handles
SIDES = dict(orient=3, pose=63, pelvis=3)
CONTACTS = [0, 1, 3, 4]  # Kimodo's 77-joint channels [L heel, L toe, L toe end, R heel, R toe, R toe end] -> [L heel, L toe, R heel, R toe]
HIPS = sc.SOMA30.index('RightLeg'), sc.SOMA30.index('LeftLeg')
INCOMPLETE = 'The automatic fill request is incomplete or malformed.'
WRONG_MODEL = 'The Kimodo model files are not the expected version. Get them again in Model Manager.'
NO_MEMORY = 'Not enough free video memory for automatic fill. Close other GPU work and try again.'


def _check(request):
    """The request as float64 arrays and plain numbers; ValueError with a sentence when it cannot be filled."""
    try:
        r = {f'{side}_{k}': np.asarray(request[f'{side}_{k}'], np.float64) for side in ('before', 'after') for k in SIDES}
        r.update(rest=np.asarray(request['rest'], np.float64), sole=np.asarray(request['sole'], np.float64))
        version, frames, seed, fps = (np.asarray(request[k]) for k in ('version', 'frames', 'seed', 'fps'))
        text, feature = np.asarray(request['text']), np.asarray(request['text_feature'])
        if text.shape != () or text.dtype.kind != 'U' or feature.shape != (FEATURE,) or feature.dtype.kind != 'f':raise ValueError
        r['text_feature'] = feature.astype(np.float32)
        if any(v.shape != () for v in (version, frames, seed, fps, r['sole'])) or r['rest'].shape != (22, 3):raise ValueError
        for side in ('before', 'after'):
            count = len(r[side+'_pelvis'])
            if not 1 <= count <= MAX_SIDE or any(r[f'{side}_{k}'].shape != (count, w) for k, w in SIDES.items()):raise ValueError
        version, frames, seed, fps = float(version), float(frames), float(seed), float(fps)
    except (KeyError, TypeError, ValueError, IndexError) as error:raise ValueError(INCOMPLETE) from error
    if version != VERSION:raise ValueError('The automatic fill request has an unsupported version.')
    if not all(np.isfinite(v).all() for v in r.values()) or not np.isfinite([frames, seed, fps]).all():
        raise ValueError('The automatic fill request holds values that are not numbers.')
    if frames != int(frames) or not 1 <= frames <= MAX_FRAMES:
        raise ValueError(f'Automatic fill supports hidden ranges of 1 to {MAX_FRAMES} frames.')
    if fps != FPS:raise ValueError('Automatic fill supports 30 fps motion only.')
    if seed != int(seed) or not 0 <= seed < SEEDS:raise ValueError(f'The automatic fill seed must be a whole number from 0 to {SEEDS-1}.')
    text = str(text[()])
    if text and not r['text_feature'].any():raise ValueError('The automatic fill request has a description without its text features.')
    r.update(frames=int(frames), seed=int(seed), sole=float(r['sole']), text=text);return r


def plan(request, smooth=None):
    """What Kimodo is asked for: the known SOMA frames of a window c0 + n + c1 around the hidden range.

    `smooth` maps a side's SOMA Hips path (F,3) to Kimodo's smooth root (Kimodo's `get_smooth_root_pos`), given the 30
    frames next to the range; without it, or for a side of fewer than 5 frames, the raw Hips path is used. Each side's floor is read under that side's given
    frames alone and joined by a straight line across the window, so a drifting capture height does not reach Kimodo.
    `path` is the smooth root of the hidden rows `path_rows` on a straight line at constant speed from the last known row
    before the range to the first after it: without it Kimodo tended to stand, then hurry.
    """
    r = _check(request);n = r['frames'];fit = sc.fit(r['rest'], r['sole']);sides = []
    for side in ('before', 'after'):
        G, J = sc.smplx_fk(r['rest'], r[side+'_orient'], r[side+'_pose'], r[side+'_pelvis']);floor = sc.floor_under(J, fit['toe'])
        local, hips = sc.to_soma(G, r[side+'_pelvis'], fit, floor);K, S = sc.soma_fk(local, hips)
        # As measured: the smooth root of each side from at most SMOOTH frames next to the range; Kimodo's smoother
        # needs at least 5 frames (fewer leave its coarse level without an acceleration term).
        near = hips[-SMOOTH:] if side == 'before' else hips[:SMOOTH]
        root = near if smooth is None or len(near) < SMOOTH_MIN else np.asarray(smooth(near), np.float64)
        sides.append(dict(J=J, K=K, S=S, root=root, floor=floor))
    b, a = sides;c0, c1 = min(CONTEXT, len(b['J'])), min(CONTEXT, len(a['J']));rows = c0+n+c1
    pick = lambda key: np.concatenate([b[key][len(b[key])-c0:], a[key][:c1]])
    S, root = pick('S'), pick('root');origin = np.array([root[0, 0], 0., root[0, 2]])
    # Kimodo's own heading: atan2(z, -x) of right hip minus left hip, at the first window frame.
    d = S[0, HIPS[0]]-S[0, HIPS[1]];sr = (root-origin)[:, [0, 2]]
    path = sr[c0-1]+(sr[c0]-sr[c0-1])*(np.arange(1, n+1)/(n+1))[:, None]
    return dict(context=(c0, c1), frames=rows, known=np.r_[0:c0, c0+n:rows], positions=S-origin, rotations=pick('K'),
                smooth_root=sr, path_rows=np.arange(c0, c0+n), path=path, heading=float(np.arctan2(d[2], -d[0])), origin=origin,
                floor=np.interp(np.arange(rows), [c0-1, c0+n], [b['floor'], a['floor']]), joints=pick('J'), fit=fit)


def finish(request, plan, output):
    """Response dict from Kimodo's output for the window: global_rot_mats (rows,J,3,3), root_positions (rows,3),
    foot_contacts (rows,6) and names (J,), the joint names of global_rot_mats."""
    r = _check(request);rows = plan['frames']
    K = np.asarray(output['global_rot_mats'], np.float64);hips = np.asarray(output['root_positions'], np.float64)+plan['origin']
    contacts = np.asarray(output['foot_contacts'])
    if K.shape[0] != rows or hips.shape != (rows, 3) or contacts.shape != (rows, 6):raise ValueError('Kimodo returned motion of an unexpected length.')
    orient, pose, pelvis = sc.to_smplx(K, hips, plan['fit'], plan['floor'], names=output['names'])
    J = sc.smplx_fk(r['rest'], orient, pose, pelvis)[1]
    # Per known frame the mean distance over the 22 joints; the seam is the last known frame before the range and the first after it.
    known = np.linalg.norm(J[plan['known']]-plan['joints'], axis=2).mean(axis=1)*100;c0 = plan['context'][0]
    return dict(version=np.int64(VERSION), context=np.array(plan['context'], np.int64), orient=orient, pose=pose, pelvis=pelvis,
                contacts=contacts[:, CONTACTS].astype(bool), end_pose_cm=np.float64(known.mean()), seam_cm=known[[c0-1, c0]])


class NoText:
    """Text encoder stand-in for a range without a description: Kimodo zeroes an empty prompt's features."""
    feature = None

    def __call__(self, texts):
        import torch
        texts = [texts] if isinstance(texts, str) else texts
        out = torch.zeros(len(texts), 1, FEATURE)
        if self.feature is not None:out[[bool(t.strip()) for t in texts]] = torch.from_numpy(self.feature)
        return out, [1]*len(texts)

    def to(self, device):return self

    def eval(self):return self


class GivenText(NoText):
    """Text encoder stand-in returning the description's feature, read beforehand by the text worker, for every non-empty
    prompt; Kimodo's interface: (features (batch, 1, 4096), [1] * batch)."""
    def __init__(self, feature):self.feature = np.asarray(feature, np.float32)


def conditioning(r):
    """(prompt, text encoder stand-in, guidance keywords of Kimodo's call) of a checked request. Without a description the
    call is version 2's: an empty prompt and Kimodo's default guidance. With one, the prompt must be the description:
    Kimodo zeroes the feature of an empty prompt."""
    if not r['text']:return '', NoText(), {}
    return r['text'], GivenText(r['text_feature']), dict(cfg_weight=[TEXT_WEIGHT, CONSTRAINT_WEIGHT])


def _save(response, path):
    tmp = path.with_name(path.name+'.tmp')
    try:
        with open(tmp, 'wb') as handle:np.savez(handle, **response)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True);raise


def _generate(request, seed, emit):
    from .compute_policy import configure_worker_cpu, gpu_memory_fraction
    from .kimodo_runtime import MODEL_FOLDER, PARAMETERS
    from .paths import MODELS
    threads = configure_worker_cpu(STAGE)
    os.environ.update(CHECKPOINT_DIR=str(MODELS/MODEL_FOLDER), HF_HUB_OFFLINE='1')
    # The real module imports transformers and peft, which this worker does not need: a description's vector is read
    # beforehand by the text worker and given to Kimodo by GivenText.
    stand_in = types.ModuleType('kimodo.model.llm2vec');stand_in.LLM2VecEncoder = None;sys.modules['kimodo.model.llm2vec'] = stand_in
    import torch
    from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet
    from kimodo.model.load_model import load_model
    from kimodo.motion_rep.smooth_root import get_smooth_root_pos
    if not torch.cuda.is_available():raise RuntimeError('Automatic fill requires a CUDA GPU.')
    torch.set_num_threads(threads)
    smooth = lambda root: get_smooth_root_pos(torch.tensor(root, dtype=torch.float32)[None])[0].double().numpy()
    p = plan(request, smooth);prompt, encoder, guidance = conditioning(_check(request))
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction(*torch.cuda.mem_get_info()))
    emit('progress', stage=STAGE, current=0, total=STEPS, message='Automatic fill: loading the Kimodo model')
    started = time.monotonic();model = load_model(MODEL_KEY, device='cuda', text_encoder=encoder)
    skeleton = model.skeleton;neutral = torch.as_tensor(skeleton.neutral_joints).detach().double().cpu().numpy()
    parents = torch.as_tensor(skeleton.joint_parents).detach().cpu().numpy()
    if sum(x.numel() for x in model.parameters()) != PARAMETERS or list(skeleton.bone_order_names) != sc.SOMA30 or \
            not np.array_equal(parents, sc.SOMA30_PARENTS) or \
            neutral.shape != sc.SOMA_NEUTRAL.shape or np.abs(neutral-sc.SOMA_NEUTRAL).max() > 1e-5:raise RuntimeError(WRONG_MODEL)
    loaded = time.monotonic()-started
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device='cuda')
    fb = FullBodyConstraintSet(skeleton, torch.tensor(p['known']), T(p['positions']), T(p['rotations']), T(p['smooth_root']))
    root2d = Root2DConstraintSet(skeleton, torch.tensor(p['path_rows']), T(p['path']))

    def steps(indices):
        for k, i in enumerate(indices):
            if k % 10 == 0:emit('progress', stage=STAGE, current=k, total=len(indices), message=f'Automatic fill: generating, step {k+1} of {len(indices)}')
            yield i
        emit('progress', stage=STAGE, current=len(indices), total=len(indices), message='Automatic fill: generated')
    torch.manual_seed(seed);started = time.monotonic()
    out = model(prompt, p['frames'], num_denoising_steps=STEPS, constraint_lst=[[fb, root2d]], first_heading_angle=p['heading'],
                post_processing=False, num_samples=1, progress_bar=steps, **guidance)
    out = {k: out[k][0].detach().cpu().numpy() for k in ('global_rot_mats', 'root_positions', 'foot_contacts')}
    generated, peak = time.monotonic()-started, torch.cuda.max_memory_allocated()/2**30
    response = finish(request, p, dict(out, names=list(model.output_skeleton.bone_order_names)));before, after = response['seam_cm']
    emit('info', stage=STAGE, message=f"Automatic fill: model loaded in {loaded:.1f} seconds, {p['frames']} frames generated in {generated:.1f} seconds, "
                                      f"known poses kept to {float(response['end_pose_cm']):.2f} cm on average, {before:.1f} cm before and {after:.1f} cm "
                                      f"after the range; peak video memory {peak:.2f} GiB.")
    return response


def main(argv, emit):
    parser = argparse.ArgumentParser(prog='kimodo_worker_boot.py');parser.add_argument('--request', required=True);parser.add_argument('--response', required=True)
    args = parser.parse_args(argv)
    try:
        try:
            with np.load(args.request, allow_pickle=False) as z:request = {k: z[k] for k in z.files}
        except (OSError, ValueError, zipfile.BadZipFile) as error:raise ValueError(f'Could not read the automatic fill request. {error}') from error
        _save(_generate(request, _check(request)['seed'], emit), Path(args.response))
    except Exception as error:
        message = str(error) or type(error).__name__
        if type(error).__name__ == 'OutOfMemoryError' or 'out of memory' in message.lower():message = NO_MEMORY
        emit('failed', stage=STAGE, message=message);return 1
    return 0
