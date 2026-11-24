"""Import a finished take from another video's project as a further take of a person.

Runs in its own process like take_join. The two videos have different cameras, so the take is placed by continuing the
person's last frame: turned about +Y to their last heading, moved horizontally to their last pelvis position, with its
own height above its own floor. Its camera-space motion is rewritten through the person's last-frame camera pairing,
so the join's seam alignment is the identity and the existing Hidden ranges methods fill the gap.
"""
from pathlib import Path
import json
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from . import take_import_check as check
from .gap_fill import rigid_params
from .take_join import Refused

STAGE = 'take_import'


def heading(orient):
    """Facing direction about +Y of one root orientation: the body's forward axis +Z projected on the floor."""
    f = Rotation.from_rotvec(np.asarray(orient, np.float64).reshape(3)).apply([0., 0., 1.]);return float(np.arctan2(f[0], f[2]))


class Body:
    """Rest pelvis and leg length of a shape from SMPL-X arrays (J_regressor, v_template, shapedirs)."""
    def __init__(self, data):
        regressor = np.asarray(data['J_regressor'], np.float64)[0]
        self.template, self.shapes = regressor@np.asarray(data['v_template'], np.float64), np.einsum('v,vci->ci', regressor, np.asarray(data['shapedirs'], np.float64)[..., :10])
        self.vertices, self.vertex_shapes = np.asarray(data['v_template'], np.float64)[:, 1], np.asarray(data['shapedirs'], np.float64)[:, 1, :10]
    def pelvis(self, betas):return self.template+self.shapes@np.asarray(betas, np.float64)[:10]
    def leg(self, betas):
        b = np.asarray(betas, np.float64)[:10];return float(self.pelvis(b)[1]-(self.vertices+self.vertex_shapes@b).min())
    def lift(self, own, receiver):
        """How much higher the pelvis must be with the receiver's shape so the feet stay where the take's own shape had them."""
        return self.leg(receiver)-self.leg(own)


def place(params, pelvis, world, incam, receiver_pelvis, betas, lift=0.):
    """(world params per result kind, camera-space params) of the imported take in the receiver's last take's world.

    `params` are the imported take's world params per kind with rest pelvis `pelvis`; `world` and `incam` the receiver's
    last take's `initial` world and camera-space params; `betas` the receiver's shape; `lift` metres added to the height."""
    p0, I = np.asarray(receiver_pelvis, np.float64), np.eye(4)
    shaped = {}
    for kind, p in params.items():
        n = len(p['transl']);out = rigid_params(p, I, pelvis, p0)  # the receiver's shape, the pelvis joint where it was
        out['transl'][:, 1] += lift;out['betas'] = np.repeat(np.asarray(betas, np.float64).reshape(1, -1), n, 0);shaped[kind] = out
    first = shaped['initial'];row = lambda v: np.asarray(v, np.float64).reshape(len(v), -1)[-1]
    turn = heading(row(world['global_orient']))-heading(first['global_orient'][0])
    T = np.eye(4);T[:3, :3] = Rotation.from_rotvec([0., turn, 0.]).as_matrix()
    T[:3, 3] = row(world['transl'])+p0-T[:3, :3]@(first['transl'][0]+p0);T[1, 3] = 0.  # horizontal only: its own height
    placed = {kind: rigid_params(p, T, p0, p0) for kind, p in shaped.items()}
    from .take_join import pair
    E = pair(row(world['global_orient']), row(incam['global_orient']), row(world['transl']), row(incam['transl']), p0)
    return placed, rigid_params(placed['initial'], E, p0, p0)


def emit(kind, **data):
    from .worker import PREFIX
    print(PREFIX+json.dumps({'type': kind, **data}, ensure_ascii=False), flush=True)


def _numpy(params):return {k: v.detach().cpu().double().numpy() for k, v in params.items()}


def _read(folder, kinds):
    """The result record of a take per kind, as saved, and its numpy params."""
    import torch
    from .motion_tools import RESULTS
    records = {k: torch.load(folder/RESULTS[k][0], map_location='cpu', weights_only=False) for k in kinds}
    return records, {k: _numpy(r[RESULTS[k][1]]) for k, r in records.items()}


def _person(root, ident):
    from .people_store import load_people
    try:state = load_people(root)
    except (OSError, ValueError, KeyError) as error:raise Refused(str(error)) from None
    person = next((p for p in (state or {}).get('people', []) if p['id'] == ident), None)
    if person is None:raise Refused('Person no longer exists. Run Find person again.')
    return person


def _source_person(source):
    """Name of the person a take of another project belongs to, for the record; None when it cannot be read."""
    from .people_store import load_people, root_for, safe_child
    try:
        root = root_for(source)
        for p in load_people(root)['people']:
            if any(safe_child(root, t['project']) == source for t in p.get('takes', [])+(p.get('joins') or [])):
                return str(p.get('name') or f"Person {p['id']}")
    except (OSError, ValueError, KeyError, TypeError):pass
    return None


HANDS = ('left_hand_pose', 'right_hand_pose')


def run(root, ident, source, gap, result):
    """Writes the imported take after the person's last take and records it in people.json; returns its folder."""
    import torch
    from .motion_tools import RESULTS, base_result, has_handflow
    from .paths import MODELS
    from .people_store import load_people, safe_child, save_people
    from .person_tracking import _link
    from .project_storage import inherit_storage
    from .storage import atomic_json
    from .take_join import RANGE_KEYS, _finger_receipt, _hand_receipt, _pipeline, _save, _stamp, _tensors, people_clip
    person = _person(root, ident)
    reason = check.refusal(root, person, source, gap, result)
    if reason:raise Refused(reason)
    model_file = MODELS/'body_models/smplx/SMPLX_NEUTRAL.npz'
    if not model_file.is_file():raise Refused('Import SMPLX_NEUTRAL.npz to import takes.')
    with np.load(model_file, allow_pickle=False) as data:shape = Body(data)
    kinds = check.kinds(result);last = check.last_take(person);lead = safe_child(root, last['project'])
    emit('progress', stage=STAGE, current=0, total=1, message='Importing the take')
    hands = {k: _finger_receipt(source, k) for k in kinds if has_handflow(k)}
    records, src = _read(source, kinds);own, mine = _read(lead, ['initial'])
    n = len(src['initial']['transl']);logits = records['initial']['net_outputs']['static_conf_logits'][0].double().numpy()
    if json.loads((source/'project.json').read_text(encoding='utf-8')).get('frames') != n or len(logits) != n or \
            any(len(v) != n for p in src.values() for v in p.values()):raise Refused(check.NOT_FINISHED)
    betas, own_betas = mine['initial']['betas'][0], src['initial']['betas'][0]
    params, incam = place(src, shape.pelvis(own_betas), mine['initial'], _numpy(own['initial']['smpl_params_incam']), shape.pelvis(betas),
                          betas, shape.lift(own_betas, betas))
    first = last['end']+gap;end = first+n
    target = safe_child(root, person['preview'])/'takes'/f'{first:09d}_{end:09d}_import_v1'
    target.mkdir(parents=True, exist_ok=True)
    clip = people_clip(root, first, end);shared = json.loads((clip/'project.json').read_text(encoding='utf-8'))
    for name in ('input.mp4', 'preview.mp4', 'preview.json'):
        if (clip/name).is_file():_link(clip/name, target/name)
    with np.load(lead/'tracking.npz', allow_pickle=False) as tracking:K = tracking['K']
    temporary = target/'tracking.partial.npz'  # Not seen by this scene's tracker: no box, nothing observed.
    np.savez_compressed(temporary, boxes=np.full((n, 4), np.nan, np.float32), observed=np.zeros(n, bool), K=K);os.replace(temporary, target/'tracking.npz')
    _save({'K': torch.from_numpy(np.asarray(K))}, target/'camera.pt')
    receipt = lead/'exmr_camera_sources.json'  # The receiver's camera, read by scene_camera and result_is_current; same folder depth.
    if receipt.is_file():atomic_json(target/receipt.name, json.loads(receipt.read_text(encoding='utf-8')))
    state = json.loads((lead/'project.json').read_text(encoding='utf-8'))
    for kind in kinds:
        name, key = RESULTS[kind];payload = {key: _tensors(params[kind])}
        if has_handflow(kind):payload.update(source=records[kind]['source'], method=records[kind]['method'])
        if kind == 'initial':
            payload.update(smpl_params_incam=_tensors(incam), source_model=own['initial'].get('source_model') or check._model(state),
                           net_outputs={'static_conf_logits': torch.from_numpy(logits.astype(np.float32))[None]})
        _save(payload, target/name)
    for kind in kinds:
        names = ('tracking.npz', 'hmr4d_results.pt', RESULTS[kind][0], *(('exmr_camera_sources.json',) if receipt.is_file() else ()))
        atomic_json(target/f'{kind}_sources.json', {x: _stamp(target/x) for x in dict.fromkeys(names)})
    for k in (*RANGE_KEYS, 'joined'):state.pop(k, None)
    state.update({k: shared[k] for k in RANGE_KEYS if k in shared})
    read = list(dict.fromkeys(['project.json', 'hmr4d_results.pt', *(RESULTS[k][0] for k in kinds)]))
    state.update(frames=n, stages={}, person_frames=[first, end], pipeline=_pipeline(check._model(state), base_result(result)),
                 imported=dict(project=str(source), person=_source_person(source), files={x: _stamp(source/x) for x in read}))
    inherit_storage(lead, target, state)
    atomic_json(target/'project.json', state)
    for kind, r in hands.items():_hand_receipt(target, kind, [dict(hand={kind: dict(receipt=r)})], n, state.get('fps', 30))
    relative = target.relative_to(Path(root).resolve()).as_posix()
    people = load_people(root);recorded = next((p for p in people['people'] if p['id'] == ident), None)  # The editor may have saved other fields meanwhile.
    if recorded is None:raise RuntimeError('Person no longer exists; the imported take was not recorded. Run Find person again.')
    recorded['takes'] = [t for t in recorded.get('takes', []) if t['project'] != relative]+[dict(project=relative, first=first, end=end)]
    save_people(root, people)
    if any(k in mine['initial'] for k in HANDS) and not any(k in src['initial'] for k in HANDS):emit('info', stage=STAGE, message=check.NO_FINGERS)
    return target


def main(argv=None):
    import argparse
    import time
    import traceback
    parser = argparse.ArgumentParser(prog='take_import_boot.py', description='Import a finished take from another project as a further take of one person.')
    parser.add_argument('--project', required=True, help='scene root folder')
    parser.add_argument('--person-id', type=int, required=True)
    parser.add_argument('--source', required=True, help='folder of the finished take in the other project')
    parser.add_argument('--gap', type=int, required=True, help="frames between the person's last take and the imported take")
    parser.add_argument('--result', default='initial', help='the result in use; the imported take provides its uncorrected results')
    args = parser.parse_args(argv);started = time.monotonic();root = Path(args.project).resolve()
    try:
        target = run(root, args.person_id, Path(args.source).resolve(), args.gap, args.result)
    except Exception as error:
        if not isinstance(error, Refused):traceback.print_exc(file=sys.stderr)
        emit('failed', stage=STAGE, message=str(error) or type(error).__name__);return 1
    emit('take_imported', project=str(root), person_id=args.person_id, projects=[str(target)])
    emit('complete', stage=STAGE, output=str(target), cached=False, elapsed=time.monotonic()-started)
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
