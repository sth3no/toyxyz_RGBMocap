"""Join a person's chain of takes into one new take with filled hidden ranges.

Runs in its own process. `assemble` is pure numpy; Torch is imported only for file IO.
"""
from pathlib import Path
import json
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .gap_fill import COPIED, INTERPOLATED, POSE_KEYS, copy_fill, fill_logits, mirror_logits, mirror_params, rigid_params, smooth_fill

STAGE = 'take_join'


def pair(world_orient, cam_orient, world_transl, cam_transl, pelvis):
    """World-to-camera of one frame; SMPL rotates about the rest pelvis (as scene_camera.paired_camera)."""
    rotation = lambda v: Rotation.from_rotvec(np.asarray(v, np.float64).reshape(3)).as_matrix()
    T = np.eye(4);T[:3, :3] = rotation(cam_orient)@rotation(world_orient).T;p = np.asarray(pelvis, np.float64)
    T[:3, 3] = np.asarray(cam_transl, np.float64).reshape(3)+p-T[:3, :3]@(np.asarray(world_transl, np.float64).reshape(3)+p)
    return T


def upright(full, pelvis):
    """Heading of a seam transform only (twist about +Y); the pelvis keeps the x, z of `full` and its own height."""
    x, y, z, w = Rotation.from_matrix(full[:3, :3]).as_quat()
    T = np.eye(4);T[:3, :3] = Rotation.from_rotvec([0., 2*np.arctan2(y, w), 0.]).as_matrix();p = np.asarray(pelvis, np.float64)
    T[[0, 2], 3] = (full[:3, :3]@p+full[:3, 3]-T[:3, :3]@p)[[0, 2]]
    return T


def _linear(a, b, frames):
    t = (np.arange(1, frames+1)/(frames+1))[:, None];return a+(b-a)*t


def _stitch(parts, fills, sources, gaps, base, total):
    out = np.empty((total, *np.shape(parts[0])[1:]), np.float64)
    for s, p in zip(sources, parts):out[s['first']-base:s['end']-base] = p
    for g, f in zip(gaps, fills):out[g['first']-base:g['end']-base] = f
    return out


def _keys(parts):
    # A pose key missing from any take (e.g. GVHMR hands) is dropped rather than invented.
    return [k for k in POSE_KEYS if all(k in p for p in parts)]+['transl']


def _rows(p, keys):return {k: np.asarray(p[k], np.float64).reshape(len(p['transl']), -1) for k in keys}


# A gap fill gets the joined neighbours a, b (params per kind, incam, logits as float64 rows in the joined world),
# the gap record and the joined pelvis; it returns the gap rows of each and its mask value.
def _smooth(a, b, g, pelvis):
    n = g['end']-g['first']
    fill = lambda x, y: smooth_fill({k: v[-2:] for k, v in x.items()}, {k: v[:2] for k, v in y.items()}, n)
    return dict(params={kind: fill(p, b['params'][kind]) for kind, p in a['params'].items()}, incam=fill(a['incam'], b['incam']),
                logits=fill_logits(a['logits'][-1], b['logits'][0], n), mask=INTERPOLATED)


def _seam(part, row, pelvis):
    g, c = part['params']['initial'], part['incam']
    return pair(g['global_orient'][row], c['global_orient'][row], g['transl'][row], c['transl'][row], pelvis)


def _copy(a, b, g, pelvis):
    n, d = g['end']-g['first'], g['donor'];dp = np.asarray(d['pelvis'], np.float64)
    rows = lambda p: {k: np.asarray(v, np.float64).reshape(n+2, -1) for k, v in p.items()}
    own, cam = {kind: rows(p) for kind, p in d['params'].items()}, rows(d['incam'])
    # The donor's row 0 is moved onto the camera pairing of the last joined frame before the gap, upright as at a seam.
    W0, W1 = _seam(a, -1, pelvis), _seam(b, 0, pelvis);dg = own['initial']
    Td = upright(np.linalg.inv(W0)@pair(dg['global_orient'][0], cam['global_orient'][0], dg['transl'][0], cam['transl'][0], dp), dg['transl'][0]+dp)
    at = lambda p, sign: dict(p, transl=p['transl']+sign*pelvis)  # copy_fill and mirror_params work on pelvis positions
    params = {}
    for kind, p in a['params'].items():
        donor = at(rigid_params(own[kind], Td, dp, pelvis), 1)
        if g['mirror']:donor = mirror_params(donor)
        params[kind] = at(copy_fill(at(p, 1), at(b['params'][kind], 1), donor, [k for k in p if k != 'transl']), -1)
    # Camera space: the copied world motion seen through a camera pairing that moves from one seam to the other.
    s = np.arange(1, n+1)/(n+1);turn = Slerp([0., 1.], Rotation.from_matrix([W0[:3, :3], W1[:3, :3]]))(s);G = params['initial']
    # A camera-space pose key that the world result lacks is interpolated.
    rest = smooth_fill({k: v[-2:] for k, v in a['incam'].items()}, {k: v[:2] for k, v in b['incam'].items()}, n) if any(k not in G for k in a['incam']) else {}
    incam = {k: G[k] if k in G else rest[k] for k in a['incam']}
    incam.update(global_orient=(turn*Rotation.from_rotvec(G['global_orient'])).as_rotvec(),
                 transl=turn.apply(G['transl']+pelvis)+W0[:3, 3]+(W1[:3, 3]-W0[:3, 3])*s[:, None]-pelvis)
    logits = np.asarray(d['logits'], np.float64)[1:-1]
    return dict(params=params, incam=incam, logits=mirror_logits(logits) if g['mirror'] else logits, mask=COPIED)


FILLS = {'smooth': _smooth, 'copy': _copy}


def assemble(sources, gaps):
    first, base = sources[0], sources[0]['first']; total = sources[-1]['end']-base
    if len(gaps) != len(sources)-1:raise ValueError('Each pair of joined takes needs one hidden range')
    for a, g, b in zip(sources, gaps, sources[1:]):
        if (g['first'], g['end']) != (a['end'], b['first']) or g['end'] <= g['first']:raise ValueError('Hidden range does not lie between its takes')
        if g['method'] not in FILLS:raise ValueError(f"Unsupported fill method: {g['method']}")
        if g['method'] == 'copy':
            d, n = g['donor'], g['end']-g['first']+2
            if not set(first['params']) <= set(d['params']) or len(d['logits']) != n or len(d['incam']['transl']) != n or \
                    any(len(d['params'][k]['transl']) != n for k in first['params']):raise ValueError('Donor arrays do not match the hidden range')
    for s in sources:
        n = s['end']-s['first']
        if len(s['logits']) != n or len(s['boxes']) != n or len(s['observed']) != n or len(s['incam']['transl']) != n or \
                any(len(p['transl']) != n for p in s['params'].values()):raise ValueError('Take arrays do not match its frame range')
    # Each take's world drifts against the fixed camera, so a take is anchored at its seam: its first frame
    # takes the heading and floor position of the previous take's last camera pairing. The tilt of one frame
    # pair is gravity noise, so it is dropped and every take keeps its own up axis and height.
    pelvis, joined = first['pelvis'], [(first['params'], first['incam'])]
    for s in sources[1:]:
        g, c = joined[-1][0]['initial'], joined[-1][1]
        last = lambda v: np.asarray(v, np.float64).reshape(len(v), -1)[-1]
        end = pair(last(g['global_orient']), last(c['global_orient']), last(g['transl']), last(c['transl']), pelvis)
        a, b = s['params']['initial'], s['incam']
        start = pair(np.asarray(a['global_orient'])[0], np.asarray(b['global_orient'])[0], np.asarray(a['transl'])[0], np.asarray(b['transl'])[0], s['pelvis'])
        T = upright(np.linalg.inv(end)@start, np.asarray(a['transl'], np.float64).reshape(-1, 3)[0]+s['pelvis'])
        # Camera space is shared by every take; only the replaced shape moves the pelvis.
        joined.append(({k: rigid_params(p, T, s['pelvis'], pelvis) for k, p in s['params'].items()},
                       rigid_params(s['incam'], np.eye(4), s['pelvis'], pelvis)))
    keys = {kind: _keys([j[0][kind] for j in joined]) for kind in first['params']};ikeys = _keys([j[1] for j in joined])
    parts = [dict(params={kind: _rows(j[0][kind], k) for kind, k in keys.items()}, incam=_rows(j[1], ikeys), logits=np.asarray(s['logits'], np.float64))
             for j, s in zip(joined, sources)]
    fills = [FILLS[g['method']](parts[i], parts[i+1], g, np.asarray(pelvis, np.float64)) for i, g in enumerate(gaps)]
    put = lambda get: _stitch([get(p) for p in parts], [get(f) for f in fills], sources, gaps, base, total)
    params = {kind: {k: put(lambda x: x['params'][kind][k]) for k in ks} for kind, ks in keys.items()}
    for kind, own in first['params'].items():params[kind]['betas'] = np.repeat(np.asarray(own['betas'], np.float64)[:1], total, 0)
    incam = {k: put(lambda x: x['incam'][k]) for k in ikeys}
    incam['betas'] = np.repeat(np.asarray(first['incam']['betas'], np.float64)[:1], total, 0)
    logits = put(lambda x: x['logits'])
    boxes = _stitch([np.asarray(s['boxes'], np.float64) for s in sources], [_linear(np.asarray(a['boxes'], np.float64)[-1], np.asarray(b['boxes'], np.float64)[0],
                    g['end']-g['first']) for a, g, b in zip(sources, gaps, sources[1:])], sources, gaps, base, total)
    mask = np.zeros(total, np.uint8); observed = np.zeros(total, bool)
    for s in sources:observed[s['first']-base:s['end']-base] = np.asarray(s['observed'], bool)
    for g, f in zip(gaps, fills):mask[g['first']-base:g['end']-base] = f['mask']
    return dict(params=params, incam=incam, logits=logits, mask=mask, boxes=boxes, observed=observed)


class Refused(ValueError):
    """A deliberate refusal with a message for the user; nothing has been written."""


UNSUPPORTED = 'Hidden ranges support the initial and FootMR results in this version; select one of them.'
RANGE_KEYS = ('start', 'end', 'frames', 'timeline', 'original_times', 'duration')


def emit(kind, **data):
    from .worker import PREFIX
    print(PREFIX+json.dumps({'type': kind, **data}, ensure_ascii=False), flush=True)


def _stamp(path):
    s = Path(path).stat();return [s.st_size, s.st_mtime_ns]


def _save(value, target):
    import torch
    temporary = target.with_name(target.name+'.tmp');torch.save(value, temporary);os.replace(temporary, target)


def _tensors(params):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v, np.float32)) for k, v in params.items()}


def _check(root, ident, result):
    """Every cheap precondition, before anything is read in bulk or written."""
    from . import hidden_ranges
    from .exmr_options import saved
    from .motion_tools import RESULTS, base_result, has_handflow, require_supported_project, result_is_current, result_sources
    from .paths import MODELS
    from .people_store import load_people, safe_child
    if not (root/'project.json').is_file() or not (root/'people.json').is_file():raise Refused('Open a scene with people before filling hidden ranges.')
    try:
        model = require_supported_project(root).key
        static = model == 'exmr' and saved(root)['static_camera']
        state = load_people(root)
    except (OSError, ValueError, KeyError) as error:raise Refused(str(error)) from None
    if model not in ('gvhmr', 'exmr'):raise Refused('Hidden ranges can be filled only in GVHMR and ExMR scenes.')
    if model == 'exmr' and not static:raise Refused('Hidden ranges need a fixed camera. Turn on Static camera in the ExMR settings.')
    if result not in RESULTS:raise Refused(f'Unknown motion result: {result}')
    kinds = result_sources(base_result(result))
    if has_handflow(result) or any(k.endswith('refined') for k in kinds):raise Refused(UNSUPPORTED)
    people = state['people'];person = next((p for p in people if p['id'] == ident), None)
    if person is None:raise Refused('Person no longer exists. Run Find person again.')
    choices = person.get('gap_fill') or {}
    chains = hidden_ranges.chains(person.get('takes', []), choices, people, ident)
    if not chains:raise Refused('No hidden range is selected for filling.')
    body = MODELS/'body_models/smplx/SMPLX_NEUTRAL.npz'
    if not body.is_file():raise Refused('Import SMPLX_NEUTRAL.npz to join takes.')
    names = list(dict.fromkeys(['tracking.npz', *(RESULTS[k][0] for k in kinds)]))
    rates = set()
    for take in (t for chain in chains for t in chain):
        folder = safe_child(root, take['project'])
        try:state = json.loads((folder/'project.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):state = None
        if not isinstance(state, dict):raise Refused(f'Take {folder.name} cannot be read. Prepare the person motion again.')
        if state.get('frames') != take['end']-take['first']:raise Refused(f'Take {folder.name} does not match its frame range. Prepare the person motion again.')
        rates.add(state.get('fps'))
        for name in names:
            if not (folder/name).is_file():raise Refused(f'Take {folder.name} has no {name}. Reconstruct it first.')
        for kind in kinds:
            if not result_is_current(folder, kind):raise Refused(f'The {kind} result of take {folder.name} is out of date. Reconstruct it first.')
    if len(rates) != 1:raise Refused('The takes to join have different frame rates.')
    gaps = []
    for chain in chains:
        gaps.append([])
        for a, b in zip(chain, chain[1:]):
            g = dict(first=a['end'], end=b['first']);method, params = hidden_ranges.choice_params(choices[hidden_ranges.gap_key(g)])
            gaps[-1].append(dict(g, method=method, params=params))
            if method != 'copy':continue
            donor = next(p for p in people if p['id'] == params['person'])
            take = gaps[-1][-1]['take'] = hidden_ranges.donor_take(donor['takes'], g, params['offset'])
            try:
                folder = safe_child(root, take['project']);manifest = json.loads((folder/'project.json').read_text(encoding='utf-8'))
                ready = manifest.get('frames') == take['end']-take['first'] and {manifest.get('fps')} == rates and \
                    all((folder/n).is_file() for n in names[1:]) and all(result_is_current(folder, k) for k in kinds)
            except (OSError, ValueError):ready = False
            if not ready:raise Refused(f"{donor.get('name') or 'Person '+str(donor['id'])}'s take is not ready to be copied from. Reconstruct and prepare it first.")
    return model, person, chains, gaps, kinds, names, body


def _pipeline(model, result):
    from .motion_tools import body_result
    return dict(model=model, result=result, htd_refine=False, footmr=body_result(result).startswith('footmr_'), handflow=False)


def _record(g):return dict(first=g['first'], end=g['end'], method=g['method'], params=g['params'])


def _donors(gaps, files):
    """fill.json records of the donor takes of a chain's copy gaps; `files(take)` gives the stamps of what is read."""
    out = {}
    for g in gaps:
        if g['method'] == 'copy' and g['take']['project'] not in out:
            t = g['take'];out[t['project']] = dict(person=g['params']['person'], project=t['project'], first=t['first'], end=t['end'], files=files(t))
    return list(out.values())


def _donor_rows(take, g):
    """The n+2 donor rows of a copy gap: scene frames gap.first-1+offset ... gap.end+offset."""
    lo = g['first']-1+g['params']['offset']-take['first'];cut = lambda v: v[lo:lo+g['end']-g['first']+2]
    return dict(pelvis=take['pelvis'], params={k: {n: cut(v) for n, v in p.items()} for k, p in take['params'].items()},
                incam={n: cut(v) for n, v in take['incam'].items()}, logits=cut(take['logits']))


def _unchanged(root, person, chain, gaps, donors, kinds, pipeline):
    """The folder of this chain's recorded join when it is current and holds what would be written, else None."""
    from .motion_tools import result_is_current
    from .people_store import join_fill, safe_child
    folder = safe_child(root, person['preview'])/'takes'/f"{chain[0]['first']:09d}_{chain[-1]['end']:09d}_fill_v1"
    for join in person.get('joins') or []:
        if join['project'] != folder.relative_to(root).as_posix() or join.get('sources') != [t['project'] for t in chain]:continue
        fill = join_fill(root, person, join)
        if fill is None or fill.get('kinds') != kinds or fill['gaps'] != [_record(g) for g in gaps] or (fill.get('donors') or []) != donors:return None
        try:written = json.loads((folder/'project.json').read_text(encoding='utf-8')).get('pipeline')
        except (OSError, ValueError):return None
        return folder if written == pipeline and all(result_is_current(folder, k) for k in kinds) else None
    return None


def _load(root, take, kinds, names, pelvis):
    import torch
    from .motion_tools import RESULTS
    from .people_store import safe_child
    folder = safe_child(root, take['project']);files = {n: _stamp(folder/n) for n in names}
    numpy = lambda params: {k: v.detach().cpu().double().numpy() for k, v in params.items()};boxes = observed = K = None
    if 'tracking.npz' in names:  # Not read from a donor take.
        with np.load(folder/'tracking.npz', allow_pickle=False) as tracking:
            boxes, observed, K = tracking['boxes'], tracking['observed'], tracking['K']
    params = {}
    for kind in kinds:
        record = torch.load(folder/RESULTS[kind][0], map_location='cpu', weights_only=False)
        params[kind] = numpy(record[RESULTS[kind][1]])
        if kind == 'initial':
            incam, model = numpy(record['smpl_params_incam']), record.get('source_model')
            logits = record['net_outputs']['static_conf_logits'][0].double().numpy()
        del record
    if {n: _stamp(folder/n) for n in names} != files:raise Refused(f'Take {folder.name} changed while it was read; try again.')
    return dict(first=take['first'], end=take['end'], project=take['project'], folder=folder, files=files, K=K,
                pelvis=pelvis(params['initial']['betas'][0]),
                params=params, incam=incam, logits=logits, boxes=boxes, observed=observed, source_model=model)


def _other_joins(person, join):
    """Recorded joins that stay beside the new one. Takes prepared from different trims may share frames, so a join
    over the same frames is replaced only when it uses one of the new join's takes, its folder, or a take that is gone."""
    live = {t['project'] for t in person.get('takes', [])};mine = set(join['sources'])
    def apart(j):
        if not (j.get('first', join['end']) < join['end'] and join['first'] < j.get('end', join['first'])):return True
        sources = j.get('sources')
        return (j['project'] != join['project'] and isinstance(sources, list) and bool(sources)
                and all(isinstance(s, str) for s in sources) and mine.isdisjoint(sources) and live.issuperset(sources))
    return [j for j in person.get('joins') or [] if apart(j)]


def _write(root, ident, sources, gaps, donors, joined, kinds, base, pipeline):
    import torch
    from .motion_tools import RESULTS
    from .people_store import join_current, load_people, safe_child, save_people
    from .person_tracking import _link
    from .project_storage import inherit_storage
    from .storage import atomic_json
    from .video import prepare_people_clip
    first, end, lead = sources[0]['first'], sources[-1]['end'], sources[0]
    person = next((p for p in load_people(root)['people'] if p['id'] == ident), None)
    if person is None:raise Refused('Person no longer exists. Run Find person again.')
    target = safe_child(root, person['preview'])/'takes'/f'{first:09d}_{end:09d}_fill_v1'
    target.mkdir(parents=True, exist_ok=True);pending = target/'fill_pending.json'
    # The marker stays after a failure, so a half-written folder is never current.
    atomic_json(pending, dict(version=1, person=ident, first=first, end=end))
    clip = prepare_people_clip(root, first, end);shared = json.loads((clip/'project.json').read_text(encoding='utf-8'))
    if shared['frames'] != end-first:raise RuntimeError('Joined take media does not match its frame range')
    for name in ('input.mp4', 'preview.mp4', 'preview.json'):
        if (clip/name).is_file():_link(clip/name, target/name)
    temporary = target/'tracking.partial.npz'
    np.savez_compressed(temporary, boxes=joined['boxes'].astype(np.float32), observed=joined['observed'], K=lead['K'])
    os.replace(temporary, target/'tracking.npz')
    _save({'K': torch.from_numpy(np.asarray(lead['K']))}, target/'camera.pt')
    receipt = lead['folder']/'exmr_camera_sources.json'  # Read by scene_camera and result_is_current; same folder depth.
    if receipt.is_file():atomic_json(target/receipt.name, json.loads(receipt.read_text(encoding='utf-8')))
    for kind in kinds:
        name, key = RESULTS[kind]
        payload = {key: _tensors(joined['params'][kind])}
        if kind == 'initial':
            payload.update(smpl_params_incam=_tensors(joined['incam']), source_model=lead['source_model'] or pipeline['model'],
                           net_outputs={'static_conf_logits': torch.from_numpy(joined['logits'].astype(np.float32))[None]})
        _save(payload, target/name)
    for kind in kinds:  # A leftover result of an earlier join with another kind becomes stale.
        names = ('tracking.npz', 'hmr4d_results.pt', RESULTS[kind][0], *(('exmr_camera_sources.json',) if receipt.is_file() else ()))
        atomic_json(target/f'{kind}_sources.json', {n: _stamp(target/n) for n in dict.fromkeys(names)})
    temporary = target/'fill_mask.partial.npy';np.save(temporary, joined['mask']);os.replace(temporary, target/'fill_mask.npy')
    lead_state = json.loads((lead['folder']/'project.json').read_text(encoding='utf-8'));state = dict(lead_state)
    for k in RANGE_KEYS:state.pop(k, None)
    state.update({k: shared[k] for k in RANGE_KEYS if k in shared})
    # The joined take holds only the fill's results, so it opens on the result it was made from.
    state.update(frames=end-first, stages={}, person_frames=[first, end], joined=True, pipeline=pipeline)
    if 'start' in lead_state:state['start'] = lead_state['start']
    inherit_storage(lead['folder'], target, state)
    atomic_json(target/'project.json', state)
    atomic_json(target/'fill.json', dict(version=1, person=ident, kind=base, kinds=list(kinds), first=first, end=end,
        sources=[dict(project=s['project'], first=s['first'], end=s['end'], files=s['files']) for s in sources],
        gaps=[_record(g) for g in gaps], donors=donors))
    pending.unlink()
    join = dict(project=target.relative_to(root).as_posix(), first=first, end=end, sources=[s['project'] for s in sources])
    state = load_people(root)  # The editor may have saved other fields meanwhile.
    person = next((p for p in state['people'] if p['id'] == ident), None)
    if person is None:raise RuntimeError('Person no longer exists; the joined take was not recorded. Run Find person again.')
    person['joins'] = _other_joins(person, join)+[join]
    if not join_current(root, person, join):raise RuntimeError('The joined take was written but does not validate; it was not recorded.')
    save_people(root, state)
    return target


def run(root, ident, result, projects):
    """Appends each joined folder to `projects` once it is recorded in people.json; True when nothing was written."""
    from .motion_tools import base_result
    model, person, chains, gaps, kinds, names, body = _check(root, ident, result)
    pipeline = _pipeline(model, result)
    with np.load(body, allow_pickle=False) as data:
        regressor = data['J_regressor'][0];template = regressor@data['v_template']
        shapes = np.einsum('v,vci->ci', regressor, data['shapedirs'][..., :10])
    pelvis = lambda betas: template+shapes@np.asarray(betas, np.float64)[:10]
    from .people_store import safe_child
    plans, loaded = [], {}
    def donor(take):
        if take['project'] not in loaded:loaded[take['project']] = _load(root, take, kinds, names[1:], pelvis)
        return loaded[take['project']]
    for chain, chain_gaps in zip(chains, gaps):
        # Filling again with unchanged inputs keeps the joined take, its correction and its previews.
        stamps = lambda t: {n: _stamp(safe_child(root, t['project'])/n) for n in names[1:]}
        kept = _unchanged(root, person, chain, chain_gaps, _donors(chain_gaps, stamps), kinds, pipeline)
        if kept is not None:plans.append((kept, None, None, None));continue
        sources = [_load(root, t, kinds, names, pelvis) for t in chain]
        records = [dict(g, **{k: g['params'][k] for k in ('mirror', 'offset')}, donor=_donor_rows(donor(g['take']), g)) if g['method'] == 'copy' else g
                   for g in chain_gaps]
        plans.append((sources, chain_gaps, _donors(chain_gaps, lambda t: donor(t)['files']), assemble(sources, records)))
    for i, (sources, chain_gaps, donors, joined) in enumerate(plans):
        emit('progress', stage=STAGE, current=i, total=len(plans), message=f'Filling hidden ranges: joined take {i+1} of {len(plans)}')
        projects.append(str(sources if joined is None else _write(root, ident, sources, chain_gaps, donors, joined, kinds, base_result(result), pipeline)))
    return all(joined is None for *_, joined in plans)


def main(argv=None):
    import argparse
    import time
    import traceback
    parser = argparse.ArgumentParser(prog='take_join_boot.py', description='Join the takes of one person across the selected hidden ranges into one new take.')
    parser.add_argument('--project', required=True, help='scene root folder')
    parser.add_argument('--person-id', type=int, required=True)
    parser.add_argument('--result', default='initial', help='initial, footmr_initial or a corrected_ variant')
    args = parser.parse_args(argv);started = time.monotonic();root = Path(args.project).resolve();projects = []
    try:
        cached = run(root, args.person_id, args.result, projects)
    except Exception as error:
        if not isinstance(error, Refused):traceback.print_exc(file=sys.stderr)
        if projects:emit('takes_joined', project=str(root), person_id=args.person_id, projects=projects)  # Earlier chains are recorded.
        emit('failed', stage=STAGE, message=str(error) or type(error).__name__);return 1
    emit('takes_joined', project=str(root), person_id=args.person_id, projects=projects)
    emit('complete', stage=STAGE, output=projects[-1], cached=cached, elapsed=time.monotonic()-started)
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
