"""Join a person's chain of takes into one new take with smooth hidden ranges.

Runs in its own process. `assemble` is pure numpy; Torch is imported only for file IO.
"""
from pathlib import Path
import json
import os
import sys
import numpy as np
from .gap_fill import INTERPOLATED, POSE_KEYS, fill_logits, rigid_params, smooth_fill

STAGE = 'take_join'


def _stitch(parts, sources, gaps, base, total, fill):
    out = np.empty((total, *np.shape(parts[0])[1:]), np.float64)
    for s, p in zip(sources, parts):out[s['first']-base:s['end']-base] = p
    for i, g in enumerate(gaps):out[g['first']-base:g['end']-base] = fill(parts[i], parts[i+1], g['end']-g['first'])
    return out


def _params(parts, sources, gaps, base, total, only=None):
    # A pose key missing from any take (e.g. GVHMR hands) is dropped rather than invented.
    keys = [k for k in POSE_KEYS if all(k in p for p in parts) and (only is None or k in only)]+['transl']
    parts = [{k: np.asarray(p[k], np.float64).reshape(len(p['transl']), -1) for k in keys} for p in parts]
    out = {k: np.empty((total, parts[0][k].shape[1]), np.float64) for k in keys}
    for s, p in zip(sources, parts):
        for k in keys:out[k][s['first']-base:s['end']-base] = p[k]
    for i, g in enumerate(gaps):
        fill = smooth_fill({k: v[-2:] for k, v in parts[i].items()}, {k: v[:2] for k, v in parts[i+1].items()}, g['end']-g['first'])
        for k in keys:out[k][g['first']-base:g['end']-base] = fill[k]
    return out


def assemble(sources, gaps):
    first, base = sources[0], sources[0]['first']; total = sources[-1]['end']-base
    if len(gaps) != len(sources)-1:raise ValueError('Each pair of joined takes needs one hidden range')
    for a, g, b in zip(sources, gaps, sources[1:]):
        if (g['first'], g['end']) != (a['end'], b['first']) or g['end'] <= g['first']:raise ValueError('Hidden range does not lie between its takes')
        if g['method'] != 'smooth':raise ValueError(f"Unsupported fill method: {g['method']}")
    for s in sources:
        n = s['end']-s['first']
        if len(s['logits']) != n or len(s['boxes']) != n or len(s['observed']) != n or len(s['incam']['transl']) != n or \
                any(len(p['transl']) != n for p in s['params'].values()):raise ValueError('Take arrays do not match its frame range')
    moved = lambda s, p, T: p if s is first else rigid_params(p, T, s['pelvis'], first['pelvis'])
    params = {}
    for kind, own in first['params'].items():
        parts = [moved(s, s['params'][kind], s['transform']) for s in sources]
        params[kind] = _params(parts, sources, gaps, base, total)
        params[kind]['betas'] = np.repeat(np.asarray(own['betas'], np.float64)[:1], total, 0)
    # Camera space is shared by every take; only the replaced shape moves the pelvis.
    incam = _params([moved(s, s['incam'], np.eye(4)) for s in sources], sources, gaps, base, total, params['initial'])
    camera = np.asarray(first['world_to_camera'], np.float64); pelvis = first['pelvis']
    for g in gaps:
        span = slice(g['first']-base, g['end']-base)
        moved_gap = rigid_params({k: v[span] for k, v in params['initial'].items() if k in incam}, camera, pelvis, pelvis)
        for k in incam:incam[k][span] = np.asarray(moved_gap[k], np.float64).reshape(g['end']-g['first'], -1)
    incam['betas'] = np.repeat(np.asarray(first['incam']['betas'], np.float64)[:1], total, 0)
    logits = _stitch([np.asarray(s['logits'], np.float64) for s in sources], sources, gaps, base, total,
                     lambda a, b, n: fill_logits(a[-1], b[0], n))
    boxes = _stitch([np.asarray(s['boxes'], np.float64) for s in sources], sources, gaps, base, total,
                    lambda a, b, n: fill_logits(a[-1], b[0], n))
    mask = np.full(total, INTERPOLATED, np.uint8); observed = np.zeros(total, bool)
    for s in sources:
        mask[s['first']-base:s['end']-base] = 0; observed[s['first']-base:s['end']-base] = np.asarray(s['observed'], bool)
    return dict(params=params, incam=incam, logits=logits, mask=mask, boxes=boxes, observed=observed)


UNSUPPORTED = 'Hidden ranges support the initial and FootMR results in this version; select one of them.'
RANGE_KEYS = ('start', 'end', 'frames', 'timeline', 'original_times', 'duration')


def command(args):
    from .paths import ROOT
    if getattr(sys, 'frozen', False) or os.environ.get('RGB_MOCAP_MANAGED') == '1':
        return [str(ROOT/'runtime/python.exe'), '-I', '-X', 'utf8', '-B', '-u', str(ROOT/'take_join_boot.py'), *map(str, args)]
    return [sys.executable, '-X', 'utf8', '-B', '-u', '-m', 'mocap.take_join', *map(str, args)]


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
    if not (root/'project.json').is_file() or not (root/'people.json').is_file():raise ValueError('Open a scene with people before filling hidden ranges.')
    model = require_supported_project(root).key
    if model not in ('gvhmr', 'exmr'):raise ValueError('Hidden ranges can be filled only in GVHMR and ExMR scenes.')
    if model == 'exmr' and not saved(root)['static_camera']:raise ValueError('Hidden ranges need a fixed camera. Turn on Static camera in the ExMR settings.')
    if result not in RESULTS:raise ValueError(f'Unknown motion result: {result}')
    kinds = result_sources(base_result(result))
    if has_handflow(result) or any(k.endswith('refined') for k in kinds):raise ValueError(UNSUPPORTED)
    person = next((p for p in load_people(root)['people'] if p['id'] == ident), None)
    if person is None:raise ValueError('Person no longer exists. Run Find person again.')
    choices = person.get('gap_fill') or {}
    chains = hidden_ranges.chains(person.get('takes', []), choices)
    if not chains:raise ValueError('No hidden range is selected for filling.')
    body = MODELS/'body_models/smplx/SMPLX_NEUTRAL.npz'
    if not body.is_file():raise ValueError('Import SMPLX_NEUTRAL.npz to join takes.')
    names = list(dict.fromkeys(['tracking.npz', *(RESULTS[k][0] for k in kinds)]))
    rates = set()
    for take in (t for chain in chains for t in chain):
        folder = safe_child(root, take['project']);state = json.loads((folder/'project.json').read_text(encoding='utf-8'))
        if state.get('frames') != take['end']-take['first']:raise ValueError(f'Take {folder.name} does not match its frame range. Prepare the person motion again.')
        rates.add(state.get('fps'))
        for name in names:
            if not (folder/name).is_file():raise ValueError(f'Take {folder.name} has no {name}. Reconstruct it first.')
        for kind in kinds:
            if not result_is_current(folder, kind):raise ValueError(f'The {kind} result of take {folder.name} is out of date. Reconstruct it first.')
    if len(rates) != 1:raise ValueError('The takes to join have different frame rates.')
    key = lambda a, b: hidden_ranges.gap_key(dict(first=a['end'], end=b['first']))
    gaps = [[dict(first=a['end'], end=b['first'], method=choices[key(a, b)]['method']) for a, b in zip(chain, chain[1:])] for chain in chains]
    return model, chains, gaps, kinds, names, body


def _cameras(root, takes):
    from .people_store import safe_child
    from .scene_camera import prepare_cameras, read_camera
    folders = [safe_child(root, t['project']) for t in takes]
    if any(read_camera(f) is None for f in folders):prepare_cameras(root)  # Same caches the scene loader writes.
    cameras = [read_camera(f) for f in folders]
    if any(c is None for c in cameras):raise ValueError('Cannot prepare the shared scene camera for these takes. Reconstruct them again.')
    return cameras


def _load(root, take, camera, kinds, names, pelvis):
    import torch
    from .motion_tools import RESULTS
    from .people_store import safe_child
    folder = safe_child(root, take['project']);files = {n: _stamp(folder/n) for n in names}
    numpy = lambda params: {k: v.detach().cpu().double().numpy() for k, v in params.items()}
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
    if {n: _stamp(folder/n) for n in names} != files:raise ValueError(f'Take {folder.name} changed while it was read; try again.')
    return dict(first=take['first'], end=take['end'], project=take['project'], folder=folder, files=files, K=K,
                world_to_camera=np.asarray(camera['world_to_camera'], np.float64), pelvis=pelvis(params['initial']['betas'][0]),
                params=params, incam=incam, logits=logits, boxes=boxes, observed=observed, source_model=model)


def _write(root, ident, sources, gaps, joined, kinds, base, model):
    import torch
    from .motion_tools import RESULTS
    from .people_store import join_current, load_people, safe_child, save_people
    from .person_tracking import _link
    from .project_storage import inherit_storage
    from .storage import atomic_json
    from .video import prepare_people_clip
    first, end, lead = sources[0]['first'], sources[-1]['end'], sources[0]
    person = next((p for p in load_people(root)['people'] if p['id'] == ident), None)
    if person is None:raise ValueError('Person no longer exists. Run Find person again.')
    target = safe_child(root, person['preview'])/'takes'/f'{first:09d}_{end:09d}_fill_v1'
    target.mkdir(parents=True, exist_ok=True);pending = target/'fill_pending.json'
    # The marker stays after a failure, so a half-written folder is never current.
    atomic_json(pending, dict(version=1, person=ident, first=first, end=end))
    clip = prepare_people_clip(root, first, end);shared = json.loads((clip/'project.json').read_text(encoding='utf-8'))
    if shared['frames'] != end-first:raise ValueError('Joined take media does not match its frame range')
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
            payload.update(smpl_params_incam=_tensors(joined['incam']), source_model=lead['source_model'] or model,
                           net_outputs={'static_conf_logits': torch.from_numpy(joined['logits'].astype(np.float32))[None]})
        _save(payload, target/name)
    for kind in kinds:  # A leftover result of an earlier join with another kind becomes stale.
        names = ('tracking.npz', 'hmr4d_results.pt', RESULTS[kind][0], *(('exmr_camera_sources.json',) if receipt.is_file() else ()))
        atomic_json(target/f'{kind}_sources.json', {n: _stamp(target/n) for n in dict.fromkeys(names)})
    temporary = target/'fill_mask.partial.npy';np.save(temporary, joined['mask']);os.replace(temporary, target/'fill_mask.npy')
    lead_state = json.loads((lead['folder']/'project.json').read_text(encoding='utf-8'));state = dict(lead_state)
    for k in RANGE_KEYS:state.pop(k, None)
    state.update({k: shared[k] for k in RANGE_KEYS if k in shared})
    state.update(frames=end-first, stages={}, person_frames=[first, end], joined=True)
    if 'start' in lead_state:state['start'] = lead_state['start']
    inherit_storage(lead['folder'], target, state)
    atomic_json(target/'project.json', state)
    atomic_json(target/'fill.json', dict(version=1, person=ident, kind=base, first=first, end=end,
        sources=[dict(project=s['project'], first=s['first'], end=s['end'], files=s['files']) for s in sources],
        gaps=[dict(first=g['first'], end=g['end'], method=g['method'], params={}) for g in gaps]))
    pending.unlink()
    join = dict(project=target.relative_to(root).as_posix(), first=first, end=end, sources=[s['project'] for s in sources])
    state = load_people(root)  # The editor may have saved other fields meanwhile.
    person = next((p for p in state['people'] if p['id'] == ident), None)
    if person is None:raise ValueError('Person no longer exists. Run Find person again.')
    person['joins'] = [j for j in person.get('joins') or [] if not (j.get('first', end) < end and first < j.get('end', first))]+[join]
    save_people(root, state)
    if not join_current(root, person, join):raise RuntimeError('The joined take was written but does not validate; it will not be used.')
    return target


def run(root, ident, result):
    from .motion_tools import base_result
    root = Path(root).resolve()
    model, chains, gaps, kinds, names, body = _check(root, ident, result)
    takes = [t for chain in chains for t in chain]
    cameras = dict(zip((t['project'] for t in takes), _cameras(root, takes)))
    with np.load(body, allow_pickle=False) as data:
        regressor = data['J_regressor'][0];template = regressor@data['v_template']
        shapes = np.einsum('v,vci->ci', regressor, data['shapedirs'][..., :10])
    pelvis = lambda betas: template+shapes@np.asarray(betas, np.float64)[:10]
    plans = []
    for chain, chain_gaps in zip(chains, gaps):
        sources = [_load(root, t, cameras[t['project']], kinds, names, pelvis) for t in chain]
        inverse = np.linalg.inv(sources[0]['world_to_camera'])
        for s in sources:s['transform'] = inverse@s['world_to_camera']
        plans.append((sources, chain_gaps, assemble(sources, chain_gaps)))
    projects = []
    for i, (sources, chain_gaps, joined) in enumerate(plans):
        emit('progress', stage=STAGE, current=i, total=len(plans), message=f'Filling hidden ranges: joined take {i+1} of {len(plans)}')
        projects.append(str(_write(root, ident, sources, chain_gaps, joined, kinds, base_result(result), model)))
    return root, projects


def main(argv=None):
    import argparse
    import time
    import traceback
    parser = argparse.ArgumentParser(prog='take_join_boot.py', description='Join the takes of one person across the selected hidden ranges into one new take.')
    parser.add_argument('--project', required=True, help='scene root folder')
    parser.add_argument('--person-id', type=int, required=True)
    parser.add_argument('--result', default='initial', help='initial, footmr_initial or a corrected_ variant')
    args = parser.parse_args(argv);started = time.monotonic()
    try:
        root, projects = run(args.project, args.person_id, args.result)
    except Exception as error:
        if not isinstance(error, ValueError):traceback.print_exc(file=sys.stderr)  # Refusals carry their own message.
        emit('failed', stage=STAGE, message=str(error) or type(error).__name__);return 1
    emit('takes_joined', project=str(root), person_id=args.person_id, projects=projects)
    emit('complete', stage=STAGE, output=projects[-1], cached=False, elapsed=time.monotonic()-started)
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
