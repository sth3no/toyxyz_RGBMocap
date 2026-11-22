"""Join a person's chain of takes into one new take with filled hidden ranges.

Runs in its own process. `assemble` is pure numpy; Torch is imported only for file IO.
"""
from pathlib import Path
import json
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .gap_fill import COPIED, GENERATED, INTERPOLATED, POSE_KEYS, contact_logits, copy_fill, fill_logits, generated_fill, mirror_logits, mirror_params, rigid_params, smooth_fill
from .hidden_ranges import typed as _typed

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
    at = lambda p, v, sign: dict(p, transl=p['transl']+sign*v)  # copy_fill and mirror_params work on pelvis positions
    params, residual = {}, {}
    for kind, p in a['params'].items():
        # The donor stays in its own world: copy_fill uses only its heading changes, its steps and its heights above its own floor.
        donor = at({k: np.asarray(v, np.float64).reshape(n+2, -1) for k, v in d['params'][kind].items()}, dp, 1)
        if g['mirror']:donor = mirror_params(donor)
        stats, keys = {}, [k for k in p if k != 'transl']
        params[kind] = at(copy_fill(at(p, pelvis, 1), at(b['params'][kind], pelvis, 1), donor, keys, g.get('height_scale', 1.), stats), pelvis, -1)
        residual[kind] = stats['residual']
    logits = np.asarray(d['logits'], np.float64)[1:-1]
    return dict(params=params, incam=_camera(a, b, params['initial'], pelvis, n), logits=mirror_logits(logits) if g['mirror'] else logits, mask=COPIED, residual=residual)


def _camera(a, b, G, pelvis, n):
    """Camera-space gap rows: the world rows G seen through a camera pairing that moves from one seam to the other."""
    W0, W1 = _seam(a, -1, pelvis), _seam(b, 0, pelvis)
    s = np.arange(1, n+1)/(n+1);turn = Slerp([0., 1.], Rotation.from_matrix([W0[:3, :3], W1[:3, :3]]))(s)
    # A camera-space pose key that the world result lacks is interpolated.
    rest = smooth_fill({k: v[-2:] for k, v in a['incam'].items()}, {k: v[:2] for k, v in b['incam'].items()}, n) if any(k not in G for k in a['incam']) else {}
    incam = {k: G[k] if k in G else rest[k] for k in a['incam']}
    incam.update(global_orient=(turn*Rotation.from_rotvec(G['global_orient'])).as_rotvec(),
                 transl=turn.apply(G['transl']+pelvis)+W0[:3, 3]+(W1[:3, 3]-W0[:3, 3])*s[:, None]-pelvis)
    return incam


AUTO_CONTEXT, AUTO_KNOWN = 45, 5  # captured frames sent on each side; known frames the generator uses on each side
MISMATCH = 'Generated motion does not match the hidden range'


def _auto(a, b, g, pelvis):
    n, A, B = g['end']-g['first'], a['params']['initial'], b['params']['initial']
    nb, na = min(AUTO_CONTEXT, len(A['transl'])), min(AUTO_CONTEXT, len(B['transl']))
    request = dict(version=1, before_orient=A['global_orient'][-nb:].copy(), before_pose=A['body_pose'][-nb:].copy(), before_pelvis=A['transl'][-nb:]+pelvis,
                   after_orient=B['global_orient'][:na].copy(), after_pose=B['body_pose'][:na].copy(), after_pelvis=B['transl'][:na]+pelvis,
                   rest=np.array(g['body']['rest'], np.float64), sole=float(g['body']['sole']), frames=n, seed=g['params']['seed'], fps=g.get('fps', 30.))
    r = g['generate'](request);c0, c1 = min(AUTO_KNOWN, nb), min(AUTO_KNOWN, na);rows = c0+n+c1
    try:
        arrays = {k: np.asarray(r[k], np.float64) for k in ('orient', 'pose', 'pelvis')};contacts = np.asarray(r['contacts']);end_pose = float(r['end_pose_cm'])
        ok = r['version'] == 1 and np.array_equal(np.asarray(r['context']), [c0, c1]) and contacts.shape == (rows, 4) and contacts.dtype == bool and np.isfinite(end_pose) and \
            all(v.shape == (rows, w) and np.isfinite(v).all() for v, w in zip(arrays.values(), (3, 63, 3)))
    except (KeyError, TypeError, ValueError, IndexError):ok = False
    if not ok:raise ValueError(MISMATCH)
    # Response rows c0-1 ... c0+n are the boundary frames and the range; generated_fill works on pelvis positions.
    cut = slice(c0-1, c0+n+1);generated = dict(global_orient=arrays['orient'][cut], body_pose=arrays['pose'][cut], transl=arrays['pelvis'][cut])
    at = lambda p, sign: dict(p, transl=p['transl']+sign*pelvis)
    params = {kind: at(generated_fill(at(p, 1), at(b['params'][kind], 1), generated, [k for k in p if k != 'transl']), -1) for kind, p in a['params'].items()}
    return dict(params=params, incam=_camera(a, b, params['initial'], pelvis, n), logits=contact_logits(contacts[c0:c0+n], a['logits'][-1], b['logits'][0]),
                mask=GENERATED, report=dict(end_pose_cm=end_pose))


FILLS = {'smooth': _smooth, 'copy': _copy, 'auto': _auto}


def assemble(sources, gaps):
    first, base = sources[0], sources[0]['first']; total = sources[-1]['end']-base
    if len(gaps) != len(sources)-1:raise ValueError('Each pair of joined takes needs one hidden range')
    for a, g, b in zip(sources, gaps, sources[1:]):
        if (g['first'], g['end']) != (a['end'], b['first']) or g['end'] <= g['first']:raise ValueError('Hidden range does not lie between its takes')
        if g['method'] not in FILLS:raise ValueError(f"Unsupported fill method: {g['method']}")
        if g['method'] == 'copy':
            d, n = g['donor'], g['end']-g['first']+2
            if not set(first['params']) <= set(d['params']) or len(d['logits']) != n or \
                    any(len(d['params'][k]['transl']) != n for k in first['params']):raise ValueError('Donor arrays do not match the hidden range')
        if g['method'] == 'auto' and not (callable(g.get('generate')) and isinstance(g.get('body'), dict) and isinstance(g.get('params'), dict) and 'seed' in g['params']):
            raise ValueError('Automatic fill needs a motion generator')
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
    # The horizontal path residual in metres per result kind of each copied gap and the report of each generated gap; None for other methods.
    return dict(params=params, incam=incam, logits=logits, mask=mask, boxes=boxes, observed=observed, residuals=[f.get('residual') for f in fills],
                reports=[f.get('report') for f in fills])


class Refused(ValueError):
    """A deliberate refusal with a message for the user; nothing has been written."""


UNSUPPORTED = 'Hidden ranges support the initial and FootMR results in this version; select one of them.'
RANGE_KEYS = ('start', 'end', 'frames', 'timeline', 'original_times', 'duration')
DONOR_NOT_READY = "{}'s take is not ready to be copied from. Reconstruct and prepare it first."
COPY_INFO = 'Frames {}–{} copied from {}: the path was adjusted by {:.1f} cm ({:.2f} cm per frame) to meet the captured motion.'
AUTO_INFO = 'Frames {}–{} generated: the captured poses on both sides were met within {:.1f} cm.'
AUTO_RATE = 'Automatic fill needs 30 fps takes; this scene has {}.'


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
            if method == 'auto':
                rate, = rates
                if rate != 30:raise Refused(AUTO_RATE.format(int(rate) if type(rate) is float and rate.is_integer() else rate))
                gaps[-1][-1]['fps'] = float(rate)
            if method != 'copy':continue
            donor = next((p for p in people if p['id'] == params['person']), None)
            if donor is None:raise Refused('Choose a person to copy from.')
            take = gaps[-1][-1]['take'] = hidden_ranges.donor_take(donor['takes'], g, params['offset'])
            name = gaps[-1][-1]['name'] = _typed(donor.get('name') or 'Person '+str(donor['id']))
            try:
                folder = safe_child(root, take['project']);manifest = json.loads((folder/'project.json').read_text(encoding='utf-8'))
                ready = manifest.get('frames') == take['end']-take['first'] and {manifest.get('fps')} == rates and \
                    all((folder/n).is_file() for n in names[1:]) and all(result_is_current(folder, k) for k in kinds)
            except (OSError, ValueError):ready = False
            if not ready:raise Refused(DONOR_NOT_READY.format(name))
    return model, person, chains, gaps, kinds, names, body


def _pipeline(model, result):
    from .motion_tools import body_result
    return dict(model=model, result=result, htd_refine=False, footmr=body_result(result).startswith('footmr_'), handflow=False)


def _record(g):return dict(first=g['first'], end=g['end'], method=g['method'], params=g['params'])


def _cm(metres):return round(float(metres)*100, 1)


def _generator(report):
    """The fill.json record of the generator of an automatic gap, beside its params: a report, not an input."""
    from .auto_fill import VERSION
    from .kimodo_runtime import identity
    return dict(name='kimodo', identity=identity(), version=VERSION, end_pose_cm=float(report['end_pose_cm']))


def _report(gaps, residuals, ends):
    """Console lines telling how much each copied path had to be bent and how closely each generated range met its sides; in cm."""
    for g, r, e in zip(gaps, residuals, ends):
        if g['method'] == 'copy' and type(r) in (int, float):
            emit('info', stage=STAGE, message=COPY_INFO.format(g['first']+1, g['end'], g['name'], r, r/(g['end']-g['first']+1)))
        if g['method'] == 'auto' and type(e) in (int, float):emit('info', stage=STAGE, message=AUTO_INFO.format(g['first']+1, g['end'], e))


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
    return dict(pelvis=take['pelvis'], params={k: {n: cut(v) for n, v in p.items()} for k, p in take['params'].items()}, logits=cut(take['logits']))


def _unchanged(root, person, chain, gaps, donors, kinds, pipeline):
    """The folder of this chain's recorded join when it is current and holds what would be written, else None."""
    from .motion_tools import result_is_current
    from .people_store import join_fill, safe_child
    folder = safe_child(root, person['preview'])/'takes'/f"{chain[0]['first']:09d}_{chain[-1]['end']:09d}_fill_v1"
    for join in person.get('joins') or []:
        if join['project'] != folder.relative_to(root).as_posix() or join.get('sources') != [t['project'] for t in chain]:continue
        fill = join_fill(root, person, join)
        if fill is None:return None
        recorded = [{k: v for k, v in x.items() if k not in ('residual_cm', 'generator')} for x in fill['gaps']]  # Reports, not inputs.
        if fill.get('kinds') != kinds or recorded != [_record(g) for g in gaps] or (fill.get('donors') or []) != donors:return None
        try:written = json.loads((folder/'project.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):return None
        written = written.get('pipeline') if isinstance(written, dict) else None
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
        if kind == 'initial':  # A donor's camera-space motion is not used.
            incam, model = numpy(record['smpl_params_incam']) if 'tracking.npz' in names else None, record.get('source_model')
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
        gaps=[dict(_record(g), **({} if r is None else dict(residual_cm=_cm(r[base]))), **({} if e is None else dict(generator=_generator(e))))
              for g, r, e in zip(gaps, joined['residuals'], joined['reports'])], donors=donors))
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
        body22 = data['J_regressor'][:22];rest = body22@data['v_template'];rest_shapes = np.einsum('jv,vci->jci', body22, data['shapedirs'][..., :10])
        vertices, vertex_shapes = data['v_template'], data['shapedirs'][..., :10]
    pelvis = lambda betas: template+shapes@np.asarray(betas, np.float64)[:10]
    def shape(take):  # rest joints (pelvis at its rest position) and lowest rest surface height of the joined shape
        b = np.asarray(take['params']['initial']['betas'][0], np.float64)[:10]
        return dict(rest=rest+rest_shapes@b, sole=float((vertices[:, 1]+vertex_shapes[:, 1]@b).min()))
    def leg(take):  # rest pelvis height above the lowest rest joint
        y = (rest+rest_shapes@np.asarray(take['params']['initial']['betas'][0], np.float64)[:10])[:, 1];return y[0]-y.min()
    def height_scale(first, donor):
        try:a, b = leg(first), leg(donor)
        except (KeyError, IndexError, ValueError):return 1.
        return float(a/b) if np.isfinite(a) and np.isfinite(b) and a > 1e-6 and b > 1e-6 else 1.
    from . import auto_fill
    from .people_store import safe_child
    plans, loaded, base = [], {}, base_result(result)
    cache = safe_child(root, person['preview'])/'auto_fill'
    echo = lambda line: print(line, flush=True)  # the worker's progress reaches the app unchanged
    generate = lambda request: auto_fill.generate(cache, request, echo=echo)
    def donor(g):
        take = g['take']
        if take['project'] not in loaded:
            try:
                d, n = _load(root, take, kinds, names[1:], pelvis), take['end']-take['first']
                if len(d['logits']) != n or any(len(v) != n for p in d['params'].values() for v in p.values()):raise ValueError('short donor arrays')
            except Exception:raise Refused(DONOR_NOT_READY.format(g['name'])) from None
            loaded[take['project']] = d
        return loaded[take['project']]
    for chain, chain_gaps in zip(chains, gaps):
        # Filling again with unchanged inputs keeps the joined take, its correction and its previews.
        stamps = lambda t: {n: _stamp(safe_child(root, t['project'])/n) for n in names[1:]}
        try:kept = _unchanged(root, person, chain, chain_gaps, _donors(chain_gaps, stamps), kinds, pipeline)
        except OSError:kept = None  # A donor file is gone; reading it below refuses with the donor's name.
        if kept is not None:plans.append((kept, chain_gaps, None, None));continue
        sources = [_load(root, t, kinds, names, pelvis) for t in chain]
        records = [dict(g, **{k: g['params'][k] for k in ('mirror', 'offset')}, donor=_donor_rows(donor(g), g), height_scale=height_scale(sources[0], donor(g)))
                   if g['method'] == 'copy' else dict(g, generate=generate, body=shape(sources[0])) if g['method'] == 'auto' else g for g in chain_gaps]
        try:joined = assemble(sources, records)
        except auto_fill.Unavailable as error:raise Refused(str(error)) from None
        plans.append((sources, chain_gaps, _donors(chain_gaps, lambda t: loaded[t['project']]['files']), joined))
    for i, (sources, chain_gaps, donors, joined) in enumerate(plans):
        emit('progress', stage=STAGE, current=i, total=len(plans), message=f'Filling hidden ranges: joined take {i+1} of {len(plans)}')
        if joined is None:
            projects.append(str(sources))
            try:
                recorded = json.loads((sources/'fill.json').read_text(encoding='utf-8'))['gaps']
                residuals, ends = [g.get('residual_cm') for g in recorded], [(g.get('generator') or {}).get('end_pose_cm') for g in recorded]
            except (OSError, ValueError, KeyError, TypeError, AttributeError):residuals = ends = []
        else:
            projects.append(str(_write(root, ident, sources, chain_gaps, donors, joined, kinds, base, pipeline)))
            residuals = [None if r is None else _cm(r[base]) for r in joined['residuals']]
            ends = [None if e is None else float(e['end_pose_cm']) for e in joined['reports']]
        _report(chain_gaps, residuals, ends)
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
