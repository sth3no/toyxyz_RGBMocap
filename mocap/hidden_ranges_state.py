"""Plain decisions behind the Hidden ranges panel; no Qt, so they test in the shared runtime."""
from pathlib import Path
import json
import numpy as np
from .hidden_ranges import METHODS, OFFSET_LIMIT, SEED_LIMIT, chains, choice_params, donor_take, gap_key, gap_label, method_issue

SUPPORTED = ('initial', 'footmr_initial')
NOT_READY = 'Reconstruct and prepare both neighbouring takes first.'
DONOR_NOT_READY = "{}'s take must be reconstructed and prepared before it can be copied from."


def disabled_reason(model, static_camera, kind, ready):
    """First reason the Fill button cannot run; `ready` is called last because it reads take files."""
    if model not in ('gvhmr', 'exmr'):return 'Hidden ranges are available for GVHMR and ExMR in this version.'
    if model == 'exmr' and not static_camera:return 'Hidden ranges need a fixed camera in this version.'
    if kind.removeprefix('corrected_') not in SUPPORTED:return 'Hidden ranges support the initial and FootMR results in this version.'
    ready = ready()  # True, False, or the reason naming a donor whose take is not ready
    return None if ready is True else ready if isinstance(ready, str) else NOT_READY


def editable(reason):
    """Choices stay editable only while the project, camera and result are supported."""
    return reason is None or reason == NOT_READY or reason.endswith(DONOR_NOT_READY.format(''))


def _people(people):return [p for p in people if isinstance(p, dict)] if isinstance(people, list) else []


def _auto_issue(gap, auto_issue):return method_issue('auto', gap) or auto_issue


def _shown(choice, gap, people, ident, auto_issue=None):
    """(method, person, mirror, offset, seed, issue) as the row shows it; a malformed record shows as Do not fill."""
    none = ('none', None, False, 0, 0, None)
    if not isinstance(choice, dict) or choice.get('method', 'none') not in METHODS:return none
    method = choice.get('method', 'none')
    if method == 'none':return none
    # Step 1 showed an over-limit Smooth join as Do not fill; keep that.
    if method == 'smooth':return ('smooth', None, False, 0, 0, None) if method_issue('smooth', gap) is None else none
    if method == 'auto':
        try:seed = choice_params(choice)[1]['seed']
        except ValueError:return none
        return 'auto', None, False, 0, seed, _auto_issue(gap, auto_issue)
    p, mirror, off = choice.get('person'), choice.get('mirror', False), choice.get('offset', 0)
    # A copy without a person is what choose() records before one is picked; it stays Copy with an issue.
    if (p is not None and type(p) is not int) or type(mirror) is not bool or type(off) is not int or abs(off) > OFFSET_LIMIT:return none
    return 'copy', p, mirror, off, 0, method_issue('copy', gap, choice, people, ident)


def gap_rows(gaps, choices, fps, people=None, ident=None, auto_issue=None):
    """One row per gap; `auto_issue` is why Automatic cannot run at all (no runtime or model), None when it can.
    `setup` marks a chosen Automatic row stopped by that, not by the range's own limit."""
    choices = choices if isinstance(choices, dict) else {};people = _people(people)
    donors = [(p['id'], str(p.get('name') or f"Person {p['id']}")) for p in people if type(p.get('id')) is int and p['id'] != ident]
    rows = []
    for g in gaps:
        method, person, mirror, offset, seed, issue = _shown(choices.get(gap_key(g)), g, people, ident, auto_issue)
        rows.append(dict(key=gap_key(g), label=gap_label(g, fps), method=method, person=person, mirror=mirror, offset=offset, seed=seed,
                         issue=issue, smooth_issue=method_issue('smooth', g), auto_issue=_auto_issue(g, auto_issue), donors=donors,
                         setup=method == 'auto' and bool(issue) and method_issue('auto', g) is None))
    return rows


def _sources(gaps, choices, people, ident):
    """{project: (take, donor name or None)} of the takes the worker would read; Automatic's runtime is not checked here."""
    choices = choices if isinstance(choices, dict) else {};people = _people(people);out = {}
    for g in gaps:
        method, person, _, offset, _, issue = _shown(choices.get(gap_key(g)), g, people, ident)
        if method == 'none' or issue:continue
        for t in (g['before'], g['after']):out[t['project']] = (t, None)
        donor = next((p for p in people if p.get('id') == person), None) if method == 'copy' else None
        take = donor_take(donor.get('takes'), g, offset) if donor else None
        if take is not None:out.setdefault(take['project'], (take, str(donor.get('name') or f'Person {person}')))
    return out


def fill_sources(gaps, choices, people=None, ident=None):
    """The takes the worker would read: both neighbours of every gap that will be filled, and the donor take of a copy."""
    return [t for t, _ in _sources(gaps, choices, people, ident).values()]


def can_fill(rows, reason):
    chosen = [r for r in rows if r['method'] != 'none']
    return reason is None and bool(chosen) and not any(r['issue'] for r in chosen)


def pending_state(gaps, choices, pending, reason, people=None, ident=None, auto_issue=None):
    """({key: issue}, can fill) with unsaved copy edits {key: (person, mirror, offset)} applied; nothing is saved."""
    choices = {k: v for k, v in choices.items()} if isinstance(choices, dict) else {}
    for key, (person, mirror, offset) in pending.items():choose(choices, key, 'copy', person, mirror, offset)
    rows = gap_rows(gaps, choices, 30, people, ident, auto_issue)
    return {r['key']: r['issue'] for r in rows}, can_fill(rows, reason)


def choose(choices, key, method, person=None, mirror=False, offset=0, seed=0):
    if method not in METHODS:raise ValueError(f'Unknown method: {method}')
    if method == 'none':choices.pop(key, None)
    elif method == 'smooth':choices[key] = {'method': 'smooth'}
    elif method == 'auto':choices[key] = {'method': 'auto', 'seed': max(0, min(SEED_LIMIT, int(seed)))}
    else:
        choices[key] = {'method': 'copy', **({} if person is None else {'person': int(person)}),
                        'mirror': bool(mirror), 'offset': max(-OFFSET_LIMIT, min(OFFSET_LIMIT, int(offset)))}


def scene_frame(root, start, fps, index):
    """Scene frame of frame `index` of a take that starts `start` seconds into the source video; None when the scene cannot be read.
    Take frame ranges count from the scene's own start, which is not the video's when the scene was prepared from a trim."""
    try:return round((start-json.loads((Path(root)/'project.json').read_text(encoding='utf-8'))['start'])*fps)+index
    except (OSError, ValueError, KeyError, TypeError):return None


def take_to_show(root, person, kind, project, frame=None):
    """Take in use to move the view to when the loaded take of this person is no longer in use; None to stay."""
    from .people_store import active_takes, owned_projects, safe_child
    try:
        project = Path(project).resolve()
        if not any(safe_child(root, p) == project for p in owned_projects(person)):return None
        active = active_takes(root, person, kind)
        if not active or any(safe_child(root, t['project']) == project for t in active):return None
        if frame is None:return active[-1]['project']
        away = lambda t: max(t['first']-frame, frame-(t['end']-1), 0)
        return min(reversed(active), key=away)['project']
    except (ValueError, KeyError, TypeError, AttributeError, OSError):return None


def unlisted_joins(root, person, kind, people=None):
    """Joined takes the take selector leaves out: not in use, and not what Fill hidden ranges would write again."""
    from .people_store import active_takes
    try:
        used = {t['project'] for t in active_takes(root, person, kind) if t.get('join')}
        wanted = [[t['project'] for t in chain] for chain in chains(person['takes'], person.get('gap_fill') or {}, people or [], person.get('id'))]
        return {j['project'] for j in person.get('joins') or [] if j['project'] not in used and j.get('sources') not in wanted}
    except (ValueError, KeyError, TypeError, AttributeError, OSError):return set()


def read_mask(project):
    try:return np.load(Path(project)/'fill_mask.npy', allow_pickle=False)
    except (OSError, ValueError):return None


def fill_runs(mask):
    """(first, end) frame runs of a joined take that were not captured."""
    if mask is None:return ()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (np.asarray(mask) != 0).astype(np.int8), [0]])))
    return tuple((int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]))


def project_fill_runs(root, state, project):
    """Timeline runs for a loaded take; only a joined take has a mask."""
    from .people_store import safe_child
    project = Path(project).resolve()
    joined = any(safe_child(root, j['project']) == project for p in state['people'] for j in p.get('joins') or [])
    return fill_runs(read_mask(project)) if joined else ()


def export_line(stem, mask):
    mask = np.asarray(mask)  # 1 is gap_fill.INTERPOLATED, 2 is gap_fill.COPIED, 3 is gap_fill.GENERATED
    i, c, g = int((mask == 1).sum()), int((mask == 2).sum()), int((mask == 3).sum())
    if not c and not g:return f'{stem}: {i} of {len(mask)} frames are interpolated across hidden ranges.'
    if not g:return f'{stem}: {i+c} of {len(mask)} frames are filled across hidden ranges ({i} interpolated, {c} copied from another person).'
    kinds = ', '.join(f'{x} {name}' for x, name in ((i, 'interpolated'), (c, 'copied from another person'), (g, 'generated')) if x)
    return f'{stem}: {i+c+g} of {len(mask)} frames are filled across hidden ranges ({kinds}).'


def panel_state(root, found, choices, model, kind, people=None, ident=None):
    """(disabled reason, manual contact note) for a person's gaps; reads only the takes that would be joined or copied."""
    from .foot_contacts import contact_state
    from .motion_tools import base_result, selected_export
    from .people_store import safe_child
    base = base_result(kind)
    try:sources = [(safe_child(root, t['project']), name) for t, name in _sources(found, choices, people, ident).values()]
    except (ValueError, KeyError, TypeError):return 'Saved take paths could not be verified. Run Find person again.', False
    def any_source(test):
        try:return any(test(project) for project, _ in sources)
        except (OSError, ValueError, KeyError, TypeError):return None
    def ready():
        try:waiting = [name for project, name in sources if selected_export(project, base) is None]
        except (OSError, ValueError, KeyError, TypeError):return False
        donor = next((name for name in waiting if name is not None), None)
        return DONOR_NOT_READY.format(donor) if donor is not None else not waiting
    static = False
    if model == 'exmr':
        from .exmr_options import saved
        try:static = saved(root)['static_camera']
        except (OSError, ValueError, KeyError):pass
    # Readiness uses the base motion: a corrected join gets its own correction after joining.
    reason = disabled_reason(model, static, kind, ready)
    return reason, bool(any_source(lambda p: contact_state(p, base).get('manual')))


def export_lines(root, state, files):
    from .people_store import safe_child
    joined = {safe_child(root, j['project']) for p in state['people'] for j in p.get('joins') or []}
    masks = ((target, read_mask(Path(project))) for project, _, _, target in files if Path(project).resolve() in joined)
    return [export_line(Path(target).stem, mask) for target, mask in masks if mask is not None]
