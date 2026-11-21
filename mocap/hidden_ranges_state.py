"""Plain decisions behind the Hidden ranges panel; no Qt, so they test in the shared runtime."""
from pathlib import Path
import numpy as np
from .hidden_ranges import METHODS, OFFSET_LIMIT, donor_take, gap_key, gap_label, method_issue

SUPPORTED = ('initial', 'footmr_initial')
NOT_READY = 'Reconstruct and prepare both neighbouring takes first.'


def disabled_reason(model, static_camera, kind, ready):
    """First reason the Fill button cannot run; `ready` is called last because it reads take files."""
    if model not in ('gvhmr', 'exmr'):return 'Hidden ranges are available for GVHMR and ExMR in this version.'
    if model == 'exmr' and not static_camera:return 'Hidden ranges need a fixed camera in this version.'
    if kind.removeprefix('corrected_') not in SUPPORTED:return 'Hidden ranges support the initial and FootMR results in this version.'
    if not ready():return NOT_READY
    return None


def editable(reason):
    """Choices stay editable only while the project, camera and result are supported."""
    return reason is None or reason == NOT_READY


def _people(people):return [p for p in people if isinstance(p, dict)] if isinstance(people, list) else []


def _shown(choice, gap, people, ident):
    """(method, person, mirror, offset, issue) as the row shows it; a malformed record shows as Do not fill."""
    none = ('none', None, False, 0, None)
    if not isinstance(choice, dict) or choice.get('method', 'none') not in METHODS:return none
    method = choice.get('method', 'none')
    if method == 'none':return none
    # Step 1 showed an over-limit Smooth join as Do not fill; keep that.
    if method == 'smooth':return ('smooth', None, False, 0, None) if method_issue('smooth', gap) is None else none
    p, mirror, off = choice.get('person'), choice.get('mirror', False), choice.get('offset', 0)
    # A copy without a person is what choose() records before one is picked; it stays Copy with an issue.
    if (p is not None and type(p) is not int) or type(mirror) is not bool or type(off) is not int or abs(off) > OFFSET_LIMIT:return none
    return 'copy', p, mirror, off, method_issue('copy', gap, choice, people, ident)


def gap_rows(gaps, choices, fps, people=None, ident=None):
    choices = choices if isinstance(choices, dict) else {};people = _people(people)
    donors = [(p['id'], str(p.get('name') or f"Person {p['id']}")) for p in people if type(p.get('id')) is int and p['id'] != ident]
    rows = []
    for g in gaps:
        method, person, mirror, offset, issue = _shown(choices.get(gap_key(g)), g, people, ident)
        rows.append(dict(key=gap_key(g), label=gap_label(g, fps), method=method, person=person, mirror=mirror, offset=offset,
                         issue=issue, smooth_issue=method_issue('smooth', g), donors=donors))
    return rows


def fill_sources(gaps, choices, people=None, ident=None):
    """The takes the worker would read: both neighbours of every gap that will be filled, and the donor take of a copy."""
    choices = choices if isinstance(choices, dict) else {};people = _people(people);out = {}
    for g in gaps:
        method, person, _, offset, issue = _shown(choices.get(gap_key(g)), g, people, ident)
        if method == 'none' or issue:continue
        out.update({t['project']: t for t in (g['before'], g['after'])})
        if method == 'copy':
            takes = next(p for p in people if p.get('id') == person).get('takes')
            take = donor_take(takes, g, offset);out[take['project']] = take
    return list(out.values())


def can_fill(rows, reason):
    chosen = [r for r in rows if r['method'] != 'none']
    return reason is None and bool(chosen) and not any(r['issue'] for r in chosen)


def choose(choices, key, method, person=None, mirror=False, offset=0):
    if method not in METHODS:raise ValueError(f'Unknown method: {method}')
    if method == 'none':choices.pop(key, None)
    elif method == 'smooth':choices[key] = {'method': 'smooth'}
    else:
        choices[key] = {'method': 'copy', **({} if person is None else {'person': int(person)}),
                        'mirror': bool(mirror), 'offset': max(-OFFSET_LIMIT, min(OFFSET_LIMIT, int(offset)))}


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
    mask = np.asarray(mask)  # 1 is gap_fill.INTERPOLATED, 2 is gap_fill.COPIED
    i, c = int((mask == 1).sum()), int((mask == 2).sum())
    if not c:return f'{stem}: {i} of {len(mask)} frames are interpolated across hidden ranges.'
    return f'{stem}: {i+c} of {len(mask)} frames are filled across hidden ranges ({i} interpolated, {c} copied from another person).'


def panel_state(root, found, choices, model, kind, people=None, ident=None):
    """(disabled reason, manual contact note) for a person's gaps; reads only the takes that would be joined or copied."""
    from .foot_contacts import contact_state
    from .motion_tools import base_result, selected_export
    from .people_store import safe_child
    base = base_result(kind)
    try:sources = [safe_child(root, t['project']) for t in fill_sources(found, choices, people, ident)]
    except (ValueError, KeyError, TypeError):return 'Saved take paths could not be verified. Run Find person again.', False
    def any_source(test):
        try:return any(test(project) for project in sources)
        except (OSError, ValueError, KeyError, TypeError):return None
    static = False
    if model == 'exmr':
        from .exmr_options import saved
        try:static = saved(root)['static_camera']
        except (OSError, ValueError, KeyError):pass
    # Readiness uses the base motion: a corrected join gets its own correction after joining.
    reason = disabled_reason(model, static, kind, lambda: any_source(lambda p: selected_export(p, base) is None) is False)
    return reason, bool(any_source(lambda p: contact_state(p, base).get('manual')))


def export_lines(root, state, files):
    from .people_store import safe_child
    joined = {safe_child(root, j['project']) for p in state['people'] for j in p.get('joins') or []}
    masks = ((target, read_mask(Path(project))) for project, _, _, target in files if Path(project).resolve() in joined)
    return [export_line(Path(target).stem, mask) for target, mask in masks if mask is not None]
