"""Plain decisions behind the Hidden ranges panel; no Qt, so they test in the shared runtime."""
from pathlib import Path
import numpy as np
from .hidden_ranges import METHODS, gap_key, gap_label, method_issue

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


def _method(choices, gap):
    method = (choices.get(gap_key(gap)) or {}).get('method', 'none')
    return method if method in METHODS and method_issue(method, gap) is None else 'none'


def gap_rows(gaps, choices, fps):
    return [dict(key=gap_key(g), label=gap_label(g, fps), method=_method(choices, g), issue=method_issue('smooth', g)) for g in gaps]


def fill_sources(gaps, choices):
    """The takes the worker would read: both neighbours of every gap that will be filled."""
    out = {}
    for g in gaps:
        if _method(choices, g) != 'none':out.update({t['project']: t for t in (g['before'], g['after'])})
    return list(out.values())


def can_fill(rows, reason):return reason is None and any(r['method'] != 'none' for r in rows)


def choose(choices, key, method):
    if method not in METHODS:raise ValueError(f'Unknown method: {method}')
    if method == 'none':choices.pop(key, None)
    else:choices[key] = {'method': method}


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


def interpolated_line(stem, mask):
    mask = np.asarray(mask)  # 1 is gap_fill.INTERPOLATED
    return f'{stem}: {int((mask == 1).sum())} of {len(mask)} frames are interpolated across hidden ranges.'


def panel_state(root, found, choices, model, kind):
    """(disabled reason, manual contact note) for a person's gaps; reads only the takes that would be joined."""
    from .foot_contacts import contact_state
    from .motion_tools import base_result, selected_export
    from .people_store import safe_child
    base = base_result(kind)
    try:sources = [safe_child(root, t['project']) for t in fill_sources(found, choices)]
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
    return [interpolated_line(Path(target).stem, mask) for target, mask in masks if mask is not None]
