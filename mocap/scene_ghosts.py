"""Hidden ranges nobody filled yet, and where the 3D view shows a checked person inside one: as a ghost standing at the
last captured frame before the range. No Qt. The ghost is drawn only; export, correction and the stream never see it."""
from .hidden_ranges import gaps
from .people_store import active_takes, safe_child

MODELS = ('gvhmr', 'exmr')  # the models whose hidden ranges can be filled


def unfilled(root, people, kind, model):
    """{person id: [dict(first, end, before, path)]}: the gaps between the takes in use for `kind`, each with the folder
    of the take before it. A current joined take stands in for its sources, so a range it fills is not listed. Empty for a
    model without Hidden ranges."""
    if model not in MODELS:return {}
    out = {}
    for person in people if isinstance(people, list) else []:
        # The folder is resolved here, once: the view looks for ghosts several times in every paint.
        try:found = [dict(first=g['first'], end=g['end'], before=g['before'], path=safe_child(root, g['before']['project']))
                     for g in gaps(active_takes(root, person, kind))]
        except (KeyError, TypeError, AttributeError, ValueError, OSError):continue
        if found:out[person['id']] = found
    return out


def count(ranges, checked):
    """(ranges, people) that are unfilled among the checked people."""
    mine = [found for ident, found in (ranges or {}).items() if ident in checked]
    return sum(len(found) for found in mine), len(mine)


def ghost(scene, ident, seconds, ranges):
    """(take, frame, (first, end)) of the person's ghost at `seconds`, or None: the take before the hidden range that
    holds this time, at its last frame. None while any take of the person shows them."""
    for found in ranges:
        take = scene.by_project.get(found.get('path'))
        if take is None or take.ident != ident:continue
        if take.end-1e-6 <= seconds < take.end+(found['end']-found['first'])/take.fps-1e-6:
            return None if scene.take_at(ident, seconds) is not None else (take, take.frames-1, (found['first'], found['end']))
    return None


def ghosts(scene, seconds, visible, hidden):
    """The ghosts of the checked people at `seconds`, by person id."""
    if not hidden:return []
    return [entry for ident in sorted(visible) if ident in hidden and (entry := ghost(scene, ident, seconds, hidden[ident])) is not None]
