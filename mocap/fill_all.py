"""What one click on Fill all hidden ranges does for the checked people of a scene, and what it says. No Qt.
A range without a method gets one here; a method chosen in Hidden ranges is kept."""
from collections import namedtuple
from copy import deepcopy

from .hidden_ranges import gap_key, gaps, method_issue
from .hidden_ranges_state import choose, gap_rows

AUTO_RATE = 'Automatic fill needs 30 fps takes; this scene has {}.'  # take_join's own sentence
EXPERIMENTAL = 'Experimental support. Generated motion is plausible, not what the person did.'
CORRECTION = 'Apply Correction then runs for all people, because the shared floor of the scene changes.'
METHODS = {'smooth': 'Smooth join', 'copy': 'Copy from person', 'auto': 'Automatic (Experimental)'}

# fills: [(gap, method)]; skipped: [(gap, reason)]; choices: the person's gap_fill with the methods of `fills` recorded.
Person = namedtuple('Person', 'ident name fills skipped choices')
Plan = namedtuple('Plan', 'people reason')  # reason: why no range of this scene can be filled now, or None


def _default(gap, fps, auto_issue):
    """(method, None) for a range nobody chose a method for, or (None, why it stays empty)."""
    if method_issue('smooth', gap) is None:return 'smooth', None
    issue = method_issue('auto', gap) or auto_issue
    if issue:return None, issue
    if fps != 30:return None, AUTO_RATE.format(int(fps) if float(fps).is_integer() else fps)
    return 'auto', None


def plan(root, people, checked, kind, model, fps, hidden, auto_issue=None, text_issue=None, check=None):
    """`hidden`: scene_ghosts.unfilled. `check(found, choices, ident)` is why the fill of these ranges cannot run (the
    Hidden ranges panel's own reason), or None; called with no ranges it gives the scene's reason."""
    if check is None:
        from .hidden_ranges_state import panel_state
        check = lambda found, choices, ident: panel_state(root, found, choices, model, kind, people, ident)[0]
    reason = check([], {}, None)
    if reason:return Plan([], reason)
    out = []
    for person in people if isinstance(people, list) else []:
        if not isinstance(person, dict) or person.get('id') not in checked or not hidden.get(person['id']):continue
        ident = person['id'];waiting = {gap_key(r) for r in hidden[ident]}
        saved = person.get('gap_fill') if isinstance(person.get('gap_fill'), dict) else {}
        choices = deepcopy(saved);fills, skipped, stopped = [], [], None
        for gap in gaps(person.get('takes', [])):
            if gap_key(gap) not in waiting:continue
            row = gap_rows([gap], saved, fps, people, ident, auto_issue, text_issue)[0]
            method, issue, chosen = row['method'], row['issue'], row['method'] != 'none'
            if not chosen:method, issue = _default(gap, fps, auto_issue)
            if issue:
                skipped.append((gap, issue));continue
            trial = deepcopy(saved)
            if not chosen:choose(trial, gap_key(gap), method)
            unready = check([gap], trial, ident)
            if unready:
                skipped.append((gap, unready))
                if chosen:stopped = unready  # the person's fill reads every range with a method, so it would be refused
                continue
            if not chosen:choose(choices, gap_key(gap), method)
            fills.append((gap, method))
        if stopped:
            skipped = sorted(skipped+[(gap, stopped) for gap, _ in fills], key=lambda item: item[0]['first']);fills, choices = [], deepcopy(saved)
        out.append(Person(ident, str(person.get('name') or f'Person {ident}'), fills, skipped, choices))
    return Plan(out, None)


def totals(found):
    """(ranges, people) that the plan fills."""
    filled = [p for p in found.people if p.fills]
    return sum(len(p.fills) for p in filled), len(filled)


def apply(state, found):
    """A copy of people.json's state with the plan's methods recorded for the people it fills."""
    state = deepcopy(state);by_id = {p.ident: p for p in found.people if p.fills}
    for person in state['people']:
        if person.get('id') in by_id:person['gap_fill'] = deepcopy(by_id[person['id']].choices)
    return state


def _count(n, one, many):return f'{n} {one if n == 1 else many}'


def _frames(gap):return f"frames {gap['first']+1}–{gap['end']}"


def summary(ranges, people):
    return f"{_count(ranges, 'hidden range', 'hidden ranges')} of {_count(people, 'person', 'people')} {'is' if ranges == 1 else 'are'} not filled."


def _skipped(found):return [f'{p.name}, {_frames(gap)}: {reason}' for p in found.people for gap, reason in p.skipped]


def confirm_text(found, corrected):
    ranges, people = totals(found)
    parts = [f"Fill {_count(ranges, 'hidden range', 'hidden ranges')} of {_count(people, 'person', 'people')}?",
             '\n'.join(f"{p.name}: " + ', '.join(f'{_frames(gap)} {METHODS[method]}' for gap, method in p.fills) for p in found.people if p.fills)]
    if _skipped(found):parts.append('Not filled:\n' + '\n'.join(_skipped(found)))
    notes = ([EXPERIMENTAL] if any(method == 'auto' for p in found.people for _, method in p.fills) else []) + ([CORRECTION] if corrected else [])
    if notes:parts.append('\n'.join(notes))
    return '\n\n'.join(parts)


def nothing_text(found):
    if found.reason:return found.reason
    return 'No hidden range can be filled. ' + ' '.join(_skipped(found)) if _skipped(found) else 'No hidden range is waiting to be filled.'


def result_text(filled, failed):
    done = 'nobody filled' if not filled else f"{_count(filled, 'person', 'people')} filled"
    if not failed:return f'Fill all hidden ranges finished: {done}.'
    return f'Fill all hidden ranges finished: {done}, {len(failed)} not. ' + ' '.join(f'{name}: {status}' for name, status in failed)


def cancelled_text(filled):
    if not filled:return 'Fill all hidden ranges was cancelled. Nobody was filled.'
    return f"Fill all hidden ranges was cancelled after {_count(filled, 'person', 'people')}."
