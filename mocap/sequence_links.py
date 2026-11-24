"""People who continue across neighbouring blocks of a sequence: who can continue whom and why not, the row of each bar,
the automatic link, stale links, and a linked row's state on disk. Pure; the states are `sequence_state.read(block,
model)` for the selected `model`, keyed by block id."""
from __future__ import annotations

from itertools import count
from pathlib import Path

from . import people_store, sequence_state
from .hidden_ranges import method_issue
from .take_import_check import gap_frames as _frames

SAME_MODEL = 'Both blocks must be tracked with the same model.'
FLOWHMR = 'FlowHMR results cannot be joined.'
NO_TAKE = '{} has no finished take in the earlier block.'
SEVERAL = '{} has several takes in this block. Fill their hidden ranges or reconstruct the block in one piece first.'
FILLED = "{}'s take contains generated frames, which cannot be joined again."
NOT_FINISHED = "{}'s take is not finished."


def _person(state, ident) -> sequence_state.Person:
    found = next((p for p in state.people if p.id == ident), None)
    return found or sequence_state.Person(ident, f'Person {ident}', people_store.DEFAULT_COLOR, False)


def rows(sequence, states) -> dict[tuple, int]:
    """The row of every (block id, person id): a linked person keeps the row of the person it continues; the others take
    the lowest rows free in their block, in people-list order."""
    out = {}
    for block in sequence.blocks:
        state = states.get(block.id);people = [p.id for p in state.people] if state else []
        taken = set()
        for ident in people:
            link = sequence.link_into(block.id, ident)
            if link is not None and link.source in out and out[link.source] not in taken:
                out[(block.id, ident)] = out[link.source];taken.add(out[link.source])
        free = (row for row in count() if row not in taken)
        for ident in people:
            if (block.id, ident) not in out:out[(block.id, ident)] = next(free)
    return out


def row_count(rows) -> int:
    return max(rows.values()) + 1 if rows else 0


def refusal(sequence, states, source, target, model) -> str | None:
    """Why `target` (later block id, person id) cannot continue `source` (earlier block id, person id), or None. The join
    continues the earlier person's last take with the later person's one whole take in use."""
    if model == 'flowhmr':return FLOWHMR
    early, late = states.get(source[0]), states.get(target[0])
    if any(state is None or state.kind in ('missing', 'untracked') for state in (early, late)):return SAME_MODEL
    giver, taker = _person(early, source[1]), _person(late, target[1])
    if not any(take.ready for take in giver.takes):return NO_TAKE.format(giver.name)
    if not max(giver.takes, key=lambda take: (take.end, take.first)).ready:return NOT_FINISHED.format(giver.name)
    if len(taker.takes) > 1:return SEVERAL.format(taker.name)
    if taker.takes and taker.takes[0].filled:return FILLED.format(taker.name)
    if not taker.takes or not taker.takes[0].ready:return NOT_FINISHED.format(taker.name)
    return None


def automatic(sequence, states, model) -> list[tuple[tuple, tuple]]:
    """(source, target) of each neighbouring pair of tracked blocks with exactly one person each, neither linked, whose
    later block's automatic link is not decided yet and which `refusal` allows."""
    out = []
    for earlier, later in zip(sequence.blocks, sequence.blocks[1:]):
        a, b = states.get(earlier.id), states.get(later.id)
        if later.id in sequence.decided or not all(s is not None and s.kind == 'tracked' and len(s.people) == 1 for s in (a, b)):
            continue
        source, target = (earlier.id, a.people[0].id), (later.id, b.people[0].id)
        if any(sequence.link_into(*person) or sequence.link_out(*person) for person in (source, target)):continue
        if refusal(sequence, states, source, target, model) is None:out.append((source, target))
    return out


def stale(sequence, states) -> list:
    """Links whose block is gone, or whose block reads a people list with a fingerprint that lacks the person or differs
    from the one the link keeps (a kept print of None matches any list). A block without such a list for the selected
    model (no folder, missing video, an out-of-date list) judges nothing."""
    ids = {block.id for block in sequence.blocks}
    def gone(person, mark):
        state = states.get(person[0])
        if person[0] not in ids:return True
        if state is None or state.kind not in ('found', 'partly', 'tracked') or state.fingerprint is None:return False
        return all(p.id != person[1] for p in state.people) or (mark is not None and mark != state.fingerprint)
    return [link for link in sequence.links if gone(link.source, link.prints[0]) or gone(link.target, link.prints[1])]


def gap_frames(sequence, later_block_id) -> int:
    """The transition before this block in whole frames at 30 fps, as the import takes it."""
    return _frames(sequence.gap(later_block_id))


def gap_issue(method, frames) -> str | None:
    """The Hidden ranges panel's own limit for `method` over a transition of `frames`."""
    return method_issue(method, dict(first=0, end=frames))


def result_stamp(folder) -> list | None:
    """[size, mtime_ns] of the base motion result file the take folder shows, or None without one."""
    from .motion_tools import RESULTS
    try:
        stat = (Path(folder) / RESULTS[sequence_state.selected_kind(folder)][0]).stat()
    except (OSError, KeyError):
        return None
    return [stat.st_size, stat.st_mtime_ns]


def _in_use(root, ident) -> list[Path]:
    """The folders of the takes in use of person `ident` of clip folder `root`; the folder itself without a people list."""
    state = people_store.load_people(root)
    if state is None:return [root.resolve()]
    person = next(p for p in state['people'] if p['id'] == ident)
    return [(root / t['project']).resolve() for t in people_store.active_takes(root, person, sequence_state.selected_kind(root))]


def _built_current(sequence, link, model) -> bool:
    """The link's recorded join still matches it: the source take is still the later person's only take in use, with the
    same result, and the transition in frames and the method are unchanged. `built['take']` is the source take folder,
    absolute or relative to the later block's clip folder."""
    built = link.built
    root = Path(sequence.block(link.target[0]).folders[model]);take = root / built['take']
    stamp = result_stamp(take)
    return built['gap'] == gap_frames(sequence, link.target[0]) and built['method'] == link.method \
        and stamp is not None and list(built['stamp']) == stamp and _in_use(root, link.target[1]) == [take.resolve()]


def _joined(sequence, row, links, model) -> bool:
    """Every link is built and current, and the receiving person (the row's first) uses a current join that owns every
    copy (`built['copy']`: the imported take's folder, absolute or relative to the first block's clip folder)."""
    try:
        if not all(_built_current(sequence, link, model) for link in links):return False
        root = Path(sequence.block(row[0][0]).folders[model])
        person = next(p for p in people_store.load_people(root)['people'] if p['id'] == row[0][1])
        copies = {(root / link.built['copy']).resolve() for link in links}
        joins = {j['project']: j for j in person.get('joins') or []}
        for take in people_store.active_takes(root, person, sequence_state.selected_kind(root)):
            if take.get('join') and copies <= {(root / s).resolve() for s in joins[take['project']]['sources']}:return True
        return False
    except Exception:   # an unreadable people list, a missing folder or a malformed record: not joined
        return False


def status(sequence, states, target, model) -> str:
    """'linked', 'joined' or 'out of date' for the whole linked row of `target` (states are taken for the interface; the
    row's state is read from the first block's people.json and the stamps its links keep)."""
    row = sequence.chain(*target)
    links = [sequence.link_into(*person) for person in row[1:]]
    if not any(link.built for link in links):return 'linked'
    return 'joined' if all(link.built for link in links) and _joined(sequence, row, links, model) else 'out of date'
