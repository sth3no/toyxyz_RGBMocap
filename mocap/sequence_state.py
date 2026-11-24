"""What a sequence block looks like on disk for one model: its state and its people rows, read from its remembered clip
folder (project.json, people.json, take readiness as Reconstruct in parts decides it). Pure reads; never raises."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import people_store

FPS = 30.  # the working clip's frame rate, for a block without a readable clip folder
_SAME = 1e-6
LABELS = {'missing': 'video missing', 'untracked': 'not tracked', 'partly': 'partly tracked', 'tracked': 'tracked'}


@dataclass
class Take:
    first: int
    end: int
    ready: bool
    joined: bool


@dataclass
class Person:
    id: int
    name: str
    color: str
    checked: bool
    takes: list[Take] = field(default_factory=list)


@dataclass
class State:
    kind: str
    reason: str
    people: list[Person]
    frames: int
    fps: float
    label: str


def kind_for(model) -> str:
    """The base result a take of `model` needs to count as finished, with the window's default options (HTD Refine, FootMR
    and HandFlow on for GVHMR); Reconstruct in parts decides on the same base of `selected_result`."""
    from .motion_tools import base_result, result_kind
    gvhmr = model == 'gvhmr'
    return base_result(result_kind(gvhmr, gvhmr, False, handflow=gvhmr))


def _state(kind, reason='', people=(), frames=0, fps=FPS) -> State:
    people = list(people)
    label = LABELS.get(kind) or ('1 person' if len(people) == 1 else f'{len(people)} people')
    return State(kind, reason, people, frames, fps, label)


def _ready(check, project) -> bool:
    try:return bool(check(project))
    except Exception:return False


def _people(root, state, kind) -> list[Person]:
    from .parts_runner import ready_result
    check = ready_result(root, kind)
    return [Person(p['id'], str(p.get('name') or f"Person {p['id']}"), p['color'], bool(p.get('checked')),
                   [Take(t['first'], t['end'], _ready(check, t['project']), bool(t.get('join'))) for t in people_store.active_takes(root, p, kind)])
            for p in state['people']]


def _manifest(block, model):
    """(kind, reason, manifest): the remembered clip folder's project.json when it matches the block's range, else the
    block's state without it."""
    folder = block.folders.get(model)
    if folder is None:
        return ('missing', f'The video file is missing: {block.video}', None) if not Path(block.video).is_file() else ('untracked', '', None)
    try:
        manifest = json.loads((Path(folder)/'project.json').read_text(encoding='utf-8'))
        if not all(type(manifest[k]) in (int, float) for k in ('start', 'end', 'fps')) or type(manifest['frames']) is not int:raise TypeError
    except Exception:return 'untracked', f'The saved results are missing: {folder}', None
    if abs(manifest['start']-block.start) > _SAME or abs(manifest['end']-block.end) > _SAME:
        return 'untracked', 'The saved results belong to another range of the video.', None
    return None, '', manifest


def read(block, model) -> State:
    """The state of `block` for `model`: missing, untracked, found (people, none finished), partly or tracked."""
    kind, reason, manifest = _manifest(block, model)
    if manifest is None:return _state(kind, reason, frames=round((block.end-block.start)*FPS))
    root, frames, fps, base = Path(block.folders[model]), manifest['frames'], float(manifest['fps']), kind_for(model)
    try:
        state = people_store.load_people(root)
        people = None if state is None else _people(root, state, base)
    except Exception as error:   # the person list's own reason; anything else in it is an invalid list too
        return _state('found', str(error) if isinstance(error, ValueError) else 'Invalid person list', frames=frames, fps=fps)
    if people is None:
        from .motion_tools import selected_export
        if not _ready(lambda folder: selected_export(folder, base) is not None, root):return _state('untracked', frames=frames, fps=fps)
        return _state('tracked', people=[Person(1, 'Person', people_store.DEFAULT_COLOR, True, [Take(0, frames, True, False)])], frames=frames, fps=fps)
    checked = [p for p in people if p.checked]
    if checked and all(p.takes and all(t.ready for t in p.takes) for p in checked):kind = 'tracked'
    elif any(t.ready for p in checked for t in p.takes):kind = 'partly'
    else:kind = 'found'
    return _state(kind, people=people, frames=frames, fps=fps)
