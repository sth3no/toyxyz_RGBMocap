"""Join one person of a scene into another and undo it, on the scene's files only (Qt-free).

Every change is first written whole to PENDING and then executed by recover(); each step gives the same result when repeated, so a crash leaves a file that recover() finishes."""
from __future__ import annotations
import copy
import json
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
import numpy as np
from .people_store import load_people, safe_child
from .storage import atomic_json

PENDING = 'people_merge_pending.json'
MODELS = ('gvhmr', 'exmr')
UNREADABLE = 'A previous Continue as did not finish and cannot be read. Run Find person again.'


class Refused(ValueError):
    """str(error) is the message for the user."""


def _name(person):
    return str(person.get('name') or f"Person {person['id']}")


def candidates(state, ident):
    return [(p['id'], _name(p)) for p in state['people'] if p['id'] != ident]


def absorbed(person):
    try:
        return [(r['person']['id'], _name(r['person'])) for r in reversed(person.get('absorbed') or [])]
    except (TypeError, KeyError, AttributeError):
        return []


def _joined_id(record):
    try:
        return record['person']['id']
    except (TypeError, KeyError):
        return None


def _observation(person):
    return person.get('observation_id', person['id'])


def _ranges(frames):
    out = []
    for f in sorted(frames):
        if out and out[-1][1] == f:
            out[-1][1] = f + 1
        else:
            out.append([f, f + 1])
    return out


def _connect(path):
    """A missing database is an error here: sqlite would create an empty one."""
    return sqlite3.connect(f'{Path(path).as_uri()}?mode=rw', uri=True)


def _frames(path, ident):
    with closing(_connect(path)) as db:
        return {r[0] for r in db.execute('SELECT frame FROM observations WHERE person=?', (ident,))}


def _database(root, state):
    try:
        path = safe_child(root, state['observations'])
    except (KeyError, TypeError, ValueError):
        path = None
    if path is None or not path.is_file():
        raise Refused('The person detection data of this scene is missing. Run Find person again.')
    return path


def _person_ids(root, projects):
    ids = {}
    for project in projects:
        try:
            ids[project] = json.loads((safe_child(root, project) / 'project.json').read_text(encoding='utf-8')).get('person_id')
        except (OSError, ValueError):
            ids[project] = None
    return ids


def continue_as(root, source_id, target_id, model, fps):
    """Person B (source) becomes part of person A (target): A gets B's takes and the frames A does not have."""
    root = Path(root)
    recover(root)
    if model not in MODELS:
        raise Refused('Continue as is available for GVHMR and ExMR in this version.')
    state = load_people(root) or dict(people=[])
    people = {p['id']: p for p in state['people']}
    if source_id == target_id or source_id not in people or target_id not in people:
        raise Refused('Choose another person of this scene.')
    database = _database(root, state)
    b, a = people[source_id], people[target_id]
    own, theirs = _frames(database, _observation(a)), _frames(database, _observation(b))
    if len(own & theirs) > round(fps * 0.5):
        raise Refused(f'{_name(b)} and {_name(a)} are both visible in {len(own & theirs)} frames, so they cannot be the same person.')
    moved = _ranges(theirs - own)
    projects = [t['project'] for t in b.get('takes', [])]
    after = own | theirs
    record = dict(person=copy.deepcopy(b), index=state['people'].index(b), takes=projects, frames=moved,
        before={k: a.get(k) for k in ('first', 'last', 'observations')}, person_ids=_person_ids(root, projects))
    state = copy.deepcopy(state)
    a = next(p for p in state['people'] if p['id'] == target_id)
    a['takes'] = [*a.get('takes', []), *copy.deepcopy(b.get('takes', []))]
    if after:
        a.update(first=min(after), last=max(after) + 1, observations=len(after))
    a['absorbed'] = [*(a.get('absorbed') or []), record]
    state['people'] = [p for p in state['people'] if p['id'] != source_id]
    state['active'] = target_id
    return _run(root, state, (_observation(b), _observation(a)), moved, {p: target_id for p in projects}, a)


def separate(root, target_id, source_id):
    """Undo continue_as: person B comes back out of person A with the frames that were moved."""
    root = Path(root)
    recover(root)
    state = load_people(root) or dict(people=[])
    a = next((p for p in state['people'] if p['id'] == target_id), None)
    ids = [_joined_id(r) for r in (a or {}).get('absorbed') or []]
    if source_id not in ids or any(p['id'] == source_id for p in state['people']):
        raise Refused('Choose a person that was joined into this one.')
    _database(root, state)
    state = copy.deepcopy(state)
    a = next(p for p in state['people'] if p['id'] == target_id)
    record = a['absorbed'][ids.index(source_id)]
    later = a['absorbed'][a['absorbed'].index(record) + 1:]
    for r in later:  # they were taken out of a list without this person, so once it is back their place moves down
        if isinstance(r, dict) and type(r.get('index')) is int:
            r['index'] += r['index'] >= record['index']
    if later and isinstance(later[0], dict):  # the earliest record left must keep A's values from before the first merge
        later[0]['before'] = record['before']
    a['absorbed'] = [r for r in a['absorbed'] if r is not record]
    gone = set(record['takes'])
    a['takes'] = [t for t in a.get('takes', []) if t['project'] not in gone]
    if a['absorbed']:  # other joins remain: measure what A has now
        rest = _frames(_database(root, state), _observation(a)) - {f for x, y in record['frames'] for f in range(x, y)}
        if rest:
            a.update(first=min(rest), last=max(rest) + 1, observations=len(rest))
    else:
        del a['absorbed']
        for key, was in record['before'].items():
            if was is None:
                a.pop(key, None)
            else:
                a[key] = was
    people = state['people']
    people.insert(min(record['index'], len(people)), copy.deepcopy(record['person']))
    return _run(root, state, (_observation(a), _observation(record['person'])), record['frames'], record['person_ids'], a)


def _run(root, state, move, frames, person_ids, person):
    """move is (from, to) observation ids; person is the one whose preview tracking is rebuilt."""
    atomic_json(root / PENDING, dict(version=1, state=state, move={'from': move[0], 'to': move[1], 'frames': frames},
        person_ids=person_ids, tracking=dict(preview=person['preview'], observation=_observation(person))))
    recover(root)
    return state


def _relabel(root, pending):
    move = pending['move']
    with closing(_connect(_database(root, pending['state']))) as db, db:
        for first, end in move['frames']:
            db.execute('UPDATE observations SET person=? WHERE person=? AND frame>=? AND frame<?', (move['to'], move['from'], first, end))


def _write_person_ids(root, pending):
    for project, value in pending['person_ids'].items():
        path = safe_child(root, project) / 'project.json'
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if value is None:
            manifest.pop('person_id', None)
        else:
            manifest['person_id'] = value
        atomic_json(path, manifest)


def _write_tracking(root, pending):
    """Same content rules as person_tracking's preview tracking.npz, rebuilt from the database rows."""
    path = safe_child(root, pending['tracking']['preview']) / 'tracking.npz'
    if not path.is_file():
        return
    with np.load(path) as old:
        K, count = old['K'].copy(), len(old['boxes'])
    boxes = np.full((count, 4), np.nan, np.float32)
    observed = np.zeros(count, bool)
    with closing(_connect(_database(root, pending['state']))) as db:
        for frame, *box in db.execute('SELECT frame,x1,y1,x2,y2 FROM observations WHERE person=? ORDER BY frame', (pending['tracking']['observation'],)):
            if 0 <= frame < count:
                boxes[frame] = box
                observed[frame] = True
    temporary = path.with_name(f'tracking.{uuid.uuid4().hex}.tmp')
    try:
        with open(temporary, 'wb') as out:
            np.savez_compressed(out, boxes=boxes, observed=observed, K=K)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish(root, pending):
    atomic_json(root / 'people.json', pending['state'])


def recover(root):
    """Finish an interrupted operation: True when there was one."""
    root = Path(root)
    path = root / PENDING
    if not path.is_file():
        return False
    try:
        pending = json.loads(path.read_text(encoding='utf-8'))
        if pending['version'] != 1:
            raise ValueError
        move, tracking = pending['move'], pending['tracking']
        if not (isinstance(pending['state']['observations'], str) and isinstance(pending['state']['people'], list) and isinstance(pending['person_ids'], dict)
                and all(type(move[k]) is int for k in ('from', 'to')) and type(tracking['observation']) is int and isinstance(tracking['preview'], str)
                and all(isinstance(f, list) and len(f) == 2 and all(type(x) is int for x in f) for f in move['frames'])
                and all(isinstance(k, str) for k in pending['person_ids'])):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise Refused(UNREADABLE) from None
    _database(root, pending['state'])  # before any step, so a missing file leaves everything as it was
    for step in (_relabel, _write_person_ids, _write_tracking, _publish):
        step(root, pending)
    path.unlink()
    return True
