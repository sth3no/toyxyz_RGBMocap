"""Join one person of a scene into another and undo it, on the scene's files only (Qt-free).

Every change is first written whole to PENDING and then executed step by step; each step gives the same result when repeated, so a file left by a failure or a crash is finished by recover()."""
from __future__ import annotations
import copy
import json
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
import numpy as np
from .people_store import load_people, read_people_json, safe_child
from .storage import atomic_json

PENDING = 'people_merge_pending.json'
DISCARDED = 'people_merge_pending.discarded.json'
SET_ASIDE = 'people_merge_pending.unreadable.json'
MODELS = ('gvhmr', 'exmr')
UNREADABLE = (f'A previous Continue as did not finish and its record cannot be read. It was set aside as {SET_ASIDE}. '
              'Check the people of this scene; Find person can be run again to start over.')


class Refused(ValueError):
    """str(error) is the message for the user."""


class Unfinished(Refused):
    """Part of the change is on disk; the pending file stays so that recover() finishes it."""


def _name(person):
    return str(person.get('name') or f"Person {person['id']}")


def candidates(state, ident):
    return [(p['id'], _name(p)) for p in state['people'] if p['id'] != ident]


def absorbed(person):
    records = person.get('absorbed') if isinstance(person, dict) else None
    out = []
    for r in reversed(records) if isinstance(records, list) else ():
        try:
            out.append((r['person']['id'], _name(r['person'])))
        except (TypeError, KeyError, AttributeError):
            continue
    return out


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


def _manifest(path):
    """A take's project.json as a dict, None when it is missing or damaged; a file that cannot be opened is an error."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _tracking(path):
    """(K, frame count) of a preview tracking.npz; raises when it is damaged."""
    with open(path, 'rb') as file, np.load(file) as old:  # own handle: np.load leaves its own open when the file is damaged
        if 'observed' not in old.files:
            raise KeyError('observed')
        return old['K'].copy(), len(old['boxes'])


def _readable(root, projects, preview):
    """Refuse before the pending file exists when a file a step reads is damaged, so nothing is half applied."""
    for project in projects:
        path = safe_child(root, project) / 'project.json'
        if path.is_file() and _manifest(path) is None:
            raise Refused(f'{Path(project, "project.json").as_posix()} cannot be read, so the people of this scene were not changed.')
    path = safe_child(root, preview) / 'tracking.npz'
    if path.is_file():
        try:
            _tracking(path)
        except Exception:
            raise Refused(f'{Path(preview, "tracking.npz").as_posix()} cannot be read, so the people of this scene were not changed.') from None


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
    _readable(root, projects, a['preview'])
    after = own | theirs
    record = dict(person=copy.deepcopy(b), index=state['people'].index(b), takes=projects, frames=moved,
        before={k: a.get(k) for k in ('first', 'last', 'observations', 'checked')}, person_ids=_person_ids(root, projects))
    state = copy.deepcopy(state)
    a = next(p for p in state['people'] if p['id'] == target_id)
    a['takes'] = [*a.get('takes', []), *copy.deepcopy(b.get('takes', []))]
    if after:
        a.update(first=min(after), last=max(after) + 1, observations=len(after))
    if b.get('checked'):  # otherwise the motion that was shown disappears from the view
        a['checked'] = True
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
    _readable(root, a['absorbed'][ids.index(source_id)].get('person_ids') or {}, a['preview'])
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
    pending = dict(version=1, state=state, move={'from': move[0], 'to': move[1], 'frames': frames},
        person_ids=person_ids, tracking=dict(preview=person['preview'], observation=_observation(person)))
    atomic_json(root / PENDING, pending)
    try:
        _relabel(root, pending)
    except Exception:  # one transaction that did not commit: nothing was applied, so there is nothing to finish
        (root / PENDING).unlink(missing_ok=True)
        raise
    try:
        for step in (_write_person_ids, _write_tracking, _publish, _retire):
            step(root, pending)
    except Exception as error:
        text = str(error).rstrip()
        raise Unfinished(f'The change could not be finished: {text}{"" if text.endswith(".") else "."} '
                         'It will be finished when this scene is opened again.') from error
    return state


def _relabel(root, pending):
    move = pending['move']
    with closing(_connect(_database(root, pending['state']))) as db, db:
        for first, end in move['frames']:
            db.execute('UPDATE observations SET person=? WHERE person=? AND frame>=? AND frame<?', (move['to'], move['from'], first, end))


def _write_person_ids(root, pending):
    for project, value in pending['person_ids'].items():
        path = safe_child(root, project) / 'project.json'
        manifest = _manifest(path)
        if manifest is None:  # damaged after the start: left as it is, like a missing one
            continue
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
    try:
        K, count = _tracking(path)
    except OSError:  # cannot be opened now: recover() tries again
        raise
    except Exception:  # damaged after the start: left as it is, like a missing one
        return
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


def _retire(root, pending):
    (root / PENDING).unlink()


def _observations(root):
    try:
        return read_people_json(root / 'people.json')['observations']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def recover(root):
    """Finish an interrupted operation: True when there was one."""
    root = Path(root)
    path = root / PENDING
    if not path.is_file():
        return False
    raw = path.read_bytes()  # a file that cannot be opened now stays for a later try
    try:
        pending = json.loads(raw.decode('utf-8'))
        if pending['version'] != 1:
            raise ValueError
        move, tracking = pending['move'], pending['tracking']
        if not (isinstance(pending['state']['observations'], str) and isinstance(pending['state']['people'], list) and isinstance(pending['person_ids'], dict)
                and all(type(move[k]) is int for k in ('from', 'to')) and type(tracking['observation']) is int and isinstance(tracking['preview'], str)
                and all(isinstance(f, list) and len(f) == 2 and all(type(x) is int for x in f) for f in move['frames'])
                and all(isinstance(k, str) for k in pending['person_ids'])):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        os.replace(path, root / SET_ASIDE)  # kept for inspection, but it no longer blocks every change
        raise Refused(UNREADABLE) from None
    if pending['state']['observations'] != _observations(root):  # Find person ran again since: replaying it would damage the new list
        os.replace(path, root / DISCARDED)
        return False
    _database(root, pending['state'])  # before any step, so a missing file leaves everything as it was
    for step in (_relabel, _write_person_ids, _write_tracking, _publish, _retire):
        step(root, pending)
    return True
