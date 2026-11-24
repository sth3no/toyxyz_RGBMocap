"""Small, relocatable person manifests; no inference imports on the GUI thread."""
from __future__ import annotations
import colorsys
import json
import secrets
from pathlib import Path
from . import hidden_ranges
from .storage import atomic_json

VERSION = 1
DEFAULT_COLOR = '#b5b5b5'


def read_people_json(path):
    """A Windows reader must allow the GUI's atomic manifest replacement."""
    import os
    if os.name!='nt':return json.loads(path.read_text(encoding='utf-8'))
    import ctypes,msvcrt
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,
        wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    handle=create(str(path),0x80000000,0x1|0x2|0x4,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:descriptor=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except Exception:
        kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle(handle);raise
    with os.fdopen(descriptor,'r',encoding='utf-8') as source:return json.load(source)


def person_color(number):
    if number == 1:
        return DEFAULT_COLOR
    hue = secrets.randbelow(10000) / 10000
    return '#' + ''.join(f'{round(c*255):02x}' for c in colorsys.hsv_to_rgb(hue, .52, .86))


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Invalid person project path')
    return path


def media_receipt(root):
    stat = (Path(root) / 'input.mp4').stat()
    return [stat.st_size, stat.st_mtime_ns]


def load_people(root):
    root = Path(root)
    path = root / 'people.json'
    if not path.is_file():
        return None
    state = read_people_json(path)
    if state.get('version') != VERSION or state.get('input') != media_receipt(root):
        raise ValueError('Person detection is out of date. Run Find person again.')
    if not isinstance(state.get('people'), list) or len(state['people']) > 256:
        raise ValueError('Invalid person list')
    ids = set()
    for person in state['people']:
        ident = person['id']
        if not isinstance(ident, int) or ident < 1 or ident in ids:
            raise ValueError('Invalid person identity')
        ids.add(ident)
        color = person['color']
        if len(color) != 7 or color[0] != '#':
            raise ValueError('Invalid person color')
        int(color[1:], 16)
        safe_child(root, person['preview'])
        for take in person.get('takes', []):
            safe_child(root, take['project'])
        joins = person.get('joins') or []
        if not isinstance(joins, list) or not all(isinstance(j, dict) and isinstance(j.get('project'), str) for j in joins):
            raise ValueError('Invalid person list')
        for join in joins:
            safe_child(root, join['project'])
    return state


def save_people(root, state):
    atomic_json(Path(root) / 'people.json', state)


def root_for(project, manifest=None):
    project = Path(project).resolve()
    if manifest is None:
        manifest = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    relative = manifest.get('people_root')
    if relative is None:
        return project
    root = (project / relative).resolve()
    # Actor projects must remain descendants of their group after relocation.
    if root == project or not project.is_relative_to(root):
        raise ValueError('Invalid person group')
    return root


def visible_runs(indices, fps, first, end, max_gap_seconds=.5):
    """Split long absences instead of fabricating full-body motion across them."""
    gap = max(1, round(float(fps) * max_gap_seconds))
    start = previous = None
    runs = []
    count = 0
    for raw in indices:
        index = int(raw)
        if not first <= index < end:
            continue
        if previous is not None and index - previous > gap + 1:
            if previous + 1 - start >= 4 and count >= 3:
                runs.append((start, previous + 1))
            start = None
            count = 0
        if start is None:
            start = index
        previous = index
        count += 1
    if start is not None and previous + 1 - start >= 4 and count >= 3:
        runs.append((start, previous + 1))
    return runs


def join_fill(root, person, join):
    """fill.json while the join is current: unchanged source files, and each filled range still a hidden range with the same choice, and unchanged donor files."""
    try:
        sources, first, end = join['sources'], join['first'], join['end']
        if not (isinstance(join['project'], str) and isinstance(sources, list) and sources and all(isinstance(x, str) for x in sources)
                and type(first) is int and type(end) is int):
            return None
        folder = safe_child(root, join['project'])
        if (folder / 'fill_pending.json').exists():
            return None
        fill = read_people_json(folder / 'fill.json')
        if fill.get('version') != 1 or not isinstance(fill.get('kind'), str) or [x['project'] for x in fill['sources']] != sources or (fill['first'], fill['end']) != (first, end):
            return None
        takes = {t['project']: (t['first'], t['end']) for t in person.get('takes', [])}
        for source in fill['sources']:
            if takes.get(source['project']) != (source['first'], source['end']):
                return None
            if not source['files']:
                return None
            base = safe_child(root, source['project'])
            for name, stamp in source['files'].items():
                stat = (base / name).stat()
                if [stat.st_size, stat.st_mtime_ns] != list(stamp):
                    return None
        found = {hidden_ranges.gap_key(g) for g in hidden_ranges.gaps(person.get('takes', []))}
        choices = person.get('gap_fill') or {}
        for gap in fill['gaps']:
            key = hidden_ranges.gap_key(gap)
            if key not in found or hidden_ranges.choice_params(choices.get(key)) != (gap['method'], gap.get('params') or {}):
                return None
        # Frames two parts share are blended without a method; they must still be shared, joinable and marked as parts.
        shared = {(s['first'], s['end']) for s in hidden_ranges.seams(person.get('takes', []))
                  if s.get('kind') == 'overlap' and s['joinable'] and hidden_ranges.marked(s, choices)}
        for overlap in fill.get('overlaps') or []:
            if (overlap['first'], overlap['end']) not in shared:
                return None
        for donor in fill.get('donors') or []:
            if not donor['files']:
                return None
            base = safe_child(root, donor['project'])
            for name, stamp in donor['files'].items():
                stat = (base / name).stat()
                if [stat.st_size, stat.st_mtime_ns] != list(stamp):
                    return None
        return fill
    except Exception:
        return None


def join_current(root, person, join):
    return join_fill(root, person, join) is not None


def join_kinds(fill):
    """Base results a joined take provides; a fill.json without `kinds` provides only its `kind`."""
    kinds = fill.get('kinds')
    return kinds if isinstance(kinds, list) else [fill['kind']]


def active_takes(root, person, kind=None):
    """Takes in use; with `kind`, a current join replaces its sources only for a result it provides."""
    takes = person.get('takes', [])
    joins = person.get('joins')
    base = None if kind is None else kind.removeprefix('corrected_')
    current, owned = [], set()
    for j in joins if isinstance(joins, list) else []:
        fill = join_fill(root, person, j)
        if fill is not None and (base is None or base in join_kinds(fill)) and owned.isdisjoint(j['sources']):
            current.append(j)
            owned.update(j['sources'])
    if not current:
        return takes
    out = [t for t in takes if t['project'] not in owned]
    out += [{'project': j['project'], 'first': j['first'], 'end': j['end'], 'join': True} for j in current]
    return sorted(out, key=lambda t: t['first'])


def replaced_projects(root, person, kind=None):
    """Source takes a current join stands in for."""
    takes = person.get('takes', [])
    if not person.get('joins') or not isinstance(takes, list):
        return set()
    used = {t['project'] for t in active_takes(root, person, kind)}
    return {t['project'] for t in takes if t['project'] not in used}


def owned_projects(person):
    return [t['project'] for t in person.get('takes', [])] + [j['project'] for j in person.get('joins') or []]


def joined_project_current(project, kind=None):
    """None for an ordinary take folder, else whether its joined take is in use (for `kind`, if given)."""
    project = Path(project).resolve()
    if not (project / 'fill.json').is_file():
        return None
    try:
        root = root_for(project)
        state = load_people(root)
        for person in state['people']:
            for join in person.get('joins') or []:
                if safe_child(root, join['project']) == project:
                    return any(t['project'] == join['project'] for t in active_takes(root, person, kind))
    except Exception:
        pass
    return False
