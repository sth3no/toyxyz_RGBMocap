from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import storage

VERSION = 1
MIN_SECONDS = 4 / 30
_SAME = 1e-6
_SHORT = 'A block must be at least 4 frames long.'
_BAD = 'This is not a sequence file.'
_NEWER = 'This sequence was saved by a newer version of RGB MoCap.'
METHODS = ('smooth', 'auto')   # the keys of Smooth join and Automatic in hidden_ranges.METHODS
DEFAULT_METHOD = METHODS[0]
DEFAULT_GAP = 1.0
_FRAME = 1 / 30
_NEIGHBOUR = 'A person can continue only a person of the block before.'


@dataclass
class Block:
    id: str
    video: str
    start: float
    end: float
    length: float
    folders: dict[str, str] = field(default_factory=dict)
    review: dict | None = None


@dataclass
class Link:
    """`target` (later block id, person id) continues `source` (earlier block id, person id) after the transition gap."""
    source: tuple
    target: tuple
    method: str = DEFAULT_METHOD
    prints: tuple = (None, None)   # people-list fingerprints of the earlier and the later block
    built: dict | None = None      # set by Join: copy, take, stamp, gap, method


def _method(method) -> str:
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    return method


def _person(item) -> tuple:
    if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str) \
            or isinstance(item[1], bool) or not isinstance(item[1], (int, str)):
        raise ValueError(_BAD)
    return item[0], item[1]


def _link(item) -> Link:
    if not isinstance(item, dict):
        raise ValueError(_BAD)
    prints, built = item.get('prints', [None, None]), item.get('built')
    if not isinstance(prints, list) or len(prints) != 2 or not (built is None or isinstance(built, dict)):
        raise ValueError(_BAD)
    return Link(_person(item.get('source')), _person(item.get('target')), item.get('method'), tuple(prints), built)


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(_BAD)
    return float(value)


def _block(item) -> Block:
    if not isinstance(item, dict):
        raise ValueError(_BAD)
    folders = item.get('folders', {})
    if not isinstance(item.get('id'), str) or not isinstance(item.get('video'), str) or not isinstance(folders, dict) \
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in folders.items()):
        raise ValueError(_BAD)
    start, end, length = _number(item.get('start')), _number(item.get('end')), _number(item.get('length'))
    if start < 0 or end <= start:
        raise ValueError(_BAD)
    review = item.get('review')
    if review is not None:
        review = _review(review, start, end)
    return Block(item['id'], item['video'], start, end, length, dict(folders), review)


def _review(item, start, end):
    from .motion_tools import RESULTS

    if not isinstance(item, dict):
        raise ValueError('Invalid review settings.')
    first = _number(item.get('start'))
    last = _number(item.get('end'))
    result = item.get('result')
    if not (math.isfinite(first) and math.isfinite(last) and start - _SAME <= first < last <= end + _SAME):
        raise ValueError('Choose a nonempty review range inside the block.')
    if result is not None and (not isinstance(result, str) or result not in RESULTS):
        raise ValueError('Unknown review result.')
    return {'start': max(first, start), 'end': min(last, end), 'result': result}


class Sequence:
    """Several videos as blocks on one timeline; each block is a range of its video."""

    def __init__(self, blocks: list[Block] | None = None, path: Path | None = None):
        self.blocks: list[Block] = blocks if blocks is not None else []
        self.path = path
        self.links: list[Link] = []
        self.gaps: dict[str, float] = {}   # later block id -> transition seconds, used while a link crosses into it
        self.decided: set[str] = set()     # later block ids whose automatic link was already decided
        self.dropped: list[Link] = []      # links the edits dropped; the controller reads and clears it

    def add(self, video, length, start=0.0, end=None, folders=None) -> Block:
        block = Block(uuid.uuid4().hex[:12], str(video), float(start), float(length if end is None else end),
                      float(length), dict(folders or {}))
        self.blocks.append(block)
        return block

    def index(self, id) -> int:
        for i, block in enumerate(self.blocks):
            if block.id == id:
                return i
        raise KeyError(id)

    def block(self, id) -> Block:
        return self.blocks[self.index(id)]

    def cut(self, id, seconds) -> tuple[Block, Block]:
        i = self.index(id); block = self.blocks[i]
        if seconds - block.start < MIN_SECONDS or block.end - seconds < MIN_SECONDS:
            raise ValueError(_SHORT)
        second = Block(uuid.uuid4().hex[:12], block.video, float(seconds), block.end, block.length)
        block.end = float(seconds); block.folders = {}; block.review = None
        self.blocks.insert(i + 1, second)
        self._drop(self.links_touching(id))
        return block, second

    def delete(self, id) -> None:
        del self.blocks[self.index(id)]
        self.gaps.pop(id, None); self.decided.discard(id)
        self._drop(self.links_touching(id))

    def move(self, id, index) -> None:
        before = [block.id for block in self.blocks]
        block = self.blocks.pop(self.index(id))
        self.blocks.insert(max(0, min(int(index), len(self.blocks))), block)
        if [block.id for block in self.blocks] != before:
            self._drop(self.links_touching(id) + [link for link in self.links if not self._neighbours(link)])

    def resize(self, id, start, end) -> None:
        block = self.block(id)
        start, end = max(0.0, min(float(start), block.length)), max(0.0, min(float(end), block.length))
        if end - start < MIN_SECONDS:
            raise ValueError(_SHORT)
        if abs(start - block.start) > _SAME or abs(end - block.end) > _SAME:
            block.folders = {}
            block.review = None
            self._drop(self.links_touching(id))
        block.start, block.end = start, end

    def remember(self, id, model, folder, start, end) -> None:
        block = self.block(id)
        if abs(start - block.start) > _SAME or abs(end - block.end) > _SAME:
            block.folders = {}
            block.review = None
            self._drop(self.links_touching(id))
        block.start, block.end = float(start), float(end)
        block.folders[model] = str(folder)

    def forget(self, id, model) -> None:
        self.block(id).folders.pop(model, None)

    def set_review(self, id, value) -> None:
        block = self.block(id)
        block.review = None if value is None else _review(value, block.start, block.end)

    def link(self, source, target, method=DEFAULT_METHOD, prints=(None, None)) -> Link:
        source, target = tuple(source), tuple(target)
        if self.index(target[0]) != self.index(source[0]) + 1:
            raise ValueError(_NEIGHBOUR)
        link = Link(source, target, _method(method), tuple(prints))
        self._add(link)
        self.gaps.setdefault(target[0], DEFAULT_GAP); self.decided.add(target[0])
        return link

    def _add(self, link) -> None:
        self.links = [old for old in self.links if old.target != link.target and old.source != link.source] + [link]

    def unlink(self, target) -> Link | None:
        link = self.link_into(*target)
        if link is not None:
            self.links.remove(link)
        return link

    def link_into(self, block_id, person) -> Link | None:
        return next((link for link in self.links if link.target == (block_id, person)), None)

    def link_out(self, block_id, person) -> Link | None:
        return next((link for link in self.links if link.source == (block_id, person)), None)

    def chain(self, block_id, person) -> list[tuple]:
        """The whole linked row of this person, from its first block to its last."""
        first = (block_id, person)
        while (link := self.link_into(*first)) is not None:
            first = link.source
        row = [first]
        while (link := self.link_out(*row[-1])) is not None:
            row.append(link.target)
        return row

    def links_touching(self, block_id) -> list[Link]:
        return [link for link in self.links if block_id in (link.source[0], link.target[0])]

    def gap(self, later_block_id) -> float:
        """The transition before this block: its length while a link crosses into it, else none."""
        if not any(link.target[0] == later_block_id for link in self.links):
            return 0.0
        return self.gaps.get(later_block_id, DEFAULT_GAP)

    def set_gap(self, later_block_id, seconds) -> None:
        if float(seconds) < _FRAME - _SAME:
            raise ValueError('The transition must be at least one frame long.')
        self.gaps[later_block_id] = float(seconds)

    def set_method(self, target, method) -> None:
        link = self.link_into(*target)
        if link is None:
            raise KeyError(target)
        link.method = _method(method)

    def _neighbours(self, link) -> bool:
        ids = [block.id for block in self.blocks]
        return link.source[0] in ids and link.target[0] in ids and ids.index(link.target[0]) == ids.index(link.source[0]) + 1

    def _drop(self, links) -> None:
        gone = [link for link in self.links if any(link is other for other in links)]
        self.links = [link for link in self.links if not any(link is other for other in gone)]
        self.dropped.extend(gone)

    def _layout(self) -> tuple[list[float], float]:
        out, at = [], 0.0
        for i, block in enumerate(self.blocks):
            if i:
                at += self.gap(block.id)
            out.append(at); at += block.end - block.start
        return out, at

    def starts(self) -> list[float]:
        return self._layout()[0]

    def total(self) -> float:
        return self._layout()[1]

    def locate(self, seconds) -> tuple[Block, float] | None:
        if not self.blocks:
            return None
        starts = self.starts()
        for i, (block, at) in enumerate(zip(self.blocks, starts)):
            if i and seconds < at:   # inside the transition gap: the earlier block's end
                return self.blocks[i - 1], self.blocks[i - 1].end
            if seconds < at + block.end - block.start:
                return block, block.start + max(0.0, seconds - at)
        return self.blocks[-1], self.blocks[-1].end

    def position(self, id, source_seconds) -> float:
        block = self.block(id)
        return self.starts()[self.index(id)] + max(block.start, min(source_seconds, block.end)) - block.start

    def plain(self) -> bool:
        return len(self.blocks) == 1   # one block, as today's single video or project, whatever its range

    def to_dict(self) -> dict:
        links = [{'source': list(l.source), 'target': list(l.target), 'method': l.method, 'prints': list(l.prints),
                  **({'built': dict(l.built)} if l.built is not None else {})} for l in self.links]
        return {'version': VERSION, 'blocks': [{'id': b.id, 'video': b.video, 'start': b.start, 'end': b.end,
                                                'length': b.length, 'folders': dict(b.folders),
                                                **({'review': dict(b.review)} if b.review is not None else {})}
                                               for b in self.blocks],
                **({'links': links} if links else {}), **({'gaps': dict(self.gaps)} if self.gaps else {}),
                **({'decided': sorted(self.decided)} if self.decided else {})}

    @classmethod
    def from_dict(cls, data) -> Sequence:
        if not isinstance(data, dict) or isinstance(data.get('version'), bool) or not isinstance(data.get('version'), int):
            raise ValueError(_BAD)
        if data['version'] > VERSION:
            raise ValueError(_NEWER)
        links, gaps, decided = data.get('links', []), data.get('gaps', {}), data.get('decided', [])
        if not isinstance(data.get('blocks'), list) or not isinstance(links, list) or not isinstance(gaps, dict) \
                or not isinstance(decided, list) or not all(isinstance(k, str) for k in [*gaps, *decided]):
            raise ValueError(_BAD)
        sequence = cls([_block(item) for item in data['blocks']])
        ids = {block.id for block in sequence.blocks}
        gaps = {k: _number(v) for k, v in gaps.items()}
        sequence.gaps = {k: v for k, v in gaps.items() if k in ids and math.isfinite(v) and v >= _FRAME - _SAME}
        sequence.decided = {k for k in decided if k in ids}
        for link in map(_link, links):   # a link to a missing block, between non-neighbours or by an unknown method is dropped
            if link.method in METHODS and sequence._neighbours(link):
                sequence._add(link)
        return sequence

    def save(self, path=None) -> None:
        if path is not None:
            self.path = Path(path)
        storage.atomic_json(self.path, self.to_dict())

    @classmethod
    def load(cls, path) -> Sequence:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            raise ValueError(_BAD) from None
        sequence = cls.from_dict(data); sequence.path = path
        return sequence
