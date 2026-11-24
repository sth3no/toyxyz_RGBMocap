from __future__ import annotations

import json
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


@dataclass
class Block:
    id: str
    video: str
    start: float
    end: float
    length: float
    folders: dict[str, str] = field(default_factory=dict)


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
    return Block(item['id'], item['video'], start, end, length, dict(folders))


class Sequence:
    """Several videos as blocks on one timeline; each block is a range of its video."""

    def __init__(self, blocks: list[Block] | None = None, path: Path | None = None):
        self.blocks: list[Block] = blocks if blocks is not None else []
        self.path = path

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
        block.end = float(seconds); block.folders = {}
        self.blocks.insert(i + 1, second)
        return block, second

    def delete(self, id) -> None:
        del self.blocks[self.index(id)]

    def move(self, id, index) -> None:
        block = self.blocks.pop(self.index(id))
        self.blocks.insert(max(0, min(int(index), len(self.blocks))), block)

    def resize(self, id, start, end) -> None:
        block = self.block(id)
        start, end = max(0.0, min(float(start), block.length)), max(0.0, min(float(end), block.length))
        if end - start < MIN_SECONDS:
            raise ValueError(_SHORT)
        if abs(start - block.start) > _SAME or abs(end - block.end) > _SAME:
            block.folders = {}
        block.start, block.end = start, end

    def remember(self, id, model, folder, start, end) -> None:
        block = self.block(id)
        if abs(start - block.start) > _SAME or abs(end - block.end) > _SAME:
            block.folders = {}
        block.start, block.end = float(start), float(end)
        block.folders[model] = str(folder)

    def forget(self, id, model) -> None:
        self.block(id).folders.pop(model, None)

    def starts(self) -> list[float]:
        out, at = [], 0.0
        for block in self.blocks:
            out.append(at); at += block.end - block.start
        return out

    def total(self) -> float:
        return sum(block.end - block.start for block in self.blocks)

    def locate(self, seconds) -> tuple[Block, float] | None:
        if not self.blocks:
            return None
        starts = self.starts()
        for block, at in zip(self.blocks, starts):
            if seconds < at + block.end - block.start:
                return block, block.start + max(0.0, seconds - at)
        return self.blocks[-1], self.blocks[-1].end

    def position(self, id, source_seconds) -> float:
        block = self.block(id)
        return self.starts()[self.index(id)] + max(block.start, min(source_seconds, block.end)) - block.start

    def plain(self) -> bool:
        return len(self.blocks) == 1   # one block, as today's single video or project, whatever its range

    def to_dict(self) -> dict:
        return {'version': VERSION, 'blocks': [{'id': b.id, 'video': b.video, 'start': b.start, 'end': b.end,
                                                'length': b.length, 'folders': dict(b.folders)} for b in self.blocks]}

    @classmethod
    def from_dict(cls, data) -> Sequence:
        if not isinstance(data, dict) or isinstance(data.get('version'), bool) or not isinstance(data.get('version'), int):
            raise ValueError(_BAD)
        if data['version'] > VERSION:
            raise ValueError(_NEWER)
        if not isinstance(data.get('blocks'), list):
            raise ValueError(_BAD)
        return cls([_block(item) for item in data['blocks']])

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
