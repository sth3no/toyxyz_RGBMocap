"""Tutorial chapters as data: what each step highlights, what it says and when it is done.

Every statement of a step comes from docs/USER_GUIDE.md. A step reads the window's state and never changes it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    text: object           # str, or callable(window) -> str
    targets: object        # callable(window) -> widgets and/or QRects in window coordinates
    done: object = None    # callable(window, events) -> bool; None: only Next moves on
    skip: object = None    # callable(window) -> bool
    waiting: str = ''      # status line while `done` is not met
    action: object = None  # callable(window) -> (label, callback) or None
    retry: object = None   # callable(window, events) -> str or None; a text sends the guide one step back with that note
    card: str = 'auto'     # 'center' keeps the card in the middle, clear of an opened menu


@dataclass(frozen=True)
class Chapter:
    key: str
    title: str
    steps: tuple
    blocked: object = None  # callable(window) -> str or None; a text holds the chapter and is shown instead of the step
    events: object = None   # callable(window) -> {name: (signal, condition callable(window) -> bool)}
