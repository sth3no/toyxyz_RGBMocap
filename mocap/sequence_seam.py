"""The seam between two neighbouring blocks of the sequence: what joining the tracking across it would do and why it
cannot be done yet (`offer`, pure but for the reads of sequence_links), and the card the timeline shows over a seam the
pointer rests on (`SeamCard`). The card only shows the offer and says which button was clicked; sequence_ui.Controller
links and joins."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import sequence_links
from .sequence_view import piece_text
from .theme import set_appearance_style, ui_px

JOIN, OPEN, TRANSITION = 'Join tracking', 'Open joined animation', 'Transition...'
TITLE = '{} → {}'
MISSING = 'The video file of {} is missing.'
NOT_TRACKED = '{} has no finished motion for the selected model. Reconstruct it first.'
PEOPLE = "More than one person here. Zoom in and right-click a person's bar in {} to choose whom it continues, then join here."
WILL = '{} will continue {}.'
PAIR = '{} continues {}.'
STATE = {'linked': '', 'joined': ' Joined.', 'out of date': ' The join is out of date.'}
LENGTH = ' Transition: {}.'
ROW = 'This row runs over {} blocks; all of them are joined together.'
HOVER_MS, LEAVE_MS = 350, 450   # the pointer rests this long before the card shows, and may be off it this long
CARD_STYLE = ('QFrame#seamCard{background:#303030;border:1px solid #477bb7;border-radius:4px}'
              'QLabel#seamTitle{color:#ededed;font-weight:600}')


@dataclass
class Offer:
    title: str                                    # 'one.mp4 → two.mp4'
    lines: list[str]                              # who continues whom, or why nothing can be joined here
    join: bool = False                            # Join tracking would start
    link: tuple | None = None                     # (source, target) the join links first: one person each, not linked
    targets: list = field(default_factory=list)   # the link targets across the seam, in the order of `lines`
    joined: tuple | None = None                   # a target whose row is joined: its animation can be opened


def offer(sequence, states, later_id, model) -> Offer:
    """What the seam before block `later_id` offers for `model`, the sequence's selected model. ValueError for the first
    block, which has no seam before it."""
    i = sequence.index(later_id)
    if i == 0:raise ValueError('The first block has no block before it.')
    earlier, later = sequence.blocks[i - 1], sequence.blocks[i]
    names = [Path(block.video).name for block in (earlier, later)];title = TITLE.format(*names)
    if model == 'flowhmr':return Offer(title, [sequence_links.FLOWHMR])
    for block, name in zip((earlier, later), names):
        state = states.get(block.id)
        if state is None or state.kind == 'missing':return Offer(title, [MISSING.format(name)])
        if state.kind in ('untracked', 'found'):return Offer(title, [NOT_TRACKED.format(name)])
    person = lambda ident:next((p.name for p in states[ident[0]].people if p.id == ident[1]), f'Person {ident[1]}')
    links = [link for link in sequence.model_links() if link.target[0] == later_id]
    if not links:
        before, after = states[earlier.id].people, states[later.id].people
        if len(before) != 1 or len(after) != 1:return Offer(title, [PEOPLE.format(names[1])])
        source, target = (earlier.id, before[0].id), (later.id, after[0].id)
        reason = sequence_links.refusal(sequence, states, source, target, model)
        if reason:return Offer(title, [reason])
        return Offer(title, [WILL.format(person(target), person(source))], True, (source, target))
    lines, pending, joined = [], False, None
    for link in links:
        state = sequence_links.status(sequence, states, link.target, model)
        length = '' if state == 'joined' else LENGTH.format(piece_text(sequence.gap(later_id), link.method))
        lines.append(PAIR.format(person(link.target), person(link.source)) + STATE[state] + length)
        if state == 'joined':joined = joined or link.target
        else:pending = True
    longest = max(len(sequence.chain(*link.target)) for link in links)
    if pending and longest > 2:lines.append(ROW.format(longest))
    return Offer(title, lines, pending, None, [link.target for link in links], joined)


class SeamCard(QFrame):
    """The card over a seam. Its buttons never take the keyboard, so the timeline keeps its keys."""
    join_clicked = Signal()
    open_clicked = Signal()
    transition_clicked = Signal()
    entered = Signal()   # the pointer came onto the card
    left = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('seamCard')
        self.later, self.offer = None, None   # the later block's id of the seam shown, and what it offers
        self.box = QVBoxLayout(self)
        self.title = QLabel();self.title.setObjectName('seamTitle')
        self.text = QLabel();self.text.setWordWrap(True)
        for label in (self.title, self.text):
            label.setTextFormat(Qt.PlainText);self.box.addWidget(label)
        self.actions = QWidget();row = QHBoxLayout(self.actions);row.setContentsMargins(0, 0, 0, 0);row.addStretch(1)
        self.transition_button, self.open_button, self.join_button = (QPushButton(text) for text in (TRANSITION, OPEN, JOIN))
        self.join_button.setObjectName('primary')
        for button, signal in ((self.transition_button, self.transition_clicked), (self.open_button, self.open_clicked),
                               (self.join_button, self.join_clicked)):
            button.setFocusPolicy(Qt.NoFocus);button.clicked.connect(lambda _=False, signal=signal:signal.emit());row.addWidget(button)
        self.box.addWidget(self.actions)
        set_appearance_style(self, CARD_STYLE)
        self.refresh_appearance()
        self.hide()

    def show_offer(self, later, offer):
        """Fill the card with the seam's offer; the caller places and shows it."""
        self.later, self.offer = later, offer
        self.text.setText('\n'.join(offer.lines))
        self.join_button.setVisible(offer.join);self.open_button.setVisible(offer.joined is not None)
        self.transition_button.setVisible(len(offer.targets) == 1)   # the method is each link's own
        self.actions.setVisible(offer.join or offer.joined is not None or len(offer.targets) == 1)
        self.fit()

    def fit(self):
        width = ui_px(340);margins = self.box.contentsMargins()
        if self.offer is not None:   # a long file name keeps both of its ends
            self.title.setText(self.title.fontMetrics().elidedText(self.offer.title, Qt.ElideMiddle, width - margins.left() - margins.right()))
        self.setFixedSize(width, self.box.totalHeightForWidth(width))

    def refresh_appearance(self):
        """Sizes of the current UI scale; the colours follow the theme through set_appearance_style."""
        margin = ui_px(10)
        self.box.setContentsMargins(margin, ui_px(8), margin, ui_px(8));self.box.setSpacing(ui_px(6))
        self.actions.layout().setSpacing(ui_px(6))
        self.fit()

    def enterEvent(self, event):
        self.entered.emit();super().enterEvent(event)

    def leaveEvent(self, event):
        self.left.emit();super().leaveEvent(event)
