"""In-window tutorial: highlights a control, explains it and waits for the user's step.

The chapters are data in tutorial_steps. MainWindow never starts the tutorial by itself; app.py and the Help menu do.
Its state is an installation-local convenience in the Resource Safety config/local.json, never project data.
"""
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from .theme import is_light_theme, set_appearance_style, theme_color, ui_px

STATE = 'tutorial'


def load_state():
    """The saved state; {} when nothing usable is saved, None when the settings cannot be read."""
    from .resource_safety import path, read_settings
    try:value = read_settings(path()).get(STATE)
    except (OSError, ValueError):return None
    return value if isinstance(value, dict) else {}


def save_state(**changes):
    from .resource_safety import update
    state = load_state()
    if state is None:return False  # an unreadable file is kept, not replaced
    try:update({STATE: {**state, **changes}})
    except (OSError, ValueError):return False
    return True


def mark_done(key):
    done = (load_state() or {}).get('done')
    done = [k for k in done if isinstance(k, str)] if isinstance(done, list) else []
    return save_state(done=done if key in done else done + [key])

DIM = True  # False draws only the ring, should the dim misbehave over the OpenGL viewport
CARD_STYLE = ('QFrame#tutorialCard{background:#303030;border:1px solid #477bb7;border-radius:4px}'
              'QLabel#tutorialHeader,QLabel#tutorialStatus{color:#a0a0a0}'
              'QLabel#tutorialTitle{color:#ededed;font-size:14px;font-weight:600}')


def place_card(window, card, target, margin):
    """Top-left of the card: beside the target where there is room (right, left, below, above), always inside the window."""
    if target is None:return QPoint((window.width() - card.width()) // 2, (window.height() - card.height()) // 2)
    right, bottom = target.x() + target.width(), target.y() + target.height()
    sides = (('right', window.width() - right, card.width()), ('left', target.x(), card.width()),
             ('below', window.height() - bottom, card.height()), ('above', target.y(), card.height()))
    fits = [side for side, room, need in sides if room >= need + 2 * margin]
    side = fits[0] if fits else max(sides, key=lambda s: s[1] - s[2])[0]
    x, y = {'right': (right + margin, target.y()), 'left': (target.x() - margin - card.width(), target.y()),
            'below': (target.x(), bottom + margin), 'above': (target.x(), target.y() - margin - card.height())}[side]
    clamp = lambda value, room: max(min(margin, max(0, room)), min(value, room - margin)) if room >= 2 * margin else max(0, room) // 2
    return QPoint(clamp(x, window.width() - card.width()), clamp(y, window.height() - card.height()))


class TutorialOverlay(QWidget):
    """Dims the window except the current targets and rings them; every click goes through to the window."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.holes = []
        for attribute in (Qt.WA_TransparentForMouseEvents, Qt.WA_NoSystemBackground):self.setAttribute(attribute)
        self.setFocusPolicy(Qt.NoFocus)

    def set_holes(self, rects):
        rects = [QRect(r) for r in rects]
        if rects != self.holes:self.holes = rects;self.update()

    def paintEvent(self, _):
        p = QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        grow, radius = ui_px(4), ui_px(4)
        rings = QPainterPath()
        for hole in self.holes:rings.addRoundedRect(QRectF(hole.adjusted(-grow, -grow, grow, grow)), radius, radius)
        if DIM:
            shade = QPainterPath();shade.addRect(QRectF(self.rect()))
            p.fillPath(shade.subtracted(rings), QColor(0, 0, 0, 60 if is_light_theme() else 90))
        p.setPen(QPen(QColor(theme_color('#74a6e0')), ui_px(2)));p.setBrush(Qt.NoBrush)
        # Half a pixel inwards, so an odd pen width still covers whole pixels.
        for hole in self.holes:
            p.drawRoundedRect(QRectF(hole.adjusted(-grow, -grow, grow, grow)).adjusted(-.5 * (ui_px(2) % 2), -.5 * (ui_px(2) % 2), 0, 0), radius, radius)


class TutorialCard(QFrame):
    """The text beside the target. Its buttons never take the keyboard, so Space still plays and arrows still step."""
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('tutorialCard')
        self.box = QVBoxLayout(self)
        top = QHBoxLayout()
        self.header = QLabel();self.header.setObjectName('tutorialHeader')
        self.cross = self._button('×', self.closed.emit)
        self.cross.setAccessibleName('End tutorial');self.cross.setToolTip('End the tutorial. Help → Tutorial starts it again.')
        top.addWidget(self.header, 1);top.addWidget(self.cross)
        self.title = QLabel();self.title.setObjectName('tutorialTitle');self.title.setWordWrap(True)
        self.text = QLabel();self.text.setWordWrap(True);self.text.setTextFormat(Qt.PlainText)
        self.status = QLabel();self.status.setObjectName('tutorialStatus');self.status.setWordWrap(True)
        self.buttons = QHBoxLayout()
        self.box.addLayout(top)
        for label in (self.title, self.text, self.status):
            label.setTextFormat(Qt.PlainText);self.box.addWidget(label)
        self.box.addLayout(self.buttons)
        self._buttons = []
        set_appearance_style(self, CARD_STYLE)
        self.refresh_appearance()

    def _button(self, label, callback):
        button = QPushButton(label);button.setFocusPolicy(Qt.NoFocus);button.clicked.connect(lambda: callback())
        return button

    def set_content(self, header, title, text, status, buttons):
        self.header.setText(header);self.title.setText(title);self.text.setText(text)
        self.status.setText(status);self.status.setVisible(bool(status))
        for button in self._buttons:
            self.buttons.removeWidget(button);button.setParent(None);button.deleteLater()
        self._buttons = [self._button(label, callback) for label, callback in buttons]
        if not self.buttons.count():self.buttons.addStretch()
        for button in self._buttons:
            if button is self._buttons[-1]:button.setObjectName('primary')  # named before it is shown, so the style applies
            self.buttons.addWidget(button)
        self.fit()

    def fit(self):
        width = ui_px(320)
        self.setFixedSize(width, self.box.totalHeightForWidth(width))

    def refresh_appearance(self):
        margin = ui_px(12)
        self.box.setContentsMargins(margin, ui_px(8), margin, margin);self.box.setSpacing(ui_px(6));self.buttons.setSpacing(ui_px(6))
        self.cross.setFixedSize(ui_px(22), ui_px(22))
        self.fit()
