"""In-window tutorial: highlights a control, explains it and waits for the user's step.

The chapters are data in tutorial_steps. MainWindow never starts the tutorial by itself; app.py and the Help menu do.
Its state is an installation-local convenience in the Resource Safety config/local.json, never project data.
"""
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from .theme import is_light_theme, set_appearance_style, theme_color, ui_px
from .tutorial_steps import CHAPTERS, hidden_ranges_usable

STATE = 'tutorial'
WELCOME = ('This tutorial shows the way from a video to an exported animation, right in this window: it highlights a control, '
           'explains it and waits for you. You can end it at any time and start it again from Help → Tutorial or '
           'File → Settings.')
HINT = 'This person has a hidden range: a gap between two takes. A short chapter shows how it can be filled.'


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
              'QLabel#tutorialTitle{color:#ededed;font-size:14px;font-weight:600}'
              'QPushButton#tutorialClose{padding:0px}')


def place_card(window, card, target, margin, avoid=()):
    """Top-left of the card: beside the target where there is room (right, left, below, above), always inside the window
    and off the rectangles to avoid when a side allows it."""
    if target is None:return QPoint((window.width() - card.width()) // 2, (window.height() - card.height()) // 2)
    right, bottom = target.x() + target.width(), target.y() + target.height()
    sides = (('right', window.width() - right, card.width()), ('left', target.x(), card.width()),
             ('below', window.height() - bottom, card.height()), ('above', target.y(), card.height()))
    spots = {'right': (right + margin, target.y()), 'left': (target.x() - margin - card.width(), target.y()),
             'below': (target.x(), bottom + margin), 'above': (target.x(), target.y() - margin - card.height())}
    clamp = lambda value, room: max(min(margin, max(0, room)), min(value, room - margin)) if room >= 2 * margin else max(0, room) // 2
    def covered(point):
        return sum(r.width() * r.height() for r in (QRect(point, card).intersected(other) for other in avoid))
    # Sides with room first, in their order; then the roomiest. Among them the first that covers nothing, else the least.
    order = sorted(sides, key=lambda s: (s[1] < s[2] + 2 * margin, 0 if s[1] >= s[2] + 2 * margin else s[2] - s[1]))
    points = [QPoint(clamp(spots[side][0], window.width() - card.width()), clamp(spots[side][1], window.height() - card.height()))
              for side, _, _ in order]
    return min(points, key=covered)


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
        self.cross = self._button('×', self.closed.emit);self.cross.setObjectName('tutorialClose')
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
            button.show()  # a layout shows a widget added to a visible parent only later; until then its row has no height
        self.fit()

    def fit(self):
        width = ui_px(320)
        self.setFixedSize(width, self.box.totalHeightForWidth(width))

    def refresh_appearance(self):
        margin = ui_px(12)
        self.box.setContentsMargins(margin, ui_px(8), margin, margin);self.box.setSpacing(ui_px(6));self.buttons.setSpacing(ui_px(6))
        self.cross.setFixedSize(ui_px(22), ui_px(22))
        self.fit()


def modal_open():return QApplication.activeModalWidget() is not None


def popup_open():return QApplication.activePopupWidget() is not None


class Tutorial(QObject):
    """Leads through one chapter: polls the window, moves on when a step is done and draws the highlight and the card."""
    finished = Signal(str, bool)  # chapter key, completed

    def __init__(self, window):
        super().__init__(window)
        self.window, self.chapter, self.index, self.held, self.note, self.events = window, None, 0, False, '', set()
        self.auto = False  # the current step was reached because the one before it was done, not by Next or Back
        self.overlay, self.card = TutorialOverlay(window), TutorialCard(window)
        self.overlay.hide();self.card.hide()
        self.card.closed.connect(self._closed)
        self.timer = QTimer(self);self.timer.setInterval(200);self.timer.timeout.connect(self.tick)
        self._links, self._shown, self._action, self._message = [], None, None, None
        # Offers the Hidden ranges chapter once the panel can really fill; started by first_run only.
        self.hint_timer = QTimer(self);self.hint_timer.setInterval(1000);self.hint_timer.timeout.connect(self.hint_tick)
        self._hint_quiet = False
        window.installEventFilter(self)

    @property
    def active(self):return self.chapter is not None

    @property
    def step(self):return self.chapter.steps[self.index] if self.chapter else None

    def eventFilter(self, watched, event):
        if watched is not self.window:return False
        if event.type() == QEvent.Resize and (self.chapter or self._message):QTimer.singleShot(0, self.tick)
        # Hidden, not Close: the window may refuse to close while a job runs. Minimising hides it too and must not end a chapter.
        elif event.type() == QEvent.Hide and not self.window.isMinimized():
            self.stop();self.hint_timer.stop()
            self._message = None;self.card.hide();self._shown = None
        return False

    def hint_tick(self):
        if self._hint_quiet or self.chapter or self._message or modal_open() or not hidden_ranges_usable(self.window):return
        state = load_state()
        quiet = lambda: setattr(self, '_hint_quiet', True)
        done = state.get('done') if state else None
        if state is None or state.get('hint_off') or (isinstance(done, list) and 'hidden_ranges' in done):return quiet()
        self.message('Hidden ranges', HINT, [("Don't show again", lambda: (quiet(), save_state(hint_off=True))), ('Not now', quiet),
                                             ('Show me', lambda: (quiet(), open_chapter(self.window, 'hidden_ranges')))], closed=quiet)

    def _ask(self, rule, *arguments, default=None):
        """A rule of a step; one that meets a control deleted meanwhile (rows are rebuilt) counts as not met."""
        if rule is None:return default
        try:return rule(self.window, *arguments)
        except RuntimeError:return default

    def _skipped(self, index):return bool(self._ask(self.chapter.steps[index].skip))

    def _beside(self, direction):
        index = self.index + direction
        while 0 <= index < len(self.chapter.steps) and self._skipped(index):index += direction
        return index if 0 <= index < len(self.chapter.steps) else None

    def start(self, chapter):
        self.stop()
        if self._message:self._message()  # a welcome or hint replaced by a chapter was answered: its closing is remembered
        self.chapter, self.index = chapter, -1
        for name, (signal, condition) in (self._ask(chapter.events) or {}).items():
            slot = lambda *_, name=name, condition=condition: condition(self.window) and self.events.add(name)
            signal.connect(slot);self._links.append((signal, slot))
        first = self._beside(1)
        if first is None:return self.stop()
        self.overlay.raise_();self.card.raise_()
        self.timer.start()
        self._enter(first)

    def stop(self, completed=False):
        if not self.chapter:return
        key, self.chapter = self.chapter.key, None
        for signal, slot in self._links:
            try:signal.disconnect(slot)
            except (RuntimeError, TypeError):pass  # the control is gone
        self._links = []
        self.timer.stop();self.overlay.hide();self.card.hide();self._shown = None
        if completed:mark_done(key)
        self.finished.emit(key, completed)

    def next(self, auto=False):
        if not self.chapter:return
        index = self._beside(1)
        if index is None:self.stop(completed=True)
        else:self._enter(index, auto=auto)

    def back(self):
        if not self.chapter:return
        index = self._beside(-1)
        if index is not None:self._enter(index, held=True)

    def _enter(self, index, held=False, note='', auto=False):
        self.index, self.held, self.note, self.auto = index, held, note, auto
        self.events.clear()
        scroll = getattr(self.window, 'settings_scroll', None)
        for target in self._ask(self.step.targets) or []:
            try:
                if scroll is not None and isinstance(target, QWidget) and scroll.isAncestorOf(target):
                    sideways = scroll.horizontalScrollBar().value()  # the settings panel only scrolls up and down
                    scroll.ensureWidgetVisible(target);scroll.horizontalScrollBar().setValue(sideways)
            except RuntimeError:pass
            break
        self.tick()

    def tick(self):
        if not self.chapter:
            if self._message:self._place(None)
            return
        if modal_open():
            self.overlay.hide();self.card.hide();return
        blocked = self._ask(self.chapter.blocked)
        if blocked:
            self.overlay.hide()
            return self._show(self.chapter.title, self.chapter.title, blocked, '', [], None)
        step = self.step
        # Only for a step the guide came to by itself: a failure left from an earlier job must not trap Next or Back.
        retry = self._ask(step.retry, self.events) if self.auto else None
        if retry:
            index = self._beside(-1)
            if index is not None:return self._enter(index, note=retry)
        if self._skipped(self.index):return self.next(auto=self.auto)
        done = bool(self._ask(step.done, self.events, default=False))
        if done and not self.held:return self.next(auto=True)
        self._render(step, done)

    def _rects(self, step):
        rects = []
        for target in self._ask(step.targets) or []:
            try:
                if isinstance(target, QRect):rect = target
                elif target.isVisible():rect = target.visibleRegion().boundingRect().translated(target.mapTo(self.window, QPoint(0, 0)))
                else:continue
            except RuntimeError:continue  # deleted while its step is shown
            if not rect.isEmpty():rects.append(rect)
        return rects

    def _render(self, step, done):
        rects = self._rects(step)
        self.overlay.setGeometry(self.window.rect());self.overlay.set_holes(rects)
        self.overlay.setVisible(not popup_open())
        shown = [i for i in range(len(self.chapter.steps)) if i == self.index or not self._skipped(i)]
        waiting = step.done is not None and not done
        text = step.text
        if callable(text):text = self._ask(text, default='') or ''
        self._action = self._ask(step.action)
        buttons = [('Back', self.back)] if self._beside(-1) is not None else []
        if self._action:buttons.append((self._action[0], lambda: self._action and self._action[1]()))
        buttons.append(('Finish' if self._beside(1) is None else 'Skip step' if waiting and step.waiting else 'Next', self.next))
        self._show(f'{self.chapter.title} · Step {shown.index(self.index) + 1} of {len(shown)}', step.title, text,
                   self.note or (step.waiting if waiting else ''), buttons, rects[0] if rects and step.card != 'center' else None, rects)

    def _show(self, header, title, text, status, buttons, target, avoid=()):
        # The card is rebuilt only when it changes: replacing a button under a pressed mouse would swallow the click.
        shown = (header, title, text, status, tuple(label for label, _ in buttons))
        if shown != self._shown:
            self.card.set_content(header, title, text, status, buttons);self._shown = shown
        self._place(target, avoid)

    def _place(self, target, avoid=()):
        # Beside the first target, the control the step is about, and off the other highlighted ones.
        self.card.move(place_card(self.window.size(), self.card.size(), target, ui_px(12), avoid))
        self.card.show();self.card.raise_()

    def refresh_appearance(self):
        self.card.refresh_appearance();self.overlay.update();self.tick()

    def message(self, title, text, buttons, closed=None):
        """A centred card outside a chapter (welcome, hint). Any button, or the cross, takes it away."""
        self.stop()
        def choice(callback):
            def run():
                self._message = None;self.card.hide();self._shown = None
                callback()
            return run
        self._message = choice(closed or (lambda: None))
        self.overlay.hide()
        self._show('', title, text, '', [(label, choice(callback)) for label, callback in buttons], None)

    def _closed(self):
        if self.chapter:self.stop()
        elif self._message:self._message()


def controller(window):
    if getattr(window, 'tutorial', None) is None:window.tutorial = Tutorial(window)
    return window.tutorial


def open_chapter(window, key):controller(window).start(CHAPTERS[key])


def smoke_run(args):
    """A scripted check of the app. Told by values: --smoke-render-mode and --smoke-playback-seconds have defaults."""
    return bool(args.smoke_screenshot or args.smoke_run_analysis or args.smoke_playback_seconds or args.smoke_report
                or args.smoke_camera_distance is not None or args.smoke_bone_overlay)


def first_run(window, welcome):
    """Called by app.py once the window is shown: the welcome on the very first start, and the Hidden ranges hint."""
    guide, state = controller(window), load_state()
    if welcome and state is not None and not state.get('seen'):
        seen = lambda: save_state(seen=True)
        guide.message('Welcome to RGB MoCap', WELCOME, [('Not now', seen), ('Start', lambda: (seen(), open_chapter(window, 'basics')))],
                      closed=seen)
    guide.hint_timer.start()
