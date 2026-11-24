"""Fill all hidden ranges in the People panel: the note and the button, the question before it starts, and the run that
fills one checked person after another with the job Fill hidden ranges uses. What it fills is decided in fill_all."""
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton

from . import fill_all
from .scene_ghosts import count
from .tooltips import tooltip_text

TITLE = 'Fill all hidden ranges'
TIP = ('Fill the hidden ranges of every checked person in one go. A range without a method gets Smooth join up to 30 frames and '
       'Automatic (Experimental) above that. The ranges and methods are listed before anything starts.')
SCENE_CHANGED = 'The scene changed, so Fill all hidden ranges stopped.'


def build(w, column):
    w.fill_all_runner = None
    w.fill_all_note = QLabel('');w.fill_all_note.setObjectName('muted');w.fill_all_note.setWordWrap(True);w.fill_all_note.hide()
    w.fill_all_button = QPushButton(TITLE);w.fill_all_button.setToolTip(tooltip_text(TIP));w.fill_all_button.hide()
    w.fill_all_button.clicked.connect(lambda: start(w))
    column.addWidget(w.fill_all_note);column.addWidget(w.fill_all_button)


def _checked(w):return {p['id'] for p in w.people_state['people'] if p.get('checked')} if w.people_state else set()


def refresh(w):
    """Shows the row while a checked person has an unfilled hidden range; the 3D view names the button only while it can be clicked."""
    if not hasattr(w, 'fill_all_button'):return
    ranges, people = count(w.scene_hidden, _checked(w))
    w.fill_all_note.setText(fill_all.summary(ranges, people) if ranges else '')
    w.fill_all_note.setVisible(bool(ranges));w.fill_all_button.setVisible(bool(ranges))
    free = bool(ranges) and not w.busy and not w.contact_saving and w.fill_all_runner is None and getattr(w, 'parts_runner', None) is None
    w.fill_all_button.setEnabled(free)
    if w.motion_view.scene_fill_offered != free:
        w.motion_view.scene_fill_offered = free;w.motion_view.update()


def confirm(w, text):
    box = QMessageBox(w)
    try:
        box.setWindowTitle(TITLE);box.setIcon(QMessageBox.Question);box.setTextFormat(Qt.PlainText);box.setText(text)
        fill = box.addButton('Fill', QMessageBox.AcceptRole);box.addButton(QMessageBox.Cancel)
        box.exec()
        return box.clickedButton() is fill
    finally:box.deleteLater()


def start(w, ask=None):
    """The button: plans, asks, records the methods of ranges that had none and starts the run. The run, or None."""
    if (w.busy or w.contact_saving or w.people_state is None or w.fill_all_runner is not None
            or getattr(w, 'parts_runner', None) is not None):return None
    from . import kimodo_runtime  # standard library only
    try:auto_issue = kimodo_runtime.issue()
    except Exception:auto_issue = kimodo_runtime.SETUP
    try:text_issue = kimodo_runtime.text_issue()
    except Exception:text_issue = kimodo_runtime.SETUP
    kind = w.active_result
    found = fill_all.plan(w.people_root, w.people_state['people'], _checked(w), kind, w.model_choice.currentData(),
                          w.info['fps'] if w.info else 30, w.scene_hidden, auto_issue, text_issue)
    if not fill_all.totals(found)[0]:
        w.status.setText(fill_all.nothing_text(found));w.log.append(w.status.text());return None
    if not (ask or confirm)(w, fill_all.confirm_text(found, kind.startswith('corrected_'))):return None
    from .people_store import save_people
    state = fill_all.apply(w.people_state, found)
    try:save_people(w.people_root, state)
    except (ValueError, OSError) as error:
        w.status.setText('Cannot save the hidden range choices: '+str(error));return None
    w.people_state = state
    w.refresh_hidden_ranges(w.current_person())  # the selected person's rows show the methods just recorded
    runner = w.fill_all_runner = Runner(w, [(p.ident, p.name) for p in found.people if p.fills], kind)
    runner.start()
    return runner


class Runner(QObject):
    """One fill batch per person, in order. A person whose fill fails is noted and the next one is filled; Cancel Job or
    another scene ends the run. For a corrected result the joined takes get no correction of their own: one Apply
    Correction for the whole scene follows, because every joined take changes the shared floor."""
    def __init__(self, window, people, kind):
        super().__init__(window)
        self.w, self.people, self.kind = window, list(people), kind
        self.root = Path(window.people_root);self.corrected = kind.startswith('corrected_')
        self.index, self.phase, self.batch, self.correcting = 0, 'next', None, False
        self.filled, self.failed = 0, []
        self.joined = len(window._joined_projects)  # joined takes written before this run
        self.timer = QTimer(self);self.timer.setInterval(200);self.timer.timeout.connect(self.tick)

    def start(self):
        self.w.log.append(f"{TITLE}: {len(self.people)} {'person' if len(self.people) == 1 else 'people'}.")
        self.w.update_readiness();self.timer.start();self.tick()

    def ours(self):
        """The window still shows the scene this run started in."""
        return self.w.people_state is not None and Path(self.w.people_root or '') == self.root

    def tick(self):
        try:self.step()
        except Exception as error:  # a timer slot must not raise: Qt would call it again every 200 ms
            self.stop(f'{TITLE} stopped: {error}')

    def outcome(self):
        """(failed, cancelled, status text) of this run's batch once it has ended, else None; as the window recorded them
        when the batch ended, since a job that starts right after it (a bone preview) sets them again."""
        w = self.w
        if w.busy and getattr(w, 'batch_serial', None) == self.batch:return None
        last = getattr(w, 'last_batch', None)
        if last is not None and last[0] == self.batch:return last[1:]
        return None if w.busy else (w.failed, w.cancelled, w.status.text())

    def step(self):
        w = self.w
        if self.phase == 'running':
            outcome = self.outcome()
            if outcome is None:return
            failed, cancelled, status = outcome
            if self.correcting:
                return self.stop(self.result()+(' Apply Correction was cancelled.' if cancelled else f' Apply Correction did not finish: {status}' if failed else ''))
            if cancelled:return self.stop(fill_all.cancelled_text(self.filled))
            if failed:self.failed.append((self.people[self.index][1], status))
            else:self.filled += 1
            self.index += 1;self.phase = 'next'
            return
        if self.phase != 'next':return
        if not w.isVisible():return self.close()  # the window was closed: nothing more starts
        if w.busy or w.contact_saving:return  # another job (a bone preview, a click): wait for it
        if not self.ours():return self.stop(SCENE_CHANGED)
        if self.index < len(self.people):
            ident, name = self.people[self.index]
            w.fill_hidden_ranges(ident, defer=self.corrected)
            if not w.busy:  # the person is gone, or the queue refused
                self.failed.append((name, 'The fill did not start.'));self.index += 1
                return
            self.batch = getattr(w, 'batch_serial', None);self.phase = 'running'
            return
        if self.corrected and len(w._joined_projects) > self.joined and not self.correcting:
            w.apply_scene_correction(self.kind)
            if w.busy:
                self.correcting = True;self.batch = getattr(w, 'batch_serial', None);self.phase = 'running'
                return
            return self.stop(self.result()+' Apply Correction did not start: '+w.status.text())
        self.stop(self.result())

    def result(self):return fill_all.result_text(self.filled, self.failed)

    def close(self):
        self.timer.stop();self.phase = 'stopped'
        if getattr(self.w, 'fill_all_runner', None) is self:self.w.fill_all_runner = None

    def stop(self, message):
        self.close()
        self.w.status.setText(message);self.w.log.append(message)
        self.w.update_readiness()
