"""The small dialog of Reconstruct in parts: part length, overlap, whether temporary frames are deleted, and what that means."""
from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout

from . import parts
from .clip_range import format_time
from .memory_fit import LIMITS
from .theme import ui_px

DELETE_TIP = ('ExMR writes every frame of a part as a picture before it reconstructs it. Deleting frames after a part finishes '
              'reduces disk use. Unfinished parts can still have temporary frames.')
NO_FRAMES = 'Only ExMR writes temporary frames.'
RESUME = 2  # a separate dialog result from Accepted (start with the current fields)


class PartsDialog(QDialog):
    def __init__(self, parent, first, end, fps, width, height, people, writes_frames, part=60., overlap=2., join_issue=None, fit=None, resume=None):
        super().__init__(parent)
        self.setWindowTitle('Reconstruct in Parts')
        self.setMinimumWidth(ui_px(420))
        self.range, self.fps, self.size, self.people, self.writes = (first, end), fps, (width, height), people, writes_frames
        layout = QVBoxLayout(self)
        # The Hidden ranges panel's own reason, so the dialog never promises a join that Fill hidden ranges refuses.
        join = (f'Neighbouring parts share the overlap, but they cannot be joined into one take: {join_issue}' if join_issue else
                'Neighbouring parts share the overlap, so Fill hidden ranges can join them into one take.')
        intro = QLabel(f'Reconstruct frames {first+1}–{end} ({format_time((end-first)/fps)}) one part after another. {join}')
        intro.setWordWrap(True);layout.addWidget(intro)
        form = QFormLayout();layout.addLayout(form)
        self.part = self._seconds('Part length, in seconds', 4., 3600., part, 10.);form.addRow('Part length:', self.part)
        # `fit`: (seconds, limit) of parts_runner.fitted, the longest part that fits the GPU and the disk now.
        self.fit_line = QLabel(f'Suggested part length: {format_time(fit[0])}, {LIMITS[fit[1]]}.' if fit else '')
        self.fit_line.setObjectName('muted');self.fit_line.setWordWrap(True);self.fit_line.setVisible(bool(fit));form.addRow(self.fit_line)
        self.overlap = self._seconds('Overlap of neighbouring parts, in seconds', .1, 600., overlap, .5);form.addRow('Overlap:', self.overlap)
        self.delete = QCheckBox('Delete temporary frames after each part');self.delete.setChecked(writes_frames);self.delete.setEnabled(writes_frames)
        self.delete.setToolTip(DELETE_TIP if writes_frames else NO_FRAMES);form.addRow(self.delete)
        self.line = QLabel();self.line.setObjectName('muted');self.line.setWordWrap(True);form.addRow(self.line)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        if resume is not None:
            first, end, count = resume
            note = QLabel(f'You can resume the saved run for frames {first+1} through {end} ({count} parts).')
            note.setWordWrap(True)
            layout.addWidget(note)
            self.buttons.addButton('Resume', QDialogButtonBox.ActionRole).clicked.connect(lambda: self.done(RESUME))
        self.buttons.button(QDialogButtonBox.Ok).setText('Reconstruct')
        self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject);layout.addWidget(self.buttons)
        for control in (self.part, self.overlap):control.valueChanged.connect(self._update)
        self.delete.toggled.connect(self._update);self._update()

    def _seconds(self, name, low, high, value, step):
        box = QDoubleSpinBox();box.setAccessibleName(name);box.setDecimals(1);box.setRange(low, high);box.setSingleStep(step)
        box.setValue(value);box.setSuffix(' s');return box

    def plan(self):
        """The parts of the current fields, or the plain reason they cannot be planned."""
        try:return parts.plan(*self.range, parts.frames(self.part.value(), self.fps), parts.frames(self.overlap.value(), self.fps))
        except ValueError as error:return str(error)

    def _update(self):
        plan = self.plan();ok = not isinstance(plan, str)
        self.line.setText(parts.summary(plan, *self.size, self.people, self.delete.isChecked(), self.writes) if ok else plan)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok and self.people > 0)

    def choice(self):return self.part.value(), self.overlap.value(), self.delete.isChecked() and self.writes


def ask(w):
    """Opens the dialog for the window's scene and starts the run it is accepted with; a refusal goes to the status bar."""
    from . import parts_runner
    reason = parts_runner.refusal(w, resuming=True)
    if reason:
        w.status.setText(reason);return None
    scene = parts_runner._scene(w.people_root);first, end = parts_runner.default_range(w, scene);fps = scene['fps']
    # The same range offers the last run's sizes, so its finished parts are planned again and skipped.
    record = parts_runner.saved(w.people_root)
    same = record is not None and (record['first'], record['end']) == (first, end)
    resume = None
    if parts_runner.resumable(w, record, scene):
        resume = record['first'], record['end'], len(record['parts'])
    if resume is None and not w.checked_people():
        w.status.setText(parts_runner.CHECK)
        return None
    fit = parts_runner.fitted(w)
    sizes = dict(part=record['part']/fps, overlap=record['overlap']/fps) if same and record['overlap'] > 0 else dict(part=fit[0])
    dialog = PartsDialog(w, first, end, fps, scene['width'], scene['height'], len(w.checked_people()), w.model_choice.currentData() == 'exmr', **sizes,
                         join_issue=parts_runner.join_issue(w), fit=fit, resume=resume)
    try:
        action = dialog.exec()
        part, overlap, delete = dialog.choice()
    finally:
        dialog.deleteLater()
    if action == RESUME:
        return parts_runner.resume(w)
    if action == QDialog.Accepted:
        return parts_runner.start(w, part, overlap, delete, first, end)
    return None
