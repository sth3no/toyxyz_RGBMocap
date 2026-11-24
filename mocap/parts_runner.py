"""Reconstruct in parts: the GUI drives the ordinary trim + Reconstruct 3D path once per part, polling the window like the
tutorial's controller. Each part becomes ordinary takes; joining them is the user's Fill hidden ranges."""
import json
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt

from . import parts as planning

MODELS = ('gvhmr', 'exmr', 'flowhmr')  # the models whose Reconstruct 3D makes takes of the people found by Find person
RECORD = 'parts_run.json'  # in the scene folder: the range and part sizes of the last run, so a later run plans the same parts
VIDEO_WAIT = 300  # ticks of 200 ms: a reloaded clip's preview must open before Reconstruct 3D accepts the trim
OPTIONS = ('htd_refine', 'footmr_refine', 'handflow_refine', 'motion_correction', 'exmr_camera', 'exmr_smoothing', 'flow_hamer', 'fast_sam',
           'flow_batch4', 'flow_gpu_crop', 'flow_hand_stride4', 'flow_sampling', 'correction_mode', 'correction_preset', 'auto_floor',
           'floor_height', 'pelvis_offset', 'foot_lock', 'hand_lock')
BUSY = 'Wait for the current job to finish.'
RUNNING = 'Reconstruct in parts is already running.'
MODEL = 'Reconstruct in parts works with GVHMR, ExMR and FlowHMR, which reconstruct the people found by Find person.'
FIND = 'Run Find person on the whole range first. All parts must share one scene and its people.'
CHECK = 'Check at least one person in the People list to reconstruct.'
KEPT = 'Finished parts are kept; Reconstruct in parts skips them next time.'


def delete_frames(take):
    """Deletes the temporary ExMR frame JPGs and their receipt.json in `take`/exmr_frames, nothing else; the count.
    exmr_backend.frame_dir writes them again when a stage needs them (it checks the receipt and the JPG count)."""
    take = Path(take).resolve();folder = take/'exmr_frames'
    if folder.is_symlink() or not folder.is_dir() or folder.resolve() != folder:return 0
    removed = 0
    for path in [*folder.glob('*.jpg'), folder/'receipt.json']:
        if path.is_symlink() or not path.is_file() or path.resolve().parent != folder:continue
        path.unlink();removed += 1
    return removed


def _scene(root):return json.loads((Path(root)/'project.json').read_text(encoding='utf-8'))


def refusal(w):
    """Why the window cannot reconstruct in parts now, or None."""
    if w.busy or getattr(w, 'contact_saving', False):return BUSY
    if getattr(w, 'parts_runner', None) is not None:return RUNNING
    if w.model_choice.currentData() not in MODELS:return MODEL
    if w.people_state is None or w.people_root is None:return FIND
    if not w.checked_people():return CHECK
    return None


def whole_view(w, scene):
    """True when the loaded clip spans the whole scene (the person preview after Find person), so its trim is scene frames."""
    return bool(w.info) and w.info['frames'] == scene['frames'] and abs(w.source_offset-float(scene['start'])) < 1e-6


def saved(root):
    try:
        record = json.loads((Path(root)/RECORD).read_text(encoding='utf-8'))
        if record.get('version') == 1 and all(type(record.get(k)) is int for k in ('first', 'end', 'part', 'overlap')):return record
    except (OSError, ValueError, AttributeError):pass
    return None


def default_range(w, scene):
    """(first, end) to split: the trim when the whole scene is shown, else the range of the last run, else the whole scene."""
    if whole_view(w, scene):return w.slider.clip_in, w.slider.clip_out
    record = saved(w.people_root)
    if record and 0 <= record['first'] < record['end'] <= scene['frames']:return record['first'], record['end']
    return 0, scene['frames']


def ready_result(kind):
    from .motion_tools import selected_export
    return lambda project: selected_export(project, kind) is not None


class PartsRunner(QObject):
    """One run over `plan` for the people checked when it starts; each part checks only those of them visible in it."""
    def __init__(self, window, plan, delete, ready=None):
        super().__init__(window)
        self.w, self.plan, self.delete = window, plan, delete
        self.root = Path(window.people_root);self.scene = _scene(self.root);self.fps = self.scene['fps']
        self.kind = window.selected_result();self.model = window.model_choice.currentData()
        self.ready = ready or ready_result(self.kind)
        self.people = [p['id'] for p in window.checked_people()]
        self.checked = {p['id']: bool(p.get('checked')) for p in window.people_state['people']}
        self.options = {name: _value(getattr(window, name)) for name in OPTIONS if hasattr(window, name)}
        self.index, self.phase, self.wait, self.stopping, self.seen = 0, 'next', 0, False, {}
        self.timer = QTimer(self);self.timer.setInterval(200);self.timer.timeout.connect(self.tick)

    def start(self):
        from .storage import atomic_json
        (first, _), (_, end) = self.plan[0], self.plan[-1]
        overlap = self.plan[0][1]-self.plan[1][0] if len(self.plan) > 1 else 0
        atomic_json(self.root/RECORD, dict(version=1, first=first, end=end, part=self.plan[0][1]-self.plan[0][0], overlap=overlap))
        self.w.cancel_button.clicked.connect(self.cancel)
        self.w.log.append(f'Reconstruct in parts: {len(self.plan)} parts, frames {first+1}–{end}.')
        self.timer.start();self.tick()

    def cancel(self):self.stopping = True

    def label(self, k=None):return f'Part {(self.index if k is None else k)+1} of {len(self.plan)}'

    def person(self, ident):return next((p for p in self.w.people_state['people'] if p['id'] == ident), None)

    def runs(self, ident, part):
        import numpy as np
        from .people_store import safe_child
        if ident not in self.seen:
            person = self.person(ident)
            if person is None:return []
            with np.load(safe_child(self.root, person['preview'])/'tracking.npz', allow_pickle=False) as data:self.seen[ident] = data['observed'].copy()
        return planning.visible(self.seen[ident], self.fps, part)

    def mark(self):
        """Marks the seams between this run's parts in people.json, so Hidden ranges lists them and Fill hidden ranges joins them."""
        from copy import deepcopy
        from .hidden_ranges import PARTS_SEAM
        from .people_store import save_people
        state = deepcopy(self.w.people_state);changed = False
        for ident in self.people:
            person = next((p for p in state['people'] if p['id'] == ident), None)
            if person is None:continue
            choices = person.get('gap_fill') if isinstance(person.get('gap_fill'), dict) else {}
            runs = {run for part in self.plan for run in self.runs(ident, part)}
            new = [key for key in planning.marks(person.get('takes', []), runs) if choices.get(key) != PARTS_SEAM]
            if new:person['gap_fill'] = {**choices, **{key: dict(PARTS_SEAM) for key in new}};changed = True
        if changed:
            save_people(self.root, state);self.w.people_state = state;self.w.populate_person_takes()

    def tick(self):
        if self.phase == 'running':
            if self.w.busy:return
            return self.finished_part()
        if self.phase == 'video':
            if self.stopping:return self.stop('Reconstruct in parts was cancelled before '+self.label().lower()+'. '+KEPT)
            if not self.w.video_readable:
                self.wait += 1
                if self.wait > VIDEO_WAIT:self.stop(f'{self.label()} did not start: the video preview did not open. {KEPT}')
                return
            return self.launch()
        if self.phase == 'next':self.next_part()

    def next_part(self):
        if self.stopping:return self.stop('Reconstruct in parts was cancelled. '+KEPT)
        while self.index < len(self.plan):
            if Path(self.w.people_root or '') != self.root or self.w.people_state is None:
                return self.stop(f'The scene changed, so Reconstruct in parts stopped before {self.label().lower()}. {KEPT}')
            part = self.plan[self.index];todo = []
            for ident in self.people:
                person = self.person(ident)
                if person is None:continue  # no longer in the list (Continue as); nothing to reconstruct for them
                runs = self.runs(ident, part)
                if not runs:
                    self.w.log.append(f"{self.label()}: {person['name']} is not visible, so this part has no take of them.")
                elif not planning.finished(person.get('takes', []), runs, self.ready):todo.append(ident)
            if todo:return self.prepare(part, todo)
            self.w.log.append(f'{self.label()}: nothing to reconstruct, skipped.')
            self.index += 1
        self.stop(None)

    def prepare(self, part, todo):
        from .people_store import safe_child
        w = self.w;a, b = part
        self.todo = todo
        if not whole_view(w, self.scene):  # after a part the window shows the person's last take; the trim needs the whole scene
            w.activate_person(todo[0], safe_child(self.root, self.person(todo[0])['preview']))
        if not whole_view(w, self.scene):return self.stop(f'{self.label()} did not start: the whole scene could not be shown. {KEPT}')
        w.slider.set_clip(a, b)
        for name, value in self.options.items():_restore(getattr(w, name), value)
        w.active_result = w.selected_result()
        self.check(set(todo))
        w.status.setText(f'{self.label()}: frames {a+1}–{b}.');w.log.append(w.status.text())
        self.phase, self.wait = 'video', 0
        self.tick()

    def check(self, wanted):
        """Checks exactly `wanted` in the People list through its own check boxes, which save people.json."""
        for row in range(self.w.people_list.count()):
            item = self.w.people_list.item(row);ident = item.data(Qt.UserRole)
            if type(ident) is int and (item.checkState() == Qt.Checked) != (ident in wanted):
                item.setCheckState(Qt.Checked if ident in wanted else Qt.Unchecked)

    def launch(self):
        w = self.w
        if (w.slider.clip_in, w.slider.clip_out) != self.plan[self.index]:return self.stop(f'{self.label()} did not start: the trim could not be set. {KEPT}')
        w.run_full()  # the Reconstruct 3D button's own path
        if not w.busy:return self.stop(f'{self.label()} did not start: {w.status.text()} {KEPT}')
        self.phase = 'running'

    def finished_part(self):
        w = self.w
        if w.cancelled or self.stopping:
            return self.stop(f'Reconstruct in parts was cancelled at {self.label().lower()}. {KEPT}' if not w.failed else
                             f'{self.label()} stopped: {w.status.text()} {KEPT}')
        if w.failed:return self.stop(f'{self.label()} stopped: {w.status.text()} {KEPT}')
        self.mark()
        if self.delete and self.model == 'exmr':self.delete_part()
        self.index += 1;self.phase = 'next'

    def delete_part(self):
        from .people_store import safe_child
        part, removed = self.plan[self.index], 0
        for ident in self.todo:
            person = self.person(ident)
            runs = set(self.runs(ident, part))
            for take in (person or {}).get('takes', []):
                if (take['first'], take['end']) in runs:removed += delete_frames(safe_child(self.root, take['project']))
        self.w.log.append(f'{self.label()}: deleted {removed} temporary frame files.')

    def stop(self, message):
        self.timer.stop();self.phase = 'stopped'
        try:self.w.cancel_button.clicked.disconnect(self.cancel)
        except (RuntimeError, TypeError):pass
        if self.w.people_state is not None and Path(self.w.people_root or '') == self.root:
            self.check({ident for ident, on in self.checked.items() if on})
            self.mark()  # also the seams of parts finished by an earlier run
        if message is None:
            unready = [self.label(k) for k in range(len(self.plan)) if not self.part_ready(k)]
            message = (f'All {len(self.plan)} parts are reconstructed. Choose Fill hidden ranges in the People panel to join them.' if not unready else
                       f'Reconstruct in parts finished, but {len(unready)} of {len(self.plan)} parts have no ready result: {", ".join(unready)}. See Console.')
        self.w.status.setText(message);self.w.log.append(message)
        if getattr(self.w, 'parts_runner', None) is self:self.w.parts_runner = None
        self.w.update_readiness()

    def part_ready(self, k):
        """Every person of the run who is visible in part k has ready takes for it; a part nobody is visible in counts as ready."""
        part = self.plan[k]
        for ident in self.people:
            runs = self.runs(ident, part)
            if runs and not planning.finished((self.person(ident) or {}).get('takes', []), runs, self.ready):return False
        return True


def _value(widget):
    from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox
    if isinstance(widget, QComboBox):return widget.currentIndex()
    if isinstance(widget, QAbstractButton):return widget.isChecked()
    if isinstance(widget, QAbstractSpinBox):return widget.value()
    return None


def _restore(widget, value):
    """Puts back a setting that reloading the scene clip reset, as load_project sets them: without signals."""
    from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox
    if value is None:return
    blocked = widget.blockSignals(True)
    try:
        if isinstance(widget, QComboBox):widget.setCurrentIndex(value)
        elif isinstance(widget, QAbstractButton):widget.setChecked(value)
        elif isinstance(widget, QAbstractSpinBox):widget.setValue(value)
    finally:widget.blockSignals(blocked)


def start(w, part_seconds, overlap_seconds, delete, first=None, end=None, ready=None):
    """Plans and starts a run; the refusal or the planning error in the status bar instead. The runner, or None."""
    reason = refusal(w)
    if reason is None:
        scene = _scene(w.people_root)
        lo, hi = default_range(w, scene) if first is None else (first, end)
        try:plan = planning.plan(lo, hi, planning.frames(part_seconds, scene['fps']), planning.frames(overlap_seconds, scene['fps']))
        except ValueError as error:reason = str(error)
    if reason:
        w.status.setText(reason);return None
    w.parts_runner = PartsRunner(w, plan, delete, ready);w.parts_runner.start();return w.parts_runner
