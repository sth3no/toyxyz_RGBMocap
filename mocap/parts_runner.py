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
    """Deletes the temporary ExMR frame JPGs and their receipt.json in `take`/exmr_frames, nothing else; (deleted, kept), where
    kept counts the files that could not be deleted (another program holds them open), which stay as they are.
    exmr_backend.frame_dir writes them again when a stage needs them (it checks the receipt and the JPG count)."""
    take = Path(take).resolve();folder = take/'exmr_frames'
    if folder.is_symlink() or not folder.is_dir() or folder.resolve() != folder:return 0, 0
    removed = kept = 0
    for path in [*folder.glob('*.jpg'), folder/'receipt.json']:
        if path.is_symlink() or not path.is_file() or path.resolve().parent != folder:continue
        try:path.unlink()
        except OSError:kept += 1
        else:removed += 1
    return removed, kept


def _scene(root):return json.loads((Path(root)/'project.json').read_text(encoding='utf-8'))


def refusal(w):
    """Why the window cannot reconstruct in parts now, or None."""
    if w.busy or getattr(w, 'contact_saving', False):return BUSY
    if getattr(w, 'parts_runner', None) is not None:return RUNNING
    if w.model_choice.currentData() not in MODELS:return MODEL
    if w.people_state is None or w.people_root is None:return FIND
    if not w.checked_people():return CHECK
    return None


def join_issue(w):
    """Why Fill hidden ranges cannot join parts made with the window's model and settings, or None: the Hidden ranges
    panel's own reason (FlowHMR, ExMR without Static Camera, HTD Refine)."""
    from .hidden_ranges_state import disabled_reason
    model = w.model_choice.currentData()
    return disabled_reason(model, model == 'exmr' and bool(w.exmr_camera.currentData()), w.selected_result(), lambda: True)


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


def ready_result(root, kind):
    """Whether a take (its project path in people.json, relative to the scene `root`) is done: its base motion is ready, as
    the Hidden ranges panel decides. A corrected result goes out of date whenever another part adds takes to the shared
    scene floor, and a joined take gets its own correction."""
    from .motion_tools import base_result, selected_export
    from .people_store import safe_child
    base = base_result(kind)
    return lambda project: selected_export(safe_child(root, project), base) is not None


class PartsRunner(QObject):
    """One run over `trims` for the people checked when it starts; each part checks only those of them visible in it."""
    def __init__(self, window, trims, delete, ready=None):
        super().__init__(window)
        self.w, self.trims, self.delete = window, trims, delete
        self.root = Path(window.people_root);self.scene = _scene(self.root);self.fps = self.scene['fps']
        # The scene frames each part's person job reconstructs: it clips the trim to the scene's end and rounds to frames.
        end = min(float(self.scene['end']), float(self.scene['start'])+self.scene['frames']/self.fps)
        self.plan = [planning.job_range(part, float(self.scene['start']), end, self.fps, self.scene['frames']) for part in trims]
        self.kind = window.selected_result();self.model = window.model_choice.currentData();self.join_issue = join_issue(window)
        self.ready = ready or ready_result(self.root, self.kind)
        self.people = [p['id'] for p in window.checked_people()]
        self.checked = {p['id']: bool(p.get('checked')) for p in window.people_state['people']}
        self.options = {name: _value(getattr(window, name)) for name in OPTIONS if hasattr(window, name)}
        # What the window showed, so every way the run ends puts back the user's view and trim.
        self.view = (window.active_person_id, Path(window.project) if window.project is not None else None, (window.slider.clip_in, window.slider.clip_out))
        self.index, self.phase, self.wait, self.stopping, self.seen, self.batch = 0, 'next', 0, False, {}, None
        self.timer = QTimer(self);self.timer.setInterval(200);self.timer.timeout.connect(self.tick)

    def start(self):
        from .storage import atomic_json
        try:
            (first, _), (_, end) = self.trims[0], self.trims[-1]
            overlap = self.trims[0][1]-self.trims[1][0] if len(self.trims) > 1 else 0
            atomic_json(self.root/RECORD, dict(version=1, first=first, end=end, part=self.trims[0][1]-self.trims[0][0], overlap=overlap))
            self.w.log.append(f'Reconstruct in parts: {len(self.plan)} parts, frames {first+1}–{end}.')
        except Exception as error:
            return self.stop(f'Reconstruct in parts did not start: {error}')
        self.timer.start();self.tick()

    def cancel(self):self.stopping = True

    def close(self):
        """The window closes: nothing more is started; the user's checks and the marks of finished parts are put back."""
        self.stopping = True;self.timer.stop();self.phase = 'stopped'
        self.tidy(view=False)
        if getattr(self.w, 'parts_runner', None) is self:self.w.parts_runner = None

    def label(self, k=None):return f'Part {(self.index if k is None else k)+1} of {len(self.plan)}'

    def ours(self):
        """The window still shows the scene this run started in, so its people list is this scene's."""
        return self.w.people_state is not None and Path(self.w.people_root or '') == self.root

    def person(self, ident):
        if not self.ours():return None
        return next((p for p in self.w.people_state['people'] if p['id'] == ident), None)

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
        from .people_store import save_people
        if not self.ours():return
        state = deepcopy(self.w.people_state);changed = False
        for ident in self.people:
            person = next((p for p in state['people'] if p['id'] == ident), None)
            if person is None:continue
            choices = person.get('gap_fill') if isinstance(person.get('gap_fill'), dict) else {}
            runs = {run for part in self.plan for run in self.runs(ident, part)}
            new = {key: value for key, value in planning.marks(person.get('takes', []), runs).items() if choices.get(key) != value}
            if new:person['gap_fill'] = {**choices, **new};changed = True
        if changed:
            save_people(self.root, state);self.w.people_state = state;self.w.populate_person_takes()

    def tick(self):
        try:self.step()
        except Exception as error:  # a timer slot must not raise: Qt would call it again every 200 ms
            self.stop(f'{self.label()} stopped: {error} {KEPT}')

    def step(self):
        if self.phase == 'running':
            outcome = self.outcome()
            if outcome is not None:self.finished_part(*outcome)
            return
        if self.phase not in ('video', 'next'):return
        if self.stopping:
            return self.stop(f'Reconstruct in parts was cancelled before {self.label().lower()}. {KEPT}' if self.index < len(self.plan) else
                             f'Reconstruct in parts was cancelled. {KEPT}')
        if self.w.busy or getattr(self.w, 'contact_saving', False):return  # another job (a bone preview, a click): wait for it
        if self.phase == 'video':
            if not self.w.video_readable:
                self.wait += 1
                if self.wait > VIDEO_WAIT:self.stop(f'{self.label()} did not start: the video preview did not open. {KEPT}')
                return
            return self.launch()
        self.next_part()

    def outcome(self):
        """(failed, cancelled, status text) of this run's batch once it has ended, else None; as the window recorded them
        when the batch ended, since a job that starts right after it (a bone preview) sets them again."""
        w = self.w
        if w.busy and getattr(w, 'batch_serial', None) == self.batch:return None
        last = getattr(w, 'last_batch', None)
        if last is not None and last[0] == self.batch:return last[1:]
        return None if w.busy else (w.failed, w.cancelled, w.status.text())

    def next_part(self):
        while self.index < len(self.plan):
            if not self.ours():
                return self.stop(f'The scene changed, so Reconstruct in parts stopped before {self.label().lower()}. {KEPT}')
            part = self.plan[self.index];todo = []
            for ident in self.people:
                person = self.person(ident)
                if person is None:continue  # no longer in the list (Continue as); nothing to reconstruct for them
                runs = self.runs(ident, part)
                if not runs:
                    self.w.log.append(f"{self.label()}: {person['name']} is not visible, so this part has no take of them.")
                elif not planning.finished(person.get('takes', []), runs, self.ready):todo.append(ident)
            if todo:return self.prepare(todo)
            self.w.log.append(f'{self.label()}: nothing to reconstruct, skipped.')
            self.index += 1
        self.stop(None)

    def prepare(self, todo):
        from .people_store import safe_child
        w = self.w;a, b = self.plan[self.index]
        self.todo = todo
        if not whole_view(w, self.scene):  # after a part the window shows the person's last take; the trim needs the whole scene
            w.activate_person(todo[0], safe_child(self.root, self.person(todo[0])['preview']))
        if not whole_view(w, self.scene):return self.stop(f'{self.label()} did not start: the whole scene could not be shown. {KEPT}')
        w.slider.set_clip(*self.trims[self.index])
        for name, value in self.options.items():_restore(getattr(w, name), value)
        w.active_result = w.selected_result()
        self.check(set(todo))
        w.status.setText(f'{self.label()}: frames {a+1}–{b}.');w.log.append(w.status.text())
        self.phase, self.wait = 'video', 0
        self.step()

    def check(self, wanted):
        """Checks exactly `wanted` in the People list through its own check boxes, which save people.json."""
        for row in range(self.w.people_list.count()):
            item = self.w.people_list.item(row);ident = item.data(Qt.UserRole)
            if type(ident) is int and (item.checkState() == Qt.Checked) != (ident in wanted):
                item.setCheckState(Qt.Checked if ident in wanted else Qt.Unchecked)

    def launch(self):
        w = self.w
        if not self.ours():return self.stop(f'The scene changed, so Reconstruct in parts stopped before {self.label().lower()}. {KEPT}')
        if (w.slider.clip_in, w.slider.clip_out) != self.trims[self.index]:return self.stop(f'{self.label()} did not start: the trim could not be set. {KEPT}')
        w.run_full()  # the Reconstruct 3D button's own path
        if not w.busy:return self.stop(f'{self.label()} did not start: {w.status.text()} {KEPT}')
        if not getattr(w, '_people_batch', False):  # run_full started something other than the person jobs of this scene
            return self.stop(f'{self.label()} did not start: another job started instead. {KEPT}')
        self.batch = getattr(w, 'batch_serial', None);self.phase = 'running'

    def finished_part(self, failed, cancelled, status):
        if failed:return self.stop(f'{self.label()} stopped: {status} {KEPT}')
        if cancelled:return self.stop(f'Reconstruct in parts was cancelled at {self.label().lower()}. {KEPT}')
        if not self.ours():return self.stop(f'The scene changed, so Reconstruct in parts stopped after {self.label().lower()}. {KEPT}')
        self.mark()
        if self.delete and self.model == 'exmr':self.delete_part()
        self.index += 1;self.phase = 'next'

    def delete_part(self):
        from .people_store import safe_child
        part, removed, kept = self.plan[self.index], 0, 0
        for ident in self.todo:
            person = self.person(ident)
            runs = set(self.runs(ident, part))
            for take in (person or {}).get('takes', []):
                if (take['first'], take['end']) in runs:
                    done, held = delete_frames(safe_child(self.root, take['project']));removed += done;kept += held
        self.w.log.append(f'{self.label()}: deleted {removed} temporary frame files.' +
                          (f' {kept} could not be deleted (another program may hold them) and stay on disk.' if kept else ''))

    def tidy(self, view=True):
        """Puts back the user's checks, marks the seams of finished parts (also those of an earlier run) and, with `view`,
        shows what the window showed before; only in this run's scene. The errors of the steps that failed."""
        errors = []
        for step in (self.restore_checks, self.mark, *((self.restore_view,) if view else ())):
            if not self.ours():break
            try:step()
            except Exception as error:errors.append(str(error))
        return errors

    def restore_checks(self):self.check({ident for ident, on in self.checked.items() if on})

    def restore_view(self):
        ident, project, trim = self.view
        if project is None:return
        if Path(self.w.project or '') != project:self.w.activate_person(ident, project)
        if Path(self.w.project or '') == project and self.w.info and trim[1] <= self.w.info['frames']:self.w.slider.set_clip(*trim)

    def stop(self, message):
        self.timer.stop();self.phase = 'stopped'
        errors = self.tidy()
        if message is None:
            try:message = self.summary()
            except Exception as error:message = f'Reconstruct in parts stopped: {error} {KEPT}'
        new = [error for error in dict.fromkeys(errors) if error not in message]  # the error that stopped the run is said once
        if new:message += f' The People list could not be updated: {new[0]}'
        self.w.status.setText(message);self.w.log.append(message)
        if getattr(self.w, 'parts_runner', None) is self:self.w.parts_runner = None
        self.w.update_readiness()

    def summary(self):
        unready = [self.label(k) for k in range(len(self.plan)) if not self.part_ready(k)]
        if unready:
            return f'Reconstruct in parts finished, but {len(unready)} of {len(self.plan)} parts have no ready result: {", ".join(unready)}. See Console.'
        if self.join_issue:return f'All {len(self.plan)} parts are reconstructed. They cannot be joined into one take: {self.join_issue}'
        return f'All {len(self.plan)} parts are reconstructed. Choose Fill hidden ranges in the People panel to join them.'

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
    """Plans and starts a run; the refusal or the planning error in the status bar instead. The runner while it runs, or None."""
    reason = refusal(w)
    if reason is None:
        try:
            scene = _scene(w.people_root)
            lo, hi = default_range(w, scene) if first is None else (first, end)
            plan = planning.plan(lo, hi, planning.frames(part_seconds, scene['fps']), planning.frames(overlap_seconds, scene['fps']))
            runner = PartsRunner(w, plan, delete, ready)
        except (OSError, ValueError, KeyError, TypeError) as error:reason = f'Reconstruct in parts did not start: {error}' if not isinstance(error, ValueError) else str(error)
    if reason:
        w.status.setText(reason);return None
    w.parts_runner = runner;w.update_readiness();runner.start();return w.parts_runner
