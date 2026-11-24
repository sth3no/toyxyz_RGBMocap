"""Reconstruct in parts: the GUI drives the ordinary trim + Reconstruct 3D path once per part, polling the window like the
tutorial's controller. Each part becomes ordinary takes; joining them is the user's Fill hidden ranges."""
import json
from pathlib import Path

from PySide6.QtCore import QObject, QTimer

from . import parts as planning

MODELS = ('gvhmr', 'exmr', 'flowhmr')  # the models whose Reconstruct 3D makes takes of the people found by Find person
RECORD = 'parts_run.json'  # the exact range, people and settings of the last run in this scene
VIDEO_WAIT = 300  # ticks of 200 ms: a reloaded clip's preview must open before Reconstruct 3D accepts the trim
OPTIONS = ('htd_refine', 'footmr_refine', 'handflow_refine', 'motion_correction', 'exmr_camera', 'exmr_smoothing', 'flow_hamer', 'fast_sam',
           'flow_batch4', 'flow_gpu_crop', 'flow_hand_stride4', 'flow_sampling', 'correction_mode', 'correction_preset', 'auto_floor',
           'floor_height', 'pelvis_offset', 'foot_lock', 'hand_lock')
BUSY = 'Wait for the current job to finish.'
RUNNING = 'Reconstruct in parts is already running.'
MODEL = 'Reconstruct in parts works with GVHMR, ExMR and FlowHMR, which reconstruct the people found by Find person.'
FIND = 'Run Find person on the whole range first. All parts must share one scene and its people.'
CHECK = 'Check at least one person in the People list to reconstruct.'
KEPT = 'Finished parts are kept. Use Resume in Reconstruct in parts to continue.'


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


def refusal(w, *, resuming=False):
    """Why the window cannot reconstruct in parts now, or None."""
    if w.busy or getattr(w, 'contact_saving', False):return BUSY
    if getattr(w, 'parts_runner', None) is not None:return RUNNING
    if w.model_choice.currentData() not in MODELS:return MODEL
    if w.people_state is None or w.people_root is None:return FIND
    if not resuming and not w.checked_people():return CHECK
    return None


def join_issue(w, *, kind=None, static=None):
    """Why Fill hidden ranges cannot join parts made with the window's model and settings, or None: the Hidden ranges
    panel's own reason (FlowHMR, ExMR without Static Camera, HTD Refine)."""
    from .hidden_ranges_state import disabled_reason
    model = w.model_choice.currentData()
    static = w.exmr_camera.currentData() if static is None else static
    return disabled_reason(model, model == 'exmr' and static, kind or w.selected_result(), lambda: True)


def whole_view(w, scene):
    """True when the loaded clip spans the whole scene (the person preview after Find person), so its trim is scene frames."""
    return bool(w.info) and w.info['frames'] == scene['frames'] and abs(w.source_offset-float(scene['start'])) < 1e-6


def saved(root):
    try:
        record = json.loads((Path(root)/RECORD).read_text(encoding='utf-8'))
        if record.get('version') in (1, 2) and all(type(record.get(k)) is int for k in ('first', 'end', 'part', 'overlap')):return record
    except (OSError, ValueError, AttributeError):pass
    return None


def people_identity(state, ids):
    people = {p['id']: p for p in state['people']}
    actors = [[people[key]['preview'], people[key].get('observation_id', key)] for key in ids]
    return [state['input'], state.get('fingerprint'), state['observations'], actors]


def resumable(w, record, scene):
    """A usable checkpoint for the same scene generation, frame grid and model."""
    from .motion_tools import result_kind
    from .people_store import media_receipt
    if not record or record.get('version') != 2 or record.get('complete') is not False:
        return False
    try:
        first, end, overlap = record['first'], record['end'], record['overlap']
        people, options, model = record['people'], record['options'], record['model']
        if not (0 <= first < end <= scene['frames'] and overlap > 0 and record['part'] > 0 and
                isinstance(people, list) and people and all(type(key) is int for key in people) and
                len(set(people)) == len(people) and model == w.model_choice.currentData() and
                scene.get('pipeline', {}).get('model', 'gvhmr') == model and
                isinstance(options, dict) and set(options) == set(OPTIONS) and
                all(_valid_value(getattr(w, name), value) for name, value in options.items()) and
                type(record['delete']) is bool and
                record['grid'] == [scene[k] for k in ('start', 'end', 'fps', 'frames')] and
                record['identity'] == people_identity(w.people_state, people) and
                record['identity'][0] == media_receipt(w.people_root)):
            return False
        gvhmr = model == 'gvhmr'
        if record['result'] != result_kind(gvhmr and options['htd_refine'], gvhmr and options['footmr_refine'],
                                           options['motion_correction'], gvhmr and options['handflow_refine']):
            return False
        for a, b, ids, depth in record['parts']:
            if not (type(a) is type(b) is type(depth) is int and first <= a < b <= end and
                    isinstance(ids, list) and ids and all(type(key) is int for key in ids) and
                    len(set(ids)) == len(ids) and set(ids) <= set(people) and 0 <= depth <= planning.MAX_SPLITS):
                return False
        for ident, a, b in record['superseded']:
            if not (type(ident) is type(a) is type(b) is int and ident in people and first <= a < b <= end):
                return False
        return bool(record['parts'])
    except (KeyError, OSError, TypeError, ValueError, IndexError):
        return False


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
    """Each part runs the window's usual person jobs for its unfinished people."""
    def __init__(self, window, trims, delete, ready=None, *, overlap=0, checkpoint=None):
        super().__init__(window)
        self.w, self.trims, self.delete = window, trims, delete
        self.root = Path(window.people_root);self.scene = _scene(self.root);self.fps = self.scene['fps']
        # The scene frames each part's person job reconstructs: it clips the trim to the scene's end and rounds to frames.
        end = min(float(self.scene['end']), float(self.scene['start'])+self.scene['frames']/self.fps)
        self.plan = [planning.job_range(part, float(self.scene['start']), end, self.fps, self.scene['frames']) for part in trims]
        self.model = window.model_choice.currentData()
        self.superseded = []
        if checkpoint is None:
            self.kind = window.selected_result()
            self.people = [p['id'] for p in window.checked_people()]
            self.options = {name: _value(getattr(window, name)) for name in OPTIONS}
            self.targets = [(self.people, 0) for _ in trims]
            first, end = trims[0][0], trims[-1][1]
            self.record = dict(version=2, first=first, end=end, part=trims[0][1]-first, overlap=overlap,
                               model=self.model, result=self.kind, people=self.people, options=self.options, delete=delete,
                               grid=[self.scene[k] for k in ('start', 'end', 'fps', 'frames')],
                               identity=people_identity(window.people_state, self.people), complete=False)
        else:
            self.record = checkpoint
            self.people = checkpoint['people']
            self.options = checkpoint['options']
            self.kind = checkpoint['result']
            self.targets = [(ids, depth) for _, _, ids, depth in checkpoint['parts']]
            self.superseded = checkpoint['superseded']
        self.ready = ready or ready_result(self.root, self.kind)
        self.join_issue = join_issue(window, kind=self.kind, static=self.options['exmr_camera'])
        # What the window showed, so every way the run ends puts back the user's view and trim.
        self.view = (window.active_person_id, Path(window.project) if window.project is not None else None, (window.slider.clip_in, window.slider.clip_out))
        self.index, self.phase, self.wait, self.stopping, self.seen, self.batch = 0, 'next', 0, False, {}, None
        self.memory_failed = False
        self.job_id = None
        self.timer = QTimer(self);self.timer.setInterval(200);self.timer.timeout.connect(self.tick)

    def save(self):
        from .storage import atomic_json
        self.record['parts'] = [[a, b, ids, depth] for (a, b), (ids, depth) in zip(self.trims, self.targets)]
        self.record['superseded'] = self.superseded
        atomic_json(self.root/RECORD, self.record)

    def start(self):
        from .job_queue_ui import note
        self.job_id = note(self.w, 'add', 'gui', 'Reconstruct in parts', project=self.root)
        if self.job_id:
            self.w._queue_timer.start()
        try:
            if self.superseded:
                self.mark()
            self.save()
            first, end = self.record['first'], self.record['end']
            self.w.log.append(f'Reconstruct in parts: {len(self.plan)} parts, frames {first+1}–{end}.')
        except Exception as error:
            return self.stop(f'Reconstruct in parts did not start: {error}')
        self.timer.start();self.tick()

    def cancel(self):self.stopping = True

    def note_memory_failure(self):
        """Hold the failure on the owning batch even if another job starts next."""
        if self.phase != 'running' or self.batch != self.w.batch_serial:
            return False
        self.memory_failed = True
        return True

    def close(self):
        """Stop scheduling and mark finished parts before the window closes."""
        self.stopping = True;self.timer.stop();self.phase = 'stopped'
        self.tidy(view=False)
        self.finish_record('cancelled', 'The window closed before reconstruction finished.')
        if getattr(self.w, 'parts_runner', None) is self:self.w.parts_runner = None

    def label(self, k=None):return f'Part {(self.index if k is None else k)+1} of {len(self.plan)}'

    def ours(self):
        """The window still shows the scene this run started in, so its people list is this scene's."""
        if self.w.people_state is None or Path(self.w.people_root or '') != self.root:
            return False
        try:
            return self.record['identity'] == people_identity(self.w.people_state, self.people)
        except KeyError:
            return False

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
        from .people_store import save_people, seam_takes
        if not self.ours():return
        state = deepcopy(self.w.people_state);changed = False
        replaced = {tuple(item) for item in self.superseded}
        for ident in self.people:
            person = next((p for p in state['people'] if p['id'] == ident), None)
            if person is None:continue
            choices = person.get('gap_fill') if isinstance(person.get('gap_fill'), dict) else {}
            for take in person.get('takes', []):
                if ((ident, take['first'], take['end']) in replaced and not take.get('parts_superseded')
                        and not self.ready(take['project'])):
                    take['parts_superseded'] = True
                    changed = True
            runs = {run for part, (ids, _) in zip(self.plan, self.targets) if ident in ids for run in self.runs(ident, part)}
            new = {key: value for key, value in planning.marks(seam_takes(person), runs).items() if choices.get(key) != value}
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
            if self.w.model_choice.currentData() != self.model:
                return self.stop(f'The model changed. {KEPT}')
            part = self.plan[self.index]
            ids, depth = self.targets[self.index]
            todo, entries = [], {}
            partial = False
            for ident in ids:
                person = self.person(ident)
                runs = self.runs(ident, part)
                if not runs:
                    self.w.log.append(f"{self.label()}: {person['name']} is not visible, so this part has no take of them.")
                    continue
                pending = self.pending(person, runs)
                if pending:
                    todo.append(ident)
                    partial |= len(pending) != len(runs)
                for run in runs:
                    entries.setdefault((run, depth), []).append(ident)
            if partial:
                # Give the existing person job exact trims so it never re-prepares a ready take.
                self.replace_part(entries)
                continue
            if todo:return self.prepare(todo)
            self.w.log.append(f'{self.label()}: nothing to reconstruct, skipped.')
            self.index += 1
        self.stop(None)

    def pending(self, person, runs):
        takes = {(t['first'], t['end']): t['project'] for t in person.get('takes', [])}
        return [run for run in runs if run not in takes or not self.ready(takes[run])]

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
        w.status.setText(f'{self.label()}: frames {a+1}–{b}.');w.log.append(w.status.text())
        if self.job_id:
            from .job_queue_ui import note
            note(w, 'phase', self.job_id, self.label(), message=w.status.text())
        self.phase, self.wait = 'video', 0
        self.step()

    def launch(self):
        w = self.w
        if not self.ours():
            return self.stop(f'The scene changed, so Reconstruct in parts stopped before {self.label().lower()}. {KEPT}')
        if w.model_choice.currentData() != self.model:
            return self.stop(f'The model changed. {KEPT}')
        if (w.slider.clip_in, w.slider.clip_out) != self.trims[self.index]:return self.stop(f'{self.label()} did not start: the trim could not be set. {KEPT}')
        self.memory_failed = False
        w.run_full(person_ids=self.todo)  # the Reconstruct 3D button's own path
        if not w.busy:return self.stop(f'{self.label()} did not start: {w.status.text()} {KEPT}')
        if not getattr(w, '_people_batch', False):  # run_full started something other than the person jobs of this scene
            return self.stop(f'{self.label()} did not start: another job started instead. {KEPT}')
        self.batch = getattr(w, 'batch_serial', None);self.phase = 'running'

    def finished_part(self, failed, cancelled, status):
        if failed:
            if self.memory_failed and not self.stopping and self.ours() and self.retry():
                return
            if self.memory_failed and not self.stopping:
                status += ' No smaller retry is available. See Console for memory details.'
            return self.stop(f'{self.label()} stopped: {status} {KEPT}', state='failed')
        if cancelled:return self.stop(f'Reconstruct in parts was cancelled at {self.label().lower()}. {KEPT}')
        if not self.ours():return self.stop(f'The scene changed, so Reconstruct in parts stopped after {self.label().lower()}. {KEPT}')
        self.mark()
        if self.delete and self.model == 'exmr' and self.part_ready(self.index):self.delete_part()
        self.index += 1;self.phase = 'next'

    def retry(self):
        """Replace unfinished visible runs with shorter ones, keeping completed takes untouched."""
        from collections import defaultdict
        from .people_store import seam_takes

        ids, depth = self.targets[self.index]
        if depth >= planning.MAX_SPLITS:
            return False
        todo = defaultdict(list)
        superseded = []
        part = self.plan[self.index]
        for ident in ids:
            person = self.person(ident)
            if person is None:
                continue
            takes = seam_takes(person)
            for run in self.runs(ident, part):
                if planning.finished(takes, [run], self.ready):
                    todo[run, depth].append(ident)
                    continue
                before, after = self.neighbour_edges(ident, run)
                children = planning.split(*run, self.record['overlap'], before=before, after=after)
                if children:
                    left, right = (self.runs(ident, child) for child in children)
                    if (len(left) == len(right) == 1 and left[0][0] == run[0] and right[0][1] == run[1]
                            and left[0][1] > right[0][0]):
                        superseded.append((ident, *run))
                        todo[left[0], depth + 1].append(ident)
                        todo[right[0], depth + 1].append(ident)
                        continue
                todo[run, depth].append(ident)
        if not superseded:
            return False
        self.superseded.extend(superseded)
        self.replace_part(todo)
        self.mark()
        self.w.log.append('The last part hit a memory limit. Retrying unfinished ranges in smaller parts.')
        self.phase = 'next'
        return True

    def neighbour_edges(self, ident, run):
        """Keep the split outside the overlaps with the adjacent planned takes."""
        a, b = run
        before, after = a, b
        for index, part in enumerate(self.plan):
            if index == self.index or part[1] <= a or part[0] >= b or ident not in self.targets[index][0]:
                continue
            for lo, hi in self.runs(ident, part):
                if lo < a < hi < b:
                    before = max(before, hi)
                if a < lo < b < hi:
                    after = min(after, lo)
        return before, after

    def replace_part(self, entries):
        ordered = sorted(entries.items())
        index = slice(self.index, self.index + 1)
        self.plan[index] = self.trims[index] = [span for (span, _), _ in ordered]
        self.targets[index] = [(ids, depth) for (_, depth), ids in ordered]
        self.save()

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
        """Mark seams and optionally restore the starting view in this run's scene."""
        errors = []
        for step in (self.mark, *((self.restore_view,) if view else ())):
            if not self.ours():break
            try:step()
            except Exception as error:errors.append(str(error))
        return errors

    def restore_view(self):
        ident, project, trim = self.view
        if project is None:return
        if Path(self.w.project or '') != project:self.w.activate_person(ident, project)
        if Path(self.w.project or '') == project and self.w.info and trim[1] <= self.w.info['frames']:self.w.slider.set_clip(*trim)

    def stop(self, message, *, state=None):
        self.timer.stop();self.phase = 'stopped'
        errors = self.tidy()
        if message is None:
            try:message = self.summary()
            except Exception as error:message = f'Reconstruct in parts stopped: {error} {KEPT}'
        new = [error for error in dict.fromkeys(errors) if error not in message]  # the error that stopped the run is said once
        if new:message += f' The People list could not be updated: {new[0]}'
        self.w.status.setText(message);self.w.log.append(message)
        if state is None:
            state = ('cancelled' if self.stopping else 'done'
                     if self.record.get('complete') and not errors else 'failed')
        self.finish_record(state, message)
        if getattr(self.w, 'parts_runner', None) is self:self.w.parts_runner = None
        self.w.update_readiness()

    def finish_record(self, state, message):
        if self.job_id:
            from .job_queue_ui import note
            result = dict(message=message, checkpoint=str(self.root / RECORD))
            note(self.w, 'finish', self.job_id, state, result=result, error=message if state != 'done' else None)
            self.job_id = None

    def summary(self):
        unready = [self.label(k) for k in range(len(self.plan)) if not self.part_ready(k)]
        if unready:
            return f'Reconstruct in parts finished, but {len(unready)} of {len(self.plan)} parts have no ready result: {", ".join(unready)}. See Console.'
        self.record['complete'] = True
        self.save()
        if self.join_issue:return f'All {len(self.plan)} parts are reconstructed. They cannot be joined into one take: {self.join_issue}'
        return f'All {len(self.plan)} parts are reconstructed. Choose Fill hidden ranges in the People panel to join them.'

    def part_ready(self, k):
        """Every person of the run who is visible in part k has ready takes for it; a part nobody is visible in counts as ready."""
        part = self.plan[k]
        for ident in self.targets[k][0]:
            runs = self.runs(ident, part)
            if runs and not planning.finished((self.person(ident) or {}).get('takes', []), runs, self.ready):return False
        return True


def _value(widget):
    from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox
    if isinstance(widget, QComboBox):return widget.currentData()
    if isinstance(widget, QAbstractButton):return widget.isChecked()
    if isinstance(widget, QAbstractSpinBox):return widget.value()
    return None


def _valid_value(widget, value):
    from PySide6.QtWidgets import QAbstractSpinBox, QComboBox
    if type(value) is not type(_value(widget)):
        return False
    if isinstance(widget, QComboBox):
        return widget.findData(value) >= 0
    if isinstance(widget, QAbstractSpinBox):
        return widget.minimum() <= value <= widget.maximum()
    return True


def _restore(widget, value):
    """Puts back a setting that reloading the scene clip reset, as load_project sets them: without signals."""
    from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox
    if value is None:return
    blocked = widget.blockSignals(True)
    try:
        if isinstance(widget, QComboBox):
            index = widget.findData(value)
            if index < 0:
                raise ValueError(f'The saved option {value!r} is unavailable.')
            widget.setCurrentIndex(index)
        elif isinstance(widget, QAbstractButton):widget.setChecked(value)
        elif isinstance(widget, QAbstractSpinBox):widget.setValue(value)
    finally:widget.blockSignals(blocked)


def fitted(w):
    """(seconds, limit) of the longest part that fits the GPU and the disk now for the window's scene, model and checked
    people; limit is a key of memory_fit.LIMITS. An unreadable GPU or disk does not limit."""
    import shutil
    from . import memory_fit
    scene = _scene(w.people_root)
    free, total = memory_fit.free_vram()
    try:disk = shutil.disk_usage(w.people_root).free
    except OSError:disk = None
    return memory_fit.part_seconds(w.model_choice.currentData(), scene['fps'], scene['width'], scene['height'], max(1, len(w.checked_people())),
                                   free, total, disk)


def start(w, part_seconds, overlap_seconds, delete, first=None, end=None, ready=None):
    """Plans and starts a run; the refusal or the planning error in the status bar instead. The runner while it runs, or None.
    `part_seconds` None plans parts of the fitted length (see fitted)."""
    reason = refusal(w);note = None
    if reason is None:
        try:
            scene = _scene(w.people_root)
            lo, hi = default_range(w, scene) if first is None else (first, end)
            if part_seconds is None:
                from .clip_range import format_time
                from .memory_fit import LIMITS
                part_seconds, limit = fitted(w);note = f'Reconstruct in parts: parts of {format_time(part_seconds)}, {LIMITS[limit]}.'
            overlap = planning.frames(overlap_seconds, scene['fps'])
            plan = planning.plan(lo, hi, planning.frames(part_seconds, scene['fps']), overlap)
            runner = PartsRunner(w, plan, delete, ready, overlap=overlap)
        except (OSError, ValueError, KeyError, TypeError) as error:reason = f'Reconstruct in parts did not start: {error}' if not isinstance(error, ValueError) else str(error)
    if reason:
        w.status.setText(reason);return None
    if note:w.log.append(note)
    return launch_runner(w, runner)


def resume(w):
    """Resume the saved ranges with the original people and reconstruction options."""
    reason = refusal(w, resuming=True)
    if reason:
        w.status.setText(reason)
        return None
    try:
        record = saved(w.people_root)
        scene = _scene(w.people_root)
        if not resumable(w, record, scene):
            raise ValueError('There is no unfinished reconstruction for this scene and model.')
        trims = [(a, b) for a, b, _, _ in record['parts']]
        runner = PartsRunner(w, trims, record['delete'], checkpoint=record)
    except (OSError, ValueError, KeyError, TypeError) as error:
        w.status.setText(f'Reconstruct in parts could not resume: {error}')
        return None
    return launch_runner(w, runner)


def launch_runner(w, runner):
    w.parts_runner = runner
    w.update_readiness()
    runner.start()
    return w.parts_runner
