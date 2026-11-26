"""The sequence in the main window: one stacked timeline with today's slider at level 3 (page "clip") and the sequence view
at levels 1 and 2 (page "sequence"), loading the selected block into the window, the playhead in sequence time and the way
between the levels. Play and the transport stay the window's own and act on the loaded block.

Left of the timeline sit the Project panel (sequence_library: the videos of the sequence's library, dragged from there to
the timeline) and the strip of timeline tools (sequence_tools)."""
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, QTimer, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QSizePolicy, QSplitter,
                               QStackedWidget, QVBoxLayout, QWidget)

from . import paths, people_store, sequence_join, sequence_links, sequence_seam, sequence_state
from .clip_range import format_time
from .parts_runner import MODELS
from .sequence import DEFAULT_METHOD, METHODS, Sequence
from .sequence_library import Library
from .sequence_tools import ToolStrip, picture
from .sequence_view import METHOD_LABELS, SequenceView, dragged_videos
from .take_import_check import gap_frames
from .theme import theme_color, ui_px
from .tooltips import tooltip_text

OTHER_MODEL = 'The sequence works with GVHMR, ExMR and FlowHMR.'
PROJECT_MODEL = 'Only GVHMR, ExMR and FlowHMR projects can be added to a sequence.'
BUSY = 'A job is running. This block can be opened when it is done.'
PLACE_BUSY = 'A job is running. Videos can be put on the timeline when it is done.'
LAST = 'A sequence needs at least one block.'
PANEL = 'The Timeline panel changed; the sequence is not available.'
CHANGE = 'This block has results. They belong to the old range and stay on disk; the changed block will show as not tracked. Change it?'
MENU = {'project': 'Add a saved GVHMR, ExMR or FlowHMR project at the end of the sequence, as a block with its results.',
        'open': 'Open a saved sequence. A sequence of two or more blocks saves itself under projects/sequences on every change.',
        'new': 'Start a new sequence. The loaded video or project stays open, as its one block when a video is loaded.'}
TIPS = {'back': 'Show the sequence timeline around this block.', 'add': 'Add a video at the end of the sequence.',
        'cut': 'Split the block under the playhead at the playhead.', 'delete': 'Remove the selected block from the sequence. Its files stay on disk.',
        'frame': 'Fit the whole sequence in the timeline.',
        'join': 'Join every linked row that is not joined yet into one animation in the scene of its first block. The later blocks are not changed.'}
STALE = 'The people of {} changed, so their links were removed.'
AUTOMATIC = '{} in {} now continues {} from {}. Drag the bar to change it.'
LINKED = '{} now continues {}.'
UNLINKED = '{} no longer continues {}.'
TRANSITION = 'Transition set to {:g} s {}.'
EXPERIMENTAL = 'Experimental: the generated motion is plausible, not what the person did.'
VIDEOS = 'Videos (*.mp4 *.mov *.avi *.mkv *.webm)'
ADDED = '{} added to the Project panel. Drag one to the timeline.'
REMOVED = '{} removed from the Project panel. {}'
KEPT = {(True, True): 'Its blocks stay on the timeline; the file stays on disk.', (True, False): 'The file stays on disk.',
        (False, True): 'Their blocks stay on the timeline; the files stay on disk.', (False, False): 'The files stay on disk.'}


def count_of(count, noun='video'):
    return f'{count} {noun}{"" if count == 1 else "s"}'


def video_length(path):
    """Seconds of a video from the metadata the preview decoder reads, without decoding a frame; ValueError with a reason."""
    import cv2
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500])
    try:
        if not capture.isOpened():raise ValueError('Cannot open the video.')
        width, height = int(capture.get(3)), int(capture.get(4))
        frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or frames <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot determine the video length and frame rate.')
        return frames / fps
    finally:
        capture.release()


def sequences_folder():
    return paths.HOME/'projects'/'sequences'


def free_name():
    """projects/sequences/Sequence <date time>.json, with a number when that name is taken."""
    folder = sequences_folder();folder.mkdir(parents=True, exist_ok=True)
    stem = f'Sequence {datetime.now():%Y-%m-%d %H%M}';path, n = folder/f'{stem}.json', 2
    while path.exists():path, n = folder/f'{stem} ({n}).json', n + 1
    return path


def same_file(a, b):
    try:return Path(a).resolve() == Path(b).resolve()
    except (OSError, TypeError):return False


def transition_issue(method, seconds):
    """Why `method` cannot fill a transition of `seconds`, in the Hidden ranges panel's words, or None."""
    if method not in METHODS:return f'Unknown method: {method}'
    return sequence_links.gap_issue(method, gap_frames(seconds))


class Controller(QObject):
    def __init__(self, w, file_menu):
        super().__init__(w)
        self.w, self.sequence, self.current = w, Sequence(), None
        self._pending = None  # (block id, source seconds or None): the block's video is opening
        self._video_info = None  # the window's info of a whole video it opened (not of a project), once this controller saw it
        self._trimming = False   # the controller sets the trim itself
        self._clicked = None     # a click's block, so its following scrub does not try the load again
        self._opening = None     # the video File > Open Video is opening, so its metadata does not start a sequence again
        self._advance_to = None
        self._yielded = False    # the window shows a project or workspace of its own that is not the loaded block
        self.joiner = None       # the Join run (sequence_join.Runner) while it runs
        self._said = None        # the latest message `_say` wrote
        self._settle = QTimer(self);self._settle.setSingleShot(True);self._settle.setInterval(250)
        self._settle.timeout.connect(self._changed)   # a dragged trim is shown and saved once it rests
        self._build()
        for text, slot, tip in (('Add Video to Sequence...', self.choose_video, 'Add a video at the end of the sequence, as a new block covering the whole video.'),
                                ('Add Project to Sequence...', self.choose_project, MENU['project']),
                                ('Open Sequence...', self.choose_sequence, MENU['open']), ('New Sequence', self.new, MENU['new'])):
            action = file_menu.addAction(text, slot);action.setToolTip(tooltip_text(tip))
            w.open_actions.append(action)   # the window disables these while a job runs
        # After the window's own slots, so the clip is set up when these run.
        w.video_view.metadata_ready.connect(self._video_ready)
        w.video_view.decode_error.connect(self._video_failed)
        w.slider.valueChanged.connect(self._playhead)
        w.slider.clip_changed.connect(self._trim_changed)
        w.slider.trim_enabled_changed.connect(lambda _:self.refresh())   # a job starts or ends
        w.model_choice.currentIndexChanged.connect(self._model_changed)
        w.parts_button.installEventFilter(self)   # enabled again when Reconstruct in parts ends
        w.slider.installEventFilter(self)
        self.view.installEventFilter(self)
        w.installEventFilter(self)
        self.refresh()

    def busy(self):
        """A job runs, or Reconstruct in parts or Join, also between their jobs."""
        return bool(self.w.busy) or getattr(self.w, 'parts_runner', None) is not None or (self.joiner is not None and self.joiner.running)

    def _reset(self, sequence):
        self._settle.stop()
        self.sequence, self.current, self._pending, self._yielded = sequence, None, None, False
        self._advance_to = None
        sequence.model = self.w.model_choice.currentData()

    def _say(self, text):
        """A message of the sequence about its links: a Console line, and the status text, written again once the event loop
        is idle, since the window's own completion or load text often follows in the same turn; never over a newer message
        of `_say`, nor while a job runs, whose progress owns the status bar."""
        self.w.log.append(text);self.w.status.setText(text);self._said = said = object()
        QTimer.singleShot(0, self, lambda:self._said is said and not self.busy() and self.w.status.setText(text))

    def _guarded(self, change, *args):
        """Run a change the window itself calls for; an error in it never decides the window's job or load."""
        try:return change(*args)
        except Exception as error:
            import traceback
            self.w.log.append(traceback.format_exc())
            self.w.status.setText(f'The sequence could not be updated: {error}. See the Console.')

    def _build(self):
        w = self.w;layout = w.timeline_panel.layout();actions = action_bar(w)
        self.clip_page, self.sequence_page = QWidget(), QWidget()
        clip = QVBoxLayout(self.clip_page);clip.setContentsMargins(0, 0, 0, 0);clip.setSpacing(0)
        for widget in (actions, w.slider):layout.removeWidget(widget)
        clip.addWidget(actions);clip.addWidget(w.slider, 1)
        self.back_button = QPushButton('← Sequence');self.back_button.clicked.connect(self.back)
        actions.layout().insertWidget(0, self.back_button)
        bar = self.bar = QWidget();bar.setObjectName('editorHeader');bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row = QHBoxLayout(bar)
        self.add_button, self.cut_button, self.delete_button, self.join_button, self.frame_button = (
            QPushButton(text) for text in ('Add Video...', 'Cut', 'Delete', 'Join', 'Frame All'))
        self.add_button.clicked.connect(self.choose_video)
        self.cut_button.clicked.connect(self.cut)
        self.delete_button.clicked.connect(lambda:self.delete())
        self.join_button.clicked.connect(lambda:self.join())
        self.frame_button.clicked.connect(lambda:self.view.frame_all())
        for button in (self.add_button, self.cut_button, self.delete_button, self.join_button, self.frame_button):row.addWidget(button)
        self.review_button = QPushButton('Review Range...')
        self.review_button.clicked.connect(self.edit_review)
        self.auto_advance = QCheckBox('Auto-advance')
        row.addWidget(self.review_button)
        row.addWidget(self.auto_advance)
        self.summary = QLabel('');self.summary.setMinimumWidth(0);self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.summary, 1)
        self.view = SequenceView()
        page = QVBoxLayout(self.sequence_page);page.setContentsMargins(0, 0, 0, 0);page.setSpacing(0)
        page.addWidget(bar);page.addWidget(self.view, 1)
        self.stack = QStackedWidget();self.stack.addWidget(self.clip_page);self.stack.addWidget(self.sequence_page)
        # The Project panel | the tools | the timeline, as an editing program lays them out.
        self.library, self.tools = Library(), ToolStrip()
        timeline = QWidget();beside = QHBoxLayout(timeline);beside.setContentsMargins(0, 0, 0, 0);beside.setSpacing(0)
        beside.addWidget(self.tools);beside.addWidget(self.stack, 1)
        self.body = QSplitter(Qt.Horizontal);self.body.setChildrenCollapsible(False)
        self.body.addWidget(self.library);self.body.addWidget(timeline)
        self.body.setStretchFactor(0, 0);self.body.setStretchFactor(1, 1)
        layout.insertWidget(1, self.body, 1)
        self._listed = None   # what the Project panel shows: the library's videos and which of them have a block
        self.stack.currentChanged.connect(lambda _:self._tools())
        self.tools.picked.connect(self.view.set_tool)
        self.view.tool_changed.connect(self.tools.set_tool)
        self.view.cut_requested.connect(self.cut_at)
        self.view.dropped.connect(lambda videos, index:self.place(videos, index))
        self.library.imported.connect(self.import_videos)
        self.library.import_requested.connect(self.choose_media)
        self.library.placed.connect(lambda video:self.place([video]))
        self.library.removed.connect(self.remove_media)
        w.slider.setAcceptDrops(True)   # a video dragged onto an opened block; see eventFilter
        self.view.selected.connect(self._select)
        self.view.opened.connect(lambda block_id, person:self._opened(block_id, person))
        self.view.scrubbed.connect(self._scrubbed)
        self.view.moved.connect(self.move)
        self.view.resized.connect(self.resize)
        self.view.removed.connect(self.delete)
        self.view.linked.connect(lambda source, target:self.link(source, target))
        self.view.unlinked.connect(lambda target:self.unlink(target))
        self.view.transition_clicked.connect(lambda target:self.edit_transition(target))
        self.view.menu_requested.connect(lambda block_id, person, point:self.person_menu(block_id, person, point))
        # The card over the seam between two blocks the pointer rests on.
        card = self.seam_card = sequence_seam.SeamCard(w)
        self._seam_wanted = None   # the seam the pointer is on, while the card waits to be shown
        self._seam_show, self._seam_hide = QTimer(self), QTimer(self)
        for timer, wait, slot in ((self._seam_show, sequence_seam.HOVER_MS, lambda:self.show_seam()),
                                  (self._seam_hide, sequence_seam.LEAVE_MS, lambda:self._seam_left())):
            timer.setSingleShot(True);timer.setInterval(wait);timer.timeout.connect(slot)
        self.view.seam_changed.connect(self._seam_changed)
        card.entered.connect(self._seam_hide.stop);card.left.connect(self._seam_hide.start)
        card.join_clicked.connect(lambda:self._seam_action(lambda:self.join_seam(card.later)))
        card.open_clicked.connect(lambda:self._seam_action(lambda:self.open_joined(card.offer.joined)))
        card.transition_clicked.connect(lambda:self._seam_action(lambda:self.edit_transition(card.offer.targets[0])))
        self._appearance()

    def _appearance(self):
        self.bar.layout().setContentsMargins(ui_px(9), ui_px(3), ui_px(6), ui_px(3))
        self.view.refresh_appearance()
        self.library.refresh_appearance();self.tools.refresh_appearance();self.seam_card.refresh_appearance()
        side = ui_px(14);scissors = QIcon()   # the Razor's scissors on the button that cuts at the playhead
        scissors.addPixmap(picture('razor', side * 2, theme_color('#c8c8c8', '#3a3a3a')), QIcon.Normal)
        scissors.addPixmap(picture('razor', side * 2, theme_color('#6e6e6e', '#a8a8a8')), QIcon.Disabled)
        self.cut_button.setIcon(scissors);self.cut_button.setIconSize(QSize(side, side))
        if not getattr(self, '_sized', False):
            self._sized = True;self.body.setSizes([ui_px(270), ui_px(1100)])

    def _tools(self):
        """The tools act on the sequence timeline (levels 1 and 2), not on an opened block's own timeline."""
        self.tools.setEnabled(self.supported() and not self.at_clip_level())

    # Levels

    def at_clip_level(self):
        return self.stack.currentWidget() is self.clip_page

    def supported(self):
        return self.w.model_choice.currentData() in MODELS

    def _show_clip(self):
        self.stack.setCurrentWidget(self.clip_page)

    def back(self):
        """Level 3 -> the sequence view, fitted on the loaded block."""
        if not self.supported():return
        self.stack.setCurrentWidget(self.sequence_page)
        if self.current in self._ids():self.view.show_block(self.current)
        else:self.view.frame_all()
        self.refresh()

    def open_block(self, block_id, person=None):
        """Level 3 of a block; with `person`, that person is activated as a click on their People row."""
        if block_id != self.current and not self.load(block_id):return
        self._show_clip()
        if person is None:return
        people = self.w.people_list
        item = next((people.item(i) for i in range(people.count()) if people.item(i).data(Qt.UserRole) == person), None)
        if item is None:return
        if people.currentItem() is item:self.w.person_selected(item)
        else:people.setCurrentItem(item)   # the row's currentItemChanged runs person_selected, as a click does

    # The sequence

    def _ids(self):
        return {block.id for block in self.sequence.blocks}

    def new(self):
        """File > New Sequence: the loaded video (whole) as its one block at level 3, else an empty sequence; the window keeps
        what it shows."""
        w = self.w
        if self.busy():
            w.status.setText(BUSY);return
        self._reset(Sequence())
        if w.video_path is not None and w.info:
            try:self.current = self.sequence.add(w.video_path, video_length(w.video_path)).id
            except ValueError:pass
        if self.current is not None:
            self._show_clip()
            if w.project is not None or w.workspace_root is not None:self._window_loaded()
        elif self.supported():self.stack.setCurrentWidget(self.sequence_page);self.view.frame_all()
        self.refresh()

    def choose_sequence(self):
        folder = sequences_folder()
        name, _ = QFileDialog.getOpenFileName(self.w, 'Open Sequence', str(folder if folder.is_dir() else folder.parent), 'Sequence (*.json)')
        if name:self.open(Path(name))

    def open(self, path):
        """A saved sequence, shown whole, with its first block loaded; a bad file keeps the current sequence."""
        if self.busy():
            self.w.status.setText(BUSY);return False
        try:sequence = Sequence.load(path)
        except (OSError, ValueError) as error:
            self.w.status.setText(f'Cannot open the sequence: {error}');return False
        self._reset(sequence)
        if sequence.blocks:self.load(sequence.blocks[0].id)
        if self.supported():self.stack.setCurrentWidget(self.sequence_page);self.view.frame_all()
        self.refresh()
        return True

    def _save(self):
        """Keep review settings even for a single block."""
        sequence = self.sequence
        if sequence.path is None and len(sequence.blocks) <= 1 and not any(b.review for b in sequence.blocks) \
                and 'library' not in sequence.to_dict():   # nor a library that holds more than the block's video
            return
        try:sequence.save(free_name() if sequence.path is None else None)
        except OSError as error:self._say(f'Cannot save the sequence: {error}')

    def _changed(self):
        self._let_go();self.refresh();self._save()

    def _let_go(self):
        """The one place where links that left the sequence (`Sequence.dropped`: by an edit, Unlink, a relink or a changed
        people list) are let go: the copies their rows recorded are taken out of the first block's take list (files stay),
        then the list is cleared; a list that cannot be saved keeps them, so the next change tries again. The actions that
        drop a joined row's link asked first."""
        if self._take_apart(self.sequence.dropped):self.sequence.dropped.clear()

    def _take_apart(self, links):
        """Withdraw the joined copies of the rows of `links` and clear their records; the window shows the changed take list.
        False with the reason in the status bar when a take list cannot be saved."""
        try:roots, lost = sequence_join.take_apart(self.sequence, links)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._say(f'Cannot take the join back: {error}');return False
        for root in roots:self._reload(root)
        if lost:self._say(' '.join(sequence_join.REJOIN.format(name) for name in dict.fromkeys(lost)))
        return True

    def _may_take_apart(self, links):
        """No row of `links` is joined, or the user agrees to take it apart."""
        return not any(link.built for link in sequence_join.rows(self.sequence, links)) or self.ask(sequence_join.TAKE_APART, 'Joined Row')

    def _may_change(self, edit):
        """`edit` (of a Sequence) would drop no link of a joined row, or the user agrees; tried on a copy of the sequence."""
        trial = deepcopy(self.sequence)
        try:edit(trial)
        except (ValueError, KeyError):return True   # the edit itself is refused, and says why
        key = lambda link:(link.source, link.target, link.model)
        kept = {key(link) for link in trial.links}
        return self._may_take_apart([link for link in self.sequence.links if key(link) not in kept])

    def _reload(self, root):
        """After a take list of `root` changed outside a job: the window, when it shows that scene, reads it again as after an
        import, staying on the take it shows while that take is still listed."""
        w = self.w
        if w.busy or w.people_root is None or not same_file(w.people_root, root):return
        try:state = people_store.load_people(w.people_root)
        except (OSError, ValueError) as error:
            self._say(f'Cannot read the people list: {error}');return
        if state is None:return
        w.people_state = state;shown = w.project
        listed = any(same_file(people_store.safe_child(w.people_root, t['project']), shown) for p in state['people']
                     for t in (p.get('takes') or [])+(p.get('joins') or [])) if shown is not None else False
        w.activate_person(w.active_person_id, shown if listed else None)
        w.request_people_scene(force=True)

    def opened_video(self, path):
        """File > Open Video: a new sequence with that one block; the window has started loading it."""
        self._opening = Path(path)
        self._guarded(self._one_video, path)

    def _one_video(self, path):
        self._reset(Sequence())
        try:self.current = self.sequence.add(path, video_length(path)).id
        except ValueError:pass   # the window reports its own video error
        self._show_clip()
        self.refresh()

    def choose_video(self):
        name, _ = QFileDialog.getOpenFileName(self.w, 'Add Video to Sequence', '', 'Videos (*.mp4 *.mov *.avi *.mkv *.webm)')
        if name:self.add_video(Path(name))

    def add_video(self, path):
        w = self.w
        if not self.supported():
            w.status.setText(OTHER_MODEL);return None
        try:length = video_length(path)
        except ValueError as error:
            w.status.setText(f'Cannot add the video: {error}');return None
        return self._append(path, length)

    def _append(self, video, length, start=0., end=None, folders=None):
        """A new last block; it is loaded when the window showed no block of the sequence, else the sequence is shown whole."""
        empty = not self.sequence.blocks or self.current not in self._ids()
        block = self.sequence.add(video, length, start, end, folders)
        if empty:
            self.load(block.id)
        else:
            self.stack.setCurrentWidget(self.sequence_page);self.view.frame_all()
        self._changed()
        return block

    # The Project panel

    def _refused(self, video, error):
        text = f'Cannot add {Path(video).name}: {error}';self.w.log.append(text);return text

    def choose_media(self):
        names, _ = QFileDialog.getOpenFileNames(self.w, 'Import Videos', '', VIDEOS)
        if names:self.import_videos([Path(name) for name in names])

    def import_videos(self, videos):
        """List videos in the Project panel (dropped files or the Import dialog); a file that is no video is named and left
        out. Nothing is tracked or loaded, so this is allowed while a job runs."""
        said, added = [], 0
        for video in videos:
            video, known = Path(video), self.sequence.media(video) is not None
            try:self.sequence.add_media(video, video_length(video))
            except ValueError as error:
                said.append(self._refused(video, error));continue
            added += not known
        if added:
            said.append(ADDED.format(count_of(added)))
            if not self.sequence.blocks and self.supported():   # the empty timeline is where they are dragged to
                self.stack.setCurrentWidget(self.sequence_page);self.view.frame_all()
            self._changed()
        if said:self.w.status.setText(' '.join(said))

    def place(self, videos, index=None):
        """Put videos on the timeline, each as a block covering its whole video, before the block at `index` (None: at the
        end), as dropped there. The blocks made; None when nothing can be placed now (the reason is in the status bar).
        The loaded block stays loaded; on an empty timeline the first new block is loaded."""
        w = self.w
        if not self.supported():
            w.status.setText(OTHER_MODEL);return None
        if self.busy():
            w.status.setText(PLACE_BUSY);return None
        blocks = self.sequence.blocks;first = not blocks;unloaded = self.current not in self._ids()
        at, placed, said = len(blocks) if index is None else max(0, min(int(index), len(blocks))), [], []
        for video in videos:
            listed = self.sequence.media(video)
            video = Path(listed.video if listed is not None else video)   # one spelling of the path in the library and the blocks
            try:length = video_length(video)
            except ValueError as error:
                said.append(self._refused(video, error));continue
            if not self._may_change(lambda s, video=video, length=length, at=at:s.insert(at, video, length)):continue
            placed.append(self.sequence.insert(at, video, length));at += 1
        if placed:
            if unloaded:self.load(placed[0].id)
            self.stack.setCurrentWidget(self.sequence_page)
            if first:self.view.frame_all()
            self._changed()
        if said:w.status.setText(' '.join(said))
        return placed

    def remove_media(self, videos):
        """Take videos out of the Project panel. Their blocks stay on the timeline and nothing on disk is touched."""
        from .sequence import _key
        used = {_key(block.video) for block in self.sequence.blocks}
        gone = [video for video in videos if self.sequence.remove_media(video)]
        if not gone:return
        what = Path(gone[0]).name if len(gone) == 1 else count_of(len(gone))
        self._changed()
        self.w.status.setText(REMOVED.format(what, KEPT[len(gone) == 1, any(_key(video) in used for video in gone)]))

    def _show_library(self):
        """Show the library again only when its videos, or which of them are on the timeline, changed."""
        used = [block.video for block in self.sequence.blocks]
        listed = [(item.video, item.length) for item in self.sequence.library], sorted(set(used))
        if listed != self._listed:
            self._listed = listed;self.library.show_media(self.sequence.library, used)

    # Existing projects

    def _project(self, folder):
        """(model, clip folder, video, start, end, video length) of the clip folder holding `folder` (a clip, person or take;
        a named project gives its active model's clip)."""
        from . import project_workspace
        if project_workspace.is_workspace(folder):folder = project_workspace.model_project(folder, project_workspace.read(folder)['active_model'])
        root = people_store.root_for(folder)
        manifest = json.loads((root/'project.json').read_text(encoding='utf-8'))
        start, end = float(manifest['start']), float(manifest['end'])
        from .project_storage import source_video
        try:video = source_video(root, manifest)
        except (OSError, ValueError, KeyError):video = Path(manifest['source']['path'])
        try:length = max(end, video_length(video))
        except ValueError:length = end   # the video is gone; the block still opens as a project
        return manifest.get('pipeline', {}).get('model', 'gvhmr'), root, video, start, end, length

    def opened_project(self, project):
        """File > Open Project: a new sequence with that project's clip as its one block, at level 3. The window's loaded
        project is read when there is one (a named workspace opens one of its model folders)."""
        self._guarded(self._one_project, project)

    def _one_project(self, project):
        self._reset(Sequence())
        try:
            model, root, video, start, end, length = self._project(self.w.project or project)
            if model in MODELS:self.current = self.sequence.add(video, length, start, end, {model:str(root)}).id
        except (OSError, ValueError, KeyError, TypeError):pass   # the window opened it; it is not a block
        self._show_clip()
        self.refresh()

    def choose_project(self):
        name, _ = QFileDialog.getOpenFileName(self.w, 'Add Project to Sequence', str(paths.HOME/'projects'), 'Project (project.json)')
        if name:self.add_project(Path(name).parent)

    def add_project(self, folder):
        w = self.w
        if not self.supported():
            w.status.setText(OTHER_MODEL);return None
        try:model, root, video, start, end, length = self._project(folder)
        except (OSError, ValueError, KeyError, TypeError) as error:
            w.status.setText(f'Cannot add the project: {error}');return None
        if model not in MODELS:
            w.status.setText(PROJECT_MODEL);return None
        return self._append(video, length, start, end, {model:str(root)})

    def prepared(self, project, manifest):
        """A job prepared a clip folder from the loaded block's video: the block takes its range and remembers it for the
        clip's model. Called from the window's job events; an error here never fails the job."""
        self._guarded(self._prepared, project, manifest)

    def _prepared(self, project, manifest):
        found = self._clip(project, manifest)
        if found is None:return
        self.sequence.remember(self.current, *found);self._yielded = False
        self._changed()

    def _clip(self, project, manifest=None):
        """(model, clip folder, start, end) of the clip holding `project` when it was made from the loaded block's video,
        else None."""
        if self.current not in self._ids():return None
        block = self.sequence.block(self.current)
        try:
            root = people_store.root_for(project)
            if manifest is None or not same_file(root, project):manifest = json.loads((root/'project.json').read_text(encoding='utf-8'))
            start, end = float(manifest['start']), float(manifest['end'])
            model, source = manifest.get('pipeline', {}).get('model', 'gvhmr'), manifest['source']['path']
        except (OSError, ValueError, KeyError, TypeError):return None
        if model not in MODELS:return None
        from .project_storage import source_video
        try:video = source_video(root, manifest)
        except (OSError, ValueError, KeyError):video = source
        if not same_file(source, block.video) and not same_file(video, block.video):return None   # made from another video
        return model, root, start, end

    def _window_loaded(self):
        """The window switched model or loaded a project by itself, and it wins: the loaded block adopts the window's clip
        when it is the block's video and range; otherwise the window keeps what it shows until a block is loaded."""
        if not self.supported() or self.current not in self._ids():return
        w, block = self.w, self.sequence.block(self.current)
        found = self._clip(w.project) if w.project is not None else None
        same = found is not None and abs(found[2] - block.start) <= 1e-6 and abs(found[3] - block.end) <= 1e-6
        self._yielded = not same
        if not same or block.folders.get(found[0]) == str(found[1]):
            self.refresh();return
        self.sequence.remember(block.id, *found)
        self._changed()

    # Editing

    def ask(self, text, title='Change Block'):
        return QMessageBox.question(self.w, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def _source_time(self):
        """(block, source seconds) under the playhead: the window's frame in the loaded block, else the drawn playhead."""
        w = self.w
        if self.current in self._ids() and w.info and self._pending is None and not self._yielded and (w.project is not None or w.info is self._video_info):
            return self.sequence.block(self.current), w.source_offset + w.slider.value() / w.info['fps']
        return self.sequence.locate(self.view.playhead)

    def cut(self):
        """Split the block under the playhead at the playhead; the second piece becomes the loaded block."""
        if self.busy() or not self.supported():return
        found = self._source_time()
        if found is not None:self._cut(*found)

    def cut_at(self, block_id, seconds):
        """The Razor's click: split that block at that time of its video. The loaded block stays loaded, unless it is the
        one cut: then its second piece is loaded, as Cut does."""
        if self.busy() or not self.supported() or block_id not in self._ids():return
        self._cut(self.sequence.block(block_id), seconds)

    def _cut(self, block, seconds):
        if block.folders and not self.ask(CHANGE) or not self._may_change(lambda s:s.cut(block.id, seconds)):return
        try:_, second = self.sequence.cut(block.id, seconds)
        except ValueError as error:
            self.w.status.setText(str(error));return
        if block.id == self.current:self.current = second.id
        self._changed();self._follow(seconds)

    def delete(self, block_id=None):
        """Remove a block from the sequence; its files stay on disk. The neighbour of a deleted loaded block is loaded."""
        block_id = block_id or self.current
        if self.busy() or block_id not in self._ids():return
        if len(self.sequence.blocks) == 1:
            self.w.status.setText(LAST);return
        if not self._may_change(lambda s:s.delete(block_id)):return
        i = self.sequence.index(block_id);self.sequence.delete(block_id)
        if block_id == self.current:
            blocks = self.sequence.blocks
            self.current = blocks[min(i, len(blocks) - 1)].id if blocks else None
        self._changed();self._follow()

    def move(self, block_id, index):
        if self.busy() or block_id not in self._ids():return
        if not self._may_change(lambda s:s.move(block_id, index)):
            self.refresh();return
        self.sequence.move(block_id, index)
        self._changed();self._follow()

    def resize(self, block_id, start, end):
        if self.busy() or block_id not in self._ids():return
        block = self.sequence.block(block_id)
        if block.folders and not self.ask(CHANGE) or not self._may_change(lambda s:s.resize(block_id, start, end)):
            self.refresh();return
        try:self.sequence.resize(block_id, start, end)
        except ValueError as error:
            self.w.status.setText(str(error));self.refresh();return
        found = self._source_time() if block_id == self.current else None
        self._changed()
        if found:self._follow(max(block.start, min(block.end, found[1])))

    def _follow(self, seconds=None):
        """Load the loaded block again when what the window shows no longer is that block; never over a project or workspace
        the window opened by itself."""
        if not self._yielded and self.current in self._ids() and not self._shows(self.sequence.block(self.current)):self.load(self.current, seconds)

    def _shows(self, block):
        """The window shows `block` for the selected model: its remembered folder, else its video trimmed to its range."""
        w, model = self.w, self.w.model_choice.currentData()
        if self._pending is not None and self._pending[0] == block.id:return True
        folder = block.folders.get(model)
        if folder and sequence_state.matches(block, model):return self._shows_folder(folder)
        if w.project is not None or not w.info or w.info is not self._video_info or not same_file(w.video_path, block.video):return False
        start, end = w.processing_range();near = .5 / w.info['fps'] + 1e-6
        return abs(start - block.start) <= near and abs(end - block.end) <= near

    def _shows_folder(self, folder):
        w = self.w
        try:return w.project is not None and same_file(people_store.root_for(w.project), folder)
        except (OSError, ValueError, KeyError):return False

    def _trim_changed(self, *_):
        """The trim of a block without results (for any model), on its video, is the block's range; while a handle is
        dragged the block is shown and saved once the trim rests."""
        w = self.w
        if self._trimming or self._yielded or self.busy() or not self.supported() or self.current not in self._ids():return
        block = self.sequence.block(self.current)
        if block.folders or w.project is not None or self._pending is not None:return
        if not w.info or w.info is not self._video_info or not same_file(w.video_path, block.video):return
        start, end = w.processing_range()
        if abs(start - block.start) <= 1e-6 and abs(end - block.end) <= 1e-6:return
        try:self.sequence.resize(block.id, start, end)
        except ValueError:return
        self._settle.start()

    def _model_changed(self, _):
        """After a model switch in a project or workspace the window's own switch stands (the block adopts its clip when it
        fits); with neither, the loaded block is opened again for that model."""
        self.refresh()
        if not self.supported() or self.busy() or self.current not in self._ids():return
        if self.w.project is not None or self.w.workspace_root is not None:
            self._window_loaded();return
        self._yielded = False
        found = self._source_time()
        self._follow(found[1] if found and found[0].id == self.current else None)

    # Loading a block

    def load(self, block_id, seconds=None):
        """Load the block into the window: its remembered folder for the selected model, else its video trimmed to its range.
        `seconds` (source time) is then sought. False with the reason in the status bar when it cannot be loaded."""
        if self.busy():
            if block_id != self.current:self.w.status.setText(BUSY)
            return False
        return self._open(block_id, seconds)

    def _open(self, block_id, seconds=None):
        """`load` without the busy check, for the Join run that keeps the sequence busy."""
        w = self.w
        block = self.sequence.block(block_id);model = w.model_choice.currentData()
        state = sequence_state.read(block, model)
        folder = block.folders.get(model)
        if folder and sequence_state.matches(block, model):   # also with an out-of-date people list
            try:opened = w.load_project(Path(folder))
            except (OSError, ValueError, KeyError) as error:   # as File > Open Project reports it; the folder stays open
                w.status.setText('Cannot open project: '+str(error));opened = self._shows_folder(folder)
            if not opened:return False
            self._pending = None;self._loaded(block_id);self._seek(seconds)
            if not Path(block.video).is_file():w.status.setText(f'The video file is missing: {block.video}')
            return True
        if not Path(block.video).is_file():
            w.status.setText(f'Cannot open the block: {state.reason or "the video file is missing."}');return False
        if folder:w.status.setText(state.reason)   # why its saved results do not open
        if w.project is None and w.info and w.info is self._video_info and self._pending is None and same_file(w.video_path, block.video):
            self._loaded(block_id);self._trim(block, seconds)
            return True
        self._pending, self._opening = (block_id, seconds), None;self._loaded(block_id)
        w.load_video(Path(block.video))
        return True

    def _loaded(self, block_id):
        self.current, self._yielded = block_id, False;self.refresh()

    def _video_ready(self, info):
        w = self.w
        pending, self._pending = self._pending, None
        whole = w.project is None and not w._workspace_video_reference   # not a project's clip
        self._video_info = w.info if whole else None
        if pending is None and (not whole or w.workspace_root is not None and self.current in self._ids()):
            self._guarded(self._window_loaded)
            self.resume_review()
            return
        if not whole or not w.info:return
        if pending is None:
            opening, self._opening = self._opening, None
            # The window opened its video again and kept its media info (leaving PEAR or MAMMA): nothing new to add.
            if w.info is not info:return
            if self._foreign(opening):self._guarded(self._foreign_video, Path(w.video_path))
            return
        block_id, seconds = pending
        if block_id in self._ids() and same_file(w.video_path, self.sequence.block(block_id).video):
            self._trim(self.sequence.block(block_id), seconds)
            self.refresh()
            self.resume_review()

    def _foreign(self, opening):
        """The window opened a video by another way than this controller or Open Video (a drop, the guide's sample video,
        the command line), also when it is the loaded block's own video."""
        w = self.w
        if w.video_path is None or getattr(w, '_pear_active', False) or w.model_choice.currentData() == 'mamma':return False
        return opening is None or not same_file(opening, w.video_path)

    def _foreign_video(self, path):
        """A video the window opened by itself: a new last block, loaded, of a sequence of two or more blocks or with a file
        (the timeline stays where it is); else a new one-block sequence."""
        sequence = self.sequence
        if len(sequence.blocks) < 2 and sequence.path is None:return self._one_video(path)
        try:block = sequence.add(path, video_length(path))
        except ValueError:return
        self._loaded(block.id);self._save()

    def _video_failed(self, _message):
        self._pending = self._opening = None   # the window reported it; the next load opens the video again
        self._advance_to = None

    def _trim(self, block, seconds):
        w, fps = self.w, self.w.info['fps']
        self._trimming = True
        try:w.slider.set_clip(round((block.start - w.source_offset) * fps), round((block.end - w.source_offset) * fps))
        finally:self._trimming = False
        w.slider.show_frames(w.slider.clip_in, w.slider.clip_out)
        if seconds is None:w.slider.setValue(w.slider.clip_in)
        else:self._seek(seconds)
        self._playhead()

    def _seek(self, seconds):
        w = self.w
        if seconds is None or not w.info:return
        w.slider.setValue(max(w.slider.minimum(), min(w.slider.maximum(), round((seconds - w.source_offset) * w.info['fps']))))

    def edit_review(self):
        from .motion_tools import RESULTS

        if self.busy() or self.current not in self._ids():
            return
        block = self.sequence.block(self.current)
        review = block.review or {'start': block.start, 'end': block.end, 'result': None}
        dialog = QDialog(self.w)
        dialog.setWindowTitle('Review range')
        form = QFormLayout(dialog)
        first, end = QDoubleSpinBox(), QDoubleSpinBox()
        for widget, key in ((first, 'start'), (end, 'end')):
            widget.setDecimals(4)
            widget.setRange(block.start, block.end)
            widget.setSuffix(' s')
            widget.setSingleStep(1. / (self.w.info or {}).get('fps', 30))
            widget.setValue(review[key])
        choices = QComboBox()
        choices.addItem('Current result', None)
        for kind in RESULTS:
            choices.addItem(kind.replace('_', ' ').capitalize(), kind)
        choices.setCurrentIndex(choices.findData(review['result']))
        form.addRow('From (source)', first)
        form.addRow('To (source)', end)
        form.addRow('Saved result', choices)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.button(QDialogButtonBox.Reset).clicked.connect(lambda: dialog.done(2))
        form.addRow(buttons)
        answer = dialog.exec()
        if answer == QDialog.Rejected:
            return
        value = None if answer == 2 else {'start': first.value(), 'end': end.value(), 'result': choices.currentData()}
        try:
            self.sequence.set_review(block.id, value)
        except ValueError as error:
            self.w.status.setText(str(error))
            return
        self._changed()
        self.w.stop_playback()

    def review_bounds(self):
        w = self.w
        left, right = w.slider.clip_in, w.slider.clip_out
        if self.at_clip_level() or self._yielded or not self.supported() or self.current not in self._ids() or not w.info:
            return left, right
        review = self.sequence.block(self.current).review
        if review is not None:
            fps = w.info['fps']
            first = max(left, round((review['start'] - w.source_offset) * fps))
            end = min(right, round((review['end'] - w.source_offset) * fps))
            if first < end:
                return first, end
        return left, right

    def review_result(self):
        from .motion_tools import result_is_current

        w = self.w
        if self.at_clip_level() or self._yielded or not self.supported() or self.current not in self._ids():
            return True
        review = self.sequence.block(self.current).review
        requested = review['result'] if review else None
        if requested is None:
            return True
        if w.project is None or not result_is_current(w.project, requested):
            w.status.setText('The saved review result is unavailable for this block.')
            return False
        if requested != w.active_result:
            w.set_active_result(requested)
        return True

    def advance_review(self):
        if (self.at_clip_level() or self._yielded or not self.auto_advance.isChecked()
                or self.busy() or self.current not in self._ids()):
            return False
        index = self.sequence.index(self.current) + 1
        if index >= len(self.sequence.blocks):
            self.w.stop_playback()
            return True
        self.w.stop_playback()
        block = self.sequence.blocks[index]
        seconds = block.review['start'] if block.review else block.start
        self._advance_to = block.id
        if self.load(block.id, seconds):
            self.resume_review()
        else:
            self._advance_to = None
        return True

    def resume_review(self):
        if (self._advance_to is not None and self.current == self._advance_to and self._pending is None
                and not self._yielded and self.w.video_readable):
            self._advance_to = None
            if not self.at_clip_level():
                self.w.toggle_play()

    # The playhead

    def _playhead(self, *_):
        w = self.w
        if self.current in self._ids() and w.info and self._pending is None and not self._yielded:
            self.view.set_playhead(self.sequence.position(self.current, w.source_offset + w.slider.value() / w.info['fps']))

    def _scrubbed(self, seconds, final):
        found = self.sequence.locate(seconds)
        if found is None:return
        block, source = found
        if block.id == self.current and self._pending is not None:
            self._pending = block.id, source
        elif block.id == self.current:
            self._seek(source)
        else:
            self.view.set_playhead(seconds)
            # A click tried this block already (its `selected` comes first); a refused load leaves the playhead in the loaded block.
            if final and (block.id == self._clicked or not self.load(block.id, source)):self._playhead()

    def _select(self, block_id):
        if block_id == self.current:return
        self._clicked = block_id;QTimer.singleShot(0, self._click_done)
        self.load(block_id)

    def _click_done(self):
        self._clicked = None

    # People who continue across neighbouring blocks

    def _states(self):
        model = self.w.model_choice.currentData()
        return {block.id:sequence_state.read(block, model) for block in self.sequence.blocks}

    def _may_link(self):
        """Linking is refused while a job runs and unavailable for a model the sequence does not support."""
        if self.busy():
            self.w.status.setText(BUSY);return False
        if not self.supported():
            self.w.status.setText(OTHER_MODEL);return False
        self.sequence.model = self.w.model_choice.currentData()
        return True

    @staticmethod
    def _name(states, ident):
        state = states.get(ident[0])
        return next((p.name for p in state.people if p.id == ident[1]), f'Person {ident[1]}') if state else f'Person {ident[1]}'

    def _video(self, ident):
        return Path(self.sequence.block(ident[0]).video).name

    @staticmethod
    def _prints(states, source, target):
        """The people-list fingerprints of the earlier and the later block, which notice Find person again."""
        return tuple(getattr(states.get(person[0]), 'fingerprint', None) for person in (source, target))

    def link(self, source, target):
        """`target` (later block id, person id) continues `source` (a person of the block before), replacing their other
        links of the selected model and keeping the method of the link `target` had; False with the reason in the status
        bar."""
        if not self._may_link():return False
        sequence, source, target = self.sequence, tuple(source), tuple(target)
        if not {source[0], target[0]} <= self._ids():return False
        old = sequence.link_into(*target)
        if old is not None and old.source == source:return True
        states = self._states()
        reason = sequence_links.refusal(sequence, states, source, target, self.w.model_choice.currentData())
        if reason:
            self.w.status.setText(reason);return False
        method, prints = old.method if old else DEFAULT_METHOD, self._prints(states, source, target)
        if not self._may_change(lambda s:s.link(source, target, method, prints)):return False
        before = list(sequence.links)
        try:sequence.link(source, target, method, prints)
        except ValueError as error:
            self.w.status.setText(str(error));return False
        sequence.dropped.extend(link for link in before if not any(link is kept for kept in sequence.links))
        self._say(LINKED.format(self._name(states, target), self._name(states, source)))   # before the change, whose own news wins
        self._changed()
        return True

    def unlink(self, target):
        """Take the link into `target` away (a joined row is taken apart, after asking); its later block is not linked by
        itself again."""
        if not self._may_link():return False
        link = self.sequence.link_into(*tuple(target))
        if link is None or not self._may_take_apart([link]):return False
        states = self._states()
        self._say(UNLINKED.format(self._name(states, link.target), self._name(states, link.source)))
        self.sequence.unlink(tuple(target));self.sequence.dropped.append(link);self._changed()
        return True

    # Join

    def _chains(self, targets=None):
        """The linked rows to join: the rows of the bars `targets`, else every linked row, in block order."""
        sequence = self.sequence
        targets = [link.target for link in sequence.model_links()] if targets is None else targets
        chains = list(dict.fromkeys(tuple(sequence.chain(*target)) for target in targets))
        return sorted((list(chain) for chain in chains if len(chain) > 1), key=lambda chain:(sequence.index(chain[0][0]), str(chain[0][1])))

    def join(self, target=None):
        """Join the row of the bar `target` (block id, person id), or every linked row: asks, then runs; False when nothing
        started."""
        if not self._may_link():return False
        if target is not None and tuple(target)[0] not in self._ids():return False
        return self._join(self._chains(None if target is None else [tuple(target)]), target is None)

    def _join(self, chains, every=False):
        """Join the rows `chains` that are not joined yet, after the question; with `every` they are all the linked rows."""
        model, states = self.w.model_choice.currentData(), self._states()
        joined = [chain for chain in chains if sequence_links.status(self.sequence, states, chain[0], model) == 'joined']
        if chains and len(joined) == len(chains):   # Join never rebuilds what is current
            self.w.status.setText(sequence_join.ALL_JOINED if every else ' '.join(sequence_join.ALREADY.format(self._name(states, chain[0])) for chain in chains));return False
        chains = [chain for chain in chains if chain not in joined]
        if not chains or not self.ask(sequence_join.QUESTION, 'Join'):return False
        self.joiner = sequence_join.Runner(self, chains);self.joiner.start()
        return True

    # The seam between two blocks

    def _seam_changed(self, later_id):
        """The pointer came onto a seam of the timeline (its card shows once it has rested there) or left it (the card goes
        after a moment, unless the pointer went onto the card)."""
        if later_id is None:
            self._seam_show.stop()
            if not self.seam_card.isHidden():self._seam_hide.start()
            return
        self._seam_hide.stop();self._seam_wanted = later_id
        if self.seam_card.isHidden() or self.seam_card.later != later_id:self._seam_show.start()

    def show_seam(self, later_id=None):
        """Show the card of the seam before block `later_id` (default: the seam the pointer is on) above that seam, and
        return what it offers. No card, and None, while a job runs, inside a block (level 3), with a model the sequence does
        not work with, and for a block that has none before it."""
        w, card = self.w, self.seam_card
        later_id = self._seam_wanted if later_id is None else later_id
        if self.busy() or not self.supported() or self.at_clip_level() or later_id not in [b.id for b in self.sequence.blocks[1:]]:
            self.hide_seam();return None
        model = self.sequence.model = w.model_choice.currentData()
        offer = sequence_seam.offer(self.sequence, self._states(), later_id, model)
        self._seam_show.stop();self._seam_hide.stop()
        card.show_offer(later_id, offer)
        above, margin = self.view.mapTo(w, QPoint(self.view.seam_rect(later_id).center().x(), self.view.TRACK_TOP)), ui_px(4)
        card.move(max(margin, min(w.width() - card.width() - margin, above.x() - card.width() // 2)),
                  max(margin, above.y() - card.height() - ui_px(2)))
        card.show();card.raise_()
        return offer

    def hide_seam(self):
        self._seam_show.stop();self._seam_hide.stop();self.seam_card.hide()

    def _seam_left(self):
        """The pointer has been off the seam for a moment: the card goes, unless the pointer is on it."""
        if not self.seam_card.underMouse():self.hide_seam()

    def _seam_action(self, action):
        """A button of the card: the card goes first, since the action may ask or open a block."""
        self.hide_seam();action()

    def join_seam(self, later_id):
        """Join the tracking across the seam before block `later_id`: one person on each side who are not linked are
        linked first, then every linked row that crosses the seam and is not joined yet is joined, after Join's question.
        False, with the reason in the status bar, when nothing started."""
        if not self._may_link() or later_id not in [b.id for b in self.sequence.blocks[1:]]:return False
        offer = sequence_seam.offer(self.sequence, self._states(), later_id, self.sequence.model)
        if not offer.join:
            self.w.status.setText(' '.join(offer.lines));return False
        if offer.link is not None and not self.link(*offer.link):return False
        return self._join(self._chains([link.target for link in self.sequence.model_links() if link.target[0] == later_id]))

    def take_back(self, target):
        """Take the joined copies of the row of the bar `target` out of its first block's take list, after asking when it has
        copies; the links stay. False when the user says No or the list cannot be saved."""
        if not self._may_link():return False
        target = tuple(target);links = [l for l in (self.sequence.link_into(*target), self.sequence.link_out(*target)) if l]
        if not self._may_take_apart(links):return False
        done = self._take_apart(links);self.refresh();self._save()
        return done

    def _joined(self, target):
        """The receiving person (block id, person id) and the folder of the joined take of `target`'s row while it is joined,
        else None."""
        model = self.w.model_choice.currentData()
        if sequence_links.status(self.sequence, {}, target, model) != 'joined':return None
        row = self.sequence.chain(*target);copies = [self.sequence.link_into(*person).built['copy'] for person in row[1:]]
        try:folder = sequence_join.joined_take(self.sequence.block(row[0][0]).folders[model], row[0][1], copies)
        except (OSError, ValueError, KeyError, StopIteration):folder = None
        return (row[0], folder) if folder is not None else None

    def open_joined(self, target):
        """Level 3 of the first block of `target`'s joined row, its receiving person active on the joined take."""
        if not self._may_link():return False
        found = self._joined(tuple(target)) if tuple(target)[0] in self._ids() else None
        if found is None:return False
        (block_id, person), folder = found
        if not self.load(block_id):return False
        self.open_block(block_id, person)
        if not same_file(self.w.project, folder):self.w.activate_person(person, folder)
        return same_file(self.w.project, folder)

    def export_joined(self, target):
        """The joined animation of `target`'s row opened, then the window's own export of the current take."""
        if not self.open_joined(target):return False
        if self.w.export_action.isEnabled():self.w.export_action.trigger()
        else:self.w.save_fbx()   # says why it cannot export now
        return True

    def _opened(self, block_id, person):
        """A double-click on a bar: the joined animation for a bar of a joined row, else the block as before."""
        if person is not None and self.supported() and not self.busy():   # while a job runs the loaded block opens as before
            self.sequence.model = self.w.model_choice.currentData()
            if self._joined((block_id, person)) is not None:return self.open_joined((block_id, person))
        self.open_block(block_id, person)

    def set_transition(self, target, method, seconds):
        """The method of the link into `target` and the transition length of its pair of blocks; False with the reason, and
        nothing changed, when the method cannot fill that length."""
        if not self._may_link():return False
        target = tuple(target)
        if target[0] not in self._ids() or self.sequence.link_into(*target) is None:return False
        issue = transition_issue(method, seconds)
        if issue:
            self.w.status.setText(issue);return False
        try:self.sequence.set_gap(target[0], seconds)
        except ValueError as error:
            self.w.status.setText(str(error));return False
        self.sequence.set_method(target, method)
        self._say(TRANSITION.format(round(float(seconds), 2), METHOD_LABELS[method]));self._changed()
        return True

    def edit_transition(self, target):
        """The transition dialog of the link into `target`: Method and Length. OK applies them; a length the method cannot
        fill keeps the dialog open with the reason."""
        if not self._may_link():return
        target = tuple(target);link = self.sequence.link_into(*target) if target[0] in self._ids() else None
        if link is None:return
        dialog = QDialog(self.w)
        dialog.setWindowTitle('Transition')
        form = QFormLayout(dialog)
        method, length = QComboBox(), QDoubleSpinBox()
        for key in METHODS:
            method.addItem(METHOD_LABELS[key], key)
        method.setCurrentIndex(method.findData(link.method))
        length.setDecimals(2)
        length.setRange(.04, 9.6)   # one frame (1/30 s) as Sequence.set_gap accepts it, in two decimals
        length.setSingleStep(.1)
        length.setSuffix(' s')
        length.setValue(self.sequence.gap(target[0]))
        note = QLabel()
        note.setObjectName('muted')
        note.setWordWrap(True)
        note.setMinimumWidth(ui_px(280))
        form.addRow('Method', method)
        form.addRow('Length', length)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        form.addRow(buttons)
        def say(text):   # the dialog grows when the note wraps onto more lines than it had room for
            note.setText(text);form.activate()
            dialog.resize(dialog.width(), max(dialog.height(), dialog.heightForWidth(dialog.width()), dialog.sizeHint().height()))
        def explain():
            say(EXPERIMENTAL if method.currentData() == 'auto' else '')
        def apply():
            issue = transition_issue(method.currentData(), length.value())
            if issue:
                say(issue);return
            if not self.set_transition(target, method.currentData(), length.value()):
                say(self.w.status.text());return   # its own refusal stays in the dialog too
            dialog.accept()
        method.currentIndexChanged.connect(explain)
        length.valueChanged.connect(explain)
        buttons.accepted.connect(apply)
        buttons.rejected.connect(dialog.reject)
        explain()
        dialog.exec()
        dialog.deleteLater()

    def person_menu(self, block_id, person, point):
        menu = self._person_menu(block_id, person)
        if menu is not None:
            menu.exec(point);menu.deleteLater()

    def _person_menu(self, block_id, person):
        """A person bar's menu: Continue after with the people of the block before (the one continued checked, a refused one
        disabled with its reason as tooltip) and Unlink when linked; None while linking is refused."""
        if not self._may_link() or block_id not in self._ids():return None
        sequence, states, target = self.sequence, self._states(), (block_id, person)
        current, i = sequence.link_into(*target), sequence.index(block_id)
        menu = QMenu(self.w);menu.setToolTipsVisible(True)
        after = QMenu('Continue after', menu);after.setToolTipsVisible(True);menu.addMenu(after)
        if i:
            earlier = sequence.blocks[i - 1]
            for giver in states[earlier.id].people:
                source = (earlier.id, giver.id);action = after.addAction(giver.name)
                action.setCheckable(True);action.setChecked(current is not None and current.source == source)
                reason = sequence_links.refusal(sequence, states, source, target, self.w.model_choice.currentData())
                if reason:
                    action.setEnabled(False);action.setToolTip(tooltip_text(reason))
                action.triggered.connect(lambda *_, source=source:self.link(source, target))
        after.setEnabled(not after.isEmpty())
        if current is not None:menu.addAction('Unlink').triggered.connect(lambda *_:self.unlink(target))
        if current is not None or sequence.link_out(*target) is not None:
            menu.addAction('Join').triggered.connect(lambda *_:self.join(target))
            if self._joined(target) is not None:
                menu.addAction('Open joined animation').triggered.connect(lambda *_:self.open_joined(target))
                menu.addAction('Export joined animation...').triggered.connect(lambda *_:self.export_joined(target))
        return menu

    def _keep_links(self, states, model):
        """Drop the links a changed people list made stale (their later block may be linked by itself again), then make the
        automatic links; never while a job runs or the window shows a project of its own. Saves and says so when anything
        changed."""
        if self.busy() or self._yielded:return
        sequence, texts = self.sequence, []
        gone = sequence_links.stale(sequence, states)
        for link in gone:
            sequence.links.remove(link);sequence.decided.discard(link.target[0]);sequence.dropped.append(link)
        texts += [STALE.format(name) for name in dict.fromkeys(self._changed_video(link, states) for link in gone)]
        if gone:self._let_go()
        for source, target in sequence_links.automatic(sequence, states, model):
            sequence.link(source, target, prints=self._prints(states, source, target))
            texts.append(AUTOMATIC.format(self._name(states, target), self._video(target), self._name(states, source), self._video(source)))
        if texts:
            self._save();self._say(' '.join(texts))

    def _changed_video(self, link, states):
        """The video name of the block whose people list no longer holds `link`, the later block first."""
        ids = self._ids()
        def changed(person, mark):
            state = states.get(person[0])
            return person[0] in ids and state is not None and state.fingerprint is not None and (
                mark not in (None, state.fingerprint) or all(p.id != person[1] for p in state.people))
        found = [person[0] for person, mark in ((link.target, link.prints[1]), (link.source, link.prints[0])) if changed(person, mark)]
        found = found or [person[0] for person in (link.target, link.source) if person[0] in ids]
        return Path(self.sequence.block(found[0]).video).name if found else 'a block'

    # State

    def refresh(self):
        """Re-read every block's state; keep the links (for the selected model, when the sequence supports it); update the
        view, the buttons and the playhead."""
        model = self.sequence.model = self.w.model_choice.currentData()   # the document's link queries see this model's links
        states, rows, row_states = self._states(), None, None
        if self.supported():
            self._keep_links(states, model)
            rows = sequence_links.rows(self.sequence, states)
            row_states = {link.target:sequence_links.status(self.sequence, states, link.target, model) for link in self.sequence.model_links()}
        self.view.show_sequence(self.sequence, states, rows, row_states)
        self.view.set_current(self.current if self.current in self._ids() else None)
        self._show_library()
        count = len(self.sequence.blocks)
        self.summary.setText(f'{count} block{"" if count == 1 else "s"} | {format_time(self.sequence.total())}')
        supported, busy, loaded = self.supported(), self.busy(), self.current in self._ids()
        self.view.editable = not busy
        self.view.set_busy(self.current if busy and loaded else None)
        self.review_button.setEnabled(supported and not busy and loaded)
        self.auto_advance.setEnabled(supported and not busy and count > 1)
        for key, button, enabled in (('back', self.back_button, True), ('add', self.add_button, not busy), ('frame', self.frame_button, True),
                                     ('cut', self.cut_button, not busy and loaded), ('delete', self.delete_button, not busy and loaded),
                                     ('join', self.join_button, not busy and bool(self.sequence.model_links()))):
            button.setEnabled(supported and enabled);button.setToolTip(tooltip_text(TIPS[key] if supported else OTHER_MODEL))
        if not supported:self._show_clip()
        self.library.setEnabled(supported);self.library.setToolTip('' if supported else tooltip_text(OTHER_MODEL))
        self._tools()
        self._playhead()
        if not self.seam_card.isHidden():self.show_seam(self.seam_card.later)   # what it offers may have changed

    def eventFilter(self, watched, event):
        w = self.w
        if watched is w.slider and event.type() == QEvent.Wheel and not event.modifiers() & Qt.ShiftModifier and event.angleDelta().y() < 0:
            full = w.slider.view_span is None or w.slider.view_span >= max(1, w.slider.maximum() - w.slider.minimum())
            if full and self.supported():
                self.back();return True
        elif watched is w.slider and event.type() in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
            # A video dragged onto an opened block's timeline goes after that block; anything else is left to the window.
            videos = dragged_videos(event.mimeData()) if self.supported() and not self.busy() else []
            if not videos:
                event.ignore();return True
            event.setDropAction(Qt.CopyAction);event.accept()
            if event.type() == QEvent.Drop:
                self.place(videos, self.sequence.index(self.current) + 1 if self.current in self._ids() else None)
            return True
        elif watched is w.parts_button and event.type() == QEvent.EnabledChange:
            self.refresh()   # update_readiness after Reconstruct in parts started or stopped
        elif watched is self.view and event.type() in (QEvent.MouseButtonPress, QEvent.Wheel):
            self.hide_seam()   # the press trims or selects, the wheel moves the seam away
        elif watched is self.view and event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._appearance()   # the UI scale or the theme changed
        elif watched is w and event.type() == QEvent.Close:
            self.library.stop()
        elif watched is w and event.type() == QEvent.ChildAdded and event.child().isWidgetType():
            event.child().installEventFilter(self)   # the guide's overlay and card are children of the window
        elif watched is not w and watched.parent() is w and event.type() == QEvent.Show:
            guide = getattr(w, 'tutorial', None)
            if guide is not None and guide.active and not self.at_clip_level():
                self._show_clip();QTimer.singleShot(0, guide.tick)   # its highlighted timeline widgets are on screen again
        return False


def action_bar(w):
    """The Timeline panel's action bar: under the shared header, above the slider; None when the panel has changed."""
    layout = w.timeline_panel.layout()
    actions = layout.itemAt(1).widget() if layout.count() > 2 else None
    if actions is None or actions is w.slider or layout.itemAt(2).widget() is not w.slider or actions.layout() is None:return None
    return actions


class Absent:
    """The window without the sequence: its hooks do nothing."""
    def __init__(self, window):
        self.window = window

    def review_bounds(self):
        return self.window.slider.clip_in, self.window.slider.clip_out

    def review_result(self):
        return True

    def advance_review(self):
        return False

    def opened_video(self, path):pass
    def opened_project(self, project):pass
    def prepared(self, project, manifest):pass


def build(w, file_menu):
    if action_bar(w) is None:   # the app starts without the sequence
        w.log.append(PANEL);w.status.setText(PANEL)
        w.sequence_ui = Absent(w);return
    w.sequence_ui = Controller(w, file_menu)
