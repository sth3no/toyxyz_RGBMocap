"""The sequence in the main window: one stacked timeline with today's slider at level 3 (page "clip") and the sequence view
at levels 1 and 2 (page "sequence"), loading the selected block into the window, the playhead in sequence time and the way
between the levels. Play and the transport stay the window's own and act on the loaded block."""
import math
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from . import sequence_state
from .clip_range import format_time
from .parts_runner import MODELS
from .sequence import Sequence
from .sequence_view import SequenceView
from .theme import ui_px
from .tooltips import tooltip_text

OTHER_MODEL = 'The sequence works with GVHMR, ExMR and FlowHMR.'
TIPS = {'back': 'Show the sequence: all blocks on one timeline.', 'add': 'Add a video at the end of the sequence.',
        'cut': 'Split the selected block at the playhead.', 'delete': 'Remove the selected block from the sequence.',
        'frame': 'Fit the whole sequence in the timeline.'}


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


def same_file(a, b):
    try:return Path(a).resolve() == Path(b).resolve()
    except (OSError, TypeError):return False


class Controller(QObject):
    def __init__(self, w, file_menu):
        super().__init__(w)
        self.w, self.sequence, self.current = w, Sequence(), None
        self._pending = None  # (block id, source seconds or None): the block's video is opening
        self._build()
        action = file_menu.addAction('Add Video to Sequence...', self.choose_video)
        action.setToolTip(tooltip_text('Add a video at the end of the sequence, as a new block covering the whole video.'))
        w.open_actions.append(action)
        # After the window's own slots, so the clip is set up when these run.
        w.video_view.metadata_ready.connect(self._video_ready)
        w.slider.valueChanged.connect(self._playhead)
        w.slider.trim_enabled_changed.connect(lambda _:self.refresh())
        w.model_choice.currentIndexChanged.connect(lambda _:self.refresh())
        w.slider.installEventFilter(self)
        self.view.installEventFilter(self)
        w.installEventFilter(self)
        self.refresh()

    def _build(self):
        w = self.w;layout = w.timeline_panel.layout()
        actions = layout.itemAt(1).widget()   # the action bar, then the slider, under the shared header
        self.clip_page, self.sequence_page = QWidget(), QWidget()
        clip = QVBoxLayout(self.clip_page);clip.setContentsMargins(0, 0, 0, 0);clip.setSpacing(0)
        for widget in (actions, w.slider):layout.removeWidget(widget)
        clip.addWidget(actions);clip.addWidget(w.slider, 1)
        self.back_button = QPushButton('← Sequence');self.back_button.clicked.connect(self.back)
        actions.layout().insertWidget(0, self.back_button)
        bar = self.bar = QWidget();bar.setObjectName('editorHeader');bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row = QHBoxLayout(bar)
        self.add_button, self.cut_button, self.delete_button, self.frame_button = (QPushButton(text) for text in ('Add Video...', 'Cut', 'Delete', 'Frame All'))
        self.add_button.clicked.connect(self.choose_video)
        self.frame_button.clicked.connect(lambda:self.view.frame_all())
        for button in (self.add_button, self.cut_button, self.delete_button, self.frame_button):row.addWidget(button)
        self.summary = QLabel('');self.summary.setMinimumWidth(0);self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.summary, 1)
        self.view = SequenceView()
        page = QVBoxLayout(self.sequence_page);page.setContentsMargins(0, 0, 0, 0);page.setSpacing(0)
        page.addWidget(bar);page.addWidget(self.view, 1)
        self.stack = QStackedWidget();self.stack.addWidget(self.clip_page);self.stack.addWidget(self.sequence_page)
        layout.insertWidget(1, self.stack, 1)
        self.view.selected.connect(lambda id:id != self.current and self.load(id))
        self.view.opened.connect(self.open_block)
        self.view.scrubbed.connect(self._scrubbed)
        self._appearance()

    def _appearance(self):
        self.bar.layout().setContentsMargins(ui_px(9), ui_px(3), ui_px(6), ui_px(3))
        self.view.refresh_appearance()

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
        self.sequence, self.current, self._pending = Sequence(), None, None
        self.refresh()

    def opened_video(self, path):
        """File > Open Video: a new sequence with that one block; the window has started loading it."""
        self.sequence, self.current, self._pending = Sequence(), None, None
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
        empty = not self.sequence.blocks or self.current not in self._ids()
        block = self.sequence.add(path, length)
        if empty:
            self.load(block.id)
        else:
            self.stack.setCurrentWidget(self.sequence_page);self.view.frame_all()
        self.refresh()
        return block

    # Loading a block

    def load(self, block_id, seconds=None):
        """Load the block into the window: its remembered folder for the selected model, else its video trimmed to its range.
        `seconds` (source time) is then sought. False with the reason in the status bar when it cannot be loaded."""
        w = self.w;block = self.sequence.block(block_id);model = w.model_choice.currentData()
        state = sequence_state.read(block, model)
        folder = block.folders.get(model)
        if folder and not state.reason:
            if not w.load_project(Path(folder)):return False
            self._pending = None;self._loaded(block_id);self._seek(seconds)
            return True
        if not Path(block.video).is_file():
            w.status.setText(f'Cannot open the block: {state.reason or "the video file is missing."}');return False
        if folder:w.status.setText(state.reason)   # why its saved results do not open
        if w.project is None and same_file(w.video_path, block.video) and (w.info or self._pending):
            if w.info and self._pending is None:
                self._loaded(block_id);self._trim(block, seconds)
            else:
                self._pending = block_id, seconds;self._loaded(block_id)   # the same video is still opening
            return True
        self._pending = block_id, seconds;self._loaded(block_id)
        w.load_video(Path(block.video))
        return True

    def _loaded(self, block_id):
        self.current = block_id;self.refresh()

    def _video_ready(self, _info):
        w = self.w
        if self._pending is None or w.project is not None or not w.info:return
        block_id, seconds = self._pending;self._pending = None
        if block_id in self._ids() and same_file(w.video_path, self.sequence.block(block_id).video):
            self._trim(self.sequence.block(block_id), seconds)
            self.refresh()

    def _trim(self, block, seconds):
        w, fps = self.w, self.w.info['fps']
        w.slider.set_clip(round((block.start - w.source_offset) * fps), round((block.end - w.source_offset) * fps))
        w.slider.show_frames(w.slider.clip_in, w.slider.clip_out)
        if seconds is None:w.slider.setValue(w.slider.clip_in)
        else:self._seek(seconds)
        self._playhead()

    def _seek(self, seconds):
        w = self.w
        if seconds is None or not w.info:return
        w.slider.setValue(max(w.slider.minimum(), min(w.slider.maximum(), round((seconds - w.source_offset) * w.info['fps']))))

    # The playhead

    def _playhead(self, *_):
        w = self.w
        if self.current in self._ids() and w.info and self._pending is None:
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
            if final:self.load(block.id, source)

    # State

    def refresh(self):
        """Re-read every block's state; update the view, the buttons and the playhead."""
        model = self.w.model_choice.currentData()
        self.view.show_sequence(self.sequence, {block.id:sequence_state.read(block, model) for block in self.sequence.blocks})
        self.view.set_current(self.current if self.current in self._ids() else None)
        count = len(self.sequence.blocks)
        self.summary.setText(f'{count} block{"" if count == 1 else "s"} | {format_time(self.sequence.total())}')
        supported = self.supported()
        for key, button in (('back', self.back_button), ('add', self.add_button), ('frame', self.frame_button)):
            button.setEnabled(supported);button.setToolTip(tooltip_text(TIPS[key] if supported else OTHER_MODEL))
        for key, button in (('cut', self.cut_button), ('delete', self.delete_button)):
            button.setEnabled(False);button.setToolTip(tooltip_text(TIPS[key] if supported else OTHER_MODEL))
        if not supported:self._show_clip()
        self._playhead()

    def eventFilter(self, watched, event):
        w = self.w
        if watched is w.slider and event.type() == QEvent.Wheel and not event.modifiers() & Qt.ShiftModifier and event.angleDelta().y() < 0:
            full = w.slider.view_span is None or w.slider.view_span >= max(1, w.slider.maximum() - w.slider.minimum())
            if full and self.supported():
                self.back();return True
        elif watched is self.view and event.type() == QEvent.FontChange:
            self._appearance()   # the UI scale changed
        elif watched is w and event.type() == QEvent.ChildAdded and event.child().isWidgetType():
            event.child().installEventFilter(self)   # the guide's overlay and card are children of the window
        elif watched is not w and watched.parent() is w and event.type() == QEvent.Show:
            guide = getattr(w, 'tutorial', None)
            if guide is not None and guide.active and not self.at_clip_level():
                self._show_clip();QTimer.singleShot(0, guide.tick)   # its highlighted timeline widgets are on screen again
        return False


def build(w, file_menu):
    w.sequence_ui = Controller(w, file_menu)
