"""The Project panel: the library of videos beside the timeline. It lists the sequence's videos with a picture and their
length, takes video files dropped from the file manager, and lets its videos be dragged to the timeline (sequence_view.MIME).
It changes nothing itself: every action is a signal. Pictures and each video's size and rate are read off the GUI thread."""
import json
import math
from pathlib import Path

from PySide6.QtCore import QMimeData, QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from .sequence_view import MIME, VIDEO_SUFFIXES
from .theme import theme_color, ui_px
from .tooltips import tooltip_text

EMPTY = 'Drop video files here, or click Import.'
IMPORT = 'Add videos to the library. Drag a video from here to the timeline to track a part of it.'
USED, UNUSED, MISSING = 'On the timeline.', 'Not on the timeline yet. Drag it to the timeline.', 'The file is missing.'
PATH = Qt.UserRole


def clock(seconds):
    """4:08 or 1:02:05, as a clip's length reads in a media bin."""
    seconds = int(round(max(0., seconds)));minutes, seconds = divmod(seconds, 60);hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes}:{seconds:02d}'


def key(path):
    from .sequence import _key
    return _key(path)


class _Reader(QObject):
    done = Signal(str, object, object)   # path, QImage or None, info dict or None (missing or unreadable)


class _Read(QRunnable):
    """Reads one frame near the start of a video and its size, rate and length; never raises."""
    def __init__(self, reader, path, width, height):
        super().__init__();self.reader, self.path, self.size = reader, path, (width, height)

    def run(self):
        image = info = None
        try:
            if Path(self.path).is_file():
                import cv2
                capture = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500])
                try:
                    if capture.isOpened():
                        width, height = int(capture.get(3)), int(capture.get(4))
                        frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
                        if width > 0 and height > 0:info = dict(width=width, height=height, fps=fps if math.isfinite(fps) else 0., frames=frames)
                        if frames > 30 and fps > 0:capture.set(cv2.CAP_PROP_POS_FRAMES, min(frames - 1, round(fps)))   # a second in: past a black first frame
                        ok, frame = capture.read()
                        if ok and frame is not None:
                            scale = min(self.size[0] / frame.shape[1], self.size[1] / frame.shape[0])
                            small = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
                finally:
                    capture.release()
        except Exception:
            image = None
        try:self.reader.done.emit(self.path, image, info)
        except RuntimeError:pass   # the panel is gone


class MediaList(QListWidget):
    """Thumbnails in a grid; dragging them out carries their paths, dropping video files on them imports the files."""
    files_dropped = Signal(object)
    delete_pressed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.IconMode);self.setResizeMode(QListView.Adjust);self.setMovement(QListView.Static)
        self.setWrapping(True);self.setWordWrap(True);self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True);self.setAcceptDrops(True);self.setDragDropMode(QAbstractItemView.DragDrop);self.setDefaultDropAction(Qt.CopyAction)
        self.viewport().setAcceptDrops(True)   # a scroll area gets its drags through its viewport
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def mimeTypes(self):return [MIME]

    def mimeData(self, items):
        mime = QMimeData();mime.setData(MIME, json.dumps([item.data(PATH) for item in items]).encode('utf-8'));return mime

    def supportedDragActions(self):return Qt.CopyAction

    def _files(self, mime):
        if mime.hasFormat(MIME):return []   # its own videos are already here
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile() and url.toLocalFile().lower().endswith(VIDEO_SUFFIXES)]

    def _take(self, event):
        if self._files(event.mimeData()):
            event.setDropAction(Qt.CopyAction);event.accept()
        else:event.ignore()

    def dragEnterEvent(self, event):self._take(event)

    def dragMoveEvent(self, event):self._take(event)

    def dropEvent(self, event):
        found = self._files(event.mimeData())
        if not found:
            event.ignore();return
        event.setDropAction(Qt.CopyAction);event.accept()
        self.files_dropped.emit(found)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_pressed.emit();event.accept()
        else:super().keyPressEvent(event)


class Library(QWidget):
    imported = Signal(object)      # video files dropped from the file manager
    placed = Signal(str)           # a video double-clicked: put it on the timeline
    removed = Signal(object)       # the selected videos, Delete pressed
    import_requested = Signal()    # the Import button

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pictures, self._info, self._asked = {}, {}, set()   # by path key: QIcon, info dict or None, read started
        self._reader = _Reader(self);self._reader.done.connect(self._read_done)
        self._pool = QThreadPool(self);self._pool.setMaxThreadCount(2)
        self._used = set()
        layout = QVBoxLayout(self);layout.setContentsMargins(0, 0, 0, 0);layout.setSpacing(0)
        self.bar = QWidget();self.bar.setObjectName('editorHeader');row = QHBoxLayout(self.bar)
        self.title = QLabel('Project');self.count = QLabel('');self.count.setObjectName('muted')
        self.import_button = QPushButton('Import...');self.import_button.setToolTip(tooltip_text(IMPORT))
        self.import_button.clicked.connect(lambda *_:self.import_requested.emit())
        row.addWidget(self.title);row.addWidget(self.count, 1);row.addWidget(self.import_button)
        self.list = MediaList();self.list.setAccessibleName('Project videos')
        self.hint = QLabel(EMPTY, self.list.viewport());self.hint.setObjectName('muted');self.hint.setAlignment(Qt.AlignCenter);self.hint.setWordWrap(True)
        self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.bar);layout.addWidget(self.list, 1)
        self.list.files_dropped.connect(self.imported.emit)
        self.list.itemDoubleClicked.connect(lambda item:self.placed.emit(item.data(PATH)))
        self.list.delete_pressed.connect(self._delete)
        self.list.viewport().installEventFilter(self)
        self.refresh_appearance()

    def refresh_appearance(self):
        self.bar.layout().setContentsMargins(ui_px(9), ui_px(3), ui_px(6), ui_px(3))
        self._picture = QSize(ui_px(96), ui_px(54))   # two rows fit the Timeline panel at its usual height
        self.list.setIconSize(self._picture);self.list.setGridSize(QSize(self._picture.width() + ui_px(14), self._picture.height() + ui_px(40)))
        self.list.setSpacing(ui_px(4));self.setMinimumWidth(ui_px(150))
        for row in range(self.list.count()):self._dress(self.list.item(row))   # the placeholders take the theme's colour
        self._place_hint()

    def eventFilter(self, watched, event):
        if watched is self.list.viewport() and event.type() == event.Type.Resize:self._place_hint()
        return False

    def stop(self):
        """The window closes: no further video is read, and the reads in progress let go of their files first."""
        self._pool.clear();self._pool.waitForDone(5000)

    def _place_hint(self):
        self.hint.setGeometry(self.list.viewport().rect().adjusted(ui_px(8), ui_px(8), -ui_px(8), -ui_px(8)))

    def _delete(self):
        chosen = [item.data(PATH) for item in self.list.selectedItems()]
        if chosen:self.removed.emit(chosen)

    def selected(self):
        return [item.data(PATH) for item in self.list.selectedItems()]

    def show_media(self, media, used):
        """The library's videos; `used` holds the videos that have a block on the timeline."""
        chosen = {key(path) for path in self.selected()}
        self._used = {key(path) for path in used}
        self.list.clear()
        for entry in media:
            item = QListWidgetItem(f'{Path(entry.video).name}\n{clock(entry.length)}')
            item.setData(PATH, entry.video);item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.list.addItem(item);self._dress(item)
            if key(entry.video) in chosen:item.setSelected(True)
        count = len(media)
        self.count.setText(f'{count} video{"" if count == 1 else "s"}' if count else '')
        self.hint.setVisible(not count)

    def _dress(self, item):
        """The item's picture and tooltip from what is known; the first time, the video is read in the background."""
        path = item.data(PATH);ident = key(path)
        item.setIcon(self._pictures.get(ident) or QIcon(self._blank()))
        lines = [str(Path(path))]
        if ident in self._info:
            info = self._info[ident]
            lines.append(MISSING if info is None else f"{info['width']}×{info['height']} · {info['fps']:.2f} fps")
        lines.append(USED if ident in self._used else UNUSED)
        item.setToolTip('\n'.join(lines))
        if ident not in self._asked:
            self._asked.add(ident)
            self._pool.start(_Read(self._reader, path, self._picture.width() * 2, self._picture.height() * 2))

    def _blank(self):
        image = QPixmap(self._picture);image.fill(QColor(theme_color('#303030', '#d4d4d4')));return image

    def _read_done(self, path, image, info):
        ident = key(path);self._info[ident] = info
        if image is not None:
            canvas = QPixmap(self._picture.width() * 2, self._picture.height() * 2);canvas.fill(QColor(theme_color('#1c1c1c', '#c4c4c4')))
            p = QPainter(canvas);p.drawImage((canvas.width() - image.width()) // 2, (canvas.height() - image.height()) // 2, image);p.end()
            self._pictures[ident] = QIcon(canvas)
        for row in range(self.list.count()):
            if key(self.list.item(row).data(PATH)) == ident:self._dress(self.list.item(row))
