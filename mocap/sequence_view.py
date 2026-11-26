"""The sequence timeline at its two upper levels: the videos as blocks (level 1) and, zoomed in, the rows of their people
under the blocks (level 2). Painted like TimelineSlider (same margins, rows, colours, grid, zoom and playhead) so the
levels read as one timeline; its time is sequence seconds. It edits nothing itself: every change is a signal.

The left button does what the chosen tool does (TOOLS): Selection selects, moves and trims blocks and links people; Razor
cuts the block under the pointer; Hand drags the view; Zoom zooms in on a click and out with Alt. Videos dragged from
the Project panel (MIME) or from the file manager are dropped between the blocks under the pointer."""
import json
import math
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from .clip_range import format_time
from .sequence import MIN_SECONDS, Sequence
from . import t3_theme
from .theme import grid_color, is_classic_theme, paint_track_bar, theme_color, track_text, ui_px

ZOOM = .22                       # per wheel notch, as the slider
NARROWEST = 12 / 30              # the slider's narrowest view, 12 frames of the 30 fps working clip
FARTHEST = 24 * 3600             # the widest view in seconds, also past the end of a shorter sequence
ZOOM_CLICK = 2                   # wheel notches of one click of the Zoom tool
TOOLS = ('select', 'razor', 'hand', 'zoom')
KEYS = {Qt.Key_V: 'select', Qt.Key_C: 'razor', Qt.Key_H: 'hand', Qt.Key_Z: 'zoom'}
CURSORS = {'select': Qt.ArrowCursor, 'razor': Qt.SplitHCursor, 'hand': Qt.OpenHandCursor, 'zoom': Qt.CrossCursor}
MIME = 'application/x-rgbmocap-videos'   # a JSON list of video paths dragged from the Project panel
VIDEO_SUFFIXES = ('.mp4', '.mov', '.avi', '.mkv', '.webm')   # as the Add Video dialog lists them
LEAST = MIN_SECONDS * (1 + 1e-9)  # a dragged edge keeps a range the document accepts despite rounding
STEPS = (.1, .2, .5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 18000, 36000)
HINT = 'Drag a video here from the Project panel, or drop a video file.'


def dragged_videos(mime):
    """The video paths a drag carries: the Project panel's own list, else the video files among dropped files."""
    if mime.hasFormat(MIME):
        try:paths = json.loads(bytes(mime.data(MIME)).decode('utf-8'))
        except ValueError:return []
        return [path for path in paths if isinstance(path, str) and path] if isinstance(paths, list) else []
    return [url.toLocalFile() for url in mime.urls() if url.isLocalFile() and url.toLocalFile().lower().endswith(VIDEO_SUFFIXES)]
VIDEO = '#6289ad'
GENERATED = ('#d8a24a', '#8a4809')   # generated frames, as TimelineSlider paints its fill_runs
METHOD_LABELS = {'smooth': 'Smooth join', 'auto': 'Automatic (Experimental)'}   # the Hidden ranges panel's names
SUFFIX = {'joined': 'joined', 'out of date': 'join out of date'}
PIECE_TIP = ("Generated transition between the end of {}'s last take and the start of {}'s take, wherever they sit in "
             "their blocks. Click to change it.")


def piece_text(seconds, method):
    return f"{seconds:.1f} s {METHOD_LABELS[method].replace(' (Experimental)', '')}"


def piece_tip(seconds, method, earlier, later):
    return f'{seconds:.1f} s {METHOD_LABELS[method]}\n' + PIECE_TIP.format(earlier, later)


class SequenceView(QWidget):
    selected = Signal(str)
    opened = Signal(str, object)    # block id, person id or None
    scrubbed = Signal(float, bool)  # sequence seconds, mouse released
    moved = Signal(str, int)
    resized = Signal(str, float, float)  # source seconds, on release
    removed = Signal(str)
    linked = Signal(object, object)            # source, target: (block id, person id) each
    unlinked = Signal(object)                  # target
    transition_clicked = Signal(object)        # target of the link whose piece was clicked
    menu_requested = Signal(str, object, object)  # block id, person id, global QPoint
    cut_requested = Signal(str, float)         # block id, source seconds: the Razor's click
    dropped = Signal(object, int)              # video paths, the index their blocks are inserted at
    tool_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.sequence, self.states = Sequence(), {}
        self.rows, self.row_states = None, {}
        self.current = self.busy = None
        self.playhead = 0.
        self.view_start, self.view_span = 0., None
        self._editable = True
        self._press = None       # the left button's gesture: mode, block, pressed x, how far it went
        self._pan_origin = None
        self.tool = TOOLS[0]
        self.razor_at = None     # x where a click of the Razor would cut, while it is over a block
        self.incoming = None     # the index a dragged video would be inserted at, while it is over the timeline
        self.setAcceptDrops(True)
        self.refresh_appearance()

    def refresh_appearance(self):
        self.LEFT, self.RIGHT = ui_px(64), ui_px(14)
        self.TRACK_TOP, self.TRACK_HEIGHT = ui_px(28), ui_px(26)
        self.RULER_HEIGHT = ui_px(25)
        self.setMinimumHeight(ui_px(190))
        self.update()

    @property
    def editable(self):
        return self._editable

    @editable.setter
    def editable(self, value):
        self._editable = bool(value)
        if not self._editable and self._press and self._press['mode'] in ('start', 'end', 'body', 'person', 'piece'):
            self._press = None;self.setCursor(CURSORS[self.tool])
        if not self._editable:self.razor_at = self.incoming = None
        self.update()

    def set_tool(self, tool):
        """Choose what the left button does; an unfinished gesture of the tool before ends."""
        if tool not in TOOLS:raise ValueError(f'Unknown tool: {tool}')
        if tool == self.tool:return
        self.tool, self._press, self._pan_origin, self.razor_at = tool, None, None, None
        self.setCursor(CURSORS[tool])
        self.tool_changed.emit(tool)
        self.update()

    # What is shown

    def show_sequence(self, sequence, states, rows=None, row_states=None):
        """`rows`: sequence_links.rows, the row of every (block id, person id); None puts each person at its index.
        `row_states`: link target -> 'linked' | 'joined' | 'out of date'. A gesture in progress ends."""
        if self._press is not None:self.setCursor(CURSORS[self.tool])
        self.sequence, self.states, self._press = sequence, dict(states), None
        self.rows, self.row_states = (None if rows is None else dict(rows)), dict(row_states or {})
        self.razor_at = None
        if self.view_span is not None and self._fits(self.view_span):
            self.view_span = None   # a view wider than the sequence stays: it was zoomed out on purpose
        self._clamp_view()
        self.update()

    def set_current(self, block_id):
        self.current = block_id;self.update()

    def set_busy(self, block_id):
        self.busy = block_id;self.update()

    def set_playhead(self, seconds):
        self.playhead = float(seconds)
        if self.view_span is not None and not self.view_start <= self.playhead <= self.view_start + self.span():
            self.view_start = self.playhead - self.span() * .2
            self._clamp_view()
        self.update()

    def frame_all(self):
        self.view_start, self.view_span = 0., None
        self.update()

    def show_block(self, block_id):
        """Zoom so the block fills most of the view, centred where the sequence allows; always level 2."""
        i = self.sequence.index(block_id);block = self.sequence.blocks[i]
        at, duration = self.sequence.starts()[i], block.end - block.start
        self.view_span = min(duration / .8, .8 * self.total())
        self.view_start = at + duration / 2 - self.view_span / 2
        self._clamp_view()
        self.update()

    def level(self):
        return 1 if self.span() >= .999 * self.total() else 2

    def block_rect(self, block_id):
        for block, _, rect in self._layout():
            if block.id == block_id:
                return rect.intersected(self._track()).toAlignedRect()
        return QRect()

    def person_rect(self, block_id, person_id):
        """The person's row inside the block, as drawn; empty at level 1."""
        if self.level() != 1:
            for block, _, rect in self._layout():
                for i, person in enumerate(self._people(block)):
                    if block.id == block_id and person.id == person_id:
                        return self._cell(rect, self._row(block.id, i, person)).intersected(self._track()).toAlignedRect()
        return QRect()

    def transition_rect(self, target):
        """The transition piece before the linked person `target` (block id, person id), as drawn; empty at level 1."""
        for link, rect in self._pieces():
            if link.target == tuple(target):
                return rect.intersected(self._track()).toAlignedRect()
        return QRect()

    # Geometry

    def total(self):
        return self.sequence.total()

    def span(self):
        return self.view_span if self.view_span is not None else max(self.total(), MIN_SECONDS)

    def _fits(self, span):
        """`span` shows the whole sequence and no more."""
        return abs(span - self.total()) <= .001 * self.total()

    def farthest(self):
        return max(self.total(), FARTHEST)

    def zoom_at(self, x, steps):
        """Zoom by wheel notches around x. Zooming in on a block that already fills the view opens it instead; the whole
        sequence is a stop on the way in and out, and the view widens past it up to `farthest`."""
        old, total = self.span(), self.total()
        if not steps or total <= 0:return
        x = max(self.LEFT, min(self.width() - self.RIGHT, x))
        fraction = (x - self.LEFT) / self._width()
        anchor = self.view_start + fraction * old
        found = self._block_at(x)
        narrowest = min(NARROWEST, total)
        if steps > 0 and found and (old <= .5 * (found[0].end - found[0].start) or old <= narrowest + 1e-9):
            self.opened.emit(found[0].id, None)
            return
        span = old * math.exp(-steps * ZOOM)
        if (old - total) * (span - total) < 0 and not self._fits(old):span = total
        span = max(narrowest, min(self.farthest(), span))
        if self._fits(span):
            self.view_start, self.view_span = 0., None
        else:
            self.view_span, self.view_start = span, anchor - fraction * span
        self._clamp_view()
        self.update()

    def ruler_step(self):
        """Seconds between two labelled ticks of the ruler: from tenths of a second up to hours."""
        target = self.span() / max(1, (self.width() - self.LEFT) / ui_px(85))
        return next((s for s in STEPS if s >= target), math.ceil(target / 3600) * 3600)

    def ruler_text(self, seconds):
        """A ruler label: with tenths while the steps are shorter than a second, else as a clock (30:00, 1:00:00)."""
        if self.ruler_step() < 1:return format_time(seconds)
        seconds = int(round(seconds));minutes, seconds = divmod(seconds, 60);hours, minutes = divmod(minutes, 60)
        return f'{hours}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes}:{seconds:02d}'

    def _insert_x(self, index):
        """Where the mark of a video dragged to `index` is drawn: the left edge of the block there, else the end."""
        rects = [rect for _, _, rect in self._layout()]
        if not rects:return self.LEFT
        return rects[index].left() if index < len(rects) else rects[-1].right()

    def _width(self):
        return max(1, self.width() - self.LEFT - self.RIGHT)

    def x_of(self, seconds):
        return self.LEFT + (seconds - self.view_start) / self.span() * self._width()

    def time_at(self, x):
        return self.view_start + (x - self.LEFT) / self._width() * self.span()

    def _clamp_view(self):
        self.view_start = max(0., min(max(0., self.total() - self.span()), self.view_start))

    def _track(self):
        return QRectF(self.LEFT, self.RULER_HEIGHT, self._width(), max(0, self.height() - self.RULER_HEIGHT))

    def _layout(self):
        """(block, sequence start, rect in the video row) per block; a block is never narrower than ui_px(6) and a widened
        one stays within the sequence's ends."""
        out, least, first, last = [], ui_px(6), self.x_of(0), self.x_of(self.total())
        for block, at in zip(self.sequence.blocks, self.sequence.starts()):
            left, right = self.x_of(at), self.x_of(at + block.end - block.start)
            if right - left < least:
                left = max(first, min(last - least, (left + right - least) / 2));right = left + least
            out.append((block, at, QRectF(left, self.TRACK_TOP + ui_px(2), right - left, self.TRACK_HEIGHT - ui_px(5))))
        return out

    def _cell(self, rect, index):
        return QRectF(rect.left(), self.TRACK_TOP + (index + 1) * self.TRACK_HEIGHT + ui_px(2), rect.width(), self.TRACK_HEIGHT - ui_px(5))

    def _people(self, block):
        state = self.states.get(block.id)
        return state.people if state else []

    def _row(self, block_id, index, person):
        return index if self.rows is None else self.rows.get((block_id, person.id), index)

    def _row_count(self):
        return max((self._row(block.id, i, person) + 1 for block in self.sequence.blocks
                    for i, person in enumerate(self._people(block))), default=0)

    def _person_in_row(self, block, row):
        return next((person for i, person in enumerate(self._people(block)) if self._row(block.id, i, person) == row), None)

    def _pieces(self, layout=None):
        """(link, rect) of each transition piece: the gap between the two blocks, in the row of the link's later person."""
        if self.level() == 1:
            return []
        rects, out = {block.id: rect for block, _, rect in layout or self._layout()}, []
        for link in self.sequence.model_links():
            earlier, later = rects.get(link.source[0]), rects.get(link.target[0])
            people = self._people(self.sequence.block(link.target[0])) if later is not None else []
            index = next((i for i, person in enumerate(people) if person.id == link.target[1]), None)
            if earlier is None or index is None or later.left() <= earlier.right():continue
            cell = self._cell(later, self._row(link.target[0], index, people[index]))
            out.append((link, QRectF(earlier.right(), cell.top(), later.left() - earlier.right(), cell.height())))
        return out

    def _piece_at(self, point):
        return next((link for link, rect in self._pieces() if rect.contains(point)), None)

    def _name(self, ident):
        return next((person.name for person in self._people(self.sequence.block(ident[0])) if person.id == ident[1]), f'Person {ident[1]}')

    def _row_state(self, ident):
        """'linked', 'joined' or 'out of date' of the linked row holding `ident`; None when it has no link."""
        chain = self.sequence.chain(*ident)
        if len(chain) < 2:
            return None
        return next((self.row_states[t] for t in chain[1:] if t in self.row_states), 'linked')

    def _drop(self, press, point):
        """What dropping the dragged person at `point` does: ('link', source) on a row of a person of the block before,
        ('unlink', None) on a drawn row of its own block that holds nobody of the block before, else None (outside the
        widget, in the ruler or the Video row, below the last row)."""
        x, y = point.x(), point.y()
        if not (0 <= x <= self.width() and self.TRACK_TOP + self.TRACK_HEIGHT <= y <= self.height()):
            return None
        row, block, person = int((y - self.TRACK_TOP) // self.TRACK_HEIGHT) - 1, press['block'], press['person']
        if row >= self._row_count():
            return None
        i = self.sequence.index(block.id)
        before = self._person_in_row(self.sequence.blocks[i - 1], row) if i else None
        link = self.sequence.link_into(block.id, person.id)
        if before is not None:
            source = (self.sequence.blocks[i - 1].id, before.id)
            return None if link is not None and link.source == source else ('link', source)
        own = next((rect for b, _, rect in self._layout() if b.id == block.id), None)
        return ('unlink', None) if link is not None and own is not None and own.left() <= x <= own.right() else None

    def _block_at(self, x):
        """The block whose drawn rectangle holds x; the narrowest wins, so a tiny block beside a long one stays clickable."""
        hits = [(rect.width(), (block, at, rect)) for block, at, rect in self._layout() if rect.left() <= x <= rect.right()]
        return min(hits, key=lambda hit: hit[0], default=(0, None))[1]

    def _hit(self, point):
        """('ruler' | 'block' | 'person' | 'empty', (block, start, rect) or None, person or None); None off the track."""
        x, y = point.x(), point.y()
        if not self.LEFT <= x <= self.width() - self.RIGHT or not 0 <= y <= self.height():
            return None
        if y < self.RULER_HEIGHT:
            return 'ruler', None, None
        found = self._block_at(x)
        row = max(0, int((y - self.TRACK_TOP) // self.TRACK_HEIGHT))
        if found and row == 0:
            return 'block', found, None
        person = self._person_in_row(found[0], row - 1) if found and row > 0 and self.level() == 2 else None
        if person is not None:
            return 'person', found, person
        return 'empty', None, None

    def _edge(self, rect, x):
        zone = min(ui_px(8), rect.width() / 4)
        return 'start' if x - rect.left() <= zone else 'end' if rect.right() - x <= zone else None

    def _edge_range(self, press, dx):
        block, dt = press['block'], dx * self.span() / self._width()
        if press['mode'] == 'start':
            return max(0., min(block.start + dt, block.end - LEAST)), block.end
        return block.start, min(block.length, max(block.end + dt, block.start + LEAST))

    def _drop_index(self, block_id, x):
        return sum(rect.center().x() < x for block, _, rect in self._layout() if block.id != block_id)

    # Mouse and keys

    def _scrub(self, x, final):
        self.playhead = max(0., min(self.total(), self.time_at(x)))
        self.scrubbed.emit(self.playhead, final)
        self.update()

    def _click(self, press):
        block, at = press['block'], press['at']
        self.selected.emit(block.id)
        # A tiny block is drawn wider than it is; the playhead still lands inside it.
        self.playhead = max(at, min(at + block.end - block.start - 1e-6, self.time_at(press['x'])))
        self.scrubbed.emit(self.playhead, True)
        self.update()

    def _tool_press(self, point, modifiers):
        """The left button with Razor, Hand or Zoom; False when the tool leaves the press to the ruler's scrub."""
        hit = self._hit(point)
        if hit and hit[0] == 'ruler' and self.tool == 'razor':return False
        self._press = None
        if self.tool == 'hand':
            self._pan_origin = (point.x(), self.view_start);self.setCursor(Qt.ClosedHandCursor)
        elif self.tool == 'zoom':
            self.zoom_at(point.x(), -ZOOM_CLICK if modifiers & Qt.AltModifier else ZOOM_CLICK)
        elif self.editable and hit and hit[0] == 'block':
            block, at, _ = hit[1]
            self.cut_requested.emit(block.id, max(block.start, min(block.end, block.start + self.time_at(point.x()) - at)))
        return True

    def mousePressEvent(self, event):
        point = event.position()
        if event.button() == Qt.MiddleButton:
            self._pan_origin = (point.x(), self.view_start)
        elif event.button() == Qt.LeftButton and self.tool != 'select' and self._tool_press(point, event.modifiers()):
            self.setFocus(Qt.MouseFocusReason)
        elif event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
            hit = self._hit(point)
            self._press = None
            piece = self._piece_at(point) if self.editable and hit else None
            if piece is not None:
                self._press = dict(mode='piece', target=piece.target, x=point.x(), far=0., dragged=False)
            elif hit and hit[0] in ('ruler', 'empty'):
                self._press = dict(mode='scrub', x=point.x())
                self._scrub(point.x(), False)
            elif hit:
                block, at, rect = hit[1]
                mode = 'click'
                if self.editable and hit[0] == 'block':
                    mode = self._edge(rect, point.x()) or 'body'
                elif self.editable and hit[0] == 'person':
                    mode = 'person'
                self._press = dict(mode=mode, block=block, at=at, person=hit[2], x=point.x(), y=point.y(), far=0., dragged=False)
                if mode in ('start', 'end'):self.setCursor(Qt.SizeHorCursor)
        elif event.button() == Qt.RightButton and self.editable and (hit := self._hit(point)) and hit[0] == 'person':
            self.menu_requested.emit(hit[1][0].id, hit[2].id, event.globalPosition().toPoint())
        else:
            super().mousePressEvent(event)
            return
        event.accept()

    def mouseMoveEvent(self, event):
        point, press = event.position(), self._press
        if self._pan_origin is not None:
            pixel, start = self._pan_origin
            self.view_start = start - (point.x() - pixel) * self.span() / self._width()
            self._clamp_view();self.update()
        elif press is None and self.tool != 'select':
            hit = self._hit(point)
            over = point.x() if self.tool == 'razor' and self.editable and hit and hit[0] == 'block' else None
            if over != self.razor_at:
                self.razor_at = over;self.update()
            self.setCursor(Qt.PointingHandCursor if hit and hit[0] == 'ruler' and self.tool == 'razor' else CURSORS[self.tool])
        elif press is None:
            hit = self._hit(point)
            self.setCursor(Qt.PointingHandCursor if hit and hit[0] == 'ruler'
                           else Qt.SizeHorCursor if hit and hit[0] == 'block' and self.editable and self._edge(hit[1][2], point.x())
                           else Qt.ArrowCursor)
        elif press['mode'] == 'scrub':
            self._scrub(point.x(), False)
        else:
            dx = point.x() - press['x'];press['far'] = max(press['far'], abs(dx))
            if press['mode'] in ('start', 'end'):
                press['range'] = self._edge_range(press, dx)
                # A press at an edge that moves less than ui_px(3) stays a click.
                press['dragged'] |= press['far'] >= ui_px(3) and press['range'] != (press['block'].start, press['block'].end)
            elif press['mode'] == 'body' and press['far'] > ui_px(8):
                press.update(dragged=True, dx=dx, to=self._drop_index(press['block'].id, point.x()))
                self.setCursor(Qt.ClosedHandCursor)
            elif press['mode'] == 'person' and (press['dragged'] or abs(point.y() - press['y']) > ui_px(8)):
                # A vertical drag of a person's bar links it; it never moves the block.
                press.update(dragged=True, dx=dx, dy=point.y() - press['y'], drop=self._drop(press, point))
                self.setCursor(Qt.ClosedHandCursor)
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_origin = None
        elif event.button() == Qt.LeftButton and self._press is None and self.tool != 'select':
            self._pan_origin = None;self.setCursor(CURSORS[self.tool])
        elif event.button() == Qt.LeftButton and self._press is not None:
            press, self._press = self._press, None
            block = press.get('block')
            if press['mode'] == 'scrub':
                self._scrub(event.position().x(), True)
            elif press['mode'] in ('start', 'end') and press['dragged']:
                start, end = self._edge_range(press, event.position().x() - press['x'])
                if (start, end) != (block.start, block.end):self.resized.emit(block.id, start, end)
            elif press['mode'] == 'body' and press['dragged']:
                to = self._drop_index(block.id, event.position().x())
                if to != self.sequence.index(block.id):self.moved.emit(block.id, to)
            elif press['mode'] == 'person' and press['dragged']:
                drop = self._drop(press, event.position())
                if drop and drop[0] == 'link':self.linked.emit(drop[1], (block.id, press['person'].id))
                elif drop:self.unlinked.emit((block.id, press['person'].id))
            elif press['mode'] == 'piece':
                if press['far'] <= ui_px(8):self.transition_clicked.emit(press['target'])
            elif press['far'] <= ui_px(8):
                self._click(press)
            self.setCursor(CURSORS[self.tool])
            self.update()
        else:
            super().mouseReleaseEvent(event)
            return
        event.accept()

    def leaveEvent(self, event):
        if self.razor_at is not None:
            self.razor_at = None;self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.tool != 'select':   # the second click of a tool is one more click
            self.mousePressEvent(event);return
        hit = self._hit(event.position()) if event.button() == Qt.LeftButton else None
        if hit and hit[0] in ('block', 'person'):
            self._press = None
            self.opened.emit(hit[1][0].id, hit[2].id if hit[2] else None)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if event.modifiers() & Qt.ShiftModifier:
            self.view_start -= steps * self.span() * .12
            self._clamp_view()
            self.update()
        else:
            self.zoom_at(event.position().x(), steps)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and self.editable and any(b.id == self.current for b in self.sequence.blocks):
            self.removed.emit(self.current)
            event.accept()
        elif event.key() in KEYS and not event.modifiers() & ~Qt.KeypadModifier:
            self.set_tool(KEYS[event.key()])
            event.accept()
        else:
            super().keyPressEvent(event)

    # Videos dragged from the Project panel or the file manager

    def _drag(self, event):
        if self.editable and dragged_videos(event.mimeData()):
            index = self._drop_index(None, event.position().x())
            if index != self.incoming:
                self.incoming = index;self.update()
            event.setDropAction(Qt.CopyAction);event.accept()
        else:
            self._no_drag();event.ignore()

    def _no_drag(self):
        if self.incoming is not None:
            self.incoming = None;self.update()

    def dragEnterEvent(self, event):self._drag(event)

    def dragMoveEvent(self, event):self._drag(event)

    def dragLeaveEvent(self, event):
        self._no_drag();event.accept()

    def dropEvent(self, event):
        videos = dragged_videos(event.mimeData()) if self.editable else []
        self._no_drag()
        if not videos:
            event.ignore();return
        event.setDropAction(Qt.CopyAction);event.accept()
        self.dropped.emit(videos, self._drop_index(None, event.position().x()))

    def event(self, event):
        if event.type() == QEvent.ToolTip:
            hit = self._hit(QPointF(event.pos()))
            piece = self._piece_at(QPointF(event.pos())) if hit else None
            text = ''
            if piece is not None:
                text = piece_tip(self.sequence.gap(piece.target[0]), piece.method, self._name(piece.source), self._name(piece.target))
            elif hit and hit[0] == 'block':
                state = self.states.get(hit[1][0].id)
                text = '\n'.join(filter(None, (hit[1][0].video, state.reason if state else '')))
            elif hit and hit[0] == 'person':
                text = hit[2].name
            if text:QToolTip.showText(event.globalPos(), text, self)
            else:QToolTip.hideText()
            return True
        return super().event(event)

    # Painting

    def paintEvent(self, _):
        p = QPainter(self)
        p.setFont(self.font())
        p.fillRect(self.rect(), QColor(theme_color('#242424')))
        p.fillRect(0, 0, self.width(), self.RULER_HEIGHT, QColor(theme_color('#303030')))
        p.setPen(QColor(theme_color('#999999')))
        p.drawText(ui_px(9), ui_px(17), 'Time')
        if not self.sequence.blocks:
            p.drawText(QRectF(0, self.RULER_HEIGHT, self.width(), self.height() - self.RULER_HEIGHT), Qt.AlignCenter, HINT)
            self._paint_marks(p)
            return
        layout = sorted(self._layout(), key=lambda item: -item[2].width())  # a tiny block is drawn over a long one
        people_rows = self._row_count() if self.level() == 2 else 0
        pieces = self._pieces(layout) if people_rows else []
        chains = {self.sequence.chain(*link.target)[0]: self._row_state(link.target) for link in self.sequence.model_links()}
        track = self._track()
        p.save()
        p.setClipRect(track)
        bars, classic = {}, is_classic_theme()
        if not classic:self._paint_grid(p)   # under the solid bars; Classic draws it over its tinted tracks
        for block, at, rect in layout:
            self._paint_block(p, block, rect)
            for i, person in enumerate(self._people(block) if people_rows else ()):
                bars[(block.id, person.id)] = self._paint_person(p, block, at, rect, self._row(block.id, i, person), person,
                                                                 SUFFIX.get(chains.get((block.id, person.id))))
        for link, rect in pieces:
            self._paint_piece(p, link, rect)
        if classic:self._paint_grid(p)
        for block, at, rect in layout:
            self._paint_label(p, block, rect)
        for first, row_state in chains.items():
            if people_rows and row_state in SUFFIX:self._paint_row(p, first, row_state, bars, pieces)
        self._paint_drag(p, layout)
        self._paint_marks(p)
        p.restore()
        p.setPen(QColor(theme_color('#aaaaaa')))
        rows = ('Video',) + (('People',) if people_rows else ())
        for row, label in enumerate(rows):
            p.drawText(QRectF(ui_px(8), self.TRACK_TOP + row * self.TRACK_HEIGHT, ui_px(53), self.TRACK_HEIGHT), Qt.AlignVCenter, label)
        self._paint_playhead(p)

    def _shown(self, rect):
        """A block as drawn: a pixel of gap to its neighbours unless that would make it narrower than ui_px(6)."""
        return rect.adjusted(1, 0, -1, 0) if rect.width() > ui_px(8) else rect

    def _paint_block(self, p, block, rect):
        state = self.states.get(block.id)
        kind, shown = state.kind if state else 'untracked', self._shown(rect)
        color = QColor(theme_color(VIDEO))
        if not is_classic_theme():
            return self._paint_block_round(p, block, shown, kind)
        if kind in ('untracked', 'missing'):
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(theme_color('#ffb45c')) if kind == 'missing' else color, 1, Qt.DashLine))
        else:
            tint = QColor(color);tint.setAlpha(80 if kind == 'tracked' else 45)
            edge = QColor(color);edge.setAlpha(150)
            p.setBrush(tint);p.setPen(QPen(edge, 1))
        p.drawRect(shown.adjusted(.5, .5, -.5, -.5))
        if block.id == self.busy:
            p.fillRect(QRectF(shown.left(), shown.bottom() - ui_px(3), shown.width(), ui_px(3)), QColor(theme_color('#d8a24a', '#8a4809')))
        if block.id == self.current:
            p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor(theme_color('#d9e8fa')), 1.5))
            p.drawRect(shown.adjusted(.75, .75, -.75, -.75))

    def _paint_block_round(self, p, block, shown, kind):
        """The block as Dark and Light draw it: a solid rounded bar once tracked, a fainter one while partly tracked, a
        dashed outline before that, a white line around the chosen block."""
        radius = ui_px(5)
        if kind in ('untracked', 'missing'):
            paint_track_bar(p, shown, VIDEO, 12)
            t3_theme.outline(p, shown, theme_color('#ffb45c') if kind == 'missing' else t3_theme.track(VIDEO), 1, radius, dashed=True)
        else:
            paint_track_bar(p, shown, VIDEO, 80 if kind == 'tracked' else 45)
        if block.id == self.busy:
            p.fillRect(QRectF(shown.left() + radius, shown.bottom() - ui_px(3), max(0., shown.width() - 2 * radius), ui_px(3)),
                       QColor(theme_color('#d8a24a', '#8a4809')))
        if block.id == self.current:
            t3_theme.outline(p, shown, theme_color('#ffffff', '#202020'), 2, radius)

    def _block_alpha(self, block):
        state = self.states.get(block.id)
        kind = state.kind if state else 'untracked'
        return 12 if kind in ('untracked', 'missing') else 80 if kind == 'tracked' else 45

    def _bars(self, block, at, cell, person):
        """(rect, take) of each of the person's takes inside its cell."""
        fps, out = self.states[block.id].fps, []
        for take in person.takes:
            left = max(cell.left(), self.x_of(at + take.first / fps))
            right = min(cell.right(), self.x_of(at + take.end / fps))
            if right > left:out.append((QRectF(left, cell.top(), max(1., right - left), cell.height()), take))
        return out

    def _paint_person(self, p, block, at, rect, row, person, suffix=None):
        """Paint the person's bars in its row and return their rects; `suffix` follows the name (a joined row's state)."""
        cell = self._cell(rect, row)
        color = QColor(theme_color(person.color))
        bars = self._bars(block, at, cell, person)
        for bar, take in bars:
            paint_track_bar(p, bar, person.color, 80 if take.ready else 25)
            if not take.ready:
                if is_classic_theme():
                    p.setBrush(Qt.NoBrush);p.setPen(QPen(color, 1, Qt.DashLine));p.drawRect(bar.adjusted(.5, .5, -.5, -.5))
                else:
                    t3_theme.outline(p, bar, t3_theme.track(person.color), 1, ui_px(5), dashed=True)
        if bars:
            left = max(self.LEFT, bars[0][0].left()) + ui_px(6)
            width = min(self.width() - self.RIGHT, cell.right()) - left - ui_px(6)
            if width > ui_px(10):
                p.setPen(track_text(person.color, '#aebeb4', 80 if bars[0][1].ready else 25))
                p.drawText(QRectF(left, cell.top(), width, cell.height()), Qt.AlignVCenter,
                           self.fontMetrics().elidedText(' | '.join(filter(None, (person.name, suffix))), Qt.ElideRight, round(width)))
        return [bar for bar, _ in bars]

    def _paint_piece(self, p, link, rect):
        """The generated transition: hatched in the colour of generated frames, solid once the row is joined."""
        color, joined = QColor(theme_color(*GENERATED)), self._row_state(link.target) == 'joined'
        if is_classic_theme():
            tint = QColor(color);tint.setAlpha(120 if joined else 35)
            p.fillRect(rect, tint)
        else:
            paint_track_bar(p, rect, GENERATED[0], 80 if joined else 20)
        if not joined:
            hatch = QColor(color);hatch.setAlpha(170)
            p.fillRect(rect.adjusted(2, 0, -2, 0) if not is_classic_theme() and rect.width() > 8 else rect, QBrush(hatch, Qt.BDiagPattern))
        width, seconds, metrics = rect.width() - 2 * ui_px(4), self.sequence.gap(link.target[0]), self.fontMetrics()
        if width > ui_px(10):
            # A narrow piece keeps its length rather than half a method name; the tooltip has the rest.
            text = piece_text(seconds, link.method)
            if metrics.horizontalAdvance(text) > width:text = f'{seconds:.1f} s'
            p.setPen(track_text(GENERATED[0], '#e6e6e6', 80 if joined else 20) if not is_classic_theme() else QColor(theme_color('#e6e6e6', '#202020')))
            p.drawText(rect.adjusted(ui_px(4), 0, -ui_px(4), 0), Qt.AlignCenter, metrics.elidedText(text, Qt.ElideRight, round(width)))

    def _paint_row(self, p, first, row_state, bars, pieces):
        """One outline around a joined row from its first bar to its last; dashed while the join is out of date."""
        chain = self.sequence.chain(*first)
        shapes = [bar for ident in chain for bar in bars.get(ident, ())] + [rect for link, rect in pieces if link.target in chain]
        if not shapes:return
        box = shapes[0]
        for shape in shapes[1:]:box = box.united(shape)
        person = next((q for q in self._people(self.sequence.block(first[0])) if q.id == first[1]), None)
        color = QColor(theme_color(person.color)) if person else QColor(theme_color('#d9e8fa'))
        if not is_classic_theme():
            return t3_theme.outline(p, box.adjusted(-2, -2, 2, 2), theme_color('#ffffff', '#202020'), 1.5, ui_px(6), dashed=row_state != 'joined')
        p.setBrush(Qt.NoBrush);p.setPen(QPen(color, 1.5, Qt.SolidLine if row_state == 'joined' else Qt.DashLine))
        p.drawRect(box.adjusted(-1, -1, 1, 1))

    def _paint_label(self, p, block, rect):
        shown = self._shown(rect).intersected(self._track())  # the text starts where the block comes into view
        width = shown.width() - 2 * ui_px(6)
        if width <= ui_px(10):return
        state = self.states.get(block.id)
        parts = (('working',) if block.id == self.busy else ()) + (Path(block.video).name, f'{format_time(block.start)}-{format_time(block.end)}') \
            + ((state.label,) if state and state.label else ())
        p.setPen(track_text(VIDEO, '#b1b9c1', self._block_alpha(block)))
        p.drawText(QRectF(shown.left() + ui_px(6), shown.top(), width, shown.height()), Qt.AlignVCenter,
                   self.fontMetrics().elidedText(' | '.join(parts), Qt.ElideRight, round(width)))

    def _paint_marks(self, p):
        """Where a dragged video would be inserted, and where the Razor would cut."""
        if self.incoming is not None:
            x = self._insert_x(self.incoming)
            p.setPen(QPen(QColor(theme_color('#74a6e0')), 2))
            p.drawLine(QPointF(x, self.TRACK_TOP), QPointF(x, self.TRACK_TOP + self.TRACK_HEIGHT))
        if self.razor_at is not None:
            p.setPen(QPen(QColor(theme_color('#d9e8fa')), 1, Qt.DashLine))
            p.drawLine(QPointF(self.razor_at, self.TRACK_TOP), QPointF(self.razor_at, self.TRACK_TOP + self.TRACK_HEIGHT))

    def _paint_grid(self, p):
        """Ticks and grid as the slider's, in sequence time; the ruler labels are drawn by the same steps."""
        span, step = self.span(), self.ruler_step()
        end = self.view_start + span
        for size, color in ((step / 5, grid_color(False)), (step, grid_color(True))):
            p.setPen(QPen(color, 1))
            for n in range(math.ceil(self.view_start / size - 1e-9), math.floor(end / size + 1e-9) + 1):
                x = self.x_of(n * size)
                p.drawLine(QPointF(x, self.RULER_HEIGHT), QPointF(x, self.height()))
        p.save()
        p.setClipping(False)
        p.setPen(QColor(theme_color('#b3b3b3')))
        for n in range(math.ceil(self.view_start / step - 1e-9), math.floor(end / step + 1e-9) + 1):
            x = self.x_of(n * step)
            p.drawText(QRectF(x - ui_px(35), ui_px(2), ui_px(70), ui_px(20)), Qt.AlignCenter, self.ruler_text(n * step))
        p.restore()

    def _paint_drag(self, p, layout):
        press = self._press
        if not press or not press.get('dragged'):return   # a scrub has no drag
        block, at = press['block'], press['at']
        if press['mode'] in ('start', 'end'):
            start, end = press['range']
            left, right = self.x_of(at + start - block.start), self.x_of(at + end - block.start)
            rect = QRectF(left, self.TRACK_TOP + 2, max(1., right - left), self.TRACK_HEIGHT - 5)
            p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor(theme_color('#d9e8fa')), 1, Qt.DashLine));p.drawRect(rect)
            edge = left if press['mode'] == 'start' else right
            p.fillRect(QRectF(edge if press['mode'] == 'start' else edge - ui_px(6), rect.top(), ui_px(6), rect.height()), QColor(139, 177, 211, 150))
            p.setPen(QPen(QColor(theme_color('#a6bed5')), 1));p.drawLine(QPointF(edge, rect.top()), QPointF(edge, rect.bottom()))
        elif press['mode'] == 'body':
            rect = next(r for b, _, r in layout if b.id == block.id)
            p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor(theme_color('#d9e8fa')), 1, Qt.DashLine))
            p.drawRect(rect.translated(press['dx'], 0))
            others = sorted((r.left(), r.right()) for b, _, r in layout if b.id != block.id)
            x = rect.left() if not others else others[0][0] if press['to'] == 0 else others[press['to'] - 1][1]
            p.setPen(QPen(QColor(theme_color('#74a6e0')), 2))
            p.drawLine(QPointF(x, self.TRACK_TOP), QPointF(x, self.TRACK_TOP + self.TRACK_HEIGHT))
        elif press['mode'] == 'person':
            self._paint_link_drag(p, layout, press)

    def _paint_link_drag(self, p, layout, press):
        """The rows of the block before that would take the dragged person, the one under the cursor outlined, and a ghost
        of the person's bars following the cursor."""
        block, person, mark = press['block'], press['person'], QColor(theme_color('#74a6e0'))
        rects, i = {b.id: r for b, _, r in layout}, self.sequence.index(block.id)
        link, drop = self.sequence.link_into(block.id, person.id), press['drop']
        if i:
            before = self.sequence.blocks[i - 1]
            for j, other in enumerate(self._people(before)):
                if link is None or link.source != (before.id, other.id):
                    fill = QColor(mark);fill.setAlpha(70)
                    p.fillRect(self._cell(rects[before.id], self._row(before.id, j, other)), fill)
        if drop:
            row = int((press['y'] + press['dy'] - self.TRACK_TOP) // self.TRACK_HEIGHT) - 1
            target = self._cell(rects[before.id] if drop[0] == 'link' else rects[block.id], row)
            p.setBrush(Qt.NoBrush);p.setPen(QPen(mark, 2));p.drawRect(target.adjusted(1, 1, -1, -1))
        index = next(j for j, other in enumerate(self._people(block)) if other.id == person.id)
        color = QColor(theme_color(person.color))
        tint = QColor(color);tint.setAlpha(70)
        for bar, _ in self._bars(block, press['at'], self._cell(rects[block.id], self._row(block.id, index, person)), person):
            ghost = bar.translated(press['dx'], press['dy'])
            p.fillRect(ghost, tint);p.setBrush(Qt.NoBrush);p.setPen(QPen(color, 1, Qt.DashLine));p.drawRect(ghost.adjusted(.5, .5, -.5, -.5))

    def _paint_playhead(self, p):
        x = self.x_of(self.playhead)
        if not self.LEFT - 1 <= x <= self.width() - self.RIGHT + 1:return
        p.setPen(QPen(QColor(theme_color('#74a6e0')), 1.4))
        p.drawLine(QPointF(x, ui_px(23)), QPointF(x, self.height()))
        p.setRenderHint(QPainter.Antialiasing)
        text = format_time(self.playhead)
        width = max(ui_px(25), self.fontMetrics().horizontalAdvance(text) + ui_px(12))
        head = QRectF(x - width / 2, ui_px(1), width, ui_px(21))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme_color('#477bb7')))
        p.drawRoundedRect(head, ui_px(3), ui_px(3))
        p.drawPolygon(QPolygonF([QPointF(x - ui_px(4), ui_px(21)), QPointF(x + ui_px(4), ui_px(21)), QPointF(x, ui_px(26))]))
        p.setPen(QColor('#ffffff'))
        p.drawText(head, Qt.AlignCenter, text)
