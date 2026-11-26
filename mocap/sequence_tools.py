"""The strip of timeline tools beside the sequence timeline: one checkable button per tool of sequence_view.TOOLS.
It only shows and picks the tool; the timeline does what the tool does."""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QButtonGroup, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from .sequence_view import TOOLS
from .theme import theme_color, ui_px
from .tooltips import tooltip_text

NAMES = {'select': 'Selection tool', 'razor': 'Razor tool', 'hand': 'Hand tool', 'zoom': 'Zoom tool'}
TIPS = {'select': 'Selection (V): click a block to select it, drag it to another place, drag its edge to trim it.',
        'razor': 'Razor (C): click a block to cut it there.',
        'hand': 'Hand (H): drag the timeline sideways.',
        'zoom': 'Zoom (Z): click to zoom in, Alt+click to zoom out.'}


def picture(tool, size, color):
    """The tool's icon, drawn in a 16-unit box: an arrow, scissors, a hand, a magnifying glass."""
    image = QPixmap(size, size);image.fill(Qt.transparent)
    p = QPainter(image);p.setRenderHint(QPainter.Antialiasing);p.scale(size / 16, size / 16)
    pen = QPen(QColor(color), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin);p.setPen(pen);p.setBrush(Qt.NoBrush)
    if tool == 'select':
        p.setBrush(QColor(color))
        p.drawPolygon(QPolygonF([QPointF(*xy) for xy in ((4, 2), (4, 13), (7, 10.5), (9, 14.5), (10.8, 13.6), (8.8, 9.7), (12.5, 9.7))]))
    elif tool == 'razor':
        for y in (4.2, 11.8):p.drawEllipse(QPointF(4.2, y), 2.1, 2.1)   # the two handles
        p.drawLine(QPointF(13.5, 2.6), QPointF(5.7, 10.4));p.drawLine(QPointF(5.7, 5.6), QPointF(8.2, 8.1))   # the blades,
        p.drawLine(QPointF(9.8, 9.7), QPointF(13.5, 13.4))                                                    # crossing at the middle
    elif tool == 'hand':
        path = QPainterPath()
        for x, top in ((5, 4.5), (7, 3), (9, 3.5), (11, 5)):   # four fingers
            path.moveTo(x, 9);path.lineTo(x, top)
        path.moveTo(5, 9);path.lineTo(3, 7.5);path.moveTo(3, 7.5);path.cubicTo(3, 12, 5, 14, 7.5, 14);path.cubicTo(10, 14, 11, 12, 11, 9)
        p.drawPath(path)
    else:
        p.drawEllipse(QRectF(2.5, 2.5, 8, 8));p.drawLine(QPointF(9.5, 9.5), QPointF(13.5, 13.5))
    p.end()
    return image


class ToolStrip(QWidget):
    picked = Signal(str)   # the user clicked a tool's button

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('editorHeader');self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        self.group = QButtonGroup(self);self.group.setExclusive(True)
        self.buttons = {}
        for tool in TOOLS:
            button = QToolButton();button.setCheckable(True);button.setAutoRaise(True);button.setFocusPolicy(Qt.NoFocus)
            button.setAccessibleName(NAMES[tool]);button.setToolTip(tooltip_text(TIPS[tool]))
            button.clicked.connect(lambda _=False, tool=tool:self._clicked(tool))
            self.group.addButton(button);layout.addWidget(button);self.buttons[tool] = button
        layout.addStretch(1)
        self.tool = TOOLS[0];self.buttons[self.tool].setChecked(True)
        self.refresh_appearance()

    def _clicked(self, tool):
        if tool != self.tool:
            self.tool = tool;self.picked.emit(tool)

    def refresh_appearance(self):
        """Sizes and icon colours of the current theme and UI scale."""
        side, gap = ui_px(18), ui_px(4)
        self.layout().setContentsMargins(gap, gap, gap, gap);self.layout().setSpacing(gap)
        for tool, button in self.buttons.items():
            icon = QIcon()
            icon.addPixmap(picture(tool, side * 2, theme_color('#c8c8c8', '#3a3a3a')), QIcon.Normal)
            icon.addPixmap(picture(tool, side * 2, theme_color('#6e6e6e', '#a8a8a8')), QIcon.Disabled)
            button.setIcon(icon);button.setIconSize(QSize(side, side));button.setFixedSize(side + ui_px(8), side + ui_px(8))

    def set_tool(self, tool):
        """Show `tool` as chosen (the timeline changed it, by a key)."""
        self.tool = tool;self.buttons[tool].setChecked(True)
