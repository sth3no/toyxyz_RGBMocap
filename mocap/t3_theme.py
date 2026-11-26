"""The Dark and Light looks: near-black or near-white flat surfaces, thin light lines, rounded controls, Inter.

The colours, radii and font follow the stock palette of the T3 Code desktop app (its `themePalettes.ts`, read
2026-11-26). The older looks stay in `theme.py` as Classic Dark and Classic Light, and every colour the rest of the
program names is still the Classic Dark one: `color()` turns it into this look's colour, so painted widgets and
widget style sheets need no second set of colours.
"""
import re
import sys

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QIcon, QLinearGradient, QPainter,
                           QPainterPath, QPalette, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QProxyStyle, QPushButton, QStyle, QWidget

from .paths import ROOT

FAMILY = 'Inter'
FONT_FILES = ('Inter-Regular.ttf', 'Inter-Medium.ttf', 'Inter-SemiBold.ttf', 'Inter-Bold.ttf')

# Written in the dark colours; LIGHT turns each one into its light counterpart.
STYLE = """
QWidget { color:#f5f5f5; font-size:12px; }
QMainWindow { background:#000000; }
QDialog { background:#0a0a0a; }
QLabel#brand { font-size:13px; font-weight:600; color:#f5f5f5; padding:0 8px; }
QLabel#muted { color:#818181; }
QLabel#status { color:#a3a3a3; padding:3px 6px; }
QLabel#taskInfo { color:#51a2ff; padding:3px 6px; }
QFrame#editor { background:#0a0a0a; border:1px solid #1f1f1f; border-radius:10px; padding:2px; }
QFrame#editor QFrame#editor { border:0; border-right:1px solid #1f1f1f; border-radius:0; padding:0; }
QWidget#editorHeader { background:transparent; border:0; }
QWidget#editorHeader QPushButton { background:transparent; border-color:transparent; color:#d4d4d4; }
QWidget#editorHeader QPushButton:hover { background:#1e1e1e; color:#f5f5f5; }
QWidget#editorHeader QPushButton:pressed { background:#161616; }
QWidget#editorHeader QPushButton:checked { background:#1e1e1e; border-color:#262626; color:#f5f5f5; }
QWidget#editorHeader QPushButton:disabled { background:transparent; border-color:transparent; color:#525252; }
QToolButton { background:transparent; border:1px solid transparent; border-radius:6px; padding:3px; }
QToolButton:hover { background:#1e1e1e; }
QToolButton:checked { background:#17213a; border-color:#264a9e; }
QTabWidget#peopleTabs::pane { background:#0a0a0a; border:0; }
QTabWidget#peopleTabs QTabBar, QTabBar#modelManagerTabs { qproperty-drawBase:0; }
QTabWidget#peopleTabs QTabBar::tab, QTabBar#modelManagerTabs::tab { background:#0a0a0a; color:#a3a3a3; border:1px solid #0a0a0a; border-radius:6px; padding:5px 10px; }
QTabWidget#peopleTabs QTabBar::tab:!selected:hover, QTabBar#modelManagerTabs::tab:!selected:hover { background:#161616; color:#f5f5f5; }
QTabWidget#peopleTabs QTabBar::tab:selected, QTabBar#modelManagerTabs::tab:selected { background:#1e1e1e; border-color:#262626; color:#f5f5f5; }
QTabBar#sidebarTabs::tab { background:transparent; color:#a3a3a3; border:1px solid transparent; border-radius:6px; padding:4px 6px; margin:4px 0 2px 0; }
QTabBar#sidebarTabs::tab:!selected:hover { background:#161616; color:#f5f5f5; }
QTabBar#sidebarTabs::tab:selected { background:#1e1e1e; border-color:#262626; color:#f5f5f5; }
QGroupBox { background:#111111; border:1px solid #1f1f1f; border-radius:10px; margin-top:0; padding:24px 3px 3px; font-weight:600; }
QGroupBox::title { subcontrol-origin:padding; subcontrol-position:top left; left:9px; top:8px; padding:0; color:#f5f5f5; }
QGroupBox QLabel { color:#a3a3a3; }
QGroupBox QLabel#muted { color:#818181; }
QPushButton { background:#171717; border:1px solid #262626; border-radius:6px; padding:5px 9px; font-weight:500; }
QPushButton:hover { background:#1e1e1e; border-color:#2f2f2f; }
QPushButton:pressed { background:#111111; }
QPushButton:checked { background:#1e1e1e; border-color:#3a3a3a; }
QPushButton:disabled { color:#525252; background:#0f0f0f; border-color:#1a1a1a; }
QPushButton#primary { background:#346bf1; border-color:#346bf1; color:#ffffff; font-weight:600; }
QPushButton#primary:hover { background:#3061d9; border-color:#3061d9; }
QPushButton#primary:disabled { background:#16233f; border-color:#16233f; color:#6f7c99; }
QPushButton#transport { padding:3px; min-width:25px; }
QPushButton[accessibleName="People Help"],QPushButton[accessibleName="Timeline Help"] { padding:3px 0; color:#a3a3a3; }
QComboBox,QAbstractSpinBox,QLineEdit { background:#141414; border:1px solid #262626; border-radius:6px; padding:3px 6px; min-height:20px; selection-background-color:#264a9e; selection-color:#ffffff; }
QComboBox:hover,QAbstractSpinBox:hover,QLineEdit:hover { border-color:#3a3a3a; }
QAbstractSpinBox:focus,QLineEdit:focus { border-color:#346bf1; }
QComboBox:disabled,QAbstractSpinBox:disabled,QLineEdit:disabled { color:#525252; background:#0f0f0f; border-color:#1a1a1a; }
QComboBox::drop-down { subcontrol-origin:padding; subcontrol-position:center right; width:18px; border:0; }
QComboBox::down-arrow { image:url("CHEVRON_DOWN"); width:10px; height:10px; }
QAbstractSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:15px; border:0; }
QAbstractSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:15px; border:0; }
QAbstractSpinBox::up-arrow { image:url("CHEVRON_UP"); width:8px; height:8px; }
QAbstractSpinBox::down-arrow { image:url("CHEVRON_DOWN"); width:8px; height:8px; }
QComboBox QAbstractItemView { background:#111111; border:1px solid #262626; selection-background-color:#1e1e1e; selection-color:#f5f5f5; outline:0; }
QCheckBox { spacing:6px; }
QCheckBox::indicator { width:14px; height:14px; border:1px solid #3a3a3a; border-radius:4px; background:#141414; }
QCheckBox::indicator:hover { border-color:#5b8af5; background:#17213a; }
QCheckBox::indicator:checked { border-color:#346bf1; background:#346bf1; image:url("CHECK_ICON"); }
QCheckBox::indicator:disabled { border-color:#262626; background:#0f0f0f; }
QCheckBox::indicator:checked:disabled { border-color:#16233f; background:#16233f; }
QListWidget#peopleList::item,QListWidget#streamPeopleList::item { min-height:24px; }
QListWidget#peopleList { border:0; }
QListWidget#peopleList::indicator,QListWidget#streamPeopleList::indicator { width:16px; height:16px; border:1px solid #3a3a3a; border-radius:4px; background:#141414; }
QListWidget#peopleList::indicator:hover,QListWidget#streamPeopleList::indicator:hover { border-color:#5b8af5; background:#17213a; }
QListWidget#peopleList::indicator:checked,QListWidget#streamPeopleList::indicator:checked { border-color:#346bf1; background:#346bf1; image:url("CHECK_ICON"); }
QListWidget#peopleList::indicator:disabled,QListWidget#streamPeopleList::indicator:disabled { border-color:#262626; background:#0f0f0f; }
QListWidget,QTreeWidget { background:#0a0a0a; selection-background-color:#1e1e1e; selection-color:#f5f5f5; outline:0; }
QFrame#editor QListWidget { border:0; }
QProgressBar { background:transparent; border:0; height:3px; }
QProgressBar::chunk { background:#346bf1; }
QTextEdit,QPlainTextEdit,QTableWidget { background:#0a0a0a; border:1px solid #1f1f1f; border-radius:6px; selection-background-color:#264a9e; selection-color:#ffffff; }
QTableWidget { selection-background-color:#1e1e1e; selection-color:#f5f5f5; gridline-color:#1f1f1f; }
QTextEdit { font-family:Consolas; font-size:11px; }
QHeaderView::section { background:#111111; color:#a3a3a3; border:0; border-bottom:1px solid #1f1f1f; padding:6px; }
QMenuBar { background:#000000; border:0; padding:2px 4px 0; }
QMenuBar::item { padding:4px 8px; border-radius:6px; background:transparent; }
QMenuBar::item:selected { background:#1e1e1e; }
QMenu { background:#111111; border:1px solid #262626; border-radius:8px; padding:4px; }
QMenu::item { padding:5px 22px; border-radius:4px; }
QMenu::item:selected { background:#1e1e1e; color:#f5f5f5; }
QMenu::item:disabled { color:#525252; }
QMenu::separator { height:1px; background:#1f1f1f; margin:4px 6px; }
QSplitter::handle { background:transparent; }
QScrollArea { background:#0a0a0a; border:0; }
QWidget#modelSidebar,QStackedWidget#modelPages,QWidget[motion_tool],
QWidget#modelSettingsViewport,QFrame#modelActions { background:#000000; }
QScrollArea#modelSettings,QStackedWidget#modelPages > QWidget,QWidget#modelSettingsViewport > QWidget { background:#000000; }
QWidget#modelSidebar QGroupBox { background:#0a0a0a; }
QScrollBar:vertical { width:9px; background:transparent; }
QScrollBar::handle:vertical { background:#2a2a2a; border-radius:3px; min-height:25px; }
QScrollBar::handle:vertical:hover { background:#3a3a3a; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical { background:transparent; }
QScrollBar:horizontal { height:9px; background:transparent; }
QScrollBar::handle:horizontal { background:#2a2a2a; border-radius:3px; min-width:25px; }
QScrollBar::handle:horizontal:hover { background:#3a3a3a; }
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal { width:0; }
QScrollBar::add-page:horizontal,QScrollBar::sub-page:horizontal { background:transparent; }
QToolTip { background:#171717; color:#f5f5f5; border:1px solid #2f2f2f; padding:6px 8px; }
"""

# The dark colours of STYLE and of DARK below, each with its light counterpart.
LIGHT = {
    '#000000': '#fafafa', '#0a0a0a': '#fcfcfc', '#0d0d0d': '#fafafa', '#0f0f0f': '#f4f4f5', '#111111': '#ffffff',
    '#131313': '#f4f4f5', '#141414': '#ffffff', '#161616': '#f4f4f5', '#171717': '#ffffff', '#1a1a1a': '#e4e4e7',
    '#1c1c1c': '#ececef', '#1e1e1e': '#f0f0f2', '#1f1f1f': '#e4e4e7', '#262626': '#d4d4d8', '#2a2a2a': '#d4d4d8',
    '#2f2f2f': '#c4c4ca', '#333333': '#c4c4ca', '#3a3a3a': '#a1a1aa', '#525252': '#a1a1aa', '#6b6b6b': '#8a8a94',
    '#818181': '#71717b', '#8f8f8f': '#63636d', '#a3a3a3': '#52525c', '#d4d4d4': '#3f3f46', '#f5f5f5': '#27272a',
    '#fafafa': '#18181b', '#346bf1': '#1b4ed8', '#3061d9': '#3160db', '#2f61da': '#1a44bd', '#5b8af5': '#3160db',
    '#51a2ff': '#1447e6', '#17213a': '#e0e6f7', '#121b34': '#e0e6f7', '#16233f': '#b8c6ee', '#6f7c99': '#f4f6fd',
    '#264a9e': '#1b4ed8', '#ffb900': '#bb4d00',
}

# Classic Dark colours named across the program, each with this look's dark colour. Greys not listed here go
# through GREYS; other colours stay as they are.
DARK = {
    '#171717': '#1f1f1f', '#181818': '#1f1f1f', '#191919': '#1f1f1f', '#1b1b1b': '#000000', '#202020': '#0a0a0a',
    '#222222': '#0a0a0a', '#242424': '#0a0a0a', '#252525': '#0a0a0a', '#292929': '#111111', '#303030': '#111111',
    '#333333': '#141414', '#353535': '#131313', '#383838': '#0d0d0d', '#454545': '#1c1c1c', '#515151': '#2a2a2a',
    '#555555': '#2a2a2a', '#666666': '#333333', '#737373': '#3a3a3a', '#747474': '#525252', '#777777': '#6b6b6b',
    '#8d8d8d': '#818181', '#999999': '#818181', '#a0a0a0': '#818181', '#a8a8a8': '#8f8f8f', '#aaaaaa': '#8f8f8f',
    '#b3b3b3': '#a3a3a3', '#b8b8b8': '#a3a3a3', '#bcbcbc': '#a3a3a3', '#bebebe': '#a3a3a3', '#c5c5c5': '#d4d4d4',
    '#d5d5d5': '#f5f5f5', '#dddddd': '#f5f5f5', '#dedede': '#f5f5f5', '#e6e6e6': '#f5f5f5', '#ededed': '#fafafa',
    '#eeeeee': '#fafafa', '#f4f4f4': '#fafafa',
    '#477bb7': '#346bf1', '#37689d': '#2f61da', '#548dce': '#3061d9', '#658fbd': '#346bf1', '#74a6e0': '#5b8af5',
    '#404c59': '#17213a', '#303f50': '#121b34', '#385571': '#17213a', '#8ed2ff': '#51a2ff', '#b9d9ff': '#51a2ff',
    '#ffb45c': '#ffb900',
}
# Classic Light colours of the accent family, each with this look's light colour.
LIGHT_ACCENTS = {
    '#2866a3': '#1b4ed8', '#205588': '#1a44bd', '#3579b9': '#3160db', '#397eb9': '#1b4ed8', '#125594': '#1447e6',
    '#245786': '#1447e6', '#174674': '#1b4ed8', '#47729b': '#3160db', '#e0ebf7': '#e0e6f7', '#dce9f7': '#e0e6f7',
    '#8a4809': '#bb4d00',
}
# Classic colours of the timeline's tracks (the video, the motion, the two feet, the two hands, generated frames),
# each with this look's livelier colour.
TRACKS = {'#6289ad': '#615fff', '#6aa186': '#00d492', '#56b6a7': '#00d3f2', '#c78f5f': '#ff8904', '#79a9db': '#a684ff',
          '#bf8dcf': '#fb64b6', '#d8a24a': '#ffb900', '#8a4809': '#ffb900'}
# Grey level of Classic Dark -> grey level of this look, for greys DARK does not name.
GREYS = ((0x00, 0x00), (0x25, 0x0a), (0x30, 0x11), (0x45, 0x1c), (0x55, 0x2a), (0x74, 0x52), (0xa0, 0x81),
         (0xb8, 0xa3), (0xd5, 0xf5), (0xff, 0xff))
# Grey level of this look's dark side -> grey level of its light side, for greys LIGHT does not name.
LIGHT_GREYS = ((0x00, 0xfa), (0x0a, 0xfc), (0x11, 0xff), (0x1f, 0xe4), (0x2a, 0xd4), (0x3a, 0xa1), (0x52, 0xa1),
               (0x81, 0x71), (0xa3, 0x52), (0xf5, 0x27), (0xff, 0x18))


def _levels(color):
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def is_grey(color):
    levels = _levels(color)
    return max(levels) - min(levels) <= 10


def _along(points, level):
    for (low, low_to), (high, high_to) in zip(points, points[1:]):
        if level <= high:
            return round(low_to + (high_to - low_to) * (level - low) / (high - low))
    return points[-1][1]


def dark(color):
    """This look's dark colour for a Classic Dark one."""
    color = color.lower()
    if color in DARK:
        return DARK[color]
    if color == '#ffffff' or not is_grey(color):
        return color
    level = _along(GREYS, round(sum(_levels(color)) / 3))
    return f'#{level:02x}{level:02x}{level:02x}'


def light(color):
    """The light counterpart of one of this look's dark colours."""
    if color in LIGHT:
        return LIGHT[color]
    if color == '#ffffff' or not is_grey(color):
        return color
    level = _along(LIGHT_GREYS, round(sum(_levels(color)) / 3))
    return f'#{level:02x}{level:02x}{min(255, level + (9 if 0x40 <= level <= 0xc0 else 3)):02x}'


def color(classic_dark, classic_light, is_light):
    """This look's colour for a Classic Dark colour. `classic_light` is what Classic Light shows for it, or None
    when Classic Light keeps the dark colour (text on a coloured bar, an overlay on the video)."""
    classic_dark = classic_dark.lower()
    if not is_light:
        return dark(classic_dark)
    if classic_light is None:
        return dark(classic_dark)
    classic_light = classic_light.lower()
    if is_grey(classic_dark) and classic_dark != '#ffffff':
        return light(dark(classic_dark))
    return LIGHT_ACCENTS.get(classic_light, classic_light)


def track(color):
    """The colour of a timeline bar for the colour the code gives it: a track's own colour from TRACKS, the green of the
    motion row for a person whose colour is a grey, any other colour of a person made livelier. None for a dark grey,
    which the code gives a row that has nothing in it."""
    color = QColor(color).name()
    if color in TRACKS:
        return TRACKS[color]
    if is_grey(color):
        return TRACKS['#6aa186'] if _levels(color)[0] >= 0x70 else None
    hue, saturation, value, _ = QColor(color).getHsvF()
    return QColor.fromHsvF(max(0., hue), min(1., saturation * 1.35 + .1), max(value, .9)).name()


def bar_opacity(alpha):
    """How solid a bar is for the alpha Classic tints it with: its usual 80 and above are solid."""
    return min(1., alpha / 80)


def paint_bar(p, rect, color, alpha, radius, is_light):
    """A rounded bar of a timeline track, solid for a Classic alpha of 80 and above and fainter below."""
    if rect.width() <= 0 or rect.height() <= 0:
        return
    vivid, opacity = track(color), bar_opacity(alpha)
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    radius = min(radius, rect.height() / 2, rect.width() / 2)
    if vivid is None:
        p.setBrush(QColor(light('#111111') if is_light else '#111111'))
    else:
        top, bottom = QColor(vivid).lighter(108), QColor(vivid).darker(112)
        top.setAlphaF(opacity);bottom.setAlphaF(opacity)
        fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        fill.setColorAt(0, top);fill.setColorAt(1, bottom)
        p.setBrush(fill)
    p.drawRoundedRect(rect, radius, radius)
    p.restore()


def bar_text(color, alpha, is_light):
    """The colour of text on a bar: white or near-black on a solid bar, whichever reads better, the look's text
    colour on a faint bar and its muted colour on an empty row."""
    vivid = track(color)
    if vivid is None:
        return QColor(light('#818181') if is_light else '#818181')
    if bar_opacity(alpha) < .7:
        return QColor(light('#f5f5f5') if is_light else '#f5f5f5')
    red, green, blue = (level / 255 for level in _levels(vivid))
    return QColor('#ffffff' if .2126 * red + .7152 * green + .0722 * blue < .5 else '#0a0a0a')


def grid(major, is_light):
    """The time grid: thin lines that stay quiet over a solid bar."""
    level = 0 if is_light else 255
    return QColor(level, level, level, (34 if major else 12) if not is_light else (40 if major else 14))


def outline(p, rect, color, width, radius, dashed=False):
    """A rounded outline, as around the chosen block."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(color), width, Qt.DashLine if dashed else Qt.SolidLine))
    inset = width / 2
    box = rect.adjusted(inset, inset, -inset, -inset)
    radius = max(0., min(radius, box.height() / 2, box.width() / 2))
    p.drawRoundedRect(box, radius, radius)
    p.restore()


def paint_person(p, row, color, swatch, is_light):
    """A row of the People list as Dark and Light draw it: a rounded background when it is the chosen row or under the
    cursor, the check box where the list answers clicks, the person's picture with round corners (their initial while
    there is none), the name, and their colour as a dot inside `swatch`, the place the list opens the colour from."""
    tone = light if is_light else (lambda value: value)
    style, box = row.widget.style(), QRectF(row.rect).adjusted(1.5, 1.5, -1.5, -1.5)
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    if row.state & (QStyle.State_Selected | QStyle.State_MouseOver):
        chosen = bool(row.state & QStyle.State_Selected)
        p.setPen(QPen(QColor(tone('#262626')), 1) if chosen else Qt.NoPen)
        p.setBrush(QColor(tone('#1e1e1e' if chosen else '#161616')))
        p.drawRoundedRect(box, 8, 8)
    p.setOpacity(1 if row.state & QStyle.State_Enabled else .45)
    check = QRectF(style.subElementRect(QStyle.SE_ItemViewItemCheckIndicator, row, row.widget))
    side = min(check.width(), check.height())
    check = QRectF(check.center().x() - side / 2, check.center().y() - side / 2, side, side).adjusted(.5, .5, -.5, -.5)
    checked = row.checkState == Qt.Checked
    p.setPen(QPen(QColor(tone('#346bf1' if checked else '#3a3a3a')), 1))
    p.setBrush(QColor(tone('#346bf1' if checked else '#141414')))
    p.drawRoundedRect(check, 4, 4)
    if checked:
        p.setPen(QPen(QColor('#ffffff'), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        left, top, unit = check.left(), check.top(), check.width() / 15
        p.drawPolyline(QPolygonF([QPointF(left + 3.6 * unit, top + 7.8 * unit), QPointF(left + 6.4 * unit, top + 10.6 * unit),
                                  QPointF(left + 11.4 * unit, top + 4.8 * unit)]))
    height = max(8., min(float(row.decorationSize.height()), box.height() - 6))
    picture = QRectF(check.right() + 9, box.center().y() - height / 2, height * 3 / 4, height)
    shape = QPainterPath()
    shape.addRoundedRect(picture, 6, 6)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(tone('#171717')))
    p.drawPath(shape)
    image = row.icon.pixmap(row.decorationSize * 2) if not row.icon.isNull() else QPixmap()
    blank = image.isNull() or image.toImage().pixelColor(image.width() // 2, image.height() // 2).alpha() == 0
    if not blank:
        p.setClipPath(shape)
        scale = max(picture.width() / image.width(), picture.height() / image.height())
        width, tall = image.width() * scale, image.height() * scale
        p.drawPixmap(QRectF(picture.center().x() - width / 2, picture.center().y() - tall / 2, width, tall), image, QRectF(image.rect()))
        p.setClipping(False)
    else:
        initial = QFont(row.font)
        initial.setWeight(QFont.DemiBold)
        initial.setPointSizeF(row.font.pointSizeF() * 1.25)
        p.setFont(initial)
        p.setPen(QColor(tone('#a3a3a3') if is_grey(QColor(color).name()) else track(color)))
        p.drawText(picture, Qt.AlignCenter, row.text.strip()[:1].upper())
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(tone('#262626')), 1))
    p.drawPath(shape)
    name = QFont(row.font)
    name.setWeight(QFont.Medium)
    p.setFont(name)
    p.setPen(QColor(tone('#f5f5f5')))
    space = QRectF(picture.right() + 9, box.top(), max(0., swatch.left() - picture.right() - 13), box.height())
    p.drawText(space, Qt.AlignVCenter | Qt.AlignLeft, QFontMetricsF(name).elidedText(row.text, Qt.ElideRight, space.width()))
    dot = QRectF(0, 0, round(side * .875), round(side * .875))
    dot.moveCenter(QRectF(swatch).center())
    p.setPen(QPen(QColor(tone('#3a3a3a')), 1))
    p.setBrush(QColor(color))
    p.drawEllipse(dot)
    p.restore()


def stylesheet(is_light):
    style = STYLE
    if is_light:
        style = re.sub(r'#[0-9a-f]{6}\b', lambda match: light(match[0]), style)
    icons = ROOT / 'assets/icons'
    suffix = '-light' if is_light else ''
    return (style.replace('CHECK_ICON', (icons / 'check.svg').as_posix())
            .replace('CHEVRON_DOWN', (icons / f'chevron-down{suffix}.svg').as_posix())
            .replace('CHEVRON_UP', (icons / f'chevron-up{suffix}.svg').as_posix()))


def palette(base, is_light):
    tone = light if is_light else (lambda value: value)
    for role, value in ((QPalette.Window, '#0a0a0a'), (QPalette.WindowText, '#f5f5f5'), (QPalette.Base, '#0a0a0a'),
                        (QPalette.AlternateBase, '#111111'), (QPalette.Text, '#f5f5f5'), (QPalette.Button, '#171717'),
                        (QPalette.ButtonText, '#f5f5f5'), (QPalette.Highlight, '#264a9e'),
                        (QPalette.HighlightedText, '#ffffff'), (QPalette.ToolTipBase, '#111111'),
                        (QPalette.ToolTipText, '#f5f5f5'), (QPalette.PlaceholderText, '#818181'),
                        (QPalette.Link, '#51a2ff'), (QPalette.Light, '#262626'), (QPalette.Midlight, '#1f1f1f'),
                        (QPalette.Mid, '#1f1f1f'), (QPalette.Dark, '#1a1a1a'), (QPalette.Shadow, '#000000')):
        base.setColor(role, QColor(tone(value)))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        base.setColor(QPalette.Disabled, role, QColor(tone('#525252')))
    return base


def load_fonts():
    """Registers the bundled Inter files (SIL Open Font License) and says whether the family is usable."""
    loaded = [QFontDatabase.addApplicationFont(str(ROOT / 'assets/fonts' / name)) >= 0 for name in FONT_FILES]
    return all(loaded)


def font(fallback, points):
    """Inter first; the bundled Noto Sans KR after it for the Hangul Inter does not have."""
    result = QFont(fallback)
    families = [FAMILY] + [family for family in fallback.families() if family != FAMILY]
    result.setFamilies(families)
    result.setPointSizeF(points)
    return result


TRANSPORT = {'|<': 'first', '<': 'previous', '>': 'next', '>|': 'last'}   # the text of a transport button -> its picture


def transport_picture(kind, size, color):
    """A transport button's picture in a 16-unit box: a triangle towards the frame it goes to, with a bar for the
    first and the last frame; the play triangle and the two bars of pause."""
    image = QPixmap(size, size)
    image.fill(Qt.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / 16, size / 16)
    if kind in ('next', 'last'):   # drawn pointing back, mirrored for forward
        p.translate(16, 0)
        p.scale(-1, 1)
    p.setPen(QPen(QColor(color), 1.3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(QColor(color))
    if kind in ('play', 'pause'):
        if kind == 'play':
            p.drawPolygon(QPolygonF([QPointF(4.6, 2.8), QPointF(4.6, 13.2), QPointF(13, 8)]))
        else:
            p.setPen(QPen(QColor(color), 2.6, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(5.2, 3.6), QPointF(5.2, 12.4))
            p.drawLine(QPointF(10.8, 3.6), QPointF(10.8, 12.4))
        p.end()
        return image
    barred = kind in ('first', 'last')
    right = 12.2 if barred else 10.8
    p.drawPolygon(QPolygonF([QPointF(right, 3.8), QPointF(right, 12.2), QPointF(right - 6, 8)]))
    if barred:
        p.setPen(QPen(QColor(color), 1.7, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(3.8, 3.8), QPointF(3.8, 12.2))
    p.end()
    return image


def dress(widget, classic, is_light, scale=100):
    """What a style sheet cannot do for one widget: a transport button shows a picture in Dark and Light and its text
    (|<, <, >, >|) in the Classic looks, the play button shows its two pictures in the look's colour, and a window's
    title bar takes the look's colour."""
    tone = light if is_light else (lambda value: value)

    def icon(kind, side):
        result = QIcon()
        result.addPixmap(transport_picture(kind, side * 2, tone('#d4d4d4')), QIcon.Normal)
        result.addPixmap(transport_picture(kind, side * 2, tone('#525252')), QIcon.Disabled)
        return result

    if isinstance(widget, QPushButton) and widget.objectName() == 'transport':
        kind = widget.property('transportKind') or TRANSPORT.get(widget.text())
        window = widget.window()
        if getattr(window, 'play_button', None) is widget and hasattr(window, 'pause_icon'):
            # The window swaps its play and pause icons itself; here they become the look's.
            if not hasattr(window, 'classic_play_icons'):
                window.classic_play_icons = window.play_icon, window.pause_icon
            window.play_icon, window.pause_icon = window.classic_play_icons if classic else (
                icon('play', round(16 * scale / 100)), icon('pause', round(16 * scale / 100)))
            widget.setIcon(window.pause_icon if widget.accessibleName() == 'Pause' else window.play_icon)
        elif kind:
            if widget.property('transportKind') is None:
                widget.setProperty('transportKind', kind)
                widget.setProperty('transportText', widget.text())
            if classic:
                widget.setIcon(QIcon())
                widget.setText(widget.property('transportText'))
            else:
                side = round(14 * scale / 100)
                widget.setText('')
                widget.setIcon(icon(kind, side))
                widget.setIconSize(QSize(side, side))
    if widget.isWindow() and widget.windowType() in (Qt.Window, Qt.Dialog, Qt.Tool):
        title_bar(widget, classic, is_light)


def title_bar(window, classic, is_light):
    """On Windows 11, paints the window's title bar in the colour of the window under it, so the program's own surface
    starts at the top edge; the Classic looks give the title bar back to Windows. Says whether it did anything."""
    if sys.platform != 'win32' or QGuiApplication.platformName() != 'windows' or window.windowHandle() is None:
        return False
    import ctypes
    tone = light if is_light else (lambda value: value)

    def reference(color):
        red, green, blue = _levels(color)
        return red | green << 8 | blue << 16

    if classic:   # 0xffffffff asks for the colour Windows would use
        dark_bar = QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
        values = ((20, int(dark_bar)), (35, 0xffffffff), (36, 0xffffffff), (34, 0xffffffff))
    else:
        canvas = '#000000' if isinstance(window, QMainWindow) else '#0a0a0a'
        values = ((20, int(not is_light)), (35, reference(tone(canvas))), (36, reference(tone('#a3a3a3'))),
                  (34, reference(tone('#1f1f1f'))))
    try:
        set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        set_attribute.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint)
        handle = int(window.winId())
        for attribute, value in values:   # immersive dark mode, caption colour, text colour, border colour
            set_attribute(handle, attribute, ctypes.byref(ctypes.c_uint(value)), 4)
    except (AttributeError, OSError):
        return False
    return True


def dress_all(app, classic, is_light, scale):
    for widget in app.allWidgets():
        dress(widget, classic, is_light, scale)


def prepare(widget):
    """What a widget needs once, when the style polishes it: a menu gets a see-through background so the corners
    outside its rounded border show what is behind them (a menu is a window of its own), a transport button and a
    window get what dress() gives them."""
    if isinstance(widget, QMenu):
        if not widget.testAttribute(Qt.WA_TranslucentBackground):
            widget.setWindowFlag(Qt.FramelessWindowHint, True)
            widget.setWindowFlag(Qt.NoDropShadowWindowHint, True)
            widget.setAttribute(Qt.WA_TranslucentBackground, True)
        return
    is_window = widget.isWindow()
    if not is_window and widget.objectName() != 'transport':
        return
    app = QApplication.instance()
    theme, scale = app.property('uiTheme') or 'dark', app.property('uiScale') or 100
    classic, is_light = theme.startswith('classic'), theme.endswith('light')
    dress(widget, classic, is_light, scale)
    if is_window:   # a button polished before its window was complete
        for button in widget.findChildren(QPushButton, 'transport'):
            dress(button, classic, is_light, scale)


class Look(QProxyStyle):
    """The application's style: Fusion, and prepare() for every widget as it is polished. A style is asked once per
    widget; an event filter on the application would be asked for every event of every widget."""

    def polish(self, target):
        result = super().polish(target)
        if isinstance(target, QWidget):
            prepare(target)
        return result


def follow_windows(app):
    """Colour the title bar of a window when it gets the focus: a window polished before Windows made it has no title
    bar to colour yet."""
    def focused(window):
        if window is None:
            return
        widget = next((candidate for candidate in app.topLevelWidgets() if candidate.windowHandle() is window), None)
        if widget is not None and widget.windowType() in (Qt.Window, Qt.Dialog, Qt.Tool):
            theme = app.property('uiTheme') or 'dark'
            title_bar(widget, theme.startswith('classic'), theme.endswith('light'))
    app.focusWindowChanged.connect(focused)
