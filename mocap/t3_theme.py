"""The Dark and Light looks: near-black or near-white flat surfaces, thin light lines, rounded controls, Inter.

The colours, radii and font follow the stock palette of the T3 Code desktop app (its `themePalettes.ts`, read
2026-11-26). The older looks stay in `theme.py` as Classic Dark and Classic Light, and every colour the rest of the
program names is still the Classic Dark one: `color()` turns it into this look's colour, so painted widgets and
widget style sheets need no second set of colours.
"""
import re

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QMenu

from .paths import ROOT

FAMILY = 'Inter'
FONT_FILES = ('Inter-Regular.ttf', 'Inter-Medium.ttf', 'Inter-SemiBold.ttf', 'Inter-Bold.ttf')

# Written in the dark colours; LIGHT turns each one into its light counterpart.
STYLE = """
QWidget { color:#f5f5f5; font-size:12px; }
QMainWindow,QDialog { background:#0a0a0a; }
QLabel#brand { font-size:13px; font-weight:600; color:#f5f5f5; padding:0 8px; }
QLabel#muted { color:#818181; }
QLabel#status { color:#a3a3a3; padding:3px 6px; }
QLabel#taskInfo { color:#51a2ff; padding:3px 6px; }
QFrame#editor { background:#0a0a0a; border:1px solid #1f1f1f; }
QWidget#editorHeader { background:#0a0a0a; border-bottom:1px solid #1f1f1f; }
QTabWidget#peopleTabs::pane { background:#0a0a0a; border:0; }
QTabWidget#peopleTabs QTabBar, QTabBar#modelManagerTabs { qproperty-drawBase:0; }
QTabWidget#peopleTabs QTabBar::tab, QTabBar#modelManagerTabs::tab { background:#0a0a0a; color:#a3a3a3; border:1px solid #0a0a0a; border-radius:6px; padding:5px 10px; }
QTabWidget#peopleTabs QTabBar::tab:!selected:hover, QTabBar#modelManagerTabs::tab:!selected:hover { background:#161616; color:#f5f5f5; }
QTabWidget#peopleTabs QTabBar::tab:selected, QTabBar#modelManagerTabs::tab:selected { background:#1e1e1e; border-color:#262626; color:#f5f5f5; }
QGroupBox { background:#111111; border:1px solid #1f1f1f; border-radius:8px; margin-top:17px; padding:9px 7px 7px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:4px; padding:0 3px; color:#f5f5f5; }
QPushButton { background:#171717; border:1px solid #262626; border-radius:6px; padding:5px 9px; font-weight:500; }
QPushButton:hover { background:#1e1e1e; border-color:#2f2f2f; }
QPushButton:pressed { background:#111111; }
QPushButton:checked { background:#1e1e1e; border-color:#3a3a3a; }
QPushButton:disabled { color:#525252; background:#0f0f0f; border-color:#1a1a1a; }
QPushButton#primary { background:#346bf1; border-color:#346bf1; color:#ffffff; font-weight:600; }
QPushButton#primary:hover { background:#3061d9; border-color:#3061d9; }
QPushButton#primary:disabled { background:#16233f; border-color:#16233f; color:#6f7c99; }
QPushButton#transport { padding:3px; min-width:25px; }
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
QListWidget#peopleList::indicator,QListWidget#streamPeopleList::indicator { width:16px; height:16px; border:1px solid #3a3a3a; border-radius:4px; background:#141414; }
QListWidget#peopleList::indicator:hover,QListWidget#streamPeopleList::indicator:hover { border-color:#5b8af5; background:#17213a; }
QListWidget#peopleList::indicator:checked,QListWidget#streamPeopleList::indicator:checked { border-color:#346bf1; background:#346bf1; image:url("CHECK_ICON"); }
QListWidget#peopleList::indicator:disabled,QListWidget#streamPeopleList::indicator:disabled { border-color:#262626; background:#0f0f0f; }
QListWidget,QTreeWidget { background:#0a0a0a; selection-background-color:#1e1e1e; selection-color:#f5f5f5; outline:0; }
QProgressBar { background:#141414; border:0; height:3px; }
QProgressBar::chunk { background:#346bf1; }
QTextEdit,QPlainTextEdit,QTableWidget { background:#0a0a0a; border:1px solid #1f1f1f; border-radius:6px; selection-background-color:#264a9e; selection-color:#ffffff; }
QTableWidget { selection-background-color:#1e1e1e; selection-color:#f5f5f5; gridline-color:#1f1f1f; }
QTextEdit { font-family:Consolas; font-size:11px; }
QHeaderView::section { background:#111111; color:#a3a3a3; border:0; border-bottom:1px solid #1f1f1f; padding:6px; }
QMenuBar { background:#0a0a0a; border-bottom:1px solid #1f1f1f; }
QMenuBar::item { padding:4px 8px; border-radius:6px; background:transparent; }
QMenuBar::item:selected { background:#1e1e1e; }
QMenu { background:#111111; border:1px solid #262626; border-radius:8px; padding:4px; }
QMenu::item { padding:5px 22px; border-radius:4px; }
QMenu::item:selected { background:#1e1e1e; color:#f5f5f5; }
QMenu::item:disabled { color:#525252; }
QMenu::separator { height:1px; background:#1f1f1f; margin:4px 6px; }
QSplitter::handle { background:#0a0a0a; }
QScrollArea { background:#0a0a0a; border:0; }
QWidget#modelSidebar,QStackedWidget#modelPages,QWidget[motion_tool],
QWidget#modelSettingsViewport,QFrame#modelActions { background:#0a0a0a; }
QWidget#modelSidebar QGroupBox { background:#111111; }
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
QToolTip { background:#111111; color:#f5f5f5; border:1px solid #262626; padding:5px; }
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


class RoundMenus(QObject):
    """Lets the rounded corners of a menu show what is behind them: a menu is a window of its own, and without a
    see-through background its corners outside the rounded border are painted square."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Polish and isinstance(watched, QMenu) and not watched.testAttribute(Qt.WA_TranslucentBackground):
            watched.setWindowFlag(Qt.FramelessWindowHint, True)
            watched.setWindowFlag(Qt.NoDropShadowWindowHint, True)
            watched.setAttribute(Qt.WA_TranslucentBackground, True)
        return False
