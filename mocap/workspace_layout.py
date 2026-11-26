"""The arrangement of the window on top of build_workspace.

People is a column of its own beside the video and the 3D viewport, the menus form the one bar at the top with the
file's line and the running job at its right end, and the menus of menus.py stand between File and Help. The widgets
are the ones build_workspace made; this module only moves them.
"""
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtWidgets import (QFrame, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QSplitter, QTabBar,
                               QVBoxLayout, QWidget)

from . import menus
from .theme import set_appearance_style, ui_px

PEOPLE_WIDTH = (190, 230, 380)   # the People column: least, at the start, most
CHIP = ('QFrame#jobChip{background:#303030;border:1px solid #555555;border-radius:10px}'
        'QFrame#jobChip QLabel{color:#d5d5d5;padding:0;background:transparent}'
        'QFrame#jobChip QProgressBar{background:#454545;border:0;border-radius:2px}'
        'QFrame#jobChip QProgressBar::chunk{background:#477bb7;border-radius:2px}')


class JobChip(QFrame):
    """The running job at the right end of the top bar: the window's own progress bar and time estimate, repeated
    small. It is there only while a job runs."""

    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self.setObjectName('jobChip')
        row = QHBoxLayout(self)
        self.bar = QProgressBar();self.bar.setTextVisible(False)
        self.text = QLabel('')
        row.addWidget(self.bar);row.addWidget(self.text)
        set_appearance_style(self, CHIP)
        self.refresh_appearance()
        self.timer = QTimer(self);self.timer.setInterval(400);self.timer.timeout.connect(self.sync);self.timer.start()
        window.progress.valueChanged.connect(self.sync)
        self.sync()

    def refresh_appearance(self):
        self.layout().setContentsMargins(ui_px(9), ui_px(2), ui_px(9), ui_px(2));self.layout().setSpacing(ui_px(7))
        self.bar.setFixedSize(ui_px(64), ui_px(4))

    def sync(self):
        w = self.window_
        busy = bool(getattr(w, 'busy', False))
        if busy:
            source = w.progress
            if (self.bar.minimum(), self.bar.maximum()) != (source.minimum(), source.maximum()):
                self.bar.setRange(source.minimum(), source.maximum())
            self.bar.setValue(source.value())
            span = source.maximum() - source.minimum()
            percent = f'{round(100 * (source.value() - source.minimum()) / span)} %' if span > 0 else ''
            text = '  '.join(filter(None, (percent, w.task_info.text())))
            if text != self.text.text():self.text.setText(text)
            self.setToolTip(w.status.text())
        if busy != self.isVisibleTo(self.parentWidget() or self):self.setVisible(busy)


class TopBarEnd(QWidget):
    """The right end of the menu bar. A menu bar measures its corner only when the corner is set, shown or hidden, so
    this sets itself again whenever its contents ask for another width."""

    def event(self, event):
        done = super().event(event)
        bar = self.parentWidget()
        if event.type() == QEvent.LayoutRequest and bar is not None and self.sizeHint().width() != self.width():
            bar.setCornerWidget(self, Qt.TopRightCorner)
            self.show()
        return done


def people_column(w):
    """Move the People panel out of the Movie Clip panel into the splitter beside it."""
    panel, editors = getattr(w, 'people_panel', None), w.workspace_split.widget(0)
    if panel is None or not isinstance(editors, QSplitter) or editors.indexOf(panel) >= 0:return False
    least, start, most = PEOPLE_WIDTH
    panel.setMinimumWidth(least);panel.setMaximumWidth(most)
    old = editors.sizes()
    editors.insertWidget(0, panel)
    editors.setCollapsible(0, False)
    for index, stretch in enumerate((0, 1, 1)):editors.setStretchFactor(index, stretch)
    editors.setSizes([start, max(260, old[0] - start), old[1] if len(old) > 1 else 650])
    return True


def top_bar(w):
    """Put the file's line and the running job at the right end of the menu bar and drop the row under the menus."""
    bar = w.menuBar()
    for label in w.centralWidget().findChildren(QLabel):
        if label.objectName() == 'brand' or label.text().startswith('Motion Workspace'):label.hide()
    end = TopBarEnd();end.setObjectName('topBarEnd')
    row = QHBoxLayout(end);row.setContentsMargins(0, 0, ui_px(8), 0);row.setSpacing(ui_px(10))
    w.filename.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    w.filename.setMaximumWidth(ui_px(560))   # a longer line is cut at its end; the window's title has it whole
    w.job_chip = JobChip(w)
    row.addWidget(w.filename);row.addWidget(w.job_chip)
    bar.setCornerWidget(end, Qt.TopRightCorner)
    end.show()
    w.top_bar_end = end


def face_group(w):
    """Put the two rows of Track Face, which stand loose between two groups, into a group of their own."""
    check = getattr(w, 'track_face', None)
    if check is None or isinstance(check.parentWidget(), QGroupBox):return None
    column = check.parentWidget().layout()
    rows = [index for index in range(column.count()) if column.itemAt(index).layout() is not None
            and any(column.itemAt(index).layout().indexOf(widget) >= 0 for widget in (check, w.neutral_face_note))]
    if len(rows) != 2:return None
    group = QGroupBox('Face');group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    box = QVBoxLayout(group);box.setSpacing(7)
    for row in [column.takeAt(index).layout() for index in reversed(rows)][::-1]:
        box.addLayout(row)
    column.insertWidget(rows[0], group)
    return group


class SectionTabs(QTabBar):
    """The names of the settings' sections above them. Choosing one scrolls the settings to that section; scrolling
    marks the section that is at the top. A section that is hidden has no name here."""

    def __init__(self, scroll, sections):
        super().__init__()
        self.setObjectName('sidebarTabs');self.setExpanding(False);self.setDrawBase(False);self.setUsesScrollButtons(False)
        self.scroll, self.sections = scroll, [section for _, section in sections]
        for name, _ in sections:self.addTab(name)
        self.tabBarClicked.connect(self.go)
        bar = scroll.verticalScrollBar()
        bar.valueChanged.connect(self.follow);bar.rangeChanged.connect(self.follow)
        self.follow()

    def top(self, section):
        return section.mapTo(self.scroll.widget(), section.rect().topLeft()).y()

    def go(self, index):
        if 0 <= index < len(self.sections):
            self.scroll.verticalScrollBar().setValue(max(0, self.top(self.sections[index]) - 4))

    def follow(self, *_):
        content, bar = self.scroll.widget(), self.scroll.verticalScrollBar()
        current = None
        for index, section in enumerate(self.sections):
            shown = section.isVisibleTo(content)
            if self.isTabVisible(index) != shown:self.setTabVisible(index, shown)
            if shown and (current is None or self.top(section) <= bar.value() + 12):current = index
        if bar.maximum() > 0 and bar.value() >= bar.maximum():   # at the very end the last section is the one in view
            current = max((index for index, section in enumerate(self.sections) if section.isVisibleTo(content)), default=current)
        if current is not None and current != self.currentIndex():
            self.blockSignals(True);self.setCurrentIndex(current);self.blockSignals(False)


def settings_sections(w):
    """Give the settings on the right their section names: Model, Correction, Face, Resources."""
    sidebar = getattr(w, 'appearance_sidebar', None)
    scroll = sidebar.findChild(QWidget, 'modelSettings') if sidebar is not None else None
    if scroll is None:return None
    face = face_group(w)
    sections = [('Model', w.model_choice.parentWidget()), ('Correction', getattr(w, 'correction_group', None)), ('Face', face),
                ('Resources', w.compute_mode.parentWidget())]
    w.settings_tabs = SectionTabs(scroll, [(name, section) for name, section in sections if section is not None])
    sidebar.layout().insertWidget(0, w.settings_tabs)
    return w.settings_tabs


def arrange(w):
    people_column(w)
    top_bar(w)
    settings_sections(w)
    menus.build(w)
