"""Hidden ranges controls under the take selector; the decisions live in hidden_ranges_state."""
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QSpinBox,
    QScrollArea, QFrame, QSizePolicy)
from .hidden_ranges import OFFSET_LIMIT, SEED_LIMIT, method_issue
from .theme import ui_px
from .hidden_ranges_state import can_fill, editable, gap_rows, pending_state

MANUAL_NOTE = 'Manual contact-lock edits on the source takes are not carried into the joined take.'
AUTO_TIP = 'Generate the missing motion from the captured poses on both sides. The result is plausible motion, not a reconstruction.'


class RangeRows(QScrollArea):
    """The rows take the height they need and scroll when the People column is shorter; a plain layout squeezes them into each other."""
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True);self.setFrameShape(QFrame.NoFrame);self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setStyleSheet('QScrollArea,QScrollArea>QWidget>QWidget{background:transparent}')
        box = QWidget();self.rows = QVBoxLayout(box);self.rows.setContentsMargins(0, 0, 0, 0);self.rows.setSpacing(3);self.setWidget(box)

    def content_height(self):
        box, width = self.widget(), max(1, self.viewport().width())
        return box.heightForWidth(width) if box.hasHeightForWidth() else box.sizeHint().height()

    def sizeHint(self):return QSize(super().sizeHint().width(), self.content_height())

    def minimumSizeHint(self):return QSize(0, min(self.content_height(), ui_px(72)))

    def resizeEvent(self, event):
        super().resizeEvent(event);self.updateGeometry()  # wrapped labels change height with the width


class HiddenRangesPanel(QWidget):
    # key, method, donor person id (None until chosen), mirror, offset, seed
    choice_changed = Signal(str, str, object, bool, int, int)
    fill_requested = Signal()
    setup_requested = Signal()  # open Model Manager to install what Automatic needs

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self);layout.setContentsMargins(0, 4, 0, 0);layout.setSpacing(3)
        layout.addWidget(QLabel('Hidden ranges'))
        self.area = RangeRows();self.rows = self.area.rows;layout.addWidget(self.area)
        self.reason = QLabel();self.reason.setObjectName('muted');self.reason.setWordWrap(True)
        self.fill = QPushButton('Fill hidden ranges')
        self.fill.setToolTip('Join the takes around the selected hidden ranges into one new take. The source takes are kept.')
        self.fill.clicked.connect(self._fill_clicked)
        self.setup = QPushButton('Set up…');self.setup.setToolTip('Open Model Manager to install what Automatic fill needs.')
        self.setup.clicked.connect(lambda: self.setup_requested.emit());self.setup.hide()
        self.manual = QLabel(MANUAL_NOTE);self.manual.setObjectName('muted');self.manual.setWordWrap(True);self.manual.hide()
        buttons = QHBoxLayout();buttons.setSpacing(6);buttons.addWidget(self.fill, 1);buttons.addWidget(self.setup)
        layout.addWidget(self.reason);layout.addLayout(buttons);layout.addWidget(self.manual)
        self._widgets, self._pending, self._seeds, self._issues, self._shown, self._rows = [], {}, {}, {}, None, ([], [], False)
        self._later = QTimer(self);self._later.setInterval(30);self._later.timeout.connect(self._rebuild);self.hide()

    def show_gaps(self, gaps, choices, fps, disabled_reason, people=None, ident=None, auto_issue=None):
        self._shown, self._pending = (gaps, choices, disabled_reason, people, ident, auto_issue), {}
        rows = gap_rows(gaps, choices, fps, people, ident, auto_issue)
        # An unsupported project, camera or result must not record choices.
        self._rows = (gaps, rows, editable(disabled_reason))
        self.fill.setEnabled(can_fill(rows, disabled_reason))
        self.setup.setVisible(any(row['setup'] for row in rows))
        self.reason.setText(disabled_reason or '');self.reason.setVisible(bool(disabled_reason))
        if not rows:self.hide()
        # An edited field is saved when a click elsewhere takes its focus, and saving shows the ranges again.
        # Replacing the rows under that click, or under an open list, would swallow it, so they wait for the mouse.
        if self._held():self._later.start()
        else:self._rebuild()

    @staticmethod
    def _held():return QApplication.mouseButtons() != Qt.NoButton or QApplication.activePopupWidget() is not None

    def _rebuild(self):
        if self._held():return
        self._later.stop()
        for widget in self._widgets:
            # Children too: a focused spin box would otherwise report editingFinished while it is removed.
            for w in (widget, *widget.findChildren(QWidget)):w.blockSignals(True)
            widget.setParent(None);widget.deleteLater()
        self._widgets, self._seeds, self._issues = [], {}, {}
        gaps, rows, enabled = self._rows
        for gap, row in zip(gaps, rows):
            label, combo = QLabel(row['label']), QComboBox()
            combo.setAccessibleName('Hidden range '+row['label'])
            combo.addItem('Do not fill', 'none');combo.addItem('Smooth join', 'smooth');combo.addItem('Copy from person', 'copy')
            combo.addItem('Automatic (Experimental)', 'auto');combo.setItemData(3, row['auto_issue'] or AUTO_TIP, Qt.ToolTipRole)
            # Only the range's own limit disables it; missing setup is shown on the chosen row with Set up….
            if method_issue('auto', gap):combo.model().item(3).setEnabled(False)
            if row['smooth_issue']:
                combo.model().item(1).setEnabled(False)
                combo.setItemData(1, row['smooth_issue'], Qt.ToolTipRole);combo.setToolTip(row['smooth_issue'])
            combo.setCurrentIndex(combo.findData(row['method']))
            combo.setEnabled(enabled)
            combo.currentIndexChanged.connect(lambda index, row=row, combo=combo: self.choice_changed.emit(
                row['key'], combo.itemData(index), row['person'], row['mirror'], row['offset'], row['seed']))
            widgets = [label, combo]
            if row['method'] == 'copy':widgets.append(self._copy_controls(row, enabled))
            if row['method'] == 'auto':widgets.append(self._auto_controls(row, enabled))
            if row['issue'] or row['method'] == 'copy':  # A copy row's issue follows the unsaved offset.
                issue = QLabel(row['issue'] or '');issue.setObjectName('muted');issue.setWordWrap(True)
                issue.setTextFormat(Qt.PlainText)  # it may name a person
                if not row['issue']:issue.hide()
                widgets.append(issue);self._issues[row['key']] = issue
            for widget in widgets:self.rows.addWidget(widget);self._widgets.append(widget)
        self.area.updateGeometry()
        self.setVisible(bool(rows))

    def _copy_controls(self, row, enabled):
        box = QWidget();layout = QVBoxLayout(box);layout.setContentsMargins(12, 0, 0, 0);layout.setSpacing(2)
        person = QComboBox();person.setAccessibleName('Copy from person, '+row['label'])
        person.setToolTip('The visible person whose motion fills this range.')
        person.addItem('Choose…', None)
        for ident, name in row['donors']:person.addItem(name, ident)
        person.setCurrentIndex(max(0, person.findData(row['person'])) if row['person'] is not None else 0)
        mirror = QCheckBox('Mirror');mirror.setChecked(row['mirror'])
        mirror.setToolTip('Swap left and right, for dancers who mirror each other.')
        offset = QSpinBox();offset.setRange(-OFFSET_LIMIT, OFFSET_LIMIT);offset.setSuffix(' f');offset.setValue(row['offset'])
        offset.setAccessibleName('Offset, '+row['label'])
        offset.setToolTip("Copy the other person's motion from earlier (negative) or later (positive) frames.")
        # Mirror and Offset each get a row: side by side they are cut off in a narrow People panel.
        line = QHBoxLayout();line.setSpacing(6);line.addWidget(QLabel('Offset'));line.addWidget(offset, 1)
        layout.addWidget(person);layout.addWidget(mirror);layout.addLayout(line)
        for widget in (person, mirror, offset):widget.setEnabled(enabled)
        emit = lambda *_: self.choice_changed.emit(row['key'], 'copy', person.currentData(), mirror.isChecked(), offset.value(), 0)
        person.currentIndexChanged.connect(emit);mirror.toggled.connect(emit)
        # Saving rewrites people.json and reloads the scene, so the offset is committed once editing ends;
        # meanwhile the row's issue and the Fill button follow the value being edited.
        offset.editingFinished.connect(lambda: offset.value() != row['offset'] and emit())
        offset.valueChanged.connect(lambda value: self._edit(row, (person.currentData(), mirror.isChecked(), value)))
        return box

    def _auto_controls(self, row, enabled):
        box = QWidget();line = QHBoxLayout(box);line.setContentsMargins(12, 0, 0, 0);line.setSpacing(6)
        seed = QSpinBox();seed.setRange(0, SEED_LIMIT);seed.setValue(row['seed']);seed.setEnabled(enabled)
        seed.setAccessibleName('Variation, '+row['label']);seed.setToolTip('Another number gives another motion for this range.')
        line.addWidget(QLabel('Variation'));line.addWidget(seed, 1)
        # Committed like Offset: saving rewrites people.json and reloads the scene.
        seed.editingFinished.connect(lambda: seed.value() != row['seed'] and self.choice_changed.emit(row['key'], 'auto', None, False, 0, seed.value()))
        self._seeds[row['key']] = (seed, row['seed'])
        return box

    def _edit(self, row, choice):
        if choice[2] == row['offset']:self._pending.pop(row['key'], None)
        else:self._pending[row['key']] = choice
        gaps, choices, reason, people, ident, auto_issue = self._shown
        issues, ok = pending_state(gaps, choices, self._pending, reason, people, ident, auto_issue)
        for key, label in self._issues.items():label.setText(issues.get(key) or '');label.setVisible(bool(issues.get(key)))
        self.fill.setEnabled(ok);self.area.updateGeometry()

    def _fill_clicked(self):
        # An offset or variation still being edited is saved first, so the worker never reads a stale one.
        # Both are read before any is saved: saving shows the rows again.
        pending = dict(self._pending)
        seeds = {key: spin.value() for key, (spin, saved) in self._seeds.items() if spin.value() != saved}
        for key, (person, mirror, offset) in pending.items():self.choice_changed.emit(key, 'copy', person, mirror, offset, 0)
        for key, seed in seeds.items():self.choice_changed.emit(key, 'auto', None, False, 0, seed)
        saved = self._shown[1] if self._shown else {}
        if (all(isinstance(saved.get(k), dict) and saved[k].get('offset') == c[2] for k, c in pending.items())
                and all(isinstance(saved.get(k), dict) and saved[k].get('seed') == s for k, s in seeds.items()) and self.fill.isEnabled()):
            self.fill_requested.emit()

    def set_manual_note(self, shown):self.manual.setVisible(bool(shown))
