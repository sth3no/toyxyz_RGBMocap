"""Hidden ranges controls under the take selector; the decisions live in hidden_ranges_state."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QSpinBox
from .hidden_ranges import OFFSET_LIMIT
from .hidden_ranges_state import can_fill, editable, gap_rows, pending_state

MANUAL_NOTE = 'Manual contact-lock edits on the source takes are not carried into the joined take.'


class HiddenRangesPanel(QWidget):
    # key, method, donor person id (None until chosen), mirror, offset
    choice_changed = Signal(str, str, object, bool, int)
    fill_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self);layout.setContentsMargins(0, 4, 0, 0);layout.setSpacing(3)
        layout.addWidget(QLabel('Hidden ranges'))
        self.rows = QVBoxLayout();self.rows.setSpacing(3);layout.addLayout(self.rows)
        self.reason = QLabel();self.reason.setObjectName('muted');self.reason.setWordWrap(True)
        self.fill = QPushButton('Fill hidden ranges')
        self.fill.setToolTip('Join the takes around the selected hidden ranges into one new take. The source takes are kept.')
        self.fill.clicked.connect(self._fill_clicked)
        self.manual = QLabel(MANUAL_NOTE);self.manual.setObjectName('muted');self.manual.setWordWrap(True);self.manual.hide()
        for widget in (self.reason, self.fill, self.manual):layout.addWidget(widget)
        self._widgets, self._pending, self._issues, self._shown = [], {}, {}, None;self.hide()

    def show_gaps(self, gaps, choices, fps, disabled_reason, people=None, ident=None):
        for widget in self._widgets:
            # Children too: a focused spin box would otherwise report editingFinished while it is removed.
            for w in (widget, *widget.findChildren(QWidget)):w.blockSignals(True)
            widget.setParent(None);widget.deleteLater()
        self._widgets, self._pending, self._issues = [], {}, {}
        self._shown = (gaps, choices, disabled_reason, people, ident)
        rows = gap_rows(gaps, choices, fps, people, ident)
        # An unsupported project, camera or result must not record choices.
        enabled = editable(disabled_reason)
        for row in rows:
            label, combo = QLabel(row['label']), QComboBox()
            combo.setAccessibleName('Hidden range '+row['label'])
            combo.addItem('Do not fill', 'none');combo.addItem('Smooth join', 'smooth');combo.addItem('Copy from person', 'copy')
            if row['smooth_issue']:
                combo.model().item(1).setEnabled(False)
                combo.setItemData(1, row['smooth_issue'], Qt.ToolTipRole);combo.setToolTip(row['smooth_issue'])
            combo.setCurrentIndex(combo.findData(row['method']))
            combo.setEnabled(enabled)
            combo.currentIndexChanged.connect(lambda index, row=row, combo=combo: self.choice_changed.emit(
                row['key'], combo.itemData(index), row['person'], row['mirror'], row['offset']))
            widgets = [label, combo]
            if row['method'] == 'copy':widgets.append(self._copy_controls(row, enabled))
            if row['issue'] or row['method'] == 'copy':  # A copy row's issue follows the unsaved offset.
                issue = QLabel(row['issue'] or '');issue.setObjectName('muted');issue.setWordWrap(True)
                if not row['issue']:issue.hide()
                widgets.append(issue);self._issues[row['key']] = issue
            for widget in widgets:self.rows.addWidget(widget);self._widgets.append(widget)
        self.fill.setEnabled(can_fill(rows, disabled_reason))
        self.reason.setText(disabled_reason or '');self.reason.setVisible(bool(disabled_reason))
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
        emit = lambda *_: self.choice_changed.emit(row['key'], 'copy', person.currentData(), mirror.isChecked(), offset.value())
        person.currentIndexChanged.connect(emit);mirror.toggled.connect(emit)
        # Saving rewrites people.json and reloads the scene, so the offset is committed once editing ends;
        # meanwhile the row's issue and the Fill button follow the value being edited.
        offset.editingFinished.connect(lambda: offset.value() != row['offset'] and emit())
        offset.valueChanged.connect(lambda value: self._edit(row, (person.currentData(), mirror.isChecked(), value)))
        return box

    def _edit(self, row, choice):
        if choice[2] == row['offset']:self._pending.pop(row['key'], None)
        else:self._pending[row['key']] = choice
        gaps, choices, reason, people, ident = self._shown
        issues, ok = pending_state(gaps, choices, self._pending, reason, people, ident)
        for key, label in self._issues.items():label.setText(issues.get(key) or '');label.setVisible(bool(issues.get(key)))
        self.fill.setEnabled(ok)

    def _fill_clicked(self):
        # An offset still being edited is saved first, so the worker never reads a stale one.
        pending = dict(self._pending)
        for key, (person, mirror, offset) in pending.items():self.choice_changed.emit(key, 'copy', person, mirror, offset)
        saved = self._shown[1] if self._shown else {}
        if all(isinstance(saved.get(k), dict) and saved[k].get('offset') == c[2] for k, c in pending.items()) and self.fill.isEnabled():
            self.fill_requested.emit()

    def set_manual_note(self, shown):self.manual.setVisible(bool(shown))
