"""Hidden ranges controls under the take selector; the decisions live in hidden_ranges_state."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
from .hidden_ranges_state import can_fill, editable, gap_rows

MANUAL_NOTE = 'Manual contact-lock edits on the source takes are not carried into the joined take.'


class HiddenRangesPanel(QWidget):
    choice_changed = Signal(str, str)
    fill_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self);layout.setContentsMargins(0, 4, 0, 0);layout.setSpacing(3)
        layout.addWidget(QLabel('Hidden ranges'))
        self.rows = QVBoxLayout();self.rows.setSpacing(3);layout.addLayout(self.rows)
        self.reason = QLabel();self.reason.setObjectName('muted');self.reason.setWordWrap(True)
        self.fill = QPushButton('Fill hidden ranges')
        self.fill.setToolTip('Join the takes around the selected hidden ranges into one new take. The source takes are kept.')
        self.fill.clicked.connect(self.fill_requested)
        self.manual = QLabel(MANUAL_NOTE);self.manual.setObjectName('muted');self.manual.setWordWrap(True);self.manual.hide()
        for widget in (self.reason, self.fill, self.manual):layout.addWidget(widget)
        self._widgets = [];self.hide()

    def show_gaps(self, gaps, choices, fps, disabled_reason):
        for widget in self._widgets:widget.blockSignals(True);widget.setParent(None);widget.deleteLater()
        self._widgets = []
        rows = gap_rows(gaps, choices, fps)
        for row in rows:
            label, combo = QLabel(row['label']), QComboBox()
            combo.setAccessibleName('Hidden range '+row['label'])
            combo.addItem('Do not fill', 'none');combo.addItem('Smooth join', 'smooth')
            if row['issue']:
                combo.model().item(1).setEnabled(False)
                combo.setItemData(1, row['issue'], Qt.ToolTipRole);combo.setToolTip(row['issue'])
            combo.setCurrentIndex(combo.findData(row['method']))
            # An unsupported project, camera or result must not record choices.
            combo.setEnabled(editable(disabled_reason))
            combo.currentIndexChanged.connect(lambda index, key=row['key'], combo=combo: self.choice_changed.emit(key, combo.itemData(index)))
            for widget in (label, combo):self.rows.addWidget(widget);self._widgets.append(widget)
        self.fill.setEnabled(can_fill(rows, disabled_reason))
        self.reason.setText(disabled_reason or '');self.reason.setVisible(bool(disabled_reason))
        self.setVisible(bool(rows))

    def set_manual_note(self, shown):self.manual.setVisible(bool(shown))
