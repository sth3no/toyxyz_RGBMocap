"""The small dialog of Import take from another project: a project folder, one finished take of it, and the gap before it."""
from pathlib import Path

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QVBoxLayout)

from .hidden_ranges import AUTO_MAX_FRAMES, SMOOTH_MAX_FRAMES
from .take_import_check import FPS, sources
from .theme import ui_px

EMPTY = 'This folder has no finished take. Choose the folder of another project.'


class ImportTakeDialog(QDialog):
    def __init__(self, parent, name):
        super().__init__(parent)
        self.setWindowTitle('Import Take')
        self.setMinimumWidth(ui_px(460))
        layout = QVBoxLayout(self)
        intro = QLabel(f'Continue {name} with a finished take from another project. The take is placed where {name} '
                       'was last seen, turned to face the same way, after the gap. Fill the gap in Hidden ranges.')
        intro.setWordWrap(True);layout.addWidget(intro)
        form = QFormLayout();layout.addLayout(form)
        row = QHBoxLayout();self.folder = QLineEdit();self.folder.setReadOnly(True);self.folder.setAccessibleName('Project folder')
        self.folder.setPlaceholderText('No project chosen')
        browse = QPushButton('Choose Project...');browse.clicked.connect(self.browse)
        row.addWidget(self.folder, 1);row.addWidget(browse);form.addRow('Project:', row)
        self.takes = QComboBox();self.takes.setAccessibleName('Take to import');form.addRow('Take:', self.takes)
        self.empty = QLabel(EMPTY);self.empty.setObjectName('muted');self.empty.setWordWrap(True);self.empty.hide();form.addRow(self.empty)
        self.gap = QDoubleSpinBox();self.gap.setAccessibleName('Gap before the imported take, in seconds')
        self.gap.setDecimals(1);self.gap.setSingleStep(.1);self.gap.setRange(.1, int(AUTO_MAX_FRAMES/FPS*10)/10);self.gap.setValue(1.);self.gap.setSuffix(' s')
        form.addRow('Gap:', self.gap)
        self.hint = hint = QLabel(f'Smooth join fills a gap of up to {SMOOTH_MAX_FRAMES/FPS:.1f} s, Automatic (Experimental) up to '
                                  f'{self.gap.maximum():.1f} s. Automatic motion is plausible, not what the person did.')
        hint.setObjectName('muted');hint.setWordWrap(True);form.addRow(hint)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText('Import')
        self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject);layout.addWidget(self.buttons)
        self._update()

    def browse(self):
        from .paths import PROJECTS
        start = self.folder.text() or (str(PROJECTS) if PROJECTS.is_dir() else '')
        folder = QFileDialog.getExistingDirectory(self, 'Choose Project', start)
        if folder:self.set_folder(folder)

    def set_folder(self, folder):
        self.folder.setText(str(folder));self.takes.clear()
        for take in sources(folder):self.takes.addItem(take['label'], str(take['project']))
        self.empty.setVisible(self.takes.count() == 0);self._update()

    def _update(self):self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.takes.count() > 0)

    def choice(self):
        """(folder of the chosen take, gap in seconds), or None before a take is listed."""
        project = self.takes.currentData()
        return (Path(project), round(self.gap.value(), 1)) if project else None


def ask(parent, person):
    """The chosen (take folder, gap seconds) for `person`, or None when the dialog is cancelled."""
    dialog = ImportTakeDialog(parent, str(person.get('name') or f"Person {person['id']}"))
    try:return dialog.choice() if dialog.exec() == QDialog.Accepted else None
    finally:dialog.deleteLater()
