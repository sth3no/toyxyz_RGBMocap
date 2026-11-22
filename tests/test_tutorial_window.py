"""The tutorial in the real main window, offscreen: menu, first start, hint, appearance and the targets of every step."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from mocap import paths, tutorial
from mocap.gui import MainWindow
from mocap.tutorial_steps import CHAPTERS

APP = QApplication.instance() or QApplication([])


class WindowCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.object(paths, 'HOME', Path(tmp.name))
        home.start()
        self.addCleanup(home.stop)
        self.settings = Path(tmp.name) / 'config/local.json'

    def window(self):
        window = MainWindow()
        def close():
            window.close();window.deleteLater();APP.processEvents()
        self.addCleanup(close)
        window.show();APP.processEvents()
        return window


class ReadinessTest(WindowCase):
    def test_window_says_whether_model_files_are_missing(self):
        w = self.window()
        self.assertIs(w.models_missing, 'Warning:' in w.readiness.text())


if __name__ == '__main__':
    unittest.main()
