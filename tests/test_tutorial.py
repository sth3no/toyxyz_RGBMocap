"""The in-window tutorial: saved state, highlight, card and the controller, on stand-in windows."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

from mocap import paths, resource_safety, tutorial

APP = QApplication.instance() or QApplication([])


class HomeCase(unittest.TestCase):
    """Local settings in a temporary folder, never the installation's."""
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.object(paths, 'HOME', Path(tmp.name))
        home.start()
        self.addCleanup(home.stop)
        self.settings = Path(tmp.name) / 'config/local.json'


class StateTest(HomeCase):
    def test_missing_file_is_an_empty_state(self):
        self.assertEqual(tutorial.load_state(), {})

    def test_saved_values_are_merged_and_other_preferences_kept(self):
        resource_safety.update({'ui_theme': 'light'})
        self.assertTrue(tutorial.save_state(seen=True))
        self.assertTrue(tutorial.mark_done('basics'))
        tutorial.mark_done('basics')
        self.assertEqual(tutorial.load_state(), {'seen': True, 'done': ['basics']})
        self.assertEqual(resource_safety.read_settings(resource_safety.path())['ui_theme'], 'light')

    def test_unreadable_file_is_none_and_is_not_overwritten(self):
        self.settings.parent.mkdir(parents=True)
        self.settings.write_text('{broken', encoding='utf-8')
        self.assertIsNone(tutorial.load_state())
        self.assertFalse(tutorial.save_state(seen=True))
        self.assertEqual(self.settings.read_text(encoding='utf-8'), '{broken')

    def test_a_value_that_is_not_an_object_is_an_empty_state(self):
        resource_safety.update({'tutorial': 'yes'})
        self.assertEqual(tutorial.load_state(), {})


if __name__ == '__main__':
    unittest.main()
