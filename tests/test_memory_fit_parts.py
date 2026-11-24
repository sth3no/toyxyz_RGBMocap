"""Reconstruct in parts takes its part length from what fits the GPU and the disk now. Needs PySide6 and ffmpeg."""
import os
import shutil
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QDialog

import test_parts_window as base  # its scene on disk and its window with a job queue that starts no worker
from mocap import memory_fit, parts_runner, parts_ui

GIB = 1024 ** 3


def system(free_vram=14 * GIB, total_vram=16 * GIB, free_disk=500e9):
    """Stand in for the GPU and the project drive."""
    return (mock.patch.object(memory_fit, 'free_vram', return_value=(free_vram, total_vram)),
            mock.patch.object(shutil, 'disk_usage', return_value=SimpleNamespace(free=free_disk)))


@unittest.skipUnless(base.FFMPEG.is_file(), 'needs the bundled ffmpeg')
class FitTest(unittest.TestCase):
    """The 10 s ExMR scene of the parts tests, 64x48, with Anna and Ben checked."""
    MODEL = 'exmr'
    END = 10.
    setUp = base.RunnerTest.setUp
    finish = base.RunnerTest.finish

    def stop(self):
        """Ends the part the run has started, as Cancel Job does, so the run is over before the window closes."""
        self.assertTrue(base.pump(lambda: bool(self.calls)), self.w.status.text())
        self.w.parts_runner.cancel();self.finish(cancelled=True)
        self.assertTrue(base.pump(lambda: self.w.parts_runner is None), self.w.status.text())

    def test_fitted_reads_the_scene_the_people_the_gpu_and_the_disk(self):
        a, b = system()
        with a, b as disk:
            self.assertEqual(parts_runner.fitted(self.w), (300., 'time'))
        self.assertEqual(disk.call_args.args[0], self.w.people_root)
        a, b = system(free_disk=1e6)  # 64x48 frames of two people: 448 frames fit a quarter of 1 MB
        with a, b:
            self.assertEqual(parts_runner.fitted(self.w), (10., 'disk'))

    def test_an_unreadable_gpu_or_disk_does_not_limit(self):
        with mock.patch.object(memory_fit, 'free_vram', return_value=(None, None)), mock.patch.object(shutil, 'disk_usage', side_effect=OSError('gone')):
            self.assertEqual(parts_runner.fitted(self.w), (300., 'time'))

    def test_start_without_a_part_length_plans_the_fitted_one(self):
        self.w.footmr_refine.setChecked(False)
        a, b = system(free_disk=1e6)
        with a, b:
            runner = parts_runner.start(self.w, None, .5, True, 0, 300)
        self.assertIsNotNone(runner, self.w.status.text())
        self.assertEqual(runner.trims, [(0, 300)])  # a 10 s part covers the 10 s scene
        self.assertIn('Reconstruct in parts: parts of 0:10.0, the longest whose temporary frames fit a quarter of the free disk.',
                      self.w.log.toPlainText())
        self.stop()

    def test_a_given_part_length_is_not_replaced(self):
        self.w.footmr_refine.setChecked(False)
        with mock.patch.object(parts_runner, 'fitted') as fitted:
            runner = parts_runner.start(self.w, 3., .5, True)
        fitted.assert_not_called()
        self.assertEqual(runner.trims, base.PLAN)
        self.stop()

    def ask(self):
        shown = []
        def execute(dialog):
            shown.append((dialog.part.value(), dialog.fit_line.text()));return QDialog.Rejected
        a, b = system(free_disk=1e6)
        with a, b, mock.patch.object(parts_ui.PartsDialog, 'exec', execute):
            parts_ui.ask(self.w)
        return shown[0]

    def test_the_dialog_offers_the_fitted_length(self):
        self.assertEqual(self.ask(), (10., 'Suggested part length: 0:10.0, the longest whose temporary frames fit a quarter of the free disk.'))

    def test_a_saved_run_of_the_same_range_still_offers_its_own_length(self):
        from mocap.storage import atomic_json
        atomic_json(self.root/parts_runner.RECORD, dict(version=1, first=0, end=300, part=150, overlap=15))
        part, line = self.ask()
        self.assertEqual(part, 5.)
        self.assertTrue(line.startswith('Suggested part length: 0:10.0'))


@unittest.skipUnless(base.FFMPEG.is_file(), 'needs the bundled ffmpeg')
class DialogTest(unittest.TestCase):
    def setUp(self):base.application()

    def dialog(self, **kw):
        d = parts_ui.PartsDialog(None, 0, 54000, 30, 1920, 1080, 1, True, **kw);self.addCleanup(d.deleteLater);return d

    def test_the_suggestion_says_what_limits_it(self):
        d = self.dialog(part=300., fit=(300., 'time'))
        self.assertEqual(d.part.value(), 300.)
        self.assertEqual(d.fit_line.text(), 'Suggested part length: 5:00.0, at most 5 minutes, so a stop costs one part at most.')
        self.assertFalse(d.fit_line.isHidden())
        self.assertEqual(self.dialog(fit=(120., 'video memory')).fit_line.text(),
                         'Suggested part length: 2:00.0, the longest that fits the free video memory now.')

    def test_without_a_suggestion_there_is_no_line(self):
        d = self.dialog()
        self.assertTrue(d.fit_line.isHidden());self.assertEqual(d.part.value(), 60.)


if __name__ == '__main__':
    unittest.main()
