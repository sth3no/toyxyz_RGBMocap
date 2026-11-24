"""The window tells the user when memory is short before and during a reconstruction, keeps running, and learns the peaks."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

from mocap import memory_fit, memory_note, paths, processing_guard  # imported here: patch.dict of sys.modules would drop a module first imported inside it

APP = QApplication.instance() or QApplication([])
GIB = 1024 ** 3
NOTE = 'This PC has 31.1 GiB of RAM; 5.0 GiB is free now. Largest users: cl 4.5 GiB.'
SLOW = ("ExMR needs up to about 12.0 GiB of RAM, more than is free, so Windows will move other programs' memory to disk "
        "and this job can run slower. ")


class FreeVramTest(unittest.TestCase):
    def test_reads_free_and_total_of_the_first_gpu(self):
        done = SimpleNamespace(stdout='14000, 16311\n8000, 8192\n')
        with patch.object(subprocess, 'run', return_value=done) as run:
            self.assertEqual(memory_fit.free_vram(), (14000 * 1024**2, 16311 * 1024**2))
        self.assertEqual(run.call_args.args[0][0], 'nvidia-smi')

    def test_no_nvidia_smi_or_no_gpu_is_unknown(self):
        with patch.object(subprocess, 'run', side_effect=OSError('missing')):
            self.assertEqual(memory_fit.free_vram(), (None, None))
        with patch.object(subprocess, 'run', return_value=SimpleNamespace(stdout='')):
            self.assertEqual(memory_fit.free_vram(), (None, None))


class WindowTest(unittest.TestCase):
    def setUp(self):
        from mocap.theme import configure_application
        if not getattr(WindowTest, 'configured', False):
            configure_application(APP);WindowTest.configured = True
        tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.home = Path(tmp.name)
        home = patch.object(paths, 'HOME', self.home);home.start();self.addCleanup(home.stop)
        from mocap.gui import MainWindow
        self.w = MainWindow()
        def close():
            self.w.busy = False;self.w.jobs = []
            self.w.close();self.w.deleteLater();APP.processEvents()
        self.addCleanup(close)
        self.w.show();APP.processEvents()
        self.w.model_choice.setCurrentIndex(self.w.model_choice.findData('exmr'))
        self.w.resource_safety.setChecked(True)

    def system(self, available, room):
        """Stand in for the PC's memory: free RAM, Windows allocation headroom and the sentence about its programs."""
        memory = SimpleNamespace(total=31.1 * GIB, available=available)
        return (patch('psutil.virtual_memory', return_value=memory), patch.object(processing_guard, 'commit_headroom', return_value=room),
                patch.object(memory_note, 'collect', return_value=NOTE))

    def begin(self, jobs, available=5 * GIB, room=40 * GIB):
        a, b, c = self.system(available, room)
        with a, b, c:memory_fit.begin(self.w, jobs)

    def seen(self):
        try:return json.loads((self.home/'config/local.json').read_text(encoding='utf-8')).get('memory_seen')
        except OSError:return None

    def test_a_reconstruction_with_too_little_free_ram_says_so_before_it_starts(self):
        self.begin([('prepare', []), ('exmr_preprocess', None)])
        self.assertEqual(self.w.status.text(), SLOW+NOTE)
        self.assertIn(SLOW+NOTE, self.w.log.toPlainText())

    def test_too_little_allocation_headroom_is_the_warning_that_wins(self):
        self.begin([('person', [])], room=9 * GIB)
        self.assertIn('ExMR needs Windows to allocate about 14.7 GiB, and Windows can allocate only 9.0 GiB more', self.w.status.text())

    def test_enough_memory_says_nothing(self):
        before = self.w.status.text()
        self.begin([('exmr_preprocess', None)], available=20 * GIB)
        self.assertEqual(self.w.status.text(), before);self.assertNotIn('needs', self.w.log.toPlainText())

    def test_a_job_without_motion_models_says_nothing(self):
        self.begin([('export', None)]);self.begin([('people', [])])
        self.assertNotIn('needs', self.w.log.toPlainText())

    def test_resource_safety_off_says_nothing(self):
        self.w.resource_safety.setChecked(False)
        self.begin([('exmr_preprocess', None)])
        self.assertNotIn('needs', self.w.log.toPlainText())

    def test_a_peak_seen_on_this_pc_raises_the_need(self):
        memory_fit.remember('exmr', 13.5 * GIB, 15 * GIB)
        self.begin([('exmr_preprocess', None)], available=13 * GIB)
        self.assertIn('ExMR needs up to about 13.5 GiB of RAM', self.w.status.text())

    def test_an_unreadable_system_does_not_keep_the_job_from_starting(self):
        with patch('psutil.virtual_memory', side_effect=OSError('denied')):
            memory_fit.begin(self.w, [('exmr_preprocess', None)])
        self.assertNotIn('needs', self.w.log.toPlainText())

    def test_start_jobs_checks_memory(self):
        a, b, c = self.system(5 * GIB, 40 * GIB)
        with a, b, c, patch.object(self.w, 'next_job'):
            self.w.start_jobs([('exmr_preprocess', ['--project', 'x'])])
        self.assertIn(SLOW+NOTE, self.w.log.toPlainText())

    def test_the_offline_guard_is_started_with_paging(self):
        self.w.process = Mock();self.w.process.processId.return_value = 7;self.w._job_resource_safety = True
        self.addCleanup(self.w.guard_timer.stop)
        with patch.object(processing_guard, 'ProcessingGuard') as guard:
            self.w.start_processing_guard()
        guard.assert_called_once_with(7, resource_safety=True, paging=True)

    def test_tight_memory_is_said_once_and_the_job_keeps_running(self):
        self.w.process = Mock();self.w.job_project = None;self.w._memory_peak = [0, 0];self.w._memory_tight_told = False
        self.w.processing_guard = SimpleNamespace(check=lambda: None, reason=None, tight=True, peak_ram=GIB, peak_commit=GIB)
        with patch.object(self.w, 'cancel_job') as cancel, patch.object(memory_note, 'collect', return_value=NOTE):
            self.w.check_processing_resources();self.w.check_processing_resources()
        cancel.assert_not_called();self.assertFalse(self.w.failed)
        self.assertEqual(self.w.status.text(), memory_fit.TIGHT+' '+NOTE)
        self.assertEqual(self.w.log.toPlainText().count(memory_fit.TIGHT), 1)

    def test_a_new_batch_can_say_tight_again(self):
        self.w._memory_tight_told = True
        self.begin([('export', None)])
        self.assertFalse(self.w._memory_tight_told)

    def test_the_peaks_of_every_stage_of_a_batch_are_kept(self):
        self.begin([('export', None)])
        for ram, commit in ((2 * GIB, 9 * GIB), (12 * GIB, 3 * GIB)):
            self.w.processing_guard = SimpleNamespace(tight=False, peak_ram=ram, peak_commit=commit)
            memory_fit.tick(self.w)
        self.assertEqual(self.w._memory_peak, [12 * GIB, 9 * GIB])

    def end(self, **state):
        w = self.w
        w._memory_peak = [12 * GIB, 15 * GIB];w._batch_has_motion = True
        w.jobs, w.busy, w.failed, w.cancelled, w.current_stage, w._batch_started_at = [], True, False, False, 'copy_bvh', None
        for name, value in state.items():setattr(w, name, value)
        w.next_job()

    def test_a_finished_reconstruction_teaches_this_pc_what_the_model_needs(self):
        self.end()
        self.assertEqual(self.seen(), {'exmr': {'ram': 12 * GIB, 'commit': 15 * GIB}})

    def test_a_failed_cancelled_or_motionless_batch_teaches_nothing(self):
        for state in (dict(failed=True), dict(cancelled=True), dict(_batch_has_motion=False)):
            self.end(**state)
            self.assertIsNone(self.seen(), state)


if __name__ == '__main__':
    unittest.main()
