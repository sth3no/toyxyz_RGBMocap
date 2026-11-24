"""The low-memory stop says how much RAM the PC has, how much is free and which programs hold the most."""
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

from mocap import memory_note, paths, processing_guard  # imported here: patch.dict of sys.modules would drop a module first imported inside it

APP = QApplication.instance() or QApplication([])
GIB = 1024 ** 3


class LargestTest(unittest.TestCase):
    def test_groups_by_name_and_sums_resident_memory(self):
        rows = [(1, 'chrome.exe', 1 * GIB), (2, 'chrome.exe', 2 * GIB), (3, 'Code.exe', 1.5 * GIB)]
        self.assertEqual(memory_note.largest(rows), [('chrome', 3 * GIB), ('Code', 1.5 * GIB)])

    def test_names_differing_only_in_case_are_one_program(self):
        rows = [(1, 'Zen.exe', GIB), (2, 'zen.exe', GIB)]
        self.assertEqual(memory_note.largest(rows), [('Zen', 2 * GIB)])

    def test_the_apps_own_processes_are_named_rgb_mocap(self):
        rows = [(10, 'python.exe', 2 * GIB), (11, 'python.exe', 3 * GIB), (12, 'python.exe', 1 * GIB)]
        self.assertEqual(memory_note.largest(rows, app_pids={10, 11}), [('RGB MoCap', 5 * GIB), ('python', 1 * GIB)])

    def test_at_most_four_largest_first_and_small_ones_left_out(self):
        rows = [(i, f'p{i}', (i + 1) * GIB) for i in range(6)] + [(9, 'tiny', 0.2 * GIB)]
        self.assertEqual([name for name, _ in memory_note.largest(rows)], ['p5', 'p4', 'p3', 'p2'])
        self.assertEqual(memory_note.largest([(1, 'tiny', 0.29 * GIB)]), [])

    def test_missing_names_and_sizes_are_skipped(self):
        self.assertEqual(memory_note.largest([(1, None, 4 * GIB), (2, 'a', None), (3, '', 4 * GIB)]), [])


class SentenceTest(unittest.TestCase):
    def test_says_total_free_and_the_largest_users(self):
        text = memory_note.sentence(31.1 * GIB, 2.1 * GIB, [('cl', 4.5 * GIB), ('dotnet', 3.9 * GIB), ('zen', 2.1 * GIB)])
        self.assertEqual(text, 'This PC has 31.1 GiB of RAM; 2.1 GiB is free now. Largest users: cl 4.5 GiB, dotnet 3.9 GiB, zen 2.1 GiB.')

    def test_without_large_users_only_the_ram_is_named(self):
        self.assertEqual(memory_note.sentence(16 * GIB, 0.5 * GIB, []), 'This PC has 16.0 GiB of RAM; 0.5 GiB is free now.')

    def test_a_stop_for_allocation_headroom_with_free_ram_says_so(self):
        # The guard also stops below 1 GiB of Windows allocation headroom (page file full) while RAM is free.
        self.assertEqual(memory_note.sentence(32 * GIB, 12 * GIB, [], commit=0.4 * GIB),
                         'This PC has 32.0 GiB of RAM; 12.0 GiB is free now, but Windows can allocate only 0.4 GiB more.')
        self.assertEqual(memory_note.sentence(32 * GIB, 1 * GIB, [], commit=0.4 * GIB),
                         'This PC has 32.0 GiB of RAM; 1.0 GiB is free now, and Windows can allocate only 0.4 GiB more.')
        self.assertEqual(memory_note.sentence(32 * GIB, 1 * GIB, [], commit=8 * GIB), 'This PC has 32.0 GiB of RAM; 1.0 GiB is free now.')


class CollectTest(unittest.TestCase):
    def test_reads_memory_and_processes(self):
        fake = SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(total=32 * GIB, available=2 * GIB),
            process_iter=lambda attrs: iter([SimpleNamespace(pid=1, info={'name': 'cl.exe', 'memory_info': SimpleNamespace(rss=4 * GIB)}),
                                             SimpleNamespace(pid=os.getpid(), info={'name': 'python.exe', 'memory_info': SimpleNamespace(rss=GIB)})]),
            Process=lambda pid: SimpleNamespace(children=lambda recursive: []))
        with patch.dict('sys.modules', psutil=fake), patch('mocap.processing_guard.commit_headroom', return_value=None):
            self.assertEqual(memory_note.collect(), 'This PC has 32.0 GiB of RAM; 2.0 GiB is free now. Largest users: cl 4.0 GiB, RGB MoCap 1.0 GiB.')
        with patch.dict('sys.modules', psutil=fake), patch('mocap.processing_guard.commit_headroom', return_value=GIB // 2):
            self.assertIn('2.0 GiB is free now, and Windows can allocate only 0.5 GiB more.', memory_note.collect())

    def test_a_failing_psutil_gives_no_note(self):
        def broken(): raise OSError('denied')
        with patch.dict('sys.modules', psutil=SimpleNamespace(virtual_memory=broken)):
            self.assertEqual(memory_note.collect(), '')


class GuardReasonTest(unittest.TestCase):
    def guard(self, available, idle):
        from mocap.processing_guard import ProcessingGuard
        guard = ProcessingGuard.__new__(ProcessingGuard)
        guard.resource_safety = True;guard.low_samples = 1;guard.cpu = guard.activity_cpu = 0.;guard.cpu_samples = {}
        guard.process = SimpleNamespace(pid=5, create_time=lambda: 1., children=lambda recursive: [],
                                        cpu_times=lambda: SimpleNamespace(user=0., system=0.))
        guard.created = 1.;guard.last_output = guard.last_cpu_change = 0.
        memory = SimpleNamespace(total=32 * GIB, available=available)
        with patch('psutil.virtual_memory', return_value=memory), patch('mocap.processing_guard.commit_headroom', return_value=None):
            message = guard.check(now=idle)
        return guard, message

    def test_a_memory_stop_reports_memory(self):
        guard, message = self.guard(GIB, 0.)
        self.assertTrue(message);self.assertEqual(guard.reason, 'memory')

    def test_a_stall_stop_reports_stall(self):
        guard, message = self.guard(16 * GIB, 1000.)
        self.assertTrue(message);self.assertEqual(guard.reason, 'stall')


class WindowTest(unittest.TestCase):
    def setUp(self):
        from mocap.theme import configure_application
        if not getattr(WindowTest, 'configured', False):
            configure_application(APP);WindowTest.configured = True
        tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        home = patch.object(paths, 'HOME', Path(tmp.name));home.start();self.addCleanup(home.stop)
        from mocap.gui import MainWindow
        self.w = MainWindow()
        def close():
            self.w.close();self.w.deleteLater();APP.processEvents()
        self.addCleanup(close)
        self.w.show();APP.processEvents()

    def stop(self, reason, message):
        """Stand in for a running job whose guard has just decided to stop it."""
        self.w.process = Mock();self.w.job_project = None
        self.w.processing_guard = SimpleNamespace(check=lambda: message, reason=reason)
        note = 'This PC has 31.1 GiB of RAM; 2.1 GiB is free now. Largest users: cl 4.5 GiB.'
        with patch.object(self.w, 'cancel_job'), patch.object(memory_note, 'collect', return_value=note) as collect:
            self.w.check_processing_resources();APP.processEvents()
        return note, collect

    def test_a_memory_stop_explains_itself_in_status_tooltip_and_console(self):
        note, collect = self.stop('memory', 'Processing stopped because available system memory is low.')
        self.assertEqual(collect.call_count, 1)
        self.assertIn(note, self.w.status.text())
        self.assertIn('memory is low', self.w.status.text())
        self.assertEqual(self.w.status.toolTip(), self.w.status.text())
        self.assertIn(note, self.w.log.toPlainText())
        self.assertTrue(self.w.log.isVisible())

    def test_a_stall_stop_has_no_memory_note(self):
        note, collect = self.stop('stall', 'Processing stopped after 180 seconds without output or CPU activity.')
        collect.assert_not_called()
        self.assertNotIn(note, self.w.status.text());self.assertNotIn(note, self.w.log.toPlainText())
        self.assertIn('180 seconds', self.w.status.text())

    def test_the_status_tooltip_follows_every_text(self):
        self.w.status.setText('Something long happened.')
        self.assertEqual(self.w.status.toolTip(), 'Something long happened.')


if __name__ == '__main__':
    unittest.main()
