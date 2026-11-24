"""The desktop guard with paging: low physical RAM is not a stop while Windows can still allocate memory."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mocap.processing_guard import ProcessingGuard

GIB = 1024 ** 3
MIB = 1024 ** 2


def process(rss=0, private=0, children=()):
    return SimpleNamespace(pid=5, create_time=lambda: 1., children=lambda recursive: list(children),
                           cpu_times=lambda: SimpleNamespace(user=0., system=0.),
                           memory_info=lambda: SimpleNamespace(rss=rss, pagefile=private))


def guard(paging, worker=None):
    with patch('psutil.Process', return_value=worker or process()):
        return ProcessingGuard(5, now=0., resource_safety=True, paging=paging)


def check(g, available, commit, total=32 * GIB, now=1.):
    memory = SimpleNamespace(total=total, available=available)
    with patch('psutil.virtual_memory', return_value=memory), patch('mocap.processing_guard.commit_headroom', return_value=commit):
        return g.check(now=now)


class PagingGuardTest(unittest.TestCase):
    def test_low_ram_with_allocation_headroom_does_not_stop(self):
        g = guard(True)
        for _ in range(4):
            self.assertIsNone(check(g, 1 * GIB, 40 * GIB))
        self.assertIsNone(g.reason)
        self.assertTrue(g.tight)

    def test_ram_under_256_mib_with_headroom_does_not_stop(self):
        g = guard(True)
        self.assertIsNone(check(g, 10 * MIB, 28 * GIB))
        self.assertTrue(g.tight)

    def test_enough_ram_is_not_tight(self):
        g = guard(True)
        check(g, 1 * GIB, 40 * GIB)
        self.assertIsNone(check(g, 12 * GIB, 40 * GIB))
        self.assertFalse(g.tight)

    def test_headroom_under_one_gib_stops_on_the_second_sample(self):
        g = guard(True)
        self.assertIsNone(check(g, 12 * GIB, GIB // 2))
        message = check(g, 12 * GIB, GIB // 2)
        self.assertIn('Windows allocation headroom 0.50 GiB', message)
        self.assertEqual(g.reason, 'memory')

    def test_headroom_under_256_mib_stops_at_once(self):
        g = guard(True)
        self.assertIn('memory is low', check(g, 12 * GIB, 100 * MIB))
        self.assertEqual(g.reason, 'memory')

    def test_low_ram_between_two_low_headroom_samples_does_not_count(self):
        g = guard(True)
        check(g, 1 * GIB, GIB // 2)
        self.assertIsNone(check(g, 1 * GIB, 40 * GIB))
        self.assertIsNone(check(g, 1 * GIB, GIB // 2))

    def test_without_a_readable_headroom_low_ram_still_stops(self):
        g = guard(True)
        check(g, 1 * GIB, None)
        self.assertIn('memory is low', check(g, 1 * GIB, None))

    def test_without_paging_low_ram_stops_as_before(self):
        g = guard(False)
        self.assertIsNone(check(g, 1 * GIB, 40 * GIB))
        self.assertIn('Available RAM 1.00 GiB', check(g, 1 * GIB, 40 * GIB))
        self.assertEqual(g.reason, 'memory')
        self.assertIn('memory is low', check(guard(False), 10 * MIB, 40 * GIB))

    def test_the_stall_check_stays(self):
        g = guard(True)
        self.assertIn('180 seconds', check(g, 1 * GIB, 40 * GIB, now=1000.))
        self.assertEqual(g.reason, 'stall')


class PeakTest(unittest.TestCase):
    def test_keeps_the_largest_working_set_and_commit_of_the_worker_tree(self):
        child = process(rss=2 * GIB, private=3 * GIB)
        worker = process(rss=1 * GIB, private=1 * GIB, children=[child])
        g = guard(True, worker)
        check(g, 12 * GIB, 40 * GIB)
        self.assertEqual((g.peak_ram, g.peak_commit), (3 * GIB, 4 * GIB))
        worker.memory_info = lambda: SimpleNamespace(rss=GIB // 2, pagefile=GIB // 2)
        check(g, 12 * GIB, 40 * GIB)
        self.assertEqual((g.peak_ram, g.peak_commit), (3 * GIB, 4 * GIB))

    def test_a_process_without_readable_memory_is_skipped(self):
        worker = process()
        def denied(): raise OSError('denied')
        worker.memory_info = denied
        g = guard(True, worker)
        self.assertIsNone(check(g, 12 * GIB, 40 * GIB))
        self.assertEqual(g.peak_ram, 0)


if __name__ == '__main__':
    unittest.main()
