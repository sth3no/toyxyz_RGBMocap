"""What a model needs in RAM, the notes about short memory and the longest part of a long range that fits."""
import json
import tempfile
import unittest
from pathlib import Path

from mocap import memory_fit

GIB = 1024 ** 3
NOTE = 'This PC has 31.1 GiB of RAM; 5.0 GiB is free now. Largest users: cl 4.5 GiB.'


class NeedTest(unittest.TestCase):
    def test_exmr_starts_from_the_measured_numbers(self):
        self.assertEqual(memory_fit.need('exmr', {}), (12.0 * GIB, 14.7 * GIB))

    def test_a_larger_peak_seen_on_this_pc_replaces_the_starting_number(self):
        seen = {'exmr': {'ram': 13 * GIB, 'commit': 16 * GIB}}
        self.assertEqual(memory_fit.need('exmr', seen), (13 * GIB, 16 * GIB))

    def test_a_smaller_peak_seen_does_not_lower_it(self):
        self.assertEqual(memory_fit.need('exmr', {'exmr': {'ram': 2 * GIB, 'commit': 3 * GIB}}), (12.0 * GIB, 14.7 * GIB))

    def test_a_model_with_no_starting_commit_takes_the_seen_one(self):
        self.assertEqual(memory_fit.need('gvhmr', {}), (6.9 * GIB, None))
        self.assertEqual(memory_fit.need('gvhmr', {'gvhmr': {'ram': 5 * GIB, 'commit': 9 * GIB}}), (6.9 * GIB, 9 * GIB))

    def test_an_unknown_model_is_known_only_once_seen(self):
        self.assertIsNone(memory_fit.need('gemx', {}))
        self.assertEqual(memory_fit.need('gemx', {'gemx': {'ram': 8 * GIB, 'commit': 10 * GIB}}), (8 * GIB, 10 * GIB))

    def test_a_damaged_record_is_ignored(self):
        self.assertEqual(memory_fit.need('exmr', {'exmr': 'x'}), (12.0 * GIB, 14.7 * GIB))
        self.assertIsNone(memory_fit.need('gemx', {'gemx': {'ram': 'big'}}))


class SeenTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root = Path(tmp.name)

    def settings(self):return json.loads((self.root/'config/local.json').read_text(encoding='utf-8'))

    def test_remember_keeps_the_peak_beside_other_local_settings(self):
        (self.root/'config').mkdir();(self.root/'config/local.json').write_text(json.dumps({'ui_theme': 'dark'}), encoding='utf-8')
        memory_fit.remember('exmr', 12 * GIB, 15 * GIB, self.root)
        self.assertEqual(self.settings(), {'ui_theme': 'dark', 'memory_seen': {'exmr': {'ram': 12 * GIB, 'commit': 15 * GIB}}})
        self.assertEqual(memory_fit.seen(self.root), {'exmr': {'ram': 12 * GIB, 'commit': 15 * GIB}})

    def test_remember_keeps_the_largest_of_each(self):
        memory_fit.remember('exmr', 12 * GIB, 15 * GIB, self.root)
        memory_fit.remember('exmr', 13 * GIB, 14 * GIB, self.root)
        memory_fit.remember('gvhmr', 4 * GIB, 6 * GIB, self.root)
        self.assertEqual(memory_fit.seen(self.root), {'exmr': {'ram': 13 * GIB, 'commit': 15 * GIB}, 'gvhmr': {'ram': 4 * GIB, 'commit': 6 * GIB}})

    def test_a_peak_under_one_gib_is_not_a_reconstruction_and_is_not_kept(self):
        memory_fit.remember('exmr', GIB // 2, GIB // 2, self.root)
        self.assertEqual(memory_fit.seen(self.root), {})
        self.assertFalse((self.root/'config/local.json').exists())

    def test_an_unreadable_settings_file_is_kept_and_nothing_fails(self):
        (self.root/'config').mkdir();(self.root/'config/local.json').write_text('{broken', encoding='utf-8')
        memory_fit.remember('exmr', 12 * GIB, 15 * GIB, self.root)
        self.assertEqual((self.root/'config/local.json').read_text(encoding='utf-8'), '{broken')
        self.assertEqual(memory_fit.seen(self.root), {})


class NotesTest(unittest.TestCase):
    def test_enough_ram_and_headroom_needs_no_note(self):
        self.assertIsNone(memory_fit.start_note('ExMR', (12 * GIB, 14.7 * GIB), 16 * GIB, 40 * GIB, NOTE))

    def test_less_free_ram_than_the_model_needs_says_it_can_be_slower(self):
        self.assertEqual(memory_fit.start_note('ExMR', (12 * GIB, 14.7 * GIB), 5 * GIB, 40 * GIB, NOTE),
                         "ExMR needs up to about 12.0 GiB of RAM, more than is free, so Windows will move other programs' memory to disk "
                         'and this job can run slower. '+NOTE)

    def test_too_little_allocation_headroom_says_the_job_can_stop(self):
        self.assertEqual(memory_fit.start_note('ExMR', (12 * GIB, 14.7 * GIB), 5 * GIB, 9 * GIB, NOTE),
                         'ExMR needs Windows to allocate about 14.7 GiB, and Windows can allocate only 9.0 GiB more, so this job can stop '
                         'for memory. Close other programs or let Windows manage a larger page file. '+NOTE)

    def test_headroom_just_above_the_need_still_warns_because_the_guard_keeps_one_gib(self):
        self.assertIn('can stop for memory', memory_fit.start_note('ExMR', (12 * GIB, 14.7 * GIB), 16 * GIB, 15.2 * GIB, NOTE))
        self.assertIsNone(memory_fit.start_note('ExMR', (12 * GIB, 14.7 * GIB), 16 * GIB, 15.8 * GIB, NOTE))

    def test_unknown_headroom_or_commit_need_leaves_only_the_ram_check(self):
        self.assertIsNone(memory_fit.start_note('GVHMR', (6.9 * GIB, None), 8 * GIB, 2 * GIB, NOTE))
        self.assertIn('can run slower', memory_fit.start_note('ExMR', (12 * GIB, 14.7 * GIB), 5 * GIB, None, NOTE))

    def test_an_unknown_need_gives_no_note(self):
        self.assertIsNone(memory_fit.start_note('GEM-X', None, GIB, GIB, NOTE))

    def test_a_missing_system_sentence_leaves_no_trailing_space(self):
        self.assertTrue(memory_fit.start_note('ExMR', (12 * GIB, None), 5 * GIB, None, '').endswith('can run slower.'))
        self.assertEqual(memory_fit.tight_note(''), memory_fit.TIGHT)

    def test_the_tight_note_says_the_job_keeps_running(self):
        self.assertEqual(memory_fit.tight_note(NOTE),
                         "Memory is tight, so Windows is moving other programs' memory to disk. This job keeps running and can be slower. "+NOTE)


class LastStepTest(unittest.TestCase):
    MEASURED = {1800: .842, 5400: .994, 10800: 1.916, 18000: 4.285, 36000: 13.637}  # reserved GiB, experiments\memory-fit\results.md

    def test_the_estimate_is_never_far_under_what_was_measured(self):
        for frames, reserved in self.MEASURED.items():
            self.assertGreater(memory_fit.last_step_bytes(frames)/GIB, reserved*.94, frames)

    def test_the_estimate_is_close_at_the_long_end(self):
        self.assertAlmostEqual(memory_fit.last_step_bytes(36000)/GIB, 13.637, delta=.3)


class PartTest(unittest.TestCase):
    def fit(self, model='exmr', fps=30., width=1920, height=1080, people=1, free_vram=14 * GIB, total_vram=16 * GIB, free_disk=500e9):
        return memory_fit.part_seconds(model, fps, width, height, people, free_vram, total_vram, free_disk)

    def test_with_room_everywhere_a_part_is_five_minutes(self):
        self.assertEqual(self.fit(), (300., 'time'))
        self.assertEqual(memory_fit.part_frames('exmr', 30., 1920, 1080, 1, 14 * GIB, 16 * GIB, 500e9), (9000, 'time'))

    def test_little_free_video_memory_shortens_an_exmr_part(self):
        seconds, limit = self.fit(free_vram=2 * GIB, total_vram=8 * GIB)
        self.assertEqual(limit, 'video memory')
        self.assertEqual(seconds, 120.)
        self.assertLessEqual(memory_fit.last_step_bytes(round(seconds*30)), .7*(2 * GIB-600 * 1024**2))

    def test_a_part_is_never_shorter_than_ten_seconds(self):
        self.assertEqual(self.fit(free_vram=.7 * GIB, total_vram=8 * GIB), (10., 'video memory'))

    def test_temporary_frames_stay_within_a_quarter_of_the_free_disk(self):
        self.assertEqual(self.fit(free_disk=2e9), (80., 'disk'))
        self.assertEqual(self.fit(free_disk=2e9, people=2), (40., 'disk'))

    def test_the_tighter_limit_is_named(self):
        self.assertEqual(self.fit(free_vram=2 * GIB, total_vram=8 * GIB, free_disk=2e9), (80., 'disk'))

    def test_unknown_video_memory_and_disk_leave_only_the_five_minutes(self):
        self.assertEqual(self.fit(free_vram=None, total_vram=None, free_disk=None), (300., 'time'))

    def test_gvhmr_follows_the_apps_own_attention_estimate(self):
        from mocap.compute_policy import gvhmr_plan
        self.assertEqual(self.fit('gvhmr'), (300., 'time'))
        seconds, limit = self.fit('gvhmr', free_vram=3.8 * GIB, total_vram=8 * GIB)
        self.assertEqual(limit, 'video memory')
        self.assertTrue(10 <= seconds < 300)
        self.assertTrue(gvhmr_plan(round(seconds*30), 3.8 * GIB, 8 * GIB)['fits'])
        self.assertFalse(gvhmr_plan(round((seconds+10)*30), 3.8 * GIB, 8 * GIB)['fits'])

    def test_gvhmr_writes_no_frames_so_the_disk_does_not_limit_it(self):
        self.assertEqual(self.fit('gvhmr', free_disk=2e9), (300., 'time'))

    def test_flowhmr_keeps_the_sixty_seconds(self):
        self.assertEqual(self.fit('flowhmr'), (60., 'default'))

    def test_every_limit_has_words_for_the_dialog(self):
        for limit in ('time', 'video memory', 'disk', 'default'):
            self.assertTrue(memory_fit.LIMITS[limit])


if __name__ == '__main__':
    unittest.main()
