import json
import tempfile
import unittest
from pathlib import Path

from mocap.sequence import MIN_SECONDS, VERSION, Sequence


class SequenceTests(unittest.TestCase):
    def test_blocks_follow_one_another(self):
        s = Sequence(); a = s.add('a.mp4', 1800.0); b = s.add('b.mp4', 1200.0, 300.0, 330.0)
        self.assertEqual(s.starts(), [0.0, 1800.0]); self.assertEqual(s.total(), 1830.0)
        self.assertEqual(s.locate(1810.0), (b, 310.0)); self.assertEqual(s.locate(9999.0), (b, 330.0))
        self.assertEqual(s.position(b.id, 305.0), 1805.0)
        self.assertEqual(s.locate(0.0), (a, 0.0)); self.assertEqual(s.locate(1800.0), (b, 300.0))
        self.assertEqual(s.position(b.id, 5.0), 1800.0); self.assertEqual(s.position(b.id, 999.0), 1830.0)
        self.assertIsNone(Sequence().locate(1.0)); self.assertEqual(Sequence().total(), 0.0)

    def test_cut_splits_and_drops_results(self):
        s = Sequence(); a = s.add('a.mp4', 60.0, folders={'exmr': 'X'})
        first, second = s.cut(a.id, 20.0)
        self.assertEqual((first.id, first.start, first.end, second.start, second.end), (a.id, 0.0, 20.0, 20.0, 60.0))
        self.assertEqual((first.folders, second.folders), ({}, {})); self.assertEqual(s.blocks, [first, second])
        self.assertNotEqual(first.id, second.id); self.assertEqual(second.video, 'a.mp4')
        with self.assertRaisesRegex(ValueError, 'at least 4 frames'): s.cut(first.id, 0.05)
        with self.assertRaisesRegex(ValueError, 'at least 4 frames'): s.cut(second.id, 59.99)
        self.assertEqual(len(s.blocks), 2)

    def test_resize_clamps_and_forgets(self):
        s = Sequence(); a = s.add('a.mp4', 60.0, 10.0, 40.0, {'exmr': 'X'})
        s.resize(a.id, 10.0, 40.0); self.assertEqual(a.folders, {'exmr': 'X'})
        s.resize(a.id, 10.0 + 1e-8, 40.0); self.assertEqual(a.folders, {'exmr': 'X'})
        s.resize(a.id, -5.0, 999.0); self.assertEqual((a.start, a.end), (0.0, 60.0)); self.assertEqual(a.folders, {})
        with self.assertRaisesRegex(ValueError, 'at least 4 frames'): s.resize(a.id, 5.0, 5.05)
        self.assertEqual((a.start, a.end), (0.0, 60.0))

    def test_remember_takes_the_prepared_range(self):
        s = Sequence(); a = s.add('a.mp4', 60.0)
        s.remember(a.id, 'exmr', 'F', 10.033333, 40.0)
        self.assertEqual((a.start, a.end, a.folders), (10.033333, 40.0, {'exmr': 'F'}))
        s.remember(a.id, 'gvhmr', 'G', 10.033333, 40.0)
        self.assertEqual(a.folders, {'exmr': 'F', 'gvhmr': 'G'})
        s.remember(a.id, 'gvhmr', 'H', 11.0, 40.0)
        self.assertEqual((a.start, a.folders), (11.0, {'gvhmr': 'H'}))
        s.forget(a.id, 'gvhmr'); s.forget(a.id, 'nope'); self.assertEqual(a.folders, {})

    def test_two_blocks_of_one_video_are_independent(self):
        s = Sequence(); a = s.add('a.mp4', 60.0, 0.0, 30.0); b = s.add('a.mp4', 60.0, 30.0, 60.0)
        s.remember(a.id, 'exmr', 'F', 0.0, 30.0)
        self.assertEqual(b.folders, {}); self.assertEqual(a.folders, {'exmr': 'F'})

    def test_move_and_delete(self):
        s = Sequence(); a = s.add('a', 10.0); b = s.add('b', 10.0); c = s.add('c', 10.0)
        s.move(a.id, 99); self.assertEqual(s.blocks, [b, c, a])
        s.move(a.id, -3); self.assertEqual(s.blocks, [a, b, c])
        s.move(c.id, 1); self.assertEqual((s.index(c.id), s.index(b.id)), (1, 2))
        s.delete(b.id); self.assertEqual(s.blocks, [a, c])
        with self.assertRaises(KeyError): s.block(b.id)
        with self.assertRaises(KeyError): s.index(b.id)

    def test_plain(self):
        s = Sequence(); self.assertFalse(s.plain())
        a = s.add('a.mp4', 60.0); self.assertTrue(s.plain())
        s.cut(a.id, 20.0); self.assertFalse(s.plain())
        t = Sequence(); t.add('a.mp4', 60.0); t.add('b.mp4', 60.0); self.assertFalse(t.plain())
        u = Sequence(); u.add('a.mp4', 60.0, 1.0, 60.0); self.assertFalse(u.plain())

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sub' / 'seq.json'
            s = Sequence(); s.add('a.mp4', 60.0, 1.0, 30.0, {'exmr': 'X'}); s.add('b.mp4', 10.0)
            self.assertEqual(s.to_dict()['version'], VERSION)
            s.save(path); self.assertEqual(s.path, path)
            loaded = Sequence.load(path)
            self.assertEqual(loaded.blocks, s.blocks); self.assertEqual(loaded.path, path)
            loaded.add('c.mp4', 5.0); loaded.save(); self.assertEqual(len(Sequence.load(path).blocks), 3)
            self.assertEqual(MIN_SECONDS, 4 / 30)

    def test_bad_files(self):
        good = {'id': 'x', 'video': 'a', 'start': 0, 'end': 5, 'length': 9, 'folders': {}}
        with self.assertRaisesRegex(ValueError, 'newer version'): Sequence.from_dict({'version': 2, 'blocks': []})
        for bad in ({'version': 1, 'blocks': 'x'}, {'version': 1, 'blocks': [dict(good, end=0)]},
                    {'version': 1, 'blocks': [dict(good, start='a')]}, {'version': 1, 'blocks': [dict(good, folders=3)]},
                    {'version': 1, 'blocks': [5]}, {'blocks': []}, [], 'text'):
            with self.assertRaisesRegex(ValueError, 'not a sequence file'): Sequence.from_dict(bad)
        self.assertEqual(Sequence.from_dict({'version': 1, 'blocks': [good]}).blocks[0].end, 5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 's.json'; path.write_text('garbage {', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'not a sequence file'): Sequence.load(path)
            path.write_text(json.dumps({'version': 2, 'blocks': []}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'newer version'): Sequence.load(path)


if __name__ == '__main__':
    unittest.main()
