import json
import tempfile
import unittest
from pathlib import Path

from mocap.sequence import DEFAULT_GAP, DEFAULT_METHOD, METHODS, MIN_SECONDS, VERSION, Link, Sequence


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
        u = Sequence(); u.add('a.mp4', 60.0, 1.0, 60.0); self.assertTrue(u.plain())   # one block, whatever its range

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

    def test_method_keys_are_those_of_hidden_ranges(self):
        from mocap import hidden_ranges_state
        from mocap.hidden_ranges import METHODS as FILL
        self.assertEqual(METHODS, ('smooth', 'auto')); self.assertEqual(DEFAULT_METHOD, 'smooth'); self.assertEqual(DEFAULT_GAP, 1.0)
        for key in METHODS:
            self.assertIn(key, FILL); choices = {}; hidden_ranges_state.choose(choices, 'g', key)
            self.assertEqual(choices['g']['method'], key)

    def test_link_only_to_the_block_before(self):
        s, a, b, c = _three()
        link = s.link((a.id, 1), (b.id, 7))
        self.assertEqual((link.source, link.target, link.method, link.prints, link.built),
                         ((a.id, 1), (b.id, 7), 'smooth', (None, None), None))
        self.assertEqual(s.links, [link])
        for source, target in (((a.id, 1), (c.id, 2)), ((b.id, 7), (a.id, 1)), ((a.id, 1), (a.id, 2))):
            with self.assertRaisesRegex(ValueError, 'only a person of the block before'): s.link(source, target)
        self.assertEqual(s.links, [link])

    def test_one_link_in_and_one_out(self):
        s, a, b, c = _three()
        s.link((a.id, 1), (b.id, 7)); second = s.link((a.id, 2), (b.id, 7))
        self.assertEqual(s.links, [second]); self.assertIsNone(s.link_out(a.id, 1)); self.assertEqual(s.link_into(b.id, 7), second)
        third = s.link((a.id, 2), (b.id, 8), 'auto', ('P', 'Q'))
        self.assertEqual(s.links, [third]); self.assertEqual(s.link_out(a.id, 2), third); self.assertIsNone(s.link_into(b.id, 7))
        self.assertEqual((third.method, third.prints), ('auto', ('P', 'Q')))
        with self.assertRaisesRegex(ValueError, 'Unknown'): s.link((a.id, 1), (b.id, 9), 'copy')
        self.assertEqual(s.links, [third])

    def test_chain_from_any_member(self):
        s, a, b, c = _three()
        s.link((a.id, 1), (b.id, 7)); s.link((b.id, 7), (c.id, 2))
        row = [(a.id, 1), (b.id, 7), (c.id, 2)]
        self.assertEqual(s.chain(a.id, 1), row); self.assertEqual(s.chain(b.id, 7), row); self.assertEqual(s.chain(c.id, 2), row)
        self.assertEqual(s.chain(b.id, 3), [(b.id, 3)])
        self.assertEqual(len(s.links_touching(b.id)), 2); self.assertEqual(s.links_touching(a.id), [s.link_into(b.id, 7)])

    def test_gap_moves_later_blocks(self):
        s = Sequence(); a = s.add('a', 10.0); b = s.add('b', 5.0)
        self.assertEqual(s.starts(), [0.0, 10.0]); self.assertEqual(s.total(), 15.0); self.assertEqual(s.gap(b.id), 0.0)
        s.link((a.id, 1), (b.id, 7))
        self.assertEqual(s.gaps, {b.id: 1.0}); self.assertEqual(s.gap(b.id), 1.0); self.assertEqual(s.gap(a.id), 0.0)
        self.assertEqual(s.starts(), [0.0, 11.0]); self.assertEqual(s.total(), 16.0)
        s.set_gap(b.id, 3.0)
        self.assertEqual(s.starts(), [0.0, 13.0]); self.assertEqual(s.total(), 18.0)
        self.assertEqual(s.locate(10.5), (a, a.end)); self.assertEqual(s.locate(12.99), (a, a.end))
        self.assertEqual(s.locate(13.0), (b, 0.0)); self.assertEqual(s.locate(14.0), (b, 1.0)); self.assertEqual(s.locate(99.0), (b, 5.0))
        self.assertEqual(s.position(b.id, b.start), 13.0); self.assertEqual(s.position(a.id, 10.0), 10.0)
        s.link((a.id, 2), (b.id, 8)); self.assertEqual(s.gap(b.id), 3.0)   # all links across one pair share its gap
        self.assertEqual(s.locate(-1.0), (a, 0.0))

    def test_gap_needs_a_frame(self):
        s = Sequence(); a = s.add('a', 10.0); b = s.add('b', 5.0); s.link((a.id, 1), (b.id, 7))
        with self.assertRaisesRegex(ValueError, 'at least one frame'): s.set_gap(b.id, 0.01)
        self.assertEqual(s.gap(b.id), 1.0); s.set_gap(b.id, 1 / 30); self.assertEqual(s.gap(b.id), 1 / 30)

    def test_set_method(self):
        s = Sequence(); a = s.add('a', 10.0); b = s.add('b', 5.0); link = s.link((a.id, 1), (b.id, 7))
        s.set_method((b.id, 7), 'auto'); self.assertEqual(link.method, 'auto')
        with self.assertRaisesRegex(ValueError, 'Unknown'): s.set_method((b.id, 7), 'copy')
        self.assertEqual(link.method, 'auto')

    def test_edits_drop_links(self):
        def linked():
            s, a, b, c = _three(); s.link((a.id, 1), (b.id, 7)); s.link((b.id, 7), (c.id, 2)); return s, a, b, c

        def check(s, gone, kept):
            self.assertEqual([(l.source, l.target) for l in s.dropped], gone)
            self.assertEqual([(l.source, l.target) for l in s.links], kept)
            ids = {x.id for x in s.blocks}
            for l in s.dropped:
                if l.target[0] in ids and not any(k.target[0] == l.target[0] for k in s.links):
                    self.assertEqual(s.gap(l.target[0]), 0.0)

        s, a, b, c = linked(); s.delete(b.id); check(s, [((a.id, 1), (b.id, 7)), ((b.id, 7), (c.id, 2))], [])
        self.assertEqual(s.starts(), [0.0, 10.0])
        s, a, b, c = linked(); s.delete(a.id); check(s, [((a.id, 1), (b.id, 7))], [((b.id, 7), (c.id, 2))])
        s, a, b, c = linked(); first, second = s.cut(c.id, 5.0); check(s, [((b.id, 7), (c.id, 2))], [((a.id, 1), (b.id, 7))])
        self.assertEqual((s.gap(first.id), s.gap(second.id)), (0.0, 0.0))
        s, a, b, c = linked(); s.cut(a.id, 5.0); check(s, [((a.id, 1), (b.id, 7))], [((b.id, 7), (c.id, 2))])
        s, a, b, c = linked(); s.resize(c.id, 0.0, 10.0); check(s, [], [((a.id, 1), (b.id, 7)), ((b.id, 7), (c.id, 2))])
        s.resize(c.id, 1.0, 10.0); check(s, [((b.id, 7), (c.id, 2))], [((a.id, 1), (b.id, 7))]); self.assertEqual(s.gap(c.id), 0.0)
        s, a, b, c = linked(); s.remember(a.id, 'exmr', 'F', 0.0, 10.0); check(s, [], [((a.id, 1), (b.id, 7)), ((b.id, 7), (c.id, 2))])
        s.remember(a.id, 'exmr', 'F', 1.0, 10.0); check(s, [((a.id, 1), (b.id, 7))], [((b.id, 7), (c.id, 2))])
        self.assertEqual((s.gap(b.id), s.gap(c.id)), (0.0, 1.0))
        s, a, b, c = linked(); s.move(a.id, 1); self.assertEqual(s.blocks, [b, a, c])   # b and c are no longer neighbours
        check(s, [((a.id, 1), (b.id, 7)), ((b.id, 7), (c.id, 2))], []); self.assertEqual(s.starts(), [0.0, 10.0, 20.0])
        s, a, b, c = linked(); s.move(c.id, 2); check(s, [], [((a.id, 1), (b.id, 7)), ((b.id, 7), (c.id, 2))])   # unchanged order
        s.move(c.id, 0); self.assertEqual(s.blocks, [c, a, b]); check(s, [((b.id, 7), (c.id, 2))], [((a.id, 1), (b.id, 7))])
        self.assertEqual((s.gap(a.id), s.gap(b.id)), (0.0, 1.0))
        s.dropped.clear(); s.delete(a.id); self.assertEqual(len(s.dropped), 1); self.assertEqual(s.links, [])

    def test_unlink_keeps_the_decision(self):
        s = Sequence(); a = s.add('a', 10.0); b = s.add('b', 5.0)
        self.assertEqual(s.decided, set()); link = s.link((a.id, 1), (b.id, 7)); self.assertEqual(s.decided, {b.id})
        self.assertEqual(s.unlink((b.id, 7)), link); self.assertEqual(s.links, []); self.assertEqual(s.decided, {b.id})
        self.assertIsNone(s.unlink((b.id, 7))); self.assertEqual(s.gap(b.id), 0.0); self.assertEqual(s.dropped, [])
        self.assertEqual(s.starts(), [0.0, 10.0])

    def test_links_round_trip_and_old_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'seq.json'
            s, a, b, c = _three()
            first = s.link((a.id, 1), (b.id, 'p7'), prints=('x', 'y'))
            first.built = {'copy': 'takes/t2', 'take': 'C:/b/takes/t0', 'stamp': [123, 456789], 'gap': 30, 'method': 'smooth'}
            s.link((b.id, 'p7'), (c.id, 2), 'auto'); s.set_gap(c.id, 2.5); s.decided.add(a.id)
            s.save(path); loaded = Sequence.load(path)
            self.assertEqual(loaded.links, s.links); self.assertEqual(loaded.gaps, s.gaps); self.assertEqual(loaded.decided, s.decided)
            self.assertEqual(loaded.links[0].built, first.built); self.assertEqual(loaded.links[0].prints, ('x', 'y'))
            self.assertIsInstance(loaded.links[0].source[1], int); self.assertIsInstance(loaded.links[0].target[1], str)
            self.assertEqual(loaded.starts(), s.starts()); self.assertEqual(loaded.dropped, [])
        plain = Sequence(); plain.add('a', 10.0); self.assertEqual(set(plain.to_dict()), {'version', 'blocks'})
        good = {'id': 'x', 'video': 'a', 'start': 0, 'end': 5, 'length': 9, 'folders': {}}
        old = Sequence.from_dict({'version': 1, 'blocks': [good, dict(good, id='y')]})
        self.assertEqual((old.links, old.gaps, old.decided, old.starts()), ([], {}, set(), [0.0, 5.0]))
        link = {'source': ['x', 1], 'target': ['y', 2], 'method': 'smooth', 'prints': [None, None]}
        data = {'version': 1, 'blocks': [good, dict(good, id='y')],
                'links': [link, dict(link, source=['x', 3], target=['z', 2]), dict(link, source=['x', 4], target=['y', 5], method='copy')],
                'gaps': {'y': 2.0}, 'decided': ['y', 'z']}
        loaded = Sequence.from_dict(data)
        self.assertEqual(loaded.links, [Link(('x', 1), ('y', 2), 'smooth', (None, None), None)])
        self.assertEqual((loaded.gap('y'), loaded.decided, loaded.dropped), (2.0, {'y'}, []))
        for bad in ({'links': {}}, {'gaps': []}, {'decided': 'y'}, {'links': [5]}, {'links': [dict(link, source='x')]},
                    {'links': [dict(link, target=['y', True])]}, {'links': [dict(link, built=3)]}, {'gaps': {'y': 'a'}}):
            with self.assertRaisesRegex(ValueError, 'not a sequence file'): Sequence.from_dict(dict(data, **bad))

    def test_review_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'seq.json'
            s = Sequence(); a = s.add('a', 10.0); b = s.add('b', 5.0)
            review = {'start': 1.0, 'end': 4.0, 'result': None}; s.set_review(b.id, review)
            s.link((a.id, 1), (b.id, 7)); self.assertEqual(b.review, review)
            s.unlink((b.id, 7)); self.assertEqual(b.review, review)
            s.link((a.id, 1), (b.id, 7)); s.save(path); loaded = Sequence.load(path)
            self.assertEqual(loaded.block(b.id).review, review); self.assertEqual(len(loaded.links), 1)


def _three():
    s = Sequence(); return s, s.add('a', 10.0), s.add('b', 10.0), s.add('c', 10.0)


if __name__ == '__main__':
    unittest.main()
