"""The sequence document's library (the videos of the Project panel) and a block inserted at a place on the timeline."""
import json
import tempfile
import unittest
from pathlib import Path

from mocap.sequence import Media, Sequence


class LibraryTests(unittest.TestCase):
    def test_a_video_is_listed_once_with_its_length(self):
        s = Sequence()
        a = s.add_media('C:/v/a.mp4', 60.0);b = s.add_media('C:/v/b.mp4', 3600.0)
        self.assertEqual(s.library, [Media('C:/v/a.mp4', 60.0), Media('C:/v/b.mp4', 3600.0)])
        self.assertIs(s.add_media('C:/v/a.mp4', 61.0), a);self.assertEqual(a.length, 61.0)   # read again: the newer length
        self.assertEqual(len(s.library), 2);self.assertIs(s.media('C:/v/b.mp4'), b);self.assertIsNone(s.media('C:/v/c.mp4'))

    def test_the_same_file_written_another_way_is_the_same_video(self):
        s = Sequence();a = s.add_media('C:/v/a.mp4', 60.0)
        self.assertIs(s.add_media('C:\\v\\.\\a.mp4', 60.0), a)
        self.assertIs(s.media(Path('C:/v/x/../a.mp4')), a);self.assertEqual(len(s.library), 1)

    def test_a_block_puts_its_video_in_the_library(self):
        s = Sequence();s.add('a.mp4', 60.0);s.add('b.mp4', 30.0, 5.0, 10.0);s.add('a.mp4', 60.0, 10.0, 20.0)
        self.assertEqual(s.library, [Media('a.mp4', 60.0), Media('b.mp4', 30.0)])

    def test_removing_a_video_from_the_library_keeps_its_blocks(self):
        s = Sequence();a = s.add('a.mp4', 60.0);s.add_media('b.mp4', 30.0)
        self.assertTrue(s.remove_media('a.mp4'));self.assertFalse(s.remove_media('a.mp4'))
        self.assertEqual(s.library, [Media('b.mp4', 30.0)]);self.assertEqual(s.blocks, [a])

    def test_a_bad_length_is_refused(self):
        s = Sequence()
        for length in (0, -1.0, float('nan'), float('inf'), '60', True):
            with self.assertRaises(ValueError):s.add_media('a.mp4', length)
        self.assertEqual(s.library, [])

    def test_the_library_is_saved_only_when_it_holds_more_than_the_blocks(self):
        s = Sequence();s.add('a.mp4', 60.0);s.add('b.mp4', 30.0)
        self.assertNotIn('library', s.to_dict())
        s.add_media('c.mp4', 10.0)
        self.assertEqual(s.to_dict()['library'], [{'video': 'a.mp4', 'length': 60.0}, {'video': 'b.mp4', 'length': 30.0}, {'video': 'c.mp4', 'length': 10.0}])
        empty = Sequence();empty.add_media('a.mp4', 60.0)
        self.assertEqual(empty.to_dict()['library'], [{'video': 'a.mp4', 'length': 60.0}])

    def test_a_removed_video_of_a_block_stays_removed_after_a_save(self):
        s = Sequence();s.add('a.mp4', 60.0);s.add('b.mp4', 30.0);s.remove_media('a.mp4')
        self.assertEqual(s.to_dict()['library'], [{'video': 'b.mp4', 'length': 30.0}])
        self.assertEqual(Sequence.from_dict(s.to_dict()).library, [Media('b.mp4', 30.0)])

    def test_save_and_load_keep_the_library_and_its_order(self):
        with tempfile.TemporaryDirectory() as folder:
            s = Sequence();s.add_media('z.mp4', 5.0);s.add('a.mp4', 60.0);s.add_media('m.mp4', 7200.0)
            path = Path(folder)/'s.json';s.save(path)
            loaded = Sequence.load(path)
            self.assertEqual(loaded.library, [Media('z.mp4', 5.0), Media('a.mp4', 60.0), Media('m.mp4', 7200.0)])
            self.assertEqual([b.video for b in loaded.blocks], ['a.mp4'])

    def test_a_file_without_a_library_lists_the_videos_of_its_blocks(self):
        data = {'version': 1, 'blocks': [{'id': 'x', 'video': 'a.mp4', 'start': 0.0, 'end': 10.0, 'length': 60.0, 'folders': {}},
                                         {'id': 'y', 'video': 'b.mp4', 'start': 0.0, 'end': 5.0, 'length': 30.0, 'folders': {}},
                                         {'id': 'z', 'video': 'a.mp4', 'start': 20.0, 'end': 30.0, 'length': 60.0, 'folders': {}}]}
        self.assertEqual(Sequence.from_dict(data).library, [Media('a.mp4', 60.0), Media('b.mp4', 30.0)])

    def test_a_damaged_library_is_not_a_sequence_file(self):
        base = {'version': 1, 'blocks': []}
        for library in ('a.mp4', [{'video': 'a.mp4'}], [{'video': 5, 'length': 1.0}], [{'video': 'a.mp4', 'length': 0}], [['a.mp4', 1.0]]):
            with self.assertRaisesRegex(ValueError, 'not a sequence file'):Sequence.from_dict({**base, 'library': library})
        self.assertEqual(Sequence.from_dict({**base, 'library': []}).library, [])


class InsertTests(unittest.TestCase):
    def test_a_video_is_inserted_between_blocks(self):
        s = Sequence();a = s.add('a.mp4', 60.0);b = s.add('b.mp4', 30.0)
        c = s.insert(1, 'c.mp4', 10.0)
        self.assertEqual(s.blocks, [a, c, b]);self.assertEqual((c.start, c.end, c.length), (0.0, 10.0, 10.0))
        self.assertEqual(s.starts(), [0.0, 60.0, 70.0]);self.assertIn(Media('c.mp4', 10.0), s.library)
        d = s.insert(0, 'a.mp4', 60.0, 5.0, 15.0);e = s.insert(99, 'b.mp4', 30.0);f = s.insert(-3, 'b.mp4', 30.0)
        self.assertEqual(s.blocks, [f, d, a, c, b, e]);self.assertEqual((d.start, d.end), (5.0, 15.0))
        self.assertEqual(len({block.id for block in s.blocks}), 6)

    def test_inserting_between_two_linked_blocks_drops_their_link(self):
        s = Sequence();s.model = 'exmr';a = s.add('a.mp4', 60.0);b = s.add('b.mp4', 30.0);c = s.add('c.mp4', 30.0)
        first = s.link((a.id, 1), (b.id, 1));second = s.link((b.id, 1), (c.id, 1))
        s.insert(1, 'x.mp4', 10.0)
        self.assertEqual(s.links, [second]);self.assertEqual(s.dropped, [first])
        s.insert(0, 'y.mp4', 10.0);s.insert(len(s.blocks), 'z.mp4', 10.0)   # beside the linked pair, not between
        self.assertEqual(s.links, [second])

    def test_an_inserted_block_lets_the_block_after_it_link_by_itself_again(self):
        s = Sequence();s.model = 'exmr';a = s.add('a.mp4', 60.0);b = s.add('b.mp4', 30.0)
        s.link((a.id, 1), (b.id, 1));self.assertIn(b.id, s.decided)
        s.insert(1, 'x.mp4', 10.0)
        self.assertNotIn(b.id, s.decided)


if __name__ == '__main__':
    unittest.main()
