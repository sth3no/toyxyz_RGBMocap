"""The Project panel, the tool strip and the sequence timeline working together in the real main window, offscreen:
videos are imported into the library, dragged to the timeline and cut with the Razor. Needs PySide6 and ffmpeg."""
import json
import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest

import test_sequence_window as base   # its window with two small real videos; not its tests by name
from mocap import sequence_ui, sequence_view
from mocap.sequence import Media, Sequence

APP = base.APP
pump = base.pump


def dragged(*videos):
    mime = QMimeData();mime.setData(sequence_view.MIME, json.dumps([str(v) for v in videos]).encode('utf-8'));return mime


@unittest.skipUnless(base.FFMPEG.is_file(), 'needs the bundled ffmpeg')
class ProjectCase(unittest.TestCase):
    setUp = base.SequenceWindowTest.setUp
    ready = base.SequenceWindowTest.ready
    open_first = base.SequenceWindowTest.open_first
    two = base.SequenceWindowTest.two
    ranges = base.SequenceWindowTest.ranges
    manifest = base.SequenceWindowTest.manifest
    folder = base.SequenceWindowTest.folder
    tracked = base.SequenceWindowTest.tracked
    loaded = base.SequenceWindowTest.loaded
    busy = base.SequenceWindowTest.busy

    def videos(self):return [Path(m.video) for m in self.c.sequence.library]

    def blocks(self):return [Path(b.video).name for b in self.c.sequence.blocks]

    def saved(self):return json.loads(self.c.sequence.path.read_text(encoding='utf-8'))


class LayoutTest(ProjectCase):
    def test_the_timeline_panel_holds_the_project_panel_the_tools_and_the_timeline_side_by_side(self):
        c, w = self.c, self.w
        for widget in (c.library, c.tools, c.stack):
            self.assertTrue(widget.isVisible());self.assertTrue(w.timeline_panel.isAncestorOf(widget))
        left = lambda widget:widget.mapTo(w.timeline_panel, QPoint(0, 0)).x()
        self.assertLess(left(c.library), left(c.tools));self.assertLess(left(c.tools), left(c.stack))
        self.assertGreater(c.stack.width(), c.library.width())
        self.assertEqual(c.library.count.text(), '');self.assertFalse(c.library.hint.isHidden())

    def test_the_cut_button_shows_the_razors_scissors(self):
        from mocap.sequence_tools import picture
        c = self.c;side = c.cut_button.iconSize().width()
        self.assertFalse(c.cut_button.icon().isNull());self.assertEqual(c.cut_button.text(), 'Cut')
        self.assertEqual(c.cut_button.icon().pixmap(side * 2, side * 2).toImage().size(), picture('razor', side * 2, '#c8c8c8').toImage().size())
        image = picture('razor', 32, '#c8c8c8').toImage()   # two round handles on the left, open blades on the right
        inked = lambda x, y:image.pixelColor(x, y).alpha() > 60
        self.assertFalse(inked(8, 8));self.assertFalse(inked(8, 24))   # the holes of the handles
        self.assertTrue(any(inked(x, 8) for x in range(2, 7)) and any(inked(x, 24) for x in range(2, 7)))
        self.assertTrue(any(inked(27, y) for y in range(3, 9)) and any(inked(27, y) for y in range(24, 30)))   # two blade tips

    def test_the_tools_work_on_the_sequence_timeline_only(self):
        c = self.c
        self.assertTrue(c.at_clip_level());self.assertFalse(c.tools.isEnabled())
        self.two()
        self.assertFalse(c.at_clip_level());self.assertTrue(c.tools.isEnabled())
        c.open_block(c.sequence.blocks[0].id);self.assertFalse(c.tools.isEnabled())
        c.back();self.assertTrue(c.tools.isEnabled())

    def test_a_picked_tool_is_the_timelines_tool_and_a_key_shows_in_the_strip(self):
        self.two();c = self.c
        QTest.mouseClick(c.tools.buttons['razor'], Qt.LeftButton)
        self.assertEqual(c.view.tool, 'razor')
        c.view.setFocus();QTest.keyClick(c.view, Qt.Key_H)
        self.assertTrue(c.tools.buttons['hand'].isChecked());self.assertEqual(c.tools.tool, 'hand')

    def test_another_model_has_no_project_panel_actions(self):
        c, w = self.c, self.w;self.two()
        w.model_choice.setCurrentIndex(w.model_choice.findData('gemx'));APP.processEvents()
        self.assertFalse(c.tools.isEnabled());self.assertFalse(c.library.isEnabled())
        self.assertIsNone(c.place([self.v2]));self.assertEqual(w.status.text(), base.OTHER)


class ImportTest(ProjectCase):
    def test_imported_videos_are_listed_and_the_empty_timeline_waits_for_them(self):
        c = self.c
        c.import_videos([self.v1, self.v2])
        self.assertEqual(self.videos(), [self.v1, self.v2]);self.assertEqual(c.sequence.blocks, [])
        self.assertEqual([c.library.list.item(i).text() for i in range(2)], ['one.mp4\n0:03', 'two.mp4\n0:03'])
        self.assertEqual(c.library.count.text(), '2 videos')
        self.assertFalse(c.at_clip_level());self.assertTrue(c.view.isVisible())   # the place to drag them to
        self.assertEqual(self.saved()['library'], [{'video': str(self.v1), 'length': 3.0}, {'video': str(self.v2), 'length': 3.0}])
        self.assertEqual(self.w.status.text(), '2 videos added to the Project panel. Drag one to the timeline.')

    def test_a_video_imported_twice_is_listed_once(self):
        c = self.c;c.import_videos([self.v1]);c.import_videos([self.v1, self.v2])
        self.assertEqual(self.videos(), [self.v1, self.v2])
        self.assertEqual(self.w.status.text(), '1 video added to the Project panel. Drag one to the timeline.')

    def test_a_file_that_is_not_a_video_is_named_and_left_out(self):
        c = self.c;bad = self.base/'notes.mp4';bad.write_text('not a video', encoding='utf-8')
        c.import_videos([bad, self.v1])
        self.assertEqual(self.videos(), [self.v1])
        self.assertIn('Cannot add notes.mp4:', self.w.status.text());self.assertIn('Cannot add notes.mp4:', self.w.log.toPlainText())

    def test_the_import_button_asks_for_several_files(self):
        c = self.c
        with mock.patch.object(sequence_ui.QFileDialog, 'getOpenFileNames', return_value=([str(self.v1), str(self.v2)], '')) as dialog:
            QTest.mouseClick(c.library.import_button, Qt.LeftButton)
        dialog.assert_called_once();self.assertEqual(self.videos(), [self.v1, self.v2])

    def test_files_dropped_on_the_project_panel_are_imported(self):
        c = self.c;c.library.imported.emit([str(self.v2)])
        self.assertEqual(self.videos(), [self.v2])

    def test_one_imported_video_is_remembered_in_a_file(self):
        c = self.c;c.import_videos([self.v1])
        self.assertEqual(self.videos(), [self.v1]);self.assertIsNotNone(c.sequence.path)   # the library holds more than the blocks

    def test_importing_is_allowed_while_a_job_runs(self):
        self.open_first();c = self.c;self.busy()
        c.import_videos([self.v2])
        self.assertEqual(self.videos(), [self.v1, self.v2]);self.assertTrue(c.at_clip_level())

    def test_open_video_lists_its_video(self):
        self.open_first()
        self.assertEqual(self.videos(), [self.v1]);self.assertIn('On the timeline', self.c.library.list.item(0).toolTip())
        self.assertIsNone(self.c.sequence.path)   # one plain video, as before

    def test_a_saved_sequence_opens_with_its_library(self):
        c = self.c;c.import_videos([self.v1, self.v2]);c.place([self.v2]);self.ready(self.v2);path = c.sequence.path
        c.new();self.assertNotEqual(c.sequence.path, path)
        self.assertTrue(c.open(path))
        self.assertEqual(self.videos(), [self.v1, self.v2]);self.assertEqual(self.blocks(), ['two.mp4'])
        self.assertEqual(c.library.count.text(), '2 videos')
        self.assertIn('Not on the timeline', c.library.list.item(0).toolTip());self.assertIn('On the timeline', c.library.list.item(1).toolTip())


class PlaceTest(ProjectCase):
    def test_the_first_video_placed_on_the_timeline_is_loaded(self):
        c = self.c;c.import_videos([self.v1, self.v2])
        blocks = c.place([self.v2], 0);self.ready(self.v2)
        self.assertEqual(self.blocks(), ['two.mp4']);self.assertEqual(c.current, blocks[0].id)
        self.assertEqual((blocks[0].start, blocks[0].end), (0., 3.));self.assertFalse(c.at_clip_level())
        self.assertIsNone(c.view.view_span)   # the first block is shown whole

    def test_a_video_is_placed_between_the_blocks_and_the_loaded_block_stays(self):
        first, second = self.two();c = self.c;current = c.current
        c.view.show_block(second.id);span = c.view.span()
        placed, = c.place([self.v1], 1)
        self.assertEqual([b.id for b in c.sequence.blocks], [first.id, placed.id, second.id])
        self.assertEqual(c.current, current);self.assertEqual(Path(self.w.video_path), self.v1)
        self.assertEqual(c.view.span(), span)   # the zoom stays where the user put it
        self.assertEqual([b['id'] for b in self.saved()['blocks']], [first.id, placed.id, second.id])

    def test_several_videos_are_placed_in_their_order(self):
        first, second = self.two();c = self.c
        c.place([self.v2, self.v1], 0)
        self.assertEqual(self.blocks(), ['two.mp4', 'one.mp4', 'one.mp4', 'two.mp4']);self.assertEqual(c.sequence.blocks[2:], [first, second])

    def test_without_a_place_the_video_goes_to_the_end(self):
        self.two();c = self.c
        c.place([self.v1])
        self.assertEqual(self.blocks(), ['one.mp4', 'two.mp4', 'one.mp4'])

    def test_a_video_dropped_on_the_timeline_from_the_file_manager_is_also_listed(self):
        self.open_first();c = self.c;c.back()
        c.view.dropped.emit([str(self.v2)], 0)
        self.assertEqual(self.blocks(), ['two.mp4', 'one.mp4']);self.assertEqual(self.videos(), [self.v1, self.v2])

    def test_a_double_click_in_the_project_panel_places_the_video_at_the_end(self):
        self.open_first();c = self.c;self.assertTrue(c.at_clip_level())
        c.library.placed.emit(str(self.v1))
        self.assertEqual(self.blocks(), ['one.mp4', 'one.mp4']);self.assertFalse(c.at_clip_level())

    def test_a_video_dropped_on_the_opened_block_goes_after_it(self):
        first, second = self.two();c, w = self.c, self.w
        c.open_block(first.id);self.assertTrue(c.at_clip_level())
        point = QPoint(w.slider.width() // 2, w.slider.height() // 2);mime = dragged(self.v2)   # kept: an event does not own its data
        enter = QDragEnterEvent(point, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier);APP.sendEvent(w.slider, enter)
        event = QDropEvent(QPointF(point), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier);APP.sendEvent(w.slider, event)
        self.assertTrue(enter.isAccepted());self.assertTrue(event.isAccepted())
        self.assertEqual([b.id for b in c.sequence.blocks][0::2], [first.id, second.id]);self.assertEqual(self.blocks(), ['one.mp4', 'two.mp4', 'two.mp4'])
        self.assertFalse(c.at_clip_level())
        text = QMimeData();text.setText('x')
        refused = QDragEnterEvent(point, Qt.CopyAction, text, Qt.LeftButton, Qt.NoModifier);APP.sendEvent(w.slider, refused)
        self.assertFalse(refused.isAccepted())

    def test_nothing_is_placed_while_a_job_runs(self):
        self.two();c = self.c;self.busy()
        self.assertIsNone(c.place([self.v1], 0))
        self.assertEqual(self.blocks(), ['one.mp4', 'two.mp4'])
        self.assertEqual(self.w.status.text(), 'A job is running. Videos can be put on the timeline when it is done.')

    def test_a_missing_file_is_not_placed(self):
        self.open_first();c = self.c
        self.assertEqual(c.place([self.base/'gone.mp4'], 0), [])
        self.assertEqual(self.blocks(), ['one.mp4']);self.assertIn('Cannot add gone.mp4:', self.w.status.text())

    def test_placing_between_a_joined_row_asks_first(self):
        first, second = self.two();c = self.c
        with mock.patch.object(c, '_may_change', return_value=False) as asked:
            self.assertEqual(c.place([self.v1], 1), [])
        asked.assert_called_once();self.assertEqual([b.id for b in c.sequence.blocks], [first.id, second.id])


class RazorTest(ProjectCase):
    def test_the_razor_cuts_the_block_it_clicks(self):
        first, second = self.two();c = self.c;current = c.current
        c.view.cut_requested.emit(second.id, 1.0)
        self.assertEqual(self.ranges(), [(0., 3.), (0., 1.), (1., 3.)])
        self.assertEqual(c.current, current)   # the loaded block was not the one cut

    def test_cutting_the_loaded_block_loads_its_second_piece(self):
        self.open_first();c = self.c;c.back();block = c.sequence.blocks[0]
        c.view.cut_requested.emit(block.id, 2.0)
        self.assertEqual(self.ranges(), [(0., 2.), (2., 3.)]);self.assertEqual(c.current, c.sequence.blocks[1].id)

    def test_a_cut_too_near_an_edge_says_why(self):
        self.open_first();c = self.c;c.back()
        c.view.cut_requested.emit(c.sequence.blocks[0].id, 0.01)
        self.assertEqual(self.ranges(), [(0., 3.)]);self.assertEqual(self.w.status.text(), 'A block must be at least 4 frames long.')

    def test_cutting_a_tracked_block_asks_first(self):
        self.open_first();c = self.c;block = self.loaded(self.tracked('one'));c.back()
        with mock.patch.object(c, 'ask', return_value=False) as ask:
            c.view.cut_requested.emit(block.id, 1.0)
        ask.assert_called_once_with(sequence_ui.CHANGE);self.assertEqual(self.ranges(), [(0., 3.)])

    def test_the_razor_does_nothing_while_a_job_runs(self):
        self.two();c = self.c;self.busy()
        c.cut_at(c.sequence.blocks[0].id, 1.0)
        self.assertEqual(self.ranges(), [(0., 3.), (0., 3.)])

    def test_the_cut_button_still_cuts_at_the_playhead(self):
        self.open_first();c, w = self.c, self.w;c.back();w.slider.setValue(45)
        c.cut_button.click()
        self.assertEqual(self.ranges(), [(0., 1.5), (1.5, 3.)])


class RemoveTest(ProjectCase):
    def test_a_video_removed_from_the_project_panel_keeps_its_blocks(self):
        self.two();c = self.c
        c.library.removed.emit([str(self.v1)])
        self.assertEqual(self.videos(), [self.v2]);self.assertEqual(self.blocks(), ['one.mp4', 'two.mp4'])
        self.assertEqual(self.saved()['library'], [{'video': str(self.v2), 'length': 3.0}])
        self.assertEqual(self.w.status.text(), 'one.mp4 removed from the Project panel. Its blocks stay on the timeline; the file stays on disk.')

    def test_a_video_without_blocks_is_just_removed(self):
        c = self.c;c.import_videos([self.v1, self.v2])
        c.library.removed.emit([str(self.v1), str(self.v2)])
        self.assertEqual(self.videos(), []);self.assertEqual(self.w.status.text(), '2 videos removed from the Project panel. The files stay on disk.')
        self.assertTrue(self.v1.is_file() and self.v2.is_file())


if __name__ == '__main__':
    unittest.main()
