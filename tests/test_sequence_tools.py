"""The sequence timeline's tools (Selection, Razor, Hand, Zoom), zooming out past the sequence, and videos dropped on it."""
import json
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent, QMouseEvent
from PySide6.QtTest import QTest

import test_sequence_view as base
from mocap import sequence_view
from mocap.sequence import Sequence

APP = base.APP


def library_mime(*paths):
    mime = QMimeData();mime.setData(sequence_view.MIME, json.dumps([str(p) for p in paths]).encode('utf-8'));return mime


def file_mime(*paths):
    mime = QMimeData();mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths]);return mime


def hover(view, point):
    """The pointer moves over the widget with no button held."""
    APP.sendEvent(view, QMouseEvent(QEvent.MouseMove, QPointF(point), QPointF(view.mapToGlobal(point)), Qt.NoButton, Qt.NoButton, Qt.NoModifier))


def drag_over(view, point, mime):
    enter = QDragEnterEvent(point, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier);APP.sendEvent(view, enter)
    move = QDragMoveEvent(point, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier);APP.sendEvent(view, move)
    return enter.isAccepted(), move.isAccepted()


def drop(view, point, mime):
    drag_over(view, point, mime)
    event = QDropEvent(QPointF(point), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier);APP.sendEvent(view, event)
    return event.isAccepted()


class ToolCase(base.ViewCase):
    def setUp(self):
        super().setUp()
        for name in ('cut_requested', 'dropped', 'tool_changed'):
            self.signals[name] = seen = []
            getattr(self.view, name).connect(lambda *args, seen=seen: seen.append(args))

    def centre(self, block):return self.view.block_rect(block.id).center()


class ZoomOutTest(ToolCase):
    def test_the_whole_sequence_is_a_stop_on_the_way_out_and_back(self):
        view = self.view;total = self.sequence.total()
        view.show_block(self.b2.id)
        for _ in range(60):
            base.wheel(view, QPoint(500, 40), -1)
            if view.view_span is None:break
        self.assertIsNone(view.view_span);self.assertEqual(view.span(), total)   # fitted, as Frame All
        base.wheel(view, QPoint(500, 40), -1)
        self.assertGreater(view.span(), total);self.assertEqual(view.view_start, 0.);self.assertEqual(view.level(), 1)
        for _ in range(60):
            base.wheel(view, QPoint(500, 40), 1)
            if view.view_span is None:break
        self.assertIsNone(view.view_span)

    def test_zooming_out_reaches_hours_and_stops_at_a_day(self):
        view = self.view
        for _ in range(200):base.wheel(view, QPoint(500, 40), -1)
        self.assertEqual(view.span(), sequence_view.FARTHEST);self.assertEqual(sequence_view.FARTHEST, 24 * 3600)
        self.assertEqual(self.signals['opened'], [])
        self.assertFalse(view.grab().isNull())
        # The blocks stay at the start, each at least its smallest drawn width.
        self.assertLess(view.block_rect(self.b3.id).right(), view.LEFT + 60)

    def test_a_sequence_longer_than_a_day_can_still_be_seen_whole(self):
        long = Sequence();long.add('C:/videos/long.mp4', 30 * 3600.)
        self.view.show_sequence(long, {});self.view.frame_all()
        base.wheel(self.view, QPoint(500, 40), -1)
        self.assertEqual(self.view.span(), 30 * 3600.)

    def test_a_refresh_keeps_a_zoomed_out_view(self):
        view = self.view
        for _ in range(3):base.wheel(view, QPoint(500, 40), -1)
        span = view.span();self.assertGreater(span, self.sequence.total())
        view.show_sequence(self.sequence, self.states)
        self.assertEqual(view.span(), span)

    def test_ruler_labels_drop_the_tenths_once_a_step_is_a_second_or_more(self):
        view = self.view
        view.view_start, view.view_span = 0., 4 * 3600.
        self.assertEqual([view.ruler_text(s) for s in (0, 1800, 3600, 5400)], ['0:00', '30:00', '1:00:00', '1:30:00'])
        view.view_span = 2.
        self.assertLess(view.ruler_step(), 1);self.assertEqual(view.ruler_text(1.5), '0:01.5')

    def test_the_ruler_reads_in_hours_when_zoomed_out(self):
        self.view.view_start, self.view.view_span = 0., 4 * 3600.
        self.assertEqual(self.view.ruler_step(), 1800)
        self.view.view_span = 24 * 3600.
        self.assertGreaterEqual(self.view.ruler_step(), 3600)


class ToolsTest(ToolCase):
    def test_the_tool_is_selection_until_another_is_set(self):
        view = self.view
        self.assertEqual((view.tool, sequence_view.TOOLS), ('select', ('select', 'razor', 'hand', 'zoom')))
        view.set_tool('razor');view.set_tool('razor')
        self.assertEqual(view.tool, 'razor');self.assertEqual(self.signals['tool_changed'], [('razor',)])
        with self.assertRaises(ValueError):view.set_tool('pen')

    def test_keys_pick_the_tools(self):
        view = self.view;view.setFocus()
        for key, tool in ((Qt.Key_C, 'razor'), (Qt.Key_H, 'hand'), (Qt.Key_Z, 'zoom'), (Qt.Key_V, 'select')):
            QTest.keyClick(view, key);self.assertEqual(view.tool, tool)
        self.assertEqual([args[0] for args in self.signals['tool_changed']], ['razor', 'hand', 'zoom', 'select'])

    def test_the_razor_cuts_the_block_where_it_is_clicked(self):
        view = self.view;view.frame_all();view.set_tool('razor')
        rect = view.block_rect(self.b2.id);point = QPoint(rect.left() + rect.width() // 4, rect.center().y())
        QTest.mouseClick(view, Qt.LeftButton, Qt.NoModifier, point)
        (block_id, seconds), = self.signals['cut_requested']
        self.assertEqual(block_id, self.b2.id)
        self.assertAlmostEqual(seconds, self.b2.start + view.time_at(point.x()) - self.sequence.starts()[1], places=6)
        self.assertAlmostEqual(seconds, 10., delta=.3)   # a quarter into 5-25 s
        self.assertEqual((self.signals['selected'], self.signals['moved'], self.signals['resized']), ([], [], []))

    def test_the_razor_neither_moves_nor_trims_a_block(self):
        view = self.view;view.frame_all();view.set_tool('razor')
        rect = view.block_rect(self.b2.id)
        base.drag(view, QPoint(rect.left() + 2, rect.center().y()), QPoint(rect.left() + 80, rect.center().y()))
        base.drag(view, rect.center(), QPoint(view.block_rect(self.b1.id).left() + 3, rect.center().y()))
        self.assertEqual((self.signals['moved'], self.signals['resized']), ([], []))

    def test_the_razor_shows_where_it_would_cut(self):
        view = self.view;view.frame_all();view.set_tool('razor')
        hover(view, self.centre(self.b2))
        self.assertEqual(view.razor_at, self.centre(self.b2).x());self.assertFalse(view.grab().isNull())
        hover(view, QPoint(view.LEFT + 5, view.height() - 5))   # below the blocks
        self.assertIsNone(view.razor_at)
        hover(view, self.centre(self.b2));view.set_tool('select')
        self.assertIsNone(view.razor_at)
        hover(view, self.centre(self.b2))
        self.assertIsNone(view.razor_at)

    def test_the_razor_still_scrubs_in_the_ruler(self):
        view = self.view;view.set_tool('razor')
        QTest.mouseClick(view, Qt.LeftButton, Qt.NoModifier, QPoint(500, 8))
        self.assertTrue(self.signals['scrubbed']);self.assertEqual(self.signals['cut_requested'], [])

    def test_a_busy_timeline_does_not_cut(self):
        view = self.view;view.set_tool('razor');view.editable = False
        QTest.mouseClick(view, Qt.LeftButton, Qt.NoModifier, self.centre(self.b2))
        self.assertEqual(self.signals['cut_requested'], [])

    def test_the_hand_drags_the_view_and_changes_nothing(self):
        view = self.view;view.show_block(self.b2.id);view.set_tool('hand')
        start = view.view_start;at = self.centre(self.b2)
        base.drag(view, at, at + QPoint(-120, 0))
        self.assertAlmostEqual(view.view_start, start + 120 * view.span() / view._width(), places=6)
        self.assertEqual([self.signals[name] for name in ('selected', 'moved', 'resized', 'scrubbed')], [[], [], [], []])

    def test_the_zoom_tool_zooms_in_on_a_click_and_out_with_alt(self):
        view = self.view;view.frame_all();view.set_tool('zoom')
        total = self.sequence.total();point = QPoint(500, 40);anchor = view.time_at(500)
        QTest.mouseClick(view, Qt.LeftButton, Qt.NoModifier, point)
        self.assertLess(view.span(), total);self.assertAlmostEqual(view.time_at(500), anchor, places=6)
        self.assertEqual(self.signals['selected'], [])
        span = view.span()
        QTest.mouseClick(view, Qt.LeftButton, Qt.AltModifier, point)
        self.assertGreater(view.span(), span)
        for _ in range(4):QTest.mouseClick(view, Qt.LeftButton, Qt.AltModifier, point)
        self.assertGreater(view.span(), total)


class DropTest(ToolCase):
    def test_a_library_video_is_dropped_between_the_blocks_under_the_pointer(self):
        view = self.view;view.frame_all()
        y = view.block_rect(self.b1.id).center().y()
        for x, index in ((view.block_rect(self.b1.id).left() + 2, 0), (view.block_rect(self.b2.id).left() + 4, 1),
                         (view.block_rect(self.b2.id).right() - 4, 2), (view.width() - view.RIGHT - 2, 3)):
            self.signals['dropped'].clear()
            self.assertTrue(drop(view, QPoint(x, y), library_mime('C:/v/new.mp4')))
            self.assertEqual(self.signals['dropped'], [(['C:/v/new.mp4'], index)], x)

    def test_the_place_is_marked_while_a_video_is_dragged_over(self):
        view = self.view;view.frame_all()
        point = QPoint(view.block_rect(self.b2.id).left() + 4, 40)
        self.assertEqual(drag_over(view, point, library_mime('C:/v/new.mp4')), (True, True))
        self.assertEqual(view.incoming, 1);self.assertFalse(view.grab().isNull())
        APP.sendEvent(view, QDragLeaveEvent())
        self.assertIsNone(view.incoming)
        drop(view, point, library_mime('C:/v/new.mp4'));self.assertIsNone(view.incoming)

    def test_video_files_from_the_file_manager_are_dropped_the_same_way(self):
        view = self.view;view.frame_all()
        mime = file_mime('C:/v/one.MP4', 'C:/v/notes.txt', 'C:/v/two.mov')
        self.assertTrue(drop(view, QPoint(view.LEFT + 2, 40), mime))
        (paths, index), = self.signals['dropped']
        self.assertEqual(([os.path.normpath(p) for p in paths], index), ([os.path.normpath('C:/v/one.MP4'), os.path.normpath('C:/v/two.mov')], 0))

    def test_other_things_are_not_taken(self):
        view = self.view
        text = QMimeData();text.setText('hello')
        self.assertEqual(drag_over(view, QPoint(500, 40), text), (False, False))
        self.assertFalse(drop(view, QPoint(500, 40), file_mime('C:/v/notes.txt')))
        self.assertEqual(self.signals['dropped'], [])

    def test_a_busy_timeline_takes_no_drop(self):
        view = self.view;view.editable = False
        self.assertFalse(drop(view, QPoint(500, 40), library_mime('C:/v/new.mp4')))
        self.assertEqual(self.signals['dropped'], []);self.assertIsNone(view.incoming)

    def test_an_empty_timeline_takes_the_first_video(self):
        view = self.view;view.show_sequence(Sequence(), {})
        self.assertTrue(drop(view, QPoint(400, 100), library_mime('C:/v/new.mp4')))
        self.assertEqual(self.signals['dropped'], [(['C:/v/new.mp4'], 0)])


if __name__ == '__main__':
    unittest.main()
