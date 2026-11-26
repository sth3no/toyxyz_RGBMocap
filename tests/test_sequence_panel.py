"""The Project panel (the library of videos beside the timeline) and the strip of timeline tools, offscreen."""
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mocap import paths, sequence_view
from mocap.sequence import Media

APP = QApplication.instance() or QApplication([])
FFMPEG = paths.ROOT/'tools/bin/ffmpeg.exe'


def application():
    if not getattr(application, 'configured', False):
        from mocap.theme import configure_application
        configure_application(APP);application.configured = True


def pump(until, seconds=20):
    deadline = time.monotonic() + seconds
    while not until() and time.monotonic() < deadline:APP.processEvents();time.sleep(.01)
    return until()


def files(*names):
    mime = QMimeData();mime.setUrls([QUrl.fromLocalFile(name) for name in names]);return mime


def send_drop(widget, mime, point=QPoint(20, 20)):
    enter = QDragEnterEvent(point, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier);APP.sendEvent(widget, enter)
    event = QDropEvent(QPointF(point), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier);APP.sendEvent(widget, event)
    return enter.isAccepted(), event.isAccepted()


class ToolStripTest(unittest.TestCase):
    def setUp(self):
        application()
        from mocap.sequence_tools import ToolStrip
        self.strip = ToolStrip();self.addCleanup(self.strip.deleteLater)
        self.picked = [];self.strip.picked.connect(self.picked.append)
        self.strip.show();APP.processEvents()

    def checked(self):return [tool for tool, button in self.strip.buttons.items() if button.isChecked()]

    def test_one_button_per_tool_with_selection_chosen(self):
        self.assertEqual(tuple(self.strip.buttons), sequence_view.TOOLS);self.assertEqual(self.checked(), ['select'])
        for tool, key in zip(sequence_view.TOOLS, 'VCHZ'):
            button = self.strip.buttons[tool]
            self.assertTrue(button.isCheckable());self.assertFalse(button.icon().isNull());self.assertIn(f'({key})', button.toolTip())
            self.assertTrue(button.accessibleName())

    def test_a_click_picks_the_tool(self):
        QTest.mouseClick(self.strip.buttons['razor'], Qt.LeftButton)
        self.assertEqual((self.picked, self.checked()), (['razor'], ['razor']))
        QTest.mouseClick(self.strip.buttons['razor'], Qt.LeftButton)   # the chosen tool stays chosen
        self.assertEqual((self.picked, self.checked()), (['razor'], ['razor']))

    def test_set_tool_shows_the_tool_without_picking_it(self):
        self.strip.set_tool('hand')
        self.assertEqual((self.picked, self.checked()), ([], ['hand']))

    def test_the_icons_follow_the_appearance(self):
        before = self.strip.buttons['zoom'].icon().cacheKey()
        self.strip.refresh_appearance()
        self.assertNotEqual(self.strip.buttons['zoom'].icon().cacheKey(), before)
        self.assertFalse(self.strip.grab().isNull())


class LibraryTest(unittest.TestCase):
    def setUp(self):
        application()
        from mocap.sequence_library import Library
        self.library = Library();self.addCleanup(self.library.deleteLater)
        self.seen = {name: [] for name in ('imported', 'placed', 'removed', 'import_requested')}
        for name, seen in self.seen.items():
            getattr(self.library, name).connect(lambda *args, seen=seen: seen.append(args))
        self.library.resize(320, 240);self.library.show();APP.processEvents()
        self.media = [Media('C:/videos/zongler.mp4', 248.0), Media('C:/videos/tanec.mp4', 3725.4)]

    def item(self, row):return self.library.list.item(row)

    def test_lists_the_videos_with_their_lengths(self):
        self.library.show_media(self.media, used={'C:/videos/tanec.mp4'})
        self.assertEqual([self.item(i).text() for i in range(2)], ['zongler.mp4\n4:08', 'tanec.mp4\n1:02:05'])
        self.assertEqual(self.library.count.text(), '2 videos')
        self.assertIn('Not on the timeline', self.item(0).toolTip());self.assertIn('On the timeline', self.item(1).toolTip())
        self.assertIn(os.path.normpath('C:/videos/tanec.mp4'), self.item(1).toolTip())
        self.assertFalse(self.item(0).icon().isNull());self.assertFalse(self.library.grab().isNull())

    def test_an_empty_library_says_how_to_fill_it(self):
        from mocap.sequence_library import EMPTY
        self.assertFalse(self.library.hint.isHidden());self.assertEqual(self.library.hint.text(), EMPTY)
        self.assertEqual(self.library.count.text(), '')
        self.library.show_media(self.media[:1], used=set())
        self.assertTrue(self.library.hint.isHidden());self.assertEqual(self.library.count.text(), '1 video')

    def test_the_selection_survives_a_refresh(self):
        self.library.show_media(self.media, used=set());self.library.list.setCurrentRow(1)
        self.library.show_media([Media('C:/videos/new.mp4', 5.0), *self.media], used=set())
        self.assertEqual([i.text().split('\n')[0] for i in self.library.list.selectedItems()], ['tanec.mp4'])

    def test_dragged_videos_carry_their_paths_for_the_timeline(self):
        self.library.show_media(self.media, used=set())
        mime = self.library.list.mimeData([self.item(1), self.item(0)])
        self.assertEqual(json.loads(bytes(mime.data(sequence_view.MIME)).decode('utf-8')), ['C:/videos/tanec.mp4', 'C:/videos/zongler.mp4'])
        self.assertFalse(mime.hasUrls())   # the main window opens dropped files; a library video is not one
        self.assertEqual(sequence_view.dragged_videos(mime), ['C:/videos/tanec.mp4', 'C:/videos/zongler.mp4'])

    def test_a_drag_started_by_qt_carries_the_same_paths_and_copies(self):
        self.library.show_media(self.media, used=set())
        model = self.library.list.model();mime = model.mimeData([model.index(1, 0)])   # as the view's own startDrag asks
        self.assertEqual(sequence_view.dragged_videos(mime), ['C:/videos/tanec.mp4'])
        self.assertTrue(model.flags(model.index(0, 0)) & Qt.ItemIsDragEnabled)
        self.assertEqual(self.library.list.defaultDropAction(), Qt.CopyAction)   # a move would take the video out of the list

    def test_video_files_dropped_from_the_file_manager_are_imported(self):
        accepted = send_drop(self.library.list.viewport(), files('C:/v/one.mp4', 'C:/v/readme.txt', 'C:/v/two.MOV'))
        self.assertEqual(accepted, (True, True))
        (dropped,), = self.seen['imported']
        self.assertEqual([os.path.normpath(p) for p in dropped], [os.path.normpath('C:/v/one.mp4'), os.path.normpath('C:/v/two.MOV')])

    def test_other_drops_are_refused(self):
        self.library.show_media(self.media, used=set())
        self.assertEqual(send_drop(self.library.list.viewport(), files('C:/v/readme.txt')), (False, False))
        own = self.library.list.mimeData([self.item(0)])
        self.assertEqual(send_drop(self.library.list.viewport(), own), (False, False))
        self.assertEqual(self.seen['imported'], [])

    def test_a_double_click_places_the_video(self):
        self.library.show_media(self.media, used=set())
        self.library.list.itemDoubleClicked.emit(self.item(1))
        self.assertEqual(self.seen['placed'], [('C:/videos/tanec.mp4',)])

    def test_delete_removes_the_selected_videos(self):
        self.library.show_media(self.media, used=set())
        self.library.list.setCurrentRow(0);self.library.list.setFocus()
        QTest.keyClick(self.library.list, Qt.Key_Delete)
        self.assertEqual(self.seen['removed'], [(['C:/videos/zongler.mp4'],)])
        self.library.list.clearSelection();QTest.keyClick(self.library.list, Qt.Key_Delete)
        self.assertEqual(len(self.seen['removed']), 1)

    def test_the_import_button_asks_for_videos(self):
        QTest.mouseClick(self.library.import_button, Qt.LeftButton)
        self.assertEqual(self.seen['import_requested'], [()])

    def test_a_missing_file_is_named(self):
        self.library.show_media(self.media[:1], used=set())
        self.assertTrue(pump(lambda:'missing' in self.item(0).toolTip()), self.item(0).toolTip())

    def test_stop_waits_for_the_reads_in_progress(self):
        self.library.show_media(self.media, used=set())
        self.library.stop()
        self.assertEqual(self.library._pool.activeThreadCount(), 0)

    def test_lengths_read_as_a_clock(self):
        from mocap.sequence_library import clock
        self.assertEqual([clock(s) for s in (0, 7.4, 59.6, 248, 3600, 3725.4, 36000)], ['0:00', '0:07', '1:00', '4:08', '1:00:00', '1:02:05', '10:00:00'])


@unittest.skipUnless(FFMPEG.is_file(), 'needs the bundled ffmpeg')
class ThumbnailTest(unittest.TestCase):
    def setUp(self):
        application()
        from mocap.sequence_library import Library
        tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.video = Path(tmp.name)/'clip.mp4'
        subprocess.run([str(FFMPEG), '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=64x48:rate=30', '-frames:v', '60', '-pix_fmt', 'yuv420p',
                        '-c:v', 'libx264', str(self.video)], check=True, creationflags=0x08000000)
        self.library = Library();self.addCleanup(self.library.deleteLater)
        self.library.resize(320, 240);self.library.show();APP.processEvents()

    def test_a_video_gets_a_picture_and_its_size_and_rate(self):
        self.library.show_media([Media(str(self.video), 2.0)], used=set())
        item = self.library.list.item(0);placeholder = item.icon().cacheKey()
        self.assertTrue(pump(lambda:'64×48' in item.toolTip()), item.toolTip())
        self.assertIn('30.00 fps', item.toolTip());self.assertNotEqual(item.icon().cacheKey(), placeholder)
        picture = item.icon().pixmap(self.library.list.iconSize()).toImage()
        colours = {picture.pixelColor(x, picture.height() // 2).rgb() for x in range(4, picture.width() - 4, 6)}
        self.assertGreater(len(colours), 3)   # the test pattern's bars, not a blank placeholder

    def test_the_picture_is_kept_when_the_list_is_shown_again(self):
        media = [Media(str(self.video), 2.0)]
        self.library.show_media(media, used=set())
        self.assertTrue(pump(lambda:'64×48' in self.library.list.item(0).toolTip()))
        key = self.library.list.item(0).icon().cacheKey()
        self.library.show_media(media, used={str(self.video)})
        self.assertEqual(self.library.list.item(0).icon().cacheKey(), key)
        self.assertIn('64×48', self.library.list.item(0).toolTip())


if __name__ == '__main__':
    unittest.main()
