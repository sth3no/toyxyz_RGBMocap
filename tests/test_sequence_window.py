"""The sequence in the real main window, offscreen: the stacked timeline, loading a block, the playhead and the levels.
Small real videos made with ffmpeg; paths.HOME is a temporary folder. Needs PySide6 and ffmpeg."""
import json, os, shutil, subprocess, tempfile, time, unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QMenu

from mocap import paths

APP = QApplication.instance() or QApplication([])
FFMPEG = paths.ROOT/'tools/bin/ffmpeg.exe'
OTHER = 'The sequence works with GVHMR, ExMR and FlowHMR.'


def application():
    if not getattr(application, 'configured', False):
        from mocap.theme import configure_application
        configure_application(APP);application.configured = True


def pump(until, seconds=30):
    deadline = time.monotonic()+seconds
    while not until() and time.monotonic() < deadline:APP.processEvents();time.sleep(.01)
    return until()


def wheel(widget, notches):
    point = QPointF(widget.width()/2, widget.height()/2)
    APP.sendEvent(widget, QWheelEvent(point, QPointF(widget.mapToGlobal(point)), QPoint(0, 0), QPoint(0, 120*notches),
                                      Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))


@unittest.skipUnless(FFMPEG.is_file(), 'needs the bundled ffmpeg')
class SequenceWindowTest(unittest.TestCase):
    def setUp(self):
        application();tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);base = self.base = Path(tmp.name)
        home = mock.patch.object(paths, 'HOME', base);home.start();self.addCleanup(home.stop)
        self.v1, self.v2 = base/'one.mp4', base/'two.mp4'
        for video, pattern in ((self.v1, 'testsrc'), (self.v2, 'testsrc2')):   # 3 s each, 90 frames at 30 fps
            subprocess.run([str(FFMPEG), '-v', 'error', '-f', 'lavfi', '-i', f'{pattern}=size=64x48:rate=30', '-frames:v', '90',
                            '-pix_fmt', 'yuv420p', '-c:v', 'libx264', str(video)], check=True, creationflags=0x08000000)
        missing = mock.patch('mocap.models.missing', lambda *a, **k:[]);missing.start();self.addCleanup(missing.stop)
        from mocap.gui import MainWindow
        w = self.w = MainWindow()
        def close():
            if w.tutorial:w.tutorial.stop()
            w.close();w.deleteLater();APP.processEvents()
        self.addCleanup(close);w.show();APP.processEvents()
        self.c = w.sequence_ui

    def ready(self, video):
        w = self.w
        self.assertTrue(pump(lambda:w.video_readable and w.info and Path(w.video_path) == video and self.c._pending is None), w.status.text())

    def open_first(self):
        """File > Open Video with v1, as the dialog returns it."""
        with mock.patch('mocap.gui.QFileDialog.getOpenFileName', lambda *a, **k:(str(self.v1), '')):self.w.choose_video()
        self.ready(self.v1)

    def two(self):
        self.open_first();self.assertIsNotNone(self.c.add_video(self.v2));APP.processEvents()
        return self.c.sequence.blocks

    def range(self):
        return tuple(round(x, 3) for x in self.w.processing_range())

    def test_open_video_is_one_block_at_level_3(self):
        self.open_first();c = self.c
        self.assertEqual(len(c.sequence.blocks), 1);self.assertTrue(c.sequence.plain())
        self.assertEqual((c.sequence.blocks[0].video, c.sequence.blocks[0].end), (str(self.v1), 3.))
        self.assertEqual(c.current, c.sequence.blocks[0].id)
        self.assertTrue(c.at_clip_level());self.assertTrue(self.w.slider.isVisible());self.assertFalse(c.view.isVisible())

    def test_menu_entry_follows_open_project_and_is_an_open_action(self):
        w = self.w;menu = next(m for m in w.menuBar().findChildren(QMenu) if m.title() == 'File')
        texts = [a.text() for a in menu.actions()]
        self.assertEqual(texts[:3], ['Open Video...', 'Open Project...', 'Add Video to Sequence...'])
        self.assertIn(menu.actions()[2], w.open_actions)
        w.busy = True
        try:
            w.update_readiness();self.assertFalse(menu.actions()[2].isEnabled())
        finally:
            w.busy = False;w.update_readiness()
        self.assertTrue(menu.actions()[2].isEnabled())

    def test_add_second_video_shows_the_sequence(self):
        first, second = self.two();c = self.c
        self.assertEqual(len(c.sequence.blocks), 2);self.assertEqual(second.video, str(self.v2))
        self.assertFalse(c.at_clip_level());self.assertTrue(c.view.isVisible());self.assertFalse(self.w.slider.isVisible())
        self.assertEqual(c.current, first.id);self.assertEqual(c.view.current, first.id)
        self.assertEqual(c.summary.text(), '2 blocks | 0:06.0')
        self.assertEqual(Path(self.w.video_path), self.v1)   # the loaded video stays

    def test_add_video_on_an_empty_window_loads_it(self):
        c = self.c;block = c.add_video(self.v2)
        self.assertIsNotNone(block);self.ready(self.v2)
        self.assertEqual(c.current, block.id);self.assertEqual(self.range(), (0., 3.))

    def test_selecting_a_block_loads_its_video_and_trim(self):
        _, second = self.two();c = self.c
        second.start, second.end = 1., 2.;c.refresh()
        c.view.selected.emit(second.id);self.ready(self.v2)
        self.assertEqual(c.current, second.id);self.assertEqual(Path(self.w.video_path), self.v2)
        self.assertEqual(self.range(), (1., 2.))
        self.assertIsNotNone(self.w.slider.view_span)   # the timeline shows the block's range
        self.assertFalse(c.at_clip_level())   # a click loads, it does not change the level

    def test_two_blocks_of_one_video_do_not_reload(self):
        self.open_first();c = self.c;c.add_video(self.v1)
        first, second = c.sequence.blocks;first.end, second.start = 1., 1.;c.refresh()
        with mock.patch.object(self.w, 'load_video') as load:
            self.assertTrue(c.load(second.id));self.assertEqual(self.range(), (1., 3.));self.assertEqual(c.current, second.id)
            self.assertEqual(self.w.slider.value(), self.w.slider.clip_in)
            self.assertTrue(c.load(first.id, .5));self.assertEqual(self.range(), (0., 1.));self.assertEqual(self.w.slider.value(), 15)
        self.assertEqual(load.call_count, 0)

    def test_tracked_block_loads_its_folder(self):
        self.open_first();c, w = self.c, self.w;model = w.model_choice.currentData()
        folder = self.base/'projects/one_project'/model;folder.mkdir(parents=True);shutil.copy(self.v1, folder/'input.mp4')
        (folder/'project.json').write_text(json.dumps(dict(version=1, frames=90, fps=30, start=0., end=3., width=64, height=48, camera='static',
                                                           source=dict(path=str(self.v1), fps=30.), pipeline=dict(model=model), stages={})), encoding='utf-8')
        block = c.sequence.blocks[0];block.folders[model] = str(folder);c.refresh()
        self.assertEqual(c.view.states[block.id].reason, '')
        self.assertTrue(c.load(block.id, 2.));self.assertIsNotNone(w.project)
        self.assertTrue(Path(w.project).resolve().is_relative_to(folder.resolve()), w.project)
        self.assertEqual(w.slider.value(), 60)
        Path(folder/'project.json').write_text('{}', encoding='utf-8');c.refresh()   # the folder no longer matches: the video loads
        self.assertTrue(c.load(block.id));self.assertTrue(w.status.text().startswith('The saved results are missing:'), w.status.text())
        self.ready(self.v1);self.assertIsNone(w.project)

    def test_playhead_maps_both_ways(self):
        first, second = self.two();c, w = self.c, self.w
        w.slider.setValue(45);self.assertAlmostEqual(c.view.playhead, c.sequence.position(first.id, 1.5))
        c.view.scrubbed.emit(2., False);self.assertEqual(w.slider.value(), 60);self.assertAlmostEqual(c.view.playhead, 2.)
        c.view.scrubbed.emit(4.5, False)   # into block 2 while dragging: only the drawn playhead moves
        self.assertEqual((c.current, w.slider.value(), Path(w.video_path)), (first.id, 60, self.v1));self.assertAlmostEqual(c.view.playhead, 4.5)
        c.view.scrubbed.emit(4.5, True);self.ready(self.v2)
        self.assertEqual(c.current, second.id);self.assertEqual(w.slider.value(), 45);self.assertAlmostEqual(c.view.playhead, 4.5)

    def test_open_and_back(self):
        first, second = self.two();c, w = self.c, self.w
        c.open_block(first.id);self.assertTrue(c.at_clip_level());self.assertEqual(c.current, first.id)
        c.back();self.assertFalse(c.at_clip_level());self.assertEqual(c.view.level(), 2)
        c.view.opened.emit(first.id, None)
        self.assertTrue(c.at_clip_level());self.assertIsNone(w.slider.view_span)
        wheel(w.slider, 1);self.assertTrue(c.at_clip_level())   # zooming in stays
        w.slider.frame_all();wheel(w.slider, -1);self.assertFalse(c.at_clip_level())   # wheel-out on the fitted slider goes back
        c.view.opened.emit(second.id, None);self.ready(self.v2)
        self.assertTrue(c.at_clip_level());self.assertEqual(c.current, second.id)
        c.back_button.click();self.assertFalse(c.at_clip_level())

    def test_open_block_with_a_person_activates_their_row(self):
        self.open_first();c, w = self.c, self.w
        rows = []
        with mock.patch.object(w, 'person_selected', lambda item, previous=None:rows.append(item.data(Qt.UserRole))):
            from PySide6.QtWidgets import QListWidgetItem
            for ident in (1, 2):
                item = QListWidgetItem(f'Person {ident}');item.setData(Qt.UserRole, ident);w.people_list.addItem(item)
            c.open_block(c.current, 2)
        self.assertEqual(rows, [2]);self.assertTrue(c.at_clip_level())

    def test_unreadable_video_is_refused(self):
        self.open_first();c = self.c;text = self.base/'notes.mp4';text.write_text('not a video', encoding='utf-8')
        self.assertIsNone(c.add_video(text))
        self.assertTrue(self.w.status.text().startswith('Cannot add the video:'), self.w.status.text())
        self.assertEqual(len(c.sequence.blocks), 1);self.assertTrue(c.at_clip_level())

    def test_other_model_keeps_today(self):
        self.two();c, w = self.c, self.w
        w.model_choice.setCurrentIndex(w.model_choice.findData('gemx'));APP.processEvents()
        self.assertTrue(c.at_clip_level())
        from mocap.tooltips import plain_tooltip
        for button in (c.back_button, c.add_button, c.cut_button, c.delete_button, c.frame_button):
            self.assertFalse(button.isEnabled(), button.text());self.assertEqual(plain_tooltip(button.toolTip()), OTHER)
        self.assertIsNone(c.add_video(self.v2));self.assertEqual(w.status.text(), OTHER);self.assertEqual(len(c.sequence.blocks), 2)
        w.slider.frame_all();wheel(w.slider, -1);self.assertTrue(c.at_clip_level())
        c.back();self.assertTrue(c.at_clip_level())
        w.model_choice.setCurrentIndex(w.model_choice.findData('exmr'));APP.processEvents()
        self.assertTrue(c.back_button.isEnabled());self.assertFalse(c.cut_button.isEnabled())   # Cut and Delete come later

    def test_guide_runs_at_level_3(self):
        self.two();c = self.c;self.assertFalse(c.at_clip_level())
        from mocap.tutorial import open_chapter
        open_chapter(self.w, 'basics');self.assertTrue(pump(lambda:c.at_clip_level(), 5))
        self.assertTrue(self.w.tutorial.active)

    def test_ui_scale_resizes_the_view(self):
        from mocap.theme import apply_appearance, current_appearance, ui_px
        old = current_appearance();self.addCleanup(lambda:apply_appearance(APP, *old))
        apply_appearance(APP, old[0], 150);APP.processEvents()
        self.assertEqual(self.c.view.LEFT, ui_px(64));self.assertNotEqual(ui_px(64), 64)


if __name__ == '__main__':unittest.main()
