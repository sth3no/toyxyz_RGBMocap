"""The sequence in the real main window, offscreen: the stacked timeline, loading a block, the playhead and the levels.
Small real videos made with ffmpeg; paths.HOME is a temporary folder. Needs PySide6 and ffmpeg."""
import json, os, shutil, subprocess, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel, QMenu

from mocap import paths, people_store
from test_parts_window import scene, write_result
import test_sequence_state   # its take builder; not its StateTest by name, which would run its tests here again

APP = QApplication.instance() or QApplication([])
FFMPEG = paths.ROOT/'tools/bin/ffmpeg.exe'
OTHER = 'The sequence works with GVHMR, ExMR and FlowHMR.'
BUSY = 'A job is running. This block can be opened when it is done.'
LAST = 'A sequence needs at least one block.'
EXPERIMENTAL = 'Experimental: the generated motion is plausible, not what the person did.'
ANNA, BEN = ((1, 'Anna', range(90)),), ((1, 'Ben', range(90)),)
TWO = ((1, 'Ben', range(90)), (2, 'Cleo', range(90)))


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

    def ranges(self):
        return [(round(b.start, 3), round(b.end, 3)) for b in self.c.sequence.blocks]

    def manifest(self, start=0., end=3., model=None, **extra):
        return dict(version=1, frames=round((end-start)*30), fps=30, start=start, end=end, width=64, height=48, camera='static',
                    source=dict(path=str(self.v1), fps=30.), pipeline=dict(model=model or self.w.model_choice.currentData()), stages={}, **extra)

    def folder(self, name, manifest):
        """A clip folder of v1 as prepare leaves it (input.mp4 and project.json), without results."""
        folder = self.base/'projects'/name/manifest['pipeline']['model'];folder.mkdir(parents=True);shutil.copy(self.v1, folder/'input.mp4')
        (folder/'project.json').write_text(json.dumps(manifest), encoding='utf-8')
        return folder

    def found(self, name, start, end):
        """A clip folder of v1 after Find person with Anna (tests/test_parts_window.py scene)."""
        manifest = self.manifest(start, end);root = self.base/'projects'/name/manifest['pipeline']['model'];root.parent.mkdir(parents=True)
        scene.video = self.v1;scene(root, manifest, ((1, 'Anna', range(manifest['frames'])),))
        return root, manifest

    def tracked(self, name, model='exmr', start=0., end=3.):
        """A finished one-person clip folder of v1 without a people list (tests/test_sequence_state.py)."""
        folder = self.folder(name, self.manifest(start, end, model, exmr_options=dict(static_camera=True, smoothing='none')))
        state = json.loads((folder/'project.json').read_text(encoding='utf-8'));state['pipeline']['result'] = 'initial'
        (folder/'project.json').write_text(json.dumps(state), encoding='utf-8');write_result(folder, 'initial', b'result')
        return folder

    def loaded(self, folder):
        """Remember `folder` for the selected model in the current block and load the block."""
        c = self.c;block = c.sequence.block(c.current);block.folders[self.w.model_choice.currentData()] = str(folder);c.refresh()
        self.assertTrue(c.load(block.id));self.assertTrue(Path(self.w.project).resolve().is_relative_to(Path(folder).resolve()))
        return block

    def busy(self):
        w = self.w;w.busy = True;w.update_readiness()
        def done():
            w.busy = False;w.update_readiness()
        self.addCleanup(done)
        return done

    def test_open_video_is_one_block_at_level_3(self):
        self.open_first();c = self.c
        self.assertEqual(len(c.sequence.blocks), 1);self.assertTrue(c.sequence.plain())
        self.assertEqual((c.sequence.blocks[0].video, c.sequence.blocks[0].end), (str(self.v1), 3.))
        self.assertEqual(c.current, c.sequence.blocks[0].id)
        self.assertTrue(c.at_clip_level());self.assertTrue(self.w.slider.isVisible());self.assertFalse(c.view.isVisible())

    def test_review_playback_preserves_trim_and_advances_to_the_next_block(self):
        first, second = self.two()
        controller, window = self.c, self.w
        controller.sequence.set_review(first.id, dict(start=.5, end=1., result=None))
        controller.sequence.set_review(second.id, dict(start=1.5, end=2., result=None))
        controller.back()
        self.assertEqual(controller.review_bounds(), (15, 30))
        self.assertEqual(self.range(), (0., 3.))
        controller.auto_advance.setChecked(True)
        window.slider.setValue(0)
        window.toggle_play()
        self.assertEqual(window.slider.value(), 15)
        self.assertTrue(controller.advance_review())
        self.ready(self.v2)
        self.assertEqual(controller.current, second.id)
        self.assertEqual(controller.review_bounds(), (45, 60))
        self.assertEqual(self.range(), (0., 3.))
        self.assertTrue(window.timer.isActive())
        window.stop_playback()

    def test_menu_entry_follows_open_project_and_is_an_open_action(self):
        w = self.w;menu = next(m for m in w.menuBar().findChildren(QMenu) if m.title() == 'File')
        texts = [a.text() for a in menu.actions()]
        self.assertEqual(texts[:6], ['Open Video...', 'Open Project...', 'Add Video to Sequence...', 'Add Project to Sequence...',
                                     'Open Sequence...', 'New Sequence'])
        entries = menu.actions()[2:6]
        for action in entries:self.assertIn(action, w.open_actions);self.assertTrue(action.toolTip())
        w.busy = True
        try:
            w.update_readiness();self.assertFalse(any(action.isEnabled() for action in entries))
        finally:
            w.busy = False;w.update_readiness()
        self.assertTrue(all(action.isEnabled() for action in entries))

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

    def test_a_video_that_fails_to_open_can_be_tried_again(self):
        self.open_first();c, w = self.c, self.w
        bad = self.base/'broken.mp4';bad.write_text('not a video', encoding='utf-8')
        block = c.sequence.add(bad, 3.);c.refresh()
        self.assertTrue(c.load(block.id))
        self.assertTrue(pump(lambda:w.status.text().startswith('Video preview:'), 10), w.status.text())   # the window's decode_error
        self.assertIsNone(c._pending)
        with mock.patch.object(w, 'load_video') as load:
            c.load(block.id)
        self.assertEqual(load.call_count, 1)

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
        self.assertTrue(c.back_button.isEnabled());self.assertTrue(c.cut_button.isEnabled());self.assertTrue(c.delete_button.isEnabled())

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

    # Task 5: editing, remembering results, jobs, models, projects, files

    def test_cut_at_the_playhead(self):
        self.open_first();c, w = self.c, self.w
        w.slider.setValue(30)   # 1.0 s of the 3 s block
        self.assertTrue(c.cut_button.isEnabled());c.cut_button.click()
        self.assertEqual(self.ranges(), [(0., 1.), (1., 3.)])
        self.assertEqual(c.current, c.sequence.blocks[1].id);self.assertEqual(self.range(), (1., 3.));self.assertEqual(w.slider.value(), 30)
        path = c.sequence.path
        self.assertEqual(path.parent, self.base/'projects/sequences');self.assertRegex(path.name, r'^Sequence \d{4}-\d{2}-\d{2} \d{4}\.json$')
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), c.sequence.to_dict())

    def test_cut_tracked_block_asks(self):
        self.open_first();c, w = self.c, self.w;folder = self.folder('one', self.manifest());block = self.loaded(folder)
        before = c.sequence.to_dict();w.slider.setValue(30)
        with mock.patch.object(c, 'ask', return_value=False) as ask:c.cut()
        ask.assert_called_once_with('This block has results. They belong to the old range and stay on disk; the changed block will show as not tracked. Change it?')
        self.assertEqual(c.sequence.to_dict(), before);self.assertIsNotNone(w.project)
        with mock.patch.object(c, 'ask', return_value=True):c.cut()
        self.assertEqual(self.ranges(), [(0., 1.), (1., 3.)]);self.assertEqual([b.folders for b in c.sequence.blocks], [{}, {}])
        self.assertEqual([c.view.states[b.id].kind for b in c.sequence.blocks], ['untracked', 'untracked'])
        self.assertTrue((folder/'project.json').is_file())
        self.ready(self.v1);self.assertIsNone(w.project);self.assertEqual(self.range(), (1., 3.))   # the video with the new piece's range

    def test_resize_from_the_view(self):
        self.open_first();c = self.c
        c.view.resized.emit(c.current, .5, 2.)
        self.assertEqual(self.ranges(), [(.5, 2.)]);self.assertEqual(self.range(), (.5, 2.))

    def test_trim_resizes_an_untracked_block(self):
        self.open_first();c, w = self.c, self.w;self.assertTrue(c.at_clip_level())
        w.slider.set_clip(15, 60)
        self.assertEqual(self.ranges(), [self.range()]);self.assertEqual(self.range(), (.5, 2.))
        pump(lambda:False, .5)   # the trim has settled
        self.assertIsNone(c.sequence.path);self.assertFalse((self.base/'projects/sequences').exists())   # one block needs no file

    def test_a_saved_sequence_keeps_saving_with_one_block(self):
        first, second = self.two();c = self.c;path = c.sequence.path
        c.delete(second.id);self.assertEqual(len(c.sequence.blocks), 1)
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), c.sequence.to_dict())

    def test_a_dragged_trim_saves_once_it_settles(self):
        self.two();c, w = self.c, self.w;c.load(c.sequence.blocks[0].id)
        with mock.patch.object(c, '_save') as save:
            for end in (60, 61, 62):w.slider.set_clip(15, end)
            self.assertEqual(self.ranges()[0], (.5, round(62/30, 3)));self.assertEqual(save.call_count, 0)
            self.assertTrue(pump(lambda:save.call_count, 5))
            pump(lambda:False, .5);self.assertEqual(save.call_count, 1)
        self.assertEqual((c.view.sequence.blocks[0].start, c.view.states.keys()), (.5, {b.id for b in c.sequence.blocks}))

    def test_trim_with_results_of_another_model_changes_nothing(self):
        self.open_first();c, w = self.c, self.w;block = c.sequence.blocks[0];block.folders['flowhmr'] = str(self.base/'other')
        if w.model_choice.currentData() == 'flowhmr':block.folders = {'exmr':str(self.base/'other')}
        before = c.sequence.to_dict();w.slider.set_clip(15, 60);pump(lambda:False, .5)
        self.assertEqual(self.range(), (.5, 2.));self.assertEqual(c.sequence.to_dict(), before)

    def test_trim_in_a_tracked_block_changes_nothing(self):
        self.open_first();c, w = self.c, self.w;block = self.loaded(self.folder('one', self.manifest()))
        before = c.sequence.to_dict()
        w.slider.set_clip(15, 60);self.assertEqual(self.range(), (.5, 2.))
        self.assertEqual(c.sequence.to_dict(), before);self.assertEqual(self.ranges(), [(0., 3.)]);self.assertTrue(block.folders)

    def prepared(self, root, manifest):
        """The worker's `prepared` event through the window's real event handler, while the job runs."""
        w = self.w;done = self.busy()
        w.handle_event(dict(type='prepared', project=str(root), manifest=manifest));done()

    def test_prepared_remembers_the_folder(self):
        self.open_first();c, w = self.c, self.w;model = w.model_choice.currentData()
        self.assertEqual(c.view.states[c.current].kind, 'untracked')
        root, manifest = self.found('one', .5, 2.5);self.prepared(root, manifest)
        block = c.sequence.block(c.current)
        self.assertEqual((block.start, block.end), (.5, 2.5));self.assertEqual(Path(block.folders[model]), root.resolve())
        self.assertEqual(c.view.states[block.id].kind, 'found');self.assertEqual(c.view.states[block.id].label, '1 person')
        self.assertTrue(Path(w.project).resolve().is_relative_to(root.resolve()))
        self.assertIsNone(c.sequence.path)   # one block: no file (ruling of fix round 1)

    def test_prepared_again_with_another_range_follows(self):
        self.open_first();c, w = self.c, self.w;model = w.model_choice.currentData()
        first, manifest = self.found('one', .5, 2.5);self.prepared(first, manifest)
        second, manifest = self.found('two', 1., 3.);self.prepared(second, manifest)
        block = c.sequence.block(c.current)
        self.assertEqual((block.start, block.end), (1., 3.));self.assertEqual(block.folders, {model:str(second.resolve())})
        self.assertNotEqual(c.view.states[block.id].kind, 'untracked');self.assertTrue((first/'people.json').is_file())

    def test_prepared_from_another_video_is_not_remembered(self):
        self.open_first();c = self.c;root, manifest = self.found('one', .5, 2.5)
        other = dict(manifest, source=dict(path=str(self.v2), fps=30.))
        (root/'project.json').write_text(json.dumps(other), encoding='utf-8');self.prepared(root, other)
        self.assertEqual(c.sequence.blocks[0].folders, {});self.assertEqual(self.ranges(), [(0., 3.)])

    def test_prepared_remembers_for_the_clip_model(self):
        self.open_first();c, w = self.c, self.w;start = w.model_choice.currentData();other = 'gvhmr' if start == 'exmr' else 'exmr'
        manifest = self.manifest(.5, 2.5, other, exmr_options=dict(static_camera=True, smoothing='none'));root = self.folder('one', manifest)
        with mock.patch.object(w, 'load_project', return_value=True):self.prepared(root, manifest)   # the combo still says the start model
        self.assertEqual(c.sequence.blocks[0].folders, {other:str(root.resolve())})

    def test_a_sequence_error_never_fails_the_job(self):
        self.open_first();c, w = self.c, self.w;root, manifest = self.found('one', .5, 2.5)
        with mock.patch.object(c.sequence, 'remember', side_effect=RuntimeError('broken')):self.prepared(root, manifest)
        self.assertFalse(w.failed);self.assertIn('RuntimeError: broken', w.log.toPlainText())
        self.assertEqual(w.status.text(), 'The sequence could not be updated: broken. See the Console.')
        with mock.patch.object(c, '_project', side_effect=RuntimeError('project')):c.opened_project(root)
        self.assertEqual(w.status.text(), 'The sequence could not be updated: project. See the Console.')
        with mock.patch('mocap.sequence_ui.video_length', side_effect=RuntimeError('video')):c.opened_video(self.v2)
        self.assertEqual(w.status.text(), 'The sequence could not be updated: video. See the Console.')

    def test_delete_keeps_files(self):
        first, second = self.two();c, w = self.c, self.w;folder = self.folder('one', self.manifest());self.loaded(folder)
        c.delete();self.assertTrue((folder/'project.json').is_file())
        self.assertEqual([b.id for b in c.sequence.blocks], [second.id]);self.assertEqual(c.current, second.id)
        self.ready(self.v2);self.assertIsNone(w.project)
        c.view.removed.emit(second.id)   # the Delete key on the last block
        self.assertEqual([b.id for b in c.sequence.blocks], [second.id]);self.assertEqual(c.current, second.id)
        self.assertEqual(w.status.text(), LAST);self.assertEqual(Path(w.video_path), self.v2)

    def test_busy_refuses_everything(self):
        first, second = self.two();c, w = self.c, self.w;before = c.sequence.to_dict()
        done = self.busy()
        self.assertFalse(c.view.editable);self.assertEqual(c.view.busy, first.id)
        for button in (c.add_button, c.cut_button, c.delete_button):self.assertFalse(button.isEnabled(), button.text())
        self.assertFalse(c.load(second.id));self.assertEqual(w.status.text(), 'A job is running. This block can be opened when it is done.')
        c.view.scrubbed.emit(4.5, True);self.assertAlmostEqual(c.view.playhead, 0.)   # the playhead stays in the busy block
        with mock.patch.object(c, 'ask', return_value=True):
            c.cut();c.delete();c.delete(second.id);c.move(first.id, 1);c.resize(first.id, .5, 2.)
            c.view.resized.emit(second.id, .5, 2.);c.view.moved.emit(second.id, 0)
        self.assertEqual(c.sequence.to_dict(), before);self.assertEqual(c.current, first.id);self.assertEqual(Path(w.video_path), self.v1)
        done()
        self.assertTrue(c.view.editable);self.assertIsNone(c.view.busy)
        for button in (c.add_button, c.cut_button, c.delete_button):self.assertTrue(button.isEnabled(), button.text())

    def test_reconstruct_in_parts_counts_as_busy(self):
        first, second = self.two();c, w = self.c, self.w
        w.parts_runner = object();self.addCleanup(lambda:(setattr(w, 'parts_runner', None), w.update_readiness()))
        w.update_readiness()   # as parts_runner.start does, also between its parts when w.busy is False
        self.assertFalse(w.busy);self.assertFalse(c.view.editable);self.assertEqual(c.view.busy, first.id);self.assertFalse(c.cut_button.isEnabled())
        self.assertFalse(c.load(second.id));self.assertEqual(w.status.text(), 'A job is running. This block can be opened when it is done.')
        w.parts_runner = None;w.update_readiness()   # as its stop does
        self.assertTrue(c.view.editable);self.assertIsNone(c.view.busy);self.assertTrue(c.cut_button.isEnabled())

    def test_missing_video_and_missing_folder_do_not_crash(self):
        first, second = self.two();c, w = self.c, self.w;model = w.model_choice.currentData()
        self.v2.unlink();gone = self.base/'gone';first.folders[model] = str(gone);c.refresh()
        self.assertEqual(c.view.states[second.id].kind, 'missing')
        c.view.selected.emit(second.id)
        self.assertEqual(w.status.text(), f'Cannot open the block: The video file is missing: {self.v2}');self.assertEqual(c.current, first.id)
        c.open_block(second.id);self.assertEqual(c.current, first.id);self.assertFalse(c.at_clip_level())
        self.assertTrue(c.load(first.id));self.assertEqual(w.status.text(), f'The saved results are missing: {gone}')
        self.ready(self.v1);w.slider.setValue(30)
        with mock.patch.object(c, 'ask', return_value=True):c.cut()
        self.assertEqual(self.ranges(), [(0., 1.), (1., 3.), (0., 3.)])
        c.delete(c.sequence.blocks[1].id);self.assertEqual(c.current, c.sequence.blocks[1].id)   # the neighbour's video is missing
        self.assertTrue(w.status.text().startswith('Cannot open the block:'), w.status.text())

    def test_model_switch_reopens_the_block(self):
        self.open_first();c, w = self.c, self.w;start = w.model_choice.currentData();other = 'gvhmr' if start == 'exmr' else 'exmr'
        folder = self.folder('one', self.manifest(model=other, exmr_options=dict(static_camera=True, smoothing='none')))
        block = c.sequence.blocks[0];block.folders[other] = str(folder);c.refresh()
        w.model_choice.setCurrentIndex(w.model_choice.findData(other))
        self.assertIsNotNone(w.project, w.status.text());self.assertTrue(Path(w.project).resolve().is_relative_to(folder.resolve()))
        w.model_choice.setCurrentIndex(w.model_choice.findData(start))   # the window drops the other model's project
        self.ready(self.v1);self.assertIsNone(w.project);self.assertEqual(self.range(), (0., 3.))
        self.assertEqual(self.ranges(), [(0., 3.)]);self.assertEqual(block.folders, {other:str(folder)})

    def named(self, name, model, start=0., end=3.):
        """A named project (File > Open Project picks its project.json) whose active model clip is a tracked() folder."""
        from mocap import project_workspace
        folder = self.tracked(name, model, start, end);state = json.loads((folder/'project.json').read_text(encoding='utf-8'))
        (folder/'project.json').write_text(json.dumps(dict(state, workspace_root='..')), encoding='utf-8')
        return project_workspace.create(folder.parent, None, dict(path=str(self.v1), duration=3., fps=30., frames=90), model), folder

    def test_open_project_and_add_project(self):
        c, w = self.c, self.w;self.two()
        named, folder = self.named('one', 'exmr', .5, 2.5)
        with mock.patch('mocap.gui.QFileDialog.getOpenFileName', lambda *a, **k:(str(named/'project.json'), '')):w.choose_project()
        self.assertTrue(Path(w.project).resolve().is_relative_to(folder.resolve()), w.status.text())
        self.assertEqual(len(c.sequence.blocks), 1);block = c.sequence.blocks[0]
        self.assertEqual((block.video, block.start, block.end, block.length), (str(self.v1), .5, 2.5, 3.))
        self.assertEqual(block.folders, {'exmr':str(folder.resolve())});self.assertEqual(c.current, block.id)
        self.assertTrue(c.at_clip_level());self.assertEqual(c.view.states[block.id].kind, 'tracked')
        other = self.folder('two', self.manifest(1., 2., 'gvhmr'))
        added = c.add_project(other);self.assertIsNotNone(added)
        self.assertEqual([b.id for b in c.sequence.blocks], [block.id, added.id])
        self.assertEqual((added.start, added.end, added.folders), (1., 2., {'gvhmr':str(other.resolve())}))
        gemx = self.folder('three', self.manifest(model='gemx'))
        self.assertIsNone(c.add_project(gemx));self.assertEqual(len(c.sequence.blocks), 2)
        self.assertEqual(w.status.text(), 'Only GVHMR, ExMR and FlowHMR projects can be added to a sequence.')
        self.assertIsNone(c.add_project(self.base/'nothing'));self.assertTrue(w.status.text().startswith('Cannot add the project:'))
        named, folder = self.named('four', 'flowhmr', 2., 3.)
        with mock.patch('mocap.sequence_ui.QFileDialog.getOpenFileName', return_value=(str(named/'project.json'), '')):c.choose_project()
        self.assertEqual(len(c.sequence.blocks), 3);self.assertEqual(c.sequence.blocks[2].folders, {'flowhmr':str(folder.resolve())})

    def test_sequence_files(self):
        self.two();c, w = self.c, self.w;path, saved = c.sequence.path, c.sequence.to_dict()
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), saved)
        c.new();self.assertEqual([(b.video, b.start, b.end) for b in c.sequence.blocks], [(str(self.v1), 0., 3.)])   # the loaded video
        self.assertTrue(c.open(path));self.assertEqual(c.sequence.to_dict(), saved);self.assertEqual(c.sequence.path, path)
        self.assertEqual(c.current, c.sequence.blocks[0].id);self.assertFalse(c.at_clip_level())
        for name, text in (('corrupt.json', '{'), ('newer.json', json.dumps(dict(saved, version=2)))):
            bad = self.base/name;bad.write_text(text, encoding='utf-8')
            self.assertFalse(c.open(bad));self.assertTrue(w.status.text().startswith('Cannot open the sequence:'), w.status.text())
            self.assertEqual(c.sequence.to_dict(), saved);self.assertEqual(c.sequence.path, path)
        self.assertIn('newer version', w.status.text())
        c.new();c.add_video(self.v1);c.add_video(self.v2)
        self.assertNotEqual(c.sequence.path, path);self.assertTrue(c.sequence.path.is_file());self.assertTrue(path.is_file())
        with mock.patch('mocap.sequence_ui.QFileDialog.getOpenFileName', return_value=(str(path), '')) as dialog:c.choose_sequence()
        self.assertEqual(Path(dialog.call_args[0][2]), self.base/'projects/sequences');self.assertEqual(c.sequence.to_dict(), saved)

    def test_a_video_loaded_by_another_path_is_added_to_a_sequence(self):
        first, second = self.two();c, w = self.c, self.w;path = c.sequence.path;self.assertFalse(c.at_clip_level())
        w.load_video(self.v1);self.ready(self.v1)   # as a drop on the window or the guide's sample video
        self.assertEqual([b.video for b in c.sequence.blocks], [str(self.v1), str(self.v2), str(self.v1)])
        self.assertEqual(c.sequence.blocks[:2], [first, second]);self.assertEqual(c.current, c.sequence.blocks[2].id)
        self.assertFalse(c.at_clip_level())   # the timeline stays where it is
        self.assertEqual(c.sequence.path, path);self.assertEqual(json.loads(path.read_text(encoding='utf-8')), c.sequence.to_dict())

    def test_a_video_loaded_by_another_path_is_added_to_a_saved_one_block_sequence(self):
        first, second = self.two();c, w = self.c, self.w;c.delete(second.id);self.assertEqual(len(c.sequence.blocks), 1)
        w.load_video(self.v2);self.ready(self.v2)
        self.assertEqual([b.video for b in c.sequence.blocks], [str(self.v1), str(self.v2)]);self.assertEqual(c.current, c.sequence.blocks[1].id)

    def test_a_video_loaded_by_another_path_starts_a_new_one_block_sequence(self):
        self.open_first();c, w = self.c, self.w;old = c.sequence.blocks[0]
        w.load_video(self.v2);self.ready(self.v2)
        self.assertEqual([b.video for b in c.sequence.blocks], [str(self.v2)]);self.assertNotEqual(c.sequence.blocks[0], old)
        self.assertEqual(c.current, c.sequence.blocks[0].id);self.assertTrue(c.at_clip_level());self.assertIsNone(c.sequence.path)

    def test_the_same_video_loaded_by_another_path_is_added_to_a_sequence(self):
        self.open_first();c, w = self.c, self.w;w.slider.setValue(30);c.cut();self.assertEqual(len(c.sequence.blocks), 2)
        w.load_video(self.v1);self.ready(self.v1)   # the loaded block's own video, dropped on the window
        self.assertEqual(self.ranges(), [(0., 1.), (1., 3.), (0., 3.)]);self.assertEqual(c.current, c.sequence.blocks[2].id)
        self.assertEqual(self.range(), (0., 3.))

    def reopen(self, flag):
        """The window opens its video again and keeps its own media info, as leaving PEAR (pear_ui.pear_leave) or MAMMA
        (mamma_ui.mamma_mode_changed) does."""
        w = self.w;info = w.info
        w._pear_media_restore = dict(info=w.info, value=w.slider.value());setattr(w, flag, True)
        w.video_view.open(Path(w.video_path))
        self.assertTrue(pump(lambda:not getattr(w, flag), 10), flag);APP.processEvents();self.assertIs(w.info, info)

    def test_a_video_the_window_reopens_keeps_the_sequence(self):
        c, w = self.c, self.w;self.open_first();w.slider.set_clip(15, 60);pump(lambda:False, .5)
        for flag in ('_pear_restoring_media', '_mamma_primary_opening'):
            before = c.sequence.to_dict();self.reopen(flag)
            self.assertEqual(c.sequence.to_dict(), before, flag);self.assertEqual(self.ranges(), [(.5, 2.)])
        c.add_video(self.v2);APP.processEvents();before = c.sequence.to_dict()
        self.reopen('_pear_restoring_media');self.assertEqual(c.sequence.to_dict(), before)   # not added either

    # The window's own model and project switches win

    def workspace(self, name, models, start=.5, end=2.5):
        """A named project as mocap/video.py prepare leaves it: its project.json with workspace_version and the source
        (mocap/project_workspace.py create), one finished clip per model at <root>/<model> whose project.json carries
        workspace_root and the project's source, each set active for its model."""
        from mocap import project_workspace
        root = project_workspace.create(self.base/'projects'/name, None, dict(path=str(self.v1), duration=3., fps=30., frames=90), models[0])
        source = project_workspace.read(root)['source']
        for model in models:
            folder = self.tracked(name, model, start, end);state = json.loads((folder/'project.json').read_text(encoding='utf-8'))
            (folder/'project.json').write_text(json.dumps(dict(state, workspace_root='..', source=source)), encoding='utf-8')
            project_workspace.set_active(root, model, folder)
        project_workspace.set_active(root, models[0], root/models[0])
        return root.resolve()

    def pick(self, model):
        self.w.model_choice.setCurrentIndex(self.w.model_choice.findData(model));pump(lambda:False, .3)

    def test_a_model_switch_in_a_project_opens_the_windows_model_folder(self):
        c, w = self.c, self.w;self.pick('exmr');root = self.workspace('one', ('exmr', 'gvhmr'))
        with mock.patch('mocap.gui.QFileDialog.getOpenFileName', lambda *a, **k:(str(root/'project.json'), '')):w.choose_project()
        self.assertEqual(Path(w.project).resolve(), root/'exmr', w.status.text());block = c.sequence.blocks[0]
        with mock.patch.object(w, 'load_video') as video:self.pick('gvhmr')
        self.assertEqual(video.call_count, 0);self.assertEqual(Path(w.project).resolve(), root/'gvhmr');self.assertEqual(Path(w.workspace_root), root)
        self.assertEqual(block.folders, {'exmr':str(root/'exmr'), 'gvhmr':str(root/'gvhmr')});self.assertEqual(c.current, block.id)
        self.assertEqual(c.view.states[block.id].kind, 'tracked')   # GVHMR's own initial result
        with mock.patch.object(w, 'load_video') as video:self.pick('exmr')
        self.assertEqual(video.call_count, 0);self.assertEqual(Path(w.project).resolve(), root/'exmr');self.assertEqual(len(c.sequence.blocks), 1)

    def test_a_model_switch_without_saved_motion_stays_in_the_project(self):
        """Open Video, Reconstruct with ExMR, then GVHMR: the window stays in the project, so the next Reconstruct uses it."""
        c, w = self.c, self.w;self.pick('exmr');self.open_first();root = self.workspace('one', ('exmr',), 0., 3.)
        clip = root/'exmr';self.prepared(clip, json.loads((clip/'project.json').read_text(encoding='utf-8')))
        self.assertEqual(Path(w.workspace_root), root);title = w.windowTitle();block = c.sequence.blocks[0]
        with mock.patch.object(w, 'load_video') as video, mock.patch.object(w, 'load_project') as project:
            self.pick('gvhmr');c.move(block.id, 0)
        self.assertEqual((video.call_count, project.call_count), (0, 0))
        self.assertEqual(Path(w.workspace_root), root);self.assertIsNone(w.project);self.assertEqual(w.windowTitle(), title)
        self.assertEqual(block.folders, {'exmr':str(clip)})
        self.pick('exmr');self.assertEqual(Path(w.project).resolve(), clip)

    def test_a_project_the_window_loads_by_itself(self):
        """As Save Project As loads its copy: a clip of the loaded block's video and range is adopted; another one is left
        alone, also by later edits, until a block is loaded."""
        first, second = self.two();c, w = self.c, self.w;model = w.model_choice.currentData()
        copy = self.workspace('copy', (model,), 0., 3.)
        self.assertTrue(w.load_project(copy/model))
        self.assertTrue(pump(lambda:first.folders.get(model) == str(copy/model), 5), first.folders)
        other = self.workspace('other', (model,), 1., 2.);self.assertTrue(w.load_project(other/model));pump(lambda:False, .5)
        self.assertEqual(first.folders, {model:str(copy/model)});self.assertEqual(self.ranges(), [(0., 3.), (0., 3.)])
        with mock.patch.object(w, 'load_video') as video, mock.patch.object(w, 'load_project') as project:
            c.move(second.id, 0);c.move(second.id, 1)
        self.assertEqual((video.call_count, project.call_count), (0, 0));self.assertEqual(Path(w.project).resolve(), other/model)
        c.view.selected.emit(second.id);self.ready(self.v2);self.assertIsNone(w.project)   # the user's own click loads again

    def test_the_last_block_cannot_be_deleted(self):
        self.open_first();c, w = self.c, self.w;block = c.sequence.blocks[0];c.back()
        c.delete();self.assertEqual(w.status.text(), LAST)
        c.view.removed.emit(block.id);self.assertEqual(c.sequence.blocks, [block]);self.assertEqual(c.current, block.id)

    def test_new_sequence_keeps_the_loaded_video_as_its_block(self):
        first, second = self.two();c, w = self.c, self.w;c.view.selected.emit(second.id);self.ready(self.v2)
        c.new()
        self.assertEqual([(b.video, b.start, b.end) for b in c.sequence.blocks], [(str(self.v2), 0., 3.)])
        self.assertEqual(c.current, c.sequence.blocks[0].id);self.assertTrue(c.at_clip_level());self.assertIsNone(c.sequence.path)

    def test_new_sequence_without_a_video_is_empty(self):
        c = self.c;c.new()
        self.assertEqual(c.sequence.blocks, []);self.assertFalse(c.at_clip_level());self.assertTrue(c.add_button.isEnabled())
        block = c.add_video(self.v1);self.ready(self.v1);self.assertEqual(c.current, block.id)

    def test_new_and_open_are_refused_while_busy(self):
        self.two();c, w = self.c, self.w;path, before = c.sequence.path, c.sequence.to_dict();self.busy()
        c.new();self.assertEqual(c.sequence.to_dict(), before);self.assertEqual(w.status.text(), BUSY)
        w.status.setText('');self.assertFalse(c.open(path));self.assertEqual(w.status.text(), BUSY);self.assertEqual(c.sequence.path, path)

    def test_a_block_with_an_out_of_date_people_list_opens_its_folder(self):
        self.open_first();c, w = self.c, self.w;root, _ = self.found('one', 0., 3.)
        people = json.loads((root/'people.json').read_text(encoding='utf-8'));people['input'] = [0, 0]
        (root/'people.json').write_text(json.dumps(people), encoding='utf-8')
        block = c.sequence.blocks[0];block.folders[w.model_choice.currentData()] = str(root);c.refresh()
        self.assertEqual(c.view.states[block.id].label, 'people list out of date')
        with mock.patch.object(w, 'load_video') as video:self.assertTrue(c.load(block.id))
        self.assertEqual(video.call_count, 0);self.assertTrue(Path(w.project).resolve().is_relative_to(root.resolve()))
        self.assertEqual(w.status.text(), 'Cannot open project: Person detection is out of date. Run Find person again.')   # as File > Open Project
        self.assertEqual(c.current, block.id);self.assertEqual(c.view.states[block.id].kind, 'found')

    def test_a_click_on_a_block_that_cannot_load_tries_once(self):
        first, second = self.two();c = self.c;self.v2.unlink();c.refresh()
        from PySide6.QtTest import QTest
        with mock.patch.object(c, 'load', wraps=c.load) as load:
            QTest.mouseClick(c.view, Qt.LeftButton, Qt.NoModifier, c.view.block_rect(second.id).center())
        self.assertEqual([call.args[0] for call in load.call_args_list], [second.id]);self.assertEqual(c.current, first.id)

    # Stage 2: people who continue across neighbouring blocks

    def clip(self, block, people, takes, mark, model='exmr'):
        """A finished scene of `model` of the block's 3 s video after Find person (tests/test_parts_window.py scene) with its
        takes (ident, first, end) as the batch leaves them (tests/test_sequence_state.py StateTest.take) and the people list's
        fingerprint `mark`, remembered for the block."""
        video = Path(block.video);manifest = self.manifest(0., 3., model, exmr_options=dict(static_camera=True, smoothing='none'))
        manifest['source'] = dict(path=str(video), fps=30.)
        root = self.base/'projects'/f'{video.stem}_{block.id}'/model;root.parent.mkdir(parents=True, exist_ok=True)
        scene.video = video;scene(root, manifest, people)
        for take in takes:test_sequence_state.StateTest.take(SimpleNamespace(root=root, manifest=manifest), *take)
        self.mark(root, mark);block.folders[model] = str(root)
        return root

    def mark(self, root, mark, people=None):
        """The people list of `root` with a new fingerprint (and only `people` when given), as Find person again leaves it."""
        state = people_store.load_people(root);state['fingerprint'] = mark
        if people is not None:state['people'] = [p for p in state['people'] if p['id'] in people]
        people_store.save_people(root, state)

    def linkable(self, later=BEN, takes=((1, 0, 90),)):
        """Two 3 s blocks tracked with ExMR: Anna alone in the first, `later` with `takes` in the second; then a refresh."""
        self.pick('exmr');first, second = self.two()
        self.clip(first, ANNA, ((1, 0, 90),), 'first');self.clip(second, later, takes, 'second');self.c.refresh()
        return first, second

    def saved(self):
        return json.loads(self.c.sequence.path.read_text(encoding='utf-8'))

    def test_one_person_each_is_linked_by_itself(self):   # after refresh: one link, status names both; unlink -> stays unlinked after another refresh
        first, second = self.linkable();c, w = self.c, self.w;target = (second.id, 1)
        self.assertEqual([(l.source, l.target, l.method, l.prints) for l in c.sequence.links], [((first.id, 1), target, 'smooth', ('first', 'second'))])
        self.assertEqual(w.status.text(), 'Ben now continues Anna. Drag the bar to change it.')
        self.assertEqual((c.view.rows[target], c.view.rows[(first.id, 1)]), (0, 0));self.assertEqual(c.view.row_states, {target:'linked'})
        self.assertEqual(c.sequence.starts(), [0., 4.]);self.assertEqual(self.saved(), c.sequence.to_dict())
        self.assertEqual(self.saved()['links'][0]['target'], [second.id, 1])
        c.refresh();self.assertEqual(len(c.sequence.links), 1)   # made once
        self.assertTrue(c.unlink(target));c.refresh();c.refresh()
        self.assertEqual(c.sequence.links, []);self.assertIn(second.id, c.sequence.decided);self.assertNotIn('links', self.saved())
        self.assertEqual(c.sequence.starts(), [0., 3.]);self.assertFalse(c.unlink(target))

    def test_two_people_are_not_linked_by_themselves(self):
        first, second = self.linkable(TWO, ((1, 0, 90), (2, 0, 90)));c = self.c
        self.assertEqual(c.sequence.links, []);self.assertEqual(c.sequence.decided, set())
        self.assertEqual(c.view.rows, {(first.id, 1):0, (second.id, 1):0, (second.id, 2):1});self.assertEqual(c.sequence.starts(), [0., 3.])

    def test_link_and_unlink_from_the_view(self):         # view.linked.emit(...) -> link saved in the sequence file; view.unlinked.emit(...) -> gone
        first, second = self.linkable(TWO, ((1, 0, 90), (2, 0, 90)));c = self.c
        c.view.linked.emit((first.id, 1), (second.id, 2))
        self.assertEqual(self.saved()['links'], [dict(source=[first.id, 1], target=[second.id, 2], method='smooth', prints=['first', 'second'], model='exmr')])
        self.assertEqual(self.saved()['gaps'], {second.id:1.});self.assertEqual(c.view.rows[(second.id, 2)], 0);self.assertEqual(c.sequence.dropped, [])
        c.view.linked.emit((first.id, 1), (second.id, 1))   # Ben's bar dragged onto Anna's row: he continues her instead of Cleo
        self.assertEqual([l.target for l in c.sequence.links], [(second.id, 1)]);self.assertEqual(self.saved()['links'][0]['target'], [second.id, 1])
        c.view.unlinked.emit((second.id, 1))
        self.assertNotIn('links', self.saved());self.assertEqual(c.sequence.links, []);self.assertEqual(c.sequence.dropped, [])

    def test_person_menu_continues_after_and_unlinks(self):   # Continue after lists the previous block's people, a refused one disabled with the reason; Unlink when linked
        first, second = self.linkable(TWO, ((1, 0, 90), (2, 0, 90)));c = self.c
        from mocap.tooltips import plain_tooltip
        menu = c._person_menu(second.id, 2);self.addCleanup(menu.deleteLater)
        self.assertEqual([a.text() for a in menu.actions()], ['Continue after'])
        after = menu.actions()[0].menu().actions();self.assertEqual([(a.text(), a.isEnabled(), a.isChecked()) for a in after], [('Anna', True, False)])
        after[0].trigger();self.assertEqual([(l.source, l.target) for l in c.sequence.links], [((first.id, 1), (second.id, 2))])
        menu = c._person_menu(second.id, 2);self.addCleanup(menu.deleteLater)
        self.assertEqual([a.text() for a in menu.actions()], ['Continue after', 'Unlink', 'Join']);self.assertTrue(menu.actions()[0].menu().actions()[0].isChecked())
        menu.actions()[1].trigger();self.assertEqual(c.sequence.links, []);self.assertNotIn('links', self.saved())
        menu = c._person_menu(first.id, 1);self.addCleanup(menu.deleteLater)   # the first block has no block before it
        self.assertFalse(menu.actions()[0].isEnabled())
        root = Path(first.folders['exmr'])
        for take in people_store.load_people(root)['people'][0]['takes']:shutil.rmtree(root/take['project'])   # Anna's take is gone
        c.refresh();menu = c._person_menu(second.id, 1);self.addCleanup(menu.deleteLater)
        anna = menu.actions()[0].menu().actions()[0]
        self.assertFalse(anna.isEnabled());self.assertEqual(plain_tooltip(anna.toolTip()), 'Anna has no finished take in the earlier block.')
        with mock.patch.object(c, '_person_menu') as built:c.view.menu_requested.emit(second.id, 1, QPoint(5, 5))
        built.assert_called_once_with(second.id, 1);built.return_value.exec.assert_called_once_with(QPoint(5, 5))

    def test_refused_link_says_why(self):                 # later person with two takes -> status is the 'several takes' text, no link
        first, second = self.linkable(BEN, ((1, 0, 40), (1, 50, 90)));c, w = self.c, self.w
        self.assertEqual(c.sequence.links, [])   # not linked by itself either
        c.view.linked.emit((first.id, 1), (second.id, 1))
        self.assertEqual(w.status.text(), 'Ben has several takes in this block. Reconstruct the block in one piece, or join parts that share frames with Fill hidden ranges, first.')
        self.assertEqual(c.sequence.links, []);self.assertFalse(c.link((first.id, 1), (second.id, 1)));self.assertNotIn('links', self.saved())

    def test_transition_dialog_applies_and_refuses(self): # set_transition(target, smooth, 2.0) -> False with the panel's limit text; (auto, 3.0) -> True, starts() moved by 3.0
        first, second = self.linkable();c, w = self.c, self.w;target = (second.id, 1)
        self.assertFalse(c.set_transition(target, 'smooth', 2.))
        self.assertEqual(w.status.text(), 'Smooth join is limited to 30 frames; this range has 60.')
        self.assertEqual((c.sequence.gap(second.id), c.sequence.link_into(*target).method), (1., 'smooth'))
        self.assertFalse(c.set_transition(target, 'auto', 10.));self.assertEqual(w.status.text(), 'Automatic fill is limited to 290 frames; this range has 300.')
        self.assertTrue(c.set_transition(target, 'auto', 3.))
        self.assertEqual(c.sequence.starts(), [0., 6.]);self.assertEqual(c.sequence.link_into(*target).method, 'auto')
        self.assertEqual((self.saved()['gaps'], self.saved()['links'][0]['method']), ({second.id:3.}, 'auto'))
        seen = []
        def step():
            dialog = APP.activeModalWidget()
            try:
                combo, spin = dialog.findChild(QComboBox), dialog.findChild(QDoubleSpinBox)
                note = next(label for label in dialog.findChildren(QLabel) if label.objectName() == 'muted')
                ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
                seen.append((dialog.windowTitle(), [combo.itemText(i) for i in range(combo.count())], combo.currentData(), spin.value(),
                             (spin.minimum(), spin.maximum(), spin.singleStep()), note.text()))
                combo.setCurrentIndex(combo.findData('smooth'));spin.setValue(2.);seen.append(note.text())
                ok.click();seen.append((dialog.isVisible(), note.text()))   # refused: the dialog stays with the reason
                combo.setCurrentIndex(combo.findData('auto'));spin.setValue(2.5);seen.append(note.text());ok.click()
            finally:
                if dialog.isVisible():dialog.reject()
        QTimer.singleShot(0, step);c.edit_transition(target)
        self.assertEqual(seen, [('Transition', ['Smooth join', 'Automatic (Experimental)'], 'auto', 3., (.03, 9.6, .1), EXPERIMENTAL), '',
                                (True, 'Smooth join is limited to 30 frames; this range has 60.'), EXPERIMENTAL])
        self.assertEqual((c.sequence.gap(second.id), c.sequence.link_into(*target).method), (2.5, 'auto'));self.assertEqual(self.saved()['gaps'], {second.id:2.5})
        with mock.patch.object(c, 'edit_transition') as edit:c.view.transition_clicked.emit(target)
        edit.assert_called_once_with(target)

    def test_transition_note_is_shown_whole(self):      # the note that appears after the dialog opened (Experimental, a refusal) is not cut off
        first, second = self.linkable();c = self.c;seen = []
        def step():
            dialog = APP.activeModalWidget()
            try:
                combo, spin = dialog.findChild(QComboBox), dialog.findChild(QDoubleSpinBox)
                note = next(label for label in dialog.findChildren(QLabel) if label.objectName() == 'muted')
                ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
                for change in (lambda:combo.setCurrentIndex(combo.findData('auto')), lambda:(combo.setCurrentIndex(combo.findData('smooth')), spin.setValue(3.), ok.click())):
                    change();APP.processEvents()
                    seen.append((bool(note.text()), note.height() >= note.heightForWidth(note.width()), dialog.height() >= dialog.sizeHint().height()))
            finally:
                if dialog.isVisible():dialog.reject()
        QTimer.singleShot(0, step);c.edit_transition((second.id, 1))
        self.assertEqual(seen, [(True, True, True), (True, True, True)])

    def test_find_person_again_drops_links(self):         # rewrite the later block's people.json with a new fingerprint -> refresh drops the link with the status text
        first, second = self.linkable(TWO, ((1, 0, 90), (2, 0, 90)));c, w = self.c, self.w;later = Path(second.folders['exmr'])
        self.assertTrue(c.link((first.id, 1), (second.id, 2)));self.assertIn(second.id, c.sequence.decided)
        self.mark(later, 'again');c.refresh()
        self.assertEqual(c.sequence.links, []);self.assertEqual(w.status.text(), 'The people of two.mp4 changed, so their links were removed.')
        self.assertNotIn(second.id, c.sequence.decided);self.assertNotIn('links', self.saved());self.assertEqual(c.sequence.dropped, [])
        self.mark(later, 'once more', people=(1,));c.refresh()   # one person each now: linked by itself again
        self.assertEqual([(l.source, l.target, l.prints) for l in c.sequence.links], [((first.id, 1), (second.id, 1), ('first', 'once more'))])
        self.assertEqual(w.status.text(), 'Ben now continues Anna. Drag the bar to change it.')

    def test_a_model_switch_keeps_links(self):            # another model's states judge nothing and link nothing; back on ExMR the row is there
        first, second = self.linkable();c = self.c;before = c.sequence.to_dict()
        self.pick('gvhmr');self.assertEqual(c.sequence.to_dict(), before)
        self.pick('flowhmr');self.assertEqual(c.sequence.to_dict(), before)
        self.pick('exmr');self.assertEqual(c.sequence.to_dict(), before);self.assertEqual(c.view.row_states, {(second.id, 1):'linked'})

    def test_a_link_belongs_to_its_model(self):           # ExMR link kept in the file and not shown under GVHMR (which links by itself), exactly as it was back on ExMR, never a 'people changed' status
        first, second = self.linkable(TWO, ((1, 0, 90), (2, 0, 90)));c, w = self.c, self.w;target = (second.id, 2)
        self.clip(first, ANNA, ((1, 0, 90),), 'g1', 'gvhmr');self.clip(second, BEN, ((1, 0, 90),), 'g2', 'gvhmr')   # other lists, no person 2
        self.assertTrue(c.link((first.id, 1), target));self.assertTrue(c.set_transition(target, 'auto', 2.))
        exmr = [dict(source=[first.id, 1], target=[second.id, 2], method='auto', prints=['first', 'second'], model='exmr')]
        said = mock.patch.object(w.status, 'setText', wraps=w.status.setText);shown = said.start();self.addCleanup(said.stop)
        opens = mock.patch.object(w, 'load_project', return_value=True);opens.start();self.addCleanup(opens.stop)   # the synthetic GVHMR scene has no motion preview to open
        self.pick('gvhmr');self.assertEqual(w.model_choice.currentData(), 'gvhmr')
        self.assertEqual([(l.source, l.target, l.model) for l in c.sequence.model_links()], [((first.id, 1), (second.id, 1), 'gvhmr')])
        self.assertEqual(c.view.row_states, {(second.id, 1):'linked'});self.assertEqual(c.sequence.gap(second.id), 2.)   # one length per pair
        self.assertEqual([l for l in self.saved()['links'] if l['model'] == 'exmr'], exmr)
        self.pick('exmr')
        self.assertEqual([(l.source, l.target, l.method, l.prints) for l in c.sequence.model_links()], [((first.id, 1), target, 'auto', ('first', 'second'))])
        self.assertEqual([l for l in self.saved()['links'] if l['model'] == 'exmr'], exmr);self.assertEqual(c.view.row_states, {target:'linked'})
        statuses = [str(call.args[0]) for call in shown.call_args_list]
        self.assertFalse([text for text in statuses if 'changed' in text], statuses)
        self.assertIn('Ben now continues Anna. Drag the bar to change it.', statuses)   # GVHMR's own link
        self.pick('gvhmr');self.assertEqual(len(c.sequence.model_links()), 1);self.assertEqual(len(c.sequence.links), 2)   # not made again

    def test_relink_by_drag_keeps_the_method(self):
        self.pick('exmr');first, second = self.two();c = self.c
        for block, mark in ((first, 'first'), (second, 'second')):self.clip(block, TWO, ((1, 0, 90), (2, 0, 90)), mark)
        c.refresh();self.assertTrue(c.link((first.id, 1), (second.id, 2)));self.assertTrue(c.set_transition((second.id, 2), 'auto', 2.))
        c.view.linked.emit((first.id, 2), (second.id, 2))   # Cleo's bar dragged onto the earlier block's other row
        self.assertEqual([(l.source, l.target, l.method) for l in c.sequence.links], [((first.id, 2), (second.id, 2), 'auto')])
        self.assertTrue(c.link((first.id, 1), (second.id, 1)));self.assertEqual(c.sequence.link_into(second.id, 1).method, 'smooth')   # a bar without a link

    def test_a_yielded_window_links_nothing(self):        # while the window shows a project of its own, nothing is linked, dropped or announced
        self.pick('exmr');first, second = self.two();c, w = self.c, self.w;c._yielded = True
        self.clip(first, ANNA, ((1, 0, 90),), 'first');self.clip(second, BEN, ((1, 0, 90),), 'second');w.status.setText('');c.refresh()
        self.assertEqual((c.sequence.links, w.status.text()), ([], ''))
        c._yielded = False;c.refresh();self.assertEqual(len(c.sequence.links), 1)

    def test_cut_drops_links(self):                      # cut of a linked block -> no links, sequence saved
        first, second = self.linkable();c, w = self.c, self.w;self.assertEqual(len(c.sequence.links), 1)
        w.slider.setValue(30)
        with mock.patch.object(c, 'ask', return_value=True):c.cut()
        self.assertEqual(self.ranges(), [(0., 1.), (1., 3.), (0., 3.)]);self.assertEqual(c.sequence.links, []);self.assertEqual(c.sequence.dropped, [])
        self.assertEqual(self.saved(), c.sequence.to_dict());self.assertNotIn('links', self.saved());self.assertEqual(c.sequence.starts(), [0., 1., 3.])

    def test_busy_refuses_linking(self):
        self.pick('exmr');first, second = self.two();c, w = self.c, self.w;target = (second.id, 1);done = self.busy()
        self.clip(first, ANNA, ((1, 0, 90),), 'first');self.clip(second, BEN, ((1, 0, 90),), 'second');c.refresh()
        self.assertEqual(c.sequence.links, []);before = c.sequence.to_dict()   # nothing is linked by itself while a job runs
        for refused in (lambda:c.link((first.id, 1), target), lambda:c.unlink(target), lambda:c.set_transition(target, 'auto', 2.)):
            w.status.setText('');self.assertFalse(refused());self.assertEqual(w.status.text(), BUSY)
        with mock.patch('mocap.sequence_ui.QMenu') as shown, mock.patch('mocap.sequence_ui.QDialog') as dialog:
            w.status.setText('');c.person_menu(second.id, 1, QPoint());self.assertEqual(w.status.text(), BUSY)
            w.status.setText('');c.edit_transition(target);self.assertEqual(w.status.text(), BUSY)
        self.assertEqual((shown.call_count, dialog.call_count), (0, 0));self.assertEqual(c.sequence.to_dict(), before)
        done();self.assertEqual([l.target for l in c.sequence.links], [target])   # made once the job ended
        self.pick('gemx');w.status.setText('')
        self.assertFalse(c.unlink(target));self.assertEqual(w.status.text(), OTHER);self.assertEqual(len(c.sequence.links), 1)

    def test_review_and_auto_advance_still_work(self):    # the other session's window test still passes with a linked row present
        first, second = self.linkable();controller, window = self.c, self.w
        self.assertEqual(len(controller.sequence.links), 1)
        controller.sequence.set_review(first.id, dict(start=.5, end=1., result=None))
        controller.sequence.set_review(second.id, dict(start=1.5, end=2., result=None))
        controller.back()
        self.assertEqual(controller.review_bounds(), (15, 30))
        self.assertEqual(self.range(), (0., 3.))
        controller.auto_advance.setChecked(True)
        window.slider.setValue(0)
        window.toggle_play()
        self.assertEqual(window.slider.value(), 15)
        self.assertTrue(controller.advance_review())
        self.assertTrue(pump(lambda:controller.current == second.id and window.project is not None and window.video_readable), window.status.text())
        self.assertTrue(Path(window.project).resolve().is_relative_to(Path(second.folders['exmr']).resolve()))
        self.assertEqual(controller.review_bounds(), (45, 60))
        self.assertTrue(pump(lambda:window.timer.isActive(), 5))
        window.stop_playback();self.assertEqual(len(controller.sequence.links), 1)


class BuildTest(unittest.TestCase):
    def test_a_changed_timeline_panel_leaves_the_window_without_the_sequence(self):
        """build takes the action bar by its place above the slider; another layout is reported, the app starts without the
        sequence and the window's three hooks do nothing."""
        from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget
        from mocap import sequence_ui
        text = 'The Timeline panel changed; the sequence is not available.'
        for order in (('header', 'slider'), ('header', 'actions', 'other', 'slider'), ('header', 'actions')):
            w = QWidget();self.addCleanup(w.deleteLater);menu = QMenu();self.addCleanup(menu.deleteLater)
            w.timeline_panel, w.slider, w.log, w.status = QWidget(w), QWidget(), QTextEdit(w), QLabel(w)
            layout = QVBoxLayout(w.timeline_panel)
            for name in order:layout.addWidget(w.slider if name == 'slider' else QWidget())
            sequence_ui.build(w, menu)
            self.assertEqual(w.log.toPlainText(), text);self.assertEqual(w.status.text(), text)
            self.assertEqual(layout.count(), len(order));self.assertEqual(menu.actions(), [])   # nothing taken out or added
            self.assertIsNone(w.sequence_ui.opened_video(Path('a.mp4')));self.assertIsNone(w.sequence_ui.opened_project(Path('a')))
            self.assertIsNone(w.sequence_ui.prepared(Path('a'), {}))


if __name__ == '__main__':unittest.main()
