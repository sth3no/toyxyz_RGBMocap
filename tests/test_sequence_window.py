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
from test_parts_window import scene, write_result

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
        self.assertTrue(c.sequence.path.is_file())   # the block changed: the sequence has its file

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
        self.assertEqual(json.loads(c.sequence.path.read_text(encoding='utf-8')), c.sequence.to_dict())

    def test_prepared_again_with_another_range_follows(self):
        self.open_first();c, w = self.c, self.w;model = w.model_choice.currentData()
        first, manifest = self.found('one', .5, 2.5);self.prepared(first, manifest)
        second, manifest = self.found('two', 1., 3.);self.prepared(second, manifest)
        block = c.sequence.block(c.current)
        self.assertEqual((block.start, block.end), (1., 3.));self.assertEqual(block.folders, {model:str(second.resolve())})
        self.assertNotEqual(c.view.states[block.id].kind, 'untracked');self.assertTrue((first/'people.json').is_file())

    def test_delete_keeps_files(self):
        first, second = self.two();c, w = self.c, self.w;folder = self.folder('one', self.manifest());self.loaded(folder)
        c.delete();self.assertTrue((folder/'project.json').is_file())
        self.assertEqual([b.id for b in c.sequence.blocks], [second.id]);self.assertEqual(c.current, second.id)
        self.ready(self.v2);self.assertIsNone(w.project)
        c.view.removed.emit(second.id)   # the Delete key
        self.assertEqual(c.sequence.blocks, []);self.assertIsNone(c.current)
        self.assertEqual(Path(w.video_path), self.v2);self.assertTrue(w.info)   # the window keeps what it shows
        self.assertFalse(c.delete_button.isEnabled())

    def test_busy_refuses_everything(self):
        first, second = self.two();c, w = self.c, self.w;before = c.sequence.to_dict()
        done = self.busy()
        self.assertFalse(c.view.editable);self.assertEqual(c.view.busy, first.id)
        for button in (c.add_button, c.cut_button, c.delete_button):self.assertFalse(button.isEnabled(), button.text())
        self.assertFalse(c.load(second.id));self.assertEqual(w.status.text(), 'A job is running. The block opens when it is done.')
        c.view.scrubbed.emit(4.5, True);self.assertAlmostEqual(c.view.playhead, 0.)   # the playhead stays in the busy block
        with mock.patch.object(c, 'ask', return_value=True):
            c.cut();c.delete();c.delete(second.id);c.move(first.id, 1);c.resize(first.id, .5, 2.)
            c.view.resized.emit(second.id, .5, 2.);c.view.moved.emit(second.id, 0)
        self.assertEqual(c.sequence.to_dict(), before);self.assertEqual(c.current, first.id);self.assertEqual(Path(w.video_path), self.v1)
        done()
        self.assertTrue(c.view.editable);self.assertIsNone(c.view.busy)
        for button in (c.add_button, c.cut_button, c.delete_button):self.assertTrue(button.isEnabled(), button.text())

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
        c.new();self.assertEqual(c.sequence.blocks, [])
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

    def test_a_video_loaded_by_another_path_starts_a_new_sequence(self):
        self.two();c, w = self.c, self.w
        w.load_video(self.v2);self.ready(self.v2)   # as a drop on the window or the guide's sample video
        self.assertEqual(len(c.sequence.blocks), 1);self.assertEqual(c.sequence.blocks[0].video, str(self.v2))
        self.assertEqual(c.current, c.sequence.blocks[0].id);self.assertTrue(c.at_clip_level())

    def test_a_click_on_a_block_that_cannot_load_tries_once(self):
        first, second = self.two();c = self.c;self.v2.unlink();c.refresh()
        from PySide6.QtTest import QTest
        with mock.patch.object(c, 'load', wraps=c.load) as load:
            QTest.mouseClick(c.view, Qt.LeftButton, Qt.NoModifier, c.view.block_rect(second.id).center())
        self.assertEqual([call.args[0] for call in load.call_args_list], [second.id]);self.assertEqual(c.current, first.id)


if __name__ == '__main__':unittest.main()
