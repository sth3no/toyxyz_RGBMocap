"""Tutorial chapters: route, wording and the rules that end, skip or repeat a step, on a stand-in window."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QRect, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGroupBox, QLabel, QListWidget, QMainWindow, QProgressBar,
    QPushButton, QVBoxLayout, QWidget)

from mocap import tutorial_steps as steps
from mocap.hidden_ranges_ui import HiddenRangesPanel
from mocap.timeline_editor import TimelineSlider

APP = QApplication.instance() or QApplication([])


class Stand(QMainWindow):
    """The controls and state the chapters read, under the names MainWindow uses."""
    def __init__(self):
        super().__init__()
        self.resize(1100, 700)
        body = QWidget();self.setCentralWidget(body);layout = QVBoxLayout(body)
        self.model_choice = QComboBox()
        for key in ('exmr', 'flowhmr', 'gvhmr', 'mamma', 'pear'):self.model_choice.addItem(key, key)
        self.model_help, self.find_person_button, self.run_button, self.details, self.cancel_button, self.play_button, self.correction_button = (
            QPushButton(name) for name in ('guide', 'find', 'run', 'console', 'cancel', 'play', 'apply'))
        self.models_group, self.correction_group = QGroupBox('models'), QGroupBox('correction')
        self.auto_button = QPushButton('auto')
        self.people_list, self.progress, self.status = QListWidget(), QProgressBar(), QLabel('status')
        self.video_view, self.motion_view, self.bone_overlay = QWidget(), QWidget(), QCheckBox('bones')
        self.slider = TimelineSlider();self.slider.setRange(0, 99)
        self.person_takes, self.hidden_ranges = QComboBox(), HiddenRangesPanel()
        for widget in (self.model_choice, self.auto_button, self.model_help, self.models_group, self.find_person_button, self.people_list, self.person_takes,
                       self.hidden_ranges, self.run_button, self.progress, self.status, self.details, self.cancel_button, self.video_view,
                       self.motion_view, self.play_button, self.bone_overlay, self.correction_group, self.correction_button, self.slider):
            layout.addWidget(widget)
        menu = self.menuBar().addMenu('File')
        self.export_action = menu.addAction('Export...')
        self.timer = QTimer(self)
        self.video_readable = self.busy = self.failed = self.cancelled = self.models_missing = self._pear_active = False
        self._batch_has_motion = False
        self.info = self.preview_data = None
        self.loaded = []

    def load_video(self, path):self.loaded.append(path)


class ChapterCase(unittest.TestCase):
    chapter = None

    def setUp(self):
        self.w = Stand();self.w.show();APP.processEvents()
        def close():
            self.w.close();self.w.deleteLater();APP.processEvents()
        self.addCleanup(close)

    def step(self, key):return next(s for s in self.chapter.steps if s.key == key)

    def text(self, key):
        text = self.step(key).text
        return text(self.w) if callable(text) else text


class BasicsTest(ChapterCase):
    chapter = steps.BASICS

    def test_chapter_is_registered_first(self):
        self.assertEqual(list(steps.CHAPTERS)[0], 'basics');self.assertIs(steps.CHAPTERS['basics'], steps.BASICS)
        self.assertEqual((steps.BASICS.key, steps.BASICS.title), ('basics', 'From video to export'))

    def test_keys_are_unique_and_in_the_route_order(self):
        self.assertEqual([s.key for s in steps.BASICS.steps],
                         ['model', 'models', 'video', 'trim', 'people', 'run', 'wait', 'review', 'correct', 'export'])

    def test_every_text_is_plain_and_short(self):
        for step in steps.BASICS.steps:
            with self.subTest(step=step.key):
                text = self.text(step.key)
                self.assertTrue(step.title and text);self.assertLessEqual(len(text), 330);self.assertNotIn('<', text)

    def test_every_target_is_a_control_or_a_rectangle(self):
        for step in steps.BASICS.steps:
            with self.subTest(step=step.key):
                targets = step.targets(self.w)
                self.assertTrue(targets);self.assertTrue(all(isinstance(t, (QWidget, QRect)) for t in targets))

    def test_the_route_says_where_reconstruct_3d_is_and_why_a_checked_person_may_be_missing(self):
        self.assertIn('at the bottom of the right panel', self.text('run'))
        self.assertIn('appears in the 3D view once reconstructed', self.text('people'))
        self.assertIn('Reconstruct 3D creates what is missing', self.text('review'))

    def test_models_step_says_what_to_do_when_a_download_fails(self):
        text = self.text('models')
        self.assertIn('Manual Download', text);self.assertIn('Import', text)
        self.assertIn('is not needed while that option is off', text);self.assertIn('HTD Refine', text)

    def test_models_step_is_skipped_when_nothing_is_missing(self):
        step = self.step('models')
        self.assertTrue(step.skip(self.w));self.assertTrue(step.done(self.w, set()))
        self.w.models_missing = True
        self.assertFalse(step.skip(self.w));self.assertFalse(step.done(self.w, set()))

    def test_video_step_ends_when_a_video_is_readable(self):
        step = self.step('video')
        self.assertFalse(step.done(self.w, set()))
        self.w.video_readable, self.w.info = True, {'fps': 30}
        self.assertTrue(step.done(self.w, set()));self.assertEqual(step.card, 'center')

    def test_people_step_needs_the_click_and_the_finished_job(self):
        step = self.step('people')
        self.w.people_list.addItem('Person 1')  # automatic tracking alone
        self.assertFalse(step.done(self.w, set()))
        self.w.busy = True
        self.assertFalse(step.done(self.w, {'find'}))
        self.w.busy = False
        self.assertTrue(step.done(self.w, {'find'}))

    def test_people_step_is_skipped_when_the_button_is_hidden(self):
        self.assertFalse(self.step('people').skip(self.w))
        self.w.find_person_button.hide()
        self.assertTrue(self.step('people').skip(self.w))

    def test_run_step_ends_when_a_reconstruction_started(self):
        self.assertFalse(self.step('run').done(self.w, {'find'}));self.assertTrue(self.step('run').done(self.w, {'run'}))

    def test_run_step_ends_when_a_reconstruction_is_already_running(self):
        step = self.step('run')
        self.w.busy = True  # Find person, a download or the stages before the motion
        self.assertFalse(step.done(self.w, set()))
        self.w._batch_has_motion = True
        self.assertTrue(step.done(self.w, set()))
        self.w.busy = False  # left from the last job
        self.assertFalse(step.done(self.w, set()))

    def test_events_count_a_click_only_when_a_job_started(self):
        events = steps.BASICS.events(self.w)
        self.assertEqual(set(events), {'run', 'find', 'export'})
        self.assertIs(events['run'][0], self.w.run_button.clicked);self.assertIs(events['find'][0], self.w.find_person_button.clicked)
        self.assertFalse(events['run'][1](self.w));self.assertFalse(events['find'][1](self.w));self.assertTrue(events['export'][1](self.w))
        self.w.busy = True
        self.assertTrue(events['run'][1](self.w));self.assertTrue(events['find'][1](self.w))

    def test_wait_step_ends_with_loaded_motion_and_retries_after_a_failure(self):
        step = self.step('wait')
        self.w.busy = True
        self.assertFalse(step.done(self.w, set()));self.assertIsNone(step.retry(self.w, set()))
        self.w.busy, self.w.failed = False, True
        self.assertIn('Reconstruct 3D again', step.retry(self.w, set()))
        self.w.failed, self.w.cancelled = False, True
        self.assertIn('Console', step.retry(self.w, set()))
        self.w.cancelled, self.w.preview_data = False, object()
        self.assertTrue(step.done(self.w, set()));self.assertIsNone(step.retry(self.w, set()))

    def test_review_step_ends_when_playback_runs(self):
        self.assertFalse(self.step('review').done(self.w, set()))
        self.w.timer.start(1000)
        self.assertTrue(self.step('review').done(self.w, set()))

    def test_export_step_ends_when_export_is_chosen(self):
        self.assertTrue(self.step('export').done(self.w, {'export'}));self.assertFalse(self.step('export').done(self.w, set()))

    def test_another_model_blocks_the_chapter(self):
        for model, blocked in (('exmr', False), ('flowhmr', False), ('gvhmr', False), ('mamma', True), ('pear', True)):
            with self.subTest(model=model):
                self.w.model_choice.setCurrentIndex(self.w.model_choice.findData(model))
                self.assertEqual(bool(steps.BASICS.blocked(self.w)), blocked)
        self.assertIn('ExMR, FlowHMR and GVHMR', steps.BASICS.blocked(self.w))

    def test_sample_action_is_offered_only_when_the_clip_exists_and_no_job_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / 'sample.mp4'
            with patch.object(steps, 'SAMPLE', clip):
                self.assertIsNone(steps.sample_action(self.w))
                clip.write_bytes(b'x')
                label, run = steps.sample_action(self.w)
                self.assertEqual(label, 'Import sample video')
                run();self.assertEqual(self.w.loaded, [clip])
                self.w.busy = True
                self.assertIsNone(steps.sample_action(self.w))
        self.assertIs(self.step('video').action, steps.sample_action)

    def test_video_step_points_at_the_sample_video_only_when_it_exists(self):
        self.assertNotIn('sample video', self.text('video'))
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / 'sample.mp4';clip.write_bytes(b'x')
            with patch.object(steps, 'SAMPLE', clip):
                self.assertTrue(self.text('video').endswith(' Or click Import sample video below.'))

    def test_track_rows_follow_the_timeline_geometry(self):
        for scale in (100, 200):
            with self.subTest(scale=scale):
                APP.setProperty('uiScale', scale);self.addCleanup(APP.setProperty, 'uiScale', None)
                self.w.slider.refresh_appearance();APP.processEvents()
                slider, origin = self.w.slider, self.w.slider.mapTo(self.w, self.w.slider.rect().topLeft())
                self.assertEqual(steps.track_rows(self.w, 0, 1), QRect(origin.x(), origin.y() + slider.TRACK_TOP, slider.width(), slider.TRACK_HEIGHT))
                self.assertEqual(steps.track_rows(self.w, 2, 2).height(), 2 * slider.TRACK_HEIGHT)
                self.assertEqual(steps.track_rows(self.w, 2, 2).y(), origin.y() + slider.TRACK_TOP + 2 * slider.TRACK_HEIGHT)

    def test_file_menu_is_the_first_item_of_the_menu_bar(self):
        rect = steps.file_menu(self.w)
        self.assertFalse(rect.isEmpty())
        self.assertEqual(rect.topLeft(), self.w.menuBar().mapTo(self.w, self.w.menuBar().actionGeometry(self.w.menuBar().actions()[0]).topLeft()))


T = lambda a, b: dict(project=f'p/takes/{a:09d}_{b:09d}_scene_v1', first=a, end=b)
PEOPLE = [dict(id=1, name='Person 1', takes=[T(0, 100), T(120, 200)]), dict(id=2, name='Person 2', takes=[T(0, 300)])]
PREPARATION = ('need_video', 'find', 'gap', 'reason')


class HiddenRangesChapterTest(ChapterCase):
    chapter = steps.HIDDEN

    def gaps(self, choices=None, reason=None):
        from mocap.hidden_ranges import gaps
        self.w.hidden_ranges.show_gaps(gaps(PEOPLE[0]['takes']), choices or {}, 30, reason, PEOPLE, 1)
        APP.processEvents()

    def test_chapter_is_registered_second(self):
        self.assertEqual(list(steps.CHAPTERS), ['basics', 'hidden_ranges'])
        self.assertEqual((steps.HIDDEN.key, steps.HIDDEN.title), ('hidden_ranges', 'Hidden ranges'))

    def test_route_order(self):
        self.assertEqual([s.key for s in steps.HIDDEN.steps], [*PREPARATION, 'takes', 'smooth', 'copy', 'automatic', 'fill', 'after', 'same_person'])

    def test_every_text_is_plain_and_short(self):
        self.gaps(reason='Reconstruct and prepare both neighbouring takes first.')
        for step in steps.HIDDEN.steps:
            with self.subTest(step=step.key):
                text = self.text(step.key)
                self.assertTrue(step.title and text);self.assertLessEqual(len(text), 330);self.assertNotIn('<', text)

    def test_every_target_is_a_control_or_a_rectangle(self):
        for step in steps.HIDDEN.steps:
            with self.subTest(step=step.key):
                targets = step.targets(self.w)
                self.assertTrue(targets);self.assertTrue(all(isinstance(t, (QWidget, QRect)) for t in targets))

    def test_honest_wording_of_automatic(self):
        for phrase in ('Experimental', 'plausible, not what the person did', 'Set up…'):self.assertIn(phrase, self.text('automatic'))
        self.assertEqual(self.step('automatic').title, 'Automatic (Experimental)')

    def test_after_step_says_other_people_may_need_apply_correction(self):
        self.assertIn("other people's takes may ask for Apply Correction again", self.text('after'))

    def test_smooth_step_names_its_limit(self):
        self.assertIn('up to 30 frames', self.text('smooth'))

    def test_a_hidden_panel_is_neither_shown_nor_usable(self):
        self.assertFalse(steps.panel_shown(self.w));self.assertFalse(steps.hidden_ranges_usable(self.w));self.assertEqual(steps.panel_reason(self.w), '')
        self.assertEqual([self.step(key).skip(self.w) for key in PREPARATION], [False, False, False, False])

    def test_a_shown_panel_with_a_reason_is_not_usable(self):
        self.gaps(reason='Hidden ranges need a fixed camera in this version.')
        self.assertTrue(steps.panel_shown(self.w));self.assertFalse(steps.hidden_ranges_usable(self.w))
        self.assertEqual(steps.panel_reason(self.w), 'Hidden ranges need a fixed camera in this version.')
        self.assertEqual([self.step(key).skip(self.w) for key in PREPARATION], [True, True, True, False])

    def test_preparation_is_skipped_when_the_panel_is_usable(self):
        self.gaps()
        self.assertTrue(steps.hidden_ranges_usable(self.w))
        self.assertEqual([self.step(key).skip(self.w) for key in PREPARATION], [True, True, True, True])

    def test_reason_step_quotes_the_panel_and_ends_when_the_reason_is_gone(self):
        self.gaps(reason='Reconstruct and prepare both neighbouring takes first.')
        self.assertIn('“Reconstruct and prepare both neighbouring takes first.”', self.text('reason'))
        self.assertFalse(self.step('reason').done(self.w, set()))
        self.gaps()
        self.assertTrue(self.step('reason').done(self.w, set()))

    def test_reason_step_asks_for_the_person_again_when_the_panel_went_away(self):
        self.assertEqual(self.text('reason'), 'Hidden ranges is shown for the selected person. Select the person who gets covered again.')

    def test_need_video_is_skipped_once_a_video_is_open(self):
        self.w.video_readable, self.w.info = True, {'fps': 30}
        self.assertTrue(self.step('need_video').skip(self.w));self.assertFalse(self.step('find').skip(self.w))

    def test_find_step_is_skipped_when_the_button_is_hidden(self):
        self.w.find_person_button.hide()
        self.assertTrue(self.step('find').skip(self.w))

    def test_gap_step_ends_when_the_panel_appears(self):
        self.assertFalse(self.step('gap').done(self.w, set()))
        self.gaps(reason='x')
        self.assertTrue(self.step('gap').done(self.w, set()))

    def test_smooth_step_ends_when_a_method_is_chosen(self):
        self.gaps()
        self.assertFalse(steps.method_chosen(self.w));self.assertFalse(self.step('smooth').done(self.w, set()))
        self.gaps({'100_120': dict(method='smooth')})
        self.assertTrue(steps.method_chosen(self.w));self.assertTrue(self.step('smooth').done(self.w, set()))

    def test_fill_step_ends_when_the_joined_take_is_current(self):
        self.w.person_takes.addItem('1–100 · Ready');self.w.person_takes.addItem('1–200 · Ready · filled')
        self.assertFalse(self.step('fill').done(self.w, set()))
        self.w.person_takes.setCurrentIndex(1)
        self.assertTrue(self.step('fill').done(self.w, set()))

    def test_need_video_mentions_the_sample_clip_only_when_it_exists(self):
        self.assertNotIn('sample video', self.text('need_video'))
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / 'sample.mp4';clip.write_bytes(b'x')
            with patch.object(steps, 'SAMPLE', clip):
                self.assertTrue(self.text('need_video').endswith(' Or click Import sample video below.'))
        self.assertIs(self.step('need_video').action, steps.sample_action)

    def test_live_preview_blocks_the_chapter(self):
        self.assertIsNone(steps.HIDDEN.blocked(self.w))
        self.w._pear_active = True
        self.assertIn('Stop PEAR', steps.HIDDEN.blocked(self.w))

    def test_find_click_is_the_only_event(self):
        events = steps.HIDDEN.events(self.w)
        self.assertEqual(set(events), {'find'});self.assertIs(events['find'][0], self.w.find_person_button.clicked)


if __name__ == '__main__':
    unittest.main()
