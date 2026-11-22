"""The tutorial in the real main window, offscreen: menu, first start, hint, appearance and the targets of every step."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QMenu, QPushButton, QWidget

from mocap import paths, tutorial
from mocap.gui import MainWindow
from mocap.tutorial_steps import CHAPTERS

APP = QApplication.instance() or QApplication([])


def application():
    """The app's own font and style, set once as app.py does: without them the settings panel is wider than the window."""
    if not getattr(application, 'configured', False):
        from mocap.theme import configure_application
        configure_application(APP);application.configured = True


class WindowCase(unittest.TestCase):
    def setUp(self):
        application()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.object(paths, 'HOME', Path(tmp.name))
        home.start()
        self.addCleanup(home.stop)
        self.settings = Path(tmp.name) / 'config/local.json'

    def window(self):
        window = MainWindow()
        def close():
            window.close();window.deleteLater();APP.processEvents()
        self.addCleanup(close)
        window.show();APP.processEvents()
        return window


class ReadinessTest(WindowCase):
    def test_window_says_whether_model_files_are_missing(self):
        w = self.window()
        self.assertIs(w.models_missing, 'Warning:' in w.readiness.text())


def press(window, label):
    next(b for b in window.tutorial.card.findChildren(QPushButton) if b.text() == label).click()


class ActionsTest(WindowCase):
    def test_reconstruct_3d_stays_in_view_however_the_settings_are_scrolled(self):
        w = self.window();w.resize(1100, 700);APP.processEvents()
        bar = w.settings_scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)  # the settings do not fit: this is the case that hid the button
        seen = []
        for value in (0, bar.maximum()):
            bar.setValue(value);APP.processEvents()
            for button in (w.run_button, w.cancel_button):
                self.assertEqual(button.visibleRegion().boundingRect().size(), button.size(), (value, button.text()))
            seen.append(w.run_button.mapTo(w, QPoint(0, 0)))
        self.assertEqual(seen[0], seen[1])
        self.assertFalse(w.settings_scroll.isAncestorOf(w.run_button))
        self.assertGreater(seen[0].y(), w.settings_scroll.mapTo(w, QPoint(0, 0)).y() + w.settings_scroll.height() - 1)

    def test_checked_people_without_a_take_are_named_to_the_viewport(self):
        w = self.window()
        w.people_state = dict(people=[dict(id=1, name='Ann', color='#ffffff', checked=True, takes=[dict(project='p')]),
                                      dict(id=3, name='Person 3', color='#ffffff', checked=True),
                                      dict(id=4, name='Bo', color='#ffffff', checked=False)])
        self.addCleanup(setattr, w, 'people_state', None)
        w.sync_scene_visibility()
        self.assertEqual(w.motion_view.scene_waiting, ['Person 3'])

    def test_a_new_viewport_waits_for_nobody(self):
        self.assertEqual(self.window().motion_view.scene_waiting, [])


class MenuTest(WindowCase):
    def test_building_the_window_starts_no_tutorial(self):
        w = self.window()
        self.assertIsNone(w.tutorial);self.assertEqual(tutorial.load_state(), {});self.assertFalse(self.settings.exists())

    def test_help_menu_offers_both_chapters_before_check_system(self):
        w = self.window()
        self.assertEqual([a.text() for a in w.tutorial_actions.values()], ['From video to export', 'Hidden ranges'])
        self.assertEqual(list(w.tutorial_actions), ['basics', 'hidden_ranges'])
        help_menu = next(m for m in w.menuBar().findChildren(QMenu) if m.title() == 'Help')
        self.assertEqual([a.text() for a in help_menu.actions()][:2], ['Tutorial', 'Check System'])
        self.assertTrue(all(a.toolTip() != a.text() for a in w.tutorial_actions.values()))

    def test_every_target_of_both_chapters_exists_in_the_real_window(self):
        w = self.window()
        for chapter in CHAPTERS.values():
            for step in chapter.steps:
                with self.subTest(chapter=chapter.key, step=step.key):
                    targets = step.targets(w)
                    self.assertTrue(targets);self.assertTrue(all(isinstance(t, (QWidget, QRect)) for t in targets))
                    step.skip and step.skip(w);step.done and step.done(w, set());step.retry and step.retry(w, set());step.action and step.action(w)
                    text = step.text(w) if callable(step.text) else step.text
                    self.assertTrue(text)
            chapter.blocked(w);self.assertTrue(chapter.events(w))

    def test_basics_runs_from_the_menu_to_the_end(self):
        w = self.window()
        w.tutorial_actions['basics'].trigger()
        guide = w.tutorial
        self.assertTrue(guide.active);self.assertEqual(guide.chapter.key, 'basics')
        seen = []
        while guide.active:
            seen.append(guide.step.key)
            self.assertTrue(guide.card.isVisible(), guide.step.key)
            self.assertTrue(w.rect().contains(guide.card.geometry()), (guide.step.key, guide.card.geometry()))
            self.assertFalse(any(guide.card.geometry().intersects(hole) for hole in guide.overlay.holes) and guide.step.card != 'center'
                             and len(guide.overlay.holes) == 1, guide.step.key)
            guide.next();APP.processEvents()
        self.assertEqual(seen[0], 'model');self.assertEqual(seen[-1], 'export');self.assertIn('run', seen)
        self.assertEqual(tutorial.load_state()['done'], ['basics'])

    def test_the_first_step_rings_the_model_selector(self):
        w = self.window()
        tutorial.open_chapter(w, 'basics')
        holes = w.tutorial.overlay.holes
        self.assertEqual(len(holes), 2)
        self.assertEqual(holes[0], w.model_choice.visibleRegion().boundingRect().translated(w.model_choice.mapTo(w, QPoint(0, 0))))
        self.assertEqual(w.settings_scroll.horizontalScrollBar().value(), 0)
        self.assertEqual(w.tutorial.overlay.geometry(), w.rect())

    def test_opening_a_chapter_twice_keeps_one_controller(self):
        w = self.window()
        tutorial.open_chapter(w, 'basics');first = w.tutorial;first.next()
        tutorial.open_chapter(w, 'hidden_ranges')
        self.assertIs(w.tutorial, first);self.assertEqual((first.chapter.key, first.step.key), ('hidden_ranges', 'need_video'))

    def test_hidden_ranges_chapter_starts_by_asking_for_a_video(self):
        w = self.window()
        w.tutorial_actions['hidden_ranges'].trigger()
        self.assertEqual(w.tutorial.card.title.text(), 'A video with a hidden person')
        self.assertEqual(w.tutorial.card.status.text(), 'Waiting for a video…')

    def test_closing_the_window_stops_the_chapter_without_recording_it(self):
        w = self.window()
        tutorial.open_chapter(w, 'basics');guide = w.tutorial
        w.close();APP.processEvents()
        self.assertFalse(guide.active);self.assertFalse(guide.timer.isActive());self.assertFalse(guide.hint_timer.isActive())
        self.assertEqual(tutorial.load_state(), {})

    def test_changing_the_appearance_resizes_the_card(self):
        from mocap.theme import apply_appearance
        from mocap.workspace_ui import resize_appearance_controls
        w = self.window()
        tutorial.open_chapter(w, 'basics')
        self.addCleanup(apply_appearance, APP, 'dark', 100)
        apply_appearance(APP, 'light', 200);resize_appearance_controls(w)
        self.assertEqual(w.tutorial.card.width(), 640);self.assertIn('#f1f3f6', w.tutorial.card.styleSheet())
        self.assertTrue(w.rect().contains(w.tutorial.card.geometry()))

    def test_appearance_can_change_before_any_tutorial_exists(self):
        from mocap.workspace_ui import resize_appearance_controls
        resize_appearance_controls(self.window())


class SettingsTest(WindowCase):
    """File > Settings... starts a chapter too."""
    def answered(self, w, answer):
        """Open Settings and let `answer(dialog)` stand for the user."""
        from PySide6.QtWidgets import QDialog
        from mocap.theme import AppearanceDialog
        from mocap.workspace_ui import show_appearance_settings
        def run(dialog):
            answer(dialog)
            return QDialog.DialogCode(dialog.result())
        with patch.object(AppearanceDialog, 'exec', run):show_appearance_settings(w)  # a plain function binds the dialog

    def test_settings_offer_both_chapters(self):
        from mocap.theme import AppearanceDialog
        dialog = AppearanceDialog(self.window());self.addCleanup(dialog.deleteLater)
        self.assertEqual({key: button.text() for key, button in dialog.tutorial_buttons.items()},
                         dict(basics='From video to export', hidden_ranges='Hidden ranges'))
        self.assertIsNone(dialog.chapter)
        self.assertTrue(all(button.toolTip() for button in dialog.tutorial_buttons.values()))

    def test_a_chapter_chosen_in_settings_starts_when_the_dialog_closes(self):
        w = self.window()
        self.answered(w, lambda dialog: dialog.tutorial_buttons['hidden_ranges'].click())
        self.assertEqual(w.tutorial.chapter.key, 'hidden_ranges');self.assertTrue(w.tutorial.card.isVisible())

    def test_ok_or_cancel_in_settings_starts_nothing(self):
        w = self.window()
        self.answered(w, lambda dialog: dialog.accept());self.answered(w, lambda dialog: dialog.reject())
        self.assertIsNone(w.tutorial)

    def test_cancel_after_looking_at_a_chapter_button_starts_nothing(self):
        w = self.window()
        def answer(dialog):
            dialog.chapter = 'basics';dialog.reject()
        self.answered(w, answer)
        self.assertIsNone(w.tutorial)

    def test_a_chapter_button_also_keeps_the_chosen_appearance(self):
        from mocap.theme import apply_appearance, current_appearance, load_appearance
        w = self.window()
        self.addCleanup(apply_appearance, APP, 'dark', 100)
        def answer(dialog):
            dialog.set_choices('light', 125);dialog.tutorial_buttons['basics'].click()
        self.answered(w, answer)
        self.assertEqual((current_appearance(), load_appearance()), (('light', 125), ('light', 125)))
        self.assertEqual(w.tutorial.chapter.key, 'basics');self.assertEqual(w.tutorial.card.width(), 400)

    def test_the_tutorial_says_it_is_in_settings_too(self):
        self.assertIn('File → Settings', tutorial.WELCOME)


class FirstRunTest(WindowCase):
    def test_first_run_shows_the_welcome_once(self):
        w = self.window()
        tutorial.first_run(w, True)
        self.assertTrue(w.tutorial.card.isVisible());self.assertEqual(w.tutorial.card.text.text(), tutorial.WELCOME)
        self.assertEqual(w.tutorial.card.title.text(), 'Welcome to RGB MoCap');self.assertFalse(w.tutorial.active)
        self.assertEqual(tutorial.load_state(), {})  # nothing is saved until the user answers
        press(w, 'Not now')
        self.assertFalse(w.tutorial.card.isVisible());self.assertIs(tutorial.load_state()['seen'], True)
        tutorial.first_run(w, True)
        self.assertFalse(w.tutorial.card.isVisible())

    def test_start_on_the_welcome_opens_the_first_chapter(self):
        w = self.window()
        tutorial.first_run(w, True);press(w, 'Start')
        self.assertEqual(w.tutorial.chapter.key, 'basics');self.assertIs(tutorial.load_state()['seen'], True)

    def test_the_cross_on_the_welcome_is_remembered_too(self):
        w = self.window()
        tutorial.first_run(w, True);press(w, '×')
        self.assertIs(tutorial.load_state()['seen'], True);self.assertFalse(w.tutorial.card.isVisible())

    def test_a_chapter_opened_from_the_menu_over_the_welcome_remembers_it(self):
        w = self.window()
        tutorial.first_run(w, True);w.tutorial_actions['basics'].trigger()
        self.assertIs(tutorial.load_state()['seen'], True)
        w.tutorial.stop();tutorial.first_run(w, True)
        self.assertFalse(w.tutorial.card.isVisible())

    def test_first_run_stays_quiet_when_the_settings_are_unreadable(self):
        self.settings.parent.mkdir(parents=True);self.settings.write_text('{broken', encoding='utf-8')
        w = self.window()
        tutorial.first_run(w, True)
        self.assertFalse(w.tutorial.card.isVisible())
        with patch.object(tutorial, 'hidden_ranges_usable', return_value=True):w.tutorial.hint_tick()
        self.assertFalse(w.tutorial.card.isVisible())

    def test_no_welcome_when_asked_not_to(self):
        w = self.window()
        tutorial.first_run(w, False)
        self.assertFalse(w.tutorial.card.isVisible());self.assertTrue(w.tutorial.hint_timer.isActive())

    def test_smoke_runs_are_recognised_by_values_not_by_defaults(self):
        from argparse import Namespace
        plain = dict(smoke_screenshot=None, smoke_run_analysis=False, smoke_playback_seconds=0, smoke_report=None,
                     smoke_camera_distance=None, smoke_render_mode='box', smoke_bone_overlay=None)
        self.assertFalse(tutorial.smoke_run(Namespace(**plain)))
        for name, value in (('smoke_screenshot', Path('a.png')), ('smoke_run_analysis', True), ('smoke_playback_seconds', 2.0),
                            ('smoke_report', Path('r.json')), ('smoke_camera_distance', 0.0), ('smoke_bone_overlay', 'off')):
            with self.subTest(name=name):self.assertTrue(tutorial.smoke_run(Namespace(**{**plain, name: value})))


class HintTest(WindowCase):
    def hinted(self, usable=True):
        w = self.window()
        tutorial.first_run(w, False)
        with patch.object(tutorial, 'hidden_ranges_usable', return_value=usable):w.tutorial.hint_tick()
        return w

    def again(self, w):
        with patch.object(tutorial, 'hidden_ranges_usable', return_value=True):w.tutorial.hint_tick()
        return w.tutorial.card.isVisible()

    def test_hint_appears_when_hidden_ranges_become_usable(self):
        w = self.hinted()
        self.assertTrue(w.tutorial.card.isVisible());self.assertEqual(w.tutorial.card.text.text(), tutorial.HINT)
        self.assertEqual([b.text() for b in w.tutorial.card.findChildren(QPushButton)], ['×', "Don't show again", 'Not now', 'Show me'])

    def test_no_hint_while_hidden_ranges_are_not_usable(self):
        self.assertFalse(self.hinted(usable=False).tutorial.card.isVisible())

    def test_not_now_is_quiet_for_this_run_only(self):
        w = self.hinted();press(w, 'Not now')
        self.assertFalse(self.again(w));self.assertNotIn('hint_off', tutorial.load_state())

    def test_dont_show_again_is_saved(self):
        w = self.hinted();press(w, "Don't show again")
        self.assertIs(tutorial.load_state()['hint_off'], True);self.assertFalse(self.again(w))
        self.assertFalse(self.hinted().tutorial.card.isVisible())  # a later start of the app

    def test_show_me_opens_the_chapter(self):
        w = self.hinted();press(w, 'Show me')
        self.assertEqual(w.tutorial.chapter.key, 'hidden_ranges')

    def test_no_hint_during_a_chapter_or_over_another_message(self):
        w = self.window()
        tutorial.first_run(w, True)
        self.assertTrue(self.again(w));self.assertEqual(w.tutorial.card.title.text(), 'Welcome to RGB MoCap')
        press(w, 'Start')
        self.again(w)
        self.assertEqual(w.tutorial.chapter.key, 'basics')

    def test_a_chapter_opened_from_the_menu_over_the_hint_keeps_it_quiet(self):
        w = self.hinted();w.tutorial_actions['basics'].trigger();w.tutorial.stop()
        self.assertFalse(self.again(w))

    def test_no_hint_once_the_chapter_was_finished(self):
        tutorial.mark_done('hidden_ranges')
        self.assertFalse(self.hinted().tutorial.card.isVisible())

    def test_no_hint_under_a_dialog(self):
        w = self.window()
        tutorial.first_run(w, False)
        with patch.object(tutorial, 'modal_open', return_value=True):self.assertFalse(self.again(w))


if __name__ == '__main__':
    unittest.main()
