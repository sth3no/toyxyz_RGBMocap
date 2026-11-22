"""The in-window tutorial: saved state, highlight, card and the controller, on stand-in windows."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea, QWidget
import shiboken6

from mocap import paths, resource_safety, tutorial
from mocap.theme import ui_px
from mocap.tutorial_steps import Chapter, Step

APP = QApplication.instance() or QApplication([])


class HomeCase(unittest.TestCase):
    """Local settings in a temporary folder, never the installation's."""
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.object(paths, 'HOME', Path(tmp.name))
        home.start()
        self.addCleanup(home.stop)
        self.settings = Path(tmp.name) / 'config/local.json'


class StateTest(HomeCase):
    def test_missing_file_is_an_empty_state(self):
        self.assertEqual(tutorial.load_state(), {})

    def test_saved_values_are_merged_and_other_preferences_kept(self):
        resource_safety.update({'ui_theme': 'light'})
        self.assertTrue(tutorial.save_state(seen=True))
        self.assertTrue(tutorial.mark_done('basics'))
        tutorial.mark_done('basics')
        self.assertEqual(tutorial.load_state(), {'seen': True, 'done': ['basics']})
        self.assertEqual(resource_safety.read_settings(resource_safety.path())['ui_theme'], 'light')

    def test_unreadable_file_is_none_and_is_not_overwritten(self):
        self.settings.parent.mkdir(parents=True)
        self.settings.write_text('{broken', encoding='utf-8')
        self.assertIsNone(tutorial.load_state())
        self.assertFalse(tutorial.save_state(seen=True))
        self.assertEqual(self.settings.read_text(encoding='utf-8'), '{broken')

    def test_a_value_that_is_not_an_object_is_an_empty_state(self):
        resource_safety.update({'tutorial': 'yes'})
        self.assertEqual(tutorial.load_state(), {})


class AppearanceCase(unittest.TestCase):
    def tearDown(self):
        APP.setProperty('uiTheme', None);APP.setProperty('uiScale', None)

    def card(self, words=60):
        card = tutorial.TutorialCard()
        card.set_content('Chapter · Step 1 of 9', 'Title', 'word ' * words, 'Waiting…', [('Back', print), ('Next', print)])
        card.refresh_appearance()
        self.addCleanup(card.deleteLater)
        return card


class PlacementTest(AppearanceCase):
    def test_card_goes_to_the_side_with_the_most_room(self):
        p = tutorial.place_card(QSize(1100, 700), QSize(320, 200), QRect(900, 300, 150, 40), 12)
        self.assertEqual(p.x() + 320 + 12, 900)  # left of a target at the right edge
        self.assertEqual(p.y(), 300)

    def test_card_prefers_the_right_side_when_it_fits(self):
        self.assertEqual(tutorial.place_card(QSize(1100, 700), QSize(320, 200), QRect(100, 100, 150, 40), 12), QPoint(262, 100))

    def test_card_goes_below_a_wide_target_at_the_top(self):
        self.assertEqual(tutorial.place_card(QSize(1100, 700), QSize(320, 200), QRect(0, 30, 1100, 60), 12), QPoint(12, 102))

    def test_card_avoids_the_other_highlighted_controls(self):
        anchor, neighbour = QRect(100, 100, 200, 300), QRect(312, 100, 400, 300)
        self.assertEqual(tutorial.place_card(QSize(1440, 880), QSize(320, 160), anchor, 12, [anchor, neighbour]), QPoint(100, 412))

    def test_card_covers_as_little_as_it_can_when_every_side_is_taken(self):
        anchor = QRect(400, 300, 300, 200)
        taken = [anchor, QRect(712, 0, 728, 880), QRect(0, 0, 388, 880), QRect(388, 512, 324, 368), QRect(400, 100, 300, 188)]
        self.assertEqual(tutorial.place_card(QSize(1440, 880), QSize(320, 160), anchor, 12, taken).y(), 128)  # above: the smallest neighbour

    def test_card_without_a_target_is_centred(self):
        self.assertEqual(tutorial.place_card(QSize(1100, 700), QSize(320, 200), None, 12), QPoint(390, 250))

    def test_card_stays_inside_the_smallest_window_at_every_scale(self):
        for scale in (75, 100, 200):
            APP.setProperty('uiScale', scale)
            card = self.card()
            for target in (QRect(0, 0, 40, 20), QRect(1000, 650, 100, 50), QRect(0, 0, 1100, 700), QRect(500, 300, 60, 30)):
                with self.subTest(scale=scale, target=target):
                    p = tutorial.place_card(QSize(1100, 700), card.size(), target, ui_px(12))
                    self.assertTrue(QRect(0, 0, 1100, 700).contains(QRect(p, card.size())), (p, card.size()))


class WidgetsTest(AppearanceCase):
    def test_card_width_follows_the_scale(self):
        for scale in (100, 200):
            with self.subTest(scale=scale):
                APP.setProperty('uiScale', scale)
                self.assertEqual(self.card().width(), 320 * scale // 100)

    def test_card_is_tall_enough_for_a_long_text(self):
        self.assertGreater(self.card(60).height(), self.card(5).height())

    def test_card_style_follows_the_theme(self):
        APP.setProperty('uiTheme', 'light')
        card = self.card()
        self.assertIn('#f1f3f6', card.styleSheet());self.assertNotIn('#303030', card.styleSheet())

    def test_card_shows_what_it_was_given(self):
        card = self.card(3)
        self.assertEqual([b.text() for b in card.findChildren(QPushButton)], ['×', 'Back', 'Next'])
        self.assertEqual(card.findChildren(QPushButton)[-1].objectName(), 'primary')
        self.assertEqual((card.header.text(), card.title.text(), card.status.text()), ('Chapter · Step 1 of 9', 'Title', 'Waiting…'))
        self.assertEqual(card.text.textFormat(), Qt.PlainText)

    def test_card_buttons_call_back_and_the_cross_reports_closing(self):
        calls = []
        card = tutorial.TutorialCard();self.addCleanup(card.deleteLater)
        card.set_content('', 'T', 'x', '', [('Go', lambda: calls.append('go'))])
        card.closed.connect(lambda: calls.append('closed'))
        for button in card.findChildren(QPushButton)[::-1]:button.click()
        self.assertEqual(calls, ['go', 'closed'])

    def test_an_empty_status_takes_no_room(self):
        card = tutorial.TutorialCard();self.addCleanup(card.deleteLater)
        card.set_content('', 'T', 'x', '', [('Go', print)])
        self.assertTrue(card.status.isHidden())

    def test_card_is_tall_enough_for_its_text_in_the_font_of_the_window(self):
        host = QWidget();host.setStyleSheet('QWidget{font-size:22px}');host.resize(1100, 900);self.addCleanup(host.deleteLater)
        card = tutorial.TutorialCard(host)
        card.set_content('Chapter · Step 1 of 9', 'Title', 'word ' * 60, 'Waiting…', [('Back', print), ('Next', print)])
        host.show();card.show();APP.processEvents()
        for label in (card.title, card.text, card.status):
            self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()), label.objectName())
        self.assertGreater(card.height(), self.card().height())

    def test_a_shown_card_grows_for_a_longer_text(self):
        host = QWidget();host.resize(1100, 900);self.addCleanup(host.deleteLater)
        card = tutorial.TutorialCard(host)
        card.set_content('', 'Welcome', 'word ' * 12, '', [('Not now', print), ('Start', print)])
        host.show();card.show();APP.processEvents()
        card.set_content('Chapter · Step 1 of 9', 'Title', 'word ' * 60, '', [('Next', print)])
        APP.processEvents()
        self.assertGreaterEqual(card.text.height(), card.text.heightForWidth(card.text.width()))

    def test_the_cross_has_room_for_its_sign(self):
        card = self.card()
        self.assertEqual(card.cross.objectName(), 'tutorialClose');self.assertIn('QPushButton#tutorialClose{padding:0px', card.styleSheet())

    def test_card_buttons_never_take_the_keyboard(self):
        self.assertTrue(all(b.focusPolicy() == Qt.NoFocus for b in self.card().findChildren(QPushButton)))

    def overlay(self):
        overlay = tutorial.TutorialOverlay();overlay.resize(400, 300);overlay.set_holes([QRect(100, 100, 80, 40)])
        self.addCleanup(overlay.deleteLater)
        return overlay

    def test_overlay_lets_the_mouse_through(self):
        overlay = self.overlay()
        self.assertTrue(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
        self.assertEqual(overlay.focusPolicy(), Qt.NoFocus)

    def test_overlay_dims_everything_but_the_hole_and_rings_it(self):
        overlay = self.overlay();image = overlay.grab().toImage()
        self.assertNotEqual(image.pixelColor(140, 120), image.pixelColor(10, 10))
        self.assertEqual(image.pixelColor(10, 10), image.pixelColor(390, 290))
        self.assertIn(QColor('#74a6e0'), [image.pixelColor(x, 120) for x in range(90, 100)])
        self.assertEqual(overlay.holes, [QRect(100, 100, 80, 40)])

    def test_overlay_without_dim_draws_only_the_ring(self):
        with patch.object(tutorial, 'DIM', False):
            overlay = self.overlay();image = overlay.grab().toImage()
        self.assertEqual(image.pixelColor(140, 120), image.pixelColor(10, 10))
        self.assertIn(QColor('#74a6e0'), [image.pixelColor(x, 120) for x in range(90, 100)])

    def test_ring_is_darker_on_the_light_theme(self):
        APP.setProperty('uiTheme', 'light')
        image = self.overlay().grab().toImage()
        self.assertIn(QColor('#2866a3'), [image.pixelColor(x, 120) for x in range(90, 100)])


class Stand(QWidget):
    """A window with three controls and the state the test steps read."""
    def __init__(self):
        super().__init__()
        self.resize(1100, 700)
        self.a, self.b, self.c = (QPushButton(name, self) for name in 'abc')
        self.a.setGeometry(50, 50, 100, 30);self.b.setGeometry(900, 300, 150, 40);self.c.setGeometry(400, 600, 120, 30)
        self.flag = self.busy = self.hide_b = False
        self.failure = self.block = ''


def demo(**changes):
    steps = dict(a=Step('a', 'First', 'Text a', lambda w: [w.a]), b=Step('b', 'Second', 'Text b', lambda w: [w.b]),
                 c=Step('c', 'Third', lambda w: 'Text c for ' + w.c.text(), lambda w: [w.c]))
    for key, fields in changes.pop('steps', {}).items():steps[key] = Step(key, steps[key].title, steps[key].text, steps[key].targets, **fields)
    return Chapter('demo', 'Demo', tuple(steps.values()), **changes)


class ControllerTest(HomeCase):
    def setUp(self):
        super().setUp()
        self.w = Stand();self.w.show();APP.processEvents()
        self.t = tutorial.Tutorial(self.w)
        self.ended = [];self.t.finished.connect(lambda key, completed: self.ended.append((key, completed)))
        def close():
            self.t.stop();self.w.close();self.w.deleteLater();APP.processEvents()
        self.addCleanup(close)

    def buttons(self):return [b.text() for b in self.t.card.findChildren(QPushButton)]

    def press(self, label):next(b for b in self.t.card.findChildren(QPushButton) if b.text() == label).click()

    def test_next_moves_through_the_steps_and_finish_records_the_chapter(self):
        self.t.start(demo())
        self.assertTrue(self.t.active);self.assertTrue(self.t.timer.isActive())
        self.assertTrue(self.t.card.isVisible());self.assertTrue(self.t.overlay.isVisible())
        self.assertEqual((self.t.card.header.text(), self.t.card.title.text(), self.t.card.text.text()), ('Demo · Step 1 of 3', 'First', 'Text a'))
        self.assertEqual(self.buttons(), ['×', 'Next'])
        self.press('Next')
        self.assertEqual(self.buttons(), ['×', 'Back', 'Next'])
        self.press('Next')
        self.assertEqual((self.t.card.text.text(), self.buttons()), ('Text c for c', ['×', 'Back', 'Finish']))
        self.press('Finish')
        self.assertEqual(self.ended, [('demo', True)]);self.assertEqual(tutorial.load_state()['done'], ['demo'])
        self.assertFalse(self.t.active);self.assertFalse(self.t.timer.isActive())
        self.assertFalse(self.t.card.isVisible());self.assertFalse(self.t.overlay.isVisible())

    def test_a_met_condition_moves_on_at_the_next_tick(self):
        self.t.start(demo(steps=dict(a=dict(done=lambda w, e: w.flag, waiting='Waiting for the flag…'))))
        self.t.tick()
        self.assertEqual(self.t.index, 0);self.assertEqual(self.t.card.status.text(), 'Waiting for the flag…')
        self.assertEqual(self.buttons(), ['×', 'Skip step'])
        self.w.flag = True;self.t.tick()
        self.assertEqual(self.t.index, 1)

    def test_a_step_that_does_not_say_it_waits_offers_next(self):
        self.t.start(demo(steps=dict(a=dict(done=lambda w, e: w.flag))))
        self.assertEqual((self.buttons(), self.t.card.status.isHidden()), (['×', 'Next'], True))

    def test_a_skipped_step_is_never_shown_and_is_not_counted(self):
        self.t.start(demo(steps=dict(b=dict(skip=lambda w: w.hide_b))))
        self.w.hide_b = True;self.t.tick()
        self.assertEqual(self.t.card.header.text(), 'Demo · Step 1 of 2')
        self.t.next()
        self.assertEqual((self.t.step.key, self.t.card.header.text()), ('c', 'Demo · Step 2 of 2'))
        self.t.back()
        self.assertEqual(self.t.step.key, 'a')

    def test_a_first_step_that_is_skipped_is_not_the_start(self):
        self.t.start(demo(steps=dict(a=dict(skip=lambda w: True))))
        self.assertEqual((self.t.step.key, self.buttons()), ('b', ['×', 'Next']))

    def test_back_holds_a_step_whose_condition_is_already_met(self):
        self.w.flag = True
        self.t.start(demo(steps=dict(b=dict(done=lambda w, e: w.flag))))
        self.t.next();self.t.tick()
        self.assertEqual(self.t.step.key, 'c')
        self.t.back();self.t.tick();self.t.tick()
        self.assertEqual((self.t.step.key, self.buttons()), ('b', ['×', 'Back', 'Next']))
        self.t.next()
        self.assertEqual(self.t.step.key, 'c')

    def test_a_signal_counts_only_when_its_condition_holds(self):
        self.t.start(demo(events=lambda w: {'go': (w.a.clicked, lambda w: w.busy)}, steps=dict(a=dict(done=lambda w, e: 'go' in e))))
        self.w.a.click();self.t.tick()
        self.assertEqual(self.t.index, 0)
        self.w.busy = True;self.w.a.click();self.t.tick()
        self.assertEqual(self.t.index, 1)

    def test_an_earlier_signal_does_not_count_for_a_later_step(self):
        self.t.start(demo(events=lambda w: {'go': (w.a.clicked, lambda w: True)}, steps=dict(b=dict(done=lambda w, e: 'go' in e))))
        self.w.a.click();self.t.next();self.t.tick()
        self.assertEqual(self.t.step.key, 'b')

    def test_retry_sends_the_guide_one_step_back_with_its_note(self):
        self.t.start(demo(steps=dict(b=dict(retry=lambda w, e: w.failure or None))))
        self.t.next();self.t.tick()
        self.assertEqual(self.t.step.key, 'b')
        self.w.failure = 'The job stopped.';self.t.tick()
        self.assertEqual((self.t.step.key, self.t.card.status.text()), ('a', 'The job stopped.'))
        self.w.failure = '';self.t.next()
        self.assertEqual((self.t.step.key, self.t.card.status.text()), ('b', ''))

    def test_a_blocked_chapter_shows_its_reason_and_does_not_move(self):
        self.w.block = 'Choose another model.'
        self.t.start(demo(blocked=lambda w: w.block or None, steps=dict(a=dict(done=lambda w, e: True))))
        self.t.tick()
        self.assertEqual((self.t.index, self.t.card.text.text(), self.buttons()), (0, 'Choose another model.', ['×']))
        self.assertFalse(self.t.overlay.isVisible())
        self.w.block = '';self.t.tick()
        self.assertEqual(self.t.index, 1);self.assertTrue(self.t.overlay.isVisible())

    def test_overlay_hole_is_the_target_rectangle_in_window_coordinates(self):
        self.t.start(demo());self.t.next()
        self.assertEqual(self.t.overlay.holes, [QRect(900, 300, 150, 40)])
        self.assertEqual(self.t.overlay.geometry(), self.w.rect())
        self.assertEqual(self.t.card.x() + self.t.card.width() + ui_px(12), 900)

    def test_a_centred_step_keeps_the_card_in_the_middle(self):
        self.t.start(demo(steps=dict(a=dict(card='center'))))
        self.assertEqual(self.t.card.pos(), tutorial.place_card(self.w.size(), self.t.card.size(), None, ui_px(12)))
        self.assertEqual(self.t.overlay.holes, [QRect(50, 50, 100, 30)])

    def test_the_card_stands_beside_the_first_target_and_off_the_others(self):
        self.w.a.setGeometry(100, 100, 200, 300);self.w.b.setGeometry(312, 100, 400, 300)
        self.t.start(Chapter('two', 'Two', (Step('two', 'Two', 'x', lambda w: [w.a, w.b]),)))
        self.assertEqual(self.t.card.pos(), QPoint(100, 400 + ui_px(12)))
        self.assertFalse(any(self.t.card.geometry().intersects(hole) for hole in self.t.overlay.holes))

    def test_a_rectangle_target_is_used_as_it_is(self):
        self.t.start(Chapter('r', 'R', (Step('r', 'R', 'x', lambda w: [QRect(10, 20, 30, 40)]),)))
        self.assertEqual(self.t.overlay.holes, [QRect(10, 20, 30, 40)])

    def test_a_hidden_target_gets_no_ring_and_the_card_stays(self):
        self.w.b.hide();self.t.start(demo());self.t.next()
        self.assertEqual(self.t.overlay.holes, []);self.assertTrue(self.t.card.isVisible())

    def test_a_deleted_target_does_not_break_the_tick(self):
        self.t.start(demo());self.t.next()
        shiboken6.delete(self.w.b);self.t.tick()
        self.assertEqual(self.t.overlay.holes, []);self.assertEqual(self.t.step.key, 'b')

    def test_a_condition_on_a_deleted_control_does_not_break_the_tick(self):
        self.t.start(demo(steps=dict(a=dict(done=lambda w, e: w.b.isChecked(), skip=lambda w: w.b.isChecked()))))
        shiboken6.delete(self.w.b);self.t.tick()
        self.assertEqual(self.t.index, 0)

    def test_starting_again_restarts_without_doubling_the_signals(self):
        self.seen = []
        chapter = demo(events=lambda w: {'go': (w.a.clicked, lambda w: self.seen.append(1) or True)})
        self.t.start(chapter);self.t.next();self.t.start(chapter)
        self.w.a.click()
        self.assertEqual((self.seen, self.t.index), ([1], 0));self.assertEqual(self.ended, [('demo', False)])

    def test_signals_are_released_when_the_chapter_ends(self):
        self.seen = []
        self.t.start(demo(events=lambda w: {'go': (w.a.clicked, lambda w: self.seen.append(1) or True)}));self.t.stop()
        self.w.a.click()
        self.assertEqual(self.seen, [])

    def test_a_modal_dialog_hides_the_guide_until_it_closes(self):
        self.t.start(demo(steps=dict(a=dict(done=lambda w, e: w.flag))))
        self.w.flag = True
        with patch.object(tutorial, 'modal_open', return_value=True):
            self.t.tick()
            self.assertFalse(self.t.card.isVisible());self.assertFalse(self.t.overlay.isVisible());self.assertEqual(self.t.index, 0)
        self.t.tick()
        self.assertTrue(self.t.card.isVisible());self.assertTrue(self.t.overlay.isVisible());self.assertEqual(self.t.index, 1)

    def test_an_open_menu_hides_the_highlight_but_keeps_the_card(self):
        self.t.start(demo())
        with patch.object(tutorial, 'popup_open', return_value=True):
            self.t.tick()
            self.assertTrue(self.t.card.isVisible());self.assertFalse(self.t.overlay.isVisible())
        self.t.tick()
        self.assertTrue(self.t.overlay.isVisible())

    def test_stop_does_not_record_the_chapter(self):
        self.t.start(demo());self.t.next();self.press('×')
        self.assertEqual(self.ended, [('demo', False)]);self.assertEqual(tutorial.load_state(), {})
        self.assertFalse(self.t.card.isVisible())

    def test_an_unchanged_card_keeps_its_buttons_between_ticks(self):
        self.t.start(demo())
        before = self.t.card.findChildren(QPushButton)
        self.t.tick();self.t.tick()
        self.assertEqual([id(b) for b in self.t.card.findChildren(QPushButton)], [id(b) for b in before])

    def test_an_action_button_runs_the_action_of_the_step(self):
        self.t.start(demo(steps=dict(a=dict(action=lambda w: None if w.busy else ('Use it', lambda: setattr(w, 'flag', True))))))
        self.assertEqual(self.buttons(), ['×', 'Use it', 'Next'])
        self.press('Use it');self.assertTrue(self.w.flag)
        self.w.busy = True;self.t.tick()
        self.assertEqual(self.buttons(), ['×', 'Next'])

    def test_the_overlay_follows_the_window_size(self):
        self.t.start(demo());self.w.resize(1300, 800);APP.processEvents()
        self.assertEqual(self.t.overlay.geometry(), QRect(0, 0, 1300, 800))

    def test_card_stays_inside_the_window_for_every_step(self):
        self.t.start(demo())
        for _ in range(3):
            self.assertTrue(self.w.rect().contains(self.t.card.geometry()), self.t.step.key);self.t.next()

    def test_a_target_in_the_settings_scroll_area_is_scrolled_into_view(self):
        area = QScrollArea(self.w);area.setGeometry(600, 0, 200, 200)
        inner = QWidget();inner.setFixedSize(180, 2000);far = QPushButton('far', inner);far.move(0, 1800);area.setWidget(inner);area.show()
        self.w.settings_scroll = area;self.w.far = far;APP.processEvents()
        self.t.start(Chapter('s', 'S', (Step('far', 'Far', 'x', lambda w: [w.far]),)));APP.processEvents();self.t.tick()
        self.assertEqual(len(self.t.overlay.holes), 1)
        self.assertTrue(QRect(600, 0, 200, 200).contains(self.t.overlay.holes[0]))

    def test_scrolling_a_target_into_view_never_moves_the_panel_sideways(self):
        area = QScrollArea(self.w);area.setGeometry(600, 0, 200, 200)
        inner = QWidget();inner.setFixedSize(600, 2000);far = QPushButton('far', inner);far.move(450, 1800);area.setWidget(inner);area.show()
        self.w.settings_scroll = area;self.w.far = far;APP.processEvents()
        self.t.start(Chapter('s', 'S', (Step('far', 'Far', 'x', lambda w: [w.far]),)));APP.processEvents()
        self.assertEqual(area.horizontalScrollBar().value(), 0);self.assertGreater(area.verticalScrollBar().value(), 0)

    def test_a_message_is_a_centred_card_without_a_chapter(self):
        calls = []
        self.t.message('Welcome', 'Hello', [('Not now', lambda: calls.append('no')), ('Start', lambda: calls.append('yes'))], closed=lambda: calls.append('closed'))
        self.assertTrue(self.t.card.isVisible());self.assertFalse(self.t.overlay.isVisible());self.assertFalse(self.t.active)
        self.assertEqual((self.t.card.title.text(), self.buttons()), ('Welcome', ['×', 'Not now', 'Start']))
        self.assertEqual(self.t.card.pos(), tutorial.place_card(self.w.size(), self.t.card.size(), None, ui_px(12)))
        self.press('Start')
        self.assertEqual(calls, ['yes']);self.assertFalse(self.t.card.isVisible())
        self.t.message('Welcome', 'Hello', [('Start', print)], closed=lambda: calls.append('closed'));self.press('×')
        self.assertEqual(calls, ['yes', 'closed']);self.assertFalse(self.t.card.isVisible())


if __name__ == '__main__':
    unittest.main()
