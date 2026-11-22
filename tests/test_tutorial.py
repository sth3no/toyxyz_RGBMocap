"""The in-window tutorial: saved state, highlight, card and the controller, on stand-in windows."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QPushButton

from mocap import paths, resource_safety, tutorial
from mocap.theme import ui_px

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


if __name__ == '__main__':
    unittest.main()
