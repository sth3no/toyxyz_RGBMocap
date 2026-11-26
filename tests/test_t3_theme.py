"""The Dark and Light looks (mocap/t3_theme.py) beside Classic Dark and Classic Light, offscreen."""
import os
import re
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QMenu, QPushButton

from mocap import t3_theme, theme

APP = QApplication.instance() or QApplication([])
HEX = r'#[0-9a-f]{6}\b'


def application():
    if not getattr(application, 'configured', False):
        theme.configure_application(APP);application.configured = True


def contrast(first, second):
    def luminance(color):
        parts = [(level / 255 / 12.92 if level / 255 <= .03928 else ((level / 255 + .055) / 1.055) ** 2.4)
                 for level in QColor(color).getRgb()[:3]]
        return .2126 * parts[0] + .7152 * parts[1] + .0722 * parts[2]
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + .05) / (low + .05)


class ColourTest(unittest.TestCase):
    """The colour rules alone, without an application."""

    def test_named_greys_become_the_looks_surfaces_and_lines(self):
        self.assertEqual(t3_theme.dark('#252525'), '#0a0a0a')  # the window
        self.assertEqual(t3_theme.dark('#303030'), '#111111')  # a panel
        self.assertEqual(t3_theme.dark('#191919'), '#1f1f1f')  # a line: lighter than the surface here, darker in Classic
        self.assertEqual(t3_theme.dark('#D5D5D5'), '#f5f5f5')  # text, in any letter case
        self.assertEqual(t3_theme.dark('#477bb7'), '#346bf1')  # the accent

    def test_other_greys_follow_the_curve_in_order(self):
        levels = [int(t3_theme.dark(f'#{level:02x}{level:02x}{level:02x}')[1:3], 16) for level in (0x26, 0x40, 0x60, 0x90, 0xc8, 0xfa)]
        self.assertEqual(levels, sorted(levels))
        self.assertTrue(all(re.fullmatch(HEX, t3_theme.dark(f'#{level:02x}{level:02x}{level:02x}')) for level in range(256)))

    def test_colours_that_are_not_grey_stay(self):
        for color in ('#d8a24a', '#91bb51', '#db6375', '#b1b9c1', '#ffffff'):
            self.assertEqual(t3_theme.dark(color), color)

    def test_light_side(self):
        self.assertEqual(t3_theme.color('#303030', '#f1f3f6', True), '#ffffff')  # a grey goes by the dark side, not by Classic Light
        self.assertEqual(t3_theme.color('#d5d5d5', '#222b35', True), '#27272a')
        self.assertEqual(t3_theme.color('#477bb7', '#2866a3', True), '#1b4ed8')  # the accent
        self.assertEqual(t3_theme.color('#d8a24a', '#8a4809', True), '#bb4d00')  # a caller's own light colour is kept or mapped
        self.assertEqual(t3_theme.color('#91bb51', '#3f6b12', True), '#3f6b12')
        self.assertEqual(t3_theme.color('#e6e6e6', None, True), '#f5f5f5')  # Classic Light keeps it, so this look keeps it light

    def test_every_colour_of_the_style_has_its_light_counterpart(self):
        missing = sorted(set(re.findall(HEX, t3_theme.STYLE)) - set(t3_theme.LIGHT) - {'#ffffff'})
        self.assertEqual(missing, [])
        self.assertEqual(sorted(set(t3_theme.DARK.values()) - set(t3_theme.LIGHT)), [])

    def test_the_style_covers_what_the_classic_style_names(self):
        names = set(re.findall(r'#[A-Za-z]+(?=[\s:,{])|Q[A-Za-z]+', theme.STYLE)) - set(re.findall(HEX, theme.STYLE))
        mine = set(re.findall(r'#[A-Za-z]+(?=[\s:,{])|Q[A-Za-z]+', t3_theme.STYLE))
        self.assertEqual(sorted(names - mine), [])

    def test_style_sheet_files_exist_and_light_has_no_dark_surface(self):
        for is_light in (False, True):
            style = t3_theme.stylesheet(is_light)
            paths = re.findall(r'url\("([^"]+)"\)', style)
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(Path(path).is_file() for path in paths), paths)
            self.assertNotIn('CHEVRON', style);self.assertNotIn('CHECK_ICON', style)
        self.assertNotIn('#0a0a0a', t3_theme.stylesheet(True))
        self.assertIn('chevron-down-light.svg', t3_theme.stylesheet(True))

    def test_text_reads_on_its_surfaces(self):
        for tone in (lambda value: value, t3_theme.light):
            for surface in ('#0a0a0a', '#111111', '#171717'):
                self.assertGreaterEqual(contrast(tone('#f5f5f5'), tone(surface)), 12)
                self.assertGreaterEqual(contrast(tone('#a3a3a3'), tone(surface)), 6)
                self.assertGreaterEqual(contrast(tone('#818181'), tone(surface)), 4.4)
            self.assertGreaterEqual(contrast('#ffffff', tone('#346bf1')), 4)


class AppearanceTest(unittest.TestCase):
    def setUp(self):
        application()
        old = theme.current_appearance()
        self.addCleanup(lambda:theme.apply_appearance(APP, *old))

    def test_the_four_looks(self):
        self.assertEqual(theme.THEME_IDS, ('dark', 'light', 'classic-dark', 'classic-light'))
        self.assertEqual(theme.DEFAULT_THEME, 'dark')
        seen = {}
        for ident in theme.THEME_IDS:
            theme.apply_appearance(APP, ident, 100)
            self.assertEqual(theme.current_appearance(), (ident, 100))
            self.assertEqual(theme.is_light_theme(), ident.endswith('light'))
            seen[ident] = (theme.theme_color('#303030'), theme.theme_color('#477bb7'), APP.palette().window().color().name())
        self.assertEqual(seen['dark'], ('#111111', '#346bf1', '#0a0a0a'))
        self.assertEqual(seen['light'], ('#ffffff', '#1b4ed8', '#fcfcfc'))
        self.assertEqual(seen['classic-dark'], ('#303030', '#477bb7', '#303030'))
        self.assertEqual(seen['classic-light'], ('#f1f3f6', '#2866a3', '#f1f3f6'))

    def test_classic_looks_show_what_they_showed(self):
        theme.apply_appearance(APP, 'classic-dark', 100)
        self.assertEqual(theme.themed_colors('a #303030 b #D5D5D5'), 'a #303030 b #D5D5D5')
        self.assertEqual(theme.theme_color('#d8a24a', '#8a4809'), '#d8a24a')
        self.assertIn('QPushButton#primary { background:#477bb7;', APP.styleSheet())
        self.assertEqual(APP.font().families()[0], 'Noto Sans KR')
        theme.apply_appearance(APP, 'classic-light', 100)
        self.assertEqual(theme.themed_colors('a #303030 b #D5D5D5'), 'a #f1f3f6 b #222b35')
        self.assertEqual(theme.theme_color('#d8a24a', '#8a4809'), '#8a4809')
        self.assertIn('QPushButton#primary { background:#2866a3;', APP.styleSheet())
        self.assertIn('QPushButton:hover { background:#d5e2ef; }', APP.styleSheet())

    def test_font_is_inter_with_the_bundled_fallback(self):
        theme.apply_appearance(APP, 'dark', 100)
        self.assertEqual(APP.font().families(), ['Inter', 'Noto Sans KR'])
        self.assertAlmostEqual(APP.font().pointSizeF(), 9)
        theme.apply_appearance(APP, 'light', 150)
        self.assertEqual(APP.font().families()[0], 'Inter')
        self.assertAlmostEqual(APP.font().pointSizeF(), 13.5)
        theme.apply_appearance(APP, 'classic-dark', 100)
        self.assertEqual(APP.font().families(), ['Noto Sans KR'])

    def test_widget_styles_written_in_classic_colours_follow_the_look(self):
        button = QPushButton('x');self.addCleanup(button.deleteLater)
        theme.set_appearance_style(button, 'QPushButton{background:#303030;border:1px solid #191919;color:#d5d5d5;padding:4px}')
        theme.apply_appearance(APP, 'dark', 100)
        self.assertEqual(button.styleSheet(), 'QPushButton{background:#111111;border:1px solid #1f1f1f;color:#f5f5f5;padding:4px}')
        theme.apply_appearance(APP, 'light', 150)
        self.assertEqual(button.styleSheet(), 'QPushButton{background:#ffffff;border:2px solid #e4e4e7;color:#27272a;padding:6px}')
        theme.apply_appearance(APP, 'classic-dark', 100)
        self.assertEqual(button.styleSheet(), 'QPushButton{background:#303030;border:1px solid #191919;color:#d5d5d5;padding:4px}')

    def test_the_style_scales(self):
        theme.apply_appearance(APP, 'dark', 150)
        self.assertIn('QPushButton { background:#171717; border:2px solid #262626; border-radius:9px; padding:8px 14px;', APP.styleSheet())
        theme.apply_appearance(APP, 'dark', 100)
        self.assertIn('QPushButton { background:#171717; border:1px solid #262626; border-radius:6px; padding:5px 9px;', APP.styleSheet())

    def test_a_menu_gets_a_see_through_background_for_its_round_corners(self):
        theme.apply_appearance(APP, 'dark', 100)
        menu = QMenu();self.addCleanup(menu.deleteLater)
        menu.addAction('One');menu.ensurePolished()
        self.assertTrue(menu.testAttribute(Qt.WA_TranslucentBackground))
        self.assertTrue(menu.windowFlags() & Qt.FramelessWindowHint)

    def test_saved_looks(self):
        import json
        import tempfile
        from unittest import mock
        from mocap import resource_safety
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'local.json'
            with mock.patch.object(resource_safety, 'path', lambda:path):
                for saved, shown in (('classic-light', 'classic-light'), ('light', 'light'), ('dark', 'dark'), ('sepia', 'dark'), (3, 'dark')):
                    path.write_text(json.dumps({'ui_theme': saved, 'ui_scale': 125}), encoding='utf-8')
                    self.assertEqual(theme.load_appearance(), (shown, 125))

    def test_settings_dialog_lists_the_four(self):
        dialog = theme.AppearanceDialog(None);self.addCleanup(dialog.deleteLater)
        self.assertEqual([dialog.theme.itemText(row) for row in range(dialog.theme.count())], ['Dark', 'Light', 'Classic Dark', 'Classic Light'])
        dialog.set_choices('classic-light', 125)
        self.assertEqual((dialog.theme.currentData(), dialog.scale.currentData()), ('classic-light', 125))


if __name__ == '__main__':
    unittest.main()
