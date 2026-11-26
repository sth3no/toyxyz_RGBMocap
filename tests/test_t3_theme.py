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

    def test_transport_buttons_show_pictures_and_classic_shows_their_text(self):
        buttons = {}
        for text in ('|<', '<', '>', '>|'):
            button = QPushButton(text);button.setObjectName('transport');self.addCleanup(button.deleteLater);buttons[text] = button
        other = QPushButton('<');self.addCleanup(other.deleteLater)   # not a transport button
        theme.apply_appearance(APP, 'dark', 100)
        self.assertEqual([(button.text(), button.icon().isNull()) for button in buttons.values()], [('', False)] * 4)
        self.assertEqual((other.text(), other.icon().isNull()), ('<', True))
        self.assertEqual(buttons['<'].iconSize().width(), 14)
        theme.apply_appearance(APP, 'light', 150)
        self.assertEqual(buttons['<'].iconSize().width(), 21)
        back, forward = (t3_theme.transport_picture(kind, 32, '#ffffff').toImage() for kind in ('previous', 'next'))
        self.assertNotEqual(back, forward)
        self.assertNotEqual(back, t3_theme.transport_picture('first', 32, '#ffffff').toImage())
        theme.apply_appearance(APP, 'classic-dark', 100)
        self.assertEqual([(button.text(), button.icon().isNull()) for button in buttons.values()],
                         [(text, True) for text in ('|<', '<', '>', '>|')])

    def test_a_transport_button_made_later_is_dressed_when_it_is_shown(self):
        theme.apply_appearance(APP, 'dark', 100)
        button = QPushButton('>|');button.setObjectName('transport');self.addCleanup(button.deleteLater)
        button.show();APP.processEvents()
        self.assertEqual((button.text(), button.icon().isNull()), ('', False))

    def test_the_title_bar_is_left_alone_where_windows_does_not_draw_it(self):
        window = QPushButton('x');self.addCleanup(window.deleteLater);window.show()
        self.assertFalse(t3_theme.title_bar(window, False, False))   # the offscreen platform has no Windows title bar

    def test_settings_dialog_lists_the_four(self):
        dialog = theme.AppearanceDialog(None);self.addCleanup(dialog.deleteLater)
        self.assertEqual([dialog.theme.itemText(row) for row in range(dialog.theme.count())], ['Dark', 'Light', 'Classic Dark', 'Classic Light'])
        dialog.set_choices('classic-light', 125)
        self.assertEqual((dialog.theme.currentData(), dialog.scale.currentData()), ('classic-light', 125))


class TimelineBarsTest(unittest.TestCase):
    """The timelines' bars: tinted tracks in the Classic looks, solid rounded bars in livelier colours in Dark and Light."""

    def setUp(self):
        application()
        old = theme.current_appearance()
        self.addCleanup(lambda:theme.apply_appearance(APP, *old))

    def bar(self, look, color, alpha=80, outline=False):
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QImage, QPainter
        theme.apply_appearance(APP, look, 100)
        image = QImage(60, 24, QImage.Format_ARGB32);image.fill(QColor('#242424'))
        p = QPainter(image);theme.paint_track_bar(p, QRectF(2, 2, 56, 20), color, alpha, outline);pen = p.pen().color().name();p.end()
        return image, pen

    def test_track_colours(self):
        self.assertEqual(t3_theme.track('#6289ad'), '#615fff')   # the video
        self.assertEqual(t3_theme.track('#6aa186'), '#00d492')   # the motion
        self.assertEqual(t3_theme.track('#b5b5b5'), '#00d492')   # a person whose colour is a grey moves in the motion's green
        self.assertIsNone(t3_theme.track('#303030'))             # a row with nothing in it
        dull, vivid = QColor('#6a8fdb'), QColor(t3_theme.track('#6a8fdb'))
        self.assertGreater(vivid.hsvSaturationF(), dull.hsvSaturationF())
        self.assertAlmostEqual(vivid.hsvHueF(), dull.hsvHueF(), places=2)
        self.assertEqual(len(set(t3_theme.TRACKS[color] for color in ('#6289ad', '#6aa186', '#56b6a7', '#c78f5f', '#79a9db', '#bf8dcf'))), 6)

    def test_classic_tints_the_track_as_before(self):
        image, pen = self.bar('classic-dark', '#6289ad')
        expected = [round(base + (top - base) * 80 / 255) for base, top in zip((0x24, 0x24, 0x24), (0x62, 0x89, 0xad))]
        for point in ((2, 2), (30, 12), (57, 21)):   # a square bar: its corner is filled
            self.assertTrue(all(abs(got - want) <= 1 for got, want in zip(image.pixelColor(*point).getRgb()[:3], expected)), point)
        self.assertEqual(image.pixelColor(0, 0).name(), '#242424')
        image, pen = self.bar('classic-dark', '#56b6a7', 90, outline=True)
        self.assertEqual(image.pixelColor(2, 12).name(), '#56b6a7')   # the line around a lock
        self.assertEqual(pen, '#56b6a7')

    def test_dark_and_light_draw_a_solid_rounded_bar(self):
        for look in ('dark', 'light'):
            image, pen = self.bar(look, '#6289ad')
            middle = image.pixelColor(30, 12)
            self.assertLess(abs(middle.hsvHueF() - QColor('#615fff').hsvHueF()), .02, look)
            self.assertGreater(middle.hsvSaturationF(), .55, look)
            self.assertEqual(image.pixelColor(2, 2).name(), '#242424', look)    # the corner is round
            self.assertEqual(image.pixelColor(57, 21).name(), '#242424', look)
            self.assertEqual(pen, '#ffffff', look)                             # white reads on the video's violet
        self.assertEqual(self.bar('dark', '#6aa186')[1], '#0a0a0a')            # near-black reads on the motion's green
        faint = self.bar('dark', '#6289ad', 25)[0].pixelColor(30, 12)
        self.assertLess(faint.valueF(), self.bar('dark', '#6289ad')[0].pixelColor(30, 12).valueF())
        self.assertEqual(self.bar('dark', '#303030')[0].pixelColor(30, 12).name(), '#111111')   # an empty row

    def test_text_and_grid(self):
        theme.apply_appearance(APP, 'classic-dark', 100)
        self.assertEqual(theme.track_text('#6289ad', '#b1b9c1').name(), '#b1b9c1')
        self.assertEqual(theme.grid_color(True).getRgb(), (115, 115, 115, 145))
        self.assertEqual(theme.grid_color(False).getRgb(), (133, 133, 133, 40))
        self.assertEqual(theme.track_tint('#56b6a7', 40).getRgb(), (0x56, 0xb6, 0xa7, 40))
        theme.apply_appearance(APP, 'classic-light', 100)
        self.assertEqual(theme.track_text('#6289ad', '#b1b9c1').name(), '#314c64')
        theme.apply_appearance(APP, 'dark', 100)
        self.assertEqual(theme.track_text('#6289ad', '#b1b9c1').name(), '#ffffff')
        self.assertEqual(theme.track_text('#6289ad', '#b1b9c1', 25).name(), '#f5f5f5')   # on a faint bar
        self.assertEqual(theme.track_text('#303030', '#aebeb4').name(), '#818181')       # on an empty row
        self.assertLess(theme.grid_color(True).alpha(), 60)
        self.assertEqual(theme.track_tint('#56b6a7', 40).name(), '#00d3f2')

    def test_the_frame_timeline_paints_its_rows_in_the_look(self):
        from mocap.timeline_editor import TimelineSlider

        def rows(look):
            theme.apply_appearance(APP, look, 100)
            slider = TimelineSlider();self.addCleanup(slider.deleteLater)
            slider.setRange(0, 99);slider.reset_trim();slider.has_motion = True;slider.resize(800, 400)   # 0-99 is a new slider's range already
            slider.set_contacts([[(10, 40)], [(50, 80)]], editable=True, identity='x', token=1)
            image, x = slider.grab().toImage(), round(slider.frame_x(15.5))
            middle = lambda row:slider.TRACK_TOP + row * slider.TRACK_HEIGHT + slider.TRACK_HEIGHT // 2
            return [image.pixelColor(x, middle(row)) for row in range(3)]

        video, motion, foot = rows('dark')
        self.assertGreater(min(color.hsvSaturationF() for color in (video, motion, foot)), .55)
        self.assertLess(abs(foot.hsvHueF() - QColor('#00d3f2').hsvHueF()), .02)
        video, motion, foot = rows('classic-dark')
        self.assertLess(max(color.hsvSaturationF() for color in (video, motion)), .45)   # tinted over the dark track

    def test_a_person_row_in_dark_shows_a_round_colour_dot(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem
        from mocap.people_ui import PeopleList

        def swatch(look):
            theme.apply_appearance(APP, look, 100)
            people = PeopleList();self.addCleanup(people.deleteLater);people.resize(170, 200)
            item = QListWidgetItem('Anna');item.setData(Qt.UserRole, 1);item.setData(Qt.UserRole + 1, '#d86a8f')
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable);item.setCheckState(Qt.Checked);people.addItem(item)
            rect, image = people.color_rect(people.model().index(0, 0)), people.viewport().grab().toImage()
            return image.pixelColor(rect.center()).name(), image.pixelColor(rect.left() + 4, rect.top() + 4).name()

        centre, corner = swatch('dark')
        self.assertEqual(centre, '#d86a8f')
        self.assertNotEqual(corner, '#d86a8f')     # a dot, not the square that fills the place
        centre, corner = swatch('classic-dark')
        self.assertEqual((centre, corner), ('#d86a8f', '#d86a8f'))


if __name__ == '__main__':
    unittest.main()
