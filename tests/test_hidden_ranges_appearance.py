"""Hidden ranges widgets follow the UI scale and theme of the appearance settings."""
import unittest

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QLabel

from mocap.hidden_ranges_ui import RangeRows
from mocap.theme import ui_px
from mocap.timeline_editor import TimelineSlider

APP = QApplication.instance() or QApplication([])


class FillBandTest(unittest.TestCase):
    def tearDown(self):
        APP.setProperty('uiTheme', None);APP.setProperty('uiScale', None)

    def band(self, theme, scale, color):
        """The slider and the pixel rows that hold `color` in a column inside the filled frames."""
        APP.setProperty('uiTheme', theme);APP.setProperty('uiScale', scale)
        slider = TimelineSlider();slider.setRange(0, 99);slider.has_motion = True;slider.fill_runs = ((40, 60),);slider.resize(800, 400)
        image = slider.grab().toImage();x = round(slider.frame_x(47))
        return slider, [y for y in range(image.height()) if image.pixelColor(x, y) == QColor(color)]

    def test_band_lies_on_the_top_edge_of_the_motion_row_at_every_scale(self):
        for scale in (100, 200):
            with self.subTest(scale=scale):
                slider, rows = self.band('dark', scale, '#d8a24a')
                top = slider.TRACK_TOP + slider.TRACK_HEIGHT
                self.assertEqual(rows, list(range(top, top + ui_px(3))))

    def test_band_is_darker_on_the_light_theme(self):
        self.assertEqual(len(self.band('light', 100, '#8a4809')[1]), 3)
        self.assertEqual(self.band('light', 100, '#d8a24a')[1], [])


class RowsHeightTest(unittest.TestCase):
    def tearDown(self):
        APP.setProperty('uiScale', None)

    def test_rows_keep_a_minimum_height_that_grows_with_the_scale(self):
        for scale in (100, 200):
            with self.subTest(scale=scale):
                APP.setProperty('uiScale', scale)
                area = RangeRows()
                for _ in range(40):area.rows.addWidget(QLabel('Frames'))
                self.assertEqual(area.minimumSizeHint().height(), 72 * scale // 100)


if __name__ == '__main__':
    unittest.main()
