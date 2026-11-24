"""Pick the trimmed range of a long video by typing its start and end times."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from mocap import clip_range, paths

APP = QApplication.instance() or QApplication([])


class ParseTest(unittest.TestCase):
    def test_seconds_minutes_and_hours(self):
        self.assertEqual(clip_range.parse('754.5'), 754.5)
        self.assertEqual(clip_range.parse('12:34.5'), 754.5)
        self.assertEqual(clip_range.parse('1:02:03.25'), 3723.25)
        self.assertEqual(clip_range.parse(' 50:00 '), 3000.)
        self.assertEqual(clip_range.parse('0'), 0.)

    def test_bad_input_says_why(self):
        for text in ('', 'abc', '1:2:3:4', '1::2', '12:75', '1:75:00', '1.5:00', ':30', '1e3', 'nan', 'inf'):
            with self.subTest(text=text), self.assertRaises(ValueError) as caught:
                clip_range.parse(text)
            self.assertTrue(str(caught.exception).endswith('.'))

    def test_negative_times_are_refused(self):
        for text in ('-5', '-1:00', '1:-5'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'negative'):
                clip_range.parse(text)


class FormatTest(unittest.TestCase):
    def test_minutes_below_an_hour_and_hours_above(self):
        self.assertEqual(clip_range.format_time(754.5), '12:34.5')
        self.assertEqual(clip_range.format_time(0), '0:00.0')
        self.assertEqual(clip_range.format_time(59.97), '1:00.0')
        self.assertEqual(clip_range.format_time(3723.25), '1:02:03.2')
        self.assertEqual(clip_range.format_time(3599.99), '1:00:00.0')

    def test_formatted_times_parse_back(self):
        for seconds in (0., 0.1, 61.3, 3599.9, 3600., 7384.5):
            self.assertAlmostEqual(clip_range.parse(clip_range.format_time(seconds)), seconds, places=6)


class FramesTest(unittest.TestCase):
    def test_converts_seconds_to_clip_frames(self):
        self.assertEqual(clip_range.frames(3000., 3030., 30., 108000, 4), (90000, 90900))
        self.assertEqual(clip_range.frames(0., 3600., 30., 108000, 4), (0, 108000))

    def test_the_end_must_come_after_the_start(self):
        for start, end in ((10., 10.), (20., 10.)):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, 'after'):
                clip_range.frames(start, end, 30., 900, 4)

    def test_beyond_the_clip_is_refused_with_its_length(self):
        with self.assertRaisesRegex(ValueError, '0:30.0'):
            clip_range.frames(10., 31., 30., 900, 4)
        with self.assertRaisesRegex(ValueError, '0:30.0'):
            clip_range.frames(30.5, 31., 30., 900, 4)

    def test_a_shown_end_rounded_up_by_a_twentieth_still_fits(self):
        self.assertEqual(clip_range.frames(0., 30., 30., 899, 4), (0, 899))

    def test_too_short_a_range_is_refused(self):
        with self.assertRaisesRegex(ValueError, '4 frames'):
            clip_range.frames(1., 1.05, 30., 900, 4)


class WindowTest(unittest.TestCase):
    def setUp(self):
        from mocap.theme import configure_application
        if not getattr(WindowTest, 'configured', False):
            configure_application(APP);WindowTest.configured = True
        tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        home = patch.object(paths, 'HOME', Path(tmp.name));home.start();self.addCleanup(home.stop)
        from mocap.gui import MainWindow
        self.w = MainWindow()
        def close():
            self.w.close();self.w.deleteLater();APP.processEvents()
        self.addCleanup(close)
        self.w.show();APP.processEvents()

    def long_clip(self, frames=108000, fps=30.):
        """The state video_ready leaves after opening an hour-long video, without decoding one."""
        w = self.w
        w.info = {'fps': fps, 'frames': frames, 'duration': frames / fps, 'width': 64, 'height': 48}
        w.slider.set_media(fps);w.slider.setRange(0, frames - 1)
        w.source_offset, w.source_end = 0., frames / fps
        w.slider.reset_trim();w.clip_changed(w.slider.clip_in, w.slider.clip_out)
        w.update_readiness();APP.processEvents()

    def type_range(self, start, end, enter=False):
        self.w.clip_from.setText(start);self.w.clip_to.setText(end)
        if enter:self.w.clip_to.returnPressed.emit()
        else:self.w.clip_set.click()
        APP.processEvents()

    def test_the_fields_show_the_whole_clip_after_loading(self):
        self.long_clip()
        self.assertEqual((self.w.clip_from.text(), self.w.clip_to.text()), ('0:00.0', '1:00:00.0'))

    def test_typing_a_range_sets_the_trim_exactly(self):
        self.long_clip()
        self.type_range('50:00', '50:30')
        self.assertEqual((self.w.slider.clip_in, self.w.slider.clip_out), (90000, 90900))
        self.assertIn('90001–90900', self.w.clip_label.text());self.assertIn('30.00 s', self.w.clip_label.text())
        self.assertEqual(self.w.processing_range(), (3000., 3030.))
        self.assertEqual((self.w.clip_from.text(), self.w.clip_to.text()), ('50:00.0', '50:30.0'))

    def test_enter_applies(self):
        self.long_clip()
        self.type_range('1:00', '2:00', enter=True)
        self.assertEqual((self.w.slider.clip_in, self.w.slider.clip_out), (1800, 3600))

    def test_the_view_shows_the_whole_chosen_range(self):
        self.long_clip()
        self.type_range('50:00', '50:30')
        slider = self.w.slider
        self.assertIsNotNone(slider.view_span);self.assertLess(slider.span(), 2000)
        left, right = slider.clip_edges()
        self.assertGreaterEqual(left, slider.LEFT);self.assertLessEqual(right, slider.width() - slider.RIGHT)
        self.assertEqual(slider.value(), 90000)  # the playhead goes to the start, as dragging the start handle does

    def test_an_invalid_entry_says_why_and_changes_nothing(self):
        self.long_clip()
        self.type_range('50:00', '50:30')
        for start, end, reason in (('abc', '50:30', 'abc'), ('50:30', '50:00', 'after'), ('50:00', '2:00:00', '1:00:00.0'),
                                   ('-1', '50:30', 'negative')):
            with self.subTest(start=start, end=end):
                self.type_range(start, end)
                self.assertEqual((self.w.slider.clip_in, self.w.slider.clip_out), (90000, 90900))
                self.assertIn(reason, self.w.status.text())

    def test_the_fields_follow_the_handles(self):
        self.long_clip()
        self.w.slider.set_clip(1800, 5400);APP.processEvents()
        self.assertEqual((self.w.clip_from.text(), self.w.clip_to.text()), ('1:00.0', '3:00.0'))

    def test_set_without_editing_keeps_the_exact_frames(self):
        self.long_clip()
        self.w.slider.set_clip(1801, 5399)
        self.w.clip_set.click();APP.processEvents()
        self.assertEqual((self.w.slider.clip_in, self.w.slider.clip_out), (1801, 5399))

    def test_disabled_exactly_when_trim_dragging_is(self):
        controls = (self.w.clip_from, self.w.clip_to, self.w.clip_set)
        self.assertFalse(any(control.isEnabled() for control in controls))
        self.long_clip()
        self.assertTrue(self.w.slider.trim_enabled)
        self.assertTrue(all(control.isEnabled() for control in controls))
        self.w.busy = True;self.w.update_readiness();APP.processEvents()
        self.assertFalse(self.w.slider.trim_enabled)
        self.assertFalse(any(control.isEnabled() for control in controls))
        self.w.busy = False;self.w.update_readiness()
        self.w.slider.trim_enabled = False;APP.processEvents()
        self.assertFalse(any(control.isEnabled() for control in controls))

    def test_the_timeline_row_fits_a_small_window(self):
        from PySide6.QtCore import QPoint
        self.long_clip()
        self.w.resize(1100, 700);APP.processEvents()
        row = self.w.clip_set.parentWidget()
        for control in (self.w.clip_from, self.w.clip_to, self.w.clip_set, self.w.reset_trim_button):
            self.assertLessEqual(control.mapTo(row, QPoint(control.width(), 0)).x(), row.width())
            self.assertGreaterEqual(control.width(), control.minimumSizeHint().width())
        self.assertGreaterEqual(self.w.clip_from.width(), self.w.clip_from.fontMetrics().horizontalAdvance('0:00:00.0'))


if __name__ == '__main__':
    unittest.main()
