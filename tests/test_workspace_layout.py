"""The window's arrangement (mocap/workspace_layout.py) and the menus between File and Help (mocap/menus.py) in the real
main window offscreen: People as a column of its own, the top bar with the file's line and the running job, the names of
the settings' sections, and menu entries that press the window's own controls. Nothing is reconstructed."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QGroupBox, QLabel, QMenuBar, QSplitter

import test_parts_window as tp
from mocap import menus, paths, theme, workspace_layout

APP, pump = tp.APP, tp.pump


def entry(menu, name):
    """The entry of `menu` called `name`, without the key written beside it."""
    return next(action for action in menu.actions() if action.text().split('\t')[0] == name)


class LayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tp.application()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.home = mock.patch.object(paths, 'HOME', Path(cls.tmp.name));cls.home.start()
        from mocap.gui import MainWindow
        cls.w = MainWindow();cls.w.resize(1440, 880);cls.w.show();pump(lambda:False, .3)

    @classmethod
    def tearDownClass(cls):
        cls.w.close();cls.w.deleteLater();APP.processEvents()
        cls.home.stop();cls.tmp.cleanup()

    def test_people_is_a_column_beside_the_video(self):
        w = self.w
        editors = w.workspace_split.widget(0)
        self.assertIsInstance(editors, QSplitter)
        self.assertIs(w.people_panel.parentWidget(), editors)
        self.assertEqual(editors.indexOf(w.people_panel), 0)
        self.assertEqual(editors.count(), 3)
        video = editors.widget(1)
        self.assertTrue(video.isAncestorOf(w.video_view))
        self.assertFalse(video.isAncestorOf(w.people_list))
        left = lambda widget:widget.mapTo(w, QPoint(0, 0)).x()
        self.assertLess(left(w.people_panel), left(video))
        self.assertLess(left(video), left(editors.widget(2)))
        least, _, most = workspace_layout.PEOPLE_WIDTH
        self.assertTrue(least <= w.people_panel.width() <= most, w.people_panel.width())
        self.assertEqual(w.people_panel.height(), video.height())
        self.assertFalse(workspace_layout.people_column(w))   # a second call moves nothing

    def test_the_file_line_is_in_the_menu_bar_and_the_row_under_it_is_gone(self):
        w = self.w
        self.assertIsInstance(w.top_bar_end.parentWidget(), QMenuBar)
        self.assertTrue(w.top_bar_end.isAncestorOf(w.filename))
        self.assertTrue(w.filename.isVisible())
        self.assertFalse(w.centralWidget().findChild(QLabel, 'brand').isVisible())
        self.assertFalse(any(label.isVisible() for label in w.centralWidget().findChildren(QLabel) if label.text().startswith('Motion Workspace')))
        bar_bottom = w.menuBar().mapTo(w, QPoint(0, w.menuBar().height())).y()
        self.assertLess(w.workspace_split.mapTo(w, QPoint(0, 0)).y() - bar_bottom, 16)   # the panels start right under the bar

    def test_the_end_of_the_bar_follows_the_length_of_the_file_line(self):
        w = self.w
        old = w.filename.text();self.addCleanup(lambda:w.filename.setText(old))
        w.filename.setText('one.mp4');pump(lambda:False, .2);short = w.top_bar_end.width()
        w.filename.setText('one.mp4  ·  Project loaded  ·  12345 frames');pump(lambda:False, .2)
        self.assertGreater(w.top_bar_end.width(), short + 100)
        self.assertLessEqual(w.top_bar_end.geometry().right(), w.menuBar().rect().right())
        w.filename.setText('x' * 400);pump(lambda:False, .2)
        self.assertLessEqual(w.filename.width(), theme.ui_px(560))    # a very long line does not push the menus out

    def test_the_running_job_shows_at_the_end_of_the_bar_only_while_it_runs(self):
        w, chip = self.w, self.w.job_chip
        def idle():
            w.busy = False;w.progress.setRange(0, 100);w.progress.setValue(0);w.task_info.setText('');chip.sync()
        self.addCleanup(idle)
        chip.sync();self.assertFalse(chip.isVisible())
        w.busy = True;w.progress.setRange(0, 100);w.progress.setValue(43);w.task_info.setText('Step left 0:42');w.status.setText('Part 2 of 5')
        chip.sync();pump(lambda:False, .2)
        self.assertTrue(chip.isVisible())
        self.assertEqual(chip.text.text(), '43 %  Step left 0:42')
        self.assertEqual((chip.bar.value(), chip.bar.maximum()), (43, 100))
        self.assertEqual(chip.toolTip(), 'Part 2 of 5')
        w.progress.setRange(0, 0);chip.sync()                      # a step without a known length
        self.assertEqual((chip.bar.minimum(), chip.bar.maximum()), (0, 0))
        self.assertEqual(chip.text.text(), 'Step left 0:42')
        idle();self.assertFalse(chip.isVisible())

    def test_the_settings_have_section_names_that_scroll_to_them(self):
        w, tabs = self.w, self.w.settings_tabs
        self.assertEqual([tabs.tabText(index) for index in range(tabs.count())], ['Model', 'Correction', 'Face', 'Resources'])
        face = w.track_face.parentWidget()
        self.assertIsInstance(face, QGroupBox)
        self.assertEqual(face.title(), 'Face')
        self.assertTrue(face.isAncestorOf(w.set_neutral_face) and face.isAncestorOf(w.reset_neutral_face) and face.isAncestorOf(w.neutral_face_note))
        column = face.parentWidget().layout()
        self.assertLess(column.indexOf(w.correction_group), column.indexOf(face))
        self.assertLess(column.indexOf(face), column.indexOf(w.compute_mode.parentWidget()))
        bar = tabs.scroll.verticalScrollBar();self.addCleanup(lambda:bar.setValue(0))
        self.assertGreater(bar.maximum(), 0, 'the settings are taller than the window in this test')
        bar.setValue(0);self.assertEqual(tabs.currentIndex(), 0)
        tabs.go(1);pump(lambda:False, .1)
        self.assertGreater(bar.value(), 0)
        self.assertEqual(tabs.currentIndex(), 1)
        self.assertLess(abs(tabs.top(w.correction_group) - bar.value()), 8)   # Correction is at the top of the view
        bar.setValue(bar.maximum());self.assertEqual(tabs.currentIndex(), 3)   # at the end the last section is the one in view
        w.correction_group.hide();self.addCleanup(w.correction_group.show);tabs.follow()
        self.assertFalse(tabs.isTabVisible(1))
        w.correction_group.show();tabs.follow();self.assertTrue(tabs.isTabVisible(1))


class MenusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tp.application()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.home = mock.patch.object(paths, 'HOME', Path(cls.tmp.name));cls.home.start()
        from mocap.gui import MainWindow
        cls.w = MainWindow();cls.w.resize(1440, 880);cls.w.show();pump(lambda:False, .3)

    @classmethod
    def tearDownClass(cls):
        cls.w.close();cls.w.deleteLater();APP.processEvents()
        cls.home.stop();cls.tmp.cleanup()

    def test_the_sections_stand_between_file_and_help(self):
        titles = [action.text() for action in self.w.menuBar().actions()]
        self.assertEqual(titles, ['File', 'View', 'Playback', 'Sequence', 'People', 'Motion', 'Help'])
        self.assertEqual(list(self.w.menus), titles[1:-1])
        self.assertEqual([action.menu() for action in self.w.menuBar().actions()[1:-1]], list(self.w.menus.values()))

    def test_the_menus_stay_usable_after_someone_asks_the_bar_for_them(self):
        import gc
        w = self.w
        asked = [action.menu() for action in w.menuBar().actions()]      # as mcp_desktop.py walks the menus
        asked += [action.menu() for menu in w.menus.values() for action in menu.actions() if action.menu() is not None]
        del asked;gc.collect()
        for title, menu in w.menus.items():
            self.assertEqual(menu.title(), title)                         # the Python objects are still the menus
            menu.aboutToShow.emit()
        direction = entry(w.menus['View'], 'View Direction').menu();direction.aboutToShow.emit()
        self.assertEqual(len(direction.actions()), w.view_choice.count())

    def test_a_box_entry_follows_and_presses_its_box(self):
        w, view = self.w, self.w.menus['View']
        box = w.viewport_checks['show_grid'];action = entry(view, 'Floor Grid')
        self.assertEqual(action.toolTip(), box.toolTip())
        was = box.isChecked();self.addCleanup(lambda:box.setChecked(was))
        view.aboutToShow.emit();self.assertEqual(action.isChecked(), was)
        action.trigger()
        self.assertEqual(box.isChecked(), not was)
        self.assertEqual(w.motion_view.show_grid, not was)        # the box did what a click on it does
        view.aboutToShow.emit();self.assertEqual(action.isChecked(), not was)
        self.assertEqual([entry(view, name).isCheckable() for name in ('Overlays', 'Bone Overlay', 'Face Overlay', 'Correction Floor', 'Motion Path',
                                                                      'Lock', 'Name', 'Camera', 'Follow Actor', 'Console')], [True] * 10)

    def test_a_button_entry_presses_its_button_and_is_grey_when_the_button_is(self):
        w, motion = self.w, self.w.menus['Motion']
        self.assertEqual([action.text() for action in motion.actions() if not action.isSeparator()],
                         ['Reconstruct 3D', 'Reconstruct in parts...', 'Prepare Preview', 'Cancel Job', 'Enable Correction', 'Track Face',
                          'Resource Safety', 'Model Manager', 'Model Guide'])
        pressed = [];w.models_button.clicked.connect(lambda:pressed.append('models'))
        with mock.patch.object(w, 'open_model_manager', lambda:None):
            entry(motion, 'Model Manager').trigger()
        self.assertEqual(pressed, ['models'])
        was = w.run_button.isEnabled();self.addCleanup(lambda:w.run_button.setEnabled(was))
        for enabled in (False, True):
            w.run_button.setEnabled(enabled);motion.aboutToShow.emit()
            self.assertEqual(entry(motion, 'Reconstruct 3D').isEnabled(), enabled)
        w.run_button.hide();self.addCleanup(w.run_button.show);motion.aboutToShow.emit()
        self.assertFalse(entry(motion, 'Reconstruct 3D').isEnabled())   # a button that is not in the window now

    def test_console_and_the_view_direction(self):
        w, view = self.w, self.w.menus['View']
        console = entry(view, 'Console');shown = w.log.isVisible();self.addCleanup(lambda:w.log.setVisible(shown))
        console.trigger();self.assertEqual(w.log.isVisible(), not shown)
        view.aboutToShow.emit();self.assertEqual(console.isChecked(), not shown)
        direction = entry(view, 'View Direction').menu();direction.aboutToShow.emit()
        names = [action.text() for action in direction.actions()]
        self.assertEqual(names, [w.view_choice.itemText(index) for index in range(w.view_choice.count())])
        self.assertEqual([action.text() for action in direction.actions() if action.isChecked()], [w.view_choice.currentText()])
        old = w.view_choice.currentText()
        w.view_choice.blockSignals(True)     # the offscreen 3D view cannot turn its camera
        def back():
            w.view_choice.setCurrentText(old);w.view_choice.blockSignals(False)
        self.addCleanup(back)
        next(action for action in direction.actions() if action.text() == 'Top').trigger()
        self.assertEqual(w.view_choice.currentText(), 'Top')

    def test_playback_entries_are_the_transport_buttons(self):
        w, playback = self.w, self.w.menus['Playback']
        self.assertEqual([action.text() for action in playback.actions() if not action.isSeparator()],
                         ['Play / Pause\tSpace', 'First Frame\tHome', 'Previous Frame\tLeft', 'Next Frame\tRight', 'Last Frame\tEnd', 'Speed',
                          'Frame All', 'Reset Trim', 'Timeline Help'])
        self.assertIs(menus.transport(w, 'Next Frame'), w.next_frame_button)
        self.assertIs(menus.transport(w, 'Previous Frame'), w.previous_frame)
        self.assertTrue(all(action.shortcut().isEmpty() for action in playback.actions()))   # the keys stay the window's own
        pressed = [];w.next_frame_button.clicked.connect(lambda:pressed.append('next'))
        was = w.next_frame_button.isEnabled();w.next_frame_button.setEnabled(True);self.addCleanup(lambda:w.next_frame_button.setEnabled(was))
        entry(playback, 'Next Frame').trigger();self.assertEqual(pressed, ['next'])
        speed = entry(playback, 'Speed').menu();speed.aboutToShow.emit()
        old = w.play_speed.currentIndex();self.addCleanup(lambda:w.play_speed.setCurrentIndex(old))
        next(action for action in speed.actions() if action.text() == '2x').trigger()
        self.assertEqual(w.play_speed.currentData(), 2.)

    def test_sequence_tools_and_people_entries(self):
        w, sequence, people = self.w, self.w.menus['Sequence'], self.w.menus['People']
        controller = w.sequence_ui
        self.assertEqual([action.text() for action in sequence.actions() if not action.isSeparator()],
                         [controller.add_button.text(), controller.cut_button.text(), 'Delete', 'Join', 'Frame All', 'Review Range...', 'Auto-advance',
                          'Selection tool\tV', 'Razor tool\tC', 'Hand tool\tH', 'Zoom tool\tZ'])
        was = controller.tools.isEnabled()
        def back():
            controller.tools.setEnabled(True);controller.tools.buttons['select'].click();controller.tools.setEnabled(was)
        self.addCleanup(back)
        controller.tools.setEnabled(False);sequence.aboutToShow.emit()      # as on a block's own timeline
        self.assertFalse(entry(sequence, 'Razor tool').isEnabled())
        controller.tools.setEnabled(True);sequence.aboutToShow.emit()
        self.assertTrue(entry(sequence, 'Razor tool').isEnabled())
        entry(sequence, 'Razor tool').trigger()
        self.assertEqual((controller.tools.tool, controller.view.tool), ('razor', 'razor'))
        sequence.aboutToShow.emit()
        self.assertEqual([action.text().split('\t')[0] for action in sequence.actions() if action.isChecked()], ['Razor tool'])
        self.assertEqual([action.text() for action in people.actions() if not action.isSeparator()],
                         ['Find person', 'Person Tracking Model', 'RIFE Frame Interpolation', 'Fill all hidden ranges', 'People Help'])
        tracker = entry(people, 'Person Tracking Model').menu();tracker.aboutToShow.emit()
        self.assertEqual([action.text() for action in tracker.actions()], ['Default', 'SAM 3.1'])

    def test_theme_and_scale_are_saved_and_applied_like_the_settings_dialog(self):
        w, view = self.w, self.w.menus['View']
        old = theme.current_appearance();saved = []
        self.addCleanup(lambda:theme.apply_appearance(APP, *old))
        with mock.patch('mocap.resource_safety.update', lambda values:saved.append(values)):
            looks = entry(view, 'Theme').menu();looks.aboutToShow.emit()
            self.assertEqual([action.text() for action in looks.actions()], ['Dark', 'Light', 'Classic Dark', 'Classic Light'])
            self.assertEqual([action.text() for action in looks.actions() if action.isChecked()], [dict((value, label) for label, value in theme.THEMES)[old[0]]])
            other = 'light' if old[0] != 'light' else 'dark'
            next(action for action in looks.actions() if action.text() == dict((value, label) for label, value in theme.THEMES)[other]).trigger()
            self.assertEqual(theme.current_appearance(), (other, old[1]))
            self.assertEqual(saved, [{'ui_theme': other, 'ui_scale': old[1]}])
            self.assertTrue(menus.set_appearance(w, theme=other));self.assertEqual(len(saved), 1)   # the same choice saves nothing
            scales = entry(view, 'UI Scale').menu();scales.aboutToShow.emit()
            self.assertEqual([action.text() for action in scales.actions() if action.isChecked()], [f'{old[1]}%'])
        with mock.patch('mocap.resource_safety.update', mock.Mock(side_effect=OSError('read-only'))), \
                mock.patch('mocap.menus.QMessageBox.warning') as warning:
            self.assertFalse(menus.set_appearance(w, theme=old[0]))
            self.assertEqual(theme.current_appearance()[0], other)      # what could not be saved is not applied
            self.assertIn('read-only', warning.call_args[0][2])


if __name__ == '__main__':
    unittest.main()
