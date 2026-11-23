"""Auto Setup: the best installed model with every refinement its files allow."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

from mocap import auto_setup, paths
from mocap.flow_hamer import MODEL_KEYS as HAMER

APP = QApplication.instance() or QApplication([])
FLOW, EXMR, GVHMR = (auto_setup.needed(key) for key in ('flowhmr', 'exmr', 'gvhmr'))
HTD, FOOTMR, HANDFLOW = (auto_setup.needed('gvhmr', **{name: True}) - GVHMR for name in ('htd', 'footmr', 'handflow'))
EVERYTHING = FLOW | EXMR | GVHMR | HTD | FOOTMR | HANDFLOW | set(HAMER)


def have(*sets):
    keys = set().union(*sets)
    return lambda key: key in keys


class NeededTest(unittest.TestCase):
    def test_every_model_needs_files_and_refinements_need_more(self):
        for keys in (FLOW, EXMR, GVHMR, HTD, FOOTMR, HANDFLOW):self.assertTrue(keys)
        self.assertIn('pva', HTD);self.assertIn('footmr', FOOTMR)

    def test_needed_matches_what_the_window_requires(self):
        from mocap.models import STAGE_REQUIREMENTS
        from mocap.motion_tools import MOTION_TOOLS
        stages = MOTION_TOOLS['gvhmr'].stages(True, True, True, handflow=True)
        self.assertEqual(auto_setup.needed('gvhmr', htd=True, footmr=True, handflow=True),
                         {key for stage in stages for key in STAGE_REQUIREMENTS.get(stage, ())})
        self.assertEqual(auto_setup.needed('flowhmr', hamer=True), FLOW | set(HAMER))


class PlanTest(unittest.TestCase):
    def test_flowhmr_is_first_with_hamer_fingers_and_the_sam_tracker(self):
        plan = auto_setup.plan(have(EVERYTHING), sam31=True)
        self.assertEqual((plan.model, plan.hamer, plan.tracker), ('flowhmr', True, 'sam31'))
        self.assertEqual((plan.htd, plan.footmr, plan.handflow), (False, False, False))  # GVHMR's refinements do not apply to it

    def test_exmr_is_next_when_flowhmr_lacks_a_file(self):
        missing = next(iter(FLOW - EXMR))
        plan = auto_setup.plan(have(EVERYTHING - {missing}), sam31=False)
        self.assertEqual((plan.model, plan.hamer, plan.tracker), ('exmr', False, 'default'))

    def test_gvhmr_gets_only_the_refinements_whose_files_are_ready(self):
        plan = auto_setup.plan(have(GVHMR, FOOTMR, HANDFLOW), sam31=False)
        self.assertEqual((plan.model, plan.htd, plan.footmr, plan.handflow), ('gvhmr', False, True, True))
        self.assertEqual(plan.skipped, ['HTD Refine', 'SAM 3.1 tracker'])

    def test_flowhmr_without_hamer_files_says_so(self):
        plan = auto_setup.plan(have(FLOW), sam31=False)
        self.assertEqual((plan.model, plan.hamer, plan.skipped), ('flowhmr', False, ['HaMeR Finger Refinement', 'SAM 3.1 tracker']))

    def test_nothing_ready_chooses_nothing(self):
        plan = auto_setup.plan(have(), sam31=False)
        self.assertIsNone(plan.model)
        self.assertEqual([name for name, count in plan.missing], ['FlowHMR', 'ExMR', 'GVHMR'])
        self.assertTrue(all(count > 0 for name, count in plan.missing))

    def test_the_note_says_what_was_chosen_why_and_what_was_left_out(self):
        text = auto_setup.describe(auto_setup.plan(have(EVERYTHING), sam31=True))
        self.assertIn('FlowHMR', text);self.assertIn('best overall motion quality in our tests', text);self.assertIn('slowest', text)
        self.assertIn('HaMeR Finger Refinement', text);self.assertIn('SAM 3.1 tracker', text);self.assertIn('Motion Correction', text)
        self.assertNotIn('Not used', text)
        text = auto_setup.describe(auto_setup.plan(have(GVHMR, FOOTMR), sam31=False))
        self.assertIn('Not used, not installed: HTD Refine, HandFlow, SAM 3.1 tracker.', text)
        self.assertIn('FlowHMR and ExMR rank higher in our tests, but files are missing.', text)
        self.assertIn('FlowHMR ranks higher in our tests, but files are missing.', auto_setup.describe(auto_setup.plan(have(EXMR), sam31=False)))
        text = auto_setup.describe(auto_setup.plan(have(), sam31=False))
        self.assertTrue(text.startswith('No model is ready.'));self.assertIn('Model Manager', text)

    def test_the_note_never_promises_more_than_the_guide(self):
        for keys in (EVERYTHING, EXMR, GVHMR):
            text = auto_setup.describe(auto_setup.plan(have(keys), sam31=False))
            self.assertIn('in our tests', text);self.assertNotIn('accurate', text)


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

    def test_the_button_sits_beside_the_model_guide(self):
        self.assertEqual(self.w.auto_button.text(), 'Auto Setup');self.assertTrue(self.w.auto_button.toolTip())
        self.assertTrue(self.w.auto_button.isVisible());self.assertTrue(self.w.auto_note.isHidden())
        self.assertEqual(self.w.auto_button.y(), self.w.model_help.y())

    def test_auto_setup_selects_the_model_its_refinements_and_the_tracker(self):
        auto_setup.run(self.w, have(EVERYTHING), True)
        self.assertEqual(self.w.model_choice.currentData(), 'flowhmr')
        self.assertTrue(self.w.flow_hamer.isChecked());self.assertTrue(self.w.motion_correction.isChecked())
        self.assertEqual(self.w.people_tracker.currentData(), 'sam31')
        self.assertFalse(self.w.auto_note.isHidden());self.assertIn('FlowHMR', self.w.auto_note.text())
        self.assertIn('Auto Setup', self.w.status.text())

    def test_auto_setup_for_gvhmr_sets_each_refinement(self):
        self.w.motion_correction.setChecked(False)
        auto_setup.run(self.w, have(GVHMR, FOOTMR), False)
        self.assertEqual(self.w.model_choice.currentData(), 'gvhmr')
        self.assertEqual((self.w.htd_refine.isChecked(), self.w.footmr_refine.isChecked(), self.w.handflow_refine.isChecked()), (False, True, False))
        self.assertTrue(self.w.motion_correction.isChecked());self.assertEqual(self.w.people_tracker.currentData(), 'default')

    def test_auto_setup_turns_hamer_off_when_its_files_are_missing(self):
        self.w.flow_hamer.setChecked(True)
        auto_setup.run(self.w, have(FLOW), False)
        self.assertEqual((self.w.model_choice.currentData(), self.w.flow_hamer.isChecked()), ('flowhmr', False))

    def test_nothing_ready_changes_nothing_and_says_so(self):
        before = self.w.model_choice.currentData()
        auto_setup.run(self.w, have(), False)
        self.assertEqual(self.w.model_choice.currentData(), before)
        self.assertTrue(self.w.auto_note.text().startswith('No model is ready.'))

    def test_a_running_job_is_left_alone(self):
        before = self.w.model_choice.currentData()
        self.w.busy = True;self.addCleanup(setattr, self.w, 'busy', False)
        auto_setup.run(self.w, have(GVHMR), False)
        self.assertEqual(self.w.model_choice.currentData(), before);self.assertTrue(self.w.auto_note.isHidden())
        self.assertIn('job', self.w.status.text())

    def test_the_button_uses_the_installed_files(self):
        with patch.object(auto_setup, 'run') as run:self.w.auto_button.click()
        run.assert_called_once_with(self.w)

    def test_the_tutorial_points_at_auto_setup(self):
        from mocap.tutorial_steps import BASICS
        step = BASICS.steps[0]
        self.assertIn(self.w.auto_button, step.targets(self.w));self.assertIn('Auto Setup', step.text)
        self.assertLessEqual(len(step.text), 330)


if __name__ == '__main__':
    unittest.main()
