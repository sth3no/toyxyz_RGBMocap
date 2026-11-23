"""Auto Setup: Fast, Detailed or Best possible from the installed files, with a recommendation for this PC."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QPushButton

from mocap import auto_setup, paths
from mocap.flow_hamer import MODEL_KEYS as HAMER

APP = QApplication.instance() or QApplication([])
FLOW, EXMR, GVHMR = (auto_setup.needed(key) for key in ('flowhmr', 'exmr', 'gvhmr'))
HTD, FOOTMR, HANDFLOW = (auto_setup.needed('gvhmr', **{name: True}) - GVHMR for name in ('htd', 'footmr', 'handflow'))
EVERYTHING = FLOW | EXMR | GVHMR | HTD | FOOTMR | HANDFLOW | set(HAMER)
SPEEDUPS = dict(fast_sam=True, batch=True, gpu_crop=True, stride=True)


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
    def test_modes_are_offered_in_the_order_of_their_cost(self):
        self.assertEqual(list(auto_setup.MODES.items()), [('fast', 'Fast'), ('detailed', 'Detailed'), ('best', 'Best possible')])

    def test_fast_takes_the_quickest_model_and_no_extra_pass(self):
        plan = auto_setup.plan(have(EVERYTHING), True, 'fast')
        self.assertEqual((plan.model, plan.tracker, plan.used), ('exmr', 'default', ['Motion Correction']))
        self.assertEqual(plan.spared, ['SAM 3.1 tracker'])

    def test_fast_on_gvhmr_leaves_its_refinements_off(self):
        plan = auto_setup.plan(have(GVHMR, HTD, FOOTMR, HANDFLOW), False, 'fast')
        self.assertEqual((plan.model, plan.htd, plan.footmr, plan.handflow), ('gvhmr', False, False, False))
        self.assertEqual(plan.spared, ['HTD Refine', 'FootMR', 'HandFlow']);self.assertEqual(plan.skipped, [])
        self.assertEqual(plan.better, ['ExMR'])

    def test_fast_falls_back_to_flowhmr_last_with_its_speed_ups(self):
        plan = auto_setup.plan(have(FLOW, HAMER), False, 'fast')
        self.assertEqual((plan.model, plan.hamer, plan.flow, plan.better), ('flowhmr', False, SPEEDUPS, ['ExMR', 'GVHMR']))

    def test_detailed_takes_the_best_model_with_its_standard_settings(self):
        plan = auto_setup.plan(have(EVERYTHING), True, 'detailed')
        self.assertEqual((plan.model, plan.hamer, plan.tracker, plan.flow), ('flowhmr', False, 'default', SPEEDUPS))
        self.assertEqual(plan.spared, ['HaMeR Finger Refinement', 'SAM 3.1 tracker']);self.assertEqual(plan.used, ['Motion Correction'])

    def test_detailed_on_gvhmr_uses_the_refinements_whose_files_are_ready(self):
        plan = auto_setup.plan(have(GVHMR, FOOTMR, HANDFLOW), False, 'detailed')
        self.assertEqual((plan.model, plan.htd, plan.footmr, plan.handflow), ('gvhmr', False, True, True))
        self.assertEqual(plan.skipped, ['HTD Refine']);self.assertEqual(plan.better, ['FlowHMR', 'ExMR'])

    def test_best_adds_every_installed_extra_pass(self):
        plan = auto_setup.plan(have(EVERYTHING), True, 'best')
        self.assertEqual((plan.model, plan.hamer, plan.tracker), ('flowhmr', True, 'sam31'))
        self.assertEqual(plan.used, ['HaMeR Finger Refinement', 'SAM 3.1 tracker', 'Motion Correction'])
        # HaMeR needs Fast SAM and 4-Frame Hand Sampling; the other two speed-ups are given up.
        self.assertEqual(plan.flow, dict(fast_sam=True, batch=False, gpu_crop=False, stride=True))

    def test_best_without_hamer_infers_hands_on_every_frame(self):
        plan = auto_setup.plan(have(FLOW), False, 'best')
        self.assertEqual((plan.model, plan.hamer, plan.flow), ('flowhmr', False, dict(fast_sam=False, batch=False, gpu_crop=False, stride=False)))
        self.assertEqual(plan.skipped, ['HaMeR Finger Refinement', 'SAM 3.1 tracker'])

    def test_exmr_is_next_when_flowhmr_lacks_a_file(self):
        missing = next(iter(FLOW - EXMR))
        plan = auto_setup.plan(have(EVERYTHING - {missing}), False, 'best')
        self.assertEqual((plan.model, plan.hamer, plan.flow, plan.better), ('exmr', False, None, ['FlowHMR']))

    def test_nothing_ready_chooses_nothing(self):
        plan = auto_setup.plan(have(), False, 'detailed')
        self.assertIsNone(plan.model)
        self.assertEqual([name for name, count in plan.missing], ['FlowHMR', 'ExMR', 'GVHMR'])
        self.assertTrue(all(count > 0 for name, count in plan.missing))

    def test_the_note_says_what_was_chosen_why_and_what_was_left_out(self):
        text = auto_setup.describe(auto_setup.plan(have(EVERYTHING), True, 'best'))
        self.assertTrue(text.startswith('Best possible · FlowHMR: the best overall motion quality in our tests, and the slowest of the three.'))
        self.assertIn('On: HaMeR Finger Refinement, SAM 3.1 tracker, Motion Correction.', text)
        self.assertIn('Off, so less is traded for speed: Adaptive Batching, GPU Hand Crops.', text)
        self.assertNotIn('Not used', text);self.assertNotIn('Left off', text)
        text = auto_setup.describe(auto_setup.plan(have(EVERYTHING), True, 'detailed'))
        self.assertIn('Left off for speed: HaMeR Finger Refinement, SAM 3.1 tracker. Best possible uses them.', text)
        text = auto_setup.describe(auto_setup.plan(have(GVHMR, FOOTMR), False, 'detailed'))
        self.assertIn('Not used, not installed: HTD Refine, HandFlow.', text)
        self.assertIn('FlowHMR and ExMR come first for this choice, but files are missing.', text)
        self.assertIn('ExMR comes first for this choice, but files are missing.', auto_setup.describe(auto_setup.plan(have(GVHMR), False, 'fast')))
        text = auto_setup.describe(auto_setup.plan(have(), False, 'fast'))
        self.assertTrue(text.startswith('No model is ready.'));self.assertIn('Model Manager', text)

    def test_the_note_never_promises_more_than_the_guide(self):
        for mode in auto_setup.MODES:
            for keys in (EVERYTHING, EXMR, GVHMR):
                text = auto_setup.describe(auto_setup.plan(have(keys), False, mode))
                self.assertIn('in our tests', text);self.assertNotIn('accurate', text)


class HardwareTest(unittest.TestCase):
    def test_the_largest_gpu_is_read_from_nvidia_smi(self):
        self.assertEqual(auto_setup.parse_gpu('NVIDIA GeForce RTX 5060 Ti, 16311\n'), ('NVIDIA GeForce RTX 5060 Ti', 16311 / 1024))
        self.assertEqual(auto_setup.parse_gpu('Small, 4096\r\nNVIDIA RTX 5090, 32607\r\n')[0], 'NVIDIA RTX 5090')
        for text in ('', 'NVIDIA-SMI has failed because it could not communicate', 'name, memory\n', 'A, B, not a number'):
            self.assertIsNone(auto_setup.parse_gpu(text), text)

    def test_recorded_peaks_come_from_the_model_guide(self):
        from mocap.model_guide import BENCHMARKS
        self.assertEqual((auto_setup.FLOW_VRAM, auto_setup.FLOW_RAM), (float(BENCHMARKS['flowhmr'][2]), float(BENCHMARKS['flowhmr'][3])))
        self.assertEqual((auto_setup.FLOW_VRAM, auto_setup.FLOW_RAM, auto_setup.HAMER_VRAM), (14.89, 22.19, 2.13))

    def test_recommendation_follows_the_recorded_peaks(self):
        self.assertEqual(auto_setup.recommend(31.8, 64)[0], 'best')
        self.assertEqual(auto_setup.recommend(17.02, 64)[0], 'best')
        self.assertEqual(auto_setup.recommend(15.93, 31.1)[0], 'detailed')  # a 16 GB card
        self.assertEqual(auto_setup.recommend(14.89, 22.19)[0], 'detailed')
        self.assertEqual(auto_setup.recommend(11.9, 64)[0], 'fast')
        self.assertEqual(auto_setup.recommend(31.8, 16)[0], 'fast')  # enough video memory, too little RAM
        self.assertEqual(auto_setup.recommend(15.93, None)[0], 'detailed')
        self.assertEqual(auto_setup.recommend(None, 64), (None, 'nvidia-smi reported no NVIDIA GPU, so there is no recommendation.'))

    def test_recommendation_says_why_and_that_the_other_choices_still_run(self):
        self.assertIn('FlowHMR peaked at 14.9 GiB of video memory and 22.2 GiB of RAM', auto_setup.recommend(11.9, 64)[1])
        self.assertIn('Low Memory', auto_setup.recommend(11.9, 64)[1])
        self.assertIn('HaMeR added 2.1 GiB', auto_setup.recommend(15.93, 31.1)[1])
        self.assertIn('our records', auto_setup.recommend(31.8, 64)[1]);self.assertIn('this PC has more', auto_setup.recommend(31.8, 64)[1])

    def test_the_line_names_this_pc_and_the_recommendation(self):
        line = auto_setup.hardware_line(('NVIDIA GeForce RTX 5060 Ti', 16311 / 1024), 31.1)
        self.assertTrue(line.startswith('This PC: NVIDIA GeForce RTX 5060 Ti, 15.9 GiB video memory, 31 GiB RAM.\nRecommended: Detailed.'), line)
        self.assertTrue(auto_setup.hardware_line(None, 31.1).startswith('This PC: no NVIDIA GPU found, 31 GiB RAM.\n'))


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

    def test_three_choices_stand_in_the_model_group(self):
        self.assertEqual({mode: button.text() for mode, button in self.w.auto_buttons.items()}, dict(fast='Fast', detailed='Detailed', best='Best'))
        self.assertTrue(all(button.isVisible() and button.toolTip() for button in self.w.auto_buttons.values()))
        self.assertEqual(self.w.auto_button.findChildren(QPushButton), list(self.w.auto_buttons.values()))
        self.assertTrue(self.w.auto_note.isHidden());self.assertTrue(self.w.auto_hardware.isHidden())
        self.assertLessEqual(self.w.auto_button.sizeHint().width(), self.w.model_choice.width())

    def test_the_choices_fit_the_panel_at_its_narrowest_with_the_star(self):
        from PySide6.QtCore import QPoint
        self.w.resize(1100, 700)
        auto_setup.show_hardware(self.w, ('NVIDIA GeForce RTX 5060 Ti', 16311 / 1024), 31.1);APP.processEvents()
        view = self.w.settings_scroll.viewport()
        for button in self.w.auto_buttons.values():
            self.assertLessEqual(button.mapTo(view, QPoint(button.width(), 0)).x(), view.width(), button.text())
            self.assertGreaterEqual(button.width(), button.sizeHint().width(), button.text())
        self.assertLessEqual(self.w.auto_hardware.mapTo(view, QPoint(self.w.auto_hardware.width(), 0)).x(), view.width())

    def test_building_the_window_does_not_ask_about_the_hardware(self):
        self.assertIsNone(getattr(self.w, 'auto_probe', None))

    def test_each_button_runs_its_mode(self):
        with patch.object(auto_setup, 'run') as run:
            for button in self.w.auto_buttons.values():button.click()
        self.assertEqual([call.args for call in run.call_args_list], [(self.w, 'fast'), (self.w, 'detailed'), (self.w, 'best')])

    def test_best_selects_the_model_its_passes_and_the_tracker(self):
        auto_setup.run(self.w, 'best', have(EVERYTHING), True)
        self.assertEqual(self.w.model_choice.currentData(), 'flowhmr')
        self.assertTrue(self.w.flow_hamer.isChecked());self.assertTrue(self.w.motion_correction.isChecked())
        self.assertEqual((self.w.fast_sam.isChecked(), self.w.flow_batch4.isChecked(), self.w.flow_gpu_crop.isChecked(),
                          self.w.flow_hand_stride4.isChecked()), (True, False, False, True))
        self.assertEqual(self.w.people_tracker.currentData(), 'sam31')
        self.assertFalse(self.w.auto_note.isHidden());self.assertIn('Best possible · FlowHMR', self.w.auto_note.text())
        self.assertIn('Auto Setup', self.w.status.text())

    def test_detailed_after_best_restores_the_standard_settings(self):
        auto_setup.run(self.w, 'best', have(FLOW), True)
        self.assertEqual((self.w.fast_sam.isChecked(), self.w.flow_hand_stride4.isChecked()), (False, False))
        auto_setup.run(self.w, 'detailed', have(EVERYTHING), True)
        self.assertEqual((self.w.model_choice.currentData(), self.w.flow_hamer.isChecked(), self.w.people_tracker.currentData()),
                         ('flowhmr', False, 'default'))
        self.assertTrue(all(box.isChecked() for box in (self.w.fast_sam, self.w.flow_batch4, self.w.flow_gpu_crop, self.w.flow_hand_stride4)))

    def test_fast_selects_exmr_and_the_default_tracker(self):
        self.w.people_tracker.setCurrentIndex(self.w.people_tracker.findData('sam31'))
        auto_setup.run(self.w, 'fast', have(EVERYTHING), True)
        self.assertEqual((self.w.model_choice.currentData(), self.w.people_tracker.currentData()), ('exmr', 'default'))

    def test_detailed_for_gvhmr_sets_each_refinement(self):
        self.w.motion_correction.setChecked(False)
        auto_setup.run(self.w, 'detailed', have(GVHMR, FOOTMR), False)
        self.assertEqual(self.w.model_choice.currentData(), 'gvhmr')
        self.assertEqual((self.w.htd_refine.isChecked(), self.w.footmr_refine.isChecked(), self.w.handflow_refine.isChecked()), (False, True, False))
        self.assertTrue(self.w.motion_correction.isChecked())

    def test_nothing_ready_changes_nothing_and_says_so(self):
        before = self.w.model_choice.currentData()
        auto_setup.run(self.w, 'best', have(), False)
        self.assertEqual(self.w.model_choice.currentData(), before)
        self.assertTrue(self.w.auto_note.text().startswith('No model is ready.'))

    def test_a_running_job_is_left_alone(self):
        before = self.w.model_choice.currentData()
        self.w.busy = True;self.addCleanup(setattr, self.w, 'busy', False)
        auto_setup.run(self.w, 'fast', have(GVHMR), False)
        self.assertEqual(self.w.model_choice.currentData(), before);self.assertTrue(self.w.auto_note.isHidden())
        self.assertIn('job', self.w.status.text())

    def test_the_hardware_line_is_shown_and_marks_the_recommended_choice(self):
        auto_setup.show_hardware(self.w, ('NVIDIA GeForce RTX 5060 Ti', 16311 / 1024), 31.1)
        self.assertFalse(self.w.auto_hardware.isHidden());self.assertIn('Recommended: Detailed.', self.w.auto_hardware.text())
        self.assertEqual({mode: button.text() for mode, button in self.w.auto_buttons.items()}, dict(fast='Fast', detailed='Detailed ★', best='Best'))
        auto_setup.show_hardware(self.w, None, 31.1)
        self.assertEqual([button.text() for button in self.w.auto_buttons.values()], ['Fast', 'Detailed', 'Best'])

    def test_the_tutorial_points_at_auto_setup(self):
        from mocap.tutorial_steps import BASICS
        step = BASICS.steps[0]
        self.assertIn(self.w.auto_button, step.targets(self.w))
        for word in ('Auto Setup', 'Fast', 'Detailed', 'Best'):self.assertIn(word, step.text)
        self.assertLessEqual(len(step.text), 330)


if __name__ == '__main__':
    unittest.main()
