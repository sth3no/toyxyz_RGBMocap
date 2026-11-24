"""Selecting another person keeps the time: the video and the 3D scene show the frame the timeline shows.
The real main window offscreen over the scene of test_fill_all_window. Needs PySide6 and ffmpeg."""
import unittest

import test_fill_all_window as fw
import test_parts_window as tp

APP=tp.APP
pump=tp.pump


@unittest.skipUnless(tp.FFMPEG.is_file(),'needs the bundled ffmpeg')
class SelectTimeTest(fw.Window):
    """Anna and Ben each have one take over the whole clip, so the timeline keeps its frame when the other is selected."""
    TAKES={1:((0,300),),2:((0,300),)}
    def shown(self):
        w=self.w
        return w.slider.value(),w.video_view.index,w.motion_view.index,round(w.motion_view.scene_time*30)
    def select(self,row):
        w=self.w;w.people_list.setCurrentRow(row);APP.processEvents()
    def test_the_other_person_is_shown_at_the_frame_of_the_timeline(self):
        w=self.w
        self.select(1);self.assertEqual(w.active_person_id,2);self.assertTrue(pump(lambda:w.video_readable))
        w.slider.setValue(110);self.assertTrue(pump(lambda:self.shown()==(110,110,110,110)),self.shown())
        self.select(0);self.assertEqual(w.active_person_id,1);self.assertTrue(pump(lambda:w.video_readable))
        self.assertTrue(pump(lambda:self.shown()==(110,110,110,110),5),self.shown())
    def test_a_take_of_the_same_person_is_shown_at_the_frame_of_the_timeline(self):
        w=self.w
        # A second take of Anna over the same frames, as reconstructing the range again with another trim leaves it.
        tp.RunnerTest.take(self,1,0,299)
        w.load_project(self.root/'p1/takes/000000000_000000300_scene_v1');self.assertTrue(pump(lambda:w.video_readable))
        self.assertGreater(w.person_takes.count(),1)
        w.slider.setValue(110);self.assertTrue(pump(lambda:self.shown()==(110,110,110,110)),self.shown())
        other=next(i for i in range(w.person_takes.count()) if i!=w.person_takes.currentIndex())
        w.person_takes.setCurrentIndex(other);APP.processEvents();self.assertTrue(pump(lambda:w.video_readable))
        self.assertTrue(pump(lambda:self.shown()==(110,110,110,110),5),self.shown())


if __name__=='__main__':
    unittest.main()
