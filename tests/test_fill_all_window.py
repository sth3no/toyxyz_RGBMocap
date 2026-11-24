"""Hidden people and Fill all hidden ranges on the real main window offscreen. The scene on disk is built with the helpers
of test_parts_window (takes as the person job and the batch leave them); the job queue starts no worker: its next step
records the batch, and the test plays the batch's end through the real end of the queue. Needs PySide6 and ffmpeg."""
import json,os,subprocess,tempfile,unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import Qt

import test_parts_window as tp
from mocap import paths,people_store as ps

APP=tp.APP
pump=tp.pump


class Window(unittest.TestCase):
    """A 10 s GVHMR scene after Find person and Reconstruct 3D: Anna has takes 1-100 and 121-200 (hidden in 101-120),
    Ben has takes 1-90 and 151-300 (hidden in 91-150)."""
    MODEL='gvhmr'
    TAKES={1:((0,100),(120,200)),2:((0,90),(150,300))}
    def setUp(self):
        tp.application();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=self.base=Path(self.tmp.name)
        home=mock.patch.object(paths,'HOME',base);home.start();self.addCleanup(home.stop)
        video=tp.scene.video=base/'video.mp4'
        subprocess.run([str(tp.FFMPEG),'-v','error','-f','lavfi','-i','testsrc=size=64x48:rate=30','-frames:v','300','-pix_fmt','yuv420p','-c:v','libx264',
                        str(video)],check=True,creationflags=0x08000000)
        self.manifest=dict(version=1,frames=300,fps=30,start=0.,end=10.,width=64,height=48,camera='static',source=dict(path=str(video),fps=30.),
                           pipeline=dict(model=self.MODEL),stages={})
        root=self.root=base/'scene';tp.scene(root,self.manifest,((1,'Anna',range(300)),(2,'Ben',range(300))))
        for ident,takes in self.TAKES.items():
            for a,b in takes:tp.RunnerTest.take(self,ident,a,b)
        missing=mock.patch('mocap.models.missing',lambda *a,**k:[]);missing.start();self.addCleanup(missing.stop)
        # The fixture has no meshes to draw, so the scene loader returns an empty scene instead of preparing cameras.
        from mocap.scene_data import PeopleScene,SceneDataLoader
        load=mock.patch.object(SceneDataLoader,'load',lambda loader,root,kind,asset,cancelled:PeopleScene(root,dict(floor=0.)));load.start();self.addCleanup(load.stop)
        from mocap.gui import MainWindow
        w=self.w=MainWindow()
        def close():
            w.close();w.deleteLater();APP.processEvents()
        self.addCleanup(close);w.show();APP.processEvents()
        patch=mock.patch('mocap.gui.missing',lambda *a,**k:[]);patch.start();self.addCleanup(patch.stop)
        patch=mock.patch.object(w,'open_model_manager',lambda:None);patch.start();self.addCleanup(patch.stop)
        w.load_project(root/'p1');self.assertTrue(pump(lambda:w.video_readable),w.status.text())
        self.assertEqual(w.people_root,root.resolve())
        # The fixture's takes have plain results, so the window shows the result without correction or refinements.
        for box in (w.motion_correction,w.htd_refine,w.footmr_refine,w.handflow_refine):
            if box.isChecked():box.click()
        APP.processEvents();self.assertEqual(w.active_result,'initial')
        self.assertTrue(pump(lambda:w.scene_data is not None),w.motion_view.scene_message)
    def ranges(self,found):return {ident:[(r['first'],r['end']) for r in rows] for ident,rows in found.items()}
    def check(self,ident,on):
        for row in range(self.w.people_list.count()):
            item=self.w.people_list.item(row)
            if item.data(Qt.UserRole)==ident:item.setCheckState(Qt.Checked if on else Qt.Unchecked)
        APP.processEvents()


@unittest.skipUnless(tp.FFMPEG.is_file(),'needs the bundled ffmpeg')
class HiddenPeopleTest(Window):
    def test_the_view_gets_the_unfilled_hidden_ranges_of_the_scene(self):
        w=self.w
        self.assertEqual(self.ranges(w.motion_view.scene_hidden),{1:[(100,120)],2:[(90,150)]})
        self.assertIs(w.scene_data.hidden,w.motion_view.scene_hidden)
    def test_a_new_scene_after_a_change_gets_them_again(self):
        w=self.w;before=w.scene_data
        w.request_people_scene(force=True);self.assertTrue(pump(lambda:w.scene_data is not None and w.scene_data is not before))
        self.assertEqual(self.ranges(w.scene_data.hidden),{1:[(100,120)],2:[(90,150)]})
    def test_a_scene_of_a_model_without_hidden_ranges_has_none(self):
        w=self.w
        with mock.patch.object(w.model_choice,'currentData',lambda *a:'flowhmr'):w.sync_scene_visibility()
        self.assertEqual(w.motion_view.scene_hidden,{})


EXPERIMENTAL='Experimental support. Generated motion is plausible, not what the person did.'
SETUP='Automatic fill is not set up. Use Set up… and install the runtime in Model Manager.'
arg=lambda args,name:args[args.index(name)+1]


class FillAllWindow(Window):
    """The window with the job queue stopped before a worker starts, and the question answered by the test."""
    AUTO=None   # why Automatic fill cannot run on this machine; None: it can
    def setUp(self):
        super().setUp();w=self.w
        from mocap import fill_all_ui,kimodo_runtime
        self.ui=fill_all_ui
        for name in ('issue','text_issue'):
            patch=mock.patch.object(kimodo_runtime,name,lambda:self.AUTO);patch.start();self.addCleanup(patch.stop)
        self.batches=[];self.asked=[];self.real_next=w.next_job
        def next_job():   # the queue's next step: a worker would start for the first job here
            if w.jobs and not w.failed and not w.cancelled:
                jobs,w.jobs=w.jobs,[];self.batches.append(jobs);w.current_stage=jobs[0][0];return
            self.real_next()
        patch=mock.patch.object(w,'next_job',next_job);patch.start();self.addCleanup(patch.stop)
    def ask(self,answer=True):
        def asked(window,text):
            self.asked.append(text);return answer
        return asked
    def start(self,answer=True):return self.ui.start(self.w,ask=self.ask(answer))
    def finish(self,failed=False,cancelled=False,status=None):
        """The end of the batch that is running, through the real end of the queue."""
        w=self.w
        if status:w.status.setText(status)
        w.failed,w.cancelled=failed,cancelled;self.real_next()
    def joined(self,ident,name='joined'):
        """The worker's message that a joined take of this person was written."""
        self.w.people_event(dict(type='takes_joined',project=str(self.root),person_id=ident,projects=[str(self.root/f'p{ident}/takes/{name}')]))
    def wait_batches(self,n):self.assertTrue(pump(lambda:len(self.batches)>=n or self.w.fill_all_runner is None),(len(self.batches),self.w.status.text()))
    def people(self):return [(arg(b[0][1],'--person-id'),b[0][0]) for b in self.batches]
    def gap_fill(self):return {p['id']:p.get('gap_fill') for p in ps.load_people(self.root)['people']}


@unittest.skipUnless(tp.FFMPEG.is_file(),'needs the bundled ffmpeg')
class FillAllTest(FillAllWindow):
    def test_the_row_counts_the_unfilled_ranges_of_checked_people(self):
        w=self.w
        self.assertFalse(w.fill_all_note.isHidden());self.assertFalse(w.fill_all_button.isHidden())
        self.assertEqual((w.fill_all_note.text(),w.fill_all_button.text(),w.fill_all_button.isEnabled()),
                         ('2 hidden ranges of 2 people are not filled.','Fill all hidden ranges',True))
        self.assertTrue(w.motion_view.scene_fill_offered)
        self.check(2,False)
        self.assertEqual(w.fill_all_note.text(),'1 hidden range of 1 person is not filled.')
        self.check(1,False)
        self.assertTrue(w.fill_all_note.isHidden());self.assertTrue(w.fill_all_button.isHidden());self.assertFalse(w.motion_view.scene_fill_offered)
    def test_the_question_lists_the_ranges_and_no_is_no(self):
        self.assertIsNone(self.start(answer=False))
        self.assertEqual(self.asked,['Fill 2 hidden ranges of 2 people?\n\nAnna: frames 101–120 Smooth join\nBen: frames 91–150 Automatic (Experimental)\n\n'+EXPERIMENTAL])
        self.assertEqual((self.batches,self.gap_fill()),([],{1:None,2:None}))
        self.assertIsNone(self.w.fill_all_runner)
    def test_yes_records_the_methods_and_fills_one_person_after_another(self):
        w=self.w;runner=self.start()
        self.assertIs(w.fill_all_runner,runner)
        self.assertEqual(self.gap_fill(),{1:{'100_120':{'method':'smooth'}},2:{'90_150':{'method':'auto','seed':0}}})
        self.assertEqual(self.people(),[('1','take_join')]);self.assertEqual(arg(self.batches[0][0][1],'--result'),'initial')
        self.assertEqual(arg(self.batches[0][0][1],'--project'),str(w.people_root))
        self.assertFalse(w.fill_all_button.isEnabled());self.assertFalse(w.motion_view.scene_fill_offered)
        self.finish();self.wait_batches(2)
        self.assertEqual(self.people(),[('1','take_join'),('2','take_join')])
        self.finish();self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual(w.status.text(),'Fill all hidden ranges finished: 2 people filled.')
        self.assertIn('Fill all hidden ranges finished: 2 people filled.',w.log.toPlainText())
        self.assertTrue(w.fill_all_button.isEnabled())
    def test_a_person_whose_fill_fails_does_not_stop_the_next(self):
        w=self.w;self.start()
        self.finish(failed=True,status='Automatic fill could not connect the motion.');self.wait_batches(2)
        self.assertEqual(self.people(),[('1','take_join'),('2','take_join')])
        self.finish();self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual(w.status.text(),'Fill all hidden ranges finished: 1 person filled, 1 not. Anna: Automatic fill could not connect the motion.')
    def test_cancel_job_ends_the_run(self):
        w=self.w;self.start()
        self.finish(cancelled=True);self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual(len(self.batches),1)
        self.assertEqual(w.status.text(),'Fill all hidden ranges was cancelled. Nobody was filled.')
    def test_another_scene_ends_the_run(self):
        w=self.w;runner=self.start()
        with mock.patch.object(runner,'ours',lambda:False):
            self.finish();self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual(len(self.batches),1)
        self.assertEqual(w.status.text(),'The scene changed, so Fill all hidden ranges stopped.')
    def test_a_closed_window_starts_nothing_more(self):
        w=self.w;self.start()
        w.hide();self.finish();self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual(len(self.batches),1)
    def test_a_joined_take_of_another_person_does_not_move_the_view(self):
        w=self.w;shown=w.project;self.start()
        self.finish();self.wait_batches(2)
        self.joined(2)                      # Ben's joined take while Anna is the selected person
        self.assertEqual(self.batches[-1][0][0],'take_join')
        w.next_job();self.assertEqual([stage for stage,_ in self.batches[-1]],['export'])   # the joined take gets its preview
        self.finish();self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual((w.project,w.active_person_id),(shown,1))
        self.assertEqual(w.status.text(),'Fill all hidden ranges finished: 2 people filled.')
    def test_nothing_that_can_be_filled_is_said_without_a_question(self):
        w=self.w;self.AUTO=SETUP;self.check(1,False)
        self.assertIsNone(self.start())
        self.assertEqual(w.status.text(),'No hidden range can be filled. Ben, frames 91–150: '+SETUP)
        self.assertEqual((self.asked,self.batches,self.gap_fill()),([],[],{1:None,2:None}))
    def test_nothing_starts_while_a_job_runs(self):
        w=self.w;w.busy=True
        try:self.assertIsNone(self.start())
        finally:w.busy=False
        self.assertEqual((self.asked,self.batches),([],[]))
    def test_fill_hidden_ranges_of_the_selected_person_is_unchanged(self):
        w=self.w;w.gap_choice_changed('100_120','smooth');APP.processEvents()
        w.fill_hidden_ranges()
        self.assertEqual(self.people(),[('1','take_join')]);self.assertFalse(w._join_defer)
        self.joined(1,'000000000_000000200_fill_v1')
        self.assertEqual(w._people_return_project,self.root/'p1/takes/000000000_000000200_fill_v1')   # the view follows the selected person's joined take
        w._people_return_project=w.project;w.jobs=[];self.finish()


@unittest.skipUnless(tp.FFMPEG.is_file(),'needs the bundled ffmpeg')
class FillAllCorrectedTest(FillAllWindow):
    """The same scene with Enable Correction on: the joined takes are corrected once, with the whole scene."""
    def setUp(self):
        super().setUp();w=self.w
        w.motion_correction.click();APP.processEvents();self.assertEqual(w.active_result,'corrected_initial')
    def test_one_apply_correction_for_the_scene_follows_the_fills(self):
        w=self.w;self.start()
        self.assertIn('Apply Correction then runs for all people, because the shared floor of the scene changes.',self.asked[0])
        self.assertEqual(arg(self.batches[0][0][1],'--result'),'corrected_initial')
        self.joined(1,'a');self.assertEqual(w.jobs,[])   # no correction of its own
        w._people_return_project=w.project   # the fixture has no joined take on disk to show
        self.finish();self.wait_batches(2)
        self.joined(2,'b');self.assertEqual(w.jobs,[])
        self.finish();self.wait_batches(3)
        stages=[stage for stage,_ in self.batches[2]]
        self.assertEqual(stages,['correct','export']*4)   # every take of the scene that has its base motion
        self.assertEqual({arg(a,'--result') for _,a in self.batches[2]},{'corrected_initial'})
        self.assertIsNotNone(w.fill_all_runner)
        self.finish();self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual(w.status.text(),'Fill all hidden ranges finished: 2 people filled.')
    def test_no_correction_when_no_joined_take_was_written(self):
        w=self.w;self.start()
        self.finish(failed=True,status='Refused.');self.wait_batches(2)
        self.finish(failed=True,status='Refused too.');self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual(len(self.batches),2)
        self.assertEqual(w.status.text(),'Fill all hidden ranges finished: nobody filled, 2 not. Anna: Refused. Ben: Refused too.')
    def test_a_cancelled_correction_is_said(self):
        w=self.w;self.start()
        self.joined(2,'b');self.finish();self.wait_batches(2);self.finish();self.wait_batches(3)
        self.finish(cancelled=True);self.assertTrue(pump(lambda:w.fill_all_runner is None))
        self.assertEqual(w.status.text(),'Fill all hidden ranges finished: 2 people filled. Apply Correction was cancelled.')


if __name__=='__main__':
    unittest.main()
