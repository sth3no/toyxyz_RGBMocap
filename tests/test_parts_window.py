"""Reconstruct in parts on the real main window offscreen; Reconstruct 3D itself is replaced by a stand-in that records the
trim and the checked people and leaves the window busy, and the test plays the end of each batch. Needs PySide6 and ffmpeg."""
import json,os,shutil,subprocess,tempfile,time,types,unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from mocap import parts_runner,paths,people_store as ps

APP=QApplication.instance() or QApplication([])
FFMPEG=paths.ROOT/'tools/bin/ffmpeg.exe'
PLAN=[(0,90),(75,165),(150,240),(225,300)]   # 300 frames in parts of 3 s sharing 0.5 s


def application():
    if not getattr(application,'configured',False):
        from mocap.theme import configure_application
        configure_application(APP);application.configured=True


def pump(until,seconds=30):
    deadline=time.monotonic()+seconds
    while not until() and time.monotonic()<deadline:APP.processEvents();time.sleep(.01)
    return until()


@unittest.skipUnless(FFMPEG.is_file(),'needs the bundled ffmpeg')
class RunnerTest(unittest.TestCase):
    """A 10 s scene after Find person: Anna is seen all the time, Ben only in frames 0-99."""
    MODEL='exmr'
    def setUp(self):
        application();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=Path(self.tmp.name)
        home=mock.patch.object(paths,'HOME',base);home.start();self.addCleanup(home.stop)
        root=self.root=base/'scene';root.mkdir()
        subprocess.run([str(FFMPEG),'-v','error','-f','lavfi','-i','testsrc=size=64x48:rate=30','-frames:v','300','-pix_fmt','yuv420p','-c:v','libx264',
                        str(root/'input.mp4')],check=True,creationflags=0x08000000)
        self.manifest=dict(version=1,frames=300,fps=30,start=0.,end=10.,width=64,height=48,camera='static',source=dict(path=str(root/'input.mp4'),fps=30.),
                           pipeline=dict(model=self.MODEL),exmr_options=dict(static_camera=True,smoothing='none'),stages={})
        (root/'project.json').write_text(json.dumps(self.manifest),encoding='utf-8')
        people=[]
        for ident,name,seen in((1,'Anna',range(300)),(2,'Ben',range(100))):
            preview=root/f'p{ident}';preview.mkdir();observed=np.zeros(300,bool);observed[list(seen)]=True
            np.savez_compressed(preview/'tracking.npz',boxes=np.tile(np.array([10,10,30,40],np.float32),(300,1)),observed=observed,K=np.eye(3,dtype=np.float32))
            (preview/'project.json').write_text(json.dumps(dict(self.manifest,people_root='..',person_id=ident,person_preview=True)),encoding='utf-8')
            shutil.copy(root/'input.mp4',preview/'input.mp4')
            people.append(dict(id=ident,name=name,color='#b5b5b5' if ident==1 else '#3366aa',checked=True,first=0,last=len(seen),observations=len(seen),
                               preview=f'p{ident}',takes=[]))
        ps.save_people(root,dict(version=ps.VERSION,input=ps.media_receipt(root),people=people,observations='x/observations.sqlite',active=1,method='default',max_gap_seconds=.5))
        from mocap.gui import MainWindow
        w=self.w=MainWindow()
        def close():
            w.close();w.deleteLater();APP.processEvents()
        self.addCleanup(close);w.show();APP.processEvents()
        w.load_project(root/'p1');self.assertTrue(pump(lambda:w.video_readable),w.status.text())
        self.assertEqual((w.people_root,w.info['frames']),(root.resolve(),300))
        self.calls,self.ready=[],set()
        def run_full():   # the Reconstruct 3D button's path stops here: the queue would start the person jobs
            self.calls.append(dict(trim=(w.slider.clip_in,w.slider.clip_out),range=w.processing_range(),people=sorted(p['id'] for p in w.checked_people()),
                                   status=w.status.text(),footmr=w.footmr_refine.isChecked(),correction=w.motion_correction.isChecked()))
            w.busy,w.failed,w.cancelled=True,False,False;w.update_readiness()
        patch=mock.patch.object(w,'run_full',run_full);patch.start();self.addCleanup(patch.stop)
    def take(self,ident,a,b,ready=True):
        """What the person job and the batch leave behind: a take folder with ExMR's temporary frames, listed in people.json."""
        relative=f'p{ident}/takes/{a:09d}_{b:09d}_scene_v1';d=self.root/relative;(d/'exmr_frames').mkdir(parents=True,exist_ok=True)
        for i in range(b-a):(d/'exmr_frames'/f'{i:06d}.jpg').write_bytes(b'jpg')
        (d/'exmr_frames'/'receipt.json').write_text('{}',encoding='utf-8');(d/'hmr4d_results.pt').write_bytes(b'result')
        if not (d/'input.mp4').exists():os.link(self.root/'input.mp4',d/'input.mp4')
        (d/'project.json').write_text(json.dumps(dict(self.manifest,frames=b-a,start=a/30,end=b/30,people_root='../../..',person_id=ident,person_preview=False,
                                                    person_frames=[a,b],pipeline=dict(model=self.MODEL,result='initial'))),encoding='utf-8')
        state=ps.load_people(self.root);person=next(p for p in state['people'] if p['id']==ident)
        person['takes']=[t for t in person['takes'] if (t['first'],t['end'])!=(a,b)]+[dict(project=relative,first=a,end=b)];ps.save_people(self.root,state)
        if ready:self.ready.add(relative)
        return d
    def end(self,failed=False,cancelled=False,status=None):
        """The end of the batch: on success each checked person's takes of the trim, then the window shows the last take."""
        w=self.w;a,b=self.calls[-1]['trim'];seen={1:range(300),2:range(100)}
        if not failed and not cancelled:
            for ident in self.calls[-1]['people']:
                lo,hi=max(a,seen[ident][0]),min(b,seen[ident][-1]+1);self.take(ident,lo,hi)
        if status:w.status.setText(status)
        w.failed,w.cancelled=failed,cancelled;w.busy=False
        w.people_state=ps.load_people(self.root);w.activate_person(self.calls[-1]['people'][0]);w.update_readiness()
    def start(self,delete=True):
        w=self.w;w.footmr_refine.setChecked(False)
        runner=parts_runner.start(w,3.,.5,delete,ready=lambda project:project in self.ready)
        self.assertIsNotNone(runner,w.status.text());return runner
    def wait_calls(self,n):self.assertTrue(pump(lambda:len(self.calls)>=n or self.w.parts_runner is None),(len(self.calls),self.w.status.text()))
    def finish_all(self):
        while self.w.parts_runner is not None:   # each part's call has been made: end it and wait for the next one
            n=len(self.calls);self.end();self.wait_calls(n+1)

    def test_parts_run_in_order_with_the_people_visible_in_each(self):
        w=self.w;self.start();self.finish_all()
        self.assertEqual([c['trim'] for c in self.calls],PLAN)
        self.assertEqual([c['people'] for c in self.calls],[[1,2],[1,2],[1],[1]])   # Ben leaves the picture at frame 100
        self.assertEqual([c['status'] for c in self.calls],['Part 1 of 4: frames 1–90.','Part 2 of 4: frames 76–165.','Part 3 of 4: frames 151–240.','Part 4 of 4: frames 226–300.'])
        self.assertEqual([tuple(round(x,6) for x in c['range']) for c in self.calls],[(a/30,b/30) for a,b in PLAN])
        self.assertTrue(all(c['footmr'] is False for c in self.calls))   # the settings survive reloading the scene clip
        self.assertEqual(w.status.text(),'All 4 parts are reconstructed. Choose Fill hidden ranges in the People panel to join them.')
        self.assertIn('Part 3 of 4: Ben is not visible, so this part has no take of them.',w.log.toPlainText())
        self.assertEqual([p['checked'] for p in ps.load_people(self.root)['people']],[True,True])   # the user's checks are back
        self.assertEqual(json.loads((self.root/'parts_run.json').read_text(encoding='utf-8')),dict(version=1,first=0,end=300,part=90,overlap=15))
        anna,ben=ps.load_people(self.root)['people']
        self.assertEqual(sorted(k for k in anna['gap_fill']),['parts_150_165','parts_225_240','parts_75_90'])
        self.assertEqual(list(ben['gap_fill']),['parts_75_90'])
        w.activate_person(1);APP.processEvents()
        self.assertEqual([x.text() for x in w.hidden_ranges._widgets][:2],['Frames 76–90 (0.5 s)','Parts overlap: blended'])
        self.assertTrue(w.parts_button.isEnabled())
    def test_finished_parts_are_skipped(self):
        for ident,a,b in((1,0,90),(2,0,90),(1,75,165),(2,75,100),(1,150,240)):self.take(ident,a,b)
        self.take(1,225,300,ready=False)   # stopped before its result was ready
        self.w.people_state=ps.load_people(self.root)   # as the earlier run left the window
        self.start();self.finish_all()
        self.assertEqual([c['trim'] for c in self.calls],[(225,300)])
        self.assertIn('Part 1 of 4: nothing to reconstruct, skipped.',self.w.log.toPlainText())
    def test_a_failed_part_stops_the_run_and_keeps_the_finished_ones(self):
        self.start();self.wait_calls(1);self.end();self.wait_calls(2)
        self.end(failed=True,status='ExMR stopped: the GPU ran out of memory.')
        self.assertTrue(pump(lambda:self.w.parts_runner is None))
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.w.status.text(),'Part 2 of 4 stopped: ExMR stopped: the GPU ran out of memory. Finished parts are kept; Reconstruct in parts skips them next time.')
        self.assertTrue((self.root/'p1/takes/000000000_000000090_scene_v1/hmr4d_results.pt').is_file())
        self.calls.clear();self.start();self.wait_calls(1);self.assertEqual(self.calls[0]['trim'],(75,165))   # a later run goes on there
    def test_cancel_job_stops_the_run(self):
        self.start();self.wait_calls(1);self.assertTrue(self.w.cancel_button.isEnabled())
        self.w.cancel_button.click();self.end(cancelled=True)
        self.assertTrue(pump(lambda:self.w.parts_runner is None))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.w.status.text(),'Reconstruct in parts was cancelled at part 1 of 4. Finished parts are kept; Reconstruct in parts skips them next time.')
    def test_temporary_frames_of_a_finished_part_are_deleted(self):
        self.start(delete=True);self.wait_calls(1);self.end();self.wait_calls(2)
        for ident in(1,2):
            d=self.root/f'p{ident}/takes/000000000_000000090_scene_v1'
            self.assertEqual(list((d/'exmr_frames').iterdir()),[]);self.assertTrue((d/'hmr4d_results.pt').is_file() and (d/'input.mp4').is_file())
        self.assertIn('Part 1 of 4: deleted 182 temporary frame files.',self.w.log.toPlainText())   # 90 frames and a receipt per person
    def test_temporary_frames_are_kept_when_asked(self):
        self.start(delete=False);self.wait_calls(1);self.end();self.wait_calls(2)
        self.assertEqual(len(list((self.root/'p1/takes/000000000_000000090_scene_v1/exmr_frames').glob('*.jpg'))),90)
    def test_refusals(self):
        w=self.w;w.busy=True
        try:self.assertIsNone(parts_runner.start(w,3.,.5,True));self.assertEqual(w.status.text(),parts_runner.BUSY)
        finally:w.busy=False
        self.assertIsNone(parts_runner.start(w,3.,2.,True));self.assertEqual(w.status.text(),'The overlap must be shorter than half a part.')
        for item in(w.people_list.item(i) for i in range(w.people_list.count())):item.setCheckState(Qt.Unchecked)
        self.assertIsNone(parts_runner.start(w,3.,.5,True));self.assertEqual(w.status.text(),parts_runner.CHECK)
        self.assertEqual(self.calls,[])
    def test_a_window_without_find_person_is_refused(self):
        from mocap.gui import MainWindow
        other=MainWindow();self.addCleanup(lambda:(other.close(),other.deleteLater()))
        self.assertIsNone(parts_runner.start(other,60.,2.,True))
        self.assertEqual(other.status.text(),'Run Find person on the whole range first. All parts must share one scene and its people.')
    def test_models_without_people_takes_are_refused(self):
        w=types.SimpleNamespace(busy=False,contact_saving=False,parts_runner=None,people_state={'people':[]},people_root=self.root,checked_people=lambda:[1])
        for model in('mamma','prompthmr','gemx','pear'):
            w.model_choice=types.SimpleNamespace(currentData=lambda model=model:model)
            self.assertEqual(parts_runner.refusal(w),parts_runner.MODEL,model)
    def test_the_button_sits_under_reconstruct_3d_and_the_window_fits(self):
        from PySide6.QtCore import QPoint
        w=self.w;layout=w.model_actions.layout()
        buttons=[layout.itemAt(i).widget() for i in range(layout.count())]
        self.assertEqual(buttons.index(w.parts_button),buttons.index(w.run_button)+1);self.assertEqual(w.parts_button.text(),'Reconstruct in parts...')
        w.resize(1100,700);APP.processEvents()
        self.assertLessEqual(w.width(),1100);self.assertLessEqual(w.height(),700)
        bottom=w.cancel_button.mapTo(w,QPoint(0,w.cancel_button.height())).y()
        self.assertLessEqual(bottom,w.status.mapTo(w,QPoint(0,0)).y())   # the pinned actions stay above the status bar
        self.assertTrue(w.parts_button.isVisible())


@unittest.skipUnless(FFMPEG.is_file(),'needs the bundled ffmpeg')
class DialogTest(unittest.TestCase):
    def setUp(self):application()
    def dialog(self,**kw):
        from mocap.parts_ui import PartsDialog
        d=PartsDialog(None,0,54000,30,1920,1080,1,True,**kw);self.addCleanup(d.deleteLater);return d
    def test_the_line_follows_the_fields(self):
        from PySide6.QtWidgets import QDialogButtonBox
        d=self.dialog();ok=d.buttons.button(QDialogButtonBox.Ok)
        self.assertEqual((d.part.value(),d.overlap.value(),d.delete.isChecked()),(60.,2.,True))
        self.assertEqual(d.line.text(),'31 parts, about 0.34 GB of temporary frames at a time.');self.assertEqual(ok.text(),'Reconstruct')
        d.delete.setChecked(False);self.assertEqual(d.line.text(),'31 parts, about 10.5 GB of temporary frames in total.')
        d.part.setValue(120.);self.assertTrue(d.line.text().startswith('15 parts, '))
        d.overlap.setValue(60.);self.assertEqual(d.line.text(),'The overlap must be shorter than half a part.');self.assertFalse(ok.isEnabled())
        d.overlap.setValue(2.);self.assertTrue(ok.isEnabled());self.assertEqual(d.choice(),(120.,2.,False))
    def test_models_that_write_no_frames(self):
        from mocap.parts_ui import PartsDialog
        d=PartsDialog(None,0,9000,30,1920,1080,2,False);self.addCleanup(d.deleteLater)
        self.assertFalse(d.delete.isEnabled());self.assertEqual(d.line.text(),'5 parts.');self.assertIs(d.choice()[2],False)
    def test_a_saved_run_offers_its_sizes(self):
        d=self.dialog(part=3.,overlap=.5);self.assertEqual((d.part.value(),d.overlap.value()),(4.,.5))   # the field's minimum is 4 s


if __name__=='__main__':unittest.main()
