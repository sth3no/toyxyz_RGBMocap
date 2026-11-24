"""Reconstruct in parts on the real main window offscreen. The Reconstruct 3D path is real (run_full, input_person_jobs,
start_jobs) up to the job queue: the queue's next_job starts no worker but records the batch, and the test plays the
batch's end through the real end of next_job. Takes are written on disk as the person job and the batch leave them, with
the files the app's own readiness checks read (selected_export, result_is_current, the shared scene floor receipt).
Needs PySide6 and ffmpeg."""
import json,os,shutil,stat,subprocess,tempfile,time,types,unittest
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
SEEN={1:range(300),2:range(100)}   # Anna is seen all the time, Ben only in frames 0-99
IDENTITY=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]


def application():
    if not getattr(application,'configured',False):
        from mocap.theme import configure_application
        configure_application(APP);application.configured=True


def pump(until,seconds=30):
    deadline=time.monotonic()+seconds
    while not until() and time.monotonic()<deadline:APP.processEvents();time.sleep(.01)
    return until()


def stamp(path):
    s=Path(path).stat();return [s.st_size,s.st_mtime_ns]


def write_result(d,kind,payload):
    """A finished result of `kind` in take `d`: the motion file, its sources receipt and the bone preview with its BVH."""
    from mocap.mixamo_data import VERSION
    from mocap.motion_tools import RESULTS
    result=d/RESULTS[kind][0];result.write_bytes(payload)
    (d/f'{kind}_sources.json').write_text(json.dumps({result.name:stamp(result)}),encoding='utf-8')
    bvh=d/f'{kind}_motion.bvh';bvh.write_bytes(b'bvh '+payload)
    (d/f'{kind}_mixamo.json').write_text(json.dumps(dict(version=VERSION,kind=kind,fps=30.,code_stats={'mixamo_data.py':stamp(paths.ROOT/'mocap/mixamo_data.py')},
                                                         sources={},outputs={bvh.name:stamp(bvh)},bvh_file=bvh.name)),encoding='utf-8')


def correct(root,takes):
    """The correct stage of a batch: one shared floor resolved over every take now in the scene (the real scene_floor
    inputs and signature), then each new take's corrected result bound to it."""
    from mocap import scene_floor
    from mocap.correction_settings import effective_options
    options=scene_floor.ensure_config(root);context=scene_floor.inputs(root)
    record=dict(version=scene_floor.VERSION,options=options,inputs=context,floor=0.,samples=0,warning='',signature=scene_floor.signature(options,context,0.),method='test')
    (root/'scene_floor_resolved.json').write_text(json.dumps(record),encoding='utf-8')
    for d in takes:
        state=json.loads((d/'project.json').read_text(encoding='utf-8'))
        (d/'corrected_initial_correction.json').write_text(json.dumps(dict(version=2,options=effective_options(d,state),fps=30,
                                                                             shared_floor=dict(signature=record['signature'],floor=0.))),encoding='utf-8')
        write_result(d,'corrected_initial',b'corrected')


def scene(root,manifest,people):
    """A scene folder after Find person: the video, the person previews with their tracking, people.json."""
    root.mkdir();shutil.copy(scene.video,root/'input.mp4')
    (root/'project.json').write_text(json.dumps(manifest),encoding='utf-8')
    rows=[]
    for ident,name,seen in people:
        preview=root/f'p{ident}';preview.mkdir();observed=np.zeros(manifest['frames'],bool);observed[list(seen)]=True
        np.savez_compressed(preview/'tracking.npz',boxes=np.tile(np.array([10,10,30,40],np.float32),(manifest['frames'],1)),observed=observed,K=np.eye(3,dtype=np.float32))
        (preview/'project.json').write_text(json.dumps(dict(manifest,people_root='..',person_id=ident,person_preview=True)),encoding='utf-8')
        shutil.copy(root/'input.mp4',preview/'input.mp4')
        rows.append(dict(id=ident,name=name,color='#b5b5b5' if ident==1 else '#3366aa',checked=True,first=0,last=len(seen),observations=len(seen),
                         preview=f'p{ident}',takes=[]))
    ps.save_people(root,dict(version=ps.VERSION,input=ps.media_receipt(root),people=rows,observations='x/observations.sqlite',active=1,method='default',max_gap_seconds=.5))


@unittest.skipUnless(FFMPEG.is_file(),'needs the bundled ffmpeg')
class RunnerTest(unittest.TestCase):
    """A 10 s ExMR scene with a static camera after Find person."""
    MODEL='exmr'
    END=10.
    def setUp(self):
        application();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=self.base=Path(self.tmp.name)
        home=mock.patch.object(paths,'HOME',base);home.start();self.addCleanup(home.stop)
        video=scene.video=base/'video.mp4'
        subprocess.run([str(FFMPEG),'-v','error','-f','lavfi','-i','testsrc=size=64x48:rate=30','-frames:v','300','-pix_fmt','yuv420p','-c:v','libx264',
                        str(video)],check=True,creationflags=0x08000000)
        self.manifest=dict(version=1,frames=300,fps=30,start=0.,end=self.END,width=64,height=48,camera='static',source=dict(path=str(video),fps=30.),
                           pipeline=dict(model=self.MODEL),exmr_options=dict(static_camera=True,smoothing='none'),stages={})
        root=self.root=base/'scene';scene(root,self.manifest,((1,'Anna',SEEN[1]),(2,'Ben',SEEN[2])))
        missing=mock.patch('mocap.models.missing',lambda *a,**k:[]);missing.start();self.addCleanup(missing.stop)   # no model files in tests
        patch=mock.patch('mocap.gui.missing',lambda *a,**k:[]);patch.start();self.addCleanup(patch.stop)
        self.dialogs=[]
        self.open_window()

    def open_window(self):
        from mocap.gui import MainWindow
        root = self.root
        w=self.w=MainWindow()
        def close():
            w.close();w.deleteLater();APP.processEvents()
        self.addCleanup(close);w.show();APP.processEvents()
        patch=mock.patch.object(w,'open_model_manager',lambda:self.dialogs.append('Model Manager'));patch.start();self.addCleanup(patch.stop)   # never a model dialog
        w.people_tracker.setCurrentIndex(w.people_tracker.findData('default'))
        w.load_project(root/'p1');self.assertTrue(pump(lambda:w.video_readable),w.status.text())
        self.assertEqual((w.people_root,w.info['frames']),(root.resolve(),300))
        self.calls,self.batches=[],[]
        self.real_next=w.next_job
        def next_job():   # the queue's next step: a worker would start for the first job here
            if w.jobs and not w.failed and not w.cancelled:
                jobs,w.jobs=w.jobs,[];self.batches.append(jobs);w.current_stage=jobs[0][0]
                if jobs[0][0]=='person':
                    arg=lambda args,name:args[args.index(name)+1]
                    self.calls.append(dict(trim=(w.slider.clip_in,w.slider.clip_out),range=w.processing_range(),people=sorted(int(arg(a,'--person-id')) for _,a in jobs),
                                           frames=(int(arg(jobs[0][1],'--first-frame')),int(arg(jobs[0][1],'--end-frame'))),result=arg(jobs[0][1],'--result'),
                                           status=w.status.text(),footmr=w.footmr_refine.isChecked(),correction=w.motion_correction.isChecked()))
                return
            self.real_next()
        patch=mock.patch.object(w,'next_job',next_job);patch.start();self.addCleanup(patch.stop)
    def take(self,ident,a,b,kind='initial',ready=True):
        """What the person job and the batch leave behind: a take folder with ExMR's temporary frames and, when `ready`, the
        result and its preview, listed in people.json; the scene camera files the shared floor reads."""
        relative=f'p{ident}/takes/{a:09d}_{b:09d}_scene_v1';d=self.root/relative;(d/'exmr_frames').mkdir(parents=True,exist_ok=True)
        for i in range(b-a):(d/'exmr_frames'/f'{i:06d}.jpg').write_bytes(b'jpg')
        (d/'exmr_frames'/'receipt.json').write_text('{}',encoding='utf-8')
        if not (d/'input.mp4').exists():os.link(self.root/'input.mp4',d/'input.mp4')
        (d/'project.json').write_text(json.dumps(dict(self.manifest,frames=b-a,start=a/30,end=b/30,people_root='../../..',person_id=ident,person_preview=False,
                                                    person_frames=[a,b],pipeline=dict(model=self.MODEL,result=kind),correction={})),encoding='utf-8')
        if ready:
            write_result(d,'initial',b'result %d %d %d'%(ident,a,b))
            (d/'scene_camera.json').write_text(json.dumps(dict(sources={'hmr4d_results.pt':stamp(d/'hmr4d_results.pt')},model=[1,2],world_to_camera=IDENTITY)),encoding='utf-8')
            if not (self.root/'people_scene.json').exists():
                (self.root/'people_scene.json').write_text(json.dumps(dict(anchor=relative,input=ps.media_receipt(self.root),
                    sources={'hmr4d_results.pt':stamp(d/'hmr4d_results.pt')},world_to_camera=IDENTITY)),encoding='utf-8')
        state=ps.load_people(self.root);person=next(p for p in state['people'] if p['id']==ident)
        person['takes']=[t for t in person['takes'] if (t['first'],t['end'])!=(a,b)]+[dict(project=relative,first=a,end=b)];ps.save_people(self.root,state)
        return d
    def end(self,failed=False,cancelled=False,status=None,then=None):
        """The end of the last batch: on success each person job's takes (one per visible run of its frames, as the person
        job makes them) and, for a corrected result, the correct stage; then the real end of the queue."""
        w=self.w;call=self.calls[-1]
        if not failed and not cancelled:
            made=[self.take(ident,a,b,call['result']) for ident in call['people'] for a,b in ps.visible_runs(SEEN[ident],30,*call['frames'])]
            if call['result'].startswith('corrected_'):correct(self.root,made)
        self.finish(failed,cancelled,status)
        if then:then()
    def finish(self,failed=False,cancelled=False,status=None,stage='export'):
        w=self.w
        if status:w.status.setText(status)
        w.failed,w.cancelled,w.current_stage=failed,cancelled,stage;self.real_next()
    def start(self,delete=True,**kw):
        w=self.w;w.footmr_refine.setChecked(False)
        runner=parts_runner.start(w,3.,.5,delete,**kw)
        self.assertIsNotNone(runner,w.status.text());return runner
    def wait_calls(self,n):self.assertTrue(pump(lambda:len(self.calls)>=n or self.w.parts_runner is None),(len(self.calls),self.w.status.text()))
    def finish_all(self):
        while self.w.parts_runner is not None:   # each part's call has been made: end it and wait for the next one
            n=len(self.calls);self.end();self.wait_calls(n+1)
    def checks(self,root=None):return [p['checked'] for p in ps.load_people(root or self.root)['people']]
    def gap(self,runner,index,action):
        """Runs `action` once between parts, before the runner's step once part `index` (from 0) waits for its reloaded clip
        and the clip is open: what the user, or the window itself, may do between two timer ticks of the runner."""
        step,done=runner.step,[]
        def wrapped():
            if runner.index==index and runner.phase=='video' and self.w.video_readable and not done:done.append(1);action()
            return step()
        runner.step=wrapped
    def rerun(self):
        """A second run with the same settings; it may end at once when every part is finished."""
        self.calls.clear();parts_runner.start(self.w,3.,.5,True);self.assertTrue(pump(lambda:self.w.parts_runner is None or self.calls))

    def test_parts_run_in_order_with_the_people_visible_in_each(self):
        w=self.w;self.start();self.finish_all()
        self.assertEqual([c['trim'] for c in self.calls],PLAN);self.assertEqual([c['frames'] for c in self.calls],PLAN)
        self.assertEqual([c['people'] for c in self.calls],[[1,2],[1,2],[1],[1]])   # Ben leaves the picture at frame 100
        self.assertEqual([c['status'] for c in self.calls],['Part 1 of 4: frames 1–90.','Part 2 of 4: frames 76–165.','Part 3 of 4: frames 151–240.','Part 4 of 4: frames 226–300.'])
        self.assertEqual([tuple(round(x,6) for x in c['range']) for c in self.calls],[(a/30,b/30) for a,b in PLAN])
        self.assertTrue(all(c['footmr'] is False for c in self.calls))   # the settings survive reloading the scene clip
        self.assertEqual(w.status.text(),'All 4 parts are reconstructed. Choose Fill hidden ranges in the People panel to join them.')
        self.assertIn('Part 3 of 4: Ben is not visible, so this part has no take of them.',w.log.toPlainText())
        self.assertEqual(self.checks(),[True,True])   # the user's checks are back
        saved = parts_runner.saved(self.root)
        self.assertEqual({key: saved[key] for key in ('version', 'first', 'end', 'part', 'overlap', 'complete')},
                         dict(version=2, first=0, end=300, part=90, overlap=15, complete=True))
        anna,ben=ps.load_people(self.root)['people']
        self.assertEqual(sorted(k for k in anna['gap_fill']),['parts_150_165','parts_225_240','parts_75_90'])
        self.assertEqual(anna['gap_fill']['parts_75_90'],dict(method='parts',takes=['p1/takes/000000000_000000090_scene_v1','p1/takes/000000075_000000165_scene_v1']))
        self.assertEqual(list(ben['gap_fill']),['parts_75_90'])
        w.activate_person(1,w.people_root/anna['takes'][0]['project']);APP.processEvents()
        self.assertEqual([x.text() for x in w.hidden_ranges._widgets][:2],['Frames 76–90 (0.5 s)','Parts overlap: blended'])
        self.assertTrue(w.parts_button.isEnabled());self.assertFalse(w.cancel_button.isEnabled())
    def test_the_view_and_trim_of_the_start_are_back_at_the_end(self):
        w=self.w;w.slider.set_clip(0,240);self.start();self.finish_all()
        self.assertEqual([c['trim'] for c in self.calls],[(0,90),(75,165),(150,240)])
        self.assertEqual((w.project,(w.slider.clip_in,w.slider.clip_out)),((self.root/'p1').resolve(),(0,240)))
    def test_the_view_is_back_after_a_failed_part(self):
        w=self.w;w.slider.set_clip(0,240);self.start();self.wait_calls(1);self.end();self.wait_calls(2)
        self.end(failed=True,status='ExMR stopped: the GPU ran out of memory.')
        self.assertTrue(pump(lambda:w.parts_runner is None))
        self.assertEqual((w.project,(w.slider.clip_in,w.slider.clip_out)),((self.root/'p1').resolve(),(0,240)))
    def test_finished_parts_are_skipped(self):
        for ident,a,b in((1,0,90),(2,0,90),(1,75,165),(2,75,100),(1,150,240)):self.take(ident,a,b)
        self.take(1,225,300,ready=False)   # stopped before its result was ready
        self.w.people_state=ps.load_people(self.root)   # as the earlier run left the window
        self.start();self.finish_all()
        self.assertEqual([c['trim'] for c in self.calls],[(225,300)])
        self.assertIn('Part 1 of 4: nothing to reconstruct, skipped.',self.w.log.toPlainText())
    def test_with_motion_correction_a_part_is_finished_when_its_base_motion_is_ready(self):
        """Every part adds takes to the shared scene floor, so the corrected results of the earlier parts go out of date;
        joining reads the base motion, and the joined take is corrected after joining."""
        from mocap.motion_tools import result_is_current,selected_export
        w=self.w;w.motion_correction.setChecked(True);self.start();self.finish_all()
        self.assertEqual({c['result'] for c in self.calls},{'corrected_initial'})
        first=self.root/'p1/takes/000000000_000000090_scene_v1';last=self.root/'p1/takes/000000225_000000300_scene_v1'
        self.assertFalse(result_is_current(first,'corrected_initial'))   # the real receipt check: the floor changed after part 1
        self.assertTrue(result_is_current(last,'corrected_initial'))
        self.assertIsNotNone(selected_export(first,'initial'))
        self.assertEqual(w.status.text(),'All 4 parts are reconstructed. Choose Fill hidden ranges in the People panel to join them.')
        w.motion_correction.setChecked(True);self.rerun()
        self.assertEqual(self.calls,[]);self.assertEqual(w.status.text(),'All 4 parts are reconstructed. Choose Fill hidden ranges in the People panel to join them.')
    def test_a_part_without_a_ready_result_is_named_at_the_end(self):
        w=self.w;self.start();self.wait_calls(1);self.end();self.wait_calls(2)
        for _ in range(2):   # parts 2 and 3: the batch ends without a result (the export did not finish)
            n=len(self.calls);call=self.calls[-1]
            for ident in call['people']:
                for a,b in ps.visible_runs(SEEN[ident],30,*call['frames']):self.take(ident,a,b,ready=False)
            self.finish();self.wait_calls(n+1)
        self.end();self.assertTrue(pump(lambda:w.parts_runner is None))
        self.assertEqual(w.status.text(),'Reconstruct in parts finished, but 2 of 4 parts have no ready result: Part 2 of 4, Part 3 of 4. See Console.')
    def test_a_failed_part_stops_the_run_and_keeps_the_finished_ones(self):
        self.start();self.wait_calls(1);self.end();self.wait_calls(2)
        self.end(failed=True,status='ExMR stopped: the GPU ran out of memory.')
        self.assertTrue(pump(lambda:self.w.parts_runner is None))
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.w.status.text(),'Part 2 of 4 stopped: ExMR stopped: the GPU ran out of memory. Finished parts are kept. Use Resume in Reconstruct in parts to continue.')
        self.assertTrue((self.root/'p1/takes/000000000_000000090_scene_v1/hmr4d_results.pt').is_file())
        self.assertEqual(self.checks(),[True,True])
        self.calls.clear();self.start();self.wait_calls(1);self.assertEqual(self.calls[0]['trim'],(75,165))   # a later run goes on there
    def test_resume_after_allocation_failure_keeps_completed_people_and_parts(self):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QDialogButtonBox
        from mocap.hidden_ranges import chains
        from mocap.parts_ui import PartsDialog, ask
        w = self.w
        self.start(); self.wait_calls(1); self.end(); self.wait_calls(2)
        first = self.root/'p1/takes/000000000_000000090_scene_v1/hmr4d_results.pt'
        ben = self.take(2, 75, 100)/'hmr4d_results.pt'
        failed = self.take(1, 75, 165, ready=False)
        before = (stamp(first), stamp(ben), (failed/'project.json').read_bytes())
        w.handle_event({'type': 'failed', 'message': 'CUDA out of memory. Tried to allocate 1 GiB.'})
        self.end(failed=True); self.wait_calls(3)
        self.assertEqual(self.calls[-1]['people'], [1])
        self.assertLess(self.calls[-1]['frames'][1]-self.calls[-1]['frames'][0], 90)
        self.end(); self.wait_calls(4)
        w.close()
        self.assertIsNone(w.parts_runner)
        self.open_window()
        fresh = self.w
        self.assertIsNot(fresh, w)
        fresh.exmr_camera.setCurrentIndex(fresh.exmr_camera.findData(False))
        with mock.patch.object(fresh, 'checked_people', return_value=[]), mock.patch.object(parts_runner, 'fitted', return_value=(4., 'video memory')):
            # Exercise the actual dialog button without leaving a modal loop running on failure.
            execute = PartsDialog.exec
            def click_resume(dialog):
                resume, = [button for button in dialog.buttons.buttons() if button.text() == 'Resume']
                dialog.part.setValue(6.)
                self.assertFalse(dialog.buttons.button(QDialogButtonBox.Ok).isEnabled())
                QTimer.singleShot(0, resume.click)
                return execute(dialog)
            with mock.patch.object(PartsDialog, 'exec', click_resume):
                self.assertIsNotNone(ask(fresh), fresh.status.text())
        self.wait_calls(1)
        self.assertTrue(fresh.exmr_camera.currentData())
        self.finish_all()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual([c['people'] for c in self.calls], [[1], [1], [1]])
        self.assertEqual((stamp(first), stamp(ben), (failed/'project.json').read_bytes()), before)
        self.assertEqual(self.checks(), [True, True])
        anna = ps.load_people(self.root)['people'][0]
        takes = ps.seam_takes(anna)
        self.assertNotIn(failed.relative_to(self.root).as_posix(), [t['project'] for t in takes])
        chain, = chains(takes, anna['gap_fill'])
        self.assertEqual((chain[0]['first'], chain[-1]['end']), (0, round(self.END * 30)))
        self.assertTrue(parts_runner.saved(self.root)['complete'])

    def test_guard_stop_and_split_limit_survive_reopen(self):
        w = self.w
        self.start(); self.wait_calls(1)
        w.processing_guard = types.SimpleNamespace(reason='memory', check=lambda: 'Windows allocation limit reached.')
        with mock.patch.object(w, 'process', mock.Mock(processId=lambda: 0)), mock.patch('mocap.memory_note.collect', return_value=''):
            w.check_processing_resources()
        self.assertTrue(w.failed)
        self.assertTrue(w.cancelled)
        self.end(failed=True, cancelled=True); self.wait_calls(2)
        w.handle_event({'type': 'failed', 'message': 'Cannot allocate memory'})
        self.end(failed=True); self.wait_calls(3)
        limited = parts_runner.saved(self.root)
        self.assertEqual(limited['parts'][0][-1], 2)
        w.cancel_job(); self.end(cancelled=True)
        self.assertTrue(pump(lambda: w.parts_runner is None))
        w.close(); self.open_window(); w = self.w
        self.assertIsNotNone(parts_runner.resume(w), w.status.text())
        self.wait_calls(1)
        self.assertEqual(self.calls[0]['trim'], tuple(limited['parts'][0][:2]))
        w.handle_event({'type': 'failed', 'message': 'Cannot allocate memory'})
        self.end(failed=True)
        self.assertTrue(pump(lambda: w.parts_runner is None))
        self.assertIn('No smaller retry is available.', w.status.text())
        self.assertEqual(parts_runner.saved(self.root), limited)
        self.assertEqual(len(self.calls), 1)

    def test_oom_keeps_a_short_interval_and_splits_a_long_one(self):
        tracking = self.root/'p2'/'tracking.npz'
        with np.load(tracking) as source:
            data = dict(source)
        data['observed'][4:] = False
        np.savez_compressed(tracking, **data)
        self.start(); self.wait_calls(1)
        self.w.handle_event({'type': 'failed', 'message': 'Cannot allocate memory'})
        self.end(failed=True); self.wait_calls(2)
        checkpoint = parts_runner.saved(self.root)
        self.assertEqual(checkpoint['parts'][0], [0, 4, [2], 0])
        self.assertEqual(checkpoint['parts'][1][2:], [[1], 1])
        self.assertEqual(checkpoint['superseded'], [[1, 0, 90]])

    def test_new_person_detection_invalidates_checkpoint(self):
        w = self.w
        self.start(); self.wait_calls(1)
        w.cancel_job(); self.end(cancelled=True)
        self.assertTrue(pump(lambda: w.parts_runner is None))
        saved = (self.root/parts_runner.RECORD).read_bytes()
        state = ps.load_people(self.root)
        state['fingerprint'] = 'new detection'
        ps.save_people(self.root, state)
        w.close(); self.open_window()
        self.assertIsNone(parts_runner.resume(self.w))
        self.assertEqual(self.calls, [])
        self.assertEqual((self.root/parts_runner.RECORD).read_bytes(), saved)

    def test_resume_skips_a_ready_interval_of_the_same_person(self):
        w = self.w
        observed = np.zeros(self.manifest['frames'], dtype=bool)
        observed[:20] = observed[40:] = True
        tracking = self.root/'p1'/'tracking.npz'
        with np.load(tracking) as source:
            data = dict(source)
        np.savez_compressed(tracking, **{**data, 'observed': observed})
        self.start(); self.wait_calls(1)
        first = self.take(1, 0, 20)/'project.json'
        ben = self.take(2, 0, 90)/'project.json'
        before = (first.read_bytes(), ben.read_bytes())
        w.cancel_job(); self.end(cancelled=True)
        self.assertTrue(pump(lambda: w.parts_runner is None))
        self.calls.clear()
        self.assertIsNotNone(parts_runner.resume(w), w.status.text())
        self.wait_calls(1)
        self.assertEqual(self.calls[0]['frames'], (40, 90))
        self.assertEqual(self.calls[0]['people'], [1])
        self.assertEqual((first.read_bytes(), ben.read_bytes()), before)

    def test_cancel_job_stops_the_run(self):
        self.start();self.wait_calls(1);self.assertTrue(self.w.cancel_button.isEnabled())
        self.w.cancel_button.click();self.end(cancelled=True)
        self.assertTrue(pump(lambda:self.w.parts_runner is None))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.w.status.text(),'Reconstruct in parts was cancelled at part 1 of 4. Finished parts are kept. Use Resume in Reconstruct in parts to continue.')
    def test_cancel_job_between_parts_stops_the_run(self):
        w=self.w;runner=self.start();self.wait_calls(1)
        def click():
            self.assertFalse(w.busy);self.assertTrue(w.cancel_button.isEnabled());w.cancel_button.click()
        self.gap(runner,1,click);self.end()
        self.assertTrue(pump(lambda:w.parts_runner is None))
        self.assertEqual(len(self.calls),1);self.assertFalse(w.cancel_button.isEnabled())
        self.assertEqual(w.status.text(),'Reconstruct in parts was cancelled before part 2 of 4. Finished parts are kept. Use Resume in Reconstruct in parts to continue.')
        self.assertIn('Part 1 of 4: deleted 182 temporary frame files.',w.log.toPlainText())   # part 1 was finished before the run stopped
    def test_find_person_between_parts_stops_the_run_and_leaves_both_scenes_alone(self):
        """The user clicks Find person while the next part's clip opens: the trim is a part's, so Find person prepares a new
        clip scene and the window loads it when it ends. The run never takes that job for its part and never writes the new
        scene's people into its own scene."""
        w=self.w;runner=self.start();self.wait_calls(1)
        clip=self.base/'clip';scene(clip,dict(self.manifest,frames=90),((1,'Someone',range(90)),))
        def find():
            w.find_people();self.assertEqual(self.dialogs,[]);self.assertEqual([s for s,_ in self.batches[-1]],['prepare','people'])
        self.gap(runner,1,find);self.end()
        self.assertTrue(pump(lambda:w.busy and len(self.batches)==2));before=(self.root/'people.json').read_bytes(),(clip/'people.json').read_bytes()
        for _ in range(10):APP.processEvents();time.sleep(.02)
        self.assertEqual(len(self.calls),1);self.assertIs(w.parts_runner,runner)   # waiting, not reconstructing
        w._people_found_root=clip;self.finish(stage='people')   # Find person ends and the window opens the new scene
        self.assertEqual(w.people_root,clip.resolve())
        self.assertTrue(pump(lambda:w.parts_runner is None or (self.root/'people.json').read_bytes()!=before[0]))
        self.assertEqual((clip/'people.json').read_bytes(),before[1])
        self.assertEqual([p['name'] for p in ps.load_people(self.root)['people']],['Anna','Ben'])   # this scene's people, not the new scene's
        anna,ben=ps.load_people(self.root)['people']
        self.assertEqual([t['project'] for t in anna['takes']],['p1/takes/000000000_000000090_scene_v1'])
        self.assertEqual([t['project'] for t in ben['takes']],['p2/takes/000000000_000000090_scene_v1'])
        self.assertTrue(pump(lambda:w.parts_runner is None));self.assertEqual(len(self.calls),1)
        self.assertEqual(w.status.text(),'The scene changed, so Reconstruct in parts stopped before part 2 of 4. Finished parts are kept. Use Resume in Reconstruct in parts to continue.')
    def test_a_bone_preview_job_between_parts_is_waited_for_and_not_taken_for_a_part(self):
        """The window prepares a bone preview by itself when a loaded take's preview is out of date (prepare_target_preview,
        scheduled by preview_loaded on a real screen): a 'rig' job that the run must wait for."""
        w=self.w;runner=self.start();self.wait_calls(1)
        rig=lambda:w.start_jobs([('rig',['--project',str(self.root/'p1/takes/000000000_000000090_scene_v1'),'--result','initial'])])
        self.gap(runner,1,rig);self.end()
        self.assertTrue(pump(lambda:len(self.batches)==2));self.assertEqual(self.batches[-1][0][0],'rig')
        for _ in range(10):APP.processEvents();time.sleep(.02)
        self.assertEqual(len(self.calls),1)
        self.finish(stage='rig');self.wait_calls(2)
        self.assertEqual(self.calls[1]['trim'],(75,165))
        self.end();self.wait_calls(3);self.assertEqual(self.calls[2]['trim'],(150,240))
    def test_a_job_that_starts_right_after_a_failed_part_does_not_hide_the_failure(self):
        w=self.w;self.start();self.wait_calls(1)
        rig=lambda:w.start_jobs([('rig',['--project',str(self.root/'p1'),'--result','initial'])])
        self.end(failed=True,status='ExMR stopped: the GPU ran out of memory.',then=rig)
        self.assertTrue(w.busy)
        self.assertTrue(pump(lambda:w.parts_runner is None))
        self.assertEqual(w.status.text(),'Part 1 of 4 stopped: ExMR stopped: the GPU ran out of memory. Finished parts are kept. Use Resume in Reconstruct in parts to continue.')
        self.finish(stage='rig');self.assertEqual(len(self.calls),1)
    def test_a_job_that_starts_right_after_a_part_is_waited_for(self):
        w=self.w;self.start();self.wait_calls(1)
        rig=lambda:w.start_jobs([('rig',['--project',str(self.root/'p1'),'--result','initial'])])
        self.end(then=rig)
        self.assertTrue(pump(lambda:'Part 1 of 4: deleted 182 temporary frame files.' in w.log.toPlainText()))   # part 1 is done
        for _ in range(10):APP.processEvents();time.sleep(.02)
        self.assertEqual(len(self.calls),1)
        self.finish(stage='rig');self.wait_calls(2);self.assertEqual(self.calls[1]['trim'],(75,165))
    def test_an_error_at_the_start_stops_the_run_with_its_reason(self):
        w=self.w;(self.root/'p2/tracking.npz').write_bytes(b'not a tracking file')
        self.assertIsNone(parts_runner.start(w,3.,.5,True))
        self.assertTrue(w.status.text().startswith('Part 1 of 4 stopped: '),w.status.text())
        self.assertTrue(w.status.text().endswith(parts_runner.KEPT))
        self.assertIsNone(w.parts_runner);self.assertTrue(w.parts_button.isEnabled());self.assertEqual(self.calls,[])
    def test_an_error_between_parts_stops_the_run_and_gives_back_the_button(self):
        w=self.w;runner=self.start();self.wait_calls(1)
        def fail(*_):raise OSError('The disk is not ready.')
        self.end(then=lambda:setattr(w,'activate_person',fail))
        self.assertTrue(pump(lambda:w.parts_runner is None))
        del w.activate_person
        self.assertEqual(w.status.text(),'Part 2 of 4 stopped: The disk is not ready. Finished parts are kept. Use Resume in Reconstruct in parts to continue.')
        self.assertFalse(runner.timer.isActive());self.assertTrue(w.parts_button.isEnabled());self.assertFalse(w.cancel_button.isEnabled())
        self.assertEqual(self.checks(),[True,True])
        for _ in range(20):APP.processEvents();time.sleep(.02)
        self.assertEqual(len(self.calls),1)
    def test_closing_the_window_between_parts_stops_the_run(self):
        w=self.w;runner=self.start();self.wait_calls(1)
        self.gap(runner,1,w.close);self.end()
        self.assertTrue(pump(lambda:w.parts_runner is None))
        self.assertFalse(runner.timer.isActive())
        for _ in range(20):APP.processEvents();time.sleep(.02)
        self.assertEqual(len(self.calls),1);self.assertEqual(self.checks(),[True,True])
    def test_temporary_frames_of_a_finished_part_are_deleted(self):
        self.start(delete=True);self.wait_calls(1);self.end();self.wait_calls(2)
        for ident in(1,2):
            d=self.root/f'p{ident}/takes/000000000_000000090_scene_v1'
            self.assertEqual(list((d/'exmr_frames').iterdir()),[]);self.assertTrue((d/'hmr4d_results.pt').is_file() and (d/'input.mp4').is_file())
        self.assertIn('Part 1 of 4: deleted 182 temporary frame files.',self.w.log.toPlainText())   # 90 frames and a receipt per person
    def test_a_frame_held_open_is_kept_and_counted(self):
        held=[]
        def hold():
            f=open(self.root/'p1/takes/000000000_000000090_scene_v1/exmr_frames/000005.jpg','rb');held.append(f);self.addCleanup(f.close)
        self.start(delete=True);self.wait_calls(1);self.end(then=hold);self.wait_calls(2)
        self.assertIn('Part 1 of 4: deleted 181 temporary frame files. 1 could not be deleted (another program may hold them) and stay on disk.',self.w.log.toPlainText())
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
    def test_the_join_is_promised_only_where_fill_hidden_ranges_joins(self):
        w=self.w
        self.assertIsNone(parts_runner.join_issue(w))
        w.exmr_camera.setCurrentIndex(w.exmr_camera.findData(False))
        self.assertEqual(parts_runner.join_issue(w),'Hidden ranges need a fixed camera in this version.')
        self.start();self.finish_all()
        self.assertEqual(w.status.text(),'All 4 parts are reconstructed. They cannot be joined into one take: Hidden ranges need a fixed camera in this version.')


@unittest.skipUnless(FFMPEG.is_file(),'needs the bundled ffmpeg')
class ShortSceneTest(RunnerTest):
    """The scene's end is one frame short of its frame count, so Reconstruct 3D makes the last part's take a frame shorter."""
    END=10.-1/30
    def test_parts_run_in_order_with_the_people_visible_in_each(self):
        self.start();self.finish_all()
        self.assertEqual([c['frames'] for c in self.calls],PLAN[:-1]+[(225,299)])
        self.assertEqual(self.w.status.text(),'All 4 parts are reconstructed. Choose Fill hidden ranges in the People panel to join them.')
        self.rerun();self.assertEqual(self.calls,[])   # all finished
        self.assertIn('parts_225_240',ps.load_people(self.root)['people'][0]['gap_fill'])
    def test_finished_parts_are_skipped(self):pass
    def test_with_motion_correction_a_part_is_finished_when_its_base_motion_is_ready(self):pass


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
    def test_the_intro_says_whether_the_parts_can_be_joined(self):
        from PySide6.QtWidgets import QLabel
        text=lambda d:d.findChildren(QLabel)[0].text()
        self.assertTrue(text(self.dialog()).endswith('Neighbouring parts share the overlap, so Fill hidden ranges can join them into one take.'))
        self.assertTrue(text(self.dialog(join_issue='Hidden ranges are available for GVHMR and ExMR in this version.')).endswith(
            'Neighbouring parts share the overlap, but they cannot be joined into one take: Hidden ranges are available for GVHMR and ExMR in this version.'))


if __name__=='__main__':unittest.main()
