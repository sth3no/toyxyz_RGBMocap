"""A finished take from another video's project, imported as a further take of a person and joined by the Hidden ranges methods."""
import contextlib,importlib.util,io,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from scipy.spatial.transform import Rotation as R
from mocap.gap_fill import rigid_params
from mocap.take_join import assemble,pair
from test_take_join import P0,PB,WA,WB,rigid,source,straight,tpair

WALK=lambda i:np.column_stack([-1.5+.012*i,.2+0*i,6-.01*i])  # the other video: another motion seen by another camera


def forward(orient):
    f=R.from_rotvec(np.asarray(orient,np.float64).reshape(-1,3)).apply([0.,0.,1.]);return np.arctan2(f[:,0],f[:,2])


class PlaceTest(unittest.TestCase):
    """The imported motion continues the receiver's last frame in heading and position and sees the receiver's camera there."""
    def setUp(self):
        from mocap import take_import
        self.ti=take_import
        self.recv=source(0,100,W=WA,pelvis=P0,betas=.1)
        self.other=source(0,90,W=WB@rigid((0.,2.,0.),(3.,0.,-4.)),pelvis=PB,betas=-.2,walk=WALK)
        self.other['params']['footmr_initial']={**self.other['params']['initial'],'body_pose':self.other['params']['initial']['body_pose']+.01}
        self.recv['params']['footmr_initial']=self.recv['params']['initial']
    def place(self,lift=0.):
        r=self.recv;return self.ti.place(self.other['params'],PB,r['params']['initial'],r['incam'],P0,r['params']['initial']['betas'][0],lift)
    def test_heading_continues_at_the_seam(self):
        params,_=self.place()
        for kind in params:
            self.assertAlmostEqual(float(forward(params[kind]['global_orient'][0])[0]),float(forward(self.recv['params']['initial']['global_orient'][-1])[0]),places=10,msg=kind)
    def test_position_continues_horizontally_and_keeps_its_own_height(self):
        params,_=self.place(lift=.03);P=params['initial']['transl']+P0;last=self.recv['params']['initial']['transl'][-1]+P0
        np.testing.assert_allclose(P[0,[0,2]],last[[0,2]],atol=1e-10)
        own=rigid_params(self.other['params']['initial'],np.eye(4),PB,P0)['transl'][:,1]+P0[1]   # the shape swap keeps the pelvis joint where it was
        np.testing.assert_allclose(P[:,1],own+.03,atol=1e-10)
    def test_motion_is_moved_rigidly(self):
        params,_=self.place();P=params['initial']['transl']+P0;raw=self.other['params']['initial']['transl']+PB
        np.testing.assert_allclose(np.linalg.norm(np.diff(P,axis=0),axis=1),np.linalg.norm(np.diff(raw,axis=0),axis=1),atol=1e-10)
        np.testing.assert_array_equal(params['initial']['body_pose'],self.other['params']['initial']['body_pose'])
        np.testing.assert_allclose(params['footmr_initial']['transl'],params['initial']['transl'],atol=1e-12)
    def test_receiver_betas_replace_the_imported_ones(self):
        params,incam=self.place()
        for p in(*params.values(),incam):np.testing.assert_array_equal(p['betas'],np.full((90,10),.1))
    def test_camera_space_is_the_receivers_last_pairing(self):
        params,incam=self.place();g,c=self.recv['params']['initial'],self.recv['incam']
        E=tpair(g['global_orient'][-1],c['global_orient'][-1],g['transl'][-1],c['transl'][-1],P0);G=params['initial']
        for f in(0,45,89):np.testing.assert_allclose(tpair(G['global_orient'][f],incam['global_orient'][f],G['transl'][f],incam['transl'][f],P0),E,atol=1e-9)
        np.testing.assert_array_equal(incam['body_pose'],G['body_pose'])
    def take(self):
        params,incam=self.place();o=self.other
        return dict(first=130,end=220,pelvis=P0,params=params,incam=incam,logits=o['logits'],boxes=np.full((90,4),np.nan),observed=np.zeros(90,bool))
    def test_the_join_seam_transform_is_the_identity(self):
        imported=self.take()
        out=assemble([self.recv,imported],[dict(first=100,end=130,method='smooth')])
        for kind in imported['params']:
            for k in('global_orient','transl','body_pose'):np.testing.assert_allclose(out['params'][kind][k][130:],imported['params'][kind][k],atol=1e-9,err_msg=f'{kind} {k}')
        np.testing.assert_allclose(out['incam']['transl'][130:],imported['incam']['transl'],atol=1e-12)
    def test_the_join_is_continuous_across_the_gap(self):
        out=assemble([self.recv,self.take()],[dict(first=100,end=130,method='smooth')]);x=out['params']['initial']['transl']
        self.assertTrue(np.isfinite(x).all())
        # The gap starts and ends at the same floor position: only the two takes' own heights differ.
        np.testing.assert_allclose(x[130,[0,2]],x[99,[0,2]],atol=1e-9)
        self.assertLess(np.linalg.norm((x[100:130]-x[99])[:,[0,2]],axis=1).max(),.5)
    def test_lift_is_the_rest_leg_difference(self):
        rng=np.random.default_rng(1);model=dict(J_regressor=np.full((2,5),.2),v_template=rng.normal(size=(5,3)),shapedirs=rng.normal(size=(5,3,10))*.01)
        b0,b1=np.zeros(10),np.full(10,.5);leg=lambda b:(np.full(5,.2)@(model['v_template']+model['shapedirs']@b))[1]-(model['v_template'][:,1]+model['shapedirs'][:,1]@b).min()
        body=self.ti.Body(model)
        self.assertAlmostEqual(body.lift(b0,b1),leg(b1)-leg(b0),places=12);self.assertEqual(body.lift(b1,b1),0.)
        np.testing.assert_allclose(body.pelvis(b1),np.full(5,.2)@(model['v_template']+model['shapedirs']@b1),atol=1e-12)


class GapTest(unittest.TestCase):
    def test_seconds_become_frames_within_the_fill_limits(self):
        from mocap import take_import_check as c
        self.assertEqual(c.gap_frames(1.0,30),30);self.assertEqual(c.gap_frames(9.6,30),288)
        for s,ok in((1/30,True),(0.01,False),(0.,False),(9.67,True),(9.7,False),(-1.,False)):self.assertIs(c.gap_issue(c.gap_frames(s,30))is None,ok,s)
        self.assertEqual(c.gap_issue(300),'The gap must be 1 to 290 frames (up to 9.7 s at 30 fps).')


from mocap import auto_fill,paths,people_store as ps,take_import_check as check,take_join
FFMPEG=paths.ROOT/'tools/bin/ffmpeg.exe'
TAKE='p1/takes/000000000_000000100_scene_v1';SOURCE='q1/takes/000000000_000000090_scene_v1'
IMPORTED='p1/takes/000000130_000000220_import_v1';JOINED='p1/takes/000000000_000000220_fill_v1'
K=np.array([[80,0,32],[0,80,24],[0,0,1]],np.float32)
def stamp(p):s=Path(p).stat();return [s.st_size,s.st_mtime_ns]
def frame(video,index):
    import cv2
    capture=cv2.VideoCapture(str(video));capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=capture.read();capture.release();assert ok,(video,index);return image.astype(np.float64)
def count(video):
    import cv2
    capture=cv2.VideoCapture(str(video));n=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release();return n


class Scene:
    """A scene of 120 video frames whose person 1 (Anna) has one take, frames 0-100, and another project with one finished take of 90 frames."""
    MODEL='gvhmr';SOURCE_MODEL=None
    def build(self,base):
        models=base/'models';(models/'body_models/smplx').mkdir(parents=True)
        rng=np.random.default_rng(0);self.template=rng.normal(size=(5,3));self.pelvis=np.full(5,.2)@self.template
        np.savez(models/'body_models/smplx/SMPLX_NEUTRAL.npz',J_regressor=np.full((2,5),.2),v_template=self.template,shapedirs=rng.normal(size=(5,3,10))*.01)
        patch=mock.patch.object(paths,'MODELS',models);patch.start();self.addCleanup(patch.stop)
        root=self.root=base/'scene';root.mkdir()
        subprocess.run([str(FFMPEG),'-v','error','-f','lavfi','-i','testsrc=size=64x48:rate=30','-frames:v','120','-pix_fmt','yuv420p','-c:v','libx264',str(root/'input.mp4')],check=True,creationflags=0x08000000)
        self.scene(root,self.MODEL,120)
        self.recv=self.write(root,TAKE,source(0,100,W=WA,pelvis=self.pelvis,betas=.1),self.MODEL,hands=self.MODEL=='exmr')
        self.person=dict(id=1,name='Anna',color='#b5b5b5',preview='p1',checked=True,first=0,last=100,observations=100,takes=[dict(project=TAKE,first=0,end=100)])
        ps.save_people(root,dict(version=ps.VERSION,input=ps.media_receipt(root),people=[self.person]))
        other=self.other=base/'other'/'gvhmr';other.mkdir(parents=True);(other/'input.mp4').write_bytes(b'another video')
        model=self.SOURCE_MODEL or self.MODEL;self.scene(other,model,300)
        self.src=self.write(other,SOURCE,source(0,90,W=WB@rigid((0.,2.,0.),(3.,0.,-4.)),pelvis=self.pelvis,betas=-.2,walk=WALK),model)
        ps.save_people(other,dict(version=ps.VERSION,input=ps.media_receipt(other),people=[dict(id=1,name='Dancer',color='#b5b5b5',preview='q1',takes=[dict(project=SOURCE,first=0,end=90)])]))
    def scene(self,root,model,frames):
        (root/'project.json').write_text(json.dumps(dict(frames=frames,fps=30,start=0.,end=frames/30,width=64,height=48,source=dict(path=str(root/'input.mp4'),fps=30.),pipeline=dict(model=model),exmr_options=dict(static_camera=True),stages={})),encoding='utf-8')
        if model=='exmr':np.save(root/'exmr_camera.npy',np.tile(np.eye(3,dtype=np.float32),(frames,1,1)))
    def write(self,root,relative,s,model,hands=False):
        import torch
        d=root/relative;d.mkdir(parents=True);n=s['end']-s['first']
        if hands:
            for p in(s['params']['initial'],s['incam']):p.update(left_hand_pose=np.full((n,45),.1),right_hand_pose=np.full((n,45),-.1))
        if model=='exmr':(d/'exmr_camera_sources.json').write_text(json.dumps(dict(root='../../..',root_stat=stamp(root/'exmr_camera.npy'),
            first=s['first'],reference_rotation=np.eye(3).tolist(),first_rotation=np.eye(3).tolist())),encoding='utf-8')
        np.savez_compressed(d/'tracking.npz',boxes=s['boxes'].astype(np.float32),observed=s['observed'],K=K)
        t=lambda p:{k:torch.from_numpy(v.astype(np.float32)) for k,v in p.items()}
        torch.save(dict(smpl_params_global=t(s['params']['initial']),smpl_params_incam=t(s['incam']),
            net_outputs=dict(static_conf_logits=torch.from_numpy(s['logits'].astype(np.float32))[None]),source_model=model),d/'hmr4d_results.pt')
        foot={**s['params']['initial'],'body_pose':s['params']['initial']['body_pose']+.01}
        torch.save(dict(smpl_params_global=t(foot),source='initial'),d/'footmr_initial.pt')
        (d/'footmr_initial_sources.json').write_text(json.dumps({k:stamp(d/k) for k in('footmr_initial.pt','hmr4d_results.pt','tracking.npz')}),encoding='utf-8')
        (d/'project.json').write_text(json.dumps(dict(json.loads((root/'project.json').read_text(encoding='utf-8')),frames=n,fps=30,start=s['first']/30,end=s['end']/30,width=64,height=48,timeline=dict(start=s['first']/30,step=1/30,frames=n),
            people_root='../../..',person_id=1,person_preview=False,person_frames=[s['first'],s['end']],pipeline=dict(model=model,result='initial'),stages={})),encoding='utf-8')
        return s


@unittest.skipUnless(importlib.util.find_spec('torch') and FFMPEG.is_file(),'needs torch and the bundled ffmpeg')
class ImportWriteTest(Scene,unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.build(Path(self.tmp.name))
    def snapshot(self):return {p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
    def run_job(self,module,*args,result='initial'):
        out,self.stderr=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(self.stderr):code=module.main(['--project',str(self.root),'--person-id','1',*args,'--result',result])
        return code,[json.loads(l[len('@@MOCAP@@'):]) for l in out.getvalue().splitlines() if l.startswith('@@MOCAP@@')]
    def bring(self,gap=30,result='initial',source=None):
        from mocap import take_import
        return self.run_job(take_import,'--source',str(source or self.other/SOURCE),'--gap',str(gap),result=result)
    def join(self,method,result='initial'):
        state=ps.load_people(self.root);state['people'][0]['gap_fill']={'100_130':method};ps.save_people(self.root,state)
        return self.run_job(take_join,result=result)
    def world(self,folder,name='hmr4d_results.pt',key='smpl_params_global'):
        import torch
        return {k:v.double().numpy() for k,v in torch.load(self.root/folder/name,weights_only=False)[key].items()}

    def test_import_writes_a_current_take_of_the_person(self):
        import torch
        from mocap import hidden_ranges
        from mocap.motion_tools import result_is_current
        code,events=self.bring();self.assertEqual(code,0,(events,self.stderr.getvalue()))
        self.assertEqual([e['type'] for e in events],['progress','take_imported','complete'])
        T=self.root/IMPORTED
        self.assertEqual(next(e for e in events if e['type']=='take_imported'),dict(type='take_imported',project=str(self.root),person_id=1,projects=[str(T)]))
        person=ps.load_people(self.root)['people'][0]
        self.assertEqual(person['takes'],[dict(project=TAKE,first=0,end=100),dict(project=IMPORTED,first=130,end=220)])
        self.assertEqual([hidden_ranges.gap_key(g) for g in hidden_ranges.gaps(person['takes'])],['100_130'])
        self.assertTrue(result_is_current(T,'initial'))
        with np.load(T/'tracking.npz') as t:
            np.testing.assert_array_equal(t['K'],K);self.assertFalse(t['observed'].any());self.assertEqual(t['boxes'].shape,(90,4));self.assertFalse(np.isfinite(t['boxes']).any())
        m=json.loads((T/'project.json').read_text(encoding='utf-8'))
        self.assertEqual((m['frames'],m['fps'],m['person_frames'],m['person_id'],m['stages'],m['people_root']),(90,30,[130,220],1,{},'../../..'))
        self.assertAlmostEqual(m['start'],130/30);self.assertAlmostEqual(m['end'],220/30);self.assertEqual(m['timeline']['frames'],90)
        self.assertEqual(m['pipeline'],dict(model=self.MODEL,result='initial',htd_refine=False,footmr=False,handflow=False))
        self.assertEqual(m['imported']['project'],str(self.other/SOURCE));self.assertEqual(m['imported']['person'],'Dancer')
        record=torch.load(T/'hmr4d_results.pt',weights_only=False);self.assertEqual(record['source_model'],self.MODEL)
        np.testing.assert_allclose(record['net_outputs']['static_conf_logits'][0].numpy(),self.src['logits'],atol=1e-5)
        self.assertEqual(torch.load(T/'camera.pt',weights_only=False)['K'].shape,(3,3))
    def test_placement_on_disk_continues_the_receiver(self):
        self.assertEqual(self.bring()[0],0)
        G,C,last,lc=self.world(IMPORTED),self.world(IMPORTED,key='smpl_params_incam'),self.world(TAKE),self.world(TAKE,key='smpl_params_incam')
        np.testing.assert_allclose((G['transl'][0]-last['transl'][-1])[[0,2]],0,atol=1e-5)
        self.assertAlmostEqual(float(forward(G['global_orient'][0])[0]),float(forward(last['global_orient'][-1])[0]),places=5)
        E=tpair(last['global_orient'][-1],lc['global_orient'][-1],last['transl'][-1],lc['transl'][-1],self.pelvis)
        np.testing.assert_allclose(tpair(G['global_orient'][40],C['global_orient'][40],G['transl'][40],C['transl'][40],self.pelvis),E,atol=1e-4)
        np.testing.assert_allclose(G['betas'],np.full((90,10),.1),atol=1e-7)
    def test_footmr_result_is_imported_too(self):
        from mocap.motion_tools import result_is_current
        self.assertEqual(self.bring(result='footmr_initial')[0],0)
        self.assertTrue(result_is_current(self.root/IMPORTED,'footmr_initial'))
        F,G=self.world(IMPORTED,'footmr_initial.pt'),self.world(IMPORTED)
        np.testing.assert_allclose(F['body_pose'],G['body_pose']+.01,atol=1e-5);np.testing.assert_allclose(F['transl'],G['transl'],atol=1e-6)
    def test_media_beyond_the_video_shows_its_last_frame(self):
        self.assertEqual(self.bring()[0],0);video=self.root/IMPORTED/'input.mp4'
        self.assertEqual(count(video),90);last=frame(self.root/'input.mp4',119)
        for i in(0,45,89):self.assertLess(np.abs(frame(video,i)-last).mean(),3.,i)
    def test_smooth_join_writes_one_take_across_the_import(self):
        from mocap.motion_tools import result_is_current
        self.assertEqual(self.bring()[0],0);code,events=self.join(dict(method='smooth'))
        self.assertEqual(code,0,(events,self.stderr.getvalue()));J=self.root/JOINED
        self.assertTrue(result_is_current(J,'initial'));person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))
        G,I=self.world(JOINED),self.world(IMPORTED)
        for k in('global_orient','transl','body_pose'):np.testing.assert_allclose(G[k][130:],I[k],atol=1e-5,err_msg=k)   # the seam transform is the identity
        self.assertEqual(list(np.load(J/'fill_mask.npy')[98:132]),[0]*2+[1]*30+[0]*2)
        self.assertEqual(count(J/'input.mp4'),220);self.assertEqual(json.loads((J/'project.json').read_text(encoding='utf-8'))['frames'],220)
        self.assertLess(np.abs(frame(J/'input.mp4',219)-frame(self.root/'input.mp4',119)).mean(),3.)
        self.assertLess(np.abs(frame(J/'input.mp4',50)-frame(self.root/'input.mp4',50)).mean(),3.)
    def test_automatic_join_across_the_import(self):
        calls=[];gen=straight(calls);self.assertEqual(self.bring()[0],0)
        with mock.patch.object(auto_fill,'generate',lambda folder,request,**kw:gen(request)):code,events=self.join(dict(method='auto',seed=3))
        self.assertEqual(code,0,(events,self.stderr.getvalue()));self.assertEqual(len(calls),1);self.assertEqual(calls[0]['frames'],30)
        self.assertEqual(list(np.load(self.root/JOINED/'fill_mask.npy')[100:130]),[3]*30)
        # The generator's two ends are at the same floor position, as the import was placed there.
        np.testing.assert_allclose((calls[0]['before_pelvis'][-1]-calls[0]['after_pelvis'][0])[[0,2]],0,atol=1e-5)
    def test_refusals_write_nothing(self):
        before=self.snapshot();src=self.other/SOURCE;m=json.loads((src/'project.json').read_text(encoding='utf-8'))
        def manifest(**change):(src/'project.json').write_text(json.dumps(dict(m,**change)),encoding='utf-8')
        stale=lambda name,key:(src/name).write_text(json.dumps({key:[0,0]}),encoding='utf-8')
        other=check.MODEL.format(check.LABELS[self.MODEL])
        cases=[(lambda:manifest(fps=25),{},check.RATE.format(25)),(lambda:manifest(pipeline=dict(model='flowhmr')),{},other),
               (lambda:manifest(pipeline=dict(model='gemx')),{},other),(lambda:None,dict(gap=0),check.GAP),(lambda:None,dict(gap=291),check.GAP),
               (lambda:None,dict(result='refined'),check.UNSUPPORTED),(lambda:None,dict(source=self.root/TAKE),check.SAME),
               (lambda:stale('initial_sources.json','hmr4d_results.pt'),{},check.NOT_FINISHED),
               (lambda:stale('footmr_initial_sources.json','footmr_initial.pt'),dict(result='footmr_initial'),check.FOOTMR),
               (lambda:None,dict(result='handflow_initial'),check.FINGERS)]
        good=(src/'footmr_initial_sources.json').read_bytes()
        for damage,kw,message in cases:
            manifest();(src/'initial_sources.json').unlink(missing_ok=True);(src/'footmr_initial_sources.json').write_bytes(good);damage()
            code,events=self.bring(**kw)
            self.assertEqual((code,events),(1,[dict(type='failed',stage='take_import',message=message)]),message);self.assertEqual(self.stderr.getvalue(),'',message)
            self.assertEqual(self.snapshot(),before,message)
        self.assertFalse((self.root/IMPORTED).exists())
    def test_a_receiver_without_a_finished_take_is_refused(self):
        (self.root/TAKE/'initial_sources.json').write_text(json.dumps({'hmr4d_results.pt':[0,0]}),encoding='utf-8');before=self.snapshot()
        self.assertEqual(self.bring(),(1,[dict(type='failed',stage='take_import',message=check.RECEIVER.format('Anna'))]));self.assertEqual(self.snapshot(),before)
    def test_refusals_are_one_sentence(self):
        for name in('UNSUPPORTED','SCENE_MODEL','STATIC','MODEL','RATE','NOT_FINISHED','FILLED','SAME','RECEIVER','LAST','FOOTMR','FINGERS','GAP'):
            text=getattr(check,name);self.assertTrue(text.endswith('.'),name);self.assertNotIn('. ',text,name)
    def test_a_second_import_continues_after_the_first(self):
        self.assertEqual(self.bring()[0],0);self.assertEqual(self.bring(gap=10)[0],0)
        takes=ps.load_people(self.root)['people'][0]['takes']
        self.assertEqual([(t['first'],t['end']) for t in takes],[(0,100),(130,220),(230,320)])
    def test_the_dialog_lists_finished_takes_of_the_other_project(self):
        listed=check.sources(self.other.parent)
        self.assertEqual([(x['project'],x['person'],x['frames'],x['model']) for x in listed],[(self.other/SOURCE,'Dancer',90,self.SOURCE_MODEL or self.MODEL)])
        self.assertIn('Dancer · frames 1–90 · ',listed[0]['label']);self.assertTrue(listed[0]['label'].endswith(' · 3.0 s'))
        (self.other/SOURCE/'initial_sources.json').write_text(json.dumps({'hmr4d_results.pt':[0,0]}),encoding='utf-8')
        self.assertEqual(check.sources(self.other.parent),[]);self.assertEqual(check.sources(self.root/'p1'),[])
    def fill(self,gaps,**extra):
        """A joined take of the other project, as take_join records it: its fill.json lists the filled hidden ranges."""
        import shutil
        joined=self.other/'q1/takes/000000000_000000090_fill_v1';shutil.copytree(self.other/SOURCE,joined,dirs_exist_ok=True)
        (joined/'fill.json').write_text(json.dumps(dict(version=1,first=0,end=90,gaps=gaps,**extra)),encoding='utf-8');return joined
    def test_a_take_with_filled_frames_is_refused(self):
        joined=self.fill([dict(first=40,end=50,method='auto',params={})]);before=self.snapshot()
        self.assertEqual(self.bring(source=joined),(1,[dict(type='failed',stage='take_import',message=check.FILLED)]));self.assertEqual(self.snapshot(),before)
        (joined/'fill.json').write_text('{',encoding='utf-8')   # unreadable: it cannot say that no frame was filled
        self.assertEqual(check.refusal(self.root,self.person,joined,30,'initial'),check.FILLED)
    def test_the_dialog_lists_no_joined_take_with_filled_frames(self):
        joined=self.fill([dict(first=40,end=50,method='smooth',params={})]);entry=dict(project='q1/takes/000000000_000000090_fill_v1',first=0,end=90,join=True)
        with mock.patch.object(ps,'active_takes',lambda root,person:[entry]):
            self.assertEqual([x['project'] for x in check.sources(self.other.parent)],[self.other/SOURCE])
            self.fill([],overlaps=[dict(first=40,end=50)])   # parts joined over shared frames: every frame was captured
            listed=check.sources(self.other.parent)
        self.assertEqual([x['project'] for x in listed],[self.other/SOURCE,joined]);self.assertIn(' · joined · ',listed[1]['label'])
        self.assertIsNone(check.refusal(self.root,self.person,joined,30,'initial'))


class ExmrImportWriteTest(ImportWriteTest):MODEL='exmr'


@unittest.skipUnless(importlib.util.find_spec('torch') and FFMPEG.is_file(),'needs torch and the bundled ffmpeg')
class GvhmrIntoExmrTest(Scene,unittest.TestCase):
    """ExMR stands its world on the floor (lowest vertex at y = 0) and GVHMR starts its pelvis at y = 0, so a take of the other model is refused."""
    MODEL='exmr';SOURCE_MODEL='gvhmr'
    setUp=ImportWriteTest.setUp;snapshot=ImportWriteTest.snapshot;run_job=ImportWriteTest.run_job;bring=ImportWriteTest.bring
    def test_a_take_of_another_model_is_refused(self):
        before=self.snapshot();message=check.MODEL.format(check.LABELS[self.MODEL])
        self.assertEqual(self.bring(),(1,[dict(type='failed',stage='take_import',message=message)]));self.assertEqual(self.snapshot(),before)
        self.assertEqual(check.refusal(self.root,self.person,self.other/SOURCE,30,'initial'),message)


class ExmrIntoGvhmrTest(GvhmrIntoExmrTest):MODEL='gvhmr';SOURCE_MODEL='exmr'


class CommandTest(unittest.TestCase):
    def test_gui_command_runs_the_import_and_loads_no_numeric_packages(self):
        import sys
        code=("import sys;from mocap.paths import worker_command;c=worker_command('take_import','--project','x');import mocap.take_import_check;"
              "print(c[-2:]==['--project','x'],'mocap.take_import' in c,sorted(m for m in ('scipy','numpy','torch','mocap.take_import') if m in sys.modules))")
        out=subprocess.run([sys.executable,'-B','-c',code],cwd=paths.ROOT,capture_output=True,text=True,check=True).stdout
        self.assertEqual(out.strip(),'True True []')
    def test_managed_command_uses_its_boot_file(self):
        from mocap import take_join_runtime
        with mock.patch.dict('os.environ',RGB_MOCAP_MANAGED='1'):c=take_join_runtime.import_command(['--project','x'])
        self.assertEqual(Path(c[c.index('-u')+1]).name,'take_import_boot.py');self.assertTrue((paths.ROOT/'take_import_boot.py').is_file())
    def test_stage_has_a_name_in_the_status_bar(self):
        self.assertIn("STAGES['take_import'] = 'Import Take'",(paths.ROOT/'mocap/gui.py').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
