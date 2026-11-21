import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from mocap.gap_fill import rigid_params
from mocap.take_join import assemble

STEP=.01
P0=np.array([0.,.3,0.]);PB=np.array([.05,.25,-.02])
def rigid(rv=(0.,0.,0.),t=(0.,0.,0.)):
    T=np.eye(4);T[:3,:3]=R.from_rotvec(rv).as_matrix();T[:3,3]=t;return T
WA=rigid((.4,-.7,.2),(.3,-.1,2.))
# A later take's world differs from A's by a heading and a floor position (gravity-consistent pairing) ...
WB,WC=WA@rigid((0.,.5,0.),(-1.,0.,3.)),WA@rigid((0.,-.9,0.),(.5,0.,-2.))
# ... or, with one frame pair's gravity noise, also by a tilt and a height: yaw 30 deg about Y after 4 deg about X.
SEAM=rigid(R.from_matrix(R.from_euler('y',30,degrees=True).as_matrix()@R.from_euler('x',4,degrees=True).as_matrix()).as_rotvec(),(.4,.07,-.3))
WT=WA@SEAM

def tpair(go_w,go_c,tr_w,tr_c,p):
    # Independent of the code under test: camera pelvis = Rw2c @ world pelvis + t.
    Rw2c=R.from_rotvec(go_c).as_matrix()@R.from_rotvec(go_w).as_matrix().T;T=np.eye(4);T[:3,:3]=Rw2c
    T[:3,3]=np.asarray(tr_c)+p-Rw2c@(np.asarray(tr_w)+p);return T

def source(first,end,W=WA,pelvis=P0,betas=0.,drift=None):
    """One continuous camera-space body motion; each take sees it in its own world W and with its own pelvis."""
    n=end-first;i=np.arange(first,end,dtype=np.float64)
    root=np.column_stack([STEP*i,.1+0*i,4+.002*i])  # camera-space pelvis position, shared by every take
    incam=dict(global_orient=np.column_stack([.3+.002*i,.1+.001*i,-.05+0*i]),body_pose=np.tile(np.linspace(-.3,.3,63),(n,1))+.001*i[:,None],
        transl=root-pelvis,betas=np.full((n,10),betas))
    world=rigid_params(incam,np.linalg.inv(W),pelvis,pelvis)
    if drift is not None:world['transl'][n//2:]+=drift  # world drifts against the camera within the take
    boxes=np.column_stack([i,i+1,i+50,i+100])
    return dict(first=first,end=end,pelvis=np.asarray(pelvis,np.float64),params={'initial':world},incam=incam,
        logits=np.tile(np.linspace(-2,2,6),(n,1))+i[:,None]/100,boxes=boxes,observed=np.ones(n,bool))

def gap(a,b):return dict(first=a['end'],end=b['first'],method='smooth')

def pair(**second):
    a=source(0,100);b=source(110,200,**{'W':WB,'pelvis':PB,**second});return a,b,assemble([a,b],[gap(a,b)])

def rot(v):return R.from_rotvec(v).as_matrix()

class AssembleTest(unittest.TestCase):
    def steps_ok(self,x,at,before=None):
        """Step into and out of a gap no larger than twice the neighbouring captured step."""
        step=lambda k:np.linalg.norm(x[k]-x[k-1])
        self.assertLess(step(at),2*step(at-1 if before is None else before),at)
    def test_two_sources_one_gap_shapes(self):
        _,_,out=pair()
        self.assertEqual(out['params']['initial']['body_pose'].shape,(200,63))
        self.assertEqual(out['params']['initial']['betas'].shape,(200,10))
        self.assertEqual(out['incam']['transl'].shape,(200,3));self.assertEqual(out['logits'].shape,(200,6))
        self.assertEqual(out['boxes'].shape,(200,4));self.assertEqual(out['mask'].dtype,np.uint8)
        self.assertEqual(int(out['mask'].sum()),10);self.assertTrue((out['mask'][100:110]==1).all())
        self.assertTrue((out['mask'][:100]==0).all() and (out['mask'][110:]==0).all())
    def test_first_source_is_untouched(self):
        a,_,out=pair()
        for k,v in a['params']['initial'].items():self.assertTrue(np.array_equal(out['params']['initial'][k][:100],v),k)
        for k,v in a['incam'].items():self.assertTrue(np.array_equal(out['incam'][k][:100],v),k)
        self.assertTrue(np.array_equal(out['logits'][:100],a['logits']))
    def test_second_source_is_anchored_at_the_seam(self):
        _,b,out=pair()
        g,c=out['params']['initial'],out['incam']
        seam=lambda k:tpair(g['global_orient'][k],c['global_orient'][k],g['transl'][k],c['transl'][k],P0)
        np.testing.assert_allclose(seam(110),seam(99),atol=1e-9);np.testing.assert_allclose(seam(99),WA,atol=1e-9)
        np.testing.assert_allclose(c['transl'][110:],b['incam']['transl']+PB-P0,atol=1e-12)  # shape swap keeps the camera pelvis
        np.testing.assert_allclose(c['global_orient'][110:],b['incam']['global_orient'],atol=1e-12)
        truth=rigid_params({k:c[k][110:] for k in('global_orient','transl')},np.linalg.inv(WA),P0,P0)  # B in A's world
        np.testing.assert_allclose(g['transl'][110:],truth['transl'],atol=1e-9)
        np.testing.assert_allclose(rot(g['global_orient'][110:]),rot(truth['global_orient']),atol=1e-9)
        np.testing.assert_allclose(g['body_pose'][110:],b['params']['initial']['body_pose'])
    def test_tilted_seam_keeps_later_take_upright(self):
        a,b,out=pair(W=WT);g,own=out['params']['initial'],b['params']['initial']
        yaw=R.from_euler('y',30,degrees=True).as_matrix()
        # Exactly the yaw part of the seam: the up axis of B stays +Y of its own world.
        np.testing.assert_allclose(rot(g['global_orient'][110:]),yaw@rot(own['global_orient']),atol=1e-9)
        np.testing.assert_allclose((g['transl'][110:]+P0)[:,1],(own['transl']+PB)[:,1],atol=1e-12)  # own pelvis heights
        # Heading and floor position at the seam still follow the camera pairing.
        np.testing.assert_allclose((g['transl'][110]+P0)[[0,2]],(SEAM[:3,:3]@(own['transl'][0]+PB)+SEAM[:3,3])[[0,2]],atol=1e-9)
        step=np.diff(g['transl'][110:],axis=0);np.testing.assert_allclose(step,np.diff(own['transl'],axis=0)@yaw.T,atol=1e-12)
        np.testing.assert_allclose(out['incam']['transl'][110:],b['incam']['transl']+PB-P0,atol=1e-12)  # camera space unchanged
    def test_upright_extracts_yaw(self):
        from mocap.take_join import upright
        T=upright(SEAM,np.array([.2,1.,.5]))
        np.testing.assert_allclose(T[:3,:3],R.from_euler('y',30,degrees=True).as_matrix(),atol=1e-12)
        self.assertEqual(T[1,3],0.)
        np.testing.assert_allclose(upright(rigid((0.,-2.,0.),(1.,2.,3.)),np.zeros(3)),rigid((0.,-2.,0.),(1.,0.,3.)),atol=1e-12)
    def test_drift_stays_in_its_take(self):
        a=source(0,100,drift=(.08,0.,-.05));b=source(110,200,W=WT,pelvis=PB)
        out=assemble([a,b],[gap(a,b)]);g,c=out['params']['initial'],out['incam']
        Rend=tpair(g['global_orient'][99],c['global_orient'][99],g['transl'][99],c['transl'][99],P0)[:3,:3]
        # Horizontally the seam step is the camera-space step; vertically each take keeps its own height.
        np.testing.assert_allclose((g['transl'][110]-g['transl'][99])[[0,2]],(Rend.T@(c['transl'][110]-c['transl'][99]))[[0,2]],atol=1e-9)
        for x in(c['transl'],g['transl'][:,[0,2]]):self.steps_ok(x,100);self.steps_ok(x,110,111)
    def test_gap_is_continuous(self):
        _,_,out=pair()
        for x in(out['params']['initial']['transl'],out['incam']['transl'],out['params']['initial']['body_pose'],out['incam']['global_orient']):
            self.steps_ok(x,100);self.steps_ok(x,110,111)
        np.testing.assert_allclose(np.diff(out['incam']['transl'][:,0]),STEP,atol=1e-9)  # camera-space line stays a line
    def test_betas_come_from_first_source(self):
        a,_,out=pair(betas=.7)
        np.testing.assert_array_equal(out['params']['initial']['betas'],np.tile(a['params']['initial']['betas'][:1],(200,1)))
        np.testing.assert_array_equal(out['incam']['betas'],np.tile(a['incam']['betas'][:1],(200,1)))
    def test_observed_false_in_gap(self):
        a=source(0,100);a['observed'][40]=False;b=source(110,200)
        out=assemble([a,b],[gap(a,b)])
        self.assertFalse(out['observed'][100:110].any());self.assertFalse(out['observed'][40])
        self.assertTrue(out['observed'][:40].all() and out['observed'][110:].all())
        np.testing.assert_allclose(out['boxes'][105],a['boxes'][99]+(b['boxes'][0]-a['boxes'][99])*6/11)
        np.testing.assert_array_equal(out['boxes'][110:],b['boxes'])
        np.testing.assert_allclose(out['logits'][105],a['logits'][99]+(b['logits'][0]-a['logits'][99])*6/11)
    def test_three_sources_chain(self):
        a,b,c=source(0,50),source(60,100,W=WB,pelvis=PB),source(115,150,W=WC)
        out=assemble([a,b,c],[gap(a,b),gap(b,c)])
        self.assertEqual(len(out['mask']),150);self.assertEqual(int(out['mask'].sum()),25)
        self.assertTrue((out['mask'][50:60]==1).all() and (out['mask'][100:115]==1).all())
        g,cam=out['params']['initial'],out['incam'];frames=np.r_[0:50,60:100,115:150]
        truth=rigid_params({k:cam[k][frames] for k in('global_orient','transl')},np.linalg.inv(WA),P0,P0)
        np.testing.assert_allclose(g['transl'][frames],truth['transl'],atol=1e-9)
    def test_tilt_does_not_accumulate_over_three_takes(self):
        seam=lambda ax,tilt,yaw,t:rigid(R.from_matrix(R.from_euler('y',yaw,degrees=True).as_matrix()@R.from_euler(ax,tilt,degrees=True).as_matrix()).as_rotvec(),t)
        a,b,c=source(0,50),source(60,100,W=WA@seam('x',4,30,(.4,.07,-.3)),pelvis=PB),source(115,150,W=WA@seam('z',3,-50,(-.2,-.05,.5)))
        out=assemble([a,b,c],[gap(a,b),gap(b,c)]);g=out['params']['initial']
        for s,P in((b,PB),(c,P0)):
            sl=slice(s['first'],s['end']);own=s['params']['initial']
            Rd=R.from_rotvec(g['global_orient'][sl]).as_matrix()@np.swapaxes(R.from_rotvec(own['global_orient']).as_matrix(),1,2)
            self.assertLess(np.abs(np.degrees(R.from_matrix(Rd).as_rotvec())[:,[0,2]]).max(),1e-6)  # only a rotation about +Y
            np.testing.assert_allclose((g['transl'][sl]+P0)[:,1],(own['transl']+P)[:,1],atol=1e-9)
    def test_hand_poses_dropped_unless_every_source_has_them(self):
        a,b=source(0,100),source(110,200)
        for s in(a,b):s['params']['initial']['left_hand_pose']=np.zeros((s['end']-s['first'],45))
        self.assertEqual(assemble([a,b],[gap(a,b)])['params']['initial']['left_hand_pose'].shape,(200,45))
        del b['params']['initial']['left_hand_pose']
        self.assertNotIn('left_hand_pose',assemble([a,b],[gap(a,b)])['params']['initial'])
    def test_rejects_mismatched_gap(self):
        a,b=source(0,100),source(110,200)
        with self.assertRaises(ValueError):assemble([a,b],[dict(first=100,end=111,method='smooth')])
import contextlib,importlib.util,io,json,subprocess,tempfile
from pathlib import Path
from unittest import mock
from mocap import paths,people_store as ps,take_join
FFMPEG=paths.ROOT/'tools/bin/ffmpeg.exe'
TAKE=lambda a,b:f'p1/takes/{a:09d}_{b:09d}_scene_v1'

def stamp(p):s=Path(p).stat();return [s.st_size,s.st_mtime_ns]

@unittest.skipUnless(importlib.util.find_spec('torch') and FFMPEG.is_file(),'needs torch and the bundled ffmpeg')
class WriteTest(unittest.TestCase):
    """Synthetic scene on disk; no inference, no user data."""
    MODEL='gvhmr'
    def setUp(self):
        import torch
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=self.root=Path(self.tmp.name)/'scene'
        models=Path(self.tmp.name)/'models';(models/'body_models/smplx').mkdir(parents=True)
        rng=np.random.default_rng(0);template=rng.normal(size=(5,3));self.pelvis=np.full(5,.2)@template
        np.savez(models/'body_models/smplx/SMPLX_NEUTRAL.npz',J_regressor=np.full((2,5),.2),v_template=template,shapedirs=rng.normal(size=(5,3,10))*.01)
        patch=mock.patch.object(paths,'MODELS',models);patch.start();self.addCleanup(patch.stop)
        root.mkdir();subprocess.run([str(FFMPEG),'-v','error','-f','lavfi','-i','testsrc=size=64x48:rate=30','-frames:v','220',
            '-pix_fmt','yuv420p','-c:v','libx264',str(root/'input.mp4')],check=True,creationflags=0x08000000)
        (root/'project.json').write_text(json.dumps(dict(frames=220,fps=30,start=0.,end=220/30,width=64,height=48,pipeline=dict(model=self.MODEL),exmr_options=dict(static_camera=True),stages={})),encoding='utf-8')
        if self.MODEL=='exmr':np.save(root/'exmr_camera.npy',np.tile(np.eye(3,dtype=np.float32),(220,1,1)))
        K=np.array([[80,0,32],[0,80,24],[0,0,1]],np.float32);takes=[]
        for (a,b),W in(((0,100),WA),((110,200),WB)):
            d=root/TAKE(a,b);d.mkdir(parents=True);s=source(a,b,W=W,pelvis=self.pelvis);n=b-a
            if self.MODEL=='exmr':(d/'exmr_camera_sources.json').write_text(json.dumps(dict(root='../../..',root_stat=stamp(root/'exmr_camera.npy'),
                first=a,reference_rotation=np.eye(3).tolist(),first_rotation=np.eye(3).tolist())),encoding='utf-8')
            np.savez_compressed(d/'tracking.npz',boxes=s['boxes'].astype(np.float32),observed=s['observed'],K=K)
            t=lambda p:{k:torch.from_numpy(v.astype(np.float32)) for k,v in p.items()}
            torch.save(dict(smpl_params_global=t(s['params']['initial']),smpl_params_incam=t(s['incam']),
                net_outputs=dict(static_conf_logits=torch.from_numpy(s['logits'].astype(np.float32))[None]),source_model=self.MODEL),d/'hmr4d_results.pt')
            foot={**s['params']['initial'],'body_pose':s['params']['initial']['body_pose']+.01}
            torch.save(dict(smpl_params_global=t(foot),source='initial'),d/'footmr_initial.pt')
            (d/'footmr_initial_sources.json').write_text(json.dumps({k:stamp(d/k) for k in('footmr_initial.pt','hmr4d_results.pt','tracking.npz')}),encoding='utf-8')
            (d/'project.json').write_text(json.dumps(dict(frames=n,fps=30,start=a/30,end=b/30,timeline=dict(start=a/30,step=1/30,frames=n),
                people_root='../../..',person_id=1,person_preview=False,person_frames=[a,b],pipeline=dict(model=self.MODEL,result='initial'),stages=dict(gvhmr='done'))),encoding='utf-8')
            takes.append(dict(project=TAKE(a,b),first=a,end=b))
        self.person=dict(id=1,color='#b5b5b5',preview='p1',takes=takes,gap_fill={'100_110':{'method':'smooth'}})
        ps.save_people(root,dict(version=ps.VERSION,input=ps.media_receipt(root),people=[self.person]))
        self.before={p:stamp(p) for p in root.rglob('*') if p.is_file()}
    def snapshot(self):return {p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
    def join(self,result='initial'):
        out,self.stderr=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(self.stderr):code=take_join.main(['--project',str(self.root),'--person-id','1','--result',result])
        return code,[json.loads(l[len('@@MOCAP@@'):]) for l in out.getvalue().splitlines() if l.startswith('@@MOCAP@@')]
    def folder(self):
        J=self.root/'p1/takes/000000000_000000200_fill_v1';return {p:(p.read_bytes(),stamp(p)) for p in [*J.rglob('*'),self.root/'people.json'] if p.is_file()}
    def test_refill_with_unchanged_inputs_writes_nothing(self):
        self.assertEqual(self.join()[0],0);before=self.folder()
        code,events=self.join()
        self.assertEqual(code,0);self.assertEqual([e['type'] for e in events],['progress','takes_joined','complete'])
        self.assertEqual(events[1]['projects'],[str(self.root/'p1/takes/000000000_000000200_fill_v1')]);self.assertIs(events[2]['cached'],True)
        self.assertEqual(self.folder(),before)  # bytes and mtimes
        self.assertIs(self.join('footmr_initial')[1][-1]['cached'],False)  # another result is written
    def test_manifest_names_the_fill_result(self):
        J=self.root/'p1/takes/000000000_000000200_fill_v1'
        self.assertEqual(self.join('corrected_footmr_initial')[0],0)
        self.assertEqual(json.loads((J/'project.json').read_text(encoding='utf-8'))['pipeline'],
            dict(model=self.MODEL,result='corrected_footmr_initial',htd_refine=False,footmr=True,handflow=False))
        self.assertEqual(json.loads((J/'fill.json').read_text(encoding='utf-8'))['kinds'],['initial','footmr_initial'])
    def test_refill_with_other_result_updates_manifest(self):
        J=self.root/'p1/takes/000000000_000000200_fill_v1'
        self.assertEqual(self.join('initial')[0],0);code,events=self.join('corrected_initial')
        self.assertEqual(code,0);self.assertIs(events[-1]['cached'],False)
        self.assertEqual(json.loads((J/'project.json').read_text(encoding='utf-8'))['pipeline']['result'],'corrected_initial')
    def test_expected_failures_are_refused_without_traceback(self):
        manifest=json.loads((self.root/'project.json').read_text(encoding='utf-8'))
        for model,options,message in(('flowhmr',{},'Hidden ranges can be filled only in GVHMR and ExMR scenes.'),('exmr',dict(static_camera='yes'),'Invalid ExMR settings'),
                ('removed_tool',{},None)):
            (self.root/'project.json').write_text(json.dumps(dict(manifest,pipeline=dict(model=model),exmr_options=options)),encoding='utf-8')
            code,events=self.join()
            self.assertEqual((code,[e['type'] for e in events]),(1,['failed']),model);self.assertEqual(self.stderr.getvalue(),'',model)
            if message:self.assertEqual(events[0]['message'],message)
    def test_writes_current_joined_take(self):
        import torch
        from mocap.motion_tools import result_is_current
        code,events=self.join()
        self.assertEqual(code,0,events);self.assertEqual([e['type'] for e in events],['progress','takes_joined','complete'])
        J=self.root/'p1/takes/000000000_000000200_fill_v1'
        self.assertEqual(events[1]['projects'],[str(J)]);self.assertEqual(events[1]['person_id'],1)
        self.assertFalse((J/'fill_pending.json').exists())
        state=ps.load_people(self.root);person=state['people'][0];join=person['joins'][0]
        self.assertEqual(join,dict(project='p1/takes/000000000_000000200_fill_v1',first=0,end=200,sources=[TAKE(0,100),TAKE(110,200)]))
        self.assertEqual(person['gap_fill'],self.person['gap_fill'])
        self.assertTrue(ps.join_current(self.root,person,join));self.assertTrue(result_is_current(J,'initial'))
        fill=json.loads((J/'fill.json').read_text(encoding='utf-8'));self.assertEqual((fill['kind'],fill['kinds']),('initial',['initial']))
        self.assertEqual(fill['gaps'],[dict(first=100,end=110,method='smooth',params={})])
        manifest=json.loads((J/'project.json').read_text(encoding='utf-8'))
        self.assertEqual((manifest['frames'],manifest['person_frames'],manifest['stages'],manifest['joined'],manifest['start']),(200,[0,200],{},True,0.))
        self.assertEqual(manifest['timeline']['frames'],200);self.assertAlmostEqual(manifest['end'],200/30)
        self.assertEqual(manifest['pipeline'],dict(model=self.MODEL,result='initial',htd_refine=False,footmr=False,handflow=False))
        self.assertIs(events[2]['cached'],False)
        record=torch.load(J/'hmr4d_results.pt',weights_only=False)
        self.assertEqual(tuple(record['smpl_params_global']['body_pose'].shape),(200,63));self.assertEqual(record['smpl_params_global']['transl'].dtype,torch.float32)
        self.assertEqual(tuple(record['net_outputs']['static_conf_logits'].shape),(1,200,6));self.assertEqual(record['source_model'],self.MODEL)
        g,c=record['smpl_params_global'],record['smpl_params_incam']
        truth=rigid_params({k:c[k][110:].double().numpy() for k in('global_orient','transl')},np.linalg.inv(WA),self.pelvis,self.pelvis)
        np.testing.assert_allclose(g['transl'][110:].numpy(),truth['transl'],atol=1e-4)  # B moved into A's world at the seam
        self.assertFalse(any(p.name=='scene_camera.json' for p in self.root.rglob('*')))  # no scene camera dependency
        self.assertEqual(int(np.load(J/'fill_mask.npy').sum()),10)
        with np.load(J/'tracking.npz') as t:self.assertEqual(t['boxes'].shape,(200,4));self.assertFalse(t['observed'][100:110].any())
        self.assertEqual(torch.load(J/'camera.pt',weights_only=False)['K'].shape,(3,3))
        for p,s in self.before.items():
            if p.name!='people.json' and 'scene_v1' in str(p):self.assertEqual(stamp(p),s,p)  # source takes untouched
    def test_leftover_kind_of_earlier_join_becomes_stale(self):
        from mocap.motion_tools import result_is_current
        J=self.root/'p1/takes/000000000_000000200_fill_v1'
        self.assertEqual(self.join('corrected_footmr_initial')[0],0)
        self.assertTrue(result_is_current(J,'footmr_initial'))
        self.assertEqual(json.loads((J/'fill.json').read_text(encoding='utf-8'))['kind'],'footmr_initial')
        self.assertEqual(self.join('initial')[0],0)
        self.assertTrue(result_is_current(J,'initial'));self.assertFalse(result_is_current(J,'footmr_initial'))
        self.assertEqual(len(ps.load_people(self.root)['people'][0]['joins']),1)
    def test_refusals_write_nothing(self):
        before=self.snapshot()
        for result,message in(('refined',take_join.UNSUPPORTED),('handflow_initial',take_join.UNSUPPORTED)):
            code,events=self.join(result);self.assertEqual(code,1);self.assertEqual(events,[dict(type='failed',stage='take_join',message=message)])
        self.assertEqual(self.snapshot(),before)  # source takes and people.json byte-identical
        state=ps.load_people(self.root);state['people'][0]['gap_fill']={};ps.save_people(self.root,state);before=self.snapshot()
        code,events=self.join();self.assertEqual(code,1);self.assertEqual(events,[dict(type='failed',stage='take_join',message='No hidden range is selected for filling.')])
        self.assertEqual(self.snapshot(),before)
        self.assertFalse((self.root/'p1/takes/000000000_000000200_fill_v1').exists());self.assertFalse((self.root/'people_media').exists())
    def test_exmr_camera_receipt_is_copied(self):
        if self.MODEL!='exmr':self.skipTest('ExMR only')
        from mocap.scene_camera import camera_sources
        self.assertEqual(self.join()[0],0);J=self.root/'p1/takes/000000000_000000200_fill_v1'
        read=lambda d:json.loads((d/'exmr_camera_sources.json').read_text(encoding='utf-8'))
        self.assertEqual(read(J),read(self.root/TAKE(0,100)))
        self.assertIn('exmr_camera_sources.json',camera_sources(J));self.assertFalse((J/'exmr_receipt.json').exists())
class ExmrWriteTest(WriteTest):MODEL='exmr'
class CommandTest(unittest.TestCase):
    def test_gui_command_loads_no_numeric_packages(self):
        import sys
        code=("import sys;from mocap.paths import worker_command;c=worker_command('take_join','--project','x');"
              "print(c[-2:]==['--project','x'],sorted(m for m in ('scipy','numpy','torch','mocap.take_join') if m in sys.modules))")
        out=subprocess.run([sys.executable,'-B','-c',code],cwd=paths.ROOT,capture_output=True,text=True,check=True).stdout
        self.assertEqual(out.strip(),'True []')
if __name__=='__main__':unittest.main()
