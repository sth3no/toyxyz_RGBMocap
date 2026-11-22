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

def source(first,end,W=WA,pelvis=P0,betas=0.,drift=None,walk=None):
    """One continuous camera-space body motion; each take sees it in its own world W and with its own pelvis."""
    n=end-first;i=np.arange(first,end,dtype=np.float64)
    root=np.column_stack([STEP*i,.1+0*i,4+.002*i]) if walk is None else walk(i)  # camera-space pelvis position, shared by every take
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

PD=np.array([-.03,.28,.04]);WD=WA@rigid((0.,1.1,0.),(1.5,0.,-.5))
WALK=lambda i:np.column_stack([-1+0*i,.1+0*i,3+STEP*i])  # another person walking towards the camera
def copied(a,b,offset=0,mirror=False,W=WD,walk=WALK,**kw):
    """A copy gap between a and b whose donor rows are the frames gap.first-1+offset ... gap.end+offset of another person."""
    d=source(a['end']-1+offset,b['first']+1+offset,W=W,pelvis=PD,walk=walk,**kw)
    return dict(gap(a,b),method='copy',mirror=mirror,offset=offset,donor=dict(pelvis=d['pelvis'],params=d['params'],logits=d['logits']))  # no camera-space donor
def pose(g,value):
    p=g['donor']['params']['initial'];p['body_pose']=np.broadcast_to(value,p['body_pose'].shape).copy();return g
def level(g,a):
    # The synthetic worlds are tilted against the walk, so put the donor's floor at the hidden person's height.
    p=g['donor']['params']['initial'];p['transl'][:,1]+=(a['params']['initial']['transl'][-1]+P0)[1]-(p['transl'][0]+g['donor']['pelvis'])[1];return g
def digest(out):
    import hashlib
    h=hashlib.sha256();bytes_=lambda name,v:(h.update(name.encode()),h.update(np.ascontiguousarray(v).tobytes()))
    for kind in sorted(out['params']):
        for k in sorted(out['params'][kind]):bytes_(f'{kind}.{k}',np.asarray(out['params'][kind][k],np.float64))
    for k in sorted(out['incam']):bytes_(k,np.asarray(out['incam'][k],np.float64))
    for k in('logits','boxes','mask','observed'):bytes_(k,out[k])
    return h.hexdigest()

class CopyAssembleTest(unittest.TestCase):
    def setUp(self):self.a,self.b=source(0,100),source(120,200,W=WB,pelvis=PB)
    def join(self,**kw):return assemble([self.a,self.b],[copied(self.a,self.b,**kw)])
    def test_copy_gap_shapes_and_mask(self):
        from mocap.gap_fill import COPIED
        out=self.join()
        self.assertTrue((out['mask'][100:120]==COPIED).all());self.assertTrue((out['mask'][:100]==0).all() and (out['mask'][120:]==0).all())
        for x in(out['params']['initial']['transl'],out['params']['initial']['body_pose'],out['incam']['transl'],out['incam']['betas'],out['logits'],out['boxes']):self.assertEqual(len(x),200)
        np.testing.assert_allclose(out['boxes'][105],self.a['boxes'][99]+(self.b['boxes'][0]-self.a['boxes'][99])*6/21)  # boxes stay linear
    def test_copy_gap_is_continuous_in_world_and_camera(self):
        out=assemble([self.a,self.b],[level(copied(self.a,self.b),self.a)])
        for x in(out['params']['initial']['transl'],out['incam']['transl']):
            mean=np.linalg.norm(np.diff(x,axis=0),axis=1).mean()
            for at in(100,120):self.assertLess(np.linalg.norm(x[at]-x[at-1]),2*mean,at)
    def test_copy_of_the_same_motion_is_exact(self):
        # The donor is the person's own camera-space motion seen in another world: the gap is the truth.
        a,b=self.a,self.b;out=assemble([a,b],[copied(a,b,walk=None)]);truth=source(0,200)
        np.testing.assert_allclose(out['incam']['transl'][100:120],truth['incam']['transl'][100:120],atol=1e-9)
        # The boundary differences are zero, so the blends change nothing: every gap frame is the truth.
        for part in('incam','params'):
            o,t=(out[part],truth[part]) if part=='incam' else (out[part]['initial'],truth[part]['initial'])
            np.testing.assert_allclose(rot(o['global_orient'][100:120]),rot(t['global_orient'][100:120]),atol=1e-9)
            np.testing.assert_allclose(o['body_pose'][100:120],t['body_pose'][100:120],atol=1e-9)
        np.testing.assert_allclose(out['params']['initial']['transl'][100:120],truth['params']['initial']['transl'][100:120],atol=1e-9)
        self.assertLess(out['residuals'][0]['initial'],1e-9)
    def test_copy_gap_uses_donor_pose(self):
        for s in(self.a,self.b):s['params']['initial']['body_pose'][:]=0;s['incam']['body_pose'][:]=0
        g=pose(copied(self.a,self.b),.1);out=assemble([self.a,self.b],[g])   # 9.9 deg per joint: 5 blend frames at each end
        np.testing.assert_allclose(out['params']['initial']['body_pose'][105:115],.1,atol=1e-12);np.testing.assert_allclose(out['incam']['body_pose'][105:115],.1,atol=1e-12)
    def test_copy_gap_offset_picks_donor_frames(self):
        g=copied(self.a,self.b,offset=-30);out=assemble([self.a,self.b],[g])
        np.testing.assert_array_equal(out['logits'][100:120],g['donor']['logits'][1:-1])
        np.testing.assert_allclose(out['logits'][100:120],source(70,90)['logits'])
    def test_copy_gap_mirror(self):
        from mocap.gap_fill import mirror_logits
        for s in(self.a,self.b):s['params']['initial']['body_pose'][:]=0
        lift=np.zeros((21,3));lift[15]=(0.,0.,-.8)  # body_pose joint 15 is the left shoulder, 16 the right; 45.8 deg: 8 blend frames
        g=pose(copied(self.a,self.b,mirror=True),lift.reshape(-1));out=assemble([self.a,self.b],[g])
        mid=out['params']['initial']['body_pose'][110].reshape(21,3)
        np.testing.assert_allclose(mid[16],(0.,0.,.8),atol=1e-12);np.testing.assert_allclose(mid[15],0.,atol=1e-12)
        np.testing.assert_array_equal(out['logits'][100:120],mirror_logits(g['donor']['logits'][1:-1]))
        plain=assemble([self.a,self.b],[pose(copied(self.a,self.b),lift.reshape(-1))])['params']['initial']['body_pose'][110].reshape(21,3)
        np.testing.assert_allclose(plain[15],(0.,0.,-.8),atol=1e-12)
    def test_copy_gap_donor_world_is_irrelevant(self):
        for mirror in(False,True):
            one=self.join(mirror=mirror,W=WA@rigid((0.,1.1,0.),(1.5,0.,-.5)))
            two=self.join(mirror=mirror,W=WA@rigid((0.,-2.3,0.),(-.7,0.,2.)))   # another heading and position on the same floor
            for kind in('params','incam'):
                part=lambda o:o['params']['initial'] if kind=='params' else o['incam']
                for k,v in part(one).items():np.testing.assert_allclose(part(two)[k],v,atol=1e-9,err_msg=f'{mirror} {kind} {k}')
    def test_copy_height_follows_the_donors_floor(self):
        # Heights are measured from each world's floor: a donor standing 0.3 higher stays 0.3 higher between the blends.
        one=self.join();two=self.join(W=WD@rigid(t=(0.,-.3,0.)))
        d=(two['params']['initial']['transl']-one['params']['initial']['transl'])[100:120]
        np.testing.assert_allclose(d[5:15,1],.3,atol=1e-9);np.testing.assert_allclose(d[:,[0,2]],0,atol=1e-9)
        g=copied(self.a,self.b);g['height_scale']=.5;half=assemble([self.a,self.b],[g])['params']['initial']['transl'][100:120]+P0
        donor=g['donor']['params']['initial']['transl']+PD
        np.testing.assert_allclose(half[5:15,1],.5*donor[6:16,1],atol=1e-9)
        np.testing.assert_allclose(half[:,[0,2]],one['params']['initial']['transl'][100:120,[0,2]]+P0[[0,2]],atol=1e-9)
    def test_copy_residual_is_reported(self):
        from mocap.gap_fill import copy_fill
        out=self.join();g=copied(self.a,self.b);J=out['params']['initial'];st={}
        rows=lambda a,b:{k:J[k][a:b]+(P0 if k=='transl' else 0) for k in('global_orient','body_pose','transl')}   # neighbours in the joined world
        copy_fill(rows(98,100),rows(120,122),dict(g['donor']['params']['initial'],transl=g['donor']['params']['initial']['transl']+PD),['global_orient','body_pose'],stats=st)
        self.assertAlmostEqual(out['residuals'][0]['initial'],st['residual'],places=12);self.assertGreater(st['residual'],.01)
        self.assertEqual(assemble([self.a,self.b],[gap(self.a,self.b)])['residuals'],[None])
    def test_copy_gap_rejects_short_donor(self):
        g=copied(self.a,self.b);g['donor']['logits']=g['donor']['logits'][1:]
        with self.assertRaises(ValueError):assemble([self.a,self.b],[g])
    def test_smooth_gap_unchanged_by_this_task(self):
        # Digest of the output of the step-1 assemble (commit afdb9dd) for this exact input.
        a=source(0,50,drift=(.08,0.,-.05));b=source(60,100,W=WT,pelvis=PB);c=source(115,150,W=WC)
        for s in(a,b,c):s['params']['footmr_initial']={**s['params']['initial'],'body_pose':s['params']['initial']['body_pose']+.01}
        self.assertEqual(digest(assemble([a,b,c],[gap(a,b),gap(b,c)])),'141c2ff1c82c1ac77cea42dfb3048f2ce5b525271e25ecf08a9be5fc66c419e2')
    def test_mixed_chain_keeps_smooth_gap(self):
        a,b,c=source(0,50),source(60,100,W=WB,pelvis=PB),source(130,170,W=WC)
        smooth=assemble([a,b,c],[gap(a,b),gap(b,c)]);mixed=assemble([a,b,c],[gap(a,b),copied(b,c)])
        self.assertEqual(list(mixed['mask'][50:60])+list(mixed['mask'][100:130]),[1]*10+[2]*30)
        for k,v in smooth['params']['initial'].items():
            np.testing.assert_array_equal(mixed['params']['initial'][k][:100],v[:100]);np.testing.assert_array_equal(mixed['params']['initial'][k][130:],v[130:])

REST=np.linspace(-.5,.5,66).reshape(22,3)
def straight(calls,end_pose=1.25,drop=0,nan=False,seam=(1.2,2.3),without=(),**response):
    """A fake generator: the given known rows on each side and a straight line across the hidden frames."""
    def generate(req):
        calls.append(req);n=req['frames'];c0,c1=min(5,len(req['before_pelvis'])),min(5,len(req['after_pelvis']))
        t=(np.arange(1,n+1)/(n+1))[:,None];line=lambda b,a:np.vstack([b[-c0:],b[-1]+(a[0]-b[-1])*t,a[:c1]])[drop:]
        out=dict(version=3,context=np.array([c0,c1]),orient=line(req['before_orient'],req['after_orient']),pose=line(req['before_pose'],req['after_pose']),
                 pelvis=line(req['before_pelvis'],req['after_pelvis']),contacts=(np.arange(c0+n+c1-drop)[:,None]+np.arange(4))%3==0,end_pose_cm=np.float64(end_pose),
                 seam_cm=np.array(seam,np.float64))
        if nan:out['pose'][c0+2,5]=np.nan
        return {k:v for k,v in {**out,**response}.items() if k not in without}
    return generate
def auto(a,b,calls,seed=7,**kw):return dict(gap(a,b),method='auto',params=dict(seed=seed),generate=straight(calls,**kw),body=dict(rest=REST,sole=.02))

class AutoAssembleTest(unittest.TestCase):
    def setUp(self):
        self.a,self.b=source(0,100),source(120,200,W=WB,pelvis=PB)
        for s in(self.a,self.b):s['params']['footmr_initial']={**s['params']['initial'],'body_pose':s['params']['initial']['body_pose']+.01}
    def join(self,calls,**kw):return assemble([self.a,self.b],[auto(self.a,self.b,calls,**kw)])
    def test_request(self):
        calls=[];out=self.join(calls);self.assertEqual(len(calls),1);q=calls[0];J=out['params']['initial']
        shapes=dict(before_orient=(45,3),before_pose=(45,63),before_pelvis=(45,3),after_orient=(45,3),after_pose=(45,63),after_pelvis=(45,3),rest=(22,3))
        for k,s in shapes.items():self.assertEqual(np.shape(q[k]),s,k)
        self.assertEqual((q['version'],q['frames'],q['seed'],q['fps']),(3,20,7,30.));self.assertEqual(float(q['sole']),.02)
        self.assertEqual((q['text'].shape,q['text'].dtype.kind,str(q['text'])),((),'U',''))   # no description: an empty text and a zero feature
        self.assertEqual((q['text_feature'].shape,q['text_feature'].dtype),((4096,),np.float32));self.assertFalse(q['text_feature'].any())
        np.testing.assert_array_equal(q['rest'],REST)
        np.testing.assert_allclose(q['before_pelvis'],J['transl'][55:100]+P0,atol=1e-12);np.testing.assert_allclose(q['after_pelvis'],J['transl'][120:165]+P0,atol=1e-12)
        np.testing.assert_allclose(q['before_orient'],J['global_orient'][55:100],atol=1e-12);np.testing.assert_allclose(q['after_pose'],J['body_pose'][120:165],atol=1e-12)
    def test_result(self):
        from mocap.gap_fill import GENERATED
        calls=[];out=self.join(calls);q=calls[0];J=out['params']['initial'];t=(np.arange(1,21)/21)[:,None]
        self.assertTrue((out['mask'][100:120]==GENERATED).all());self.assertTrue((out['mask'][:100]==0).all() and (out['mask'][120:]==0).all())
        self.assertEqual(out['reports'],[dict(end_pose_cm=1.25,seam_cm=[1.2,2.3])]);self.assertIs(type(out['reports'][0]['end_pose_cm']),float)
        self.assertEqual([type(x) for x in out['reports'][0]['seam_cm']],[float,float])
        for k,v in self.a['params']['initial'].items():np.testing.assert_array_equal(J[k][:100],v)   # captured rows at and before the boundary
        np.testing.assert_allclose(J['transl'][[99,120]]+P0,np.r_[q['before_pelvis'][-1:],q['after_pelvis'][:1]],atol=1e-12)
        # The boundary rows of the line are the captured poses, so the fill is the line itself.
        line=lambda b,a:b[-1]+(a[0]-b[-1])*t
        np.testing.assert_allclose(J['transl'][100:120]+P0,line(q['before_pelvis'],q['after_pelvis']),atol=1e-12)
        np.testing.assert_allclose(J['body_pose'][100:120],line(q['before_pose'],q['after_pose']),atol=1e-12)
        np.testing.assert_allclose(rot(J['global_orient'][100:120]),rot(line(q['before_orient'],q['after_orient'])),atol=1e-12)
        c=(np.arange(5,25)[:,None]+np.arange(4))%3==0
        np.testing.assert_array_equal(out['logits'][100:120,:4],np.where(c,4.,-8.))
        np.testing.assert_allclose(out['logits'][100:120,4:],self.a['logits'][-1,4:]+(out['logits'][120,4:]-self.a['logits'][-1,4:])*t,atol=1e-12)
        for x in(J['transl'],out['incam']['transl']):
            mean=np.linalg.norm(np.diff(x,axis=0),axis=1).mean()
            for at in(100,120):self.assertLess(np.linalg.norm(x[at]-x[at-1]),2*mean,at)
    def test_second_kind_blends_to_its_own_boundaries(self):
        out=self.join([]);F,J=out['params']['footmr_initial'],out['params']['initial']
        np.testing.assert_allclose(F['body_pose'][105:115],J['body_pose'][105:115],atol=1e-12)   # 1 degree per joint: 5 blend rows
        self.assertLess(np.abs(F['body_pose'][100]-F['body_pose'][99]).max(),.003);self.assertLess(np.abs(F['body_pose'][119]-F['body_pose'][120]).max(),.003)
        self.assertGreater(np.abs(F['body_pose'][100]-J['body_pose'][100]).max(),.008)
        np.testing.assert_allclose(F['transl'],J['transl'],atol=1e-12)
    def test_short_neighbour(self):
        a=source(0,4);calls=[];out=assemble([a,self.b],[auto(a,self.b,calls)])
        self.assertEqual(len(calls[0]['before_pelvis']),4);self.assertEqual(len(calls[0]['after_pelvis']),45)
        self.assertEqual(out['params']['initial']['body_pose'].shape,(200,63));self.assertTrue(np.isfinite(out['params']['initial']['body_pose']).all())
    def test_bad_response(self):
        for kw in(dict(drop=1),dict(nan=True),dict(version=1),dict(context=np.array([4,5])),dict(end_pose=np.nan),dict(without=('seam_cm',)),
                  dict(seam=(1.,np.nan)),dict(seam=(1.,2.,3.)),dict(seam=1.)):
            with self.assertRaisesRegex(ValueError,'^Generated motion does not match the hidden range$',msg=kw):self.join([],**kw)
    def test_seam_limit(self):
        from mocap import take_join
        self.assertEqual(take_join.AUTO_SEAM_LIMIT_CM,25.)
        self.assertEqual(self.join([],seam=(25.,25.))['reports'][0]['seam_cm'],[25.,25.])   # at the limit: written
        for seam,worst in(((25.1,3.),25.1),((3.,80.4),80.4)):
            with self.assertRaises(take_join.Refused) as caught:self.join([],seam=seam)
            self.assertEqual(str(caught.exception),'Automatic fill could not connect the motion on both sides of frames 101–120: the generated motion misses '
                f'the captured poses next to the range by {worst:.0f} cm. The takes before and after it do not fit together. Check the frames next to the range, '
                'reconstruct those takes again, or choose Do not fill.')
    def test_gap_without_generator(self):
        g=auto(self.a,self.b,[]);del g['generate']
        with self.assertRaisesRegex(ValueError,'^Automatic fill needs a motion generator$'):assemble([self.a,self.b],[g])
        g=auto(self.a,self.b,[]);g['params']={}
        with self.assertRaisesRegex(ValueError,'^Automatic fill needs a motion generator$'):assemble([self.a,self.b],[g])
    def test_generator_gets_a_copy_of_the_rest_joints(self):
        rest=REST.copy();g=auto(self.a,self.b,[]);g['body']=dict(rest=rest,sole=.02);inner=g['generate']
        def generate(req):req['rest'][:]=0;return inner(req)
        g['generate']=generate;assemble([self.a,self.b],[g]);np.testing.assert_array_equal(rest,REST)
    def test_description_reaches_the_request(self):
        calls,texts=[],[];feature=np.arange(4096,dtype=np.float32)/4096
        def describe(text):texts.append(text);return feature
        g=auto(self.a,self.b,calls);g['params']=dict(seed=7,text='A person waves.');g['describe']=describe
        out=assemble([self.a,self.b],[g]);q=calls[0]
        self.assertEqual(texts,['A person waves.']);self.assertEqual((q['text'].shape,str(q['text'])),((),'A person waves.'))
        self.assertEqual(q['text_feature'].dtype,np.float32);np.testing.assert_array_equal(q['text_feature'],feature)
        self.assertEqual(out['reports'],[dict(end_pose_cm=1.25,seam_cm=[1.2,2.3])])
        g=auto(self.a,self.b,[]);g['describe']=describe;assemble([self.a,self.b],[g]);self.assertEqual(len(texts),1)   # no description: not called
        g=auto(self.a,self.b,[]);g['params']=dict(seed=7,text='A person waves.')
        with self.assertRaisesRegex(ValueError,'^Automatic fill needs a motion generator$'):assemble([self.a,self.b],[g])   # a description needs `describe`
    def test_other_methods_report_nothing(self):
        a,b=source(0,100),source(120,200,W=WB,pelvis=PB)
        self.assertEqual(assemble([a,b],[gap(a,b)])['reports'],[None]);self.assertEqual(assemble([a,b],[copied(a,b)])['reports'],[None])
import contextlib,importlib.util,io,json,subprocess,tempfile
from pathlib import Path
from unittest import mock
from mocap import auto_fill,kimodo_runtime,paths,people_store as ps,take_join
FFMPEG=paths.ROOT/'tools/bin/ffmpeg.exe'
TAKE=lambda a,b:f'p1/takes/{a:09d}_{b:09d}_scene_v1'
DONOR='p2/takes/000000000_000000220_scene_v1'

def stamp(p):s=Path(p).stat();return [s.st_size,s.st_mtime_ns]

@unittest.skipUnless(importlib.util.find_spec('torch') and FFMPEG.is_file(),'needs torch and the bundled ffmpeg')
class WriteTest(unittest.TestCase):
    """Synthetic scene on disk; no inference, no user data."""
    MODEL='gvhmr'
    def setUp(self):
        import torch
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=self.root=Path(self.tmp.name)/'scene'
        models=Path(self.tmp.name)/'models';(models/'body_models/smplx').mkdir(parents=True)
        rng=np.random.default_rng(0);template=self.template=rng.normal(size=(5,3));self.pelvis=np.full(5,.2)@template
        np.savez(models/'body_models/smplx/SMPLX_NEUTRAL.npz',J_regressor=np.full((2,5),.2),v_template=template,shapedirs=rng.normal(size=(5,3,10))*.01)
        patch=mock.patch.object(paths,'MODELS',models);patch.start();self.addCleanup(patch.stop)
        self.text_cache=Path(self.tmp.name)/'cache/kimodo_text';patch=mock.patch.object(auto_fill,'TEXT_CACHE',self.text_cache);patch.start();self.addCleanup(patch.stop)
        root.mkdir();subprocess.run([str(FFMPEG),'-v','error','-f','lavfi','-i','testsrc=size=64x48:rate=30','-frames:v','220',
            '-pix_fmt','yuv420p','-c:v','libx264',str(root/'input.mp4')],check=True,creationflags=0x08000000)
        (root/'project.json').write_text(json.dumps(dict(frames=220,fps=30,start=0.,end=220/30,width=64,height=48,pipeline=dict(model=self.MODEL),exmr_options=dict(static_camera=True),stages={})),encoding='utf-8')
        if self.MODEL=='exmr':np.save(root/'exmr_camera.npy',np.tile(np.eye(3,dtype=np.float32),(220,1,1)))
        takes=[self.take(a,b,W,walk) for (a,b),W,walk in(((0,100),WA,None),((110,200),WB,None),((0,220),WD,WALK))]
        self.person=dict(id=1,color='#b5b5b5',preview='p1',takes=takes[:2],gap_fill={'100_110':{'method':'smooth'}})
        other=dict(id=2,name='Anna',color='#3366aa',preview='p2',takes=takes[2:])
        ps.save_people(root,dict(version=ps.VERSION,input=ps.media_receipt(root),people=[self.person,other]))
        self.before={p:stamp(p) for p in root.rglob('*') if p.is_file()}
    def take(self,a,b,W,walk=None):
        """Writes one reconstructed take of person 1 (or, with `walk`, the donor take of person 2) and returns its people.json record."""
        import torch
        root=self.root;d=root/(TAKE(a,b) if walk is None else DONOR);d.mkdir(parents=True);s=source(a,b,W=W,pelvis=self.pelvis,walk=walk);n=b-a
        K=np.array([[80,0,32],[0,80,24],[0,0,1]],np.float32)
        if walk is not None:self.donor=s
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
            people_root='../../..',person_id=1 if walk is None else 2,person_preview=False,person_frames=[a,b],pipeline=dict(model=self.MODEL,result='initial'),stages=dict(gvhmr='done'))),encoding='utf-8')
        return dict(project=TAKE(a,b) if walk is None else DONOR,first=a,end=b)
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
        self.assertEqual(fill['gaps'],[dict(first=100,end=110,method='smooth',params={})]);self.assertEqual(fill['donors'],[])
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
    def test_unreadable_take_manifest_is_refused_without_traceback(self):
        manifest=self.root/TAKE(110,200)/'project.json';text=manifest.read_text(encoding='utf-8')
        for damage in(lambda:manifest.write_text(text[:40],encoding='utf-8'),lambda:manifest.write_text('[]',encoding='utf-8'),manifest.unlink):
            damage();code,events=self.join()
            self.assertEqual((code,events),(1,[dict(type='failed',stage='take_join',message='Take 000000110_000000200_scene_v1 cannot be read. Prepare the person motion again.')]))
            self.assertEqual(self.stderr.getvalue(),'')
        self.assertFalse((self.root/'p1/takes/000000000_000000200_fill_v1').exists())
    def test_exmr_camera_receipt_is_copied(self):
        if self.MODEL!='exmr':self.skipTest('ExMR only')
        from mocap.scene_camera import camera_sources
        self.assertEqual(self.join()[0],0);J=self.root/'p1/takes/000000000_000000200_fill_v1'
        read=lambda d:json.loads((d/'exmr_camera_sources.json').read_text(encoding='utf-8'))
        self.assertEqual(read(J),read(self.root/TAKE(0,100)))
        self.assertIn('exmr_camera_sources.json',camera_sources(J));self.assertFalse((J/'exmr_receipt.json').exists())
    def choose(self,**choice):
        state=ps.load_people(self.root);state['people'][0]['gap_fill']={'100_110':{'method':'copy','person':2,'mirror':False,'offset':5,**choice}}
        ps.save_people(self.root,state)
    def owned(self,relative):return {p:(p.read_bytes(),stamp(p)) for p in (self.root/relative).rglob('*') if p.is_file()}
    def test_copy_join_records_donors(self):
        import torch
        J,D=self.root/'p1/takes/000000000_000000200_fill_v1',self.root/DONOR
        self.choose();donor,sources=self.owned('p2'),self.owned(TAKE(0,100))|self.owned(TAKE(110,200))
        code,events=self.join();self.assertEqual(code,0,events);self.assertIs(events[-1]['cached'],False)
        fill=json.loads((J/'fill.json').read_text(encoding='utf-8'));r=fill['gaps'][0]['residual_cm']
        self.assertEqual(fill['gaps'],[dict(first=100,end=110,method='copy',params=dict(person=2,mirror=False,offset=5),residual_cm=r)])
        self.assertIs(type(r),float);self.assertGreater(r,0)   # beside params, never inside: currency compares params
        info=dict(type='info',stage='take_join',message=f'Frames 101–110 copied from Anna: the path was adjusted by {r:.1f} cm ({r/11:.2f} cm per frame) to meet the captured motion.')
        self.assertEqual([e['type'] for e in events],['progress','info','takes_joined','complete']);self.assertEqual(events[1],info)
        code,events=self.join();self.assertIs(events[-1]['cached'],True);self.assertEqual(events[1],info)   # reported again from fill.json
        self.assertEqual(fill['donors'],[dict(person=2,project=DONOR,first=0,end=220,files={'hmr4d_results.pt':stamp(D/'hmr4d_results.pt')})])
        person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))
        self.assertEqual(list(np.load(J/'fill_mask.npy')[95:115]),[0]*5+[2]*10+[0]*5)
        logits=torch.load(J/'hmr4d_results.pt',weights_only=False)['net_outputs']['static_conf_logits'][0].double().numpy()
        np.testing.assert_allclose(logits[100:110],self.donor['logits'][105:115],atol=1e-5)  # gap frame f copies donor frame f+offset
        self.assertIs(self.join()[1][-1]['cached'],True)
        self.choose(mirror=True);self.assertIs(self.join()[1][-1]['cached'],False)
        self.assertEqual(json.loads((J/'fill.json').read_text(encoding='utf-8'))['gaps'][0]['params'],dict(person=2,mirror=True,offset=5))
        self.assertEqual(self.join('footmr_initial')[0],0)
        self.assertEqual(json.loads((J/'fill.json').read_text(encoding='utf-8'))['donors'][0]['files'].keys(),{'hmr4d_results.pt','footmr_initial.pt'})
        self.assertEqual(self.owned('p2'),donor);self.assertEqual(self.owned(TAKE(0,100))|self.owned(TAKE(110,200)),sources)  # never written
        self.assertEqual(self.join()[0],0);person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))
        (D/'hmr4d_results.pt').write_bytes((D/'hmr4d_results.pt').read_bytes())  # a re-reconstructed donor
        self.assertFalse(ps.join_current(self.root,person,person['joins'][0]))
        self.assertIs(self.join()[1][-1]['cached'],False)
        person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))
    def test_donor_name_that_reads_as_markup_is_reported_as_typed(self):
        state=ps.load_people(self.root);state['people'][1]['name']='Anna <b>';ps.save_people(self.root,state);self.choose()
        code,events=self.join();self.assertEqual(code,0,events)
        self.assertIn(' copied from <span>Anna &lt;b&gt;</span>: ',events[1]['message'])
        for text in('Anna','Tom & Jerry','a > b'):self.assertEqual(take_join._typed(text),text)
        self.assertEqual(take_join._typed('a &lt; b'),'<span>a &amp;lt; b</span>')
    def test_copy_from_unreadable_or_short_donor_is_refused(self):
        import torch
        D=self.root/DONOR/'hmr4d_results.pt';self.choose();good=D.read_bytes()
        record=torch.load(D,weights_only=False);record['smpl_params_global']={k:v[:200] for k,v in record['smpl_params_global'].items()}
        for write in(lambda:torch.save(record,D),lambda:D.write_bytes(b'not a result')):   # shorter than its manifest; unreadable
            write();before=self.snapshot();code,events=self.join()
            self.assertEqual((code,events),(1,[dict(type='failed',stage='take_join',message="Anna's take is not ready to be copied from. Reconstruct and prepare it first.")]))
            self.assertEqual(self.stderr.getvalue(),'');self.assertEqual(self.snapshot(),before)
        D.write_bytes(good);self.assertEqual(self.join()[0],0)
    def test_joined_manifest_that_is_not_an_object_is_rewritten(self):
        J=self.root/'p1/takes/000000000_000000200_fill_v1'
        self.assertEqual(self.join()[0],0);(J/'project.json').write_text('[1]',encoding='utf-8')
        code,events=self.join();self.assertEqual(code,0,self.stderr.getvalue());self.assertIs(events[-1]['cached'],False)
    def test_copy_refusals_write_nothing(self):
        J=self.root/'p1/takes/000000000_000000200_fill_v1'
        self.choose();(self.root/DONOR/'initial_sources.json').write_text(json.dumps({'hmr4d_results.pt':[0,0]}),encoding='utf-8')
        before=self.snapshot();code,events=self.join()
        self.assertEqual((code,events),(1,[dict(type='failed',stage='take_join',message="Anna's take is not ready to be copied from. Reconstruct and prepare it first.")]))
        self.assertEqual(self.stderr.getvalue(),'');self.assertEqual(self.snapshot(),before);self.assertFalse(J.exists())
        self.choose(offset=200);before=self.snapshot();code,events=self.join()  # the donor no longer covers the range
        self.assertEqual((code,events[0]['message']),(1,'No hidden range is selected for filling.'));self.assertEqual(self.snapshot(),before)
    def automatic(self,seed=7,text=None,**choices):
        choice={'method':'auto','seed':seed,**({} if text is None else {'text':text})}
        state=ps.load_people(self.root);state['people'][0]['gap_fill']={'100_110':choice,**choices};ps.save_people(self.root,state)
    def generator(self,calls):
        gen=straight(calls);return mock.patch.object(auto_fill,'generate',lambda folder,request,**kw:gen(request))
    def test_automatic_join(self):
        J=self.root/'p1/takes/000000000_000000200_fill_v1';calls=[];self.automatic()
        with self.generator(calls):code,events=self.join()
        self.assertEqual(code,0,events);self.assertEqual(len(calls),1);self.assertIs(events[-1]['cached'],False)
        q=calls[0];self.assertEqual((q['fps'],q['seed'],q['frames']),(30.,7,10))
        np.testing.assert_allclose(q['rest'],np.full((2,5),.2)@self.template,atol=1e-12);self.assertAlmostEqual(q['sole'],self.template[:,1].min(),places=12)
        fill=json.loads((J/'fill.json').read_text(encoding='utf-8'))
        self.assertEqual(fill['gaps'],[dict(first=100,end=110,method='auto',params=dict(seed=7),
            generator=dict(name='kimodo',identity=kimodo_runtime.identity(),version=3,end_pose_cm=1.25,seam_cm=[1.2,2.3]))])
        self.assertEqual(str(q['text']),'');self.assertFalse(q['text_feature'].any())
        info=dict(type='info',stage='take_join',message='Frames 101–110 generated: the captured poses next to the range were missed by 1.2 cm before it and 2.3 cm after it.')
        self.assertEqual([e['type'] for e in events],['progress','info','takes_joined','complete']);self.assertEqual(events[1],info)
        self.assertEqual(list(np.load(J/'fill_mask.npy')[95:115]),[0]*5+[3]*10+[0]*5)
        person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))
        with self.generator(calls):code,events=self.join()
        self.assertEqual((code,len(calls)),(0,1));self.assertIs(events[-1]['cached'],True);self.assertEqual(events[1],info)   # reported again from fill.json
        self.automatic(seed=8);person=ps.load_people(self.root)['people'][0];self.assertFalse(ps.join_current(self.root,person,person['joins'][0]))
        with self.generator(calls):code,events=self.join()
        self.assertEqual((code,len(calls)),(0,2));self.assertIs(events[-1]['cached'],False);self.assertEqual(calls[1]['seed'],8)
    def test_report_is_rounded(self):
        J=self.root/'p1/takes/000000000_000000200_fill_v1';self.automatic()
        gen=straight([],end_pose=1.23456,seam=(7.86,3.04))
        with mock.patch.object(auto_fill,'generate',lambda folder,request,**kw:gen(request)):code,events=self.join()
        self.assertEqual(code,0,events);g=json.loads((J/'fill.json').read_text(encoding='utf-8'))['gaps'][0]['generator']
        self.assertEqual((g['end_pose_cm'],g['seam_cm']),(1.23,[7.9,3.0]))
        self.assertEqual(events[1]['message'],'Frames 101–110 generated: the captured poses next to the range were missed by 7.9 cm before it and 3.0 cm after it.')
    def test_seam_beyond_the_limit_is_refused(self):
        J=self.root/'p1/takes/000000000_000000200_fill_v1';self.automatic()
        with self.generator([]):self.assertEqual(self.join()[0],0)
        runs,seam=[],[(3.,25.1)];real=auto_fill.generate
        def run(command):
            args=dict(zip(command[1::2],command[2::2]));runs.append(command)
            with np.load(args['--request'],allow_pickle=False) as z:request={k:z[k] for k in z.files}
            np.savez(args['--response'],**straight([],seam=seam[0])(request));return 0,[]
        patches=[mock.patch.object(kimodo_runtime,'issue',lambda:None),mock.patch.object(kimodo_runtime,'command',lambda args:['kimodo',*map(str,args)]),
                 mock.patch.object(auto_fill,'generate',lambda folder,request,**kw:real(folder,request,run=run,**kw))]
        for p in patches:p.start();self.addCleanup(p.stop)
        message=('Automatic fill could not connect the motion on both sides of frames 101–110: the generated motion misses the captured poses next to the range '
                 'by 25 cm. The takes before and after it do not fit together. Check the frames next to the range, reconstruct those takes again, or choose Do not fill.')
        cache=self.root/'p1/auto_fill';taken=lambda:{p:v for p,v in self.snapshot().items() if cache not in p.parents}   # all but the cache
        self.automatic(seed=8);before=taken()
        for count in(1,1):   # the second Fill is served from the cache and refuses at once
            code,events=self.join();self.assertEqual((code,events),(1,[dict(type='failed',stage='take_join',message=message)]));self.assertEqual(len(runs),count)
            self.assertEqual(self.stderr.getvalue(),'');self.assertEqual(taken(),before);self.assertEqual([p.suffix for p in cache.iterdir()],['.npz'])
        person=ps.load_people(self.root)['people'][0];self.assertEqual(len(person['joins']),1)
        self.automatic(seed=7);person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))   # the earlier take is kept
        code,events=self.join();self.assertEqual(code,0);self.assertIs(events[-1]['cached'],True)
        seam[0]=(25.,25.);self.automatic(seed=9);code,events=self.join()   # at the limit
        self.assertEqual((code,len(runs)),(0,2),events);self.assertIs(events[-1]['cached'],False)
        self.assertEqual(json.loads((J/'fill.json').read_text(encoding='utf-8'))['gaps'][0]['generator']['seam_cm'],[25.,25.])
    def test_automatic_range_is_served_from_the_cache(self):
        import torch
        state=ps.load_people(self.root);person=state['people'][0]
        person['takes']=[person['takes'][0],self.take(110,150,WB),self.take(160,200,WC)];ps.save_people(self.root,state)
        self.automatic(seed=3,**{'150_160':{'method':'smooth'}});runs=[];real=auto_fill.generate
        def run(command):
            args=dict(zip(command[1::2],command[2::2]));runs.append(command)
            with np.load(args['--request'],allow_pickle=False) as z:request={k:z[k] for k in z.files}
            np.savez(args['--response'],**straight([])(request));return 0,[]
        patches=[mock.patch.object(kimodo_runtime,'issue',lambda:None),mock.patch.object(kimodo_runtime,'command',lambda args:['kimodo',*map(str,args)]),
                 mock.patch.object(auto_fill,'generate',lambda folder,request,**kw:real(folder,request,run=run,**kw))]
        for p in patches:p.start();self.addCleanup(p.stop)
        rows=lambda J:{k:v[100:110].numpy() for k,v in torch.load(J/'hmr4d_results.pt',weights_only=False)['smpl_params_global'].items()}
        self.assertEqual(self.join()[0],0);self.assertEqual(len(runs),1);first=rows(self.root/'p1/takes/000000000_000000200_fill_v1')
        self.assertEqual([p.suffix for p in (self.root/'p1/auto_fill').iterdir()],['.npz'])
        self.automatic(seed=3);code,events=self.join();J=self.root/'p1/takes/000000000_000000150_fill_v1'
        self.assertEqual((code,len(runs)),(0,1),events);self.assertIs(events[-1]['cached'],False)
        self.assertEqual(next(e for e in events if e['type']=='takes_joined')['projects'],[str(J)])
        for k,v in rows(J).items():np.testing.assert_array_equal(v,first[k],k)
        person=ps.load_people(self.root)['people'][0];self.assertEqual([j['project'] for j in person['joins']],['p1/takes/000000000_000000150_fill_v1'])
        self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))
    def test_earlier_range_changes_later_automatic_ranges(self):
        # Known property: the request is expressed in the world and body of the chain's first take, so filling or unfilling an
        # earlier range of the same person changes it, and the later Automatic range is generated again (and may differ).
        state=ps.load_people(self.root);person=state['people'][0]
        person['takes']=[person['takes'][0],self.take(110,150,WB),self.take(160,200,WC)];ps.save_people(self.root,state)
        runs,requests=[],[];real=auto_fill.generate
        def run(command):
            args=dict(zip(command[1::2],command[2::2]));runs.append(command)
            with np.load(args['--request'],allow_pickle=False) as z:request={k:z[k] for k in z.files}
            requests.append(request);np.savez(args['--response'],**straight([])(request));return 0,[]
        patches=[mock.patch.object(kimodo_runtime,'issue',lambda:None),mock.patch.object(kimodo_runtime,'command',lambda args:['kimodo',*map(str,args)]),
                 mock.patch.object(auto_fill,'generate',lambda folder,request,**kw:real(folder,request,run=run,**kw))]
        for p in patches:p.start();self.addCleanup(p.stop)
        alone,joined={'150_160':{'method':'auto','seed':3}},{'100_110':{'method':'smooth'},'150_160':{'method':'auto','seed':3}}
        for choices,count in((alone,1),(joined,2),(alone,2),(joined,2)):   # each chain's request is generated once, then served from the cache
            state=ps.load_people(self.root);state['people'][0]['gap_fill']=choices;ps.save_people(self.root,state)
            code,events=self.join();self.assertEqual((code,len(runs)),(0,count),events)
        self.assertGreater(np.abs(requests[0]['before_pelvis']-requests[1]['before_pelvis']).max(),1e-3)   # another world
        self.assertEqual((requests[0]['frames'],requests[0]['seed']),(requests[1]['frames'],requests[1]['seed']))
    def test_automatic_needs_30_fps(self):
        for take in(TAKE(0,100),TAKE(110,200)):
            p=self.root/take/'project.json';p.write_text(json.dumps(dict(json.loads(p.read_text(encoding='utf-8')),fps=25)),encoding='utf-8')
        self.automatic();before=self.snapshot();calls=[]
        with self.generator(calls):code,events=self.join()
        self.assertEqual((code,events),(1,[dict(type='failed',stage='take_join',message='Automatic fill needs 30 fps takes; this scene has 25.')]))
        self.assertEqual((calls,self.stderr.getvalue()),([],''));self.assertEqual(self.snapshot(),before)
    def test_unavailable_generator_is_refused(self):
        def missing(folder,request,**kw):raise auto_fill.Unavailable(kimodo_runtime.SETUP)
        self.automatic();before=self.snapshot()
        with mock.patch.object(auto_fill,'generate',missing):code,events=self.join()
        self.assertEqual((code,events),(1,[dict(type='failed',stage='take_join',message=kimodo_runtime.SETUP)]))
        self.assertEqual(self.stderr.getvalue(),'');self.assertEqual(self.snapshot(),before)
        self.assertFalse((self.root/'p1/takes/000000000_000000200_fill_v1').exists());self.assertNotIn('joins',ps.load_people(self.root)['people'][0])
    def fill(self):return json.loads((self.root/'p1/takes/000000000_000000200_fill_v1/fill.json').read_text(encoding='utf-8'))
    def test_automatic_join_with_description(self):
        calls,texts=[],[];feature=np.full(4096,.5,np.float32);self.automatic(text='  A person  waves. ')
        def describe(text,**kw):texts.append(text);return feature
        def join():
            with self.generator(calls),mock.patch.object(auto_fill,'describe',describe):return self.join()
        code,events=join();self.assertEqual(code,0,events);self.assertEqual(texts,['A person waves.'])
        q=calls[0];self.assertEqual(str(q['text']),'A person waves.');np.testing.assert_array_equal(q['text_feature'],feature)
        generator=dict(name='kimodo',identity=kimodo_runtime.identity(),version=3,end_pose_cm=1.25,seam_cm=[1.2,2.3])
        self.assertEqual(self.fill()['gaps'],[dict(first=100,end=110,method='auto',params=dict(seed=7,text='A person waves.'),
            generator=dict(generator,text_encoder=kimodo_runtime.text_identity()))])
        info=dict(type='info',stage='take_join',message='Frames 101–110 generated from the description "A person waves.": the captured poses next to the range '
            'were missed by 1.2 cm before it and 2.3 cm after it.')
        self.assertEqual([e['type'] for e in events],['progress','info','takes_joined','complete']);self.assertEqual(events[1],info)
        person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]))
        code,events=join();self.assertEqual((code,len(calls),len(texts)),(0,1,1));self.assertIs(events[-1]['cached'],True);self.assertEqual(events[1],info)
        self.automatic(text='A person sits.');person=ps.load_people(self.root)['people'][0];self.assertFalse(ps.join_current(self.root,person,person['joins'][0]))
        code,events=join();self.assertEqual((code,len(calls),texts[1:]),(0,2,['A person sits.']));self.assertEqual(str(calls[1]['text']),'A person sits.')
        self.automatic();code,events=join()   # the description removed: generated as before, no text encoder
        self.assertEqual((code,len(calls),len(texts)),(0,3,2));self.assertEqual(str(calls[2]['text']),'');self.assertFalse(calls[2]['text_feature'].any())
        self.assertEqual(self.fill()['gaps'][0],dict(first=100,end=110,method='auto',params=dict(seed=7),generator=generator))
        self.assertEqual(events[1]['message'],'Frames 101–110 generated: the captured poses next to the range were missed by 1.2 cm before it and 2.3 cm after it.')
    def workers(self):
        """Real auto_fill.generate and auto_fill.describe in front of fake workers; returns the lists of generation and text runs."""
        runs,reads=[],[];generate,describe=auto_fill.generate,auto_fill.describe
        def run(command):
            args=dict(zip(command[1::2],command[2::2]));runs.append(command)
            with np.load(args['--request'],allow_pickle=False) as z:request={k:z[k] for k in z.files}
            runs[-1]=request;np.savez(args['--response'],**straight([])(request));return 0,[]
        def read(command):
            args=dict(zip(command[2::2],command[3::2]));reads.append(Path(args['--prompt-file']).read_text(encoding='utf-8'))
            np.save(args['--out'],np.full(4096,len(reads),np.float32));return 0,[]
        patches=[mock.patch.object(kimodo_runtime,'issue',lambda:None),mock.patch.object(kimodo_runtime,'text_issue',lambda:None),
                 mock.patch.object(kimodo_runtime,'command',lambda args:['kimodo',*map(str,args)]),
                 mock.patch.object(auto_fill,'generate',lambda folder,request,**kw:generate(folder,request,run=run,**kw)),
                 mock.patch.object(auto_fill,'describe',lambda text,**kw:describe(text,run=read,**kw))]
        for p in patches:p.start();self.addCleanup(p.stop)
        return runs,reads
    def test_one_reading_per_description(self):
        state=ps.load_people(self.root);person=state['people'][0]
        person['takes']=[person['takes'][0],self.take(110,140,WB),self.take(150,170,WC),self.take(180,200,WD)];ps.save_people(self.root,state)
        same={'method':'auto','seed':3,'text':'A person waves.'};self.automatic(seed=3,text='A person waves.',**{'140_150':same,'170_180':same})
        runs,reads=self.workers();code,events=self.join()
        self.assertEqual((code,len(runs),reads),(0,3,['A person waves.']),events)   # three ranges, one reading
        self.assertEqual([str(r['text']) for r in runs],['A person waves.']*3);self.assertTrue(all(r['text_feature'][0]==1 for r in runs))
        self.automatic(seed=3,text='A person waves.',**{'140_150':dict(same,text='A person sits.'),'170_180':same});code,events=self.join()
        self.assertEqual((code,len(runs),reads[1:]),(0,4,['A person sits.']),events)   # only the changed range is generated again
        self.assertEqual((str(runs[3]['text']),runs[3]['text_feature'][0]),('A person sits.',2.))
        self.assertEqual([g['params'] for g in self.fill()['gaps']],[dict(seed=3,text='A person waves.'),dict(seed=3,text='A person sits.'),dict(seed=3,text='A person waves.')])
        self.assertEqual(len(list(self.text_cache.iterdir())),2)
    def test_description_needs_the_text_files(self):
        sentence='A description needs the Kimodo text adapters and the Llama 3 weights. Use Set up… to get them in Model Manager.'
        runs,reads=self.workers();patch=mock.patch.object(kimodo_runtime,'text_issue',lambda:sentence);patch.start();self.addCleanup(patch.stop)
        self.automatic(text='A person waves.');before=self.snapshot();code,events=self.join()
        self.assertEqual((code,events),(1,[dict(type='failed',stage='take_join',message=sentence)]));self.assertEqual((runs,reads,self.stderr.getvalue()),([],[],''))
        self.assertEqual(self.snapshot(),before);self.assertFalse(self.text_cache.exists());self.assertFalse((self.root/'p1/takes/000000000_000000200_fill_v1').exists())
        self.automatic();code,events=self.join()   # without a description the range is still filled
        self.assertEqual((code,len(runs),reads),(0,1,[]),events);self.assertNotIn('text_encoder',self.fill()['gaps'][0]['generator'])
    def test_step_4_join_stays_current(self):
        calls=[];self.automatic()
        with self.generator(calls):self.assertEqual(self.join()[0],0)
        J=self.root/'p1/takes/000000000_000000200_fill_v1';fill=self.fill();self.assertEqual(fill['gaps'][0]['params'],dict(seed=7))
        fill['gaps'][0]['generator']['version']=2;(J/'fill.json').write_text(json.dumps(fill),encoding='utf-8')   # as step 4 wrote it
        for text in(None,'',' \n '):
            self.automatic(text=text);person=ps.load_people(self.root)['people'][0];self.assertTrue(ps.join_current(self.root,person,person['joins'][0]),repr(text))
            with self.generator(calls):code,events=self.join()
            self.assertEqual((code,len(calls)),(0,1));self.assertIs(events[-1]['cached'],True)
class ExmrWriteTest(WriteTest):MODEL='exmr'
def _take(a,b):return dict(project=f't{a}_{b}',first=a,end=b)
def _join(first,end,*sources,project=None):return dict(project=project or f'j{first}_{end}',first=first,end=end,sources=list(sources))
class OtherJoinsTest(unittest.TestCase):
    """Which recorded joins a new join replaces; takes prepared from different trims may share frames."""
    def kept(self,takes,joins,new):return [j['project'] for j in take_join._other_joins(dict(takes=takes,joins=joins),new)]
    def test_join_of_other_takes_over_shared_frames_stays(self):
        from mocap import hidden_ranges
        takes=[_take(0,100),_take(120,200),_take(150,260),_take(280,400)]
        chains=hidden_ranges.chains(takes,{'100_120':dict(method='smooth'),'260_280':dict(method='smooth')})
        self.assertEqual([[t['project'] for t in c] for c in chains],[['t0_100','t120_200'],['t150_260','t280_400']])  # they share frames 150-200
        early,late=_join(0,200,'t0_100','t120_200'),_join(150,400,'t150_260','t280_400')
        self.assertEqual(self.kept(takes,[early],late),['j0_200']);self.assertEqual(self.kept(takes,[late],early),['j150_400'])
    def test_join_sharing_a_take_or_the_folder_is_replaced(self):
        takes=[_take(0,100),_take(120,200),_take(220,300)]
        self.assertEqual(self.kept(takes,[_join(0,200,'t0_100','t120_200')],_join(0,300,'t0_100','t120_200','t220_300')),[])  # the chain grew
        self.assertEqual(self.kept(takes,[_join(0,300,'t0_100','t120_200','t220_300')],_join(120,300,'t120_200','t220_300')),[])  # it shrank
        self.assertEqual(self.kept(takes,[_join(0,200,'t220_300',project='j0_200')],_join(0,200,'t0_100','t120_200')),[])  # same folder
    def test_join_of_takes_that_are_gone_is_replaced_and_a_distant_one_stays(self):
        takes=[_take(0,100),_take(120,200),_take(500,600),_take(620,700)];new=_join(0,200,'t0_100','t120_200')
        self.assertEqual(self.kept(takes,[_join(150,400,'gone_a','gone_b'),dict(project='j10_20',first=10,end=20),dict(project='bad',first=0,end=9,sources=[1])],new),[])
        self.assertEqual(self.kept(takes,[_join(500,700,'t500_600','t620_700'),_join(900,990,'gone_a','gone_b'),dict(project='unranged')],new),['j500_700','j900_990','unranged'])
class CommandTest(unittest.TestCase):
    def test_gui_command_loads_no_numeric_packages(self):
        import sys
        code=("import sys;from mocap.paths import worker_command;c=worker_command('take_join','--project','x');"
              "print(c[-2:]==['--project','x'],sorted(m for m in ('scipy','numpy','torch','mocap.take_join') if m in sys.modules))")
        out=subprocess.run([sys.executable,'-B','-c',code],cwd=paths.ROOT,capture_output=True,text=True,check=True).stdout
        self.assertEqual(out.strip(),'True []')
if __name__=='__main__':unittest.main()
