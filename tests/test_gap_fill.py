import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from mocap import gap_fill as g

def pose(n,orient=(0,0,0),transl=(0,0,0)):
    return dict(global_orient=np.tile(np.asarray(orient,float),(n,1)),body_pose=np.zeros((n,63)),
                transl=np.tile(np.asarray(transl,float),(n,1)),betas=np.zeros((n,10)))

def donor_walk(n, heading=0.):                       # walks 0.1 m per frame along its facing, starting at the origin
    d=np.array([np.sin(heading),0,np.cos(heading)])
    p=pose(n,orient=(0,heading,0));p['transl']=np.arange(n)[:,None]*.1*d+np.array([0,1.,0]);return p

class GapFillTest(unittest.TestCase):
    def test_rigid_keeps_world_pelvis_when_betas_differ(self):
        T=np.eye(4);T[:3,:3]=R.from_euler('y',90,degrees=True).as_matrix();T[:3,3]=[1,0,2]
        a,b=np.array([0,.9,0]),np.array([0,1.,0])
        out=g.rigid_params(pose(3,transl=(1,0,0)),T,a,b)
        np.testing.assert_allclose(out['transl'][0]+b,T[:3,:3]@(np.array([1,0,0])+a)+T[:3,3],atol=1e-9)
        np.testing.assert_allclose(R.from_rotvec(out['global_orient'][0]).as_matrix(),T[:3,:3],atol=1e-9)
        np.testing.assert_array_equal(out['body_pose'],np.zeros((3,63)))
    def test_constant_pose_stays_constant(self):
        out=g.smooth_fill(pose(2,transl=(1,2,3)),pose(2,transl=(1,2,3)),5)
        self.assertEqual(out['transl'].shape,(5,3));self.assertEqual(out['body_pose'].shape,(5,63))
        np.testing.assert_allclose(out['transl'],np.tile([1,2,3],(5,1)),atol=1e-9)
        self.assertNotIn('left_hand_pose',out);self.assertNotIn('betas',out)
    def test_slerp_midpoint(self):
        out=g.smooth_fill(pose(2),pose(2,orient=(0,np.pi/2,0)),1)
        np.testing.assert_allclose(out['global_orient'][0],[0,np.pi/4,0],atol=1e-9)
    def test_slerp_takes_short_way(self):
        out=g.smooth_fill(pose(2,orient=(0,np.radians(170),0)),pose(2,orient=(0,np.radians(-170),0)),1)
        self.assertAlmostEqual(abs(np.degrees(R.from_rotvec(out['global_orient'][0]).magnitude())),180,places=6)
    def test_root_matches_boundary_velocity(self):
        before=pose(2);before['transl']=np.array([[-.1,0,0],[0,0,0]])
        after=pose(2);after['transl']=np.array([[1.1,0,0],[1.2,0,0]])
        out=g.smooth_fill(before,after,10)['transl']
        self.assertAlmostEqual(out[0,0]-0,.1,delta=.02);self.assertAlmostEqual(1.1-out[-1,0],.1,delta=.02)
        self.assertTrue(np.all(np.diff(np.r_[0,out[:,0],1.1])>0))
    def test_tangent_clamp_prevents_overshoot(self):
        before=pose(2);before['transl']=np.array([[-5.,0,0],[0,0,0]])
        after=pose(2);after['transl']=np.array([[.1,0,0],[.1,0,0]])
        out=g.smooth_fill(before,after,10)['transl']
        self.assertLess(out[:,0].max(),.2)
    def test_logits(self):
        out=g.fill_logits(np.zeros(6),np.full(6,3.),2)
        np.testing.assert_allclose(out,[[1.]*6,[2.]*6])
    def test_yaw(self):
        np.testing.assert_allclose(g.yaw(np.array([[0,.5,0],[0,-2.,0]])),[.5,-2.],atol=1e-12)
        tilted=(R.from_rotvec([0,.7,0])*R.from_rotvec([.2,0,0])).as_rotvec()
        self.assertAlmostEqual(float(g.yaw(tilted)),.7,places=9)
    def test_mirror_twice_is_identity(self):
        p=donor_walk(4,.3);p['body_pose']=np.random.default_rng(0).normal(0,.3,(4,63))
        q=g.mirror_params(g.mirror_params(p))
        for k in ('global_orient','body_pose','transl'):np.testing.assert_allclose(R.from_rotvec(q[k].reshape(-1,3)).as_matrix() if k!='transl' else q[k],
            R.from_rotvec(p[k].reshape(-1,3)).as_matrix() if k!='transl' else p[k],atol=1e-9)
    def test_mirror_swaps_arm_and_keeps_forward(self):
        p=donor_walk(3);p['body_pose'][:,15*3:15*3+3]=[0,0,1.]          # left shoulder (joint 16 → body index 15)
        p['transl'][:,0]+=np.array([0,.2,.4])                          # sidestep along +X while walking +Z
        q=g.mirror_params(p)
        np.testing.assert_allclose(q['body_pose'][:,16*3:16*3+3],np.tile([0,0,-1.],(3,1)),atol=1e-12)   # right shoulder, mirrored axis
        np.testing.assert_allclose(q['body_pose'][:,15*3:15*3+3],0,atol=1e-12)
        np.testing.assert_allclose(q['transl'][:,0],-p['transl'][:,0],atol=1e-12)                       # sidestep flipped
        np.testing.assert_allclose(q['transl'][:,[1,2]],p['transl'][:,[1,2]],atol=1e-12)                # height and forward kept
    def test_mirror_logits(self):
        np.testing.assert_array_equal(g.mirror_logits(np.arange(12.).reshape(2,6)),[[2,3,0,1,5,4],[8,9,6,7,11,10]])
    def test_copy_is_exact_at_both_boundaries(self):
        d=donor_walk(12);b=pose(2,transl=(5,1,5));a=pose(2,transl=(5.3,1,6.4))
        out=g.copy_fill(b,a,d,('global_orient','body_pose'))
        self.assertEqual(out['transl'].shape,(10,3));self.assertEqual(out['body_pose'].shape,(10,63))
        step=np.linalg.norm(np.diff(np.vstack([b['transl'][-1],out['transl'],a['transl'][0]]),axis=0),axis=1)
        self.assertLess(step.max(),2*step.mean())                         # no jump at either seam
    def test_copy_follows_the_hidden_persons_heading(self):
        d=donor_walk(12);b=pose(2,orient=(0,np.pi/2,0),transl=(0,1,0));a=pose(2,orient=(0,np.pi/2,0),transl=(1.1,1,0))
        out=g.copy_fill(b,a,d,('global_orient','body_pose'))
        np.testing.assert_allclose(out['transl'],np.c_[np.arange(1,11)*.1,np.ones(10),np.zeros(10)],atol=1e-9)   # walks +X, not +Z
        np.testing.assert_allclose(g.yaw(out['global_orient']),np.pi/2,atol=1e-9)
    def test_copy_blends_pose_at_the_edges(self):
        d=donor_walk(32);d['body_pose'][:]=.6;b=pose(2,transl=(0,1,0));a=pose(2,transl=(0,1,3.1))   # n=30; 59.5 deg per joint: 10 blend frames
        st={};v=g.copy_fill(b,a,d,('global_orient','body_pose'),stats=st)['body_pose'][:,0];self.assertEqual(st['blend'],(10,10))
        self.assertTrue(np.all(np.diff(np.r_[0,v[:11]])>0));np.testing.assert_allclose(v[10:20],.6,atol=1e-12)   # rises from the boundary, then the donor's
        np.testing.assert_allclose(v[:10],.6*g._smoothstep(np.arange(1,11)/11),atol=1e-12);np.testing.assert_allclose(v[::-1],v,atol=1e-12)
    def test_copy_blend_length_follows_the_difference(self):
        for n,want in((40,15),(20,10)):
            d=donor_walk(n+2);b=pose(2,transl=(0,1,0));a=pose(2,transl=(0,1,.1*(n+1)));b['body_pose'][:,3:6]=[np.pi/2,0,0]   # 90 deg on one joint
            st={};o=g.copy_fill(b,a,d,('global_orient','body_pose'),stats=st);self.assertEqual(st['blend'],(want,5))
            j=np.vstack([b['body_pose'][-1:,3:6],o['body_pose'][:,3:6]]);step=np.degrees((R.from_rotvec(j[1:])*R.from_rotvec(j[:-1]).inv()).magnitude())
            if n==40:self.assertLess(step.max(),10)
            np.testing.assert_allclose(o['body_pose'][want:,3:6],0,atol=1e-12)
    def test_copy_difference_composes_on_the_right(self):
        n=12;d=donor_walk(n+2);d['body_pose'][:,:3]=np.c_[np.zeros(n+2),.08*np.arange(n+2),np.full(n+2,.4)]   # turning joint
        b=pose(2,transl=(0,1,0));a=pose(2,transl=(0,1,1.3));b['body_pose'][:,:3]=[.5,0,0];a['body_pose'][:,:3]=d['body_pose'][-1,:3]
        st={};o=g.copy_fill(b,a,d,('global_orient','body_pose'),stats=st)['body_pose'][:,:3];B=st['blend'][0]
        v=R.from_rotvec(d['body_pose'][:,:3]);D0=(v[0].inv()*R.from_rotvec([.5,0,0])).as_rotvec();L0=(R.from_rotvec([.5,0,0])*v[0].inv()).as_rotvec()
        for j in range(1,B+1):
            u=1-g._smoothstep(j/(B+1))
            np.testing.assert_allclose(o[j-1],(v[j]*R.from_rotvec(D0*u)).as_rotvec(),atol=1e-12)
            self.assertGreater(np.linalg.norm(o[j-1]-(R.from_rotvec(L0*u)*v[j]).as_rotvec()),1e-3)   # left side would differ
        np.testing.assert_allclose(o[B:],d['body_pose'][B+1:n+1,:3],atol=1e-12)
    def test_copy_of_a_twin_is_exact(self):
        rng=np.random.default_rng(3);N,f0,f1=60,20,40;t=np.arange(N)[:,None]
        m=dict(global_orient=(R.from_rotvec(np.c_[0*t,.02*t-.5,0*t])*R.from_rotvec(.1*np.sin(t/7+rng.normal(0,1,(1,3))))).as_rotvec(),
               body_pose=.3*np.sin(t/9+rng.normal(0,3,(1,63))),left_hand_pose=.2*np.sin(t/5+rng.normal(0,3,(1,45))),right_hand_pose=.2*np.cos(t/6+rng.normal(0,3,(1,45))),
               transl=np.c_[np.sin(t/10),.9+.05*np.sin(t/4),.03*t],betas=np.zeros((N,10)))
        keys=('global_orient','body_pose','left_hand_pose','right_hand_pose');rows=lambda a,b:{k:v[a:b] for k,v in m.items()}
        M=R.from_rotvec([0,1.3,0])
        moved=lambda d:dict(d,global_orient=(M*R.from_rotvec(d['global_orient'])).as_rotvec(),transl=M.apply(d['transl'])+[3,0,-2])   # same floor
        for name,d in(('same world',rows(f0-1,f1+1)),('turned and moved',moved(rows(f0-1,f1+1))),('mirrored back',g.mirror_params(g.mirror_params(moved(rows(f0-1,f1+1)))))):
            st={};out=g.copy_fill(rows(f0-2,f0),rows(f1,f1+2),d,keys,stats=st)
            for k in(*keys,'transl'):np.testing.assert_allclose(out[k],m[k][f0:f1],atol=1e-9,err_msg=f'{name} {k}')
            self.assertLess(st['residual'],1e-9)
    def test_copy_height_scale(self):
        n=20;d=donor_walk(n+2);b=pose(2,transl=(0,.8,0));a=pose(2,transl=(0,.8,.1*(n+1)))       # donor pelvis at 1.0
        np.testing.assert_allclose(g.copy_fill(b,a,d,('global_orient','body_pose'),height_scale=.8)['transl'][:,1],.8,atol=1e-12)
        a['transl'][:,1]=.6;st={};y=g.copy_fill(b,a,d,('global_orient','body_pose'),height_scale=.8,stats=st)['transl'][:,1]   # reappears 0.2 lower
        B=st['blend'][1];self.assertEqual(B,5);np.testing.assert_allclose(y[:n-B],.8,atol=1e-12)
        self.assertTrue(np.all(np.diff(np.r_[.8,y[n-B:],.6])<0))                          # sinks only inside the end blend
        np.testing.assert_allclose(y[n-B:],.8-.2*(1-g._smoothstep(np.arange(B,0,-1)/(B+1))),atol=1e-12)
    def test_copy_heading_change_across_the_gap(self):
        n=10;d=donor_walk(n+2);d['transl'][:,0]=.03*np.arange(n+2)**1.5                    # curved donor path facing +Z
        b=pose(2,orient=(0,.3,0),transl=(1,1,2));a=pose(2,orient=(0,.9,0),transl=(2.,1,2.5))
        out=g.copy_fill(b,a,d,('global_orient','body_pose'));acc=[b['transl'][-1]];th=[]
        for i in range(1,n+2):
            t=.3+i/(n+1)*.6;c,s=np.cos(t),np.sin(t);th.append(t)
            acc.append(acc[-1]+np.array([[c,0,s],[0,1,0],[-s,0,c]])@(d['transl'][i]-d['transl'][i-1]))
        acc=np.array(acc);res=acc[-1]-a['transl'][0];want=np.array([acc[i]-i/(n+1)*res for i in range(1,n+1)])
        np.testing.assert_allclose(out['transl'][:,[0,2]],want[:,[0,2]],atol=1e-12);np.testing.assert_allclose(out['transl'][:,1],1,atol=1e-12)
        np.testing.assert_allclose(g.yaw(out['global_orient']),th[:n],atol=1e-12)
    def test_copy_short_gap_and_missing_donor_hands(self):
        b=pose(2);a=pose(2);b['left_hand_pose']=np.zeros((2,45));a['left_hand_pose']=np.full((2,45),.2)
        out=g.copy_fill(b,a,donor_walk(3),('global_orient','body_pose','left_hand_pose'))                       # one-frame gap
        self.assertEqual(out['left_hand_pose'].shape,(1,45));np.testing.assert_allclose(out['left_hand_pose'],.1,atol=1e-9)
        self.assertTrue(all(np.isfinite(v).all() for v in out.values()))
    def test_mirror_with_heading_flips_the_donors_side(self):
        p=donor_walk(3,np.pi/2);p['transl']+=np.arange(3)[:,None]*[0,0,-.2]   # facing +X, so its left is -Z
        q=g.mirror_params(p)
        np.testing.assert_allclose(q['transl'],p['transl']*[1,1,-1],atol=1e-12)   # sidestep to its right, forward and height kept
        m=R.from_rotvec(q['global_orient']).as_matrix()
        np.testing.assert_allclose(np.linalg.det(m),1,atol=1e-12);np.testing.assert_allclose(m@m.transpose(0,2,1),np.tile(np.eye(3),(3,1,1)),atol=1e-12)
        np.testing.assert_allclose(g.yaw(q['global_orient']),np.pi/2,atol=1e-12)
    def test_mirror_swaps_hands_and_keeps_other_keys(self):
        p=donor_walk(2);rng=np.random.default_rng(1);p['left_hand_pose']=rng.normal(0,.3,(2,45));p['right_hand_pose']=rng.normal(0,.3,(2,45))
        p['betas'][:]=.5;q=g.mirror_params(p);flip=np.tile([1,-1,-1],15)
        np.testing.assert_allclose(q['right_hand_pose'],p['left_hand_pose']*flip,atol=1e-12)
        np.testing.assert_allclose(q['left_hand_pose'],p['right_hand_pose']*flip,atol=1e-12)
        np.testing.assert_array_equal(q['betas'],p['betas'])
    def test_mirror_turns_a_lean_to_the_left_into_a_lean_to_the_right(self):
        lean=R.from_rotvec([0,0,-.3])                                                 # SMPL +X is the body's left: up tilts to +X
        for heading in(0.,np.pi/2):
            p=pose(1,orient=(R.from_rotvec([0,heading,0])*lean).as_rotvec(),transl=(0,1,0));q=g.mirror_params(p)
            left=R.from_rotvec([0,heading,0]).apply([1,0,0]);up=lambda x:R.from_rotvec(x['global_orient'][0]).apply([0,1,0])
            self.assertAlmostEqual(up(p)@left,np.sin(.3),places=12);self.assertAlmostEqual(up(q)@left,-np.sin(.3),places=12)   # now leans to the right
            self.assertAlmostEqual(up(q)[1],np.cos(.3),places=12);self.assertAlmostEqual(float(g.yaw(q['global_orient'])[0]),heading,places=12)
    def test_copy_heading_wraps_the_short_way(self):
        d=donor_walk(5);b=pose(1,orient=(0,3.1,0));a=pose(1,orient=(0,-3.1,0),transl=(0,1,0))
        y=g.yaw(g.copy_fill(b,a,d,('global_orient',))['global_orient'])
        s=np.arange(1,4)/4;want=3.1+s*(2*np.pi-6.2);want=np.where(want>np.pi,want-2*np.pi,want)
        np.testing.assert_allclose(np.abs(y),np.abs(want),atol=1e-9)   # facing matches at both ends, so the blends change nothing
        self.assertTrue(np.all(np.abs(y)>3.0));self.assertTrue(np.all(np.diff(np.unwrap(np.r_[3.1,y,-3.1]))>0))
    def test_copy_spreads_the_residual(self):
        d=donor_walk(10);b=pose(1,transl=(0,1,0));a=pose(1,transl=(.4,1,1.2))      # donor path ends at (0,1,.9): residual (-.4,0,-.3)
        out=g.copy_fill(b,a,d,('global_orient',))['transl'];s=np.arange(1,9)/9
        np.testing.assert_allclose(out,np.c_[.4*s,np.ones(8),.1*np.arange(1,9)+.3*s],atol=1e-12)
        np.testing.assert_allclose(out[-1]+[0,0,.1]+(a['transl'][0]-[0,1,.9])/9,a['transl'][0],atol=1e-12)   # step n+1 lands on after[0]
    def test_copy_does_not_mutate_and_handles_tiny_gaps(self):
        import copy
        for n in (1,2,3):
            d=g.mirror_params(donor_walk(n+2,.4));d['body_pose'][:]=.3;b=pose(2,orient=(0,np.pi,0));a=pose(2,orient=(0,-2.,0),transl=(.3,1,.2))
            for x in (b,a):x['right_hand_pose']=np.full((2,45),.1)
            keep=copy.deepcopy((b,a,d));out=g.copy_fill(b,a,d,('global_orient','body_pose','right_hand_pose'))
            for x,y in zip((b,a,d),keep):
                for k in y:np.testing.assert_array_equal(x[k],y[k])
            self.assertEqual(set(out),{'global_orient','body_pose','right_hand_pose','transl'})
            self.assertTrue(all(v.shape[0]==n and v.dtype==np.float64 and np.isfinite(v).all() for v in out.values()))
        p=donor_walk(3);keep=copy.deepcopy(p);g.mirror_params(p)
        for k in p:np.testing.assert_array_equal(p[k],keep[k])

def motion(N=40,seed=3):
    """A smooth motion with hands; transl is the pelvis position."""
    rng=np.random.default_rng(seed);t=np.arange(N)[:,None]
    return dict(global_orient=(R.from_rotvec(np.c_[0*t,.02*t-.5,0*t])*R.from_rotvec(.1*np.sin(t/7+rng.normal(0,1,(1,3))))).as_rotvec(),
                body_pose=.3*np.sin(t/9+rng.normal(0,3,(1,63))),left_hand_pose=.2*np.sin(t/5+rng.normal(0,3,(1,45))),right_hand_pose=.2*np.cos(t/6+rng.normal(0,3,(1,45))),
                transl=np.c_[np.sin(t/10),.9+0*t,.03*t])
KEYS=('global_orient','body_pose','left_hand_pose','right_hand_pose')

class GeneratedFillTest(unittest.TestCase):
    def setUp(self):
        self.m=motion();self.f0,self.f1=10,30;rows=lambda a,b:{k:v[a:b].copy() for k,v in self.m.items()}
        self.before,self.after=rows(self.f0-2,self.f0),rows(self.f1,self.f1+2)
        self.gen={k:v for k,v in rows(self.f0-1,self.f1+1).items() if k in('global_orient','body_pose','transl')}   # no hands
    def test_matching_boundaries_keep_the_generated_rows(self):
        out=g.generated_fill(self.before,self.after,self.gen,KEYS)
        self.assertEqual(set(out),{*KEYS,'transl'})
        for k in('global_orient','body_pose','transl'):np.testing.assert_allclose(out[k],self.gen[k][1:-1],atol=1e-12,err_msg=k)
    def test_left_offset_fades_out_and_leaves_the_right_half(self):
        n=self.f1-self.f0;half=n//2;gen={k:v.copy() for k,v in self.gen.items()};O=R.from_euler('z',12,degrees=True)
        j=gen['body_pose'][:half+1,9:12];gen['body_pose'][:half+1,9:12]=(R.from_rotvec(j)*O).as_rotvec()   # rows 0..half, one joint
        gen['transl'][:half+1]+=[.03,0,0]
        out=g.generated_fill(self.before,self.after,gen,KEYS)
        b,t=self.before['body_pose'][-1,9:12],self.m['body_pose'][self.f0,9:12]
        angle=lambda x,y:np.degrees((R.from_rotvec(x).inv()*R.from_rotvec(y)).magnitude())
        self.assertLessEqual(angle(out['body_pose'][0,9:12],b),angle(t,b)+1)   # continues the boundary within 1 degree
        p0=self.before['transl'][-1]
        self.assertLessEqual(np.linalg.norm(out['transl'][0]-p0),np.linalg.norm(self.m['transl'][self.f0]-p0)+.003)
        np.testing.assert_allclose(out['body_pose'][5:],gen['body_pose'][6:n+1],atol=1e-12)   # B = 5: the correction ends there
        np.testing.assert_allclose(out['transl'][5:],gen['transl'][6:n+1],atol=1e-12)
        self.assertGreater(angle(out['body_pose'][4,9:12],gen['body_pose'][5,9:12]),1e-3)
        for k in('global_orient','body_pose','transl'):np.testing.assert_allclose(out[k][half:],self.m[k][self.f0+half:self.f1],atol=1e-12,err_msg=k)
    def test_height_step_fades_without_a_jump(self):
        gen=dict(self.gen,transl=self.gen['transl']+[0,.1,0])
        y=np.r_[self.before['transl'][-1,1],g.generated_fill(self.before,self.after,gen,KEYS)['transl'][:,1],self.after['transl'][0,1]]
        self.assertLess(np.abs(np.diff(y)).max(),.025);self.assertGreater(np.abs(np.diff(y)).max(),.02)
        np.testing.assert_allclose(y[6:-6],.9+.1,atol=1e-12)   # rows beyond the blends keep the generated height
    def test_keys_absent_from_the_generated_motion_are_slerped(self):
        n=self.f1-self.f0;self.before['left_hand_pose'][:]=0;self.after['left_hand_pose'][:]=.2
        out=g.generated_fill(self.before,self.after,self.gen,KEYS)
        np.testing.assert_allclose(out['left_hand_pose'],np.tile(.2*np.arange(1,n+1)[:,None]/(n+1),(1,45)),atol=1e-12)
        self.assertEqual(out['right_hand_pose'].shape,(n,45))
    def short(self,n,shift_after):
        """Truth rows around n hidden frames; the generated rows are offset by 5 cm and 0.3 rad on one joint, and so is `after` when shift_after."""
        rows=lambda a,b:{k:v[a:b].copy() for k,v in self.m.items()};f0=self.f0;f1=f0+n;O=R.from_rotvec([0,0,.3])
        offset=lambda d:dict(d,transl=d['transl']+[.05,0,0],body_pose=np.c_[d['body_pose'][:,:9],(R.from_rotvec(d['body_pose'][:,9:12])*O).as_rotvec(),d['body_pose'][:,12:]])
        gen=offset({k:v for k,v in rows(f0-1,f1+1).items() if k in('global_orient','body_pose','transl')});after=rows(f1,f1+2)
        return rows(f0-2,f0),offset(after) if shift_after else after,gen,rows(f0-1,f1+1)
    def test_short_ranges_correct_every_row(self):
        for n in(1,2,3,9):
            before,after,gen,truth=self.short(n,False)   # the same offset at both ends: every row is the truth again
            out=g.generated_fill(before,after,gen,KEYS)
            self.assertTrue(all(v.shape[0]==n and np.isfinite(v).all() for v in out.values()),n)
            np.testing.assert_allclose(out['transl'],truth['transl'][1:-1],atol=1e-12,err_msg=str(n))
            np.testing.assert_allclose(out['body_pose'],truth['body_pose'][1:-1],atol=1e-12,err_msg=str(n))
            np.testing.assert_allclose(out['global_orient'],truth['global_orient'][1:-1],atol=1e-12,err_msg=str(n))
    def test_short_ranges_spread_a_one_sided_offset(self):
        for n in(1,2,3,9):
            before,after,gen,truth=self.short(n,True)   # the offset only at the start: it is spread evenly over the n+1 steps
            out=g.generated_fill(before,after,gen,KEYS)
            p=np.vstack([before['transl'][-1:],out['transl'],after['transl'][:1]]);t=truth['transl']
            step=np.linalg.norm(np.diff(p,axis=0),axis=1);true=np.linalg.norm(np.diff(t,axis=0),axis=1)
            self.assertLessEqual((step-true).max(),.05/(n+1)+1e-9,n)
            self.assertGreater(step.max(),.05/(n+1)/2,n)   # the offset is actually crossed
            q=R.from_rotvec(np.vstack([before['body_pose'][-1:,9:12],out['body_pose'][:,9:12],after['body_pose'][:1,9:12]]));r=R.from_rotvec(truth['body_pose'][:,9:12])
            step=(q[:-1].inv()*q[1:]).magnitude();true=(r[:-1].inv()*r[1:]).magnitude()
            self.assertLessEqual((step-true).max(),.3/(n+1)+1e-9,n)
    def test_long_ranges_keep_the_end_blends(self):
        for shift in(False,True):   # 17 degrees: 5 blend rows each side; between them the generated rows are kept
            before,after,gen,truth=self.short(20,shift);out=g.generated_fill(before,after,gen,KEYS)
            np.testing.assert_allclose(out['transl'][5:15],gen['transl'][6:16],atol=1e-12);np.testing.assert_allclose(out['body_pose'][5:15],gen['body_pose'][6:16],atol=1e-12)
            np.testing.assert_allclose(out['transl'][0],truth['transl'][1]+np.array([.05,0,0])*g._smoothstep(1/6),atol=1e-12)

class ContactLogitsTest(unittest.TestCase):
    def test_values(self):
        c=np.array([[1,0,0,1],[0,1,1,0],[0,0,0,0]],bool)
        out=g.contact_logits(c,np.zeros(6),np.full(6,4.))
        self.assertEqual(out.shape,(3,6))
        np.testing.assert_array_equal(out[:,:4],np.where(c,4.,-8.))
        np.testing.assert_allclose(out[:,4:],[[1.,1.],[2.,2.],[3.,3.]])
