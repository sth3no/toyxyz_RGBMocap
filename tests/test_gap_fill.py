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
        d=donor_walk(14);d['body_pose'][:]=.6;b=pose(2,transl=(0,1,0));a=pose(2,transl=(0,1,1.3))
        v=g.copy_fill(b,a,d,('global_orient','body_pose'))['body_pose'][:,0]
        self.assertTrue(0<v[0]<v[1]<v[4]<.6);np.testing.assert_allclose(v[5:7],.6,atol=1e-9);self.assertTrue(0<v[-1]<v[-5]<.6)
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
    def test_copy_heading_wraps_the_short_way(self):
        d=donor_walk(5);b=pose(1,orient=(0,3.1,0));a=pose(1,orient=(0,-3.1,0),transl=(0,1,0))
        y=g.yaw(g.copy_fill(b,a,d,('global_orient',))['global_orient'])
        s=np.arange(1,4)/4;want=3.1+s*(2*np.pi-6.2);want=np.where(want>np.pi,want-2*np.pi,want)
        self.assertAlmostEqual(abs(y[1]),abs(want[1]),places=9)   # frames 0 and 2 are edge-blended (B=1)
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
