import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from mocap import gap_fill as g

def pose(n,orient=(0,0,0),transl=(0,0,0)):
    return dict(global_orient=np.tile(np.asarray(orient,float),(n,1)),body_pose=np.zeros((n,63)),
                transl=np.tile(np.asarray(transl,float),(n,1)),betas=np.zeros((n,10)))

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
