import sys, unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap import soma_convert as sc

# SMPL-X neutral, zero betas: J_regressor[:22] @ v_template, and the lowest v_template height.
REST = np.array([
    [0.0031, -0.3514, 0.0120], [0.0613, -0.4442, -0.0140], [-0.0601, -0.4553, -0.0092], [0.0004, -0.2415, -0.0156],
    [0.1160, -0.8229, -0.0234], [-0.1044, -0.8177, -0.0260], [0.0098, -0.1097, -0.0215], [0.0726, -1.2260, -0.0552],
    [-0.0889, -1.2284, -0.0462], [-0.0015, -0.0574, 0.0069], [0.1198, -1.2840, 0.0630], [-0.1277, -1.2868, 0.0728],
    [-0.0137, 0.1077, -0.0247], [0.0448, 0.0275, -0.0003], [-0.0492, 0.0269, -0.0065], [0.0111, 0.2682, -0.0040],
    [0.1641, 0.0852, -0.0158], [-0.1518, 0.0804, -0.0191], [0.4182, 0.0131, -0.0582], [-0.4229, 0.0439, -0.0456],
    [0.6702, 0.0363, -0.0607], [-0.6722, 0.0394, -0.0609]])
SOLE = -1.3018
F = 40
_rng = np.random.default_rng(0)
ORIENT, POSE, PELVIS = _rng.normal(0, .5, (F, 3)), _rng.normal(0, .5, (F, 63)), _rng.uniform(-1, 1, (F, 3))


def angle(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1)))


class SomaConvertTest(unittest.TestCase):
    def setUp(self):
        self.f = sc.fit(REST, SOLE); self.G, self.J = sc.smplx_fk(REST, ORIENT, POSE, PELVIS)

    def soma(self, floor=0.07):
        local, hips = sc.to_soma(self.G, PELVIS, self.f, floor); return (local, hips) + sc.soma_fk(local, hips)

    def test_neutral_is_a_t_pose(self):
        i = sc.SOMA30.index; N = sc.SOMA_NEUTRAL
        np.testing.assert_allclose(N[i('Hips')], 0, atol=1e-6)
        np.testing.assert_allclose(N[i('LeftHand')], [0.724, 0.429, -0.013], atol=2e-3)
        np.testing.assert_allclose(N[i('RightFoot')], [-0.100, -0.938, -0.017], atol=2e-3)

    def test_round_trip_is_exact(self):
        local, hips, K, S = self.soma(); o2, p2, pel2 = sc.to_smplx(K, hips, self.f, 0.07)
        G2, J2 = sc.smplx_fk(REST, o2, p2, pel2)
        self.assertLess(np.abs(J2 - self.J).max(), 1e-9)
        self.assertLess(np.abs(pel2 - PELVIS).max(), 1e-9)

    def test_limbs_point_the_same_way(self):
        S = self.soma()[3]; i = sc.SOMA30.index
        for sj, sc_, kj, kc in sc.LIMBS:
            self.assertLess(angle(self.J[:, sc_] - self.J[:, sj], S[:, i(kc)] - S[:, i(kj)]).max(), 1e-6, (kj, kc))

    def test_hip_midpoints_coincide(self):
        S = self.soma(0.07)[3]; mid = S[:, [22, 26]].mean(1); want = self.J[:, [1, 2]].mean(1) * self.f['scale']
        want[:, 1] = (self.J[:, [1, 2], 1].mean(1) - 0.07) * self.f['scale'] + self.f['lift']
        np.testing.assert_allclose(mid, want, atol=1e-9)

    def test_scale_is_hip_above_ankle(self):
        self.assertTrue(1.05 < self.f['scale'] < 1.15, self.f['scale'])

    def test_floor_follows_a_per_frame_line(self):
        d = np.linspace(0, .5, F); J2 = PELVIS + d[:, None] * [0, 1, 0]
        a = sc.to_soma(self.G, PELVIS, self.f, 0.07 + 0 * d); b = sc.to_soma(self.G, J2, self.f, 0.07 + d)
        np.testing.assert_allclose(a[0], b[0]); np.testing.assert_allclose(a[1], b[1], atol=1e-12)

    def test_standing_foot_sits_where_kimodo_expects(self):
        pel = np.array([[0.2, REST[0, 1] - SOLE, -0.3]]); G, J = sc.smplx_fk(REST, np.zeros((1, 3)), np.zeros((1, 63)), pel)
        S = sc.soma_fk(*sc.to_soma(G, pel, self.f, 0.0))[1]; i = sc.SOMA30.index
        for n in ('LeftFoot', 'RightFoot'): self.assertAlmostEqual(S[0, i(n), 1], 0.066, delta=.015)
        for n in ('LeftToeBase', 'RightToeBase'): self.assertAlmostEqual(S[0, i(n), 1], 0.017, delta=.015)

    def test_neck2_is_halfway(self):
        K = self.soma()[2]; a, b = K[:, 4], K[:, 6]; mid = K[:, 5]
        d1 = Rot.from_matrix(np.swapaxes(a, 1, 2) @ mid).magnitude(); d2 = Rot.from_matrix(np.swapaxes(mid, 1, 2) @ b).magnitude()
        d = Rot.from_matrix(np.swapaxes(a, 1, 2) @ b).magnitude()
        np.testing.assert_allclose(d1, d / 2, atol=1e-9); np.testing.assert_allclose(d2, d / 2, atol=1e-9)

    def test_extra_joints_are_looked_up_by_name(self):
        _, hips, K, _ = self.soma(); base = sc.to_smplx(K, hips, self.f, 0.07)
        names = ['Extra%d' % n for n in range(47)] + sc.SOMA30; order = np.random.default_rng(1).permutation(77)
        K77 = np.empty((F, 77, 3, 3)); K77[:] = np.eye(3); K77[:, 47:] = K; names = [names[k] for k in order]; K77 = K77[:, order]
        for a, b in zip(base, sc.to_smplx(K77, hips, self.f, 0.07, names)): np.testing.assert_allclose(a, b, atol=1e-12)

    def test_floor_under(self):
        J = np.zeros((100, 22, 3)); J[:, 10, 1] = 0.5; J[:, 11, 1] = np.where(np.arange(100) < 90, 0.12, 0.30)
        self.assertAlmostEqual(sc.floor_under(J, self.f['toe']), 0.12 - self.f['toe'], places=9)


if __name__ == '__main__':
    unittest.main()
