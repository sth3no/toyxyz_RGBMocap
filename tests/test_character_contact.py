"""Character self-contacts: SMPL-X joint layout, finger mapping, offset scaling and the character skin's LBS."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap.retarget import character_contact as cc  # noqa: E402
from mocap.self_contact import matrix_quat  # noqa: E402


class Layout(unittest.TestCase):
    def test_smplx_parents(self):
        p = cc.SMPLX_PARENTS
        self.assertEqual(len(p), 55)
        self.assertEqual(list(p[22:25]), [15, 15, 15])
        # Left index chain 25-27 hangs off the left wrist, right thumb 52-54 off the right wrist.
        self.assertEqual(list(p[25:28]), [20, 25, 26])
        self.assertEqual(list(p[52:55]), [21, 52, 53])

    def test_finger_joints_semantic_and_mixamo(self):
        semantic = {'leftIndexProximal': 'L_index_proximal', 'rightThumbDistal': 'R_thumb_distal', 'leftLittleIntermediate': 'L_little_2'}
        bones = set(semantic.values())
        self.assertEqual(cc.finger_joints('humanoid', {'semantic': semantic}, bones), {25: 'L_index_proximal', 54: 'R_thumb_distal', 32: 'L_little_2'})
        self.assertEqual(cc.finger_joints('mixamo', {}, {'mixamorig:LeftHandPinky2', 'mixamorig:RightHandIndex1'}),
                         {32: 'mixamorig:LeftHandPinky2', 40: 'mixamorig:RightHandIndex1'})

    def test_scaled_tracks(self):
        tracks = [dict(id='a', offset=[0.1, -0.2, 0.3], distance=0.16), dict(id='b', offset='auto', distance='auto')]
        out = cc.scaled_tracks(tracks, 0.5)
        np.testing.assert_allclose(out[0]['offset'], [0.05, -0.1, 0.15])
        self.assertAlmostEqual(out[0]['distance'], 0.08)
        self.assertEqual((out[1]['offset'], out[1]['distance']), ('auto', 'auto'))
        self.assertEqual(tracks[0]['offset'], [0.1, -0.2, 0.3])

    def test_runs(self):
        self.assertEqual(cc.runs(np.array([0, 1, 1, 0, 1], bool)), [(1, 3), (4, 5)])


def box(lo, hi):
    """Closed, outward-facing triangle box: (vertices (8, 3), faces (12, 3))."""
    x = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    v = np.asarray(lo) + x * (np.asarray(hi) - np.asarray(lo))
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


class OffsetMapping(unittest.TestCase):
    def test_goal_keeps_its_place_on_a_differently_shaped_segment(self):
        # A head joint offset at the side of a 20 cm cube lands at the side of a wide, flat 40 x 10 x 20 cm head.
        sv, sf = box([-.1, -.1, -.1], [.1, .1, .1])
        weights = np.zeros((8, 55)); weights[:, 15] = 1
        source = dict(vertices=sv, faces=sf, weights=weights, rest=np.zeros((55, 3)))
        cv, cf = box([-.2, .95, -.1], [.2, 1.05, .1])
        rest = np.zeros((55, 3)); rest[15] = [0, 1, 0]
        skin = cc.CharacterSkin(cv, weights, rest, cf, None)
        offset, note = cc.mapped_offset([.1, 0., 0.], 15, 'Head', None, source, skin)
        np.testing.assert_allclose(offset, [.2, 0., 0.], atol=1e-9)
        self.assertEqual(note['segment'], 'head')
        self.assertAlmostEqual(note['source_skin_gap_m'], 0., places=9)

    def test_only_torso_neck_and_head_offsets_are_mapped(self):
        sv, sf = box([-.1, -.1, -.1], [.1, .1, .1])
        weights = np.zeros((8, 55)); weights[:, 15] = 1
        source = dict(vertices=sv, faces=sf, weights=weights, rest=np.zeros((55, 3)))
        cv, cf = box([-.2, .95, -.1], [.2, 1.05, .1])
        rest = np.zeros((55, 3)); rest[15] = [0, 1, 0]
        skin = cc.CharacterSkin(cv, weights, rest, cf, None)
        tracks = [dict(id='head', target='Head', offset=[.1, 0., 0.], distance='auto'),
                  dict(id='arm', target='RightForeArm', offset=[.1, 0., 0.], distance='auto')]
        out, notes = cc.character_tracks(tracks, .5, source, skin)
        np.testing.assert_allclose(out[0]['offset'], [.2, 0., 0.], atol=1e-9)
        np.testing.assert_allclose(out[1]['offset'], [.05, 0., 0.])
        self.assertEqual(list(notes), ['head'])


class Skinning(unittest.TestCase):
    def test_lbs_moves_rigid_parts_with_their_joint(self):
        rng = np.random.default_rng(0)
        rest = np.zeros((55, 3))
        rest[:22] = rng.normal(scale=.3, size=(22, 3))
        for j in range(22, 55):
            rest[j] = rest[cc.SMPLX_PARENTS[j]] + rng.normal(scale=.02, size=3)
        # One vertex per joint, fully weighted to it, at that joint's rest position plus an offset.
        offset = rng.normal(scale=.05, size=(55, 3))
        shaped = rest + offset
        weights = np.eye(55)
        faces = np.array([[0, 1, 2]])
        hands = np.repeat(np.eye(3)[None, None], 2, 0).repeat(30, 1)
        skin = cc.CharacterSkin(shaped, weights, rest, faces, hands)
        local = Rotation.random(2 * 22, random_state=1).as_matrix().reshape(2, 22, 3, 3)
        quat = matrix_quat(local)
        root = rng.normal(size=(2, 3))
        skin.groups = {'all': np.arange(55)}
        posed = skin.vertices(root, quat, np.arange(2), ['all'])['all']
        from mocap.correction import fk
        joints, world = fk(rest[:22], root, quat)
        expected = joints + np.einsum('tjab,jb->tja', world, offset[:22])
        np.testing.assert_allclose(posed[:, :22], expected, atol=1e-9)

    def test_solve_on_target_skin_without_optional_source_video_modules(self):
        from mocap import self_contact
        from mocap.self_contact_store import normalize_tracks

        body = np.array([
            [0, 1.0, 0], [.14, .99, 0], [-.14, .99, 0], [0, 1.1, 0],
            [.14, .54, 0], [-.14, .54, 0], [0, 1.25, 0], [.14, .10, 0],
            [-.14, .10, 0], [0, 1.38, 0], [.14, .06, .18], [-.14, .06, .18],
            [0, 1.53, 0], [.1, 1.44, 0], [-.1, 1.44, 0], [0, 1.67, 0],
            [.2, 1.43, 0], [-.2, 1.43, 0], [.42, 1.41, 0], [-.42, 1.41, 0],
            [.64, 1.4, 0], [-.64, 1.4, 0],
        ])
        head = body[15]
        shapes = [box(point - .035, point + .035) for point in body]
        verts = np.concatenate([shape[0] for shape in shapes])
        faces = np.concatenate([shape[1] + 8 * i for i, shape in enumerate(shapes)])
        weights = np.zeros((len(verts), 55))
        weights[np.arange(len(verts)), np.repeat(np.arange(22), 8)] = 1
        rest = np.vstack([body, np.repeat(head[None], 33, axis=0)])
        frames = 40
        hands = np.broadcast_to(np.eye(3), (frames, 30, 3, 3))
        skin = cc.CharacterSkin(verts, weights, rest, faces, hands)
        local = np.tile(np.eye(3), (frames, 22, 1, 1))
        local[:, 16] = Rotation.from_euler('z', -35, degrees=True).as_matrix()
        local[:, 18] = Rotation.from_euler('y', -30, degrees=True).as_matrix()
        quats = matrix_quat(local)
        tracks = normalize_tracks([dict(effector='LeftHand', target='Head', ranges=[[8, 32]])], frames)
        report = self_contact.apply(body, np.tile(body[0], (frames, 1)), quats,
                                    np.array([[.035, 0, 0], [-.035, 0, 0]]),
                                    30., tracks, [[], []], skin=skin)
        self.assertTrue(report['changed'])
        self.assertTrue(np.isfinite(quats).all())


if __name__ == '__main__':
    unittest.main()
