"""SMPL-X body pose <-> Kimodo SOMA (30 joints). numpy/scipy only, float64.

Conventions: Y up, +Z forward, left = +X, metres, both rest poses are T-poses in these axes. `pelvis` always means the
SMPL-X pelvis joint position (not `transl`); SOMA `hips` is the SOMA Hips joint position. A joint's global rotation
is carried over (times a fixed offset on the ten limb bones), SOMA has one more neck joint which gets the rotation
halfway between neck and head, positions are scaled by the leg-length ratio because Kimodo has one fixed body.
`floor` is the height of the floor in SMPL-X coordinates (scalar, or one value per frame for a sloping line).
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

SMPLX_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
SOMA30 = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Jaw', 'LeftEye', 'RightEye',
          'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumbEnd', 'LeftHandMiddleEnd',
          'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumbEnd', 'RightHandMiddleEnd',
          'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase']
SOMA30_PARENTS = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28])
# Kimodo's somaskel30 rest joints (joints.p), Hips at the origin.
SOMA_NEUTRAL = np.array([
    [0.000000, 0.000000, 0.000000], [-0.000137, 0.050038, -0.000537], [-0.000137, 0.121291, -0.000836],
    [-0.000137, 0.196791, -0.008995], [-0.001954, 0.459904, -0.014529], [-0.001954, 0.536998, 0.008497],
    [-0.001954, 0.598287, 0.028034], [-0.001928, 0.603043, 0.058984], [0.030110, 0.652089, 0.103903],
    [-0.034179, 0.651906, 0.103617], [0.016079, 0.429163, 0.042139], [0.165278, 0.429163, -0.012884],
    [0.452671, 0.429163, -0.012910], [0.723611, 0.429163, -0.012884], [0.846297, 0.396961, 0.035447],
    [0.913730, 0.426034, -0.013224], [-0.013938, 0.428594, 0.043146], [-0.164310, 0.428594, -0.012310],
    [-0.451677, 0.428594, -0.012336], [-0.723013, 0.428594, -0.012310], [-0.845655, 0.396480, 0.035731],
    [-0.913019, 0.425528, -0.012625], [0.100432, -0.084345, 0.025957], [0.100432, -0.516563, 0.017927],
    [0.100432, -0.938114, -0.016888], [0.100432, -0.988708, 0.115427], [-0.100473, -0.082953, 0.026203],
    [-0.100473, -0.516575, 0.018148], [-0.100473, -0.937749, -0.016636], [-0.100473, -0.988545, 0.116206]])
# Height of the SOMA ankle and toe joints above the floor while standing, read from Kimodo's own motions.
SOMA_STANDING = (0.066, 0.017)
# SOMA joint -> SMPL-X joint whose global rotation it takes. Neck2 is handled separately.
FROM_SMPLX = dict(Hips=0, Spine1=3, Spine2=6, Chest=9, Neck1=12, Head=15, Jaw=15, LeftEye=15, RightEye=15,
                  LeftShoulder=13, LeftArm=16, LeftForeArm=18, LeftHand=20, LeftHandThumbEnd=20, LeftHandMiddleEnd=20,
                  RightShoulder=14, RightArm=17, RightForeArm=19, RightHand=21, RightHandThumbEnd=21, RightHandMiddleEnd=21,
                  LeftLeg=1, LeftShin=4, LeftFoot=7, LeftToeBase=10, RightLeg=2, RightShin=5, RightFoot=8, RightToeBase=11)
# SMPL-X joint -> SOMA joint whose global rotation it takes on the way back.
TO_SMPLX = {0: 'Hips', 3: 'Spine1', 6: 'Spine2', 9: 'Chest', 12: 'Neck1', 15: 'Head', 13: 'LeftShoulder', 16: 'LeftArm',
            18: 'LeftForeArm', 20: 'LeftHand', 14: 'RightShoulder', 17: 'RightArm', 19: 'RightForeArm', 21: 'RightHand',
            1: 'LeftLeg', 4: 'LeftShin', 7: 'LeftFoot', 10: 'LeftToeBase', 2: 'RightLeg', 5: 'RightShin', 8: 'RightFoot', 11: 'RightToeBase'}
# Limb bones with an offset: (SMPL-X joint, SMPL-X child, SOMA joint, SOMA child).
LIMBS = [(1, 4, 'LeftLeg', 'LeftShin'), (2, 5, 'RightLeg', 'RightShin'), (4, 7, 'LeftShin', 'LeftFoot'), (5, 8, 'RightShin', 'RightFoot'),
         (7, 10, 'LeftFoot', 'LeftToeBase'), (8, 11, 'RightFoot', 'RightToeBase'), (16, 18, 'LeftArm', 'LeftForeArm'),
         (17, 19, 'RightArm', 'RightForeArm'), (18, 20, 'LeftForeArm', 'LeftHand'), (19, 21, 'RightForeArm', 'RightHand')]
_T = lambda m: np.swapaxes(m, -1, -2)


def _align(a, b):
    """Smallest rotation (3,3) taking direction a to direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b); axis = np.cross(a, b); s = np.linalg.norm(axis)
    return np.eye(3) if s < 1e-9 else Rot.from_rotvec(axis / s * np.arctan2(s, a @ b)).as_matrix()


def fit(rest, sole):
    """How this body maps to SOMA: dict(offsets=(30,3,3), scale, anchor=(3,), lift, toe).

    rest (22,3) are the SMPL-X rest joints, sole the lowest rest surface height. scale is SOMA size over SMPL-X size by the
    hip joints' height above the ankles; anchor is where SOMA Hips sits in the pelvis frame (hip joint midpoints coincide);
    lift is added to SOMA heights so a standing foot sits where Kimodo expects it; toe is the rest height of joints 10, 11
    above the sole.
    """
    rest = np.asarray(rest, np.float64); N = SOMA_NEUTRAL; A = np.tile(np.eye(3), (30, 1, 1))
    for sj, sc, kj, kc in LIMBS:
        k, c = SOMA30.index(kj), SOMA30.index(kc); A[k] = _align(N[c] - N[k], rest[sc] - rest[sj])
    scale = float((N[[22, 26], 1].mean() - N[[24, 28], 1].mean()) / (rest[[1, 2], 1].mean() - rest[[7, 8], 1].mean()))
    anchor = (rest[[1, 2]].mean(0) - rest[0]) - N[[22, 26]].mean(0) / scale
    ankle, toe = rest[[7, 8], 1].mean() - sole, rest[[10, 11], 1].mean() - sole
    return dict(offsets=A, scale=scale, anchor=anchor, lift=float(np.mean(SOMA_STANDING) - scale * (ankle + toe) / 2), toe=float(toe))


def smplx_fk(rest, orient, pose, pelvis):
    """Global rotations G (F,22,3,3) and joint positions J (F,22,3) from SMPL-X axis-angle orient (F,3), pose (F,63), pelvis (F,3)."""
    rest = np.asarray(rest, np.float64)
    aa = np.concatenate([np.asarray(orient, np.float64)[:, None], np.asarray(pose, np.float64).reshape(-1, 21, 3)], 1)
    local = Rot.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    G = np.empty_like(local); J = np.empty((len(local), 22, 3)); G[:, 0] = local[:, 0]; J[:, 0] = np.asarray(pelvis, np.float64)
    for j in range(1, 22):
        p = SMPLX_PARENTS[j]; G[:, j] = G[:, p] @ local[:, j]; J[:, j] = J[:, p] + G[:, p] @ (rest[j] - rest[p])
    return G, J


def to_soma(G, pelvis, fit, floor):
    """SOMA local rotations (F,30,3,3) and Hips positions (F,3) from SMPL-X global rotations and pelvis positions.

    Height is (y - floor) * scale + lift; floor is a scalar or (F,).
    """
    G = np.asarray(G, np.float64); K = np.empty((len(G), 30, 3, 3)); A = fit['offsets']
    for k, name in enumerate(SOMA30):
        if name != 'Neck2': K[:, k] = G[:, FROM_SMPLX[name]] @ A[k]
    K[:, 5] = G[:, 12] @ Rot.from_rotvec(Rot.from_matrix(_T(G[:, 12]) @ G[:, 15]).as_rotvec() * .5).as_matrix()
    local = K.copy()
    for k in range(1, 30): local[:, k] = _T(K[:, SOMA30_PARENTS[k]]) @ K[:, k]
    at = np.asarray(pelvis, np.float64) + G[:, 0] @ fit['anchor']
    hips = at * fit['scale']; hips[:, 1] = (at[:, 1] - floor) * fit['scale'] + fit['lift']
    return local, hips


def soma_fk(local, hips):
    """Global rotations K (F,30,3,3) and joint positions S (F,30,3) from SOMA local rotations and Hips positions."""
    K = np.empty_like(local); S = np.empty((len(local), 30, 3)); K[:, 0] = local[:, 0]; S[:, 0] = hips
    for j in range(1, 30):
        p = SOMA30_PARENTS[j]; K[:, j] = K[:, p] @ local[:, j]; S[:, j] = S[:, p] + K[:, p] @ (SOMA_NEUTRAL[j] - SOMA_NEUTRAL[p])
    return K, S


def to_smplx(K, hips, fit, floor, names=SOMA30):
    """SMPL-X orient (F,3), pose (F,63) and pelvis positions (F,3) from SOMA global rotations and Hips positions.

    The inverse of `to_soma`. K may hold more joints than 30 in any order; `names` gives the name of each of its joints.
    """
    K = np.asarray(K, np.float64); hips = np.asarray(hips, np.float64); F = len(K); G = np.empty((F, 22, 3, 3)); names = list(names)
    for j, name in TO_SMPLX.items(): G[:, j] = K[:, names.index(name)] @ fit['offsets'][SOMA30.index(name)].T
    local = G.copy()
    for j in range(1, 22): local[:, j] = _T(G[:, SMPLX_PARENTS[j]]) @ G[:, j]
    aa = Rot.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(F, 22, 3)
    at = hips / fit['scale']; at[:, 1] = (hips[:, 1] - fit['lift']) / fit['scale'] + floor
    return aa[:, 0], aa[:, 1:].reshape(F, 63), at - G[:, 0] @ fit['anchor']


def floor_under(J, toe):
    """Floor height for joints J (F,22,3): 10th percentile over frames of the lower of joints 10 and 11, minus toe."""
    return float(np.percentile(np.minimum(J[:, 10, 1], J[:, 11, 1]), 10) - toe)
