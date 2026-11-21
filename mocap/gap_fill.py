import numpy as np
from scipy.spatial.transform import Rotation as R

CAPTURED, INTERPOLATED, COPIED, GENERATED = 0, 1, 2, 3
POSE_KEYS = ('global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose')

def rigid_params(params, transform, pelvis_from, pelvis_to):
    # SMPL rotates about the shape-dependent rest pelvis, so transl needs the pelvis offsets of both takes
    T = np.asarray(transform, np.float64); M, t = T[:3, :3], T[:3, 3]
    out = {k: np.array(v, copy=True) for k, v in params.items()}
    go = np.asarray(params['global_orient'], np.float64).reshape(-1, 3)
    out['global_orient'] = (R.from_matrix(M) * R.from_rotvec(go)).as_rotvec()
    tr = np.asarray(params['transl'], np.float64).reshape(-1, 3)
    out['transl'] = (tr + np.asarray(pelvis_from, np.float64)) @ M.T + t - np.asarray(pelvis_to, np.float64)
    return out

def _slerp(a, b, ts):
    a = np.asarray(a, np.float64).reshape(-1, 3); b = np.asarray(b, np.float64).reshape(-1, 3)
    r0 = R.from_rotvec(a); rel = (r0.inv() * R.from_rotvec(b)).as_rotvec()  # rotvec angle is <= pi: short way
    return np.stack([(r0 * R.from_rotvec(rel * t)).as_rotvec() for t in ts])  # (frames, k, 3)

def _hermite(before, after, frames):
    p0, p1 = before[-1], after[0]; n = frames + 1
    lim = 3 * np.linalg.norm(p1 - p0)
    def tangent(v):
        m = v * n; k = np.linalg.norm(m)
        return m * (lim / k) if k > lim else m
    m0, m1 = tangent(before[-1] - before[-2]), tangent(after[1] - after[0])
    t = (np.arange(1, frames + 1) / n)[:, None]
    return ((2*t**3 - 3*t**2 + 1) * p0 + (t**3 - 2*t**2 + t) * m0 + (-2*t**3 + 3*t**2) * p1 + (t**3 - t**2) * m1)

def smooth_fill(before, after, frames):
    ts = np.arange(1, frames + 1) / (frames + 1); out = {}
    for k in POSE_KEYS:
        if k in before and k in after:
            b = np.asarray(before[k], np.float64); a = np.asarray(after[k], np.float64)
            out[k] = _slerp(b[-1], a[0], ts).reshape(frames, -1)
    out['transl'] = _hermite(np.asarray(before['transl'], np.float64).reshape(-1, 3),
                             np.asarray(after['transl'], np.float64).reshape(-1, 3), frames)
    return out

def fill_logits(before_last, after_first, frames):
    a = np.asarray(before_last, np.float64); b = np.asarray(after_first, np.float64)
    t = (np.arange(1, frames + 1) / (frames + 1))[:, None]
    return a + (b - a) * t
