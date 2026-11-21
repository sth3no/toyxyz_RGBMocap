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

BLEND_RATE, BLEND_MIN, BLEND_MAX = 6.0, 5, 30  # degrees per frame; frames
MIRROR_BODY = ((0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16), (17, 18), (19, 20))
HANDS = ('left_hand_pose', 'right_hand_pose')
FLIP = np.array([1., -1., -1.])

def _wrap(a): return np.pi - np.mod(np.pi - a, 2 * np.pi)  # (-pi, pi]

def _ry(a):
    a = np.atleast_1d(np.asarray(a, np.float64)); z = np.zeros_like(a)
    return R.from_rotvec(np.stack([z, a, z], -1))

def yaw(orient):
    o = np.asarray(orient, np.float64); q = R.from_rotvec(o.reshape(-1, 3)).as_quat()
    return _wrap(2 * np.arctan2(q[:, 1], q[:, 3])).reshape(o.shape[:-1])  # swing-twist about +Y

def mirror_params(params, row=0):
    out = {k: np.array(v, copy=True) for k, v in params.items() if k not in HANDS}
    rot = R.from_rotvec(np.asarray(params['global_orient'], np.float64).reshape(-1, 3)).as_matrix()
    f = rot[row] @ [0., 0., 1.]; f[1] = 0; k = np.linalg.norm(f)
    f = f / k if k > 1e-12 else np.array([0., 0., 1.])
    nrm = np.array([f[2], 0., -f[0]]); S = np.eye(3) - 2 * np.outer(nrm, nrm)
    out['global_orient'] = R.from_matrix(S @ rot * [-1., 1., 1.]).as_rotvec()  # S R diag(-1,1,1) is proper
    p = np.asarray(params['transl'], np.float64).reshape(-1, 3); c = p[row]
    out['transl'] = c + (p - c) @ S
    if 'body_pose' in params:
        b = np.asarray(params['body_pose'], np.float64).reshape(len(p), -1, 3) * FLIP; m = b.copy()
        for i, j in MIRROR_BODY: m[:, i], m[:, j] = b[:, j], b[:, i]
        out['body_pose'] = m.reshape(len(p), -1)
    for i, k in enumerate(HANDS):
        if k in params:
            h = np.asarray(params[k], np.float64); out[HANDS[1 - i]] = (h.reshape(len(h), -1, 3) * FLIP).reshape(h.shape)
    return out

def mirror_logits(logits, pairs=((0, 2), (1, 3), (4, 5))):
    # columns: L ankle, L foot, R ankle, R foot, L wrist, R wrist
    x = np.asarray(logits); out = np.array(x, copy=True)
    for a, b in pairs: out[..., [a, b]] = x[..., [b, a]]
    return out

def _blend(diffs, n):
    """Frames over which a boundary difference fades out: more for a larger difference, at most half the range."""
    deg = max([np.degrees(np.linalg.norm(d, axis=-1)).max() for d in diffs] + [0.])
    return min(n // 2, int(np.clip(np.ceil(deg / BLEND_RATE - 1e-9), BLEND_MIN, BLEND_MAX)))

def copy_fill(before, after, donor, keys, height_scale=1.0, stats=None):
    """Gap rows from the donor's n+2 rows (rows 0 and n+1 face the boundary frames); transl is the pelvis position."""
    f = lambda d, k: np.asarray(d[k], np.float64)
    dp = f(donor, 'transl').reshape(-1, 3); n = len(dp) - 2
    if n < 1: raise ValueError('Copy needs at least one hidden frame')
    s = np.arange(n + 2) / (n + 1); dy = yaw(f(donor, 'global_orient').reshape(-1, 3))
    y0 = yaw(f(before, 'global_orient').reshape(-1, 3)[-1]) - dy[0]
    y1 = yaw(f(after, 'global_orient').reshape(-1, 3)[0]) - dy[-1]
    turn = _ry(y0 + s * _wrap(y1 - y0))  # rows 0..n+1; row 0 turns by y0, row n+1 by y1
    p0, p1 = f(before, 'transl').reshape(-1, 3)[-1], f(after, 'transl').reshape(-1, 3)[0]
    acc = p0 + np.cumsum(turn[1:].apply(np.diff(dp, axis=0)), 0)
    res = acc[-1] - p1; out = {'transl': acc[:n] - s[1:n + 1, None] * res}
    # Heights are measured from each world's floor; the boundary differences fade out with the pose blends below.
    h = height_scale * dp[:, 1]; out['transl'][:, 1] = h[1:-1]
    rows, diffs = {}, {}
    for k in keys:
        if k not in donor: continue
        b, a = np.atleast_2d(f(before, k))[-1].reshape(-1, 3), np.atleast_2d(f(after, k))[0].reshape(-1, 3)
        r = R.from_rotvec(f(donor, k).reshape(-1, 3))
        if k == 'global_orient': r = turn * r
        v = r.as_rotvec().reshape(n + 2, -1, 3); rows[k] = v
        # Local rotations compose on the right: donor row · D equals the boundary pose.
        diffs[k] = [(R.from_rotvec(v[i]).inv() * R.from_rotvec(x)).as_rotvec() for i, x in ((0, b), (-1, a))]
    B = [_blend([d[side] for d in diffs.values()], n) for side in (0, 1)]
    u = lambda j, side: 1 - _smoothstep(j / (B[side] + 1))
    for j in range(1, B[0] + 1): out['transl'][j - 1, 1] += u(j, 0) * (p0[1] - h[0])
    for j in range(1, B[1] + 1): out['transl'][n - j, 1] += u(j, 1) * (p1[1] - h[-1])
    for k in keys:
        if k not in donor:
            b, a = np.atleast_2d(f(before, k)), np.atleast_2d(f(after, k))
            out[k] = _slerp(b[-1], a[0], s[1:n + 1]).reshape(n, -1); continue
        v = rows[k]; o = v[1:-1].copy(); D0, D1 = diffs[k]
        for j in range(1, B[0] + 1): o[j - 1] = (R.from_rotvec(v[j]) * R.from_rotvec(D0 * u(j, 0))).as_rotvec()
        for j in range(1, B[1] + 1): o[n - j] = (R.from_rotvec(v[n + 1 - j]) * R.from_rotvec(D1 * u(j, 1))).as_rotvec()
        out[k] = o.reshape(n, -1)
    if stats is not None: stats.update(residual=float(np.linalg.norm(res[[0, 2]])), blend=tuple(B))
    return out

def _smoothstep(x): return 3 * x * x - 2 * x ** 3
