"""Two takes of one person that share frames (parts of one long range): the later part is moved into the earlier part's
world by the camera pairings of the shared frames, and the shared frames fade from one estimate to the other. numpy only."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from .gap_fill import POSE_KEYS, _smoothstep


def weights(n):
    """Weight of the later part on each of `n` shared frames: smoothstep from near 0 to near 1, symmetric."""
    return _smoothstep(np.arange(1, n+1)/(n+1))


def align(fulls, pivots):
    """One heading-and-floor transform for all shared frames, as take_join.upright gives for one frame.
    `fulls` are the per-frame transforms from the later world to the earlier one (inv(earlier pairing) @ later pairing),
    `pivots` the later part's pelvis positions in its own world. The heading is the circular mean of each frame's twist
    about +Y; the horizontal offset is the least-squares fit of the pivots; the height is left to each part."""
    fulls = np.asarray(fulls, np.float64);pivots = np.asarray(pivots, np.float64).reshape(-1, 3)
    q = R.from_matrix(fulls[:, :3, :3]).as_quat();twist = 2*np.arctan2(q[:, 1], q[:, 3])
    heading = np.arctan2(np.sin(twist).mean(), np.cos(twist).mean())
    T = np.eye(4);T[:3, :3] = R.from_rotvec([0., heading, 0.]).as_matrix()
    moved = np.einsum('nij,nj->ni', fulls[:, :3, :3], pivots)+fulls[:, :3, 3]
    T[[0, 2], 3] = (moved-pivots@T[:3, :3].T).mean(0)[[0, 2]]
    return T


def blend(a, b, w):
    """Rows of the shared frames: per joint slerp for rotations, linear for everything else. `a` and `b` hold the same keys."""
    out = {}
    for k, x in a.items():
        x, y = np.asarray(x, np.float64), np.asarray(b[k], np.float64)
        if k in POSE_KEYS:
            n = len(x);r0 = R.from_rotvec(x.reshape(-1, 3));step = (r0.inv()*R.from_rotvec(y.reshape(-1, 3))).as_rotvec()
            out[k] = (r0*R.from_rotvec(step*np.repeat(w, x.reshape(n, -1).shape[1]//3)[:, None])).as_rotvec().reshape(x.shape)
        else:out[k] = x+(y-x)*np.reshape(w, (-1,)+(1,)*(x.ndim-1))
    return out
