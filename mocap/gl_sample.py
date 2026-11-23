"""Synthetic viewport geometry for an on-screen OpenGL check without models.

Used only by the hidden ``app.py --smoke-gl-sample`` option. Nothing here is
derived from SMPL-X or any registered model or character. It drives the real
GPU paths: the SMPL-X-layout skin and normals kernels, the export bone overlay
and a skinned, textured character, so a platform port can be checked visually.
"""
import numpy as np

# A 22-joint stick figure in the SMPL-X joint order (metres, Y up, +X left).
JOINTS = np.array([
    [0, .95, 0], [.09, .90, 0], [-.09, .90, 0], [0, 1.05, 0], [.10, .50, 0], [-.10, .50, 0],
    [0, 1.18, 0], [.10, .08, 0], [-.10, .08, 0], [0, 1.30, 0], [.10, .02, .10], [-.10, .02, .10],
    [0, 1.50, 0], [.07, 1.42, 0], [-.07, 1.42, 0], [0, 1.62, 0], [.18, 1.42, 0], [-.18, 1.42, 0],
    [.45, 1.42, 0], [-.45, 1.42, 0], [.70, 1.42, 0], [-.70, 1.42, 0]], dtype=float)
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def grid_faces(rows, cols, wrap):
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols if wrap else cols - 1), indexing="ij")
    a = r * cols + c
    b = r * cols + (c + 1) % cols
    return np.concatenate((np.stack((a, b, b + cols), -1), np.stack((a, b + cols, a + cols), -1))).reshape(-1, 3).astype(np.uint32)


def body_mesh():
    """Ellipsoid body in the packed SMPL-X LBS layout, bent at the spine."""
    from .mesh_data import adjacency
    rows, cols = 64, 64
    theta = np.linspace(.04, np.pi - .04, rows)[:, None]
    phi = np.linspace(0, 2 * np.pi, cols, endpoint=False)[None]
    points = np.stack((.22 * np.sin(theta) * np.cos(phi), .80 * np.cos(theta) + .85 * np.ones_like(phi),
                       .14 * np.sin(theta) * np.sin(phi)), -1).reshape(-1, 3)
    faces = grid_faces(rows, cols, True)
    count = len(points)
    directions = np.zeros((506, count, 3))
    # One pose blendshape (feature 880) bulges the belly forward.
    directions[20, :, 2] = .08 * np.exp(-((points[:, 1] - .9) / .15) ** 2) * (points[:, 2] > 0)
    weights = np.zeros((count, 55))
    upper = np.clip((points[:, 1] - 1.0) / .25, 0, 1)
    weights[:, 0] = 1 - upper
    weights[:, 3] = upper
    packed = np.concatenate((points.ravel(), directions.ravel(), weights.ravel())).astype(np.float32)
    offsets, neighbors = adjacency(faces.astype(np.int64), count)
    asset = dict(key=("gl-sample-body",), packed=packed, faces=faces, offsets=offsets, neighbors=neighbors,
                 vertex_count=np.array(count))
    transforms = np.tile(np.eye(4), (55, 1, 1))
    angle = np.radians(25)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    pivot = np.array([0, 1.0, 0])
    transforms[3, :3, :3] = rotation
    transforms[3, :3, 3] = pivot - rotation @ pivot
    frame = np.zeros(1389, np.float32)
    frame[:880] = transforms.reshape(-1)
    frame[880] = 1
    return asset, frame[None]


def bone_overlay():
    """52-bone export skeleton: the stick figure plus three-joint fingers."""
    from .bone_overlay import geometry
    rest = [*JOINTS];parents = [*PARENTS];ends = [np.zeros(3)] * 22
    for wrist, sign in ((20, 1), (21, -1)):
        for finger in range(5):
            parent = wrist
            for joint in range(3):
                rest.append(JOINTS[wrist] + [sign * (.05 + .025 * joint), 0, .04 - .02 * finger])
                parents.append(parent);ends.append(np.array([sign * .02, 0, 0]))
                parent = len(rest) - 1
    rest = np.asarray(rest);ends = np.asarray(ends)
    for leaf, tip in ((15, (0, .12, 0)), (10, (0, 0, .06)), (11, (0, 0, .06))):
        ends[leaf] = tip
    asset = dict(key="gl-sample-rig", bones=geometry(rest, np.asarray(parents), ends, ("gl-sample",)))
    frames = np.tile(np.eye(4), (1, 52, 1, 1))
    frames[0, :, :3, 3] = rest
    return asset, frames.astype(np.float32)


def character():
    """A checker-textured cylinder skinned to two bones, beside the actor."""
    from .mesh_data import adjacency
    from .name_overlay import character_samples
    rows, cols = 24, 32
    height = np.linspace(0, 1.6, rows)[:, None]
    phi = np.linspace(0, 2 * np.pi, cols, endpoint=False)[None]
    points = np.stack((.25 * np.cos(phi) + 0 * height, height + 0 * phi, .25 * np.sin(phi) + 0 * height), -1).reshape(-1, 3)
    faces = grid_faces(rows, cols, True)
    count = len(points)
    upper = np.clip(points[:, 1] / 1.6, 0, 1)
    rows_ = np.column_stack((np.arange(count) * 2, np.full(count, 2)))
    weights = np.column_stack((np.tile([0, 1], count), np.column_stack((1 - upper, upper)).ravel()))
    offsets, neighbors = adjacency(faces.astype(np.int64), count)
    u = (np.arange(count) % cols) / cols;v = (np.arange(count) // cols) / (rows - 1)
    corners = np.column_stack((faces.ravel(), u[faces.ravel()] * 4, v[faces.ravel()] * 4)).astype(np.float32)
    checker = ((np.arange(64)[:, None] // 8 + np.arange(64)[None] // 8) % 2).astype(bool)
    image = np.full((64, 64, 4), 255, np.uint8)
    image[checker, :3] = (230, 120, 40)
    asset = dict(key="gl-sample-character", vertices=points.astype(np.float32), rows=rows_.astype(np.float32),
                 weights=weights.astype(np.float32), faces=faces, offsets=offsets, neighbors=neighbors,
                 colors=np.tile([.8, .5, .3], (count, 1)).astype(np.float32),
                 textures=dict(corners=corners, images={"image0": image},
                               batches=[dict(start=0, count=len(corners), image="image0", color=[1, 1, 1, 1])]))
    asset["name_samples"] = character_samples(asset)
    matrices = np.tile(np.eye(4), (2, 1, 1))
    matrices[:, 0, 3] = -1.1
    angle = np.radians(-20)
    matrices[1, :2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    matrices[1, :3, 3] += np.array([0, .8, 0]) - matrices[1, :3, :3] @ [0, .8, 0]
    return asset, matrices.astype(np.float32)


class SampleCharacters:
    """Minimal stand-in for the character workspace used by MotionView."""

    def __init__(self):
        asset, self.matrices = character()
        self.data = dict(asset=asset, palette=2)
        self.characters = [dict(name="GL sample character", visible=True)]

    def character_draws(self):
        yield self.characters[0], self.data, self.matrices, 0, None


def install(view):
    """Show the synthetic actor, bone overlay and character in a MotionView."""
    from .preview_data import motion_summary
    joints = np.concatenate((JOINTS, np.repeat(JOINTS[15:16], 33, axis=0)))[None]
    mesh_asset, mesh_frames = body_mesh()
    view.set_motion(joints, motion_summary(joints), None, None, mesh_frames, mesh_asset)
    bones, frames = bone_overlay()
    view.target_asset = bones
    view.target_frames = frames
    view.show_bones = True
    view.actor_name = "GL sample"
    view.character_workspace = SampleCharacters()
    view.update()
