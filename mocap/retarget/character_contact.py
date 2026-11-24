"""Self-contacts solved again on a retargeted character's own skin.

The project's self-contact rows were solved on the SMPL-X body; on a character with other proportions the same pose puts
the hands inside or off its body. After the retarget bake, the rows are solved once more by the unchanged self-contact
solver on the character: its canonical 22 joints share SMPL-X's body joint order and parents, and its skin is laid out
like SMPL-X's 55 joints (body bones on joints 0-21, finger bones on SMPL-X's finger joints 25-54, every other bone on its
nearest mapped ancestor), so the solver's segments, palm and grip code apply as they are.

Hair is soft and made of open, overlapping strand shells, which the solver's closed-segment inside test cannot read (a
hand pushed out of it slid to the hair ends): hair meshes belong to no segment, so hands touch the scalp and face under it.
They are found by mesh name, or listed in the mapping file as "soft_meshes".

Differences from the SMPL-X pass: no pose correctives (plain linear blend skinning, as the character is exported), no video
terms (the video camera fits the SMPL-X body, not the character), and explicit row offsets/distances scale with the
character's leg-length ratio. Fingers keep the baked finger pose; the solver changes arm (and reciprocal leg) rotations.
"""
import json
import re
from pathlib import Path

import numpy as np

from ..correction import fk
from ..hand_lock import palm_offsets
from ..self_contact import Skin, apply, matrix_quat, quat_matrix, segment_names, segment_vertices, Segments
from .mapping import PARENTS

# SMPL-X 55-joint parents: body (22), jaw and eyes on the head, then per hand MANO order index, middle, pinky, ring, thumb.
FINGER_ORDER = ('Index', 'Middle', 'Little', 'Ring', 'Thumb')
PHALANGES = ('Proximal', 'Intermediate', 'Distal')
MIXAMO_FINGER = {'Index': 'Index', 'Middle': 'Middle', 'Little': 'Pinky', 'Ring': 'Ring', 'Thumb': 'Thumb'}
SMPLX_PARENTS = np.array(list(PARENTS) + [15, 15, 15] + [
    (20 + h if p == 0 else 25 + 15 * h + 3 * c + p - 1) for h in (0, 1) for c in range(5) for p in range(3)])
METHOD = 'character self-contact v1'
SOFT_MESH = re.compile(r'hair|kami|髪', re.IGNORECASE)


def soft_vertices(folder, kind, rig, count):
    """(vertex ids, mesh names) of the soft meshes (hair) in the preview's vertex order (scene meshes in import order)."""
    geometry = np.load(Path(folder) / f'{kind}-contact-geometry.npz')
    names = [str(n) for n in geometry['mesh_names']]; counts = np.asarray(geometry['mesh_counts'], int)
    if counts.sum() != count:raise ValueError('Character meshes changed since import. Import the character again.')
    listed = rig.get('mapping_options', {}).get('soft_meshes')
    soft = [n for n in names if (n in listed if listed is not None else SOFT_MESH.search(n))]
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ids = [np.arange(s, s + c) for name, s, c in zip(names, starts, counts) if name in soft]
    return (np.concatenate(ids) if ids else np.zeros(0, int)), soft


def finger_joints(kind, rig, bones):
    """{SMPL-X finger joint (25-54): character bone} for the fingers the rig maps."""
    out = {}
    for h, side in enumerate(('left', 'right')):
        for c, finger in enumerate(FINGER_ORDER):
            for p, phalanx in enumerate(PHALANGES):
                if kind in ('humanoid', 'vrm'):
                    name = rig.get('semantic', {}).get(side + finger + phalanx)
                elif kind == 'mixamo':
                    stem = ('Left' if h == 0 else 'Right') + 'Hand' + MIXAMO_FINGER[finger] + str(p + 1)
                    name = next((b for b in bones if b.split(':')[-1] == stem), None)
                else:
                    name = None
                if name in bones:
                    out[25 + 15 * h + 3 * c + p] = name
    return out


class CharacterSkin(Skin):
    """Skin with the character's mesh on SMPL-X's 55-joint layout; plain LBS, fingers from the baked take."""

    def __init__(self, shaped, weights, rest, faces, hands, exclude=()):
        self.shaped = np.asarray(shaped, float); self.weights = np.asarray(weights, float); self.rest = np.asarray(rest, float)
        self.parents = [int(p) for p in SMPLX_PARENTS]; self.mean = np.repeat(np.eye(3)[None], 55, 0); self.hands = hands
        self.posedirs = None
        self.groups = {k: v[~np.isin(v, exclude)] for k, v in segment_vertices(self.weights).items()}
        self.segments = Segments(faces, self.groups)

    def vertices(self, root, quat, rows, names, fingers=None):
        ids = [self.groups[name] for name in names]; every = np.concatenate(ids); m = len(root)
        rot = np.repeat(self.mean[None], m, 0)
        rot[:, :22] = quat_matrix(quat)
        if self.hands is not None:rot[:, 25:55] = self.hands[rows]
        for h, value in (fingers or {}).items():rot[:, 25 + 15 * h:40 + 15 * h] = value
        world = np.empty_like(rot); joints = np.empty((m, 55, 3)); world[:, 0] = rot[:, 0]; joints[:, 0] = root
        for j in range(1, 55):
            p = self.parents[j]; world[:, j] = world[:, p] @ rot[:, j]
            joints[:, j] = joints[:, p] + np.einsum('tij,j->ti', world[:, p], self.rest[j] - self.rest[p])
        shift = joints - np.einsum('tjab,jb->tja', world, self.rest)
        w = self.weights[every]
        blend = (w @ world.reshape(m, 55, 9)).reshape(m, -1, 3, 3)
        out = np.einsum('tvab,vb->tva', blend, self.shaped[every]) + w @ shift
        return dict(zip(names, np.split(out, np.cumsum([len(i) for i in ids])[:-1], 1)))


def load_skin(folder, kind, stance):
    """CharacterSkin of a baked take folder (preview mesh, keyed bones, skeleton) and its 22 canonical rest joints."""
    folder = Path(folder)
    rig = json.loads((folder / f'{kind}-rig.json').read_text(encoding='utf-8'))
    parent = {b['name']: b['parent'] for b in rig['bones']}
    skeleton = np.load(folder / f'{kind}-skeleton.npz')
    heads = dict(zip(skeleton['names'].tolist(), skeleton['heads']))
    keyed = json.loads((folder / 'bake.json').read_text(encoding='utf-8'))['keyed_bones']
    palette = sorted(n for n in keyed if n in parent)  # blender_io.preview palette for a baked take
    names = [str(n) for n in stance['names']]
    rest = np.empty((55, 3)); rest[:22] = stance['rest']; rest[22:25] = stance['rest'][15]
    fingers = finger_joints(kind, rig, set(parent))
    joint = {name: j for j, name in enumerate(names)}  # a virtual slot shares its bone with the later real slot
    joint.update({bone: j for j, bone in fingers.items()})
    for j in range(25, 55):
        # Unmapped finger joints sit on their parent joint (zero length) and carry no skin.
        rest[j] = heads[fingers[j]] if j in fingers else rest[SMPLX_PARENTS[j]]

    def joint_of(bone):
        while bone is not None and bone not in joint:
            bone = parent.get(bone)
        return 0 if bone is None else joint[bone]

    preview = np.load(folder / 'preview.npz')
    rows, influences = preview['rows'], preview['weights']
    to_joint = np.array([joint_of(b) for b in palette] + [0])
    weights = np.zeros((len(rows), 55))
    for v, (start, count) in enumerate(rows):
        for bone, w in influences[start:start + count]:
            weights[v, to_joint[int(bone)]] += w
    frames = np.load(folder / 'preview-frames.npy', mmap_mode='r')
    index = {b: i for i, b in enumerate(palette)}
    hands = np.repeat(np.eye(3)[None, None], len(frames), 0).repeat(30, 1)
    for j, bone in fingers.items():
        up = SMPLX_PARENTS[j]
        owner = names[up] if up < 22 else fingers.get(up)
        if bone in index and owner in index:
            R = np.asarray(frames[:, index[bone], :3, :3], float); P = np.asarray(frames[:, index[owner], :3, :3], float)
            hands[:, j - 25] = P.swapaxes(-1, -2) @ R
    soft, soft_names = soft_vertices(folder, kind, rig, len(rows))
    skin = CharacterSkin(preview['vertices'], weights, rest, preview['faces'].astype(np.int64), hands, soft)
    skin.soft_meshes = soft_names
    return skin


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def scaled_tracks(tracks, ratio):
    """Explicit SMPL-X offsets and hand distances, scaled to the character."""
    out = []
    for t in tracks:
        t = dict(t)
        if isinstance(t.get('offset'), (list, tuple)):t['offset'] = [float(x) * ratio for x in t['offset']]
        if isinstance(t.get('distance'), (int, float)) and not isinstance(t.get('distance'), bool):t['distance'] = float(t['distance']) * ratio
        out.append(t)
    return out


def mapped_offset(offset, tj, part, side, source, skin):
    """An explicit joint offset moved from the SMPL-X target segment onto the character's: the SMPL-X goal point keeps its
    position relative to the segment's rest bounding box, then lands on the character's skin at the SMPL-X skin distance
    (a scaled offset misses stylised proportions: a joint offset at a realistic head's ear lands off an anime head).
    (offset, report) or (None, reason) when the segments cannot be compared."""
    from ..self_contact import Segment, signed_distance
    name = segment_names(part, side)[0]
    src_groups = segment_vertices(source['weights'])
    src_ids, ids = src_groups.get(name), skin.groups.get(name)
    if src_ids is None or ids is None or len(src_ids) < 4 or len(ids) < 4:return None, f'no {name} segment to map onto'
    sv, cv = source['vertices'][src_ids], skin.shaped[ids]
    goal = source['rest'][tj] + np.asarray(offset, float)
    src_gap = float(signed_distance(goal[None], sv, Segment(source['faces'], src_ids), 1.)[0][0])
    lo, hi, clo, chi = sv.min(0), sv.max(0), cv.min(0), cv.max(0)
    mapped = clo + (goal - lo) / np.maximum(hi - lo, 1e-9) * (chi - clo)
    sd, surface = signed_distance(mapped[None], cv, skin.segments[name], 1.)
    q = surface[0]; out = mapped - q if sd[0] >= 0 else q - mapped
    n = out / max(np.linalg.norm(out), 1e-12)
    landed = q + n * min(max(src_gap, 0.), .02)
    return landed - skin.rest[tj], dict(segment=name, source_offset_m=list(map(float, offset)), source_skin_gap_m=src_gap,
                                        offset_m=(landed - skin.rest[tj]).tolist())


def character_tracks(tracks, ratio, source, skin):
    """Tracks for the character: explicit joint offsets on the torso, neck and head mapped onto its skin, everything else
    (limbs, hand-to-hand, distances) scaled. Returns (tracks, per-track mapping notes).

    Stylised proportions break scaling on heads and torsos (NecoBody: hands 1.1-1.6 cm off the head with scaled offsets,
    0.3 cm mapped); limbs are near cylinders that scale well, and a mapped arm-on-arm goal kept the hand 2 cm inside."""
    from ..self_contact import parse_target, FLAT_PARTS
    out, notes = [], {}
    for original, t in zip(tracks, scaled_tracks(tracks, ratio)):
        tj, side, part = parse_target(t['target'])
        if isinstance(original.get('offset'), (list, tuple)) and part in FLAT_PARTS and source is not None:
            offset, note = mapped_offset(original['offset'], tj, part, side, source, skin)
            if offset is not None:t = dict(t, offset=[float(x) for x in offset])
            notes[t.get('id')] = note if offset is not None else dict(scaled=True, reason=note)
        out.append(t)
    return out, notes


def run(folder, kind, project, result, emit=lambda *a, **k: None):
    """Solve the project's self-contact rows on the baked take in `folder`; rewrites motion-<kind>-stance.npz when anything
    changes (the pre-contact stance is kept as motion-<kind>-stance-before-contact.npz). None when the project has no rows."""
    from .. import self_contact_store as store
    from ..motion_tools import base_result
    folder = Path(folder)
    stance_path = folder / f'motion-{kind}-stance.npz'
    stance = dict(np.load(stance_path))
    fps = float(np.load(folder / 'motion-source.npz')['fps']); n = len(stance['root'])
    tracks, _, warning = store.read(Path(project), base_result(result), {'frames': n, 'fps': fps})
    if not [t for t in tracks if t.get('enabled', True) and t['ranges']]:
        return None if not warning else dict(method=METHOD, changed=[], tracks=[], warning=warning)
    if not (folder / f'{kind}-skeleton.npz').exists():
        return dict(method=METHOD, changed=[], tracks=[], warning='Import the character again to solve self-contacts on it.')
    skin = load_skin(folder, kind, stance)
    rest = np.asarray(stance['rest'], float); roots = np.asarray(stance['root'], float).copy()
    quats = matrix_quat(stance['local']).copy()
    check, _ = fk(rest, roots, quats)
    drift = float(np.abs(check - stance['points']).max())
    if drift > 1e-6:raise ValueError(f'Character stance does not match its joints ({drift:.2e} m)')
    active = np.load(folder / 'motion-source-contacts.npz')['active']
    foot_ranges = [runs(active[:, f]) for f in (0, 1)]
    offsets = palm_offsets(skin.shaped, skin.weights, skin.rest)
    ratio = float(stance['ratio'])
    neutral = folder / 'motion-source-neutral.npz'
    source = dict(np.load(neutral)) if neutral.exists() else None
    if source is not None and source['weights'].shape[1] != 55:source = None  # SMPL-H sources: offsets are only scaled
    solved, mapping = character_tracks(tracks, ratio, source, skin)
    report = apply(rest, roots, quats, offsets, fps, solved, foot_ranges, emit, skin, None)
    report = dict(report, method=METHOD, ratio=ratio, palm_offsets=offsets.tolist(), soft_meshes=skin.soft_meshes, offset_mapping=mapping)
    if warning:report['warning'] = ' '.join(filter(None, (report.get('warning'), warning)))
    if report['changed']:
        np.savez(folder / f'motion-{kind}-stance-before-contact.npz', **stance)
        points, world = fk(rest, roots, quats)
        stance.update(local=quat_matrix(quats), points=points, world=world)
        np.savez(stance_path, **stance)
    return report
