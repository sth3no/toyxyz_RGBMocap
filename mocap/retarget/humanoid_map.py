"""Humanoid bone mappings for FBX characters that do not use Mixamo bone names.

A mapping is keyed by humanoid bone names in VRM spelling (`hips`, `leftUpperLeg`, `leftIndexProximal`, ...), which are
Unity Humanoid's names with the first letter lowered and spaces removed. It comes from a file next to the character:

- `<file>.humanoid.json`: {"version": 1, "bones": {"hips": "Hips", ...}, "missing_spine": "low" | "mid"}
- `<file>.meta`: a Unity model import file whose Humanoid avatar (`humanDescription`) is configured.

Stdlib only: this runs in the bundled Blender as well as in the app.
"""
import json
import re
from pathlib import Path

VERSION = 1
MAX_SIDECAR_BYTES = 4 * 1024 * 1024
MISSING_SPINE = ('low', 'mid')
# Unity Humanoid treats these as optional; every other body bone of the 22-joint canonical skeleton is required.
OPTIONAL = {'chest', 'upperChest', 'neck', 'leftShoulder', 'rightShoulder'}
FINGERS = [side + finger + part for side in ('left', 'right') for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Little')
           for part in ('Proximal', 'Intermediate', 'Distal')]


class MappingError(ValueError):
    pass


def unity_name(human_name):
    """'Left Index Proximal' / 'LeftUpperLeg' -> 'leftIndexProximal' / 'leftUpperLeg'."""
    compact = human_name.replace(' ', '')
    return compact[:1].lower() + compact[1:]


def parse_unity_meta(text):
    """Bone mapping of a Unity model `.meta` with a configured Humanoid avatar, or None when it has none."""
    if 'humanDescription:' not in text:
        return None
    human = text.split('humanDescription:', 1)[1].split('\n    skeleton:', 1)[0]
    pairs = re.findall(r'-\s*boneName:\s*(.+?)\s*\n\s*humanName:\s*(.+?)\s*\n', human)
    bones = {unity_name(h.strip().strip('"\'')): b.strip().strip('"\'') for b, h in pairs}
    return bones or None


def load_sidecar(model_path):
    """(bones, options, source) from a mapping file next to `model_path`, or None when there is none."""
    model_path = Path(model_path)
    explicit = model_path.with_name(model_path.stem + '.humanoid.json')
    if explicit.is_file():
        data = json.loads(read_sidecar(explicit))
        if (not isinstance(data, dict) or data.get('version') != VERSION
                or not isinstance(data.get('bones'), dict)):
            raise MappingError(f'{explicit.name}: expected {{"version": {VERSION}, "bones": {{...}}}}')
        if any(not isinstance(value, str) or not value.strip() for value in data['bones'].values()):
            raise MappingError(f'{explicit.name}: mapped bones must be nonempty names.')
        options = {k: v for k, v in data.items() if k not in {'version', 'bones'}}
        soft = options.get('soft_meshes')
        if soft is not None and (not isinstance(soft, list) or not all(isinstance(item, str) for item in soft)):
            raise MappingError(f'{explicit.name}: soft_meshes must contain mesh names.')
        return dict(data['bones']), options, explicit.name
    meta = model_path.with_name(model_path.name + '.meta')
    if meta.is_file():
        bones = parse_unity_meta(read_sidecar(meta))
        if bones:
            return bones, {}, meta.name
    return None


def read_sidecar(path):
    with path.open('rb') as source:
        data = source.read(MAX_SIDECAR_BYTES + 1)
    if len(data) > MAX_SIDECAR_BYTES:
        raise MappingError(f'{path.name}: humanoid mapping exceeds 4 MiB.')
    return data.decode('utf-8-sig')


def body_mapping(vrm_names, bones, semantic, missing_spine='low'):
    """Bone name per canonical joint (in `vrm_names` order) for a Unity-Humanoid-style mapping.

    A rig without a joint the canonical skeleton has gets a virtual joint: the slot reuses the bone of the real joint
    that follows it, so it sits at that bone's head with a zero-length segment and the solver's chain matches the real
    one exactly. The virtual slot always has the lower index, so name -> index lookups (built in order, last wins)
    resolve to the real slot, which is the one keyed.
    - no upper chest, missing_spine='low': Spine1 at the chest head; the spine bone follows the source's lower segment.
    - no upper chest, missing_spine='mid': Spine at the spine head; the spine bone follows the source's middle segment.
    - no neck: Neck at the head bone's head. No shoulder: Shoulder at the upper arm's head.
    """
    if missing_spine not in MISSING_SPINE:
        raise MappingError(f'missing_spine must be one of {", ".join(MISSING_SPINE)}')
    names = set(bones)
    unknown = sorted(k for k, v in semantic.items() if v and v not in names and (k in vrm_names or k in FINGERS))
    if unknown:
        raise MappingError('Mapped bones not found in the character: ' + ', '.join(f'{k}={semantic[k]}' for k in unknown))
    slot = {n: i for i, n in enumerate(vrm_names)}
    mapping = [semantic.get(n) or None for n in vrm_names]
    missing = [n for n, v in zip(vrm_names, mapping) if v is None and n not in OPTIONAL]
    if missing:
        raise MappingError('Missing required humanoid bones: ' + ', '.join(missing))
    if mapping[slot['chest']] is None:
        raise MappingError('Rigs with a single spine bone (no chest) are not supported yet.')
    if mapping[slot['upperChest']] is None:
        chest, spine = mapping[slot['chest']], mapping[slot['spine']]
        mapping[slot['upperChest']] = chest
        if missing_spine == 'low':
            mapping[slot['chest']] = chest
        else:
            mapping[slot['chest']] = spine
            mapping[slot['spine']] = spine
    if mapping[slot['neck']] is None:
        mapping[slot['neck']] = mapping[slot['head']]
    for side in ('left', 'right'):
        if mapping[slot[side + 'Shoulder']] is None:
            mapping[slot[side + 'Shoulder']] = mapping[slot[side + 'UpperArm']]
    distinct = len(set(mapping))
    virtual = len(mapping) - distinct
    expected = sum(semantic.get(n) is None for n in OPTIONAL)
    if virtual != expected:
        raise MappingError('Humanoid bones must have distinct mappings')
    return mapping
