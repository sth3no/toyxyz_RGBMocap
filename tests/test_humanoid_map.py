"""Humanoid mappings for non-Mixamo FBX characters: Unity .meta parsing, sidecar files and virtual joints."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap.retarget import humanoid_map as hm  # noqa: E402
from mocap.retarget.mapping import VRM, body_mapping  # noqa: E402

META = """ModelImporter:
  humanDescription:
    serializedVersion: 3
    human:
    - boneName: Hips
      humanName: Hips
      limit:
        min: {x: 0, y: 0, z: 0}
    - boneName: L_UpperLeg
      humanName: LeftUpperLeg
      limit:
        min: {x: 0, y: 0, z: 0}
    - boneName: L_index_proximal
      humanName: Left Index Proximal
      limit:
        min: {x: 0, y: 0, z: 0}
    skeleton:
    - name: NecoBody(Clone)
      parentName:
  animationType: 3
"""

# A VRChat-style rig with two spine bones, as NecoBody.
FULL = {'hips': 'Hips', 'leftUpperLeg': 'L_UpperLeg', 'rightUpperLeg': 'R_UpperLeg', 'spine': 'Spine', 'chest': 'Chest',
        'leftLowerLeg': 'L_LowerLeg', 'rightLowerLeg': 'R_LowerLeg', 'leftFoot': 'L_Foot', 'rightFoot': 'R_Foot',
        'leftToes': 'L_Toe', 'rightToes': 'R_Toe', 'neck': 'Neck', 'leftShoulder': 'L_Shoulder', 'rightShoulder': 'R_Shoulder',
        'head': 'Head', 'leftUpperArm': 'L_UpperArm', 'rightUpperArm': 'R_UpperArm', 'leftLowerArm': 'L_LowerArm',
        'rightLowerArm': 'R_LowerArm', 'leftHand': 'L_hand', 'rightHand': 'R_hand'}
BONES = list(FULL.values()) + ['UpperChest', 'oppai']
SLOT = {n: i for i, n in enumerate(VRM)}


def last_wins(mapping):
    return {name: i for i, name in enumerate(mapping)}


class UnityMeta(unittest.TestCase):
    def test_parses_human_bones_only(self):
        self.assertEqual(hm.parse_unity_meta(META), {'hips': 'Hips', 'leftUpperLeg': 'L_UpperLeg', 'leftIndexProximal': 'L_index_proximal'})

    def test_meta_without_avatar(self):
        self.assertIsNone(hm.parse_unity_meta('ModelImporter:\n  animationType: 2\n'))


class Sidecar(unittest.TestCase):
    def test_json_wins_over_meta_and_keeps_options(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'Neco.fbx'
            (Path(d) / 'Neco.fbx.meta').write_text(META, encoding='utf-8')
            self.assertEqual(hm.load_sidecar(model)[2], 'Neco.fbx.meta')
            (Path(d) / 'Neco.humanoid.json').write_text(json.dumps(dict(version=1, bones=FULL, missing_spine='mid')), encoding='utf-8')
            bones, options, source = hm.load_sidecar(model)
            self.assertEqual((bones, options, source), (FULL, {'missing_spine': 'mid'}, 'Neco.humanoid.json'))

    def test_none_without_files(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(hm.load_sidecar(Path(d) / 'Neco.fbx'))


class Mapping(unittest.TestCase):
    def test_complete_rig_maps_one_to_one(self):
        semantic = dict(FULL, upperChest='UpperChest')
        mapping = body_mapping('humanoid', BONES, semantic)
        self.assertEqual(mapping, [semantic[n] for n in VRM])

    def test_missing_upper_chest_low(self):
        mapping = body_mapping('humanoid', BONES, FULL, {'missing_spine': 'low'})
        self.assertEqual([mapping[SLOT[n]] for n in ('spine', 'chest', 'upperChest')], ['Spine', 'Chest', 'Chest'])
        self.assertEqual(last_wins(mapping)['Chest'], SLOT['upperChest'])

    def test_missing_upper_chest_mid(self):
        mapping = body_mapping('humanoid', BONES, FULL, {'missing_spine': 'mid'})
        self.assertEqual([mapping[SLOT[n]] for n in ('spine', 'chest', 'upperChest')], ['Spine', 'Spine', 'Chest'])
        self.assertEqual(last_wins(mapping)['Spine'], SLOT['chest'])

    def test_missing_neck_and_shoulders_reuse_the_next_bone(self):
        semantic = {k: v for k, v in FULL.items() if k not in {'neck', 'leftShoulder', 'rightShoulder'}}
        mapping = body_mapping('humanoid', BONES, semantic)
        self.assertEqual(mapping[SLOT['neck']], 'Head')
        self.assertEqual(mapping[SLOT['leftShoulder']], 'L_UpperArm')
        lookup = last_wins(mapping)
        self.assertEqual((lookup['Head'], lookup['L_UpperArm']), (SLOT['head'], SLOT['leftUpperArm']))

    def test_errors(self):
        cases = [
            ({k: v for k, v in FULL.items() if k != 'leftToes'}, 'Missing required humanoid bones: leftToes'),
            ({k: v for k, v in FULL.items() if k != 'chest'}, 'single spine bone'),
            (dict(FULL, head='Kopf'), 'not found in the character: head=Kopf'),
            (dict(FULL, rightHand='L_hand'), 'distinct'),
        ]
        for semantic, message in cases:
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    body_mapping('humanoid', BONES, semantic)
        with self.assertRaisesRegex(ValueError, 'missing_spine'):
            body_mapping('humanoid', BONES, FULL, {'missing_spine': 'top'})

    def test_existing_kinds_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'Expected a Mixamo humanoid rig'):
            body_mapping('mixamo', BONES, {})
        with self.assertRaisesRegex(ValueError, 'Unsupported humanoid mapping'):
            body_mapping('vrm', BONES, FULL)


if __name__ == '__main__':
    unittest.main()
