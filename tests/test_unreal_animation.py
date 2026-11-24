import tempfile
import unittest
from pathlib import Path

from mocap import unreal_animation


class UnrealAnimationTest(unittest.TestCase):
    def test_binary_fbx_hierarchy_and_root(self):
        from mocap.fbx_writer import _element, node
        import struct

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'motion.fbx'
            with path.open('wb') as stream:
                stream.write(b'Kaydara FBX Binary  \x00\x1a\x00' + struct.pack('<I', 7400))
                _element(stream, node('Objects', children=[
                    node('Model', 9, 'SourceAnimation\0\1Model', 'Null'),
                    node('Model', 10, 'Hips\0\1Model', 'LimbNode'),
                    node('Model', 11, 'Head\0\1Model', 'LimbNode'),
                    node('Model', 12, 'Head_end\0\1Model', 'LimbNode'),
                    node('Model', 40, 'Mesh\0\1Model', 'Mesh'),
                ]))
                _element(stream, node('Connections', children=[
                    node('C', 'OO', 9, 0), node('C', 'OO', 10, 9),
                    node('C', 'OO', 11, 10), node('C', 'OO', 12, 11),
                    node('C', 'OP', 10, 40),
                ]))
                stream.write(bytes(13))
            source = unreal_animation.fbx_hierarchy(path)
            self.assertEqual(source, {'SourceAnimation': None, 'Hips': 'SourceAnimation',
                                      'Head': 'Hips', 'Head_end': 'Head'})
            unreal_animation.check_hierarchy(source, source | {'Socket': 'Head'})
            unreal_animation.check_hierarchy(source, {key: value for key, value in source.items()
                                                      if key != 'Head_end'})
            with self.assertRaisesRegex(ValueError, 'parent'):
                unreal_animation.check_hierarchy(source, source | {'Head': 'SourceAnimation'})
            with self.assertRaisesRegex(ValueError, 'root'):
                unreal_animation.check_hierarchy(source, {'Hips': None, 'Head': 'Hips'})
            with self.assertRaisesRegex(ValueError, 'missing'):
                unreal_animation.check_hierarchy(source, {'SourceAnimation': None, 'Hips': 'SourceAnimation'})

    def test_target_paths_and_destination_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            editor = root / 'UnrealEditor-Cmd.exe'
            project = root / 'Game.uproject'
            editor.touch()
            project.touch()
            target = unreal_animation.profile_values(root, editor.name, project.name,
                '/Game/Characters/Body.Body', '/Game/Motions', 'ue5_mannequin', 'quinn')
            self.assertEqual(target['editor'], str(editor))
            self.assertEqual(target['project'], str(project))
            self.assertEqual(target['ue5_target'], 'quinn')
            with self.assertRaisesRegex(ValueError, '/Game'):
                unreal_animation.profile_values(root, editor.name, project.name,
                    '/Game/Characters/Body.Body', '/Game/../Engine', 'generic', None)
