"""Run inside Unreal Editor with the local import request written by RGB MoCap."""
import json
import os
from pathlib import Path
import sys

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.storage import atomic_json
from mocap.unreal_animation import check_hierarchy, fbx_hierarchy


def target_rig(profile):
    mesh = unreal.load_asset(profile['mesh'])
    if not isinstance(mesh, unreal.SkeletalMesh):
        raise ValueError('The target asset is not an Unreal Skeletal Mesh.')
    skeleton = mesh.get_editor_property('skeleton')
    if not isinstance(skeleton, unreal.Skeleton):
        raise ValueError('The target Skeletal Mesh has no Skeleton.')
    pose = unreal.AnimPoseExtensions.get_reference_pose(skeleton)
    bones = {}
    for name in pose.get_bone_names():
        parent = str(mesh.get_bone_parent(name))
        bones[str(name)] = None if parent == 'None' else parent
    if not bones or sum(parent is None for parent in bones.values()) != 1:
        raise ValueError('Select a Skeletal Mesh with the complete reference skeleton.')
    return skeleton, bones


def import_one(path, destination, skeleton, bones, dry_run):
    asset = destination + '/' + path.stem
    if unreal.EditorAssetLibrary.does_asset_exist(asset):
        raise ValueError(f'An asset already exists at {asset}.')
    check_hierarchy(fbx_hierarchy(path), bones)
    if dry_run:
        return dict(state='compatible', asset=asset)
    settings = unreal.FbxImportUI()
    settings.automated_import_should_detect_type = False
    settings.import_as_skeletal = True
    settings.import_mesh = settings.import_materials = settings.import_textures = False
    settings.create_physics_asset = False
    settings.import_animations = True
    settings.mesh_type_to_import = settings.original_import_type = unreal.FBXImportType.FBXIT_ANIMATION
    settings.skeleton = skeleton
    settings.anim_sequence_import_data.set_editor_property('use_default_sample_rate', False)
    task = unreal.AssetImportTask()
    task.filename = str(path)
    task.destination_path = destination
    task.destination_name = path.stem
    task.automated = True
    task.replace_existing = False
    task.save = False
    task.options = settings
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    objects = task.get_objects()
    if len(objects) != 1 or not isinstance(objects[0], unreal.AnimSequence):
        raise RuntimeError('Unreal did not create one animation sequence.')
    sequence = objects[0]
    if sequence.get_editor_property('skeleton') != skeleton:
        raise RuntimeError('Unreal created a sequence with a different skeleton.')
    if sequence.get_outermost().get_name() != asset:
        raise RuntimeError('Unreal used a different asset name.')
    if not unreal.EditorAssetLibrary.save_loaded_asset(sequence):
        raise RuntimeError(f'Unreal could not save {asset}.')
    return dict(state='imported', asset=sequence.get_path_name(), seconds=sequence.get_play_length())


def main(request):
    path = Path(request)
    with path.open(encoding='utf-8') as stream:
        data = json.load(stream)
    receipt = path.with_name('result.json')
    profile = data['profile']
    dry_run = data['dry_run']
    report = dict(dry_run=dry_run, animations=[], failed=False)
    try:
        unreal.SystemLibrary.execute_console_command(None, 'Interchange.FeatureFlags.Import.FBX 0')
        skeleton, bones = target_rig(profile)
        report['skeleton'] = skeleton.get_path_name()
        for name in data['files']:
            row = {'file': name}
            try:
                row.update(import_one(Path(name), profile['destination'], skeleton, bones, dry_run))
            except (OSError, ValueError, TypeError, RuntimeError) as error:
                row.update(state='failed', error=str(error))
                report['failed'] = True
            report['animations'].append(row)
            atomic_json(receipt, report)
    except (AttributeError, OSError, ValueError, TypeError, RuntimeError) as error:
        report.update(failed=True, error=str(error))
    atomic_json(receipt, report)


if __name__ == '__main__':
    main(os.environ.pop('RGBMOCAP_UNREAL_REQUEST'))
