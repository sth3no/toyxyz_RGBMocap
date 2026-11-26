"""Inspect animation FBX hierarchies and run an explicit Unreal import profile."""
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

from .storage import atomic_json

FBX_MAGIC = b'Kaydara FBX Binary  \x00\x1a\x00'
ASSET_FOLDER = re.compile(r'/Game(?:/[A-Za-z0-9_]+)*')
ASSET_NAME = re.compile(r'[A-Za-z0-9_]+')
MAX_METADATA = 2 * 1024 * 1024


def profile_values(base, editor, project, mesh, destination, fbx_profile, ue5_target):
    editor, project = ((Path(base) / Path(value)).resolve(strict=True) for value in (editor, project))
    if editor.name.casefold() != 'unrealeditor-cmd.exe' or not editor.is_file():
        raise ValueError('Select UnrealEditor-Cmd.exe from the Unreal Engine installation.')
    if project.suffix.lower() != '.uproject' or not project.is_file():
        raise ValueError('Select an existing Unreal .uproject file.')
    package, dot, object_name = mesh.partition('.')
    if (not ASSET_FOLDER.fullmatch(package) or package == '/Game' or
            dot and object_name != package.rsplit('/', 1)[1]):
        raise ValueError('Select the Skeletal Mesh by its /Game asset path.')
    if not ASSET_FOLDER.fullmatch(destination):
        raise ValueError('Select an Unreal destination folder under /Game.')
    if fbx_profile not in ('generic', 'ue5_mannequin'):
        raise ValueError('Invalid FBX export profile.')
    if (fbx_profile == 'ue5_mannequin' and ue5_target not in ('manny', 'quinn') or
            fbx_profile == 'generic' and ue5_target is not None):
        raise ValueError('Choose Manny or Quinn for the UE5 Mannequin export profile.')
    return dict(version=1, editor=str(editor), project=str(project), mesh=mesh, destination=destination,
                fbx_profile=fbx_profile, ue5_target=ue5_target)


def load_profile(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        raw = stream.read(MAX_METADATA + 1)
    if len(raw) > MAX_METADATA:
        raise ValueError('The Unreal import profile is too large.')
    data = json.loads(raw.decode('utf-8-sig'))
    fields = ('editor', 'project', 'mesh', 'destination', 'fbx_profile')
    if (not isinstance(data, dict) or data.get('version') != 1 or
            not all(isinstance(data.get(key), str) and data[key] for key in fields)):
        raise ValueError('Use a version 1 Unreal import profile.')
    return profile_values(path.parent, *(data[key] for key in fields), data.get('ue5_target'))


def _value(stream):
    """Read a scalar from a Model or Connection record, never a geometry or animation array."""
    kind = stream.read(1)
    if kind in (b'I', b'L'):
        fmt = '<i' if kind == b'I' else '<q'
        return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]
    if kind == b'S':
        size = struct.unpack('<I', stream.read(4))[0]
        if size <= 4096:
            data = stream.read(size)
            if len(data) == size:
                return data.decode('utf-8')
    raise ValueError('Invalid FBX skeleton metadata.')


def _nodes(stream, header, limit):
    while stream.tell() + header.size <= limit:
        start = stream.tell()
        end, count, properties, name_size = header.unpack(stream.read(header.size))
        if not end:
            break
        name = stream.read(name_size)
        children = stream.tell() + properties
        if not start < end <= limit or not stream.tell() <= children <= end:
            raise ValueError('Invalid FBX node offsets.')
        yield name, count, children, end
        stream.seek(end)


def _name(value):
    # The Legacy importer strips the namespace and normalizes spaces in bone names.
    return value.split('\0\1', 1)[0].rsplit(':', 1)[-1].strip().replace(' ', '-')


def fbx_hierarchy(path):
    """Read the binary FBX model graph without materializing animation or mesh arrays."""
    model, parents = {}, {}
    with Path(path).open('rb') as stream:
        if stream.read(len(FBX_MAGIC)) != FBX_MAGIC:
            raise ValueError('Choose a binary animation FBX.')
        try:
            version = struct.unpack('<I', stream.read(4))[0]
            header = struct.Struct('<QQQB' if version >= 7500 else '<IIIB')
            limit = os.fstat(stream.fileno()).st_size
            creator = ''
            object_links = []
            for group, _, children, end in _nodes(stream, header, limit):
                if group == b'Creator':
                    creator = _value(stream)
                if group not in (b'Objects', b'Connections'):
                    continue
                stream.seek(children)
                for kind, count, properties_end, _ in _nodes(stream, header, end):
                    if kind == b'Model' and count >= 3:
                        ident, name, model_type = (_value(stream) for _ in range(3))
                        if ident in model:
                            raise ValueError('Duplicate FBX model ID.')
                        model[ident] = (_name(name), model_type)
                    elif kind == b'C' and count >= 3:
                        rel, child, parent = (_value(stream) for _ in range(3))
                        object_links.append((rel, child, parent))
                    if stream.tell() > properties_end:
                        raise ValueError('Invalid FBX property offsets.')
                    if len(model) + len(object_links) > 100000:
                        raise ValueError('The FBX skeleton metadata is too large.')
        except (AttributeError, IndexError, struct.error, TypeError) as error:
            raise ValueError('Invalid FBX skeleton metadata.') from error
    for rel, child, parent in object_links:
        if rel == 'OO' and child in model and (parent in model or parent == 0):
            if child in parents:
                raise ValueError('A model in the FBX has more than one parent.')
            parents[child] = parent
    bones = {ident for ident, (_, kind) in model.items() if kind in ('LimbNode', 'Root', 'Limb')}
    if not bones:
        raise ValueError('The FBX has no skeleton bones.')
    included = set(bones)
    for bone in bones:
        seen = {bone}
        parent = parents.get(bone)
        while parent in model:
            if parent in seen:
                raise ValueError('Cycle in the FBX skeleton hierarchy.')
            seen.add(parent)
            if ('blender' in creator.casefold() and not parents.get(parent) and
                    model[parent][0].casefold() == 'armature' and model[parent][1] == 'Null'):
                break
            included.add(parent)
            parent = parents.get(parent)
    out = {model[key][0]: model[p][0] if (p := parents.get(key)) in included else None
           for key in sorted(included)}
    if len(out) != len(included) or len({name.casefold() for name in out}) != len(out) or '' in out:
        raise ValueError('Duplicate or empty FBX bone name.')
    return out


def check_hierarchy(source, target):
    source_root = [name for name, parent in source.items() if parent is None]
    target_root = [name for name, parent in target.items() if parent is None]
    if len(source_root) != 1 or len(target_root) != 1 or source_root[0].casefold() != target_root[0].casefold():
        raise ValueError(f'The FBX root {source_root} does not match the target root {target_root}.')
    target_names = {name.casefold(): parent for name, parent in target.items()}
    internal = set(source.values())
    for name, parent in source.items():
        key = name.casefold()
        if key not in target_names:
            if name not in internal and parent and name == parent + '_end':
                continue  # Native FBX terminal marker.
            raise ValueError(f'The target skeleton is missing the FBX bone {name}.')
        expected = target_names[key]
        if (parent or '').casefold() != (expected or '').casefold():
            raise ValueError(f'The parent of {name} is {parent} in the FBX and {expected} in the target.')


def safe_asset_name(name):
    return re.sub(r'[^A-Za-z0-9_]+', '_', name)


def file_list(files):
    paths = [Path(path).resolve(strict=True) for path in files]
    if not 1 <= len(paths) <= 4096:
        raise ValueError('Select between 1 and 4096 animation FBX files.')
    names = set()
    for path in paths:
        if not path.is_file() or path.suffix.casefold() != '.fbx':
            raise ValueError(f'Choose an existing FBX file: {path}')
        if not ASSET_NAME.fullmatch(path.stem):
            raise ValueError(f'Use letters, digits and underscores for the FBX asset name: {path.name}')
        if path.stem.casefold() in names:
            raise ValueError(f'More than one FBX would create {path.stem}.')
        names.add(path.stem.casefold())
    return paths


def import_files(profile, files, report, *, dry_run=False):
    paths = file_list(files)
    from .paths import ROOT
    if set_phase := getattr(report, 'set_phase', None):
        set_phase('unreal_check' if dry_run else 'unreal_import', project=profile['project'])
    with tempfile.TemporaryDirectory(prefix='rgbmocap-unreal-') as folder:
        root = Path(folder)
        request, response = root / 'request.json', root / 'result.json'
        atomic_json(request, dict(profile=profile, files=[str(path) for path in paths], dry_run=dry_run))
        script = ROOT / 'tools' / 'unreal_animation_commandlet.py'
        args = [profile['editor'], profile['project'], '-run=pythonscript', f'-script={script}',
                '-unattended', '-nullrhi', '-nosplash', '-nosound', '-nop4', '-NoCrashDialog', f'-abslog={root / "unreal.log"}']
        env = dict(os.environ, RGBMOCAP_UNREAL_REQUEST=str(request))
        report(f'Checking {len(paths)} animation FBX file(s) in Unreal.')
        log = root / 'unreal.log'
        failure = None
        try:
            with subprocess.Popen(args, cwd=Path(profile['project']).parent, env=env, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as editor:
                try:
                    code = editor.wait(timeout=1800)
                except BaseException:
                    editor.kill()
                    raise
            if code:
                failure = f'Unreal exited with code {code}.'
        except subprocess.TimeoutExpired:
            failure = 'The Unreal import did not finish within 30 minutes.'
        if response.is_file():
            data = json.loads(response.read_text(encoding='utf-8'))
            if failure or len(data.get('animations', [])) != len(paths):
                data.update(failed=True, error=data.get('error') or failure or 'Unreal stopped before processing every file.')
            return data
        detail = ''
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 4000))
                detail = stream.read().decode('utf-8', errors='replace').strip()
        raise RuntimeError((failure or 'Unreal did not return an import report. Check the Python Editor Script Plugin.')
                           + ('\n' + detail if detail else ''))


def run(args, report):
    if args.unreal_action == 'profile':
        path = args.output.resolve()
        if path.suffix.lower() != '.json':
            raise ValueError('Choose a .json destination for the Unreal import profile.')
        values = profile_values(Path.cwd(), args.editor, args.project, args.mesh, args.destination,
                                args.fbx_profile, args.ue5_target)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(values, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return {'profile': str(path), 'mesh': values['mesh'], 'destination': values['destination']}
    return import_files(load_profile(args.profile), args.fbx, report, dry_run=args.dry_run)
