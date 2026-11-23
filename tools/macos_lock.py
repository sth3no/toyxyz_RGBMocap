"""Derive installer/runtime-lock-macos.json from the Windows lock: same package versions, Apple Silicon wheels.

Maintainer tool; run it with any Python that has `packaging`, after changing the Windows lock or
runtime_registry.py. It works like tools/linux_lock.py, with three differences:

- Only the shared, gui and exmr profiles are converted. ExMR runs on Apple GPUs (MPS); FlowHMR,
  GVHMR and their add-ons are CUDA-only, so flow/hand/htd stay out of the lock and
  runtime_registry.command reports them as missing.
- CUDA local builds (for example torch 2.7.1+cu128) are replaced by the same public PyPI version,
  which is the CPU/MPS build for macOS. runtime_registry.version_matches accepts it on macOS only.
- Each platform wheel is the best macOS arm64 (or universal2) wheel of the same version, except
  the wheels in UNLOADABLE, which fail to load on current macOS; the next best wheel is used.

Nothing is installed.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from packaging.requirements import Requirement
from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linux_lock import WINDOWS_ENV, candidates, entry, get, inspect_wheel, lock_id  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('shared', 'gui', 'exmr')
# Newest deployment target up to macOS 14 first, then newer targets only when nothing older exists.
MACOS = [*range(14, 10, -1), 15]
PLATFORMS = ([f'macosx_{major}_0_arm64' for major in MACOS] + [f'macosx_{major}_0_universal2' for major in MACOS]
             + [f'macosx_10_{minor}_universal2' for minor in range(16, 8, -1)])
# scipy 1.15.2's Accelerate wheel ships a gfortran-built _spropack that dyld on macOS 27 rejects
# ("__DATA/__thread_bss has a zero-fill section type"), breaking `import scipy.spatial`. Its
# OpenBLAS macosx_12_0 wheel of the same version loads.
UNLOADABLE = {'scipy-1.15.2-cp312-cp312-macosx_14_0_arm64.whl'}
BASE_ENV = {'os_name': 'posix', 'sys_platform': 'darwin', 'platform_system': 'Darwin', 'platform_machine': 'arm64',
            'implementation_name': 'cpython', 'platform_python_implementation': 'CPython', 'extra': ''}


def public(version):
    return version.split('+', 1)[0]


def macos_wheel(name, version, python):
    major, minor = map(int, python.split('.')[:2])
    order = {tag: i for i, tag in enumerate([*cpython_tags((major, minor), platforms=PLATFORMS),
                                             *compatible_tags((major, minor), platforms=PLATFORMS)])}
    def rank(filename):
        return min((order.get(Tag(t.interpreter, t.abi, t.platform), 10**9) for t in parse_wheel_filename(filename)[3]), default=10**9)
    found = [row for row in candidates(name, version)
             if parse_wheel_filename(row[0])[1] == Version(version) and row[0] not in UNLOADABLE]
    best = min(found, key=lambda row: rank(row[0]), default=None)
    if best is None or rank(best[0]) == 10**9:
        raise RuntimeError(f'No macOS arm64 wheel for {name}=={version} (Python {python}).')
    return best


def minimum_macos(filename):
    """Lowest macOS version a wheel accepts (from its platform tag), or None for pure wheels."""
    versions = [tuple(map(int, m.groups())) for t in parse_wheel_filename(filename)[3]
                if (m := re.fullmatch(r'macosx_(\d+)_(\d+)_(?:arm64|universal2)', t.platform))]
    return min(versions) if versions else None


def convert_group(group, base_names):
    python = group['python'];env = {**BASE_ENV, 'python_version': '.'.join(python.split('.')[:2]), 'python_full_version': python}
    win_env = {**WINDOWS_ENV, 'python_version': env['python_version'], 'python_full_version': python}
    packages, installed, queue = [], 0, []
    names = {canonicalize_name(p['name']) for p in group['packages']} | base_names
    for package in group['packages']:
        if package.get('bundled'):
            packages.append(dict(package));installed += package['bytes'] * 3  # rebuilt pure sources; compressed estimate
            continue
        pure = package['filename'].endswith('-none-any.whl')
        version = public(package['version'])
        row = (package['filename'], package['url'], package['sha256'], package['bytes']) if pure else macos_wheel(package['name'], version, python)
        queue.append((package['name'], version, row))
    while queue:
        name, version, row = queue.pop(0)
        size, requires = inspect_wheel(row[1], row[3])
        packages.append(entry(name, version, row));installed += size
        print(f"  {group['key']}: {row[0]} ({size / 1024**2:.0f} MiB installed)", flush=True)
        for text in requires:
            requirement = Requirement(text)
            if requirement.marker is None or not requirement.marker.evaluate(env) or requirement.marker.evaluate(win_env):
                continue  # Only requirements that macOS needs and Windows does not are added.
            key = canonicalize_name(requirement.name)
            if key in names: continue
            pins = [s.version for s in requirement.specifier if s.operator == '==']
            if len(pins) != 1: raise RuntimeError(f'{name} needs unpinned macOS-only {text}; pin it explicitly.')
            names.add(key);queue.append((key, pins[0], macos_wheel(key, pins[0], python)))
    packages.sort(key=lambda p: canonicalize_name(p['name']))
    return dict(group, packages=packages), installed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--uv-sha256', required=True, help='SHA-256 of the extracted aarch64-apple-darwin uv binary for the pinned uv_version')
    parser.add_argument('--output', type=Path, default=ROOT / 'installer/runtime-lock-macos.json')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.uv_sha256): raise SystemExit('Invalid --uv-sha256')
    windows = json.loads((ROOT / 'installer/runtime-lock.json').read_text(encoding='utf-8'))
    groups, sizes, names = [], {}, {}
    for group in windows['groups']:
        if group['key'] not in PROFILES: continue
        print('Group', group['key'], flush=True)
        base = names.get(group['base'], set()) if group['base'] else set()
        converted, sizes[group['key']] = convert_group(group, base)
        names[group['key']] = {canonicalize_name(p['name']) for p in converted['packages']}
        groups.append(converted)
    unique = {p['sha256']: p['bytes'] for g in groups for p in g['packages'] if p.get('url')}
    download, installed = sum(unique.values()), sum(sizes.values())
    floor = max(filter(None, (minimum_macos(p['filename']) for g in groups for p in g['packages'])), default=(11, 0))
    lock = dict(windows, groups=groups, python_versions=sorted({g['python'] for g in groups}),
                all_runtimes=False, minimum_macos='.'.join(map(str, floor)),
                download_bytes=download, installed_bytes_estimate=installed,
                # Verified downloads + installed runtimes + the largest group's transient extraction + 2 GiB margin.
                fresh_free_bytes_required=download + installed + max(sizes.values()) + 2 * 1024**3,
                uv_sha256=args.uv_sha256,
                registry_sha256=hashlib.sha256((ROOT / 'mocap/runtime_registry.py').read_bytes()).hexdigest())
    lock.pop('id', None);lock['id'] = lock_id(lock)
    temporary = args.output.with_name(args.output.name + '.partial')
    temporary.write_text(json.dumps(lock, indent=2) + '\n', encoding='utf-8');temporary.replace(args.output)
    print(f'Wrote {args.output}: {download / 1024**3:.2f} GiB download, {installed / 1024**3:.2f} GiB installed, '
          f'{lock["fresh_free_bytes_required"] / 1024**3:.2f} GiB free space required, macOS {lock["minimum_macos"]} or newer.')


if __name__ == '__main__':
    sys.exit(main())
