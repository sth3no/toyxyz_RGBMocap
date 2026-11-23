"""Install a pinned Blender.app as an optional macOS support component."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from pinned_download import download

FOLDER = 'blender-macos'
RECEIPT = '.rgbmocap-blender.json'  # next to Blender.app: files inside the signed bundle stay untouched


def executable(app):
    return Path(app) / 'Contents/MacOS/Blender'


def verify(app, version):
    result = subprocess.run([str(executable(app)), '--version'], capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=120)
    if result.returncode or result.stdout.split()[:2] != ['Blender', version]:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError('Blender verification failed. See docs/MACOS.md. ' + detail)


def copy_from_image(image, app_name, destination):
    """Copy the app out of the read-only disk image without Finder, prompts or auto-open."""
    with TemporaryDirectory(prefix='rgbmocap-blender-mount-') as mount:
        subprocess.run(['/usr/bin/hdiutil', 'attach', '-nobrowse', '-readonly', '-noautoopen', '-noverify',
                        '-mountpoint', mount, str(image)], check=True, stdout=subprocess.DEVNULL, timeout=600)
        try:
            source = Path(mount) / app_name
            if source.is_symlink() or not executable(source).is_file():
                raise RuntimeError('The Blender disk image does not contain ' + app_name)
            subprocess.run(['/usr/bin/ditto', str(source), str(destination)], check=True, timeout=1800)
        finally:
            detached = subprocess.run(['/usr/bin/hdiutil', 'detach', mount], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, timeout=120)
            if detached.returncode:
                subprocess.run(['/usr/bin/hdiutil', 'detach', '-force', mount], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=120)


def prepare(root):
    pin = json.loads((root / 'installer/macos-blender-lock.json').read_text(encoding='utf-8'))
    app_name = pin['app']
    if not app_name.endswith('.app') or Path(app_name).name != app_name:
        raise ValueError('Invalid Blender application name')

    parent = root / 'runtime_character'
    if parent.is_symlink():
        raise RuntimeError('The character runtime path is a symbolic link.')
    folder = parent / FOLDER
    if os.path.lexists(folder):
        if folder.is_symlink() or not (folder / RECEIPT).is_file():
            raise RuntimeError('Existing unmanaged Blender was preserved. Use RGBMOCAP_BLENDER to select it.')
        if json.loads((folder / RECEIPT).read_text(encoding='utf-8')) != pin:
            raise RuntimeError('A different managed Blender is installed. See docs/MACOS.md.')
        verify(folder / app_name, pin['version'])
        print('Keeping managed Blender ' + pin['version'], flush=True)
        return

    image = download(root / '.setup/macos-blender-downloads', 'blender-' + pin['version'], pin,
                     filename=Path(pin['url']).name, allowed_hosts={'download.blender.org'})
    parent.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='.blender-stage-', dir=parent) as temporary:
        build = Path(temporary) / FOLDER
        build.mkdir()
        copy_from_image(image, app_name, build / app_name)
        verify(build / app_name, pin['version'])
        (build / RECEIPT).write_text(json.dumps(pin, indent=2), encoding='utf-8')
        if os.path.lexists(folder):
            raise RuntimeError('The Blender destination was created during setup and was preserved.')
        build.rename(folder)
    print('Installed Blender ' + pin['version'] + ' at ' + str(folder / app_name), flush=True)
