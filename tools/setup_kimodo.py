"""Install a pinned, optional Kimodo overlay for automatic fill without changing base packages."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.kimodo_runtime import PROFILE_VERSION, REVISION, location
from tools.session_lock import app_lock

PACKAGES = (
    'hydra-core==1.3.8 --hash=sha256:d5f184c291691cba835c7f04bc4e36270ad88b1d7e394a3712f5af084c30b394',
    'pydantic==2.14.4 --hash=sha256:653a2ad9f6d8858529c77c72717d0ea991d6c6837fe683246b2f4b2bbfc3fadc',
    'pydantic-core==2.50.5 --hash=sha256:baa107902f8e38011a5f79836d905eb6bfaac1d4b0bd09fc86509b711d0b09c8',
    'annotated-types==0.8.0 --hash=sha256:f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0',
    'typing-inspection==0.4.5 --hash=sha256:bc210535442b39e1bf2805e0c7a17dfd6a4ace760c84c7f7d16245d23a6b1c39',
)
# No wheel exists; it is pure Python and builds with the runtime's own setuptools.
SOURCES = (
    'antlr4-python3-runtime==4.9.3 --hash=sha256:f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b',
)
# Robot meshes and demo files are not used by automatic fill.
LEFT_OUT = ('assets/skeletons/g1skel34', 'assets/demo')


def verify(folder):
    code = r'''
import sys, types
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[1])
from pathlib import Path
# The real module imports transformers and peft, which are not installed and not needed without text.
stand_in = types.ModuleType('kimodo.model.llm2vec')
stand_in.LLM2VecEncoder = None
sys.modules['kimodo.model.llm2vec'] = stand_in
import numpy as np, torch
import kimodo
from kimodo.skeleton.registry import build_skeleton
from mocap.soma_convert import SOMA30, SOMA_NEUTRAL
assert Path(kimodo.__file__).is_relative_to(Path(sys.argv[1]))
skeleton = build_skeleton(30)
assert list(skeleton.bone_order_names) == list(SOMA30)
neutral = torch.as_tensor(skeleton.neutral_joints).detach().double().cpu().numpy()
assert neutral.shape == SOMA_NEUTRAL.shape and np.abs(neutral - SOMA_NEUTRAL).max() <= 1e-5
if not torch.cuda.is_available():
    raise RuntimeError("Automatic fill requires a CUDA GPU.")
x = torch.randn((64, 64), device="cuda")
assert (x @ x).isfinite().all().item()
assert 'transformers' not in sys.modules
'''
    subprocess.run([sys.executable, '-I', '-X', 'utf8', '-B', '-c', code, str(folder), str(ROOT)],
                   check=True, timeout=180)


def pip(requirements, overlay, *options):
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull)
    subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install',
                    '--no-deps', '--no-cache-dir', '--require-hashes', *options,
                    '--index-url', 'https://pypi.org/simple', '--target', str(overlay),
                    '-r', str(requirements)], check=True, env=env, timeout=300)


def build(stage: Path) -> Path:
    overlay = stage / 'overlay'
    checkout = stage / 'upstream'
    checkout.mkdir()
    subprocess.run(['git', '-C', str(checkout), 'init', '-q'], check=True, timeout=30)
    git = ['git', '-C', str(checkout)]
    subprocess.run([*git, 'fetch', '--depth=1', 'https://github.com/nv-tlabs/kimodo.git', REVISION],
                   check=True, timeout=300)
    fetched = subprocess.check_output([*git, 'rev-parse', 'FETCH_HEAD'], text=True, timeout=30).strip()
    if fetched != REVISION:
        raise RuntimeError('Unexpected Kimodo source revision.')
    subprocess.run([*git, 'checkout', '-q', '--detach', REVISION], check=True, timeout=60)
    wheels = stage / 'wheels.txt'
    wheels.write_text('\n'.join(PACKAGES) + '\n', encoding='utf-8')
    pip(wheels, overlay, '--only-binary=:all:')
    if subprocess.run([sys.executable, '-I', '-c', 'import setuptools'], timeout=60).returncode:
        raise RuntimeError('The ExMR runtime cannot import setuptools, which builds antlr4-python3-runtime. '
                           'Run install.cmd to repair the ExMR runtime.')
    sources = stage / 'sources.txt'
    sources.write_text('\n'.join(SOURCES) + '\n', encoding='utf-8')
    pip(sources, overlay, '--no-binary', 'antlr4-python3-runtime', '--no-build-isolation')
    package = checkout / 'kimodo'
    left_out = {package / name for name in LEFT_OUT}
    shutil.copytree(package, overlay / 'kimodo', ignore=lambda folder, names: [
        name for name in names if name == '__pycache__' or Path(folder) / name in left_out])
    shutil.copy2(checkout / 'LICENSE', overlay / 'KIMODO_LICENSE')
    shutil.copy2(checkout / 'ATTRIBUTIONS.MD', overlay / 'KIMODO_ATTRIBUTIONS.MD')
    verify(overlay)
    return overlay


def install():
    from mocap.runtime_registry import activate
    activate(ROOT, 'exmr')
    try:
        existing = location()
    except RuntimeError:
        pass
    else:
        verify(existing)
        print('Verified Kimodo runtime retained.')
        return
    with tempfile.TemporaryDirectory(prefix='.kimodo-', dir=ROOT / '.setup') as name:
        overlay = build(Path(name))
        (overlay / 'runtime.json').write_text(json.dumps(dict(version=PROFILE_VERSION, revision=REVISION)),
                                              encoding='utf-8')
        destination = ROOT / 'runtime_kimodo'
        backup = None
        if destination.exists():
            backup = ROOT / '.setup' / ('kimodo-backup-' + uuid.uuid4().hex)
            destination.rename(backup)
        try:
            overlay.rename(destination)
        except OSError:
            if backup is not None:
                backup.rename(destination)
            raise
    print('Kimodo runtime ready. Get the optional Kimodo model in Model Manager.')


if __name__ == '__main__':
    try:
        with app_lock(ROOT):
            install()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f'Kimodo setup failed: {error}')
