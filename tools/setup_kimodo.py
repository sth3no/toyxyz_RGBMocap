"""Install a pinned, optional Kimodo overlay for automatic fill without changing base packages."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.kimodo_runtime import PROFILE_VERSION, REVISION, location
from tools.session_lock import app_lock

STAGE = 'kimodo_setup'
STEPS = 5
FREE = 1024 ** 3
NO_SPACE = 'Not enough free disk space to set up automatic fill (1 GiB needed).'
NO_GIT = 'Git was not found. Install Git for Windows or run install.cmd again.'
NO_DOWNLOAD = 'Automatic fill setup could not download its files. Check the internet connection and try again.'

PACKAGES = (
    'hydra-core==1.3.8 --hash=sha256:d5f184c291691cba835c7f04bc4e36270ad88b1d7e394a3712f5af084c30b394',
    'pydantic==2.14.4 --hash=sha256:653a2ad9f6d8858529c77c72717d0ea991d6c6837fe683246b2f4b2bbfc3fadc',
    'pydantic-core==2.50.5 --hash=sha256:baa107902f8e38011a5f79836d905eb6bfaac1d4b0bd09fc86509b711d0b09c8',
    'annotated-types==0.8.0 --hash=sha256:f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0',
    'typing-inspection==0.4.5 --hash=sha256:bc210535442b39e1bf2805e0c7a17dfd6a4ace760c84c7f7d16245d23a6b1c39',
    # Text encoder (version 2). transformers 5.1.0 refuses huggingface_hub 2.x, so the overlay's 1.33.0
    # shadows the runtime's copy inside the Kimodo worker only.
    'transformers==5.1.0 --hash=sha256:de534b50c9b2ce6217fc56421075a1734241fb40704fdc90f50f6a08fc533d59',
    'peft==0.22.0 --hash=sha256:397f09710e3e1b6ad730211468839aa271732b51a50e99d4f44f2a6e8d9a9f0b',
    'tokenizers==0.22.2 --hash=sha256:c9ea31edff2968b44a88f97d784c2f16dc0729b8b143ed004699ebca91f05c48',
    'regex==2026.11.19 --hash=sha256:6b379ac8a3d0f21f2490e3b6264bb82e0bf572c0c14c44d6f12b8e4e3fa37117',
    'httpx==0.28.1 --hash=sha256:d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad',
    'httpcore==1.0.9 --hash=sha256:2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55',
    'huggingface_hub==1.33.0 --hash=sha256:04e434b06e100eddbce9a6e817d72693a7884b10a79bd67ab48080d5c07eb899',
)
# No wheel exists; it is pure Python and builds with the runtime's own setuptools.
SOURCES = (
    'antlr4-python3-runtime==4.9.3 --hash=sha256:f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b',
)
# Robot meshes and demo files are not used by automatic fill.
LEFT_OUT = ('assets/skeletons/g1skel34', 'assets/demo')


def run(command, timeout, **options):
    """Run a tool, passing its output through line by line; a failure carries its last output line."""
    last, expired = '', []
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', **options) as process:
        def stop():
            expired.append(True)
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(process.pid)], capture_output=True)
            process.kill()
        timer = threading.Timer(timeout, stop)
        timer.start()
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(line, flush=True)
                    last = line
        finally:
            timer.cancel()
        code = process.wait()
    if expired:
        raise subprocess.TimeoutExpired(command, timeout, output=last)
    if code:
        raise subprocess.CalledProcessError(code, command, output=last)


def check(code, *args):
    try:
        run([sys.executable, '-I', '-X', 'utf8', '-B', '-u', '-c', code, *map(str, args)], timeout=180)
    except subprocess.SubprocessError as error:
        raise RuntimeError('The Kimodo runtime check failed: ' + (error.output or type(error).__name__)) from error


def verify(folder):
    code = r'''
import sys, types
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[1])
from pathlib import Path
# The real module imports transformers and peft, which the generator does not need: the text worker reads a description.
stand_in = types.ModuleType('kimodo.model.llm2vec')
stand_in.LLM2VecEncoder = None
sys.modules['kimodo.model.llm2vec'] = stand_in
import numpy as np, torch
import kimodo
from kimodo.skeleton.registry import build_skeleton
from mocap.soma_convert import SOMA30, SOMA30_PARENTS, SOMA_NEUTRAL
assert Path(kimodo.__file__).is_relative_to(Path(sys.argv[1]))
skeleton = build_skeleton(30)
assert list(skeleton.bone_order_names) == list(SOMA30)
assert np.array_equal(torch.as_tensor(skeleton.joint_parents).cpu().numpy(), SOMA30_PARENTS)
neutral = torch.as_tensor(skeleton.neutral_joints).detach().double().cpu().numpy()
assert neutral.shape == SOMA_NEUTRAL.shape and np.abs(neutral - SOMA_NEUTRAL).max() <= 1e-5
if not torch.cuda.is_available():
    raise RuntimeError("Automatic fill requires a CUDA GPU.")
x = torch.randn((64, 64), device="cuda")
assert (x @ x).isfinite().all().item()
assert 'transformers' not in sys.modules
'''
    check(code, folder, ROOT)
    # The text encoder, imported for real: the overlay's transformers and huggingface_hub, not the runtime's.
    text = r'''
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import transformers
from kimodo.model.llm2vec.llm2vec import LLM2Vec
from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
assert transformers.__version__ == '5.1.0', transformers.__version__
assert Path(transformers.__file__).is_relative_to(Path(sys.argv[1])), transformers.__file__
'''
    check(text, folder)


def pip(requirements, overlay, *options):
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull)
    run([sys.executable, '-I', '-u', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install',
         '--no-deps', '--no-cache-dir', '--require-hashes', *options,
         '--index-url', 'https://pypi.org/simple', '--target', str(overlay),
         '-r', str(requirements)], env=env, timeout=300)


def build(stage: Path, step=lambda message: None) -> Path:
    """Assemble and verify the overlay in `stage`; `step` is called before each of its four phases."""
    overlay = stage / 'overlay'
    checkout = stage / 'upstream'
    checkout.mkdir()
    step('Fetching the Kimodo source')
    run(['git', '-C', str(checkout), 'init', '-q'], timeout=30)
    git = ['git', '-C', str(checkout)]
    run([*git, 'fetch', '--progress', '--depth=1', 'https://github.com/nv-tlabs/kimodo.git', REVISION], timeout=300)
    fetched = subprocess.check_output([*git, 'rev-parse', 'FETCH_HEAD'], text=True, timeout=30).strip()
    if fetched != REVISION:
        raise RuntimeError('Unexpected Kimodo source revision.')
    run([*git, 'checkout', '-q', '--detach', REVISION], timeout=60)
    step('Installing the Kimodo packages')
    wheels = stage / 'wheels.txt'
    wheels.write_text('\n'.join(PACKAGES) + '\n', encoding='utf-8')
    pip(wheels, overlay, '--only-binary=:all:')
    step('Building antlr4-python3-runtime')
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
    step('Checking the Kimodo runtime')
    verify(overlay)
    return overlay


def install(emit=None):
    """Keep the overlay when its receipt is valid and it passes its check, else install it again; with `emit`, progress
    goes to the app as events instead of plain lines. Returns the folder and whether a new runtime was installed."""
    from mocap.runtime_registry import activate
    activate(ROOT, 'exmr')
    current = iter(range(STEPS))
    def step(message):
        if emit is None:
            print(message + '...', flush=True)
        else:
            emit('progress', stage=STAGE, current=next(current), total=STEPS, message=message)
    try:
        existing = location()
    except RuntimeError:
        pass
    else:
        try:
            verify(existing)
        except RuntimeError as error:
            print(f'{error} Installing the Kimodo runtime again.', flush=True)
        else:
            print('Verified Kimodo runtime retained.', flush=True)
            return existing, False
    # A stopped or cancelled setup does not unwind; its staging folders are removed before the free space is measured.
    for left in (ROOT / '.setup').glob('.kimodo-*'):
        shutil.rmtree(left, ignore_errors=True)
    if shutil.disk_usage(ROOT).free < FREE:
        raise RuntimeError(NO_SPACE)
    if shutil.which('git') is None:
        raise RuntimeError(NO_GIT)
    (ROOT / '.setup').mkdir(exist_ok=True)
    # Cleaning the staging folder can fail on a file still held open; after the swap that must not report failure.
    with tempfile.TemporaryDirectory(prefix='.kimodo-', dir=ROOT / '.setup', ignore_cleanup_errors=True) as name:
        overlay = build(Path(name), step)
        step('Installing the Kimodo runtime')
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
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)  # the replaced runtime; a file still in use is left behind
    print('Kimodo runtime ready. Get the optional Kimodo model in Model Manager.', flush=True)
    return destination, True


def from_app():
    """Run as the app's `kimodo_setup` job: no lock (the app holds it and runs one job at a time), events on stdout."""
    from mocap.worker_progress import emit
    try:
        folder, installed = install(emit)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        last = error.output if isinstance(error.output, str) else ''  # bytes or None from a bare subprocess call
        emit('failed', stage=STAGE, message=' '.join(filter(None, (NO_DOWNLOAD, last.strip()))))
        return 1
    except Exception as error:
        emit('failed', stage=STAGE, message=str(error) or type(error).__name__)
        return 1
    emit('complete', stage=STAGE, output=str(folder),
         message='Automatic fill runtime installed.' if installed else 'Automatic fill runtime verified.')
    return 0


def main(arguments):
    if arguments == ['--from-app']:
        return from_app()
    if arguments:
        raise SystemExit('Usage: setup_kimodo.py [--from-app]')
    try:
        with app_lock(ROOT):
            install()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f'Kimodo setup failed: {error}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
