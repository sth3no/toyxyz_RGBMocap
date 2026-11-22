"""Optional Kimodo overlay on the pinned Windows ExMR runtime, used by automatic fill of hidden ranges."""

import json
import sys

from .paths import ROOT

REVISION = 'e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954'
PROFILE_VERSION = 1
MODEL_FOLDER = 'kimodo'
MODEL_NAME = 'Kimodo-SOMA-RP-v1.2'
PARAMETERS = 283281777
SETUP = 'Close RGB MoCap and run Setup_Kimodo.cmd to prepare automatic fill.'
WORKFLOWS = frozenset(('gvhmr', 'exmr'))  # the workflows whose hidden ranges can be filled automatically


def location():
    folder = ROOT / 'runtime_kimodo'
    try:
        receipt = json.loads((folder / 'runtime.json').read_text(encoding='utf-8'))
        if (receipt != {'version': PROFILE_VERSION, 'revision': REVISION}
                or not (folder / 'kimodo/model/load_model.py').is_file()):
            raise ValueError('Kimodo runtime is out of date')
    except (OSError, ValueError) as error:
        raise RuntimeError(SETUP) from error
    return folder


def issue():
    try:
        location()
    except RuntimeError as error:
        return str(error)
    from .models import missing
    return 'Get the Kimodo model in Model Manager to use automatic fill.' if missing('kimodo_fill') else None


def identity():
    from .models import BY_KEY
    return f"{REVISION}:{PROFILE_VERSION}:{BY_KEY['kimodo'].expected_sha256}"


def activate(folder):
    sys.path.insert(0, str(folder))


def command(args):
    location()
    python = ROOT / 'runtime_exmr/python.exe'
    if not python.is_file():
        raise RuntimeError('The ExMR runtime is missing. Run install.cmd before Setup_Kimodo.cmd.')
    return [str(python), '-I', '-X', 'utf8', '-B', '-u',
            str(ROOT / 'kimodo_worker_boot.py'), *map(str, args)]
