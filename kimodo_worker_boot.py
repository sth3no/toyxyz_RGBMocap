"""Generate one hidden range with Kimodo in the verified ExMR runtime and its Kimodo overlay."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from mocap.worker_progress import emit


def run(arguments):
    try:
        from mocap import kimodo_runtime
        from mocap.runtime_registry import activate
        activate(root, 'exmr')
        kimodo_runtime.activate(kimodo_runtime.location())
        from mocap.kimodo_fill import main
    except Exception as error:
        emit('failed', stage='take_join', message=str(error) or type(error).__name__)
        return 1
    return main(arguments, emit)


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1:]))
