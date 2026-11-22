"""Kimodo worker in the verified ExMR runtime and its Kimodo overlay: generate one hidden range, or, with a first argument
`text`, read a description into Kimodo's text vector."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from mocap.worker_progress import emit


def run(arguments):
    text = arguments[:1] == ['text']
    try:
        from mocap import kimodo_runtime
        from mocap.runtime_registry import activate
        activate(root, 'exmr')
        kimodo_runtime.activate(kimodo_runtime.location())
        if text:
            from mocap.kimodo_text import main
        else:
            from mocap.kimodo_fill import main
    except Exception as error:
        emit('failed', stage='take_join', message=str(error) or type(error).__name__)
        return 1
    return main(arguments[1:] if text else arguments, emit)


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1:]))
