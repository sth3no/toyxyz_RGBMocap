"""Automatic fill of one hidden range: a cache of generated ranges in front of the Kimodo worker.

Numpy and the standard library only; the worker runs in its own runtime, started by `generate`.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np

VERSION = 1
KNOWN = 5  # known frames the worker uses on each side (take_join.AUTO_KNOWN)
PREFIX = '@@MOCAP@@'
NO_MOTION = 'Automatic fill returned no usable motion.'
STOPPED = 'Automatic fill stopped unexpectedly.'


class Unavailable(RuntimeError):
    """Automatic fill cannot produce this range; the message is a sentence for the user."""


def key(request, identity):
    """Cache key of a request for a generator `identity`: every array's name, dtype, shape and bytes, in sorted key order."""
    h = hashlib.sha256(json.dumps([VERSION, identity]).encode())
    for name in sorted(request):
        a = np.ascontiguousarray(request[name]);h.update(json.dumps([name, a.dtype.str, list(a.shape)]).encode());h.update(a.tobytes())
    return h.hexdigest()


def _read(path, request):
    """The response stored at `path` when it is a complete answer to `request`, else None."""
    try:
        with np.load(path, allow_pickle=False) as z:r = {k: z[k] for k in z.files}
        n = int(request['frames']);context = [min(KNOWN, len(request['before_pelvis'])), min(KNOWN, len(request['after_pelvis']))];rows = n+sum(context)
        ok = r['version'].shape == () and r['version'] == 1 and np.array_equal(r['context'], context) and r['contacts'].shape == (rows, 4) and \
            r['contacts'].dtype == bool and r['end_pose_cm'].shape == () and np.isfinite(r['end_pose_cm']) and \
            all(r[k].shape == (rows, w) and r[k].dtype.kind == 'f' and np.isfinite(r[k]).all() for k, w in (('orient', 3), ('pose', 63), ('pelvis', 3)))
    except Exception:return None
    return r if ok else None


def _event(line):
    try:value = json.loads(line[len(PREFIX):]) if line.startswith(PREFIX) else None
    except ValueError:return {}
    return value if isinstance(value, dict) else {}


def _run(command, echo):
    """Run the worker; each stdout line is passed to `echo` as it arrives, except its `failed` event, which the caller reports."""
    lines = []
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
        for line in process.stdout:
            line = line.rstrip('\r\n');lines.append(line)
            if _event(line).get('type') != 'failed':echo(line)
    return process.returncode, lines


def generate(folder, request, run=None, echo=print):
    """The worker's response to `request`, served from `folder/<key>.npz` when it is there; raises Unavailable."""
    from . import kimodo_runtime
    folder = Path(folder);name = key(request, kimodo_runtime.identity());done = folder/f'{name}.npz'
    cached = _read(done, request) if done.is_file() else None
    if cached is not None:return cached
    problem = kimodo_runtime.issue()
    if problem:raise Unavailable(problem)
    asked, partial = folder/f'{name}.request.npz', folder/f'{name}.partial.npz'
    try:command = kimodo_runtime.command(['--request', asked, '--response', partial])
    except RuntimeError as error:raise Unavailable(str(error)) from None
    folder.mkdir(parents=True, exist_ok=True)
    leftovers = (asked, partial, partial.with_name(partial.name+'.tmp'))  # also those of a run that was killed
    try:
        for p in leftovers[1:]:p.unlink(missing_ok=True)
        with open(asked, 'wb') as handle:np.savez(handle, **{k: np.asarray(v) for k, v in request.items()})
        try:code, lines = (run or (lambda c: _run(c, echo)))(command)
        except OSError:raise Unavailable(STOPPED) from None
        if code != 0:
            failed = [e for e in map(_event, lines) if e.get('type') == 'failed']
            raise Unavailable(str(failed[-1].get('message') or STOPPED) if failed else STOPPED)
        response = _read(partial, request)
        if response is None:raise Unavailable(NO_MOTION)
        os.replace(partial, done)
    finally:
        for p in leftovers:p.unlink(missing_ok=True)
    return response
