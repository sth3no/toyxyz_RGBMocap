"""Planning a long range of a scene in consecutive parts that share a few frames; no Qt, no file access.

Frames are scene frames: indices of the scene clip's frame grid, as take ranges are."""
import numpy as np

from .people_store import visible_runs

MIN_FRAMES = 4  # the shortest take a person job makes
# Temporary ExMR frame JPGs (exmr_frames): about 188 KB per 1920x1080 frame, measured in one local project
# (65 MB for 353 frames). Other sizes are scaled by their pixel count; the real size depends on the picture.
FRAME_BYTES = 188_000


def frames(seconds, fps):return int(round(float(seconds)*float(fps)))


def issue(part, overlap):
    """Why parts of `part` frames sharing `overlap` frames cannot be planned, or None."""
    if part < MIN_FRAMES:return f'A part must be at least {MIN_FRAMES} frames long.'
    if overlap < 1:return 'The overlap must be at least one frame.'
    if 2*overlap >= part:return 'The overlap must be shorter than half a part.'
    return None


def plan(first, end, part, overlap):
    """[(start, end)] of consecutive parts of [first, end): each `part` frames long and sharing `overlap` frames with the
    next; the last one also takes a remainder shorter than a third of a part."""
    reason = issue(part, overlap)
    if reason:raise ValueError(reason)
    if end <= first:raise ValueError('The range is empty.')
    out, start = [], first
    while True:
        stop = start+part
        if end-stop < part/3:
            out.append((start, end));return out
        out.append((start, stop));start = stop-overlap


def job_range(part, start, end, fps, frames):
    """The scene frames the person job reconstructs for the trim `part`, as input_person_jobs and the job compute them for a
    scene that starts `start` and ends `end` seconds into the video: the trim's end is clipped to the scene's end, both
    edges are rounded to frames and kept inside the scene's `frames`."""
    a, b = part
    first, last = start+a/fps, min(end, start+b/fps)
    return max(0, round((first-start)*fps)), min(frames, round((last-start)*fps))


def visible(observed, fps, part):
    """Visible runs of a person inside a part, as the person job finds them: `observed` is the person's whole-scene
    tracking mask (the person preview's tracking.npz), and each run becomes one take."""
    return visible_runs(np.flatnonzero(np.asarray(observed, bool)), fps, *part)


def finished(takes, runs, ready):
    """True when every run has a take of exactly that range whose result is ready (`ready(project)`)."""
    have = {(t['first'], t['end']): t['project'] for t in takes}
    return bool(runs) and all(run in have and ready(have[run]) for run in runs)


def marks(takes, runs):
    """{gap_fill key: mark} of the seams (shared frames, or a meeting point) between two takes that are both runs of the
    planned parts (`runs`, a set of (first, end)); Hidden ranges lists and joins only these, so overlapping takes of other
    trims stay apart. A mark names its two takes, so another take with the same seam frames does not inherit it."""
    from .hidden_ranges import parts_mark, seam_key, seams
    span = lambda t: (t['first'], t['end'])
    return {seam_key(s): parts_mark(s) for s in seams(takes) if s.get('kind') and span(s['before']) in runs and span(s['after']) in runs}


def frame_bytes(width, height):return round(FRAME_BYTES*width*height/(1920*1080))


def summary(parts, width, height, people, delete, frames=True):
    """The dialog's line: how many parts, and an estimate of the temporary frames on disk at once (deleted after each part)
    or in total (kept). `frames` False for a model that writes none."""
    count = f"{len(parts)} part{'' if len(parts) == 1 else 's'}"
    if not frames:return count+'.'
    lengths = [b-a for a, b in parts]
    size = frame_bytes(width, height)*people*(max(lengths) if delete else sum(lengths))/1e9
    return f"{count}, about {size:.2f} GB of temporary frames {'at a time' if delete else 'in total'}." if size < 1 else \
        f"{count}, about {size:.1f} GB of temporary frames {'at a time' if delete else 'in total'}."
