"""What a model needs in memory, what to tell the user when it is short, and the longest part of a long range that fits the
GPU and the disk. No Qt and no Torch: the window calls begin, tick and end; the rest is plain arithmetic.

A model takes a fixed amount of RAM whatever the clip length (measured 2026-11-24, docs/VALIDATION.md): ExMR's working set
peaks near 12 GiB for a second or two while Sapiens2 loads and stays under 7.3 GiB afterwards. Shorter parts do not lower it.
Clip length matters only for the last ExMR step, which holds every frame on the GPU, and for the temporary frames on disk."""
import math

from .compute_policy import GIB, gpu_reserve

# Starting numbers in GiB: (peak working set, peak private commit) of the worker during Reconstruct 3D. ExMR was measured
# here; GVHMR with its add-ons and FlowHMR are the 60-frame process-tree peaks of docs/PERFORMANCE.md, their commit unknown.
# A larger peak that a finished reconstruction reached on this PC replaces them (see remember).
NEEDS = {'exmr': (12.0, 14.7), 'gvhmr': (6.9, None), 'flowhmr': (8.4, None)}
SEEN = 'memory_seen'  # in config/local.json: {model: {'ram': bytes, 'commit': bytes}}
SMALLEST = GIB  # a batch that peaked lower reused cached results and says nothing about the model
STAGES = frozenset(('person', 'track', 'pva', 'features', 'gvhmr', 'refine', 'footmr', 'handflow', 'flow_camera', 'flow_features',
                    'flowhmr', 'exmr_preprocess', 'exmr_features', 'exmr'))  # a batch with one of these loads motion models
TIGHT = "Memory is tight, so Windows is moving other programs' memory to disk. This job keeps running and can be slower."

# The last ExMR step, measured to 36,000 frames (it failed at 54,000 with 14 GiB of video memory free): reserved PyTorch
# video memory is about LAST_FIXED + LAST_SQUARE x frames x frames GiB.
LAST_FIXED, LAST_SQUARE = .85, 9.9e-9
VRAM_SHARE = .7  # of the free video memory budget, for the last step
DISK_SHARE = .25  # of the free disk, for one part's temporary frames
LONGEST, SHORTEST, STEP, DEFAULT = 300., 10., 10., 60.  # seconds
LIMITS = {'time': 'at most 5 minutes, so a stop costs one part at most',
          'video memory': 'the longest that fits the free video memory now',
          'disk': 'the longest whose temporary frames fit a quarter of the free disk',
          'default': 'the default for this model; no limit was measured for it'}


def _bytes(record, key):
    value = record.get(key) if isinstance(record, dict) else None
    return value if type(value) in (int, float) and value >= SMALLEST else None


def need(model, seen):
    """(RAM bytes, commit bytes or None) a reconstruction with `model` needs, or None when nothing is known: the starting
    numbers, raised by the largest peak seen on this PC."""
    start = NEEDS.get(model);record = seen.get(model) if isinstance(seen, dict) else None
    ram, commit = _bytes(record, 'ram'), _bytes(record, 'commit')
    if start is None:return (ram, commit) if ram else None
    first = start[0]*GIB;second = start[1]*GIB if start[1] else None
    return max(first, ram or 0), max(second or 0, commit or 0) or None


def seen(root=None):
    """The peaks kept in this installation's local settings; {} when there are none or the file cannot be read."""
    from .resource_safety import path, read_settings
    try:value = read_settings(path(root)).get(SEEN, {})
    except (OSError, ValueError):return {}
    return value if isinstance(value, dict) else {}


def remember(model, ram, commit, root=None):
    """Keeps the largest working set and commit a finished reconstruction with `model` reached. Local to this installation,
    never in a project or an inference identity; a failure to save is not the job's failure."""
    if not model or ram < SMALLEST:return
    from .resource_safety import update
    try:
        records = dict(seen(root));old = records.get(model) if isinstance(records.get(model), dict) else {}
        new = dict(ram=max(int(ram), int(_bytes(old, 'ram') or 0)), commit=max(int(commit), int(_bytes(old, 'commit') or 0)))
        if new == old:return
        records[model] = new;update({SEEN: records}, root)
    except (OSError, ValueError, TypeError):pass


def _with(text, note):return f'{text} {note}' if note else text


def start_note(label, needs, available, commit_room, note):
    """The sentence before a reconstruction starts, or None: Windows cannot allocate what the model needs (the job can stop),
    or less RAM is free than it needs (the job can be slower). `note` is memory_note's sentence about the PC."""
    if not needs:return None
    ram, commit = needs
    if commit and commit_room is not None and commit_room < commit+GIB:  # the guard stops under 1 GiB of headroom
        return _with(f'{label} needs Windows to allocate about {commit/GIB:.1f} GiB, and Windows can allocate only {commit_room/GIB:.1f} GiB '
                     'more, so this job can stop for memory. Close other programs or let Windows manage a larger page file.', note)
    if available < ram:
        return _with(f"{label} needs up to about {ram/GIB:.1f} GiB of RAM, more than is free, so Windows will move other programs' memory "
                     'to disk and this job can run slower.', note)
    return None


def tight_note(note):return _with(TIGHT, note)


def begin(w, jobs):
    """A batch starts: forget the last batch's peaks and say in the status bar and Console when memory is short for it."""
    w._memory_peak = [0, 0];w._memory_tight_told = False
    try:
        model = w.model_choice.currentData()
        if not {stage for stage, _ in jobs} & STAGES or not w.resource_safety.isChecked():return
        needs = need(model, seen())
        if not needs:return
        import psutil
        from . import memory_note
        from .motion_tools import MOTION_TOOLS
        from .processing_guard import commit_headroom
        memory = psutil.virtual_memory();room = commit_headroom()
        if start_note('', needs, memory.available, room, '') is None:return
        text = start_note(MOTION_TOOLS[model].label, needs, memory.available, room, memory_note.collect())
        w.status.setText(text);w.log.append(text)
    except Exception:pass  # a note must never keep a job from starting


def tick(w):
    """Every guard check: keep the batch's peaks and say once per batch that memory became tight."""
    guard = w.processing_guard
    if guard is None:return
    peak = getattr(w, '_memory_peak', None)
    if peak is None:peak = w._memory_peak = [0, 0]
    peak[0] = max(peak[0], getattr(guard, 'peak_ram', 0));peak[1] = max(peak[1], getattr(guard, 'peak_commit', 0))
    if getattr(guard, 'tight', False) and not getattr(w, '_memory_tight_told', False):
        w._memory_tight_told = True
        from . import memory_note
        text = tight_note(memory_note.collect())
        w.status.setText(text);w.log.append(text)


def end(w):
    """A batch ended: a finished reconstruction's peaks become what this PC knows the model needs."""
    peak = getattr(w, '_memory_peak', None);w._memory_peak = None
    if not peak or w.failed or w.cancelled or not getattr(w, '_batch_has_motion', False):return
    remember(w.model_choice.currentData(), *peak)


def free_vram():
    """(free, total) bytes of the first NVIDIA GPU as nvidia-smi reports them, or (None, None)."""
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
        free, total = (int(value)*1024**2 for value in out.strip().splitlines()[0].split(','))
        return free, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):return None, None


def last_step_bytes(frames):
    """Reserved video memory of the last ExMR step for a take of `frames` frames."""
    return (LAST_FIXED+LAST_SQUARE*frames*frames)*GIB


def part_frames(model, fps, width, height, people, free_vram=None, total_vram=None, free_disk=None):
    """(frames, limit) of the longest part of a long range that fits now; limit is a key of LIMITS. An unknown video memory
    or disk does not limit."""
    seconds, limit = part_seconds(model, fps, width, height, people, free_vram, total_vram, free_disk)
    return round(seconds*fps), limit


def part_seconds(model, fps, width, height, people, free_vram=None, total_vram=None, free_disk=None):
    """part_frames in whole tens of seconds, between 10 seconds and 5 minutes."""
    if model not in ('exmr', 'gvhmr'):return DEFAULT, 'default'
    frames, limit = LONGEST*fps, 'time'
    if free_vram is not None and total_vram:
        if model == 'exmr':
            budget = VRAM_SHARE*max(0, free_vram-gpu_reserve(total_vram))/GIB
            fits = math.sqrt(max(0., budget-LAST_FIXED)/LAST_SQUARE)
        else:
            from .compute_policy import gvhmr_plan
            low, high = 0, int(frames)  # the estimate grows with the frames, so the longest that fits is found by halving
            while low < high:
                middle = (low+high+1)//2
                if gvhmr_plan(middle, free_vram, total_vram)['fits']:low = middle
                else:high = middle-1
            fits = low
        if fits < frames:frames, limit = fits, 'video memory'
    if model == 'exmr' and free_disk is not None:  # only ExMR writes temporary frames
        from .parts import frame_bytes
        fits = free_disk*DISK_SHARE/(frame_bytes(width, height)*max(1, people))
        if fits < frames:frames, limit = fits, 'disk'
    return max(SHORTEST, math.floor(frames/fps/STEP)*STEP), limit
