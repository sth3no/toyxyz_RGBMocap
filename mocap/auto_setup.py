"""Auto Setup: Fast, Detailed or Best possible from the installed files, with a recommendation for this PC.

It only sets what the user could set by hand, with pipelines the app already has; nothing is run and no result of one
model is mixed into another's. The model order, its wording and the memory figures come from the user guide's
comparison and the Model Guide's records of our test PC; they are observations, not accuracy scores.
"""
from dataclasses import dataclass, field
import sys

MODES = {'fast': 'Fast', 'detailed': 'Detailed', 'best': 'Best possible'}
QUALITY = ('flowhmr', 'exmr', 'gvhmr')  # best overall quality first
SPEED = ('exmr', 'gvhmr', 'flowhmr')    # shortest recorded time first
ORDERS = {'fast': SPEED, 'detailed': QUALITY, 'best': QUALITY}
WHY = {'flowhmr': 'the best overall motion quality in our tests, and the slowest of the three',
       'exmr': 'generally fast, with good overall motion quality in our tests',
       'gvhmr': 'the lowest overall quality of the three in our tests, sometimes better for unusual poses'}
OTHERS = 'MAMMA, PromptHMR and GEM-X need their own kind of input and are never chosen here.'
REFINEMENTS = {'gvhmr': (('htd', 'HTD Refine'), ('footmr', 'FootMR'), ('handflow', 'HandFlow')),
               'flowhmr': (('hamer', 'HaMeR Finger Refinement'),), 'exmr': ()}
# Which mode switches a refinement on when its files are installed. Detailed keeps GVHMR's own defaults and leaves
# FlowHMR's extra finger pass to Best possible.
USES = {'fast': (), 'detailed': ('htd', 'footmr', 'handflow'), 'best': ('htd', 'footmr', 'handflow', 'hamer')}
SPEEDUPS = {'fast_sam': 'Fast SAM', 'batch': 'Adaptive Batching', 'gpu_crop': 'GPU Hand Crops', 'stride': '4-Frame Hand Sampling'}


def _records():
    from .model_guide import BENCHMARKS
    return float(BENCHMARKS['flowhmr'][2]), float(BENCHMARKS['flowhmr'][3])


FLOW_VRAM, FLOW_RAM = _records()  # recorded peaks of FlowHMR on the test PC, GiB
HAMER_VRAM = 2.13                 # what the HaMeR pass added there (docs/models/FLOWHMR.md)


def needed(model, htd=False, footmr=False, handflow=False, hamer=False):
    """The model files a reconstruction with these refinements requires, as the readiness line of the window counts them."""
    from .models import STAGE_REQUIREMENTS
    from .motion_tools import MOTION_TOOLS
    keys = {key for stage in MOTION_TOOLS[model].stages(htd, footmr, True, handflow=handflow) for key in STAGE_REQUIREMENTS.get(stage, ())}
    if hamer and model == 'flowhmr':
        from .flow_hamer import MODEL_KEYS
        keys.update(MODEL_KEYS)
    return keys


@dataclass
class Plan:
    mode: str = 'detailed'
    model: str = None
    htd: bool = False
    footmr: bool = False
    handflow: bool = False
    hamer: bool = False
    tracker: str = 'default'
    flow: dict = None                            # FlowHMR's speed-ups, when FlowHMR is the model
    used: list = field(default_factory=list)     # switched on
    spared: list = field(default_factory=list)   # installed, left off because this mode saves the time
    skipped: list = field(default_factory=list)  # wanted by this mode, not installed
    better: list = field(default_factory=list)   # labels of models earlier in this mode's order that lack files
    missing: list = field(default_factory=list)  # (label, number of missing files) when no model is ready


def plan(ready, sam31, mode='detailed'):
    """`ready(key)`: that model file is installed; `sam31`: the SAM 3.1 tracker can run."""
    from .motion_tools import MOTION_TOOLS
    order = ORDERS[mode]
    lacking = {model: [key for key in needed(model) if not ready(key)] for model in order}
    model = next((model for model in order if not lacking[model]), None)
    if model is None:return Plan(mode, missing=[(MOTION_TOOLS[m].label, len(lacking[m])) for m in QUALITY])
    result = Plan(mode, model, better=[MOTION_TOOLS[m].label for m in order[:order.index(model)]])
    for name, label in REFINEMENTS[model]:
        installed_ = all(ready(key) for key in needed(model, **{name: True}))
        wanted = name in USES[mode]
        setattr(result, name, wanted and installed_)
        (result.used if wanted and installed_ else result.skipped if wanted else result.spared if installed_ else []).append(label)
    if mode == 'best':
        result.tracker = 'sam31' if sam31 else 'default'
        (result.used if sam31 else result.skipped).append('SAM 3.1 tracker')
    elif sam31:result.spared.append('SAM 3.1 tracker')
    result.used.append('Motion Correction')
    if model == 'flowhmr':
        # HaMeR needs Fast SAM and 4-Frame Hand Sampling (the window forces both on) and replaces the fingers anyway.
        result.flow = dict(fast_sam=mode != 'best' or result.hamer, batch=mode != 'best', gpu_crop=mode != 'best',
                           stride=mode != 'best' or result.hamer)
    return result


def describe(result):
    from .motion_tools import MOTION_TOOLS
    if result.model is None:
        return ('No model is ready. Missing files: ' + ', '.join(f'{label} {count}' for label, count in result.missing)
                + '. Open Model Manager to download or import them.')
    lines = [f'{MODES[result.mode]} · {MOTION_TOOLS[result.model].label}: {WHY[result.model]}.', 'On: ' + ', '.join(result.used) + '.']
    off = [label for key, label in SPEEDUPS.items() if result.flow and not result.flow[key]]
    if off:lines.append('Off, so less is traded for speed: ' + ', '.join(off) + '.')
    if result.spared:lines.append('Left off for speed: ' + ', '.join(result.spared) + '.' + (' Best possible uses them.' if result.mode != 'best' else ''))
    if result.skipped:lines.append('Not used, not installed: ' + ', '.join(result.skipped) + '.')
    if result.better:lines.append(' and '.join(result.better) + (' come' if len(result.better) > 1 else ' comes')
                                  + ' first for this choice, but files are missing.')
    return '\n'.join(lines + [OTHERS])


def installed():
    """(ready, sam31) of this installation."""
    from .models import BY_KEY, ready
    from . import sam31_runtime
    try:sam31_runtime.location();sam31 = ready(BY_KEY['sam31_people'])
    except (RuntimeError, KeyError):sam31 = False
    return (lambda key: ready(BY_KEY[key])), sam31


def run(window, mode, ready=None, sam31=None):
    """An Auto Setup button. Sets the same controls a user would, so every choice stays visible and can be changed."""
    if window.busy or window.contact_saving:
        window.status.setText('Auto Setup waits until the current job has finished.');return
    if ready is None:ready, sam31 = installed()
    result = plan(ready, sam31, mode)
    window.auto_note.setText(describe(result));window.auto_note.show()
    if result.model is None:
        window.status.setText('Auto Setup: no model is ready. Open Model Manager.');return
    window.model_choice.setCurrentIndex(window.model_choice.findData(result.model))
    if window.model_choice.currentData() != result.model:return  # the window refused the change
    if result.model == 'gvhmr':
        for control, value in ((window.htd_refine, result.htd), (window.footmr_refine, result.footmr), (window.handflow_refine, result.handflow)):
            control.setChecked(value)
    if result.model == 'flowhmr':
        window.flow_hamer.setChecked(result.hamer)
        for control, key in ((window.fast_sam, 'fast_sam'), (window.flow_batch4, 'batch'), (window.flow_gpu_crop, 'gpu_crop'),
                             (window.flow_hand_stride4, 'stride')):
            control.setChecked(result.flow[key])
    window.motion_correction.setChecked(True)
    window.people_tracker.setCurrentIndex(window.people_tracker.findData(result.tracker))
    window.update_readiness()
    window.status.setText(f'Auto Setup ({MODES[mode]}) chose {window.model_choice.currentText()}. Open a video, then Find person and Reconstruct 3D.')


def parse_gpu(text):
    """(name, GiB of video memory) of the largest GPU in `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`."""
    found = []
    for line in text.splitlines():
        name, _, memory = line.rpartition(',')
        try:found.append((name.strip(), int(memory.strip()) / 1024))
        except ValueError:continue
    return max(found, key=lambda gpu: gpu[1]) if found else None


def recommend(vram, ram):
    """(mode or None, reason) from the recorded peaks. Every mode still runs on less: the app switches stages to Low Memory."""
    records = (f'In our records FlowHMR peaked at {FLOW_VRAM:.1f} GiB of video memory and {FLOW_RAM:.1f} GiB of RAM')
    if vram is None:return None, 'nvidia-smi reported no NVIDIA GPU, so there is no recommendation.'
    if vram < FLOW_VRAM or (ram is not None and ram < FLOW_RAM):
        return 'fast', records + '. With less, Detailed and Best possible still run, more slowly, as stages switch to Low Memory.'
    if vram < FLOW_VRAM + HAMER_VRAM:
        return 'detailed', records + (f', and HaMeR added {HAMER_VRAM:.1f} GiB. Best possible still runs here; '
                                      'its extra passes may switch stages to Low Memory.')
    return 'best', records + f', and HaMeR added {HAMER_VRAM:.1f} GiB; this PC has more.'


def hardware_line(gpu, ram):
    mode, reason = recommend(gpu[1] if gpu else None, ram)
    pc = (f'{gpu[0]}, {gpu[1]:.1f} GiB video memory' if gpu else 'no NVIDIA GPU found') + (f', {ram:.0f} GiB RAM' if ram else '')
    return f'This PC: {pc}.\n' + (f'Recommended: {MODES[mode]}. ' if mode else '') + reason


def show_hardware(window, gpu, ram):
    mode = recommend(gpu[1] if gpu else None, ram)[0]
    window.auto_hardware.setText(hardware_line(gpu, ram));window.auto_hardware.show()
    for key, button in window.auto_buttons.items():
        button.setText(button.property('label') + (' ★' if key == mode else ''))


def probe_hardware(window):
    """Ask nvidia-smi without blocking the window; app.py calls this once the window is shown."""
    from PySide6.QtCore import QProcess
    import psutil
    ram = psutil.virtual_memory().total / 2 ** 30
    if sys.platform == 'darwin':
        # macOS has no CUDA GPU or nvidia-smi; inference on Apple GPUs is a later port phase.
        show_hardware(window, None, ram)
        window.auto_hardware.setText(f'This PC: Mac, {ram:.0f} GiB unified memory.\n'
                                     'AI inference on macOS comes in a later phase: there is no CUDA GPU, so there is no recommendation.')
        return
    process = window.auto_probe = QProcess(window)
    done = []
    def finish(*_):
        if done:return  # finished and errorOccurred can both arrive
        done.append(1)
        text = bytes(process.readAllStandardOutput()).decode('utf-8', 'replace') if process.exitStatus() == QProcess.NormalExit else ''
        show_hardware(window, parse_gpu(text), ram)
    process.finished.connect(finish);process.errorOccurred.connect(finish)
    process.start('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'])
