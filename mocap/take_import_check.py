"""What can be imported as a further take of a person, and why not; standard library only, so the GUI can ask before a job starts."""
import json
from pathlib import Path

from .hidden_ranges import AUTO_MAX_FRAMES

FPS = 30
UNSUPPORTED = 'Takes cannot be imported for HTD Refine results in this version; switch HTD Refine off.'
SCENE_MODEL = 'Takes can be imported only into GVHMR and ExMR scenes.'
STATIC = 'Takes can be imported only into a scene with a fixed camera; turn on Static camera in the ExMR settings.'
LABELS = {'gvhmr': 'GVHMR', 'exmr': 'ExMR'}
MODEL = 'Only {0} takes can be imported into this {0} scene, because other models put the floor at another height.'
RATE = 'Only 30 fps takes can be imported; the chosen take has {} fps.'
NOT_FINISHED = 'The chosen take has no finished result; reconstruct it in its own project first.'
SAME = 'The chosen take is already in this scene; choose a take from another project.'
RECEIVER = '{} has no finished take to continue; reconstruct one first.'
LAST = "{}'s last take is not finished; reconstruct it first, because the imported take continues from it."
FOOTMR = 'The chosen take has no FootMR result; reconstruct it with FootMR on, or switch FootMR off here.'
FINGERS = 'The chosen take has no finger result; reconstruct it with HandFlow on, or switch HandFlow off here.'
GAP = f'The gap must be 1 to {AUTO_MAX_FRAMES} frames (up to {AUTO_MAX_FRAMES/FPS:.1f} s at {FPS} fps).'


def gap_frames(seconds, fps=FPS):
    return round(float(seconds)*fps)


def gap_issue(frames):
    """Smooth join fills 1 to 30 frames and Automatic 1 to 290; a gap no fill method accepts is refused."""
    return None if 1 <= frames <= AUTO_MAX_FRAMES else GAP


def _manifest(folder):
    try:
        state = json.loads((Path(folder)/'project.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def _model(state):return (state.get('pipeline') or {}).get('model', 'gvhmr')


def kinds(result):
    """The results an imported take must provide for `result`: its uncorrected bases; None for HTD Refine."""
    from .motion_tools import base_result, result_sources
    needed = [k for k in result_sources(base_result(result)) if not k.startswith('corrected_')]
    return None if any(k.endswith('refined') for k in needed) else needed


def finished(folder, needed=('initial',)):
    from .motion_tools import RESULTS, result_is_current
    folder = Path(folder)
    return all((folder/RESULTS[k][0]).is_file() and result_is_current(folder, k) for k in needed)


def last_take(person):
    takes = [t for t in person.get('takes') or [] if isinstance(t, dict)]
    return max(takes, key=lambda t: (t['end'], t['first'])) if takes else None


def refusal(root, person, source, frames, result):
    """The first reason the take in folder `source` cannot be imported for `person` of scene `root`, or None."""
    from .people_store import safe_child
    root, source = Path(root).resolve(), Path(source).resolve()
    scene = _manifest(root) or {}
    model = _model(scene)
    if model not in LABELS:return SCENE_MODEL
    if model == 'exmr':
        from .exmr_options import saved
        try:static = saved(root)['static_camera']
        except (OSError, ValueError, KeyError):static = False
        if not static:return STATIC
    needed = kinds(result)
    if needed is None:return UNSUPPORTED
    name = str(person.get('name') or f"Person {person.get('id')}")
    if source.is_relative_to(root):return SAME
    state = _manifest(source)
    if state is None:return NOT_FINISHED
    # ExMR stands its world on the floor; GVHMR starts the pelvis at height 0: a take keeps its own height, so only one model fits.
    if _model(state) != model:return MODEL.format(LABELS[model])
    if not finished(source):return NOT_FINISHED
    if state.get('fps') != FPS or scene.get('fps', FPS) != FPS:return RATE.format(_rate(state.get('fps')))
    for kind in needed:
        if not finished(source, [kind]):return FINGERS if kind.startswith('handflow_') else FOOTMR
    last = last_take(person)
    try:folders = [safe_child(root, t['project']) for t in person.get('takes') or []]
    except (ValueError, KeyError, TypeError):folders = []
    if last is None or not any(finished(f) for f in folders):return RECEIVER.format(name)
    if not finished(safe_child(root, last['project']), needed):return LAST.format(name)
    return gap_issue(frames)


def _rate(value):
    return int(value) if isinstance(value, float) and value.is_integer() else value


def scene_roots(folder):
    """Scenes with people inside a chosen project folder: the folder itself, a model folder of a workspace, or one of its clips."""
    folder = Path(folder)
    found = [*folder.glob('people.json'), *folder.glob('*/people.json'), *folder.glob('*/clips/*/people.json')]
    return list(dict.fromkeys(p.parent.resolve() for p in found if p.is_file()))


def sources(folder):
    """Finished takes and current joined takes of every person of the scenes in `folder`, as dicts for the import dialog."""
    from .motion_tools import MOTION_TOOLS
    from .people_store import active_takes, load_people, safe_child
    out = []
    for root in scene_roots(folder):
        try:state = load_people(root)
        except (OSError, ValueError, KeyError, TypeError):continue
        if not state:continue
        for person in state['people']:
            name = str(person.get('name') or f"Person {person['id']}")
            try:takes = person.get('takes', [])+[t for t in active_takes(root, person) if t.get('join')]
            except (OSError, ValueError, KeyError, TypeError):continue
            for take in takes:
                try:
                    project = safe_child(root, take['project']);manifest = _manifest(project)
                    if manifest is None or not finished(project):continue
                except (OSError, ValueError, KeyError, TypeError):continue
                model = _model(manifest);fps = manifest.get('fps') or FPS;frames = manifest.get('frames', take['end']-take['first'])
                label = MOTION_TOOLS[model].label if model in MOTION_TOOLS else model
                seconds = frames/fps
                out.append(dict(root=root, person=name, project=project, model=model, frames=frames, fps=fps, seconds=seconds,
                                label=f"{name} · frames {take['first']+1}–{take['end']}{' · filled' if take.get('join') else ''} · {label} · {seconds:.1f} s"))
    return out
