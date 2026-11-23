"""Auto Setup: select the best installed model and switch on every refinement its files allow.

It only sets what the user could set by hand, with pipelines the app already has; nothing is run and no result of one
model is mixed into another's. The order and its wording come from the user guide's comparison of our test videos.
"""
from dataclasses import dataclass, field

ORDER = ('flowhmr', 'exmr', 'gvhmr')
WHY = {'flowhmr': 'the best overall motion quality in our tests, and the slowest of the three',
       'exmr': 'generally fast, with good overall motion quality in our tests',
       'gvhmr': 'the lowest overall quality of the three in our tests, sometimes better for unusual poses'}
OTHERS = 'MAMMA, PromptHMR and GEM-X need their own kind of input and are never chosen here.'


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
    model: str = None
    htd: bool = False
    footmr: bool = False
    handflow: bool = False
    hamer: bool = False
    tracker: str = 'default'
    used: list = field(default_factory=list)     # refinements switched on
    skipped: list = field(default_factory=list)  # refinements of this model that are not installed
    better: list = field(default_factory=list)   # labels of models that rank higher but lack files
    missing: list = field(default_factory=list)  # (label, number of missing files) when no model is ready


REFINEMENTS = {'gvhmr': (('htd', 'HTD Refine'), ('footmr', 'FootMR'), ('handflow', 'HandFlow')),
               'flowhmr': (('hamer', 'HaMeR Finger Refinement'),), 'exmr': ()}


def plan(ready, sam31):
    """`ready(key)`: that model file is installed; `sam31`: the SAM 3.1 tracker can run."""
    from .motion_tools import MOTION_TOOLS
    lacking = {model: sorted(key for key in needed(model) if not ready(key)) for model in ORDER}
    model = next((model for model in ORDER if not lacking[model]), None)
    if model is None:return Plan(missing=[(MOTION_TOOLS[m].label, len(lacking[m])) for m in ORDER])
    result = Plan(model, better=[MOTION_TOOLS[m].label for m in ORDER[:ORDER.index(model)]])
    for name, label in REFINEMENTS[model]:
        on = all(ready(key) for key in needed(model, **{name: True}))
        setattr(result, name, on);(result.used if on else result.skipped).append(label)
    result.tracker = 'sam31' if sam31 else 'default'
    (result.used if sam31 else result.skipped).append('SAM 3.1 tracker')
    result.used.append('Motion Correction')
    return result


def describe(result):
    from .motion_tools import MOTION_TOOLS
    if result.model is None:
        return ('No model is ready. Missing files: ' + ', '.join(f'{label} {count}' for label, count in result.missing)
                + '. Open Model Manager to download or import them.')
    lines = [f'{MOTION_TOOLS[result.model].label}: {WHY[result.model]}.', 'On: ' + ', '.join(result.used) + '.']
    if result.skipped:lines.append('Not used, not installed: ' + ', '.join(result.skipped) + '.')
    if result.better:lines.append(' and '.join(result.better) + (' rank' if len(result.better) > 1 else ' ranks')
                                  + ' higher in our tests, but files are missing.')
    return '\n'.join(lines + [OTHERS])


def installed():
    """(ready, sam31) of this installation."""
    from .models import BY_KEY, ready
    from . import sam31_runtime
    try:sam31_runtime.location();sam31 = ready(BY_KEY['sam31_people'])
    except (RuntimeError, KeyError):sam31 = False
    return (lambda key: ready(BY_KEY[key])), sam31


def run(window, ready=None, sam31=None):
    """The Auto Setup button. Sets the same controls a user would, so every choice stays visible and can be changed."""
    if window.busy or window.contact_saving:
        window.status.setText('Auto Setup waits until the current job has finished.');return
    if ready is None:ready, sam31 = installed()
    result = plan(ready, sam31)
    window.auto_note.setText(describe(result));window.auto_note.show()
    if result.model is None:
        window.status.setText('Auto Setup: no model is ready. Open Model Manager.');return
    window.model_choice.setCurrentIndex(window.model_choice.findData(result.model))
    if window.model_choice.currentData() != result.model:return  # the window refused the change
    if result.model == 'gvhmr':
        for control, value in ((window.htd_refine, result.htd), (window.footmr_refine, result.footmr), (window.handflow_refine, result.handflow)):
            control.setChecked(value)
    if result.model == 'flowhmr':window.flow_hamer.setChecked(result.hamer)
    window.motion_correction.setChecked(True)
    window.people_tracker.setCurrentIndex(window.people_tracker.findData(result.tracker))
    window.update_readiness()
    window.status.setText(f'Auto Setup chose {window.model_choice.currentText()}. Open a video, then Find person and Reconstruct 3D.')
