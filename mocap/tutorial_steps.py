"""Tutorial chapters as data: what each step highlights, what it says and when it is done.

Every statement of a step comes from docs/USER_GUIDE.md. A step reads the window's state and never changes it.
"""
from dataclasses import dataclass

from PySide6.QtCore import QRect
from .paths import ROOT


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    text: object           # str, or callable(window) -> str
    targets: object        # callable(window) -> widgets and/or QRects in window coordinates
    done: object = None    # callable(window, events) -> bool; None: only Next moves on
    skip: object = None    # callable(window) -> bool
    waiting: str = ''      # status line while `done` is not met
    action: object = None  # callable(window) -> (label, callback) or None
    retry: object = None   # callable(window, events) -> str or None; a text sends the guide one step back with that note
    card: str = 'auto'     # 'center' keeps the card in the middle, clear of an opened menu


@dataclass(frozen=True)
class Chapter:
    key: str
    title: str
    steps: tuple
    blocked: object = None  # callable(window) -> str or None; a text holds the chapter and is shown instead of the step
    events: object = None   # callable(window) -> {name: (signal, condition callable(window) -> bool)}


SAMPLE = ROOT / 'assets/tutorial/sample.mp4'  # optional clip shipped with the app; without it the user's own video is used


def sample_action(window):
    if window.busy or not SAMPLE.is_file():return None
    return 'Use the sample clip', lambda: window.load_video(SAMPLE)


def part(window, widget, rect):
    """A rectangle inside a widget, in window coordinates."""
    return QRect(widget.mapTo(window, rect.topLeft()), rect.size())


def track_rows(window, first, count):
    """Timeline rows: 0 Video, 1 Motion, 2-3 feet, 4-5 hands."""
    slider = window.slider
    return part(window, slider, QRect(0, slider.TRACK_TOP + first * slider.TRACK_HEIGHT, slider.width(), count * slider.TRACK_HEIGHT))


def file_menu(window):
    bar = window.menuBar()
    return part(window, bar, bar.actionGeometry(bar.actions()[0]))


def video_open(window, events=None):return bool(window.video_readable and window.info is not None)


def job_started(window):return bool(window.busy)


def people_found(window, events):return 'find' in events and not window.busy and window.people_list.count() > 0


def models_ready(window, events=None):return not getattr(window, 'models_missing', False)


def job_stopped(window, events):
    if not window.busy and (window.failed or window.cancelled):
        return 'The job stopped before it finished. Open Console for the reason, then click Reconstruct 3D again.'


PEOPLE = Step('people', 'Find the people',
    'Click Find person, then check the People rows you want to reconstruct. Without it, one person is tracked automatically '
    'and the identity may switch when people overlap.',
    lambda w: [w.find_person_button, w.people_list], done=people_found,
    skip=lambda w: not w.find_person_button.isVisible(), waiting='Waiting for Find person…')

BASICS = Chapter('basics', 'From video to export', (
    Step('model', 'Choose a motion model',
         'The AI Motion Model creates the 3D motion from the video. ExMR is the starting choice: body and finger tracking, '
         'generally fast. ? Model Guide compares motion quality, finger support, speed and memory use.',
         lambda w: [w.model_choice, w.model_help]),
    Step('models', 'Prepare the model files',
         'The selected model needs its files before it can run. Open Model Manager, download or import the missing files and '
         'close the dialog. This step ends when the list here shows every required file as ready.',
         lambda w: [w.models_group], done=models_ready, skip=models_ready, waiting='Waiting for the model files…'),
    Step('video', 'Open a video',
         'Choose File → Open Video…, or drop a video file into the window. The original video is never modified.',
         lambda w: [file_menu(w)], done=video_open, waiting='Waiting for a video…', action=sample_action, card='center'),
    Step('trim', 'Trim the clip (optional)',
         'Drag the ends of the Video track to limit playback and reconstruction to a part of the clip. '
         'Reset Trim restores the full clip.',
         lambda w: [track_rows(w, 0, 1)]),
    PEOPLE,
    Step('run', 'Reconstruct 3D',
         'Click Reconstruct 3D to create 3D motion for the checked people in the trimmed range, with the selected model and '
         'settings. Completed steps are reused when they are still valid.',
         lambda w: [w.run_button], done=lambda w, e: 'run' in e, waiting='Waiting for your click…'),
    Step('wait', 'While it runs',
         'The bar and the status line show the progress. Console shows details and error messages. '
         'Cancel Job stops the queue; completed results are kept.',
         lambda w: [w.progress, w.status, w.details, w.cancel_button],
         done=lambda w, e: not w.busy and w.preview_data is not None, retry=job_stopped,
         waiting='Waiting for the reconstruction to finish…'),
    Step('review', 'Review the result',
         'Press Space or the play button and compare the video, the mesh and the Bone Overlay, the skeleton used for export. '
         'In the 3D Viewport, drag to orbit, middle or right drag to pan, wheel to zoom.',
         lambda w: [w.video_view, w.motion_view, w.play_button, w.bone_overlay], done=lambda w, e: w.timer.isActive()),
    Step('correct', 'Motion Correction',
         'Contact correction reduces sliding when feet rest on the floor. Change the settings here or edit the L foot and '
         'R foot tracks on the timeline, then click Apply Correction. It updates the preview and the export without running '
         'the AI model again.',
         lambda w: [w.correction_group, track_rows(w, 2, 2)]),
    Step('export', 'Export',
         'File → Export… saves the displayed, applied animation as FBX, BVH or a portable Blender project. File → Save Project '
         'As… keeps the video, motion and settings in a folder you choose. Help → Tutorial has this chapter again, and one '
         'about Hidden ranges.',
         lambda w: [file_menu(w)], done=lambda w, e: 'export' in e, card='center'),
), blocked=lambda w: None if w.model_choice.currentData() in ('exmr', 'flowhmr', 'gvhmr') else
    'This chapter covers ExMR, FlowHMR and GVHMR. Choose one of them in AI Motion Model to continue.',
   events=lambda w: {'run': (w.run_button.clicked, job_started), 'find': (w.find_person_button.clicked, job_started),
                     'export': (w.export_action.triggered, lambda w: True)})

CHAPTERS = {BASICS.key: BASICS}
