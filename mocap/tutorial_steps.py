"""Tutorial chapters as data: what each step highlights, what it says and when it is done.

Every statement of a step comes from docs/USER_GUIDE.md. A step reads the window's state and never changes it.
"""
from dataclasses import dataclass

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QComboBox
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


def reconstructing(window, events):
    """Reconstruct 3D clicked on its step, or a reconstruction that was started before the step came up."""
    return 'run' in events or bool(window.busy and getattr(window, '_batch_has_motion', False))


def people_found(window, events):return 'find' in events and not window.busy and window.people_list.count() > 0


def models_ready(window, events=None):return not getattr(window, 'models_missing', False)


def job_stopped(window, events):
    if not window.busy and (window.failed or window.cancelled):
        return 'The job stopped before it finished. Open Console for the reason, then click Reconstruct 3D again.'


PEOPLE = Step('people', 'Find the people',
    'Click Find person, then check the People rows you want to reconstruct. Without it, one person is tracked automatically '
    'and the identity may switch when people overlap. A checked person appears in the 3D view once reconstructed.',
    lambda w: [w.find_person_button, w.people_list], done=people_found,
    skip=lambda w: not w.find_person_button.isVisible(), waiting='Waiting for Find person…')

BASICS = Chapter('basics', 'From video to export', (
    Step('model', 'Choose a motion model',
         'The AI Motion Model creates the 3D motion from the video. ExMR is the starting choice: body and finger tracking, '
         'generally fast. ? Model Guide compares motion quality, finger support, speed and memory use.',
         lambda w: [w.model_choice, w.model_help]),
    Step('models', 'Prepare the model files',
         'The selected model needs its files before it can run. Open Model Manager and click Download All. If a download '
         'fails, use Manual Download and Import on that row. A file of an optional refinement, such as HTD Refine, is not '
         'needed while that option is off. This step ends when every required file is ready.',
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
         'Click Reconstruct 3D, at the bottom of the right panel, to create 3D motion for the checked people in the trimmed '
         'range, with the selected model and settings. Completed steps are reused when they are still valid.',
         lambda w: [w.run_button], done=reconstructing, waiting='Waiting for your click…'),
    Step('wait', 'While it runs',
         'The bar and the status line show the progress. Console shows details and error messages. '
         'Cancel Job stops the queue; completed results are kept.',
         lambda w: [w.progress, w.status, w.details, w.cancel_button],
         done=lambda w, e: not w.busy and w.preview_data is not None, retry=job_stopped,
         waiting='Waiting for the reconstruction to finish…'),
    Step('review', 'Review the result',
         'Press Space or the play button and compare the video, the mesh and the Bone Overlay, the skeleton used for export. '
         'A checked person is drawn only when they have a result for the selected model and options; the 3D view names who '
         'is missing, and Reconstruct 3D creates what is missing.',
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


def panel_shown(window, events=None):return window.hidden_ranges.isVisible()


def panel_reason(window):
    """Why the panel cannot fill; the panel decides it, so the chapter never lists the supported models itself."""
    panel = window.hidden_ranges
    return panel.reason.text() if panel.isVisible() and not panel.reason.isHidden() else ''


def hidden_ranges_usable(window, events=None):return panel_shown(window) and not panel_reason(window)


def method_chosen(window, events=None):
    return any(combo.currentData() != 'none' for combo in window.hidden_ranges.findChildren(QComboBox)
               if combo.accessibleName().startswith('Hidden range '))


def need_video_text(window):
    return ('A hidden range is the gap between two takes of the same person, for example while the person is covered by an '
            'object. Open a video from a fixed camera in which a person is fully covered for more than half a second.'
            + (' Or use the sample clip.' if sample_action(window) else ''))


def reason_text(window):
    reason = panel_reason(window)
    if not reason:return 'Hidden ranges is shown for the selected person. Select the person who gets covered again.'
    return (f'Hidden ranges says: “{reason}” Follow it; Reconstruct 3D creates the motion of the checked people. '
            'This step ends when the panel shows no reason.')


HIDDEN = Chapter('hidden_ranges', 'Hidden ranges', (
    Step('need_video', 'A video with a hidden person', need_video_text, lambda w: [file_menu(w)], done=video_open,
         skip=lambda w: panel_shown(w) or video_open(w), waiting='Waiting for a video…', action=sample_action, card='center'),
    Step('find', PEOPLE.title, PEOPLE.text, PEOPLE.targets, done=people_found,
         skip=lambda w: panel_shown(w) or PEOPLE.skip(w), waiting=PEOPLE.waiting),
    Step('gap', 'Select the covered person',
         'Check the people and select the one who gets covered. When that person has two takes with a gap between them, '
         'Hidden ranges appears under the People list. If it does not appear, this video has no hidden range for that person.',
         lambda w: [w.people_list], done=panel_shown, skip=panel_shown, waiting='Waiting for a hidden range…'),
    Step('reason', 'Make the range fillable', reason_text, lambda w: [w.hidden_ranges, w.run_button], done=hidden_ranges_usable, skip=hidden_ranges_usable,
         waiting='Waiting for the panel…'),
    Step('takes', 'Takes and hidden ranges',
         'This person has more than one take, and the selector switches between them. The gap between two consecutive takes '
         'is a hidden range. By default nothing is filled: the gap stays empty and each take is used as before.',
         lambda w: [w.person_takes]),
    Step('smooth', 'Smooth join',
         'Every range has a method. Smooth join is plain interpolation between the last frame before the gap and the first '
         'frame after it, with no model. It is available only for gaps of up to 30 frames. Choose a method for a range to go on.',
         lambda w: [w.hidden_ranges.area], done=method_chosen, waiting='Waiting for a method…'),
    Step('copy', 'Copy from person',
         'Copy from person fills the range with the motion of another person of the same scene. It does not understand '
         'choreography; it reuses what the other person did, which suits dancers who perform the same moves. Mirror swaps '
         'left and right, Offset shifts the copied motion in time.',
         lambda w: [w.hidden_ranges.area]),
    Step('automatic', 'Automatic (Experimental)',
         'Automatic generates the motion of the range from the captured poses just before and just after it, with NVIDIA '
         'Kimodo. Experimental: generated motion is plausible, not what the person did. An optional Description says what '
         'the person does. If something is missing, the row says what and Set up… opens Model Manager.',
         lambda w: [w.hidden_ranges.area, w.hidden_ranges.setup]),
    Step('fill', 'Fill hidden ranges',
         'Fill hidden ranges writes one new joined take that covers the takes and the chosen gaps. The source takes are kept '
         'unchanged. The take selector shows the joined take with “· filled”.',
         lambda w: [w.hidden_ranges.fill], done=lambda w, e: '· filled' in w.person_takes.currentText(),
         waiting='Waiting for the joined take…'),
    Step('after', 'After a fill',
         "Filled frames are marked by a thin band on the timeline. The joined take needs Apply Correction, and other people's "
         'takes may ask for Apply Correction again, because the shared scene floor includes the joined take. Export file '
         'names end with _filled.',
         lambda w: [track_rows(w, 1, 1), w.correction_button]),
    Step('same_person', 'One person found as two',
         'After an occlusion the tracker sometimes finds the same human again as a second person. Right-click the person that '
         'should disappear, choose Continue as and pick who they really are; the gap then appears in Hidden ranges. Separate '
         'again undoes it. No motion is changed and no file is moved or deleted.',
         lambda w: [w.people_list]),
), blocked=lambda w: 'Hidden ranges belong to offline reconstruction. Stop PEAR and choose GVHMR or ExMR to continue.'
                     if getattr(w, '_pear_active', False) else None,
   events=lambda w: {'find': (w.find_person_button.clicked, job_started)})

CHAPTERS = {BASICS.key: BASICS, HIDDEN.key: HIDDEN}
