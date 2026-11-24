"""Headless entry for the RGB MoCap workflow commands."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .cli_arguments import camera_refinement_flag, option_parsers
from .motion_tools import MOTION_TOOLS, RESULTS
from .paths import ROOT

COMMAND_SCHEMA = 1
OFFLINE_MODELS = tuple(key for key, tool in MOTION_TOOLS.items() if not tool.live)


def correction_flags(parser):
    parser.add_argument('--preset', choices=('weak', 'normal', 'strong'))
    parser.add_argument('--mode', choices=('flat', 'multi_level'))
    floor = parser.add_mutually_exclusive_group()
    floor.add_argument('--floor', type=float, help='Manual floor in metres.')
    floor.add_argument('--auto-floor', action='store_true')
    parser.add_argument('--pelvis-offset', type=float)
    for option in ('--foot-lock', '--hand-lock'):
        parser.add_argument(option, action=argparse.BooleanOptionalAction)
    parser.add_argument('--compute', choices=('auto', 'low_memory'))


def take_flags(parser):
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--person', type=int, action='append', help='Include this person ID. Repeat for more people.')
    selection.add_argument('--take', action='append', help='Include this saved take ID from project inspect.')
    selection.add_argument('--all-takes', action='store_true')
    parser.add_argument('--result', choices=tuple(RESULTS))


def parser():
    root = argparse.ArgumentParser(prog='RGBMoCapCLI', description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    rec = commands.add_parser('reconstruct', help='Prepare and reconstruct a video.')
    rec.add_argument('video', type=Path)
    rec.add_argument('--model', default='exmr', choices=OFFLINE_MODELS)
    rec.add_argument('--project', type=Path, help='Destination or existing workspace for this video.')
    rec.add_argument('--start', type=float)
    rec.add_argument('--end', type=float)
    rec.add_argument('--camera', choices=('static', 'moving'))
    rec.add_argument('--smoothing', action=argparse.BooleanOptionalAction)
    rec.add_argument('--correction', action=argparse.BooleanOptionalAction)
    rec.add_argument('--rife', action=argparse.BooleanOptionalAction)
    rec.add_argument('--find-person', action='store_true', help='Run the Default person tracker and use its checked people.')
    rec.add_argument('--person', type=int, action='append', help='Select a found person ID. Repeat to select more people.')
    rec.add_argument('--find-only', action='store_true', help='Save the found people without reconstructing.')
    rec.add_argument('--tracker', choices=('default', 'sam31'), default='default')
    rec.add_argument('--sam31-prompt', help='Text prompt for SAM 3.1.')
    rec.add_argument('--sam31-time', type=float, help='Marker time in video seconds.')
    rec.add_argument('--sam31-point', type=float, nargs=2, action='append', metavar=('X','Y'))
    rec.add_argument('--people', type=int, help='Expected count (0 Auto) for PromptHMR or MAMMA.')
    rec.add_argument('--interaction', action=argparse.BooleanOptionalAction)
    rec.add_argument('--camera-video', action='append', type=Path, default=[], help='Additional synchronized MAMMA camera video.')
    rec.add_argument('--calibration', type=Path, help='MAMMA manual camera calibration.')
    rec.add_argument('--height', type=float, help='MAMMA reference height in metres.')
    rec.add_argument('--active-camera', type=int, help='Initial MAMMA camera number (1-based).')
    rec.add_argument('--camera-interval', type=int, choices=(-1, 0, 1, 2, 5, 10, 30),
                     help='FlowHMR camera sampling interval in frames (-1 static, 0 auto).')
    for option in ('--fast-sam', '--hamer', '--htd', '--footmr', '--handflow'):
        rec.add_argument(option, action=argparse.BooleanOptionalAction)
    camera_refinement_flag(rec, 'Save the background camera refinement preference (FlowHMR / ExMR).')
    correction_flags(rec)

    correct = commands.add_parser('correct', help='Update the selected take correction.')
    correct.add_argument('project', type=Path)
    correct.add_argument('--model', default='exmr', choices=OFFLINE_MODELS)
    take_flags(correct)
    correction_flags(correct)

    exp = commands.add_parser('export', help='Export the selected committed motion.')
    exp.add_argument('project', type=Path)
    exp.add_argument('--model', default='exmr', choices=OFFLINE_MODELS)
    take_flags(exp)
    exp.add_argument('--format', default='fbx', choices=('fbx','bvh','blend'))
    exp.add_argument('--fbx-profile', default='generic', choices=('generic', 'ue5_mannequin'))
    exp.add_argument('--ue5-target', choices=('manny', 'quinn'))
    for option in ('--in-place', '--include-mesh', '--fbx-compatibility', '--prepend-tpose',
                   '--place-at-origin', '--symmetric-rest', '--overwrite'):
        exp.add_argument(option, action='store_true')
    exp.add_argument('--rest-pose', choices=('t', 'a'))
    exp.add_argument('--bvh-profile', choices=('generic','biped'))
    exp.add_argument('--combined', action='store_true', help='Put selected Blender takes on one timeline.')
    exp.add_argument('--whole-scene', action='store_true', help='MAMMA native Blender scene.')
    exp.add_argument('--face', choices=('txt','livelink_csv'))
    exp.add_argument('--raw-face', action='store_true')
    exp.add_argument('--no-camera-animation', action='store_true')
    camera_refinement_flag(exp, 'Override background refinement for the animated Blender camera (FlowHMR / ExMR).')
    exp.add_argument('--unreal-profile', type=Path, help='Import the exported FBX files using this Unreal profile.')
    outputs = exp.add_mutually_exclusive_group(required=True)
    outputs.add_argument('--output', type=Path)
    outputs.add_argument('--output-dir', type=Path)

    project = commands.add_parser('project', help='Read saved project information.')
    project_actions = project.add_subparsers(dest='project_action', required=True)
    copy = project_actions.add_parser('copy', help='Save an independent project copy.')
    copy.add_argument('project', type=Path)
    copy.add_argument('--model', choices=OFFLINE_MODELS)
    copy.add_argument('--output', type=Path, required=True)
    copy.add_argument('--dry-run', action='store_true')
    inspect = project_actions.add_parser('inspect')
    inspect.add_argument('project', type=Path)
    inspect.add_argument('--model', default='exmr', choices=OFFLINE_MODELS)
    take_flags(inspect)

    opn = commands.add_parser('open', help='Open the desktop app with an optional video or project.')
    opn.add_argument('path', type=Path, nargs='?')
    check = commands.add_parser('check', help='Check offline model files.')
    check.add_argument('--model', choices=OFFLINE_MODELS, default='exmr')
    unreal = commands.add_parser('unreal', help='Save an Unreal import profile or import animation FBX files.')
    unreal_actions = unreal.add_subparsers(dest='unreal_action', required=True)
    profile = unreal_actions.add_parser('profile', help='Save an Unreal animation import profile.')
    profile.add_argument('--editor', type=Path, required=True)
    profile.add_argument('--project', type=Path, required=True)
    profile.add_argument('--mesh', required=True)
    profile.add_argument('--destination', required=True)
    profile.add_argument('--fbx-profile', choices=('generic', 'ue5_mannequin'), default='generic')
    profile.add_argument('--ue5-target', choices=('manny', 'quinn'))
    profile.add_argument('--output', type=Path, required=True)
    unreal_import = unreal_actions.add_parser('import', help='Check and import animation FBX files.')
    unreal_import.add_argument('--profile', type=Path, required=True)
    unreal_import.add_argument('fbx', type=Path, nargs='+')
    unreal_import.add_argument('--dry-run', action='store_true', help='Check compatibility without importing assets.')
    jobs = commands.add_parser('jobs', help='Read and manage local workflow jobs.')
    actions = jobs.add_subparsers(dest='job_action', required=True)
    listing = actions.add_parser('list', help='List active and recent workflow jobs.')
    listing.add_argument('--limit', type=int, default=50)
    for action in ('status', 'cancel'):
        actions.add_parser(action).add_argument('job_id')
    for action in actions.choices.values():
        action.add_argument('--json', action='store_true', help='Print one JSON result to stdout.')
    option_parsers(commands, OFFLINE_MODELS, take_flags)
    for sub in (rec, correct, exp, inspect, copy, opn, check, profile, unreal_import):
        sub.add_argument('--json', action='store_true', help='Print one JSON result to stdout.')
    for sub in (rec, correct, exp):
        sub.add_argument('--dry-run', action='store_true', help='Inspect the plan without writing files.')
    return root


def open_app(path):
    from tools.session_lock import app_lock
    with app_lock(ROOT):
        pass  # The launched desktop process acquires and holds this itself.
    arguments = []
    if path is not None:
        path = path.resolve(strict=True)
        if path.is_dir() and (path / 'project.json').is_file():
            arguments = ['--project', str(path)]
        elif path.is_file():
            flag = '--project' if path.name == 'project.json' else '--video'
            arguments = [flag, str(path.parent if flag == '--project' else path)]
        else:
            raise ValueError('Choose a video or a project containing project.json.')
    command = [str(ROOT / 'runtime/python.exe'), '-I', '-X', 'utf8', '-u', '-B',
               str(ROOT / 'launch_source.py'), *arguments]
    subprocess.Popen(command, cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE, close_fds=True)
    return {'launch_requested': True, 'path': str(path) if path is not None else None}


def format_output(result, machine):
    print(json.dumps(result, ensure_ascii=False, indent=None if machine else 2))


def writes_installation(args):
    read_only = (args.command in {'open', 'check', 'jobs'}
                 or args.command == 'project' and args.project_action == 'inspect'
                 or args.command == 'models' and args.action == 'list'
                 or args.command == 'face' and args.action == 'inspect'
                 or args.command == 'pear' and args.list_devices
                 or args.command == 'contacts' and args.action in {'list', 'self-list'})
    return not read_only and not getattr(args, 'dry_run', False)


class JobReport:
    def __init__(self, jobs, job_id):
        self.jobs, self.job_id = jobs, job_id

    def __call__(self, message):
        print(message, file=sys.stderr, flush=True)

    def set_phase(self, phase, **fields):
        self.jobs.phase(self.job_id, phase, **fields)

    def check_owner(self):
        if not self.jobs.owned(self.job_id):
            raise KeyboardInterrupt


@contextmanager
def wait_for_installation(report):
    from tools.session_lock import AppBusyError, app_lock
    report.set_phase('waiting_for_installation', message='Waiting for RGB MoCap to release the installation lock.')
    with ExitStack() as stack:
        while True:
            report.check_owner()
            try:
                stack.enter_context(app_lock(ROOT))
            except AppBusyError:
                time.sleep(.5)
            else:
                break
        yield


def dispatch(args, report, managed):
    from tools.session_lock import app_lock
    from tools.source_launch import verify

    lock = nullcontext()
    if writes_installation(args):
        lock = wait_for_installation(report) if managed else app_lock(ROOT)
    with lock:
        if managed:
            report.check_owner()
        report.set_phase(args.command)
        verify(ROOT)
        from .runtime_registry import activate
        activate(ROOT, 'gui')
        from .runtime_limits import configure_preview_cpu
        configure_preview_cpu()
        if args.command == 'open':
            return open_app(args.path)
        from .cli_workflows import execute
        return execute(args, report)


def main(argv=None):
    from sqlite3 import Error as DatabaseError
    from .job_queue import Jobs, failure_reason

    args = parser().parse_args(argv)
    jobs = Jobs()
    job_id = None
    managed_id = os.environ.pop('RGBMOCAP_MCP_JOB_ID', None)
    code, result = 0, {}
    try:
        if args.command == 'jobs':
            if args.job_action == 'list':
                result = {'jobs': jobs.list(args.limit)}
            else:
                action = jobs.get if args.job_action == 'status' else jobs.cancel
                result = {'job': action(args.job_id)}
        else:
            if managed_id:
                job_id = managed_id
            else:
                project = getattr(args, 'project', None) or getattr(args, 'video', None)
                job_id = jobs.add('cli', args.command, project=project.resolve() if project else None)
            result = dispatch(args, JobReport(jobs, job_id), bool(managed_id))
            if result.get('failed'):
                code = 1
    except KeyboardInterrupt:
        code, result = 130, {'error': 'Cancelled.'}
    except (FileNotFoundError, ValueError) as error:
        code, result = 2, {'error': str(error)}
    except (DatabaseError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        code, result = 1, {'error': str(error)}
    if job_id and not managed_id:
        state = 'cancelled' if code == 130 else 'failed' if code else 'done'
        try:
            jobs.finish(job_id, state, result=result, error=failure_reason(result) if code else None, returncode=code)
        except (DatabaseError, OSError) as error:
            print(f'Could not save the job result: {error}', file=sys.stderr)
        result = {'job_id': job_id, **result}
    if args.json or not code:
        format_output({'schema_version': COMMAND_SCHEMA, 'ok': not code, **result}, args.json)
    else:
        print(result.get('error', json.dumps(result, ensure_ascii=False)), file=sys.stderr)
    return code
