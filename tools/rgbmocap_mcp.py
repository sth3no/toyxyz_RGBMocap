"""Local stdio MCP server for RGB MoCap."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
from sqlite3 import Error as DatabaseError
import sys
import tempfile
from threading import Event, Lock, Thread
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.cli import parser, writes_installation
from mocap.job_queue import Jobs, failure_reason
from mocap.mcp_cli import commands


VERSIONS = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')
REQUEST_BYTES = 1024 * 1024
LOG_BYTES = 16 * 1024
PYTHON = [sys.executable, '-I', '-X', 'utf8', '-u', '-B']


def tool(name, description, properties=None, required=(), *, read=False):
    return dict(name=name, description=description,
                inputSchema=dict(type='object', properties=properties or {}, required=list(required),
                                 additionalProperties=False),
                annotations=dict(readOnlyHint=read, destructiveHint=not read, openWorldHint=False))


TEXT = {'type': 'string'}
JOB = {'job_id': TEXT}
CONTROLS = (
    tool('job_list', 'List active and recent jobs from the desktop, CLI and MCP.',
         {'limit': {'type': 'integer', 'minimum': 1, 'maximum': 200}}, read=True),
    tool('job_status', 'Read the phase, result and error of a local job.', JOB, JOB, read=True),
    tool('job_cancel', 'Request cancellation of a desktop or MCP workflow job.', JOB, JOB),
    tool('desktop_start', 'Start a managed RGB MoCap window controlled by this MCP session.',
         {'path': TEXT}),
    tool('desktop_stop', 'Close the window started by this MCP session. Normal close prompts still apply.'),
    tool('desktop_state', 'Read the project, selected result, frame and status of the MCP window.', read=True),
    tool('desktop_controls', 'List visible desktop controls, menu commands and modal dialog choices.', read=True),
    tool('desktop_invoke', 'Click or set a control returned by desktop_controls. For choices supply the listed value.',
         {'control_id': TEXT, 'value': {'description': 'Text, numeric or boolean value for an editable control.'}},
         ('control_id',)),
    tool('desktop_seek', 'Seek the MCP window to the given 1-based timeline frame.',
         {'frame': {'type': 'integer', 'minimum': 1}}, ('frame',)),
    tool('desktop_capture', 'Capture the RGB MoCap workspace as a PNG.', read=True),
    tool('desktop_pointer', 'Click a coordinate in the captured RGB MoCap window or modal dialog.',
         {'x': {'type': 'integer'}, 'y': {'type': 'integer'},
          'button': {'type': 'string', 'enum': ['left', 'right']},
          'clicks': {'type': 'integer', 'enum': [1, 2]}}, ('x', 'y')),
    tool('desktop_drag', 'Drag between two coordinates in the RGB MoCap window.',
         {key: {'type': 'integer'} for key in ('x', 'y', 'to_x', 'to_y')},
         ('x', 'y', 'to_x', 'to_y')),
    tool('desktop_key', 'Send a shortcut to the RGB MoCap window, for example Space or Ctrl+S.',
         {'shortcut': TEXT}, ('shortcut',)),
)


def stop_tree(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8, check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if process.poll() is None:
        process.terminate()


@dataclass
class Job:
    argv: list
    stdout: Path
    stderr: Path
    queued: bool
    process: subprocess.Popen | None = None
    stopped: bool = False

    def start(self, job_id):
        with self.stdout.open('wb') as output, self.stderr.open('wb') as log:
            self.process = subprocess.Popen([*PYTHON, str(ROOT / 'cli_boot.py'), *self.argv], cwd=ROOT,
                                            env=dict(os.environ, RGBMOCAP_MCP_JOB_ID=job_id),
                                            stdin=subprocess.DEVNULL, stdout=output, stderr=log,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def log(self):
        if not self.stderr.is_file():
            return ''
        with self.stderr.open('rb') as stream:
            stream.seek(max(0, self.stderr.stat().st_size - LOG_BYTES))
            return stream.read(LOG_BYTES).decode('utf-8', errors='replace')

    def finish(self, jobs, job_id, code):
        with self.stdout.open('rb') as output:
            payload = output.read(REQUEST_BYTES + 1)
        result, error = None, None
        if len(payload) > REQUEST_BYTES:
            error = 'CLI result exceeds the MCP response limit.'
        elif payload:
            try:
                result = json.loads(payload)
            except (UnicodeError, ValueError):
                result = {'output': payload.decode('utf-8', errors='replace')}
        log = self.log()
        if self.stopped or code == 130:
            state, error = 'cancelled', 'Cancelled.'
        elif code:
            state = 'failed'
            error = error or (failure_reason(result) if result and 'output' not in result
                              else log.strip() or f'CLI exited with code {code}.')
        else:
            state = 'done'
        jobs.finish(job_id, state, result=result, error=error, returncode=code, log=log)

    def discard_logs(self):
        self.stdout.unlink(missing_ok=True)
        self.stderr.unlink(missing_ok=True)


class Desktop:
    def __init__(self, path):
        args = [*PYTHON, str(ROOT / 'launch_source.py'), '--mcp-stdio']
        if path is not None:
            source = Path(path).resolve(strict=True)
            if source.is_dir():
                source = source / 'project.json'
            if not source.is_file():
                raise ValueError('Choose an existing video or project.json.')
            args += ['--project', str(source.parent)] if source.name == 'project.json' else ['--video', str(source)]
        self.responses = Queue()
        self.process = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        Thread(target=self.read, daemon=True).start()

    def read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except (OSError, ValueError):
            pass
        finally:
            self.responses.put(None)

    def call(self, action, args, *, timeout=15):
        if self.process.poll() is not None:
            raise RuntimeError('RGB MoCap desktop session ended. Check the MCP log.')
        request = dict(id=uuid4().hex, action=action, arguments=args)
        self.process.stdin.write(json.dumps(request, ensure_ascii=False).encode('utf-8') + b'\n')
        self.process.stdin.flush()
        try:
            while True:
                response = self.responses.get(timeout=timeout)
                if response is None:
                    raise RuntimeError('RGB MoCap desktop disconnected. Check the MCP log.')
                if response['id'] == request['id']:
                    break
        except Empty:
            raise TimeoutError('RGB MoCap desktop did not respond.') from None
        if 'error' in response:
            raise ValueError(response['error'])
        return response['result']

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Leave the GUI running if its normal close flow needs attention.
            pass


class Toolkit:
    def __init__(self, folder, *, jobs=None):
        self.folder = Path(folder)
        self.parser = parser()
        self.commands = {command.name: command for command in commands(self.parser) if command.path[0] != 'jobs'}
        self.registry = jobs if jobs is not None else Jobs(ROOT)
        self.jobs = {}
        self.desktop = None
        self.lock = Lock()
        self.closed = Event()
        self.wake = Event()
        self.monitor = Thread(target=self.watch, name='rgbmocap-jobs', daemon=True)
        self.monitor.start()

    def tools(self):
        return [command.tool() for command in self.commands.values()] + list(CONTROLS)

    def advance(self):
        with self.lock:
            for job_id, job in list(self.jobs.items()):
                if job.process is not None:
                    code = job.process.poll()
                    if code is None and (self.closed.is_set() or self.registry.cancel_requested(job_id)):
                        stop_tree(job.process)
                        job.stopped = True
                        code = job.process.wait(timeout=10)
                    if code is None:
                        continue
                    job.finish(self.registry, job_id, code)
                elif self.closed.is_set() or self.registry.cancel_requested(job_id):
                    self.registry.finish(job_id, 'cancelled', error='Cancelled before starting.')
                elif not job.queued or self.registry.claim(job_id):
                    try:
                        job.start(job_id)
                    except OSError as error:
                        self.registry.finish(job_id, 'failed', error=str(error), returncode=1)
                    else:
                        continue
                else:
                    continue
                del self.jobs[job_id]
                job.discard_logs()

    def watch(self):
        while True:
            try:
                self.advance()
            except (DatabaseError, OSError, ValueError, subprocess.SubprocessError) as error:
                print(f'RGB MoCap job monitor: {error}', file=sys.stderr)
            if self.closed.is_set() and not self.jobs:
                return
            self.wake.wait(.25)
            self.wake.clear()

    def call(self, name, args):
        if name in self.commands:
            argv = self.commands[name].argv(args)
            try:
                parsed = self.parser.parse_args(argv)
            except SystemExit as error:
                raise ValueError('Invalid command arguments.') from error
            queued = writes_installation(parsed)
            project = getattr(parsed, 'project', None) or getattr(parsed, 'video', None)
            with self.lock:
                job_id = self.registry.add('mcp', name, project=(ROOT / project).resolve() if project else None,
                                           queued=queued)
                self.jobs[job_id] = Job(argv, self.folder / (job_id + '.stdout'),
                                        self.folder / (job_id + '.stderr'), queued)
            self.wake.set()
            return {'job_id': job_id, 'state': 'queued' if queued else 'running'}
        if name == 'job_list':
            if args.keys() - {'limit'}:
                raise ValueError('Expected an optional job count.')
            return {'jobs': self.registry.list(args.get('limit', 50))}
        if name in {'job_status', 'job_cancel'}:
            if set(args) != {'job_id'}:
                raise ValueError('Choose an existing local job_id.')
            job_id = args['job_id']
            if name == 'job_cancel':
                result = self.registry.cancel(job_id)
                self.wake.set()
                return result
            with self.lock:
                result = self.registry.get(job_id)
                if job_id in self.jobs and result['state'] == 'running':
                    result['log'] = self.jobs[job_id].log()
            return result
        if name == 'desktop_start':
            if args.keys() - {'path'} or ('path' in args and not isinstance(args['path'], str)):
                raise ValueError('Expected an optional video or project path.')
            if self.desktop is not None and self.desktop.process.poll() is None:
                raise ValueError('There is already an active MCP desktop.')
            self.desktop = Desktop(args.get('path'))
            return self.desktop.call('state', {}, timeout=40)
        if name.startswith('desktop_') and any(item['name'] == name for item in CONTROLS):
            if self.desktop is None:
                raise ValueError('Start the MCP desktop first.')
            return self.desktop.call(name.removeprefix('desktop_'), args)
        raise ValueError(f'Unknown tool {name!r}.')

    def close(self):
        self.closed.set()
        self.wake.set()
        if self.desktop is not None:
            self.desktop.close()
        self.monitor.join(timeout=20)
        if self.monitor.is_alive():
            # A failing registry must not prevent child cleanup on disconnect.
            with self.lock:
                for job in self.jobs.values():
                    if job.process is not None:
                        stop_tree(job.process)


def reply(request, toolkit):
    if not isinstance(request, dict) or request.get('jsonrpc') != '2.0':
        return dict(jsonrpc='2.0', id=None, error=dict(code=-32600, message='Invalid request'))
    if 'id' not in request:
        return None  # MCP notifications do not receive a reply.
    header = dict(jsonrpc='2.0', id=request['id'])
    params = request.get('params', {})
    if not isinstance(params, dict):
        return {**header, 'error': dict(code=-32602, message='Invalid params')}
    match request.get('method'):
        case 'initialize':
            version = params.get('protocolVersion')
            result = dict(protocolVersion=version if version in VERSIONS else VERSIONS[0],
                          capabilities={'tools': {}}, serverInfo={'name': 'rgbmocap', 'version': '1.0.0'})
        case 'ping':
            result = {}
        case 'tools/list':
            result = {'tools': toolkit.tools()}
        case 'tools/call':
            name = params.get('name')
            args = params.get('arguments', {})
            try:
                if not isinstance(name, str) or not isinstance(args, dict):
                    raise ValueError('Expected a tool name and an arguments object.')
                data = toolkit.call(name, args)
                if name == 'desktop_capture':
                    result = {'content': [dict(type='image', mimeType='image/png', data=data['png'])]}
                else:
                    result = dict(content=[dict(type='text', text=json.dumps(data, ensure_ascii=False))],
                                  structuredContent=data)
            except (DatabaseError, KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
                result = dict(isError=True, content=[dict(type='text', text=str(error))])
        case _:
            return {**header, 'error': dict(code=-32601, message='Method not found')}
    return {**header, 'result': result}


def serve(input_stream, output_stream, toolkit):
    while line := input_stream.readline(REQUEST_BYTES + 1):
        if len(line) > REQUEST_BYTES:
            break
        try:
            result = reply(json.loads(line), toolkit)
        except (UnicodeError, json.JSONDecodeError):
            result = dict(jsonrpc='2.0', id=None, error=dict(code=-32700, message='Invalid JSON'))
        if result is not None:
            output_stream.write(json.dumps(result, ensure_ascii=False).encode('utf-8') + b'\n')
            output_stream.flush()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='rgbmocap-mcp-') as folder:
        toolkit = Toolkit(folder)
        try:
            serve(sys.stdin.buffer, sys.stdout.buffer, toolkit)
        finally:
            toolkit.close()
