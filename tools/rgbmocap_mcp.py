"""Local stdio MCP server for RGB MoCap."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
from threading import Thread
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.cli import parser
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
    tool('job_status', 'Read a local CLI job result and its latest log lines.', JOB, JOB, read=True),
    tool('job_cancel', 'Cancel a CLI job started by this MCP session.', JOB, JOB),
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
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8, check=False)
    else:
        process.terminate()


@dataclass
class Job:
    name: str
    process: subprocess.Popen
    stdout: Path
    stderr: Path
    cancelled: bool = False

    def status(self):
        code = self.process.poll()
        state = 'running' if code is None else 'cancelled' if self.cancelled else 'done' if not code else 'failed'
        with self.stderr.open('rb') as log:
            log.seek(max(0, self.stderr.stat().st_size - LOG_BYTES))
            tail = log.read(LOG_BYTES).decode('utf-8', errors='replace')
        reply = dict(tool=self.name, state=state, returncode=code, log=tail)
        if code is not None:
            with self.stdout.open('rb') as output:
                payload = output.read(REQUEST_BYTES + 1)
            if len(payload) > REQUEST_BYTES:
                reply['error'] = 'CLI result exceeds the MCP response limit.'
            else:
                try:
                    reply['result'] = json.loads(payload)
                except (UnicodeError, ValueError):
                    reply['output'] = payload.decode('utf-8', errors='replace')
        return reply


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
    def __init__(self, folder):
        self.folder = Path(folder)
        self.commands = {command.name: command for command in commands(parser())}
        self.jobs = {}
        self.desktop = None

    def tools(self):
        return [command.tool() for command in self.commands.values()] + list(CONTROLS)

    def call(self, name, args):
        if name in self.commands:
            command = self.commands[name]
            argv = command.argv(args)
            job_id = uuid4().hex
            stdout = self.folder / (job_id + '.stdout')
            stderr = self.folder / (job_id + '.stderr')
            with stdout.open('wb') as output, stderr.open('wb') as log:
                process = subprocess.Popen([*PYTHON, str(ROOT / 'cli_boot.py'), *argv], cwd=ROOT,
                                           stdin=subprocess.DEVNULL, stdout=output, stderr=log)
            self.jobs[job_id] = Job(name, process, stdout, stderr)
            return {'job_id': job_id, 'state': 'running'}
        if name in {'job_status', 'job_cancel'}:
            if set(args) != {'job_id'} or args['job_id'] not in self.jobs:
                raise ValueError('Choose an existing local job_id.')
            job = self.jobs[args['job_id']]
            if name == 'job_cancel' and job.process.poll() is None:
                stop_tree(job.process)
                job.cancelled = True
            return {'job_id': args['job_id'], **job.status()}
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
        for job in self.jobs.values():
            if job.process.poll() is None:
                stop_tree(job.process)
                job.process.wait(timeout=10)
        if self.desktop is not None:
            self.desktop.close()


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
            except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
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
