import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from mocap import kimodo_runtime
from tools import setup_kimodo as setup

DOWNLOAD = 'Automatic fill setup could not download its files. Check the internet connection and try again.'


def events(text):
    return [json.loads(line[len('@@MOCAP@@'):]) for line in text.splitlines() if line.startswith('@@MOCAP@@')]


class FromAppTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.lock = mock.Mock(side_effect=AssertionError('the app holds the lock'))
        self.build = mock.Mock(side_effect=AssertionError('build must not run'))
        for target in (mock.patch.object(setup, 'ROOT', self.root), mock.patch.object(kimodo_runtime, 'ROOT', self.root),
                       mock.patch.object(setup, 'app_lock', self.lock), mock.patch.object(setup, 'build', self.build),
                       mock.patch('mocap.runtime_registry.activate'),
                       mock.patch('shutil.which', return_value='C:/Git/cmd/git.exe'),
                       mock.patch('shutil.disk_usage', return_value=mock.Mock(free=2 * 1024 ** 3))):
            target.start(); self.addCleanup(target.stop)

    def run_setup(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = setup.main(['--from-app'])
        return code, events(out.getvalue())

    def assert_failed(self, message):
        code, seen = self.run_setup()
        self.assertEqual((code, seen), (1, [dict(type='failed', stage='kimodo_setup', message=message)]))

    def test_too_little_free_disk_fails_before_anything_is_fetched(self):
        with mock.patch('shutil.disk_usage', return_value=mock.Mock(free=1024 ** 3 - 1)):
            self.assert_failed('Not enough free disk space to set up automatic fill (1 GiB needed).')
        self.build.assert_not_called()

    def test_missing_git_fails_before_anything_is_fetched(self):
        with mock.patch('shutil.which', return_value=None):
            self.assert_failed('Git was not found. Install Git for Windows or run install.cmd again.')
        self.build.assert_not_called()

    def test_a_failed_download_keeps_the_existing_runtime(self):
        old = self.root / 'runtime_kimodo'; old.mkdir(); (old / 'runtime.json').write_text('{"version": 1}', encoding='utf-8')
        self.build.side_effect = RuntimeError(DOWNLOAD + ' fatal: unable to access')
        self.assert_failed(DOWNLOAD + ' fatal: unable to access')
        self.assertEqual([p.name for p in old.iterdir()], ['runtime.json'])
        self.assertEqual((old / 'runtime.json').read_text(encoding='utf-8'), '{"version": 1}')
        self.assertEqual([p.name for p in (self.root / '.setup').iterdir()], [])

    def test_a_tool_that_fails_outside_a_named_step_is_not_blamed_on_the_connection(self):
        self.build.side_effect = subprocess.CalledProcessError(1, ['tool'], output='boom')
        self.assert_failed('Automatic fill setup failed. boom')
        self.build.side_effect = subprocess.TimeoutExpired(['tool'], 30)
        self.assert_failed('Automatic fill setup failed.')

    @staticmethod
    def build_overlay(stage, step):
        for message in ('Fetching', 'Installing', 'Building', 'Checking'):
            step(message)
        overlay = stage / 'overlay'; (overlay / 'kimodo/model').mkdir(parents=True)
        (overlay / 'kimodo/model/load_model.py').write_text('', encoding='utf-8')
        (overlay / 'new').write_text('', encoding='utf-8')
        return overlay

    def valid_runtime(self):
        """An installed runtime with a valid receipt, as location() accepts it."""
        folder = self.root / 'runtime_kimodo'; (folder / 'kimodo/model').mkdir(parents=True)
        (folder / 'kimodo/model/load_model.py').write_text('', encoding='utf-8')
        (folder / 'runtime.json').write_text(json.dumps(dict(version=2, revision=kimodo_runtime.REVISION)), encoding='utf-8')
        (folder / 'old').write_text('', encoding='utf-8')
        return folder

    def test_a_successful_build_is_installed_and_reported(self):
        old = self.root / 'runtime_kimodo'; old.mkdir(); (old / 'runtime.json').write_text('{"version": 1}', encoding='utf-8')
        self.build.side_effect = self.build_overlay
        code, seen = self.run_setup()
        folder = self.root / 'runtime_kimodo'
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((folder / 'runtime.json').read_text(encoding='utf-8')),
                         dict(version=2, revision=kimodo_runtime.REVISION))
        self.assertEqual(kimodo_runtime.location(), folder)
        progress = [e for e in seen if e['type'] == 'progress']
        self.assertEqual([(e['stage'], e['current'], e['total']) for e in progress],
                         [('kimodo_setup', n, 5) for n in range(5)])
        self.assertTrue(all(e['message'] for e in progress))
        self.assertEqual(seen[5:], [dict(type='complete', stage='kimodo_setup', output=str(folder),
                                         message='Automatic fill runtime installed.')])
        self.assertEqual(len(seen), 6)
        self.lock.assert_not_called()
        self.assertEqual([p.name for p in (self.root / '.setup').iterdir()], [])   # the replaced runtime's backup is removed

    def test_a_valid_runtime_that_passes_its_check_is_kept(self):
        folder = self.valid_runtime()
        with mock.patch.object(setup, 'verify') as verify:
            code, seen = self.run_setup()
        verify.assert_called_once_with(folder); self.build.assert_not_called()
        self.assertEqual((code, seen), (0, [dict(type='complete', stage='kimodo_setup', output=str(folder),
                                                 message='Automatic fill runtime verified.')]))
        self.assertTrue((folder / 'old').is_file())

    def test_a_valid_runtime_that_fails_its_check_is_installed_again(self):
        folder = self.valid_runtime(); self.build.side_effect = self.build_overlay
        with mock.patch.object(setup, 'verify', side_effect=RuntimeError('The Kimodo runtime check failed: x')) as verify:
            code, seen = self.run_setup()
        verify.assert_called_once_with(folder); self.build.assert_called_once()
        self.assertEqual(code, 0); self.assertEqual(seen[-1], dict(type='complete', stage='kimodo_setup', output=str(folder),
                                                                    message='Automatic fill runtime installed.'))
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ['kimodo', 'new', 'runtime.json'])   # the broken one replaced
        self.assertEqual([p.name for p in (self.root / '.setup').iterdir()], [])

    def test_leftovers_of_cancelled_runs_are_removed(self):
        setup_folder = self.root / '.setup'
        for name in ('.kimodo-abc', '.kimodo-def'):(setup_folder / name / 'upstream').mkdir(parents=True)
        pack = setup_folder / '.kimodo-abc/upstream/.git/objects/pack/pack-1.pack'; pack.parent.mkdir(parents=True)
        pack.write_bytes(b'pack'); os.chmod(pack, stat.S_IREAD)   # Git marks what it fetched read-only
        (setup_folder / 'app.lock').write_text('', encoding='utf-8'); (setup_folder / 'kimodo-backup-1').mkdir()
        self.build.side_effect = self.build_overlay
        self.assertEqual(self.run_setup()[0], 0)
        self.assertEqual(sorted(p.name for p in setup_folder.iterdir()), ['app.lock', 'kimodo-backup-1'])   # only the setup's own leftovers
        (setup_folder / '.kimodo-ghi').mkdir()
        with mock.patch('shutil.disk_usage', return_value=mock.Mock(free=1024 ** 3 - 1)):   # removed before the space is measured
            self.assert_failed('Not enough free disk space to set up automatic fill (1 GiB needed).')
        self.assertFalse((setup_folder / '.kimodo-ghi').exists())

    def test_a_cleanup_error_after_the_swap_does_not_fail_the_install(self):
        self.build.side_effect = self.build_overlay
        with mock.patch.object(setup.tempfile, 'TemporaryDirectory', wraps=setup.tempfile.TemporaryDirectory) as made:
            self.assertEqual(self.run_setup()[0], 0)
        self.assertIs(made.call_args.kwargs.get('ignore_cleanup_errors'), True)


class BuildFailureTest(unittest.TestCase):
    """The sentence a failing tool of each step gets; `run` is replaced, so nothing is fetched or built."""
    def fail_at(self, match, error=None):
        def run(command, timeout, **options):
            if match(command):
                raise error or subprocess.CalledProcessError(1, command, output='last line')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(setup, 'run', run), \
                mock.patch.object(setup.subprocess, 'check_output', return_value=setup.REVISION + '\n'), \
                mock.patch.object(setup.subprocess, 'run', return_value=mock.Mock(returncode=0)):
            with self.assertRaises(RuntimeError) as caught:
                setup.build(Path(tmp))
        return str(caught.exception)

    def test_only_the_steps_that_download_blame_the_connection(self):
        self.assertEqual(self.fail_at(lambda c: 'fetch' in c), DOWNLOAD + ' last line')
        self.assertEqual(self.fail_at(lambda c: '--only-binary=:all:' in c), DOWNLOAD + ' last line')
        self.assertEqual(self.fail_at(lambda c: 'fetch' in c, subprocess.TimeoutExpired(['git'], 300)), DOWNLOAD)
        self.assertEqual(self.fail_at(lambda c: 'init' in c), 'Automatic fill setup could not unpack the Kimodo source. last line')
        self.assertEqual(self.fail_at(lambda c: 'checkout' in c), 'Automatic fill setup could not unpack the Kimodo source. last line')
        self.assertEqual(self.fail_at(lambda c: '--no-build-isolation' in c),
                         'Automatic fill setup could not get or build antlr4-python3-runtime. last line')


class RunTest(unittest.TestCase):
    def test_output_is_passed_through_and_a_failure_names_the_last_line(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(subprocess.CalledProcessError) as caught:
            setup.run([sys.executable, '-c', "print('first');print('last');raise SystemExit(3)"], timeout=60)
        self.assertEqual(out.getvalue(), 'first\nlast\n')
        self.assertEqual((caught.exception.returncode, caught.exception.output), (3, 'last'))

    def test_a_silent_tool_is_stopped_at_its_timeout(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.TimeoutExpired):
            setup.run([sys.executable, '-c', 'import time;time.sleep(60)'], timeout=1)


if __name__ == '__main__':
    unittest.main()
