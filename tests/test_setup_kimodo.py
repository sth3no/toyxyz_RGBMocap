import contextlib
import io
import json
from pathlib import Path
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
        self.build.side_effect = subprocess.CalledProcessError(128, ['git', 'fetch'], output='fatal: unable to access')
        self.assert_failed(DOWNLOAD + ' fatal: unable to access')
        self.assertEqual([p.name for p in old.iterdir()], ['runtime.json'])
        self.assertEqual((old / 'runtime.json').read_text(encoding='utf-8'), '{"version": 1}')
        self.assertEqual([p.name for p in (self.root / '.setup').iterdir()], [])

    def test_a_successful_build_is_installed_and_reported(self):
        old = self.root / 'runtime_kimodo'; old.mkdir(); (old / 'runtime.json').write_text('{"version": 1}', encoding='utf-8')
        def build(stage, step):
            for message in ('Fetching', 'Installing', 'Building', 'Checking'):
                step(message)
            overlay = stage / 'overlay'; (overlay / 'kimodo/model').mkdir(parents=True)
            (overlay / 'kimodo/model/load_model.py').write_text('', encoding='utf-8')
            return overlay
        self.build.side_effect = build
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
        self.assertEqual(seen[5:], [dict(type='complete', stage='kimodo_setup', output=str(folder))])
        self.assertEqual(len(seen), 6)
        self.lock.assert_not_called()


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
