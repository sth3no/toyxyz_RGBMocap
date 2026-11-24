from contextlib import redirect_stdout
from io import StringIO
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mocap.job_queue import Jobs


class JobQueueTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.jobs = Jobs(self.root)
        self.other = Jobs(self.root)

    def test_fifo_and_independent_read_only_work(self):
        first = self.jobs.add('mcp', 'reconstruct', queued=True)
        second = self.other.add('cli', 'export', queued=True)
        search = self.other.add('mcp', 'check')
        self.assertEqual(self.jobs.get(search)['state'], 'running')
        self.assertFalse(self.other.claim(second))
        self.assertTrue(self.jobs.claim(first))
        self.assertTrue(self.other.owned(first))
        self.assertFalse(self.other.claim(second))
        self.jobs.finish(first, 'done', result={'project': 'complete'})
        self.assertTrue(self.other.claim(second))
        self.assertEqual(self.other.get(first)['result'], {'project': 'complete'})

    def test_gui_state_and_cancellation_are_visible_across_connections(self):
        job = self.jobs.add('gui', 'Reconstruction', project='scene')
        self.jobs.phase(job, 'exmr', message='Frame 1 of 2')
        self.assertEqual(self.other.get(job)['phase'], 'exmr')
        self.assertTrue(self.other.cancel(job)['cancel_requested'])
        self.assertTrue(self.jobs.cancel_requested(job))
        self.jobs.finish(job, 'failed', error='Out of memory', returncode=1)
        self.jobs.finish(job, 'done')
        self.assertEqual(self.other.get(job)['error'], 'Out of memory')
        self.assertEqual(self.other.get(job)['state'], 'failed')

    def test_cancel_queued_and_reconcile_dead_owner(self):
        stale = self.jobs.add('mcp', 'reconstruct', queued=True)
        cancelled = self.jobs.add('mcp', 'export', queued=True)
        self.assertEqual(self.other.cancel(cancelled)['state'], 'cancelled')
        self.assertFalse(self.jobs.claim(cancelled))
        self.assertFalse(self.other.owned(cancelled))
        with patch('mocap.job_queue.owner_alive', return_value=False):
            found = {job['job_id']: job for job in self.other.list()}
        self.assertEqual(found[stale]['state'], 'interrupted')
        self.assertIn('exited', found[stale]['error'])
        self.assertEqual(found[cancelled]['state'], 'cancelled')


    def test_foreground_cli_result_and_phase_visible_through_mcp(self):
        from mocap.cli import main
        from tools.rgbmocap_mcp import Toolkit

        def run(args, report, managed):
            self.assertFalse(managed)
            report.set_phase('correct')
            current = self.other.list()[0]
            self.assertEqual((current['state'], current['phase']), ('running', 'correct'))
            return {'failed': True, 'error': 'Missing motion'}

        output = StringIO()
        with patch('mocap.paths.HOME', self.root), patch('mocap.cli.dispatch', side_effect=run), redirect_stdout(output):
            self.assertEqual(main(['correct', 'take', '--json']), 1)
        job_id = json.loads(output.getvalue())['job_id']
        toolkit = Toolkit(self.root, jobs=self.other)
        self.addCleanup(toolkit.close)
        job = toolkit.call('job_status', {'job_id': job_id})
        self.assertEqual((job['source'], job['state'], job['error']), ('cli', 'failed', 'Missing motion'))

    def test_cli_managed_command_waits_until_installation_is_available(self):
        from mocap.cli import wait_for_installation
        from tools.session_lock import AppBusyError
        from contextlib import nullcontext

        report = Mock()
        with patch('tools.session_lock.app_lock', side_effect=[AppBusyError(), nullcontext()]) as lock, \
                patch('mocap.cli.time.sleep'):
            with wait_for_installation(report):
                self.assertEqual(lock.call_count, 2)
        self.assertEqual(report.set_phase.call_args.args[0], 'waiting_for_installation')

    def test_mcp_process_completion_and_disconnect_are_persistent(self):
        from tools import rgbmocap_mcp as mcp

        def simulate(job, job_id):
            script = 'import time; time.sleep(60)' if job.queued else 'print(' + repr('{"ok": true}') + ')'
            with job.stdout.open('wb') as out, job.stderr.open('wb') as err:
                job.process = subprocess.Popen([mcp.PYTHON[0], '-I', '-c', script],
                                               stdout=out, stderr=err, stdin=subprocess.DEVNULL)

        with patch.object(mcp.Job, 'start', simulate):
            toolkit = mcp.Toolkit(self.root, jobs=self.jobs)
            self.addCleanup(toolkit.close)
            first = toolkit.call('check', {})['job_id']
            waiting = toolkit.call('correct', {'project': 'take'})['job_id']
            next_job = toolkit.call('correct', {'project': 'another'})['job_id']
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                if self.other.get(first)['state'] == 'done' and self.other.get(waiting)['state'] == 'running':
                    break
                time.sleep(.05)
            self.assertEqual(self.other.get(first)['result'], {'ok': True})
            self.assertEqual(self.other.get(next_job)['state'], 'queued')
            self.assertEqual(self.other.get(waiting)['state'], 'running')
            toolkit.close()
        reconnected = mcp.Toolkit(self.root, jobs=self.other)
        self.addCleanup(reconnected.close)
        self.assertEqual(reconnected.call('job_status', {'job_id': first})['state'], 'done')
        self.assertEqual(reconnected.call('job_status', {'job_id': waiting})['state'], 'cancelled')
        self.assertEqual(reconnected.call('job_status', {'job_id': next_job})['state'], 'cancelled')

    def test_offscreen_batch_and_queue_dialog(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest('PySide6 unavailable')
        from mocap import gui, job_queue_ui

        app = QApplication.instance() or QApplication([])
        with patch('mocap.paths.HOME', self.root):
            window = gui.MainWindow()
            self.addCleanup(window.close)
            started = Mock()
            with patch.object(window, 'next_job', started):
                window.start_jobs([('diagnose', [])])
            started.assert_called_once()
            job = self.other.list()[0]
            self.assertEqual((job['source'], job['state']), ('gui', 'running'))
            dialog = job_queue_ui.QueueDialog(window)
            dialog.cancel.click()
            self.assertTrue(self.other.get(job['job_id'])['cancel_requested'])
            job_queue_ui.poll(window)
            self.assertTrue(window.cancelled)
            window.next_job()
            self.assertEqual(self.other.get(job['job_id'])['state'], 'cancelled')
            dialog.close()
            app.processEvents()
