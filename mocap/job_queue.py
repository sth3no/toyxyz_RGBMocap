"""Installation-local job state shared by the desktop, CLI and MCP."""
from contextlib import closing, contextmanager
import json
import os
import re
import sqlite3
import time
from uuid import uuid4


TERMINAL = ('done', 'failed', 'cancelled', 'interrupted')
NOT_SET = object()


def failure_reason(result):
    if result.get('error'):
        return result['error']
    if isinstance(result.get('unreal'), dict) and result['unreal'].get('failed'):
        return failure_reason(result['unreal'])
    for key in ('jobs', 'animations', 'errors', 'face_errors'):
        for row in result.get(key) or ():
            if isinstance(row, dict) and row.get('error'):
                return row['error']
            if isinstance(row, str):
                return row
    return 'The workflow reported a failure. See the job result for details.'


def owner_alive(pid, started):
    import psutil
    try:
        return psutil.Process(pid).create_time() == started
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


class Jobs:
    def __init__(self, root=None):
        if root is None:
            from .paths import HOME
            root = HOME
        self.path = root / '.setup' / 'job-queue.sqlite3'

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=2)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, source TEXT NOT NULL, tool TEXT NOT NULL,
                    state TEXT NOT NULL, phase TEXT NOT NULL, project TEXT, message TEXT,
                    result TEXT, error TEXT, returncode INTEGER, log TEXT,
                    owner_pid INTEGER NOT NULL, owner_started REAL NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    queued_write INTEGER NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS active_jobs ON jobs (state, queued_write)
                WHERE state IN ('queued', 'running')
            """)
            yield db

    def add(self, source, tool, *, project=None, queued=False):
        import psutil
        job_id = uuid4().hex
        now = time.time()
        with self.connection() as db:
            db.execute("""
                INSERT INTO jobs (job_id, source, tool, state, phase, project, owner_pid,
                                  owner_started, created_at, updated_at, queued_write)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (job_id, source, tool, 'queued' if queued else 'running',
                  'queued' if queued else 'starting', str(project) if project else None,
                  os.getpid(), psutil.Process().create_time(), now, now, int(queued)))
        return job_id

    def _recover(self, db):
        owners = db.execute("SELECT DISTINCT owner_pid, owner_started FROM jobs WHERE state IN ('queued', 'running')")
        for pid, started in owners.fetchall():
            if not owner_alive(pid, started):
                db.execute("""
                    UPDATE jobs SET state='interrupted', error='The process that owned this job exited.', updated_at=?
                    WHERE owner_pid=? AND owner_started=? AND state IN ('queued', 'running')
                """, (time.time(), pid, started))

    def owned(self, job_id):
        with self.connection() as db:
            row = self._row(db, job_id)
            return (row['state'] == 'running' and not row['cancel_requested']
                    and owner_alive(row['owner_pid'], row['owner_started']))

    def _next_write(self, db):
        self._recover(db)
        if db.execute("SELECT 1 FROM jobs WHERE queued_write=1 AND state='running' LIMIT 1").fetchone():
            return None
        row = db.execute("SELECT job_id FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
        return row['job_id'] if row else None

    def claim(self, job_id):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if self._next_write(db) != job_id:
                return False
            db.execute("UPDATE jobs SET state='running', phase='starting', message=NULL, updated_at=? WHERE job_id=?",
                       (time.time(), job_id))
            return True

    def phase(self, job_id, phase, *, message=None, project=NOT_SET):
        with self.connection() as db:
            db.execute("""
                UPDATE jobs SET phase=?, message=?, project=CASE WHEN ? THEN project ELSE ? END, updated_at=?
                WHERE job_id=? AND state IN ('queued', 'running')
            """, (phase, message, project is NOT_SET, str(project) if project is not NOT_SET and project else None,
                  time.time(), job_id))

    def finish(self, job_id, state, *, result=None, error=None, returncode=None, log=None):
        if state not in TERMINAL:
            raise ValueError('Expected a terminal job state.')
        payload = json.dumps(result, ensure_ascii=False) if result is not None else None
        if payload and len(payload.encode('utf-8')) > 1024 * 1024:
            payload = None
            error = error or 'The result exceeds the saved job size limit. See the command output.'
        with self.connection() as db:
            db.execute("""
                UPDATE jobs SET state=?, result=?, error=?, returncode=?, log=?, updated_at=?
                WHERE job_id=? AND state IN ('queued', 'running')
            """, (state, payload, error, returncode, log, time.time(), job_id))

    @staticmethod
    def _view(row):
        fields = ('job_id', 'source', 'tool', 'state', 'phase', 'project', 'message', 'error',
                  'returncode', 'log', 'created_at', 'updated_at', 'cancel_requested')
        result = {field: row[field] for field in fields if row[field] is not None}
        result['cancel_requested'] = bool(result['cancel_requested'])
        if 'result' in row.keys() and row['result'] is not None:
            result['result'] = json.loads(row['result'])
        return result

    def _row(self, db, job_id):
        if not isinstance(job_id, str) or not re.fullmatch('[a-f0-9]{32}', job_id):
            raise ValueError('Choose an existing local job ID.')
        row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None:
            raise ValueError('Choose an existing local job ID.')
        return row

    def get(self, job_id):
        with self.connection() as db:
            self._recover(db)
            return self._view(self._row(db, job_id))

    def list(self, limit=50):
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError('Choose a job count between 1 and 200.')
        with self.connection() as db:
            self._recover(db)
            rows = db.execute("""
                SELECT job_id, source, tool, state, phase, project, message, error, returncode,
                       NULL AS log, created_at, updated_at, cancel_requested
                FROM jobs ORDER BY (state IN ('queued', 'running')) DESC, rowid DESC LIMIT ?
            """, (limit,))
            return [self._view(row) for row in rows]

    def cancel(self, job_id):
        with self.connection() as db:
            self._recover(db)
            row = self._row(db, job_id)
            if row['source'] == 'cli' and row['state'] == 'running':
                raise ValueError('Stop this foreground CLI command with Ctrl+C in its terminal.')
            db.execute("""
                UPDATE jobs SET cancel_requested=1, updated_at=?,
                    error=CASE WHEN state='queued' THEN 'Cancelled before starting.' ELSE error END,
                    state=CASE WHEN state='queued' THEN 'cancelled' ELSE state END
                WHERE job_id=? AND state IN ('queued', 'running')
            """, (time.time(), job_id))
            return self._view(self._row(db, job_id))

    def cancel_requested(self, job_id):
        with self.connection() as db:
            return bool(self._row(db, job_id)['cancel_requested'])
