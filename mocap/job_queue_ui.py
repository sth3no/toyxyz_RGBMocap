"""Desktop status and cancellation for the shared workflow queue."""
import json
from sqlite3 import Error as DatabaseError

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView,
                               QLabel, QPlainTextEdit, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .job_queue import Jobs
from .theme import ui_px


STORE_ERRORS = (DatabaseError, OSError, ValueError)


def note(window, method, *args, **kwargs):
    try:
        return getattr(Jobs(), method)(*args, **kwargs)
    except STORE_ERRORS as error:
        window.log.append(f'Could not update the job queue: {error}')
        return None


def poll(window):
    parts = getattr(window, 'parts_runner', None)
    part_id = getattr(parts, 'job_id', None)
    batch_id = getattr(window, '_queue_job_id', None)
    if part_id and not parts.stopping and note(window, 'cancel_requested', part_id):
        window.cancel_job()
    elif (batch_id and not window.cancelled and not window.failed
          and window._queue_cancel_sent != batch_id and note(window, 'cancel_requested', batch_id)):
        window._queue_cancel_sent = batch_id
        window.cancel_job(stop_parts=False)
    if not part_id and not batch_id:
        window._queue_timer.stop()


def finish_batch(window, *, message=None, state=None):
    job_id = getattr(window, '_queue_job_id', None)
    if job_id is None:
        return
    window._queue_job_id = None
    message = window.status.text() if message is None else message
    if state is None:
        state = 'failed' if window.failed else 'cancelled' if window.cancelled else 'done'
    paths = getattr(window, '_motion_export_targets', ())
    result = {'message': message}
    if getattr(window, 'project', None):
        result['project'] = str(window.project)
    if window.current_stage in ('save_fbx', 'copy_bvh', 'save_blend', 'save_blend_scene', 'stream_reference') and paths:
        result['outputs'] = paths
    error = message if state != 'done' else None
    if error == 'Processing stopped. See the console for details.':
        row = note(window, 'get', job_id)
        error = (row or {}).get('message') or error
    note(window, 'finish', job_id, state, result=result if state == 'done' else None, error=error)


class QueueDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle('Job Queue')
        self.resize(ui_px(760), ui_px(430))
        self.jobs = Jobs()
        self.rows = None
        layout = QVBoxLayout(self)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Job', 'Source', 'State', 'Phase', 'Project'])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.details)
        layout.addWidget(self.table, 2)
        self.detail = QPlainTextEdit(readOnly=True)
        layout.addWidget(self.detail, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.cancel = buttons.addButton('Cancel Job', QDialogButtonBox.ActionRole)
        self.cancel.clicked.connect(self.cancel_selected)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.timer = QTimer(self, interval=1000)
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        self.timer.start()

    def selected(self):
        item = self.table.item(self.table.currentRow(), 0)
        return item.data(Qt.UserRole) if item is not None else None

    def refresh(self):
        try:
            rows = self.jobs.list()
        except STORE_ERRORS as error:
            self.status.setText(f'Could not read the job queue: {error}')
            self.cancel.setEnabled(False)
            return
        self.status.setText('Active and recent workflow jobs.' if rows else 'No workflow jobs yet.')
        if rows != self.rows:
            current = self.selected()
            self.rows = rows
            self.table.setRowCount(len(rows))
            match = -1
            for index, row in enumerate(rows):
                fields = ('tool', 'source', 'state', 'phase', 'project')
                for column, field in enumerate(fields):
                    item = QTableWidgetItem(str(row.get(field, '')))
                    if column == 0:
                        item.setData(Qt.UserRole, row['job_id'])
                    self.table.setItem(index, column, item)
                if row['job_id'] == current:
                    match = index
            if rows:
                self.table.selectRow(max(0, match))
        self.details()

    def details(self):
        job_id = self.selected()
        if not job_id:
            self.detail.clear()
            self.cancel.setEnabled(False)
            return
        try:
            row = self.jobs.get(job_id)
        except STORE_ERRORS as error:
            self.detail.setPlainText(str(error))
            self.cancel.setEnabled(False)
            return
        lines = [row['job_id']]
        if row['cancel_requested'] and row['state'] == 'running':
            lines.append('Cancellation requested.')
        if 'error' in row or 'message' in row:
            lines.append(row.get('error') or row['message'])
        if 'result' in row:
            lines.append(json.dumps(row['result'], ensure_ascii=False, indent=2))
        if 'log' in row:
            lines.append(row['log'])
        text = '\n'.join(lines)
        if self.detail.toPlainText() != text:
            self.detail.setPlainText(text)
        self.cancel.setEnabled((row['state'] == 'queued' or row['state'] == 'running' and row['source'] != 'cli')
                               and not row['cancel_requested'])

    def cancel_selected(self):
        try:
            self.jobs.cancel(self.selected())
        except STORE_ERRORS as error:
            self.status.setText(str(error))
            return
        self.refresh()


def show(window):
    dialog = QueueDialog(window)
    dialog.exec()
    dialog.deleteLater()
