"""Join of a linked row of the sequence: the takes of its later bars are imported, one after another, into the scene of its
first block as further takes of the receiving person, each followed by the export the window queues after an import; the
new gaps get the links' methods and one Fill hidden ranges joins them. Every step is a batch on the window's own job queue,
waited for as Fill all hidden ranges and Reconstruct in parts wait for theirs. Taking a join back takes its copies out of
the receiving person's take list; no file is ever deleted."""
from pathlib import Path

from PySide6.QtCore import QObject, QTimer

from . import people_store, sequence_links, sequence_state

QUESTION = ("Join copies the linked takes into the scene of the first block and joins them there. In that block the joined "
            "animation becomes the person's take in use; the later blocks are not changed. Unlink puts the first block back. "
            "No file is deleted. Join now?")
TAKE_APART = 'This row is joined. Taking it apart puts the first block back to its own takes; no file is deleted. Continue?'
IMPORTING = 'Joining {}: importing {} of {}.'
FILLING = 'Joining {}: filling the transitions.'
JOINED = 'Joined {} across {} blocks. Double-click the row to open it.'
STOPPED = 'Join stopped: {}'
CANCELLED = 'cancelled'
NOT_LOADED = 'the first block could not be opened.'
SCENE = 'the window shows another scene.'
NOT_STARTED = 'the job did not start.'
NOT_RECORDED = 'the imported take was not recorded.'


def withdraw(root, person_id, copies) -> int:
    """Take the takes whose folder is in `copies` (relative to the clip folder `root` or absolute) out of the takes of person
    `person_id` in `root`'s people.json, and drop every join of that person built on one of them; saved through
    save_people, and only when something changes. How many takes were taken out; no file or folder is deleted."""
    root = Path(root);state = people_store.load_people(root)
    person = next((p for p in state['people'] if p['id'] == person_id), None) if state else None
    if person is None:return 0
    gone = {(root/copy).resolve() for copy in copies}
    out = lambda project:(root/project).resolve() in gone
    takes = [t for t in person.get('takes') or [] if not out(t['project'])]
    joins = [j for j in person.get('joins') or [] if not any(out(s) for s in j.get('sources') or [])]
    if len(takes) == len(person.get('takes') or []) and len(joins) == len(person.get('joins') or []):return 0
    count = len(person['takes'])-len(takes);person['takes'] = takes
    if 'joins' in person:person['joins'] = joins
    people_store.save_people(root, state)
    return count


def row(pool, link) -> list:
    """The links of `link`'s row for its model, among `pool`."""
    same = [l for l in pool if l.model == link.model];seen = [link]
    for step in (lambda cur:next((l for l in same if l.target == cur.source), None), lambda cur:next((l for l in same if l.source == cur.target), None)):
        cur = link
        while (cur := step(cur)) is not None and not any(cur is s for s in seen):seen.append(cur)
    return seen


def rows(sequence, links) -> list:
    """Every link of the rows of `links`, over the sequence's links and the ones it just dropped."""
    pool = sequence.links+[l for l in sequence.dropped if not any(l is k for k in sequence.links)];out = []
    for link in links:
        out += [l for l in row(pool, link) if not any(l is o for o in out)]
    return out


def take_apart(sequence, links) -> list[Path]:
    """Withdraw the copies the rows of `links` recorded and clear their `built`; the clip folders whose people list changed.
    A list that cannot be saved raises, and its links keep their record."""
    groups = {}
    for link in rows(sequence, links):
        if link.built:groups.setdefault((link.built.get('root'), link.built.get('person')), []).append(link)
    roots = []
    for (root, person), group in groups.items():
        if root is not None and withdraw(root, person, [l.built['copy'] for l in group if 'copy' in l.built]):roots.append(Path(root))
        for link in group:link.built = None
    return roots


def joined_take(root, ident, copies) -> Path | None:
    """The folder of the current join of person `ident` of `root` that holds every copy, or None."""
    root = Path(root);person = next(p for p in people_store.load_people(root)['people'] if p['id'] == ident)
    joins = {j['project']:j for j in person.get('joins') or []};copies = {(root/c).resolve() for c in copies}
    for take in people_store.active_takes(root, person, sequence_state.selected_kind(root)):
        if take.get('join') and copies <= {(root/s).resolve() for s in joins[take['project']]['sources']}:
            return people_store.safe_child(root, take['project'])
    return None


def _relative(folder, root) -> str:
    try:return Path(folder).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:return str(folder)


class Runner(QObject):
    """Joins the given rows (each a list of (block id, person id) from Sequence.chain) one after another. A stop by a
    refusal, a failed job, Cancel Job or another scene leaves the window idle on the first block and the row not joined;
    the copies imported so far are recorded, so the next Join or Unlink takes them out first."""
    def __init__(self, controller, chains):
        super().__init__(controller)
        self.c, self.w, self.chains = controller, controller.w, [list(chain) for chain in chains]
        self.model = self.w.model_choice.currentData()
        self.running, self.stopping, self.waited, self.phase, self.batch = False, False, False, 'next', None
        self.k, self.stage, self.done = 0, 'begin', []
        self.timer = QTimer(self);self.timer.setInterval(200);self.timer.timeout.connect(self.tick)

    def start(self):
        self.running = True
        self.w.cancel_button.clicked.connect(self.cancel)   # also between the jobs, as Reconstruct in parts
        self.c.refresh();self.w.update_readiness();self.timer.start();self.tick()

    def cancel(self):self.stopping = True

    def tick(self):
        try:self.step()
        except Exception as error:  # a timer slot must not raise: Qt would call it again every 200 ms
            self.stop(str(error) or type(error).__name__)

    def outcome(self):
        """(failed, cancelled, status text) of this run's batch once it has ended, else None (as fill_all_ui.Runner)."""
        w = self.w
        if w.busy and getattr(w, 'batch_serial', None) == self.batch:return None
        last = getattr(w, 'last_batch', None)
        if last is not None and last[0] == self.batch:return last[1:]
        return None if w.busy else (w.failed, w.cancelled, w.status.text())

    def step(self):
        w = self.w
        if self.phase == 'running':
            outcome = self.outcome()
            if outcome is None:return
            self.phase = 'next'
            if not getattr(self, 'ended_'+self.stage)(*outcome):return
        if self.phase != 'next':return
        if not w.isVisible():return self.close()   # the window was closed: nothing more starts
        if self.stopping:return self.stop(CANCELLED)
        if w.busy or w.contact_saving:   # another job (a bone preview): wait for it
            self.waited = True;return
        if self.waited:
            self.waited = False
            if w.cancelled:return self.stop(CANCELLED)   # Cancel Job on that job ends the run too
        if self.stage != 'begin' and not self.ours():return self.stop(SCENE)
        getattr(self, self.stage)()

    def ours(self):
        w = self.w
        return w.people_state is not None and w.model_choice.currentData() == self.model and Path(w.people_root or '').resolve() == self.root.resolve()

    def person(self):
        return next(p for p in people_store.load_people(self.root)['people'] if p['id'] == self.ident)

    def run(self, message):
        self.batch, self.phase = getattr(self.w, 'batch_serial', None), 'running'
        self.w.status.setText(message);self.w.log.append(message)

    # The steps of one row

    def begin(self):
        """Opens the first block, refuses before anything changes, takes earlier copies of this row out and activates the
        receiving person."""
        if self.k == len(self.chains):return self.stop(None)
        c, w, chain = self.c, self.w, self.chains[self.k]
        sequence = c.sequence;self.links = [sequence.link_into(*person) for person in chain[1:]]
        first = sequence.block(chain[0][0]);folder = first.folders.get(self.model)
        states = c._states();self.name = c._name(states, chain[0])
        if folder is None:return self.stop(NOT_LOADED)
        self.root, self.ident = Path(folder), chain[0][1]
        if not c._open(first.id) or not self.ours():return self.stop(NOT_LOADED)
        reason = self.refusal(states)
        if reason:return self.stop(reason)
        take_apart(sequence, self.links);c._save();c._reload(self.root)
        w.activate_person(self.ident)
        self.i, self.stage = 0, 'importing'
        self.importing()

    def refusal(self, states):
        """Why this row cannot be joined at all: a link rule, a transition its method cannot fill, the Hidden ranges panel's
        own reason (FlowHMR, ExMR without Static Camera, HTD Refine), Automatic not set up; or None."""
        from .parts_runner import join_issue
        sequence = self.c.sequence
        for link in self.links:
            reason = sequence_links.refusal(sequence, states, link.source, link.target, self.model) or \
                sequence_links.gap_issue(link.method, sequence_links.gap_frames(sequence, link.target[0]))
            if reason:return reason
        reason = join_issue(self.w)
        if reason or not any(link.method == 'auto' for link in self.links):return reason
        from . import kimodo_runtime  # standard library only
        try:return kimodo_runtime.issue()
        except Exception:return kimodo_runtime.SETUP

    def importing(self):
        """Imports the take of the next later bar, after the transition of its pair of blocks."""
        from . import take_import_check
        if self.i == len(self.links):return self.methods()
        w, sequence, link = self.w, self.c.sequence, self.links[self.i]
        later = Path(sequence.block(link.target[0]).folders[self.model])
        self.source = sequence_links._in_use(later, link.target[1])[0]   # the one take in use, as the row's state reads it
        self.later, self.frames = later, sequence_links.gap_frames(sequence, link.target[0])
        person = self.person()
        reason = take_import_check.refusal(self.root, person, self.source, self.frames, w.active_result)
        if reason:return self.stop(reason)
        self.before, self.stamp = {t['project'] for t in person.get('takes') or []}, sequence_links.result_stamp(self.source)
        if not w.start_take_import(self.ident, self.source, self.frames):return self.stop(w.status.text() or NOT_STARTED)
        self.run(IMPORTING.format(self.name, self.i+1, len(self.links)))

    def ended_importing(self, failed, cancelled, status):
        """Records the copy on its link as soon as the import wrote it, whatever became of the export after it."""
        link = self.links[self.i]
        new = [t for t in self.person().get('takes') or [] if t['project'] not in self.before]
        if new:
            link.built = dict(copy=new[-1]['project'], take=_relative(self.source, self.later), stamp=self.stamp, gap=self.frames,
                              method=link.method, root=str(self.root.resolve()), person=self.ident)
            self.c._save()
        if cancelled or failed or not new:
            self.stop(CANCELLED if cancelled else status if failed else NOT_RECORDED);return False
        self.i += 1
        return True

    def methods(self):
        """Gives every new gap its link's method, then starts Fill hidden ranges for the receiving person."""
        from .hidden_ranges import gap_key, gaps
        from .hidden_ranges_state import choose
        w = self.w;state = people_store.load_people(self.root)
        person = next(p for p in state['people'] if p['id'] == self.ident)
        choices = person['gap_fill'] if isinstance(person.get('gap_fill'), dict) else {}
        methods = {link.built['copy']:link.method for link in self.links}
        for gap in gaps(person['takes']):
            if gap['after']['project'] in methods:choose(choices, gap_key(gap), methods[gap['after']['project']])
        person['gap_fill'] = choices
        people_store.save_people(self.root, state);w.people_state = state;w.refresh_hidden_ranges(w.current_person())
        w.fill_hidden_ranges(self.ident)
        if not w.busy:return self.stop(w.status.text() or NOT_STARTED)
        self.stage = 'filling';self.run(FILLING.format(self.name))

    def ended_filling(self, failed, cancelled, status):
        if cancelled or failed:
            self.stop(CANCELLED if cancelled else status);return False
        self.done.append(JOINED.format(self.name, len(self.chains[self.k])))
        self.k, self.stage = self.k+1, 'begin'
        return True

    # The end

    def close(self):
        self.timer.stop();self.phase, self.running = 'stopped', False
        try:self.w.cancel_button.clicked.disconnect(self.cancel)
        except (RuntimeError, TypeError):pass
        if getattr(self.c, 'joiner', None) is self:self.c.joiner = None

    def stop(self, reason):
        """Ends the run: `reason` None after the last row, else why it stopped."""
        self.close()
        message = ' '.join(self.done) if reason is None else STOPPED.format(reason)
        self.w.status.setText(message);self.w.log.append(message)
        self.c.refresh();self.w.update_readiness()
