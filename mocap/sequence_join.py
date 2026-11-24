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
REJOIN = 'Fill hidden ranges again for {} in the first block to rejoin their own takes.'
ALL_JOINED = 'Every linked row is already joined.'
ALREADY = '{} is already joined.'


_ANY = object()   # withdraw without a fingerprint check


def _without(root, person, copies, joins=()) -> tuple[int, bool]:
    """`person` (a people.json entry, changed in place) without the takes whose folder is in `copies` and the joins built
    on them, with each recorded join of `joins` put back while it is current for the takes left. (takes taken out, whether
    a recorded join could not be put back)."""
    root = Path(root);gone = {(root/copy).resolve() for copy in copies}
    out = lambda project:(root/project).resolve() in gone
    takes = person.get('takes') or [];kept = [t for t in takes if not out(t['project'])]
    have = [j for j in person.get('joins') or [] if not any(out(s) for s in j.get('sources') or [])];lost = False
    if 'takes' in person:person['takes'] = kept
    for join in joins or []:
        if any(j.get('project') == join.get('project') for j in have):continue
        try:current = people_store.join_current(root, {**person, 'joins':have+[join]}, join)
        except Exception:current = False
        if current:have.append(join)
        else:lost = True
    if have or 'joins' in person:person['joins'] = have
    return len(takes)-len(kept), lost


def _withdraw(root, person_id, copies, joins=(), fingerprint=_ANY) -> tuple[int, bool, str | None]:
    """withdraw, also saying whether people.json changed and the name of a person whose own join could not be put back."""
    root = Path(root);state = people_store.load_people(root)
    person = next((p for p in state['people'] if p['id'] == person_id), None) if state else None
    if person is None or fingerprint is not _ANY and state.get('fingerprint') != fingerprint:return 0, False, None
    before = (list(person.get('takes') or []), list(person.get('joins') or []))
    count, lost = _without(root, person, copies, joins)
    changed = (person.get('takes') or [], person.get('joins') or []) != before
    if changed:people_store.save_people(root, state)
    return count, changed, str(person.get('name') or f'Person {person_id}') if lost else None


def withdraw(root, person_id, copies, joins=(), fingerprint=_ANY) -> int:
    """Take the takes whose folder is in `copies` (relative to the clip folder `root` or absolute) out of the takes of person
    `person_id` in `root`'s people.json, drop every join of that person built on one of them and put back the person's own
    recorded `joins` that are current again; saved through save_people, and only when something changes. With a
    `fingerprint`, a people list made again since (another fingerprint) is left alone. How many takes were taken out; no
    file or folder is deleted."""
    return _withdraw(root, person_id, copies, joins, fingerprint)[0]


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


def take_apart(sequence, links) -> tuple[list[Path], list[str]]:
    """Withdraw the copies the rows of `links` recorded (putting back the receiver's own joins they recorded) and clear their
    `built`. (the clip folders whose people list changed, the names of receivers whose own join could not be put back). A
    receiving list made again since (another fingerprint) is left alone. A list that cannot be saved raises, and its links
    keep their record."""
    groups = {}
    for link in rows(sequence, links):
        if link.built:groups.setdefault((link.built.get('root'), link.built.get('person')), []).append(link)
    roots, lost = [], []
    for (root, person), group in groups.items():
        if root is not None:
            built = [link.built for link in group]
            mark = next((b['fingerprint'] for b in built if 'fingerprint' in b), _ANY)
            _, changed, name = _withdraw(root, person, [b['copy'] for b in built if 'copy' in b], [j for b in built for j in b.get('joins') or []], mark)
            if changed:roots.append(Path(root))
            if name:lost.append(name)
        for link in group:link.built = None
    return roots, lost


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
        reason = self.refusal(*self.judged(states, first.id))
        if reason:return self.stop(reason)
        take_apart(sequence, self.links);c._save();c._reload(self.root)
        self.own = [dict(j) for j in self.person().get('joins') or []]   # the receiver's own joins, which Unlink puts back
        w.activate_person(self.ident)
        self.i, self.stage = 0, 'importing'
        self.importing()

    def judged(self, states, first):
        """(states, receiving person) as they will be once the copies this row recorded are taken out, read without
        changing anything: the link rules and the import judge the take list the run will continue."""
        from copy import deepcopy
        from dataclasses import replace
        state = deepcopy(people_store.load_people(self.root))
        person = next((p for p in state['people'] if p['id'] == self.ident), None) if state else None
        if person is None:return states, None
        built = [l.built for l in rows(self.c.sequence, self.links) if l.built and l.built.get('person') == self.ident
                 and Path(l.built.get('root') or '').resolve() == self.root.resolve() and l.built.get('fingerprint', state.get('fingerprint')) == state.get('fingerprint')]
        _without(self.root, person, [b['copy'] for b in built if 'copy' in b], [j for b in built for j in b.get('joins') or []])
        fresh = sequence_state._people(self.root, {'people':[person]}, sequence_state.selected_kind(self.root))[0]
        old = states[first]
        return {**states, first:replace(old, people=[fresh if p.id == self.ident else p for p in old.people])}, person

    def refusal(self, states, person):
        """Why this row cannot be joined at all: a link rule, a transition its method cannot fill, the import's own refusal
        as far as it can be known before the first import, the Hidden ranges panel's own reason (FlowHMR, ExMR without
        Static Camera, HTD Refine), Automatic not set up; or None. Keeps the source take of each link."""
        from . import take_import_check
        from .parts_runner import join_issue
        sequence = self.c.sequence;self.sources = []
        for link in self.links:
            frames = sequence_links.gap_frames(sequence, link.target[0])
            reason = sequence_links.refusal(sequence, states, link.source, link.target, self.model) or sequence_links.gap_issue(link.method, frames)
            if reason:return reason
            later = Path(sequence.block(link.target[0]).folders[self.model])
            source = sequence_links._in_use(later, link.target[1])[0]   # the one take in use, as the row's state reads it
            reason = person is not None and take_import_check.refusal(self.root, person, source, frames, self.w.active_result)
            if reason:return reason
            self.sources.append((later, source, frames))
        reason = join_issue(self.w)
        if reason or not any(link.method == 'auto' for link in self.links):return reason
        from . import kimodo_runtime  # standard library only
        try:return kimodo_runtime.issue()
        except Exception:return kimodo_runtime.SETUP

    def importing(self):
        """Imports the take of the next later bar, after the transition of its pair of blocks."""
        from . import take_import_check
        if self.i == len(self.links):return self.methods()
        w = self.w;self.later, self.source, self.frames = self.sources[self.i]
        person = self.person()
        reason = take_import_check.refusal(self.root, person, self.source, self.frames, w.active_result)
        if reason:return self.stop(reason)
        self.before, self.stamp = {t['project'] for t in person.get('takes') or []}, sequence_links.result_stamp(self.source)
        self.mark = people_store.load_people(self.root).get('fingerprint')
        if not w.start_take_import(self.ident, self.source, self.frames):return self.stop(w.status.text() or NOT_STARTED)
        self.run(IMPORTING.format(self.name, self.i+1, len(self.links)))

    def ended_importing(self, failed, cancelled, status):
        """Records the copy on its link as soon as the import wrote it, whatever became of the export after it."""
        link = self.links[self.i]
        new = [t for t in self.person().get('takes') or [] if t['project'] not in self.before]
        if new:
            link.built = dict(copy=new[-1]['project'], take=_relative(self.source, self.later), stamp=self.stamp, gap=self.frames,
                              method=link.method, root=str(self.root.resolve()), person=self.ident, fingerprint=self.mark,
                              **({'joins':self.own} if self.i == 0 else {}))
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
            method = methods.get(gap['after']['project'])
            if method and (choices.get(gap_key(gap)) or {}).get('method') != method:choose(choices, gap_key(gap), method)   # a choice of that method keeps its seed and description
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
