"""Join of a linked row of the sequence in the real main window offscreen: the runner over the window's own job queue (take
import, the export queued after it, the gap methods, Fill hidden ranges), taking a join back, opening and exporting it.
The blocks are ExMR scenes with synthetic finished takes (tests/test_take_import.py Scene.write) in real clip folders. The
queue's next_job starts no worker: the test plays each job as it would end. Take import and Fill hidden ranges run for
real in-process (take_import.main, take_join.main) and their events go through the window's own handle_event; the export
job writes the bone preview the export leaves (tests/test_parts_window.py write_result). Needs torch, PySide6 and ffmpeg."""
import contextlib, importlib.util, io, json, os, shutil, subprocess, tempfile, unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np

import test_parts_window as tp
from mocap import paths, people_store as ps, sequence_links
from mocap.sequence import Sequence
from test_take_import import WALK, Scene
from test_take_join import WA, WB, WC, rigid, source, straight

APP, pump, FFMPEG = tp.APP, tp.pump, tp.FFMPEG
QUESTION = ("Join copies the linked takes into the scene of the first block and joins them there. In that block the joined "
            "animation becomes the person's take in use; the later blocks are not changed. Unlink puts the first block back. "
            "No file is deleted. Join now?")
TAKE_APART = 'This row is joined. Taking it apart puts the first block back to its own takes; no file is deleted. Continue?'
TAKE = 'p{}/takes/000000000_000000090_scene_v1'
arg = lambda args, name: args[args.index(name)+1]


def preview(d, kind='initial'):
    """What the export job leaves in take folder `d`: the bone preview and its receipt (tests/test_parts_window.py write_result)."""
    from mocap.mixamo_data import VERSION
    bvh = d/f'{kind}_motion.bvh';bvh.write_bytes(b'bvh '+d.name.encode())
    (d/f'{kind}_mixamo.json').write_text(json.dumps(dict(version=VERSION, kind=kind, fps=30., code_stats={'mixamo_data.py':tp.stamp(paths.ROOT/'mocap/mixamo_data.py')},
                                                     sources={}, outputs={bvh.name:tp.stamp(bvh)}, bvh_file=bvh.name)), encoding='utf-8')


def files(root):
    return {p.relative_to(root).as_posix():p.read_bytes() for p in Path(root).rglob('*') if p.is_file()}


@unittest.skipUnless(FFMPEG.is_file(), 'needs the bundled ffmpeg')
class WithdrawTest(unittest.TestCase):
    def test_withdraw_takes_copies_and_their_join_out(self):   # pure: people.json before/after; files still on disk; another person's takes untouched
        from mocap.sequence_join import withdraw
        tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);base = Path(tmp.name)
        video = tp.scene.video = base/'video.mp4'
        subprocess.run([str(FFMPEG), '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=64x48:rate=30', '-frames:v', '30', '-pix_fmt', 'yuv420p',
                        '-c:v', 'libx264', str(video)], check=True, creationflags=0x08000000)
        root = base/'scene';tp.scene(root, dict(frames=30, fps=30, start=0., end=1.), ((1, 'Anna', range(30)), (2, 'Ben', range(30))))
        own, copy, later, other = 'p1/takes/a', 'p1/takes/b_import_v1', 'p1/takes/c_import_v1', 'p2/takes/b_import_v1'
        for name in (own, copy, later, other, 'p1/takes/j', 'p1/takes/k', 'p2/takes/j'):(root/name).mkdir(parents=True);(root/name/'project.json').write_text('{}')
        state = ps.load_people(root);anna, ben = state['people']
        anna.update(takes=[dict(project=own, first=0, end=10), dict(project=copy, first=20, end=30), dict(project=later, first=40, end=50)],
                    joins=[dict(project='p1/takes/j', first=0, end=50, sources=[own, copy, later]), dict(project='p1/takes/k', first=0, end=10, sources=[own])],
                    gap_fill={'10_20':{'method':'smooth'}})
        ben.update(takes=[dict(project='p2/takes/a', first=0, end=10), dict(project=other, first=20, end=30)],
                   joins=[dict(project='p2/takes/j', first=0, end=30, sources=['p2/takes/a', other])])
        ps.save_people(root, state);before, ben_before = files(root), json.dumps(ben, sort_keys=True)
        self.assertEqual(withdraw(root, 1, [copy, str(root/later)]), 2)   # relative to the clip folder or absolute
        anna, ben = ps.load_people(root)['people']
        self.assertEqual(anna['takes'], [dict(project=own, first=0, end=10)])
        self.assertEqual([j['project'] for j in anna['joins']], ['p1/takes/k'])   # a join without the copies stays
        self.assertEqual(anna['gap_fill'], {'10_20':{'method':'smooth'}});self.assertEqual(json.dumps(ben, sort_keys=True), ben_before)
        after = files(root);self.assertEqual(set(after), set(before))   # no file or folder is deleted
        self.assertEqual({k:v for k, v in after.items() if k != 'people.json'}, {k:v for k, v in before.items() if k != 'people.json'})
        saved = after['people.json'];self.assertEqual(withdraw(root, 1, [copy]), 0);self.assertEqual(withdraw(root, 3, [other]), 0)
        self.assertEqual(files(root)['people.json'], saved)   # nothing to take out: nothing written
        self.assertEqual(withdraw(root, 2, [other], fingerprint='made again'), 0);self.assertEqual(files(root)['people.json'], saved)   # another list: left alone
        state = ps.load_people(root);del state['people'][1]['takes'];ps.save_people(root, state)
        self.assertEqual(withdraw(root, 2, [other]), 0);self.assertEqual(ps.load_people(root)['people'][1]['joins'], [])   # a person without a takes key: its join on the copy still goes


@unittest.skipUnless(importlib.util.find_spec('torch') and FFMPEG.is_file(), 'needs torch and the bundled ffmpeg')
class JoinTest(unittest.TestCase):
    def setUp(self):
        tp.application();tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);base = self.base = Path(tmp.name)
        home = mock.patch.object(paths, 'HOME', base);home.start();self.addCleanup(home.stop)
        models = base/'models';(models/'body_models/smplx').mkdir(parents=True)   # tests/test_take_import.py Scene.build
        rng = np.random.default_rng(0);template = rng.normal(size=(5, 3));self.pelvis = np.full(5, .2)@template
        np.savez(models/'body_models/smplx/SMPLX_NEUTRAL.npz', J_regressor=np.full((2, 5), .2), v_template=template, shapedirs=rng.normal(size=(5, 3, 10))*.01)
        self.videos = []
        for name, pattern in (('one', 'testsrc'), ('two', 'testsrc2'), ('three', 'rgbtestsrc')):   # 3 s each, 90 frames at 30 fps
            video = base/f'{name}.mp4';self.videos.append(video)
            subprocess.run([str(FFMPEG), '-v', 'error', '-f', 'lavfi', '-i', f'{pattern}=size=64x48:rate=30', '-frames:v', '90', '-pix_fmt', 'yuv420p',
                            '-c:v', 'libx264', str(video)], check=True, creationflags=0x08000000)
        from mocap import auto_fill, kimodo_runtime
        from mocap.scene_data import PeopleScene, SceneDataLoader
        self.generated = [];generate = straight(self.generated)
        for patch in (mock.patch.object(paths, 'MODELS', models), mock.patch('mocap.models.missing', lambda *a, **k:[]),
                      mock.patch.object(SceneDataLoader, 'load', lambda loader, root, kind, asset, cancelled:PeopleScene(root, dict(floor=0.))),
                      mock.patch.object(kimodo_runtime, 'issue', lambda:None), mock.patch.object(kimodo_runtime, 'text_issue', lambda:None),
                      mock.patch.object(auto_fill, 'generate', lambda folder, request, **kw:generate(request))):
            patch.start();self.addCleanup(patch.stop)
        from mocap.gui import MainWindow
        w = self.w = MainWindow()
        def close():
            w.close();w.deleteLater();APP.processEvents()
        self.addCleanup(close);w.show();APP.processEvents()
        for patch in (mock.patch('mocap.gui.missing', lambda *a, **k:[]), mock.patch.object(w, 'open_model_manager', lambda:None)):
            patch.start();self.addCleanup(patch.stop)
        self.c = w.sequence_ui;w.model_choice.setCurrentIndex(w.model_choice.findData('exmr'));pump(lambda:False, .2)
        self.jobs, self.played, self.real_next = [], 0, w.next_job
        def next_job():   # the queue's next step: a worker would start for the next job here
            if w.jobs and not w.failed and not w.cancelled:
                job = w.jobs.pop(0);self.jobs.append(job);w.current_stage = job[0];return
            self.real_next()
        patch = mock.patch.object(w, 'next_job', next_job);patch.start();self.addCleanup(patch.stop)

    # Building the blocks

    def scene(self, video, people, W, receiver, own=False):
        """A finished ExMR scene of the 3 s `video` after Find person and Reconstruct 3D: each person of `people` ((id, name))
        has one whole take, frames 0-90, or with `own` two takes 0-40 and 50-90 joined by their own Smooth join; the people
        list's fingerprint is the video's name. `people` None: reconstructed without Find person (no people list)."""
        root = self.base/'projects'/video.stem/'exmr';root.parent.mkdir(parents=True)
        manifest = dict(version=1, frames=90, fps=30, start=0., end=3., width=64, height=48, camera='static', source=dict(path=str(video), fps=30.),
                        pipeline=dict(model='exmr'), exmr_options=dict(static_camera=True, smoothing='none'), stages={})
        if people is None:   # tests/test_sequence_window.py tracked()
            root.mkdir();shutil.copy(video, root/'input.mp4')
            (root/'project.json').write_text(json.dumps(dict(manifest, pipeline=dict(model='exmr', result='initial'))), encoding='utf-8')
            tp.write_result(root, 'initial', b'result');return root
        tp.scene.video = video;tp.scene(root, manifest, [(ident, name, range(90)) for ident, name in people])
        np.save(root/'exmr_camera.npy', np.tile(np.eye(3, dtype=np.float32), (90, 1, 1)))
        state = ps.load_people(root)
        for person in state['people']:
            ident = person['id'];person['takes'] = []
            for a, b in ((0, 40), (50, 90)) if own else ((0, 90),):
                relative = f'p{ident}/takes/{a:09d}_{b:09d}_scene_v1';d = root/relative
                motion = source(a, b, W=W@rigid((0., .3*ident, 0.)), pelvis=self.pelvis, betas=.1 if receiver else -.2, walk=None if receiver else WALK)
                Scene.write(None, root, relative, motion, 'exmr', hands=receiver)
                m = json.loads((d/'project.json').read_text(encoding='utf-8'));m['person_id'] = ident
                (d/'project.json').write_text(json.dumps(m), encoding='utf-8');os.link(root/'input.mp4', d/'input.mp4');preview(d)
                person['takes'].append(dict(project=relative, first=a, end=b))
            if own:person['gap_fill'] = {'40_50':{'method':'smooth'}}
        state['fingerprint'] = video.stem;ps.save_people(root, state)
        if own:   # Fill hidden ranges of each person, as the job leaves it
            from mocap import take_join
            for person in state['people']:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(take_join.main(['--project', str(root), '--person-id', str(person['id']), '--result', 'initial']), 0)
                preview(root/f"p{person['id']}/takes/000000000_000000090_fill_v1")
        return root

    def blocks(self, *people, own=False):
        """A saved sequence of one block per entry of `people` (the people of its scene), opened in the window; with `own`
        the people of the first block have their own filled hidden range."""
        sequence = Sequence();self.roots = []
        for k, (video, persons, W) in enumerate(zip(self.videos, people, (WA, WB, WC))):
            root = self.scene(video, persons, W, k == 0, own and k == 0);self.roots.append(root)
            sequence.add(video, 3., folders={'exmr':str(root)})
        path = self.base/'projects'/'sequences'/'s.json';path.parent.mkdir(parents=True, exist_ok=True);sequence.save(path)
        self.assertTrue(self.c.open(path), self.w.status.text());APP.processEvents()
        if (self.roots[0]/'people.json').is_file():self.assertEqual(self.w.people_root, self.roots[0].resolve())
        return self.c.sequence.blocks

    def linked(self, n=2):
        """n blocks with one person each, linked by the app itself."""
        blocks = self.blocks(*[((1, name),) for name in ('Anna', 'Ben', 'Cleo')[:n]])
        self.assertEqual(len(self.c.sequence.links), n-1, self.w.status.text())
        return blocks

    # Playing the jobs

    def wait_job(self, stage):
        self.assertTrue(pump(lambda:len(self.jobs) > self.played or self.c.joiner is None), (len(self.jobs), self.w.status.text()))
        self.assertGreater(len(self.jobs), self.played, self.w.status.text());self.assertEqual(self.jobs[self.played][0], stage)
        return self.jobs[self.played][1]

    def play(self):
        """The running job ends as its worker would leave it, then the queue goes on."""
        from mocap import take_import, take_join
        w = self.w;stage, args = self.jobs[self.played];self.played += 1
        if stage in ('take_import', 'take_join'):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                (take_import if stage == 'take_import' else take_join).main(args)
            for line in out.getvalue().splitlines():
                event = json.loads(line[len('@@MOCAP@@'):]) if line.startswith('@@MOCAP@@') else None
                if event and event['type'] not in ('progress', 'complete'):w.handle_event(event)   # these two only move the task bar of a started worker
        elif stage == 'export':
            preview(Path(arg(args, '--project')), arg(args, '--result'))
        w.next_job()

    def fail_job(self, status='Processing failed. The GPU ran out of memory.', cancelled=False):
        w = self.w;self.played += 1;w.status.setText(status);w.failed, w.cancelled = not cancelled, cancelled;w.jobs.clear();w.next_job()

    def play_all(self):
        while True:
            pump(lambda:len(self.jobs) > self.played or self.c.joiner is None)
            if len(self.jobs) <= self.played:break
            self.play()
        self.assertIsNone(self.c.joiner, self.w.status.text())

    def join(self, target=None):
        with mock.patch.object(self.c, 'ask', return_value=True) as asked:self.assertTrue(self.c.join(target), self.w.status.text())
        self.assertEqual(asked.call_args.args[0], QUESTION)

    def person(self, k=0, ident=1):
        return next(p for p in ps.load_people(self.roots[k])['people'] if p['id'] == ident)

    def status(self, target):
        return sequence_links.status(self.c.sequence, {}, target, 'exmr')

    def saved(self):
        return json.loads(self.c.sequence.path.read_text(encoding='utf-8'))

    # The tests

    def test_join_two_blocks(self):            # ask -> True: import started with gap == 30 frames and the later person's take folder; after the batches: link.built set and saved, gap method chosen == the link's method, fill started for the receiving person; status 'Joined ...'; row state 'joined'
        first, second = self.linked();c, w = self.c, self.w;target = (second.id, 1);link = c.sequence.link_into(*target)
        self.assertEqual(self.status(target), 'linked');later = files(self.roots[1]);self.join()
        job = self.wait_job('take_import')
        self.assertEqual((arg(job, '--gap'), Path(arg(job, '--source')), arg(job, '--person-id'), Path(arg(job, '--project'))),
                         ('30', self.roots[1]/TAKE.format(1), '1', self.roots[0].resolve()))
        self.assertEqual(w.status.text(), 'Joining Anna: importing 1 of 1.')
        self.assertTrue(c.busy());self.assertFalse(c.load(second.id))   # the sequence is busy for the whole run
        self.play();self.assertEqual(self.wait_job('export'), ['--project', str(self.roots[0].resolve()/'p1/takes/000000120_000000210_import_v1'), '--result', 'initial'])
        self.play();job = self.wait_job('take_join')
        self.assertEqual((arg(job, '--person-id'), Path(arg(job, '--project'))), ('1', self.roots[0].resolve()))
        self.assertEqual(w.status.text(), 'Joining Anna: filling the transitions.')
        built = dict(copy='p1/takes/000000120_000000210_import_v1', take=TAKE.format(1), stamp=sequence_links.result_stamp(self.roots[1]/TAKE.format(1)),
                     gap=30, method='smooth', root=str(self.roots[0].resolve()), person=1, fingerprint='one', joins=[])
        self.assertEqual(link.built, built);self.assertEqual(self.saved()['links'][0]['built'], built)
        self.assertEqual(self.person()['gap_fill'], {'90_120':{'method':'smooth'}})   # the link's method for the new gap
        self.play_all()
        self.assertEqual([stage for stage, _ in self.jobs], ['take_import', 'export', 'take_join', 'export'])
        self.assertEqual(w.status.text(), 'Joined Anna across 2 blocks. Double-click the row to open it.')
        self.assertEqual(self.status(target), 'joined');self.assertEqual(c.view.row_states, {target:'joined'})
        self.assertFalse(c.busy());self.assertFalse(w.busy);self.assertEqual(c.current, first.id)
        self.assertEqual(w.project, (self.roots[0]/'p1/takes/000000000_000000210_fill_v1').resolve())   # the joined take is shown
        self.assertEqual(files(self.roots[1]), later)   # the later block is not changed

    def test_join_asks_first(self):            # ask -> False: no job started, nothing changed
        self.linked();c, w = self.c, self.w;before, people = c.sequence.to_dict(), files(self.roots[0])
        with mock.patch.object(c, 'ask', return_value=False) as asked:self.assertFalse(c.join())
        asked.assert_called_once_with(QUESTION, 'Join')
        self.assertEqual((self.jobs, w.busy, c.busy(), c.joiner), ([], False, False, None))
        self.assertEqual(c.sequence.to_dict(), before);self.assertEqual(files(self.roots[0]), people)

    def test_join_three_blocks_in_order(self): # two imports, the second after the first's export batch; one fill
        first, second, third = self.linked(3);c, w = self.c, self.w
        self.assertEqual(c.sequence.chain(first.id, 1), [(first.id, 1), (second.id, 1), (third.id, 1)])
        self.join();self.wait_job('take_import');self.play();self.wait_job('export')
        self.assertEqual(len(self.jobs), 2);self.play()   # the second import only once the first's export batch has ended
        job = self.wait_job('take_import')
        self.assertEqual((arg(job, '--gap'), Path(arg(job, '--source'))), ('30', self.roots[2]/TAKE.format(1)))
        self.assertEqual(w.status.text(), 'Joining Anna: importing 2 of 2.')
        self.play_all()
        self.assertEqual([stage for stage, _ in self.jobs], ['take_import', 'export', 'take_import', 'export', 'take_join', 'export'])
        self.assertEqual([(t['first'], t['end']) for t in self.person()['takes']], [(0, 90), (120, 210), (240, 330)])
        self.assertEqual(self.person()['gap_fill'], {'90_120':{'method':'smooth'}, '210_240':{'method':'smooth'}})
        self.assertEqual(w.status.text(), 'Joined Anna across 3 blocks. Double-click the row to open it.')
        self.assertEqual(self.status((third.id, 1)), 'joined');self.assertEqual(c.view.row_states, {(second.id, 1):'joined', (third.id, 1):'joined'})

    def test_import_refusal_stops(self):       # make the import's own check refuse (e.g. receiving scene without Static Camera): status 'Join stopped: <that text>', no built, window idle
        from mocap import take_import_check as check
        first, second = self.linked();c, w = self.c, self.w
        take = self.roots[1]/TAKE.format(1);m = json.loads((take/'project.json').read_text(encoding='utf-8'))
        m['pipeline']['model'] = 'gvhmr';(take/'project.json').write_text(json.dumps(m), encoding='utf-8')   # a take of another model
        c.refresh();self.assertEqual(c.view.states[second.id].kind, 'tracked')   # the link rules do not see it
        before = files(self.roots[0]);self.join()
        self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), 'Join stopped: '+check.MODEL.format('ExMR'))
        self.assertEqual(self.jobs, []);self.assertIsNone(c.sequence.links[0].built);self.assertFalse(w.busy);self.assertFalse(c.busy())
        self.assertEqual(c.current, first.id);self.assertEqual(self.status((second.id, 1)), 'linked')
        self.assertEqual({k:v for k, v in files(self.roots[0]).items() if k != 'people.json'}, {k:v for k, v in before.items() if k != 'people.json'})
        self.assertEqual(self.person()['takes'], [dict(project=TAKE.format(1), first=0, end=90)])

    def test_refusals_before_anything_is_imported(self):   # the link rules and the Hidden ranges panel's reason stop the run before an import
        first, second = self.linked();c, w = self.c, self.w
        with mock.patch('mocap.parts_runner.join_issue', return_value='Hidden ranges do not support HTD Refine results in this version.'):self.join()
        self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), 'Join stopped: Hidden ranges do not support HTD Refine results in this version.');self.assertEqual(self.jobs, [])
        for take in self.person(1)['takes']:shutil.rmtree(self.roots[1]/take['project'])   # Ben's take is gone
        self.join();self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), "Join stopped: Ben's take is not finished.");self.assertEqual(self.jobs, [])

    def test_failed_job_stops_cleanly(self):   # play the import batch as failed: 'Join stopped: ...', runner not running, sequence not busy, next join() can start
        first, second = self.linked();c, w = self.c, self.w
        self.join();self.wait_job('take_import');runner = c.joiner
        self.fail_job();self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), 'Join stopped: Processing failed. The GPU ran out of memory.')
        self.assertFalse(runner.running);self.assertFalse(c.busy());self.assertFalse(w.busy);self.assertIsNone(c.sequence.links[0].built)
        self.assertEqual(c.current, first.id);self.assertEqual(w.people_root, self.roots[0].resolve());self.assertEqual(self.status((second.id, 1)), 'linked')
        self.assertTrue(c.load(second.id))   # the sequence is free again
        self.join();self.wait_job('take_import');self.play_all()
        self.assertEqual(self.status((second.id, 1)), 'joined')

    def test_cancel_stops_between_jobs(self):
        first, second = self.linked();c, w = self.c, self.w
        self.join();self.wait_job('take_import');self.play();self.wait_job('export');self.play()
        w.batch_serial += 1;w.busy, w.cancelled, w.failed = True, False, False   # a job the window starts by itself after the batch
        self.assertFalse(pump(lambda:len(self.jobs) > 2, .6))
        w.cancelled, w.busy = True, False   # Cancel Job on that job
        self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), 'Join stopped: cancelled');self.assertEqual(len(self.jobs), 2)
        link = c.sequence.links[0];self.assertEqual(link.built['copy'], 'p1/takes/000000120_000000210_import_v1')   # recorded, so the next Join takes it out first
        self.assertEqual(self.status((second.id, 1)), 'out of date');self.assertFalse(c.busy())
        self.join();job = self.wait_job('take_import')
        self.assertEqual(self.person()['takes'], [dict(project=TAKE.format(1), first=0, end=90)])   # the earlier copy is out of the take list
        self.assertEqual(arg(job, '--gap'), '30')
        c.joiner.cancel();self.play();self.wait_job('export');self.play()   # Cancel Job with no job of the window running: the run stops before the next job
        self.assertTrue(pump(lambda:c.joiner is None));self.assertEqual(w.status.text(), 'Join stopped: cancelled');self.assertEqual(len(self.jobs), 4)

    def test_cancel_job_during_a_job_stops_the_run(self):
        first, second = self.linked();c, w = self.c, self.w
        self.join();self.wait_job('take_import');self.assertTrue(w.cancel_button.isEnabled())
        w.cancel_button.click();self.assertTrue(w.cancelled)
        self.fail_job('Cancelled. Completed stages will be reused next time.', cancelled=True)
        self.assertTrue(pump(lambda:c.joiner is None));self.assertEqual(w.status.text(), 'Join stopped: cancelled');self.assertEqual(len(self.jobs), 1)

    def test_a_closed_window_starts_nothing_more(self):
        first, second = self.linked();c, w = self.c, self.w
        self.join();self.wait_job('take_import');runner = c.joiner;w.hide();self.play();self.play()
        self.assertTrue(pump(lambda:c.joiner is None));self.assertFalse(runner.running);self.assertEqual(len(self.jobs), 2)

    def test_join_again_after_gap_change(self):  # joined; set_transition(auto, 3.0) -> row 'out of date'; join -> earlier copy withdrawn first, import with gap 90
        first, second = self.linked();c, w = self.c, self.w;target = (second.id, 1)
        self.join();self.play_all();self.assertEqual(self.status(target), 'joined')
        self.assertTrue(c.set_transition(target, 'auto', 3.));self.assertEqual(self.status(target), 'out of date')
        self.assertEqual(c.view.row_states, {target:'out of date'})
        self.join();job = self.wait_job('take_import')
        anna = self.person();self.assertEqual(anna['takes'], [dict(project=TAKE.format(1), first=0, end=90)]);self.assertEqual(anna.get('joins') or [], [])
        self.assertIsNone(c.sequence.links[0].built)
        self.assertEqual(arg(job, '--gap'), '90');self.play_all()
        self.assertEqual([(t['first'], t['end']) for t in self.person()['takes']], [(0, 90), (180, 270)])
        self.assertEqual(self.person()['gap_fill']['90_180'], {'method':'auto', 'seed':0});self.assertEqual(len(self.generated), 1)
        self.assertEqual(self.status(target), 'joined');self.assertEqual(c.sequence.links[0].built['gap'], 90)
        for old in ('p1/takes/000000120_000000210_import_v1', 'p1/takes/000000000_000000210_fill_v1'):self.assertTrue((self.roots[0]/old).is_dir(), old)

    def test_unlink_takes_the_join_back(self):   # ask -> True: copies out of the take list, the person's own take is in use again, link gone, files on disk
        first, second = self.linked();c, w = self.c, self.w;target = (second.id, 1)
        self.join();self.play_all();on_disk = set(files(self.roots[0]))
        with mock.patch.object(c, 'ask', return_value=False) as asked:self.assertFalse(c.unlink(target))
        asked.assert_called_once_with(TAKE_APART, 'Joined Row');self.assertEqual(len(c.sequence.links), 1);self.assertEqual(self.status(target), 'joined')
        with mock.patch.object(c, 'ask', return_value=True):self.assertTrue(c.unlink(target))
        anna = self.person();self.assertEqual(anna['takes'], [dict(project=TAKE.format(1), first=0, end=90)]);self.assertEqual(anna.get('joins') or [], [])
        self.assertEqual([t['project'] for t in ps.active_takes(self.roots[0], anna, 'initial')], [TAKE.format(1)])
        self.assertEqual(c.sequence.links, []);self.assertNotIn('links', self.saved())
        self.assertTrue(on_disk <= set(files(self.roots[0])))   # every file is still on disk
        self.assertEqual(w.project, (self.roots[0]/TAKE.format(1)).resolve())   # the window shows the person's own take again
        self.assertEqual([w.person_takes.itemData(i) for i in range(w.person_takes.count())], [TAKE.format(1)])

    def test_an_edit_takes_the_join_back_after_asking(self):   # Delete of a block of a joined row asks first; No changes nothing, Yes takes the copies out
        first, second = self.linked();c, w = self.c, self.w
        self.join();self.play_all();before = c.sequence.to_dict()
        with mock.patch.object(c, 'ask', return_value=False) as asked:c.delete(second.id)
        asked.assert_called_once_with(TAKE_APART, 'Joined Row');self.assertEqual(c.sequence.to_dict(), before)
        with mock.patch.object(c, 'ask', return_value=True):c.delete(second.id)
        self.assertEqual(len(c.sequence.blocks), 1);self.assertEqual(c.sequence.links, [])
        self.assertEqual(self.person()['takes'], [dict(project=TAKE.format(1), first=0, end=90)])

    def test_two_rows_do_not_touch_each_other(self):  # rows A and B across the same pair: joining / unlinking A leaves B's person's takes and built untouched
        first, second = self.blocks(((1, 'Anna'), (2, 'Ben')), ((1, 'Cleo'), (2, 'Dan')));c = self.c
        self.assertTrue(c.link((first.id, 1), (second.id, 1)));self.assertTrue(c.link((first.id, 2), (second.id, 2)))
        self.join((first.id, 2));self.play_all();self.assertEqual(self.status((second.id, 2)), 'joined')
        ben, built = self.person(0, 2), dict(c.sequence.link_into(second.id, 2).built)
        self.join((second.id, 1));self.play_all()   # row A, from its later bar
        self.assertEqual(self.status((second.id, 1)), 'joined');self.assertEqual(self.w.status.text(), 'Joined Anna across 2 blocks. Double-click the row to open it.')
        self.assertEqual((self.person(0, 2), c.sequence.link_into(second.id, 2).built), (ben, built));self.assertEqual(self.status((second.id, 2)), 'joined')
        with mock.patch.object(c, 'ask', return_value=True):self.assertTrue(c.unlink((second.id, 1)))
        self.assertEqual(self.person()['takes'], [dict(project=TAKE.format(1), first=0, end=90)])
        self.assertEqual((self.person(0, 2), c.sequence.link_into(second.id, 2).built), (ben, built));self.assertEqual(self.status((second.id, 2)), 'joined')
        self.assertEqual(self.saved()['links'][0]['built'], built)

    def test_open_and_export_joined(self):       # open_joined: first block loaded, clip level, receiving person active on the joined take; export_joined triggers the window's export action once
        first, second = self.linked();c, w = self.c, self.w;target = (second.id, 1)
        joined = (self.roots[0]/'p1/takes/000000000_000000210_fill_v1').resolve()
        self.join();self.play_all();c.back();self.assertTrue(c.load(second.id));self.assertFalse(c.at_clip_level())
        self.assertTrue(c.open_joined(target))
        self.assertEqual((c.current, c.at_clip_level(), w.active_person_id, w.project), (first.id, True, 1, joined))
        self.assertEqual(w.person_takes.currentData(), 'p1/takes/000000000_000000210_fill_v1')
        triggered = [];w.export_action.triggered.connect(lambda *_:triggered.append(1))
        c.back();self.assertTrue(c.load(second.id))
        with mock.patch('mocap.source_export_dialog.SourceExportDialog') as dialog:self.assertTrue(c.export_joined(target))
        self.assertEqual(len(triggered), 1);dialog.assert_called_once();self.assertEqual(w.project, joined)
        menu = c._person_menu(second.id, 1);self.addCleanup(menu.deleteLater)
        self.assertEqual([a.text() for a in menu.actions()], ['Continue after', 'Unlink', 'Join', 'Open joined animation', 'Export joined animation...'])
        with mock.patch.object(c, 'open_joined') as opened, mock.patch.object(c, 'open_block') as block:
            c.view.opened.emit(second.id, 1);c.view.opened.emit(first.id, 1)   # double-click on a bar of the joined row
            c.set_transition(target, 'auto', 3.);c.view.opened.emit(second.id, 1)   # out of date: the block as before
        self.assertEqual([call.args for call in opened.call_args_list], [((second.id, 1),), ((first.id, 1),)])
        block.assert_called_once_with(second.id, 1)

    def test_menu_of_a_linked_bar_offers_join(self):
        first, second = self.linked();c = self.c
        menu = c._person_menu(first.id, 1);self.addCleanup(menu.deleteLater)
        self.assertEqual([a.text() for a in menu.actions()], ['Continue after', 'Join'])
        with mock.patch.object(c, 'join') as join:menu.actions()[1].trigger()
        join.assert_called_once_with((first.id, 1))
        self.assertTrue(c.join_button.isEnabled());self.assertEqual(c.join_button.text(), 'Join')

    def test_people_import_dialog_path_unchanged(self):  # the People row action still starts the same job with the dialog's values (existing take-import window tests stay green)
        from mocap import take_import_check as check
        self.linked();w = self.w;other = self.roots[1]/TAKE.format(1)
        with mock.patch('mocap.take_import_ui.ask', lambda parent, person:(other, 20.)):w.people_list.importRequested.emit(1)
        self.assertEqual((w.status.text(), self.jobs, w.busy), (check.GAP, [], False))
        with mock.patch('mocap.take_import_ui.ask', lambda parent, person:(other, 1.)):w.people_list.importRequested.emit(1)
        self.assertEqual(self.jobs, [('take_import', ['--project', str(self.roots[0].resolve()), '--person-id', '1', '--source', str(other), '--gap', '30',
                                                     '--result', 'initial'])])
        self.assertEqual((w._people_batch, w._people_return_id, w._import_result, w._hidden_refresh), (True, 1, 'initial', True))
        self.play();self.play();self.assertFalse(w.busy)
        self.assertEqual(self.person()['takes'][-1], dict(project='p1/takes/000000120_000000210_import_v1', first=120, end=210))

    # Fix round 1

    def test_join_again_after_a_failed_export(self):   # import played, its export failed: the next Join takes the recorded copy out first and joins
        first, second = self.linked();c, w = self.c, self.w;target = (second.id, 1)
        self.join();self.wait_job('take_import');self.play();self.wait_job('export');self.fail_job()
        self.assertTrue(pump(lambda:c.joiner is None));self.assertEqual(w.status.text(), 'Join stopped: Processing failed. The GPU ran out of memory.')
        self.assertEqual(c.sequence.links[0].built['copy'], 'p1/takes/000000120_000000210_import_v1');self.assertEqual(self.status(target), 'out of date')
        self.join();job = self.wait_job('take_import')   # not refused because of the copy without its preview
        self.assertEqual(self.person()['takes'], [dict(project=TAKE.format(1), first=0, end=90)]);self.assertEqual(arg(job, '--gap'), '30')
        self.play_all();self.assertEqual(self.status(target), 'joined')

    def test_a_fill_that_fails_stops_cleanly(self):
        first, second = self.linked();c, w = self.c, self.w;target = (second.id, 1)
        self.join();self.wait_job('take_import');self.play();self.play();self.wait_job('take_join')
        self.fail_job('Processing failed. Automatic fill could not connect the motion.')
        self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), 'Join stopped: Processing failed. Automatic fill could not connect the motion.')
        self.assertFalse(c.busy());self.assertEqual(self.status(target), 'out of date');self.assertEqual(c.current, first.id)

    def test_an_error_inside_the_run_stops_it(self):
        first, second = self.linked();c, w = self.c, self.w
        with mock.patch('mocap.hidden_ranges_state.choose', side_effect=RuntimeError('boom')):
            self.join();runner = c.joiner;self.wait_job('take_import');self.play();self.play()
            self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), 'Join stopped: boom');self.assertFalse(runner.running);self.assertFalse(runner.timer.isActive())
        self.assertFalse(c.busy());self.assertEqual(len(self.jobs), 2)

    def test_a_receiver_without_a_people_list_cannot_be_linked(self):   # automatic link not made, link and menu entry refused with the reason, nothing runs
        first, second = self.blocks(None, ((1, 'Ben'),));c, w = self.c, self.w
        text = 'Run Find person in one.mp4 first. A block reconstructed without it has no person to continue.'
        self.assertEqual(c.sequence.links, [])
        self.assertFalse(c.link((first.id, 1), (second.id, 1)));self.assertEqual(w.status.text(), text)
        from mocap.tooltips import plain_tooltip
        menu = c._person_menu(second.id, 1);self.addCleanup(menu.deleteLater);entry = menu.actions()[0].menu().actions()[0]
        self.assertFalse(entry.isEnabled());self.assertEqual(plain_tooltip(entry.toolTip()), text)
        self.assertFalse(c.join());self.assertEqual(self.jobs, [])

    def test_join_skips_rows_that_are_joined(self):   # Join all imports only the row not joined yet; with all joined it says so and asks nothing; the menu's Join of a joined row only says so
        first, second = self.blocks(((1, 'Anna'), (2, 'Ben')), ((1, 'Cleo'), (2, 'Dan')));c, w = self.c, self.w
        self.assertTrue(c.link((first.id, 1), (second.id, 1)));self.assertTrue(c.link((first.id, 2), (second.id, 2)))
        self.join((first.id, 2));self.play_all();n = len(self.jobs)
        self.join();self.play_all()
        self.assertEqual([arg(a, '--person-id') for s, a in self.jobs[n:] if s in ('take_import', 'take_join')], ['1', '1'])
        self.assertEqual(w.status.text(), 'Joined Anna across 2 blocks. Double-click the row to open it.')
        with mock.patch.object(c, 'ask') as asked:
            self.assertFalse(c.join());self.assertEqual(w.status.text(), 'Every linked row is already joined.')
            self.assertFalse(c.join((second.id, 2)));self.assertEqual(w.status.text(), 'Ben is already joined.')
        asked.assert_not_called();self.assertEqual(len(self.jobs), n+4);self.assertIsNone(c.joiner)

    def test_a_choice_tuned_in_the_panel_is_kept(self):   # same gap, same method: the user's seed and description stay
        first, second = self.linked();c, w = self.c, self.w;target = (second.id, 1)
        self.assertTrue(c.set_transition(target, 'auto', 1.));self.join();self.play_all();self.assertEqual(self.status(target), 'joined')
        w.gap_choice_changed('90_120', 'auto', seed=7);self.assertEqual(self.status(target), 'out of date')   # a description would need the text runtime
        self.join();self.play_all()
        self.assertEqual(self.person()['gap_fill']['90_120'], {'method':'auto', 'seed':7})
        self.assertEqual(self.status(target), 'joined');self.assertEqual(len(self.generated), 2)

    def test_unlink_puts_the_receivers_own_join_back(self):   # receiver with an own filled join -> Join -> Unlink -> the takes in use are the own join again
        own = 'p1/takes/000000000_000000090_fill_v1'
        first, second = self.blocks(((1, 'Anna'),), ((1, 'Ben'),), own=True);c, w = self.c, self.w;target = (second.id, 1)
        self.assertEqual([t['project'] for t in ps.active_takes(self.roots[0], self.person(), 'initial')], [own])
        self.assertEqual(len(c.sequence.links), 1, w.status.text())
        self.join();self.play_all();self.assertEqual(self.status(target), 'joined')
        self.assertEqual([t['project'] for t in ps.active_takes(self.roots[0], self.person(), 'initial')], ['p1/takes/000000000_000000210_fill_v1'])
        with mock.patch.object(c, 'ask', return_value=True):self.assertTrue(c.unlink(target))
        anna = self.person();self.assertEqual([t['project'] for t in ps.active_takes(self.roots[0], anna, 'initial')], [own])
        self.assertEqual([j['project'] for j in anna['joins']], [own]);self.assertEqual(w.project, (self.roots[0]/own).resolve())
        self.assertTrue(c.link((first.id, 1), target));self.join();self.play_all()   # the own join is recorded again by this Join
        state = ps.load_people(self.roots[0]);state['people'][0]['gap_fill']['40_50'] = {'method':'auto', 'seed':0};ps.save_people(self.roots[0], state)
        with mock.patch.object(c, 'ask', return_value=True):self.assertTrue(c.unlink(target))
        self.assertEqual(w.status.text(), 'Fill hidden ranges again for Anna in the first block to rejoin their own takes.')
        self.assertEqual(self.person().get('joins') or [], [])

    def test_a_list_made_again_keeps_its_takes(self):   # Find person again in the receiving block: the stale link's record is cleared, its people list is not touched
        first, second = self.linked();c, w = self.c, self.w
        self.join();self.play_all()
        state = ps.load_people(self.roots[0]);state['fingerprint'] = 'again';ps.save_people(self.roots[0], state);before = files(self.roots[0])['people.json']
        c.refresh();self.assertEqual(c.sequence.dropped, []);self.assertEqual(files(self.roots[0])['people.json'], before)
        self.assertEqual([(l.prints, l.built) for l in c.sequence.links], [(('again', 'two'), None)])   # linked again by itself, with no record

    def test_a_failed_take_back_is_tried_again(self):   # a list that cannot be saved keeps the dropped links, and the next change takes the copies out
        first, second = self.linked();c, w = self.c, self.w
        self.join();self.play_all()
        with mock.patch.object(ps, 'save_people', side_effect=OSError('disk full')), mock.patch.object(c, 'ask', return_value=True):
            self.assertTrue(c.unlink((second.id, 1)))
        self.assertEqual(w.status.text(), 'Cannot take the join back: disk full');self.assertEqual(len(c.sequence.dropped), 1)
        self.assertEqual(len(self.person()['takes']), 2)
        c._changed();self.assertEqual(c.sequence.dropped, []);self.assertEqual(self.person()['takes'], [dict(project=TAKE.format(1), first=0, end=90)])

    # Final fix wave

    def test_a_copy_is_recorded_while_its_export_runs(self):   # the import wrote its take, the batch has not ended: a tick records it and saves; the batch then fails: Join again leaves one copy
        from mocap import take_import
        first, second = self.linked();c, w = self.c, self.w;target = (second.id, 1);copy = 'p1/takes/000000120_000000210_import_v1'
        self.join();args = self.wait_job('take_import')
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):take_import.main(args)   # the worker's take_imported effect
        for line in out.getvalue().splitlines():
            event = json.loads(line[len('@@MOCAP@@'):]) if line.startswith('@@MOCAP@@') else None
            if event and event['type'] not in ('progress', 'complete'):w.handle_event(event)
        self.assertEqual(self.person()['takes'][-1]['project'], copy);self.assertTrue(w.busy)   # the batch has not ended
        c.joiner.tick()
        self.assertEqual(c.sequence.links[0].built['copy'], copy);self.assertEqual(self.saved()['links'][0]['built']['copy'], copy)
        self.fail_job();self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), 'Join stopped: Processing failed. The GPU ran out of memory.')
        self.join();self.play_all();self.assertEqual(self.status(target), 'joined')
        self.assertEqual(len([t for t in self.person()['takes'] if '_import_' in t['project']]), 1, self.person()['takes'])

    def test_an_old_link_into_a_block_without_a_people_list_says_why(self):
        first, second = self.blocks(None, ((1, 'Ben'),));c, w = self.c, self.w
        c.sequence.link((first.id, 1), (second.id, 1));self.join();self.assertTrue(pump(lambda:c.joiner is None))
        self.assertEqual(w.status.text(), 'Join stopped: Run Find person in one.mp4 first. A block reconstructed without it has no person to continue.')
        self.assertEqual(self.jobs, [])

    def test_a_double_click_while_a_job_runs_opens_the_loaded_block(self):
        first, second = self.linked();c, w = self.c, self.w
        self.join();self.play_all();c.back();self.assertFalse(c.at_clip_level());w.busy = True
        try:c.view.opened.emit(first.id, 1);self.assertTrue(c.at_clip_level())
        finally:w.busy = False


if __name__ == '__main__':
    unittest.main()
