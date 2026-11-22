import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap import auto_fill, kimodo_runtime

PREFIX = '@@MOCAP@@'


def request(seed=7, frames=20, before=45, after=30):
    rng = np.random.default_rng(0)
    side = lambda n: (rng.normal(size=(n, 3)), rng.normal(size=(n, 63)), rng.normal(size=(n, 3)))
    b, a = side(before), side(after)
    return dict(version=3, before_orient=b[0], before_pose=b[1], before_pelvis=b[2], after_orient=a[0], after_pose=a[1], after_pelvis=a[2],
                rest=rng.normal(size=(22, 3)), sole=-1.3, frames=frames, seed=seed, fps=30., text=np.array(''), text_feature=np.zeros(4096, np.float32))


def response(req, rows=None, end_pose=.8, seam=(1.5, 2.5)):
    c0, c1 = min(5, len(req['before_pelvis'])), min(5, len(req['after_pelvis']));rows = c0+int(req['frames'])+c1 if rows is None else rows
    return dict(version=np.int64(3), context=np.array([c0, c1], np.int64), orient=np.zeros((rows, 3)), pose=np.full((rows, 63), .1),
                pelvis=np.ones((rows, 3)), contacts=np.zeros((rows, 4), bool), end_pose_cm=np.float64(end_pose), seam_cm=np.array(seam, np.float64))


class FakeRun:
    """Stands in for the worker: reads the request file and writes `answer(request)` to the response path."""
    def __init__(self, answer=response, code=0, lines=()):self.answer, self.code, self.lines, self.calls = answer, code, list(lines), []

    def __call__(self, command):
        self.calls.append(command);args = dict(zip(command[1::2], command[2::2]))
        with np.load(args['--request'], allow_pickle=False) as z:req = {k: z[k] for k in z.files}
        self.requests = getattr(self, 'requests', [])+[req];out = self.answer(req)
        if out is not None:np.savez(args['--response'], **out)
        return self.code, self.lines


class KeyTest(unittest.TestCase):
    def test_key(self):
        r = request();k = auto_fill.key(r, 'id')
        self.assertEqual(k, auto_fill.key(dict(reversed(list(r.items()))), 'id'))   # dict order does not matter
        self.assertNotEqual(k, auto_fill.key(r, 'other'));self.assertNotEqual(k, auto_fill.key(request(seed=8), 'id'))
        for name in ('before_pose', 'after_pelvis', 'rest'):
            changed = dict(r, **{name: r[name].copy()});changed[name][3, 1] += 1e-9
            self.assertNotEqual(k, auto_fill.key(changed, 'id'), name)
        self.assertNotEqual(k, auto_fill.key(dict(r, sole=-1.2), 'id'));self.assertNotEqual(k, auto_fill.key(dict(r, frames=21), 'id'))
        with mock.patch.object(auto_fill, 'VERSION', 1):self.assertNotEqual(k, auto_fill.key(r, 'id'))   # a version-1 cache entry is never found

    def test_identity_holds_the_workers_code(self):
        with tempfile.TemporaryDirectory() as name:
            files = [Path(name)/'kimodo_fill.py', Path(name)/'soma_convert.py']
            for f in files:f.write_bytes(b'code of '+f.name.encode())
            with mock.patch.object(auto_fill, 'SOURCES', tuple(files)):
                first = auto_fill.identity();self.assertTrue(first.startswith(kimodo_runtime.identity()+':'), first)
                for f in files:
                    f.write_bytes(f.read_bytes()+b' changed');self.assertNotEqual(auto_fill.identity(), first, f.name);first = auto_fill.identity()
        self.assertEqual([f.name for f in auto_fill.SOURCES], ['kimodo_fill.py', 'soma_convert.py']);self.assertTrue(all(f.is_file() for f in auto_fill.SOURCES))


    def test_text_cache_is_in_the_data_folder(self):
        from mocap.paths import HOME
        self.assertEqual(auto_fill.TEXT_CACHE, HOME/'cache'/'kimodo_text')


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder = Path(self.tmp.name)/'person'/'auto_fill'
        for name, value in (('issue', lambda: None), ('command', lambda args: ['kimodo', *map(str, args)])):
            patch = mock.patch.object(kimodo_runtime, name, value);patch.start();self.addCleanup(patch.stop)
        self.request = request();self.done = self.folder/(auto_fill.key(self.request, auto_fill.identity())+'.npz')

    def files(self):return sorted(p.name for p in self.folder.iterdir()) if self.folder.exists() else []

    def test_generates_once_then_serves_the_cache(self):
        run = FakeRun();lines = []
        out = auto_fill.generate(self.folder, self.request, run=run, echo=lines.append)
        self.assertEqual(len(run.calls), 1);self.assertEqual(self.files(), [self.done.name])
        self.assertEqual(out['pose'].shape, (30, 63));self.assertEqual(float(out['end_pose_cm']), .8);np.testing.assert_array_equal(out['seam_cm'], [1.5, 2.5])
        sent = run.requests[0];self.assertEqual(set(sent), set(self.request))
        for k, v in self.request.items():np.testing.assert_array_equal(sent[k], v)
        again = auto_fill.generate(self.folder, self.request, run=run)
        self.assertEqual(len(run.calls), 1);np.testing.assert_array_equal(again['pose'], out['pose'])   # served from the cache

    def test_failed_worker(self):
        lines = ['@@MOCAP@@{"type": "progress", "stage": "take_join", "current": 0, "total": 100, "message": "x"}',
                 PREFIX+json.dumps(dict(type='failed', stage='take_join', message='Not enough free video memory for automatic fill.'))]
        for code, found, message in ((1, lines, 'Not enough free video memory for automatic fill.'), (2, [], auto_fill.STOPPED), (1, ['noise'], auto_fill.STOPPED)):
            with self.assertRaises(auto_fill.Unavailable) as caught:auto_fill.generate(self.folder, self.request, run=FakeRun(lambda r: None, code, found))
            self.assertEqual(str(caught.exception), message);self.assertEqual(self.files(), [])

    def test_no_usable_motion(self):
        for answer in (lambda r: None, lambda r: response(r, rows=29), lambda r: dict(response(r), contacts=np.zeros((30, 4))),
                       lambda r: dict(response(r), end_pose_cm=np.float64(np.nan)), lambda r: {k: v for k, v in response(r).items() if k != 'pelvis'},
                       lambda r: {k: v for k, v in response(r).items() if k != 'seam_cm'}, lambda r: response(r, seam=(1., np.nan)),
                       lambda r: response(r, seam=(1., 2., 3.)), lambda r: dict(response(r), version=np.int64(1))):
            with self.assertRaises(auto_fill.Unavailable) as caught:auto_fill.generate(self.folder, self.request, run=FakeRun(answer))
            self.assertEqual(str(caught.exception), 'Automatic fill returned no usable motion.');self.assertEqual(self.files(), [])

    def test_bad_cache_is_regenerated(self):
        self.folder.mkdir(parents=True);np.savez(self.done, **response(self.request, rows=25))
        run = FakeRun();out = auto_fill.generate(self.folder, self.request, run=run)
        self.assertEqual(len(run.calls), 1);self.assertEqual(out['orient'].shape, (30, 3));self.assertEqual(self.files(), [self.done.name])

    def test_older_cache_is_not_trusted(self):
        self.folder.mkdir(parents=True)
        for version in (1, 2):
            old = {k: v for k, v in response(self.request).items() if k != 'seam_cm' or version == 2};old['version'] = np.int64(version)
            np.savez(self.done, **old);run = FakeRun();out = auto_fill.generate(self.folder, self.request, run=run)
            self.assertEqual(len(run.calls), 1);self.assertEqual(int(out['version']), 3);self.assertEqual(self.files(), [self.done.name])

    def test_description_is_part_of_the_key(self):
        k = auto_fill.key(self.request, 'id');feature = np.zeros(4096, np.float32);feature[9] = 1
        self.assertNotEqual(k, auto_fill.key(dict(self.request, text=np.array('A person waves.'), text_feature=feature), 'id'))
        self.assertNotEqual(auto_fill.key(dict(self.request, text=np.array('A person waves.'), text_feature=feature), 'id'),
                            auto_fill.key(dict(self.request, text=np.array('A person sits.'), text_feature=feature), 'id'))
        self.assertEqual(auto_fill.VERSION, 3)

    def test_cleanup_failure_keeps_the_result(self):
        real = Path.unlink
        def unlink(path, missing_ok=False):
            if path.name.endswith(('.request.npz', '.partial.npz.tmp')):raise PermissionError(13, 'in use', str(path))
            return real(path, missing_ok=missing_ok)
        with mock.patch.object(Path, 'unlink', unlink):out = auto_fill.generate(self.folder, self.request, run=FakeRun())
        self.assertEqual(out['pose'].shape, (30, 63));self.assertTrue(self.done.is_file())

    def test_working_files_that_cannot_be_written(self):
        def refuse(*args, **kw):raise PermissionError(13, 'Access is denied', 'auto_fill')
        for target in ((Path, 'mkdir'), (np, 'savez')):
            run = FakeRun()
            with mock.patch.object(*target, refuse), self.assertRaises(auto_fill.Unavailable) as caught:auto_fill.generate(self.folder, self.request, run=run)
            self.assertTrue(str(caught.exception).startswith('Automatic fill could not write its working files: '), str(caught.exception))
            self.assertIn('Access is denied', str(caught.exception));self.assertEqual(run.calls, []);self.assertEqual(self.files(), [])

    def test_leftovers_of_a_killed_run(self):
        self.folder.mkdir(parents=True);stem = self.done.name[:-4]
        for suffix in ('.request.npz', '.partial.npz', '.partial.npz.tmp'):(self.folder/(stem+suffix)).write_bytes(b'left by a killed run')
        run = FakeRun();auto_fill.generate(self.folder, self.request, run=run)
        self.assertEqual(len(run.calls), 1);self.assertEqual(self.files(), [self.done.name])

    def test_missing_runtime_writes_nothing(self):
        run = FakeRun()
        with mock.patch.object(kimodo_runtime, 'issue', lambda: kimodo_runtime.SETUP):
            with self.assertRaises(auto_fill.Unavailable) as caught:auto_fill.generate(self.folder, self.request, run=run)
        self.assertEqual(str(caught.exception), kimodo_runtime.SETUP);self.assertEqual(run.calls, []);self.assertFalse(self.folder.exists())

    def test_default_run_echoes_progress_and_keeps_the_failed_event(self):
        script = "print('@@MOCAP@@{\"type\": \"progress\", \"message\": \"step\"}', flush=True);print('@@MOCAP@@{\"type\": \"failed\", \"message\": \"No GPU.\"}');raise SystemExit(1)"
        echoed = [];code, lines = auto_fill._run([sys.executable, '-c', script], echoed.append)
        self.assertEqual(code, 1);self.assertEqual(len(lines), 2);self.assertEqual(echoed, lines[:1])

    def test_echo_failure_stops_the_worker(self):
        import time
        script = "import time;print('@@MOCAP@@{\"type\": \"progress\"}', flush=True);time.sleep(60)"
        def echo(line):raise KeyboardInterrupt
        started = time.monotonic()
        with self.assertRaises(KeyboardInterrupt):auto_fill._run([sys.executable, '-c', script], echo)
        self.assertLess(time.monotonic()-started, 30)

    def test_imports_without_torch(self):
        import subprocess
        out = subprocess.run([sys.executable, '-B', '-c', "import sys;import mocap.auto_fill;print(sorted(m for m in ('torch','mocap.models','mocap.kimodo_runtime') if m in sys.modules))"],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), '[]')


def vector(text):
    """A stand-in text feature: different for each text, never all zero."""
    return np.random.default_rng(list(text.encode('utf-8'))).normal(size=4096).astype(np.float32)


class FakeText:
    """Stands in for the text worker: `text --prompt-file <file> --out <file.npy>`; writes `answer(prompt bytes)`."""
    def __init__(self, answer=lambda raw: vector(raw.decode('utf-8')), code=0, lines=()):self.answer, self.code, self.lines, self.calls, self.prompts = answer, code, list(lines), [], []

    def __call__(self, command):
        self.calls.append(command);self.mode = command[1];args = dict(zip(command[2::2], command[3::2]))
        raw = Path(args['--prompt-file']).read_bytes();self.prompts.append(raw);out = self.answer(raw)
        if out is not None:np.save(args['--out'], out)
        return self.code, self.lines


class DescribeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder = Path(self.tmp.name)/'cache'/'kimodo_text'
        for target, name, value in ((auto_fill, 'TEXT_CACHE', self.folder), (kimodo_runtime, 'text_issue', lambda: None),
                                    (kimodo_runtime, 'command', lambda args: ['kimodo', *map(str, args)])):
            patch = mock.patch.object(target, name, value);patch.start();self.addCleanup(patch.stop)

    def files(self):return sorted(p.name for p in self.folder.iterdir()) if self.folder.exists() else []

    def test_reads_once_per_text(self):
        run = FakeText();lines = []
        out = auto_fill.describe('A person waves.', run=run, echo=lines.append)
        self.assertEqual((out.shape, out.dtype), ((4096,), np.float32));np.testing.assert_array_equal(out, vector('A person waves.'))
        self.assertEqual((len(run.calls), run.mode), (1, 'text'));self.assertEqual(self.files(), [auto_fill.text_key('A person waves.')+'.npy'])
        np.testing.assert_array_equal(auto_fill.describe('A person waves.', run=run), out);self.assertEqual(len(run.calls), 1)   # served from the cache
        other = auto_fill.describe('A person sits.', run=run);self.assertEqual(len(run.calls), 2);self.assertFalse(np.array_equal(other, out))
        self.assertEqual(len(self.files()), 2)

    def test_prompt_file_is_utf8_and_exact(self):
        text = 'Člověk mává pravou rukou — „ahoj“.';run = FakeText();auto_fill.describe(text, run=run)
        self.assertEqual(run.prompts, [text.encode('utf-8')]);args = dict(zip(run.calls[0][2::2], run.calls[0][3::2]))
        self.assertEqual(Path(args['--prompt-file']).parent, self.folder);self.assertEqual(Path(args['--out']).suffix, '.npy')
        self.assertEqual(self.files(), [auto_fill.text_key(text)+'.npy'])   # the prompt and partial files are removed

    def test_key(self):
        k = auto_fill.text_key('A person waves.');self.assertEqual(len(k), 64);self.assertNotEqual(k, auto_fill.text_key('A person sits.'))
        with mock.patch.object(kimodo_runtime, 'text_identity', lambda: 'other'):self.assertNotEqual(auto_fill.text_key('A person waves.'), k)
        from mocap import kimodo_text
        with mock.patch.object(kimodo_text, 'VERSION', 2):self.assertNotEqual(auto_fill.text_key('A person waves.'), k)
        source = Path(self.tmp.name)/'kimodo_text.py';source.write_bytes(b'code')
        with mock.patch.object(auto_fill, 'TEXT_SOURCE', source):
            first = auto_fill.text_key('A person waves.');source.write_bytes(b'code changed');self.assertNotEqual(auto_fill.text_key('A person waves.'), first)
        self.assertEqual(auto_fill.TEXT_SOURCE.name, 'kimodo_text.py');self.assertTrue(auto_fill.TEXT_SOURCE.is_file())

    def test_invalid_cache_is_read_again(self):
        done = self.folder/(auto_fill.text_key('A person waves.')+'.npy')
        for bad in (np.zeros(4095, np.float32), vector('x').astype(np.float64), np.where(np.arange(4096) == 2, np.nan, 1).astype(np.float32), b'not an array'):
            self.folder.mkdir(parents=True, exist_ok=True)
            if isinstance(bad, bytes):done.write_bytes(bad)
            else:np.save(done, bad)
            run = FakeText();out = auto_fill.describe('A person waves.', run=run)
            self.assertEqual(len(run.calls), 1);np.testing.assert_array_equal(out, vector('A person waves.'));self.assertEqual(self.files(), [done.name])

    def test_missing_text_files_write_nothing(self):
        sentence = 'A description needs the Kimodo text adapters and the Llama 3 weights. Use Set up… to get them in Model Manager.'
        run = FakeText()
        with mock.patch.object(kimodo_runtime, 'text_issue', lambda: sentence), self.assertRaises(auto_fill.Unavailable) as caught:
            auto_fill.describe('A person waves.', run=run)
        self.assertEqual(str(caught.exception), sentence);self.assertEqual(run.calls, []);self.assertFalse(self.folder.exists())

    def test_failed_worker(self):
        failed = PREFIX+json.dumps(dict(type='failed', stage='take_join', message='Not enough free video memory to read the description.'))
        for code, lines, message in ((1, [failed], 'Not enough free video memory to read the description.'), (1, [], auto_fill.STOPPED)):
            with self.assertRaises(auto_fill.Unavailable) as caught:auto_fill.describe('A person waves.', run=FakeText(lambda raw: None, code, lines))
            self.assertEqual(str(caught.exception), message);self.assertEqual(self.files(), [])
        for answer in (lambda raw: None, lambda raw: np.zeros(12, np.float32), lambda raw: np.full(4096, np.inf, np.float32)):
            with self.assertRaises(auto_fill.Unavailable) as caught:auto_fill.describe('A person waves.', run=FakeText(answer))
            self.assertEqual(str(caught.exception), auto_fill.NO_FEATURE);self.assertEqual(self.files(), [])

    def test_leftovers_are_removed(self):
        self.folder.mkdir(parents=True);stem = auto_fill.text_key('A person waves.')
        for suffix in ('.prompt.txt', '.partial.npy', '.partial.npy.tmp'):(self.folder/(stem+suffix)).write_bytes(b'left by a killed run')
        run = FakeText();auto_fill.describe('A person waves.', run=run)
        self.assertEqual(len(run.calls), 1);self.assertEqual(self.files(), [stem+'.npy'])


if __name__ == '__main__':unittest.main()
