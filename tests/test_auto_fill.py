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
    return dict(version=1, before_orient=b[0], before_pose=b[1], before_pelvis=b[2], after_orient=a[0], after_pose=a[1], after_pelvis=a[2],
                rest=rng.normal(size=(22, 3)), sole=-1.3, frames=frames, seed=seed, fps=30.)


def response(req, rows=None, end_pose=.8):
    c0, c1 = min(5, len(req['before_pelvis'])), min(5, len(req['after_pelvis']));rows = c0+int(req['frames'])+c1 if rows is None else rows
    return dict(version=np.int64(1), context=np.array([c0, c1], np.int64), orient=np.zeros((rows, 3)), pose=np.full((rows, 63), .1),
                pelvis=np.ones((rows, 3)), contacts=np.zeros((rows, 4), bool), end_pose_cm=np.float64(end_pose))


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


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder = Path(self.tmp.name)/'person'/'auto_fill'
        for name, value in (('issue', lambda: None), ('command', lambda args: ['kimodo', *map(str, args)])):
            patch = mock.patch.object(kimodo_runtime, name, value);patch.start();self.addCleanup(patch.stop)
        self.request = request();self.done = self.folder/(auto_fill.key(self.request, kimodo_runtime.identity())+'.npz')

    def files(self):return sorted(p.name for p in self.folder.iterdir()) if self.folder.exists() else []

    def test_generates_once_then_serves_the_cache(self):
        run = FakeRun();lines = []
        out = auto_fill.generate(self.folder, self.request, run=run, echo=lines.append)
        self.assertEqual(len(run.calls), 1);self.assertEqual(self.files(), [self.done.name])
        self.assertEqual(out['pose'].shape, (30, 63));self.assertEqual(float(out['end_pose_cm']), .8)
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
                       lambda r: dict(response(r), end_pose_cm=np.float64(np.nan)), lambda r: {k: v for k, v in response(r).items() if k != 'pelvis'}):
            with self.assertRaises(auto_fill.Unavailable) as caught:auto_fill.generate(self.folder, self.request, run=FakeRun(answer))
            self.assertEqual(str(caught.exception), 'Automatic fill returned no usable motion.');self.assertEqual(self.files(), [])

    def test_bad_cache_is_regenerated(self):
        self.folder.mkdir(parents=True);np.savez(self.done, **response(self.request, rows=25))
        run = FakeRun();out = auto_fill.generate(self.folder, self.request, run=run)
        self.assertEqual(len(run.calls), 1);self.assertEqual(out['orient'].shape, (30, 3));self.assertEqual(self.files(), [self.done.name])

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

    def test_imports_without_torch(self):
        import subprocess
        out = subprocess.run([sys.executable, '-B', '-c', "import sys;import mocap.auto_fill;print(sorted(m for m in ('torch','mocap.models','mocap.kimodo_runtime') if m in sys.modules))"],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), '[]')


if __name__ == '__main__':unittest.main()
