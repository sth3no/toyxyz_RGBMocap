import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap import kimodo_fill as kf, kimodo_runtime, soma_convert as sc

# SMPL-X neutral, zero betas (as tests/test_soma_convert.py): J_regressor[:22] @ v_template, and the lowest v_template height.
REST = np.array([
    [0.0031, -0.3514, 0.0120], [0.0613, -0.4442, -0.0140], [-0.0601, -0.4553, -0.0092], [0.0004, -0.2415, -0.0156],
    [0.1160, -0.8229, -0.0234], [-0.1044, -0.8177, -0.0260], [0.0098, -0.1097, -0.0215], [0.0726, -1.2260, -0.0552],
    [-0.0889, -1.2284, -0.0462], [-0.0015, -0.0574, 0.0069], [0.1198, -1.2840, 0.0630], [-0.1277, -1.2868, 0.0728],
    [-0.0137, 0.1077, -0.0247], [0.0448, 0.0275, -0.0003], [-0.0492, 0.0269, -0.0065], [0.0111, 0.2682, -0.0040],
    [0.1641, 0.0852, -0.0158], [-0.1518, 0.0804, -0.0191], [0.4182, 0.0131, -0.0582], [-0.4229, 0.0439, -0.0456],
    [0.6702, 0.0363, -0.0607], [-0.6722, 0.0394, -0.0609]])
SOLE = -1.3018
GAP, SIDE = 60, 45


def walk(frames=GAP, before=SIDE, after=SIDE, seed=0):
    """A person facing +Z, walking 1.5 m along +Z across the hidden range, standing-like poses with a slow leg swing."""
    rng = np.random.default_rng(seed); step = 1.5 / (frames + 1); height = REST[0, 1] - SOLE

    def side(times):
        t = np.asarray(times, np.float64); pose = rng.normal(0, .02, (len(t), 63)); swing = .25 * np.sin(t * 2 * np.pi / 30)
        pose[:, 0] += swing; pose[:, 3] -= swing; pose[:, 9] += .1 + .1 * np.abs(swing); pose[:, 12] += .1 + .1 * np.abs(swing)
        pose[:, 47] -= 1.2; pose[:, 50] += 1.2                                      # shoulders: arms down from the T-pose
        orient = rng.normal(0, .01, (len(t), 3)); pelvis = np.c_[np.full(len(t), .2), np.full(len(t), height), t * step]
        return orient, pose, pelvis
    b, a = side(np.arange(-before, 0) + 1), side(np.arange(frames + 1, frames + 1 + after))
    return dict(version=np.int64(1), before_orient=b[0], before_pose=b[1], before_pelvis=b[2], after_orient=a[0], after_pose=a[1], after_pelvis=a[2],
                rest=REST.copy(), sole=np.float64(SOLE), frames=np.int64(frames), seed=np.int64(3), fps=np.float64(30.))


def synthetic_output(p):
    """Kimodo-like output: the plan's known SOMA rows, a linear root path between them, the last known rotations held."""
    c0, c1 = p['context']; rows = p['frames']; K = np.empty((rows, 30, 3, 3)); hips = np.empty((rows, 3))
    K[:c0] = p['rotations'][:c0]; K[rows - c1:] = p['rotations'][c0:]; K[c0:rows - c1] = p['rotations'][c0 - 1]
    hips[p['known']] = p['positions'][:, 0]; gap = np.arange(c0, rows - c1)
    hips[gap] = p['positions'][c0 - 1, 0] + (gap - (c0 - 1))[:, None] / (rows - c1 - (c0 - 1)) * (p['positions'][c0, 0] - p['positions'][c0 - 1, 0])
    contacts = np.random.default_rng(1).random((rows, 6)) > .5
    return dict(global_rot_mats=K, root_positions=hips, foot_contacts=contacts, names=list(sc.SOMA30))


class PlanTest(unittest.TestCase):
    def test_window(self):
        p = kf.plan(walk()); self.assertEqual(p['context'], (5, 5)); self.assertEqual(p['frames'], 70)
        np.testing.assert_array_equal(p['known'], [0, 1, 2, 3, 4, 65, 66, 67, 68, 69])
        self.assertEqual(p['positions'].shape, (10, 30, 3)); self.assertEqual(p['rotations'].shape, (10, 30, 3, 3))
        np.testing.assert_allclose(p['smooth_root'][0], 0, atol=1e-12); np.testing.assert_allclose(p['positions'][0, 0, [0, 2]], 0, atol=1e-12)
        self.assertIsInstance(p['heading'], float); self.assertLess(abs(p['heading']), .1)       # facing +Z

    def test_floor_is_a_line_between_the_sides(self):
        r = walk(); p = kf.plan(r); f = sc.fit(REST, SOLE)
        fl = sc.floor_under(sc.smplx_fk(REST, r['before_orient'], r['before_pose'], r['before_pelvis'])[1], f['toe'])
        fr = sc.floor_under(sc.smplx_fk(REST, r['after_orient'], r['after_pose'], r['after_pelvis'])[1], f['toe'])
        self.assertEqual(p['floor'].shape, (70,)); np.testing.assert_allclose(p['floor'][:5], fl); np.testing.assert_allclose(p['floor'][65:], fr)
        np.testing.assert_allclose(p['floor'][4:66], np.linspace(fl, fr, 62), atol=1e-12)

    def test_short_side(self):
        p = kf.plan(walk(before=4)); self.assertEqual(p['context'], (4, 5)); self.assertEqual(p['frames'], 69)
        np.testing.assert_array_equal(p['known'], [0, 1, 2, 3, 64, 65, 66, 67, 68])

    def test_smooth_root_from_each_side(self):
        seen = []
        def smooth(root):seen.append(len(root));return root + [.1, 0, .2]
        p = kf.plan(walk(before=2), smooth); self.assertEqual(seen, [30])          # 30 frames next to the range; a 2-frame side keeps its raw root
        np.testing.assert_allclose(p['smooth_root'][0], 0, atol=1e-12)
        np.testing.assert_allclose(p['smooth_root'][2:] - p['positions'][2:, 0][:, [0, 2]], np.tile([.1, .2], (5, 1)), atol=1e-12)

    def test_smoother_sees_the_frames_next_to_the_range(self):
        r = walk(); seen = []
        def smooth(root):seen.append(root.copy());return root
        p = kf.plan(r, smooth); full = kf.plan(r)
        self.assertEqual([len(x) for x in seen], [30, 30])
        np.testing.assert_allclose(seen[0][-5:], full['positions'][:5, 0] + full['origin'], atol=1e-12)    # before: its last 30 frames
        np.testing.assert_allclose(seen[1][:5], full['positions'][5:, 0] + full['origin'], atol=1e-12)     # after: its first 30 frames

    def test_short_sides_skip_kimodos_smoother(self):
        def smooth(root):                         # Kimodo's smoother fails ('Factor is exactly singular') below 5 frames
            if len(root) < 5:raise RuntimeError('Factor is exactly singular')
            return root
        for count in (1, 2, 3, 4, 5, 6):
            p = kf.plan(walk(before=count, after=count), smooth); self.assertEqual(p['context'], (min(5, count),) * 2)

    def test_refusals(self):
        for change in (dict(frames=np.int64(291)), dict(fps=np.float64(25.)), dict(seed=np.int64(1000))):
            with self.assertRaises(ValueError) as e:kf.plan(dict(walk(), **change))
            self.assertTrue(str(e.exception).endswith('.'), str(e.exception))
        r = walk(); r['after_pose'][3, 7] = np.nan
        with self.assertRaises(ValueError) as e:kf.plan(r)
        self.assertTrue(str(e.exception).endswith('.'))
        r = walk(); del r['rest']
        with self.assertRaises(ValueError):kf.plan(r)
        with self.assertRaises(ValueError):kf.plan(dict(walk(), before_pose=np.zeros((45, 60))))


class FinishTest(unittest.TestCase):
    def test_known_rows_come_back(self):
        r = walk(); p = kf.plan(r); o = synthetic_output(p); out = kf.finish(r, p, o)
        self.assertEqual(int(out['version']), 1); np.testing.assert_array_equal(out['context'], [5, 5])
        for k, w in (('orient', 3), ('pose', 63), ('pelvis', 3)):self.assertEqual(out[k].shape, (70, w))
        for k in ('orient', 'pose', 'pelvis'):
            np.testing.assert_allclose(out[k][:5], r['before_' + k][-5:], atol=1e-6); np.testing.assert_allclose(out[k][65:], r['after_' + k][:5], atol=1e-6)
        self.assertLess(float(out['end_pose_cm']), 1e-3)
        self.assertEqual(out['contacts'].dtype, bool); np.testing.assert_array_equal(out['contacts'], o['foot_contacts'][:, [0, 1, 3, 4]])

    def test_extra_output_joints_are_found_by_name(self):
        r = walk(); p = kf.plan(r); o = synthetic_output(p); base = kf.finish(r, p, o)
        K = np.tile(np.eye(3), (70, 77, 1, 1)); order = np.random.default_rng(2).permutation(77); K[:, order[:30]] = o['global_rot_mats']
        names = ['Extra%d' % k for k in range(77)]
        for k, name in zip(order[:30], sc.SOMA30):names[k] = name
        out = kf.finish(r, p, dict(o, global_rot_mats=K, names=names))
        for k in ('orient', 'pose', 'pelvis'):np.testing.assert_allclose(out[k], base[k], atol=1e-12)

    def test_wrong_row_count(self):
        r = walk(); p = kf.plan(r); o = synthetic_output(p)
        with self.assertRaises(ValueError):kf.finish(r, p, dict(o, root_positions=o['root_positions'][:-1]))


class FloorDriftTest(unittest.TestCase):
    def test_floor_drift_stays_out_of_kimodo(self):
        r = walk(); d = dict(r, before_pelvis=r['before_pelvis'] + [0, .12, 0], after_pelvis=r['after_pelvis'] + [0, .03, 0])
        p, q = kf.plan(r), kf.plan(d)
        self.assertLess(np.abs(q['positions'] - p['positions']).max(), 1e-3); self.assertLess(np.abs(q['smooth_root'] - p['smooth_root']).max(), 1e-3)
        o = synthetic_output(p); a, b = kf.finish(r, p, o), kf.finish(d, q, o)
        rise = b['pelvis'][:, 1] - a['pelvis'][:, 1]
        np.testing.assert_allclose(rise[:5], .12, atol=1e-9); np.testing.assert_allclose(rise[65:], .03, atol=1e-9)
        np.testing.assert_allclose(rise[4:66], np.linspace(.12, .03, 62), atol=1e-9)
        np.testing.assert_allclose(b['pelvis'][:, [0, 2]], a['pelvis'][:, [0, 2]], atol=1e-12)


def events(text):
    return [json.loads(line[len('@@MOCAP@@'):]) for line in text.splitlines() if line.startswith('@@MOCAP@@')]


@unittest.skipUnless(kimodo_runtime.issue() is None, 'Kimodo runtime or model is not installed')
class WorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(); cls.dir = Path(cls.folder.name)
        cls.request = cls.dir / 'request.npz'; np.savez(cls.request, **walk())

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def run_worker(self, request, response):
        return subprocess.run(kimodo_runtime.command(['--request', request, '--response', response]), capture_output=True, text=True, encoding='utf-8', timeout=900)

    def test_generates_and_repeats(self):
        outs = []
        for k in range(2):
            path = self.dir / f'response{k}.npz'; p = self.run_worker(self.request, path)
            self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:]); self.assertFalse(Path(str(path) + '.tmp').exists())
            self.assertFalse([e for e in events(p.stdout) if e['type'] == 'failed']); self.assertTrue(all(e.get('stage') == 'take_join' for e in events(p.stdout)))
            with np.load(path, allow_pickle=False) as z:outs.append({k: z[k] for k in z.files})
        o = outs[0]; self.assertEqual(int(o['version']), 1); np.testing.assert_array_equal(o['context'], [5, 5])
        for k, w in (('orient', 3), ('pose', 63), ('pelvis', 3)):self.assertEqual(o[k].shape, (70, w)); self.assertTrue(np.isfinite(o[k]).all())
        self.assertEqual(o['contacts'].shape, (70, 4)); self.assertEqual(o['contacts'].dtype, bool)
        self.assertLess(float(o['end_pose_cm']), 6); self.assertGreaterEqual(np.linalg.norm(o['pelvis'][64] - o['pelvis'][5]), 1.2)
        for k in ('orient', 'pose', 'pelvis', 'end_pose_cm'):np.testing.assert_allclose(outs[1][k], o[k], atol=1e-4)
        np.testing.assert_array_equal(outs[1]['contacts'], o['contacts'])

    def test_missing_request(self):
        path = self.dir / 'none.npz'; p = self.run_worker(self.dir / 'missing.npz', path)
        self.assertEqual(p.returncode, 1); found = events(p.stdout)
        self.assertEqual([e['type'] for e in found], ['failed']); self.assertEqual(found[0]['stage'], 'take_join'); self.assertFalse(path.exists())

    def test_four_frames_before(self):
        request, path = self.dir / 'short.npz', self.dir / 'short_response.npz'; np.savez(request, **walk(before=4))
        p = self.run_worker(request, path); self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['context'], [4, 5]); self.assertEqual(z['pelvis'].shape, (69, 3)); self.assertTrue(np.isfinite(z['pelvis']).all())

    def test_unreadable_request(self):
        request, path = self.dir / 'broken.npz', self.dir / 'broken_response.npz'; request.write_bytes(b'not a zip file')
        p = self.run_worker(request, path); found = events(p.stdout)
        self.assertEqual(p.returncode, 1); self.assertEqual([e['type'] for e in found], ['failed'])
        self.assertTrue(found[0]['message'].startswith('Could not read the automatic fill request.'), found[0]['message']); self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
