import json, re, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap import hidden_ranges, kimodo_runtime, kimodo_text as kt

GATE = ROOT / 'experiments/kimodo-spike/out/text/full.npz'   # the step 5 gate's vectors (gitignored; compared when present)
WALK = 'A person walks forward.'


class NormaliseTest(unittest.TestCase):
    def test_whitespace_collapses_to_one_line(self):
        self.assertEqual(kt.normalise('  A person\twalks\n\nforward.\r\n '), WALK)
        self.assertEqual(kt.normalise('a   b'), 'a b')

    def test_limit(self):
        self.assertEqual(kt.normalise('x' * 200), 'x' * 200)
        self.assertEqual(kt.normalise(' ' * 50 + 'x' * 200 + '\n' * 50), 'x' * 200)
        with self.assertRaises(ValueError) as caught:kt.normalise('x' * 201)
        self.assertEqual(str(caught.exception), 'A description can have at most 200 characters.')

    def test_quotes_and_non_ascii_kept(self):
        text = 'Osoba "skáče" a mává — it\'s 3× faster'
        self.assertEqual(kt.normalise(text), text)

    def test_only_spaces(self):
        self.assertEqual(kt.normalise(' \t\n '), ''); self.assertEqual(kt.normalise(''), '')

    def test_one_definition(self):
        self.assertIs(kt.normalise, hidden_ranges.normalise_text); self.assertEqual(hidden_ranges.TEXT_LIMIT, 200)

    def test_files_and_version(self):
        from mocap.paths import MODELS
        self.assertEqual(kt.VERSION, 1)
        self.assertEqual(kt.files(), {k: MODELS / 'kimodo_text' / k for k in ('llama', 'mntp', 'supervised')})

    def test_imports_without_torch(self):
        code = ('import sys;sys.path.insert(0,sys.argv[1]);import mocap.kimodo_text as m;m.normalise(" a ");m.files();'
                'print([k for k in ("torch","transformers","numpy","safetensors") if k in sys.modules])')
        p = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(ROOT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr); self.assertEqual(p.stdout.strip(), '[]')


def events(text):
    return [json.loads(line[len('@@MOCAP@@'):]) for line in text.splitlines() if line.startswith('@@MOCAP@@')]


@unittest.skipUnless(kimodo_runtime.text_issue() is None, 'Kimodo runtime or text files are not installed')
class WorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(); cls.dir = Path(cls.folder.name); cls.walk = []
        for k in range(2):cls.walk.append(cls.encode(WALK, f'walk{k}'))

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    @classmethod
    def encode(cls, text, name):
        prompt, out = cls.dir / f'{name}.txt', cls.dir / f'{name}.npy'; prompt.write_text(text, encoding='utf-8')
        return cls.run_worker(prompt, out), out

    @classmethod
    def run_worker(cls, prompt, out):
        return subprocess.run(kimodo_runtime.command(['text', '--prompt-file', prompt, '--out', out]), capture_output=True, text=True, encoding='utf-8', timeout=900)

    def check_success(self, p, out):
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:]); self.assertFalse(Path(str(out) + '.tmp').exists())
        found = events(p.stdout); self.assertFalse([e for e in found if e['type'] == 'failed']); self.assertTrue(all(e.get('stage') == 'take_join' for e in found))
        progress = [e for e in found if e['type'] == 'progress']; self.assertEqual((progress[-1]['current'], progress[-1]['total']), (32, 32))
        self.assertNotIn('FutureWarning', p.stderr)
        info = [e['message'] for e in found if e['type'] == 'info']; self.assertEqual(len(info), 1)
        match = re.fullmatch(r'Description read in [0-9]+\.[0-9] seconds; peak video memory ([0-9]+\.[0-9]{2}) GiB\.', info[0]); self.assertTrue(match, info[0])
        vector = np.load(out, allow_pickle=False); self.assertEqual((vector.shape, vector.dtype), ((4096,), np.float32)); self.assertTrue(np.isfinite(vector).all())
        return vector, float(match[1])

    def test_walk(self):
        (p, out), _ = self.walk; vector, peak = self.check_success(p, out)
        self.assertTrue(100 < np.linalg.norm(vector) < 160, np.linalg.norm(vector)); self.assertLess(peak, 3)

    def test_repeats(self):
        (a, b) = [self.check_success(p, out)[0] for p, out in self.walk]; np.testing.assert_allclose(a, b, atol=1e-3)

    @unittest.skipUnless(GATE.is_file(), "the gate's vectors are not on disk")
    def test_equals_the_gate(self):
        vector = self.check_success(*self.walk[0])[0]
        with np.load(GATE, allow_pickle=False) as z:gate = z['features'][list(z['prompts']).index(WALK)]
        cosine = float(vector @ gate / np.linalg.norm(vector) / np.linalg.norm(gate))
        self.assertGreater(cosine, 0.9999)

    def test_non_ascii_and_quotes(self):
        p, out = self.encode('Osoba "skáče" —\na person\tjumps, says “hi”.', 'czech'); vector = self.check_success(p, out)[0]
        self.assertTrue(100 < np.linalg.norm(vector) < 160, np.linalg.norm(vector))

    def test_missing_prompt_file(self):
        out = self.dir / 'none.npy'; p = self.run_worker(self.dir / 'missing.txt', out)
        self.assertEqual(p.returncode, 1); found = events(p.stdout)
        self.assertEqual([e['type'] for e in found], ['failed']); self.assertEqual(found[0]['stage'], 'take_join'); self.assertFalse(out.exists())

    def test_too_long(self):
        p, out = self.encode('x' * 201, 'long'); found = events(p.stdout)
        self.assertEqual(p.returncode, 1); self.assertEqual(found, [dict(type='failed', stage='take_join', message='A description can have at most 200 characters.')])
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
