import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mocap import kimodo_runtime as runtime, models, paths

SENTENCE = 'Close RGB MoCap and run Setup_Kimodo.cmd to prepare automatic fill.'


class ImportTest(unittest.TestCase):
    def test_import_loads_no_numeric_or_gui_packages(self):
        code = ("import sys;import mocap.kimodo_runtime;"
                "print(sorted(m for m in ('numpy','scipy','torch','PySide6') if m in sys.modules))")
        out = subprocess.run([sys.executable, '-B', '-c', code], cwd=paths.ROOT, capture_output=True, text=True,
                             check=True).stdout
        self.assertEqual(out.strip(), '[]')


class LocationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.folder = self.root / 'runtime_kimodo'
        target = patch.object(runtime, 'ROOT', self.root); target.start(); self.addCleanup(target.stop)

    def prepare(self, revision=runtime.REVISION):
        (self.folder / 'kimodo/model').mkdir(parents=True, exist_ok=True)
        (self.folder / 'kimodo/model/load_model.py').write_text('', encoding='utf-8')
        (self.folder / 'runtime.json').write_text(json.dumps(dict(version=runtime.PROFILE_VERSION, revision=revision)),
                                                  encoding='utf-8')

    def test_missing_runtime_raises_the_sentence(self):
        with self.assertRaisesRegex(RuntimeError, '^' + SENTENCE + '$'):
            runtime.location()

    def test_other_revision_raises(self):
        self.prepare('0' * 40)
        with self.assertRaisesRegex(RuntimeError, SENTENCE):
            runtime.location()

    def test_missing_package_raises(self):
        self.prepare(); (self.folder / 'kimodo/model/load_model.py').unlink()
        with self.assertRaisesRegex(RuntimeError, SENTENCE):
            runtime.location()

    def test_prepared_runtime_is_returned(self):
        self.prepare()
        self.assertEqual(runtime.location(), self.folder)

    def test_command_starts_the_exmr_python_with_the_boot_file(self):
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, 'ExMR runtime is missing'):
            runtime.command(['--x'])
        python = self.root / 'runtime_exmr/python.exe'; python.parent.mkdir(); python.write_bytes(b'')
        command = runtime.command(['--x'])
        self.assertEqual(command[:6], [str(python), '-I', '-X', 'utf8', '-B', '-u'])
        self.assertEqual(command[-2:], [str(self.root / 'kimodo_worker_boot.py'), '--x'])

    def test_command_needs_the_runtime(self):
        with self.assertRaisesRegex(RuntimeError, SENTENCE):
            runtime.command(['--x'])

    def test_issue(self):
        self.assertEqual(runtime.issue(), SENTENCE)
        self.prepare()
        with patch.object(models, 'missing', return_value=[models.BY_KEY['kimodo']]) as missing:
            self.assertEqual(runtime.issue(), 'Get the Kimodo model in Model Manager to use automatic fill.')
        missing.assert_called_with('kimodo_fill')
        with patch.object(models, 'missing', return_value=[]):
            self.assertIsNone(runtime.issue())

    def test_identity_names_the_revision_and_the_weights(self):
        identity = runtime.identity()
        self.assertEqual(identity, f"{runtime.REVISION}:{runtime.PROFILE_VERSION}:{models.BY_KEY['kimodo'].expected_sha256}")

    def test_activate_puts_the_overlay_first(self):
        with patch.object(sys, 'path', list(sys.path)):
            runtime.activate(self.folder)
            self.assertEqual(sys.path[0], str(self.folder))

    def test_constants(self):
        self.assertEqual((runtime.REVISION, runtime.PROFILE_VERSION, runtime.MODEL_FOLDER, runtime.MODEL_NAME, runtime.PARAMETERS),
                         ('e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954', 1, 'kimodo', 'Kimodo-SOMA-RP-v1.2', 283281777))


if __name__ == '__main__':
    unittest.main()
