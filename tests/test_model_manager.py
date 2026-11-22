import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mocap import model_manager as manager, models
from mocap.storage import atomic_json


class InstalledModelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.root = base / 'installation'
        self.root.mkdir()
        self.external = base / 'original.ckpt'
        self.external.write_bytes(b'Keep the original')
        paths = dict(HOME=self.root, ROOT=self.root, MODELS=self.root / 'models',
                     UPSTREAM=self.root / 'third_party')
        for target in (patch.multiple(manager, **paths), patch.object(models, 'MODELS', paths['MODELS'])):
            target.start()
            self.addCleanup(target.stop)
        self.model = models.BY_KEY['flow_sam']
        self.payload = self.model.path
        self.payload.parent.mkdir(parents=True)
        self.payload.write_bytes(b'Example installed model')
        self.receipt = manager.receipt_path(self.model)
        self.receipt.write_bytes(b'Record')
        self.keys = [self.model.key]
        self.alias = next(path for path, _ in manager.runtime_aliases(self.model)
                          if 'FlowHMR' in path.parts)
        self.alias.parent.mkdir(parents=True)

    def test_shared_workflows_use_actual_dependencies(self):
        users = manager.model_consumers(manager.workflow_catalog())
        self.assertEqual(users['flow_sam'], ('flowhmr', 'exmr'))
        self.assertEqual(users['flowhmr'], ('flowhmr',))
        self.assertEqual(set(users['flow_vggt']), {'flowhmr', 'mamma', 'pear'})
        self.assertEqual(set(users['sam31_people']), {'flowhmr', 'exmr', 'gvhmr'})

    def test_removes_known_links_but_keeps_import_source_and_unrelated_files(self):
        atomic_json(self.receipt, {'source': str(self.external)})
        os.link(self.payload, self.alias)
        held_by_user = self.external.with_name('outside-hardlink.ckpt')
        os.link(self.payload, held_by_user)
        other = self.payload.with_name('custom.ckpt')
        other.write_bytes(b'Other model')
        plan = manager.removal_plan(self.keys)
        self.assertEqual({item.path for item in plan}, {self.payload, self.receipt, self.alias})
        self.assertEqual(manager.physical_bytes(plan), self.payload.stat().st_size + self.receipt.stat().st_size)
        self.assertEqual(manager.physical_bytes(plan, reclaim=True), self.receipt.stat().st_size)
        manager.remove_models(self.keys, plan)
        self.assertTrue(all(not item.path.exists() for item in plan))
        self.assertEqual(held_by_user.read_bytes(), b'Example installed model')
        self.assertEqual(self.external.read_bytes(), b'Keep the original')
        self.assertEqual(other.read_bytes(), b'Other model')
        self.assertFalse(manager.journal_path().exists())
        self.payload.write_bytes(b'Reimported model')
        self.assertIn(self.model, manager.installed_models())

    def test_refuses_a_model_changed_after_confirmation(self):
        plan = manager.removal_plan(self.keys)
        self.payload.write_bytes(b'New version')
        with self.assertRaisesRegex(ValueError, 'changed after confirmation'):
            manager.remove_models(self.keys, plan)
        self.assertEqual(self.payload.read_bytes(), b'New version')
        self.assertFalse(manager.journal_path().exists())

    def test_interrupted_removal_rolls_back_from_journal(self):
        plan = manager.removal_plan(self.keys)
        rename = Path.rename

        def interrupt(path, destination):
            if path == self.receipt:
                raise KeyboardInterrupt
            return rename(path, destination)

        with patch.object(Path, 'rename', interrupt), self.assertRaises(KeyboardInterrupt):
            manager.remove_models(self.keys, plan)
        self.assertFalse(self.payload.exists())
        self.assertTrue(manager.journal_path().exists())
        self.assertTrue(manager.recover_removal())
        self.assertEqual(self.payload.read_bytes(), b'Example installed model')
        self.assertEqual(self.receipt.read_bytes(), b'Record')
        self.assertFalse(manager.journal_path().exists())

    def test_recovery_purges_committed_staging_without_touching_replacement(self):
        plan = manager.removal_plan(self.keys)
        with patch.object(manager, 'recover_removal', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            manager.remove_models(self.keys, plan)
        self.payload.write_bytes(b'Replacement')
        self.assertTrue(manager.recover_removal())
        self.assertEqual(self.payload.read_bytes(), b'Replacement')
        self.assertFalse(self.receipt.exists())

    def test_pear_removes_active_bundle_and_preserves_old_generation(self):
        from mocap import pear_models

        pear = models.BY_KEY['pear_geometry'].path
        active = pear.parent / 'geometry-active' / 'assets'
        old = pear.parent / 'geometry-old' / 'assets'
        relative = 'body/geometry.bin'
        for directory in (active, old):
            (directory / relative).parent.mkdir(parents=True)
        payload = active / relative
        payload.write_bytes(b'Geometry')
        os.link(payload, old / relative)
        stamp = payload.stat()
        manifest = dict(bundle_sha256='test-bundle', files=[dict(path=relative, bytes=stamp.st_size)])
        atomic_json(pear, dict(folder='geometry-active/assets', bundle_sha256='test-bundle',
                               files={relative:[stamp.st_size, stamp.st_mtime_ns]}))
        with patch.object(pear_models, 'MODELS', manager.MODELS), patch.object(pear_models, 'spec', return_value=manifest):
            plan = manager.removal_plan(['pear_geometry'])
            self.assertEqual(manager.physical_bytes(plan, reclaim=True), pear.stat().st_size)
            manager.remove_models(['pear_geometry'], plan)
            self.assertFalse(pear_models.geometry_ready())
        self.assertFalse(active.parent.exists())
        self.assertEqual((old / relative).read_bytes(), b'Geometry')

    def test_rejects_unrecognized_path_in_recovery_record(self):
        custom = manager.MODELS / 'my-model.ckpt'
        custom.write_bytes(b'Separate model')
        atomic_json(manager.journal_path(), dict(
            version=1,id='0'*32,committed=True,keys=self.keys,files=[
                dict(path=str(custom),fingerprint=manager.fingerprint(manager.entry(custom, manager.HOME)))
            ]))
        with self.assertRaisesRegex(ValueError, 'leaves managed storage'):
            manager.recover_removal()
        self.assertEqual(custom.read_bytes(), b'Separate model')

    def test_external_symlink_cannot_be_removed_as_a_managed_model(self):
        self.payload.unlink()
        try:
            self.payload.symlink_to(self.external)
        except OSError as error:
            self.skipTest(f'File symlinks unavailable on this system: {error}')
        self.assertIn(self.model, manager.installed_models())
        with self.assertRaisesRegex(ValueError, 'stored externally'):
            manager.removal_plan(self.keys)
        self.assertEqual(self.external.read_bytes(), b'Keep the original')


KIMODO_KEYS = ('kimodo', 'kimodo_config', 'kimodo_stats_body_mean', 'kimodo_stats_body_std',
               'kimodo_stats_global_root_mean', 'kimodo_stats_global_root_std',
               'kimodo_stats_local_root_mean', 'kimodo_stats_local_root_std')


class KimodoModelTests(unittest.TestCase):
    def test_records_are_pinned_files_of_one_snapshot(self):
        self.assertEqual(models.STAGE_REQUIREMENTS['kimodo_fill'], KIMODO_KEYS)
        sizes = {}
        for key in KIMODO_KEYS:
            model = models.BY_KEY[key]
            self.assertRegex(model.expected_sha256, '^[0-9a-f]{64}$')
            self.assertGreater(model.minimum_bytes, 0)
            self.assertTrue(model.relative.startswith('kimodo/Kimodo-SOMA-RP-v1.2/'))
            path = model.relative.removeprefix('kimodo/Kimodo-SOMA-RP-v1.2/')
            self.assertEqual(model.download_url, 'https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/'
                             '0e3c89024d896064b6851201373f7834784a386b/' + path)
            sizes[path] = model.minimum_bytes
        self.assertEqual(sizes['model.safetensors'], 1133177724)
        self.assertEqual(sum(sizes.values()), 1133209886 - 1519 - 10229 - 13003)
        self.assertEqual(models.BY_KEY['kimodo'].name, 'Kimodo Motion Generation (Optional)')
        self.assertEqual(models.BY_KEY['kimodo_config'].name, 'Kimodo Configuration (Optional)')
        self.assertEqual(models.BY_KEY['kimodo_stats_global_root_std'].name, 'Kimodo Statistics (global root std) (Optional)')
        self.assertTrue(all(models.BY_KEY[key].name.endswith(' (Optional)') for key in KIMODO_KEYS))

    def test_all_missing_in_an_empty_models_folder(self):
        with tempfile.TemporaryDirectory() as name, patch.object(models, 'MODELS', Path(name)):
            self.assertEqual([model.key for model in models.missing('kimodo_fill')], list(KIMODO_KEYS))

    def test_a_file_without_receipt_is_not_ready(self):
        model = models.BY_KEY['kimodo_stats_local_root_mean']
        with tempfile.TemporaryDirectory() as name, patch.object(models, 'MODELS', Path(name)):
            model.path.parent.mkdir(parents=True)
            model.path.write_bytes(b'x' * model.minimum_bytes)
            self.assertFalse(models.ready(model))
            stat = model.path.stat()
            atomic_json(manager.receipt_path(model), dict(sha256=model.expected_sha256,
                                                          stat=[stat.st_size, stat.st_mtime_ns]))
            self.assertTrue(models.ready(model))

    def test_download_all_leaves_kimodo_out(self):
        for key in KIMODO_KEYS:
            self.assertTrue(manager.download_all_skips(key))
        for key in ('sam31_people', 'rife49'):
            self.assertTrue(manager.download_all_skips(key))
        for key in ('gvhmr', 'exmr', 'exmr_config', 'smplx'):
            self.assertFalse(manager.download_all_skips(key))

    def test_listed_under_gvhmr_and_exmr_only(self):
        from mocap.kimodo_runtime import WORKFLOWS
        self.assertEqual(WORKFLOWS, frozenset(('gvhmr', 'exmr')))
        users = manager.model_consumers(manager.workflow_catalog())
        for key in KIMODO_KEYS:
            self.assertEqual(set(users[key]), {'gvhmr', 'exmr'})


class Response:
    def __init__(self, url, failing):self.url, self.failing, self.headers = url, failing, {}
    def __enter__(self):return self
    def __exit__(self, *exc):return False
    def raise_for_status(self):
        if self.url in self.failing:
            import requests
            raise requests.HTTPError('503 Server Error')
    def iter_content(self, size):yield b'payload of ' + self.url.encode()


class KimodoDownloadTests(unittest.TestCase):
    """Auto Download on the Kimodo Motion Generation row fetches all eight files; network replaced."""
    def setUp(self):
        import dataclasses, hashlib
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        records = {key: dataclasses.replace(models.BY_KEY[key], minimum_bytes=1, expected_sha256=hashlib.sha256(
            b'payload of ' + models.BY_KEY[key].download_url.encode()).hexdigest()) for key in KIMODO_KEYS}
        self.urls, self.failing = [], set()
        def get(url, **kw):self.urls.append(url); return Response(url, self.failing)
        for target in (patch.object(models, 'MODELS', Path(tmp.name)), patch.dict(models.BY_KEY, records), patch('requests.get', get)):
            target.start(); self.addCleanup(target.stop)
        self.url = lambda key: models.BY_KEY[key].download_url

    def test_companions_first_then_the_weights(self):
        path = models.download('kimodo')
        self.assertEqual(self.urls, [self.url(key) for key in KIMODO_KEYS[1:]] + [self.url('kimodo')])
        self.assertEqual(path, models.BY_KEY['kimodo'].path); self.assertEqual(models.missing('kimodo_fill'), [])

    def test_ready_companions_are_skipped(self):
        models.download('kimodo_config'); models.download('kimodo_stats_body_std'); self.urls.clear()
        models.download('kimodo')
        self.assertEqual(self.urls, [self.url(key) for key in KIMODO_KEYS[1:] if key not in ('kimodo_config', 'kimodo_stats_body_std')] + [self.url('kimodo')])
        self.urls.clear(); models.download('kimodo'); self.assertEqual(self.urls, [])

    def test_a_failed_companion_stops_before_the_weights(self):
        import requests
        self.failing.add(self.url('kimodo_stats_global_root_mean'))
        with self.assertRaises(requests.HTTPError):models.download('kimodo')
        self.assertNotIn(self.url('kimodo'), self.urls); self.assertFalse(models.ready(models.BY_KEY['kimodo']))

    def test_other_rows_download_alone(self):
        models.download('kimodo_stats_body_mean'); self.assertEqual(self.urls, [self.url('kimodo_stats_body_mean')])


if __name__ == '__main__':
    unittest.main()
