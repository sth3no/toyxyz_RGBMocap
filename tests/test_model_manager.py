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
        self.assertEqual(models.BY_KEY['kimodo'].name, 'Kimodo Motion Generation (Experimental, optional)')
        self.assertEqual(models.BY_KEY['kimodo_config'].name, 'Kimodo Configuration (Optional)')
        self.assertEqual(models.BY_KEY['kimodo_stats_global_root_std'].name, 'Kimodo Statistics (global root std) (Optional)')
        self.assertTrue(all(models.BY_KEY[key].name.endswith(' (Optional)') for key in KIMODO_KEYS[1:]))

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
        for key in KIMODO_KEYS:  # members are counted for their group's row in Installed Models
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


MNTP = 'https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp'
TEXT_FILES = (  # key, folder, repository, revision, file, size, SHA-256
    ('kimodo_text', 'mntp', MNTP, '31474e395ada192e8ed1586db6be79fb3b70c9c0', 'adapter_model.safetensors', 167829552, 'de9c8736618a13173c6a1623cdef1b75e86c69317f1073ae82cd516ac36a632d'),
    ('kimodo_text_mntp_config', 'mntp', MNTP, '31474e395ada192e8ed1586db6be79fb3b70c9c0', 'adapter_config.json', 794, '1d3bbf14142865885ea8d0dbc1fb261adf4f5d14ddbdc939efb47c4913037813'),
    ('kimodo_text_model_config', 'mntp', MNTP, '31474e395ada192e8ed1586db6be79fb3b70c9c0', 'config.json', 781, '416c570276487324e49dca7c3d79e212b948d9cb2282fa734b1fa8014de041d1'),
    ('kimodo_text_tokenizer', 'mntp', MNTP, '31474e395ada192e8ed1586db6be79fb3b70c9c0', 'tokenizer.json', 9085671, '52b583ee1699c6ca4468f70885bd5f96dd1f7517292362a67581038afc04f4d8'),
    ('kimodo_text_tokenizer_config', 'mntp', MNTP, '31474e395ada192e8ed1586db6be79fb3b70c9c0', 'tokenizer_config.json', 51042, '0655c52e874137e4a455228155d982c4ea18419f461a10ceb57054995fec4223'),
    ('kimodo_text_special_tokens', 'mntp', MNTP, '31474e395ada192e8ed1586db6be79fb3b70c9c0', 'special_tokens_map.json', 335, '849070cae53bd45439e64ce5b1ddd650a66081b1bd47895c5a58939a05055579'),
    ('kimodo_text_supervised', 'supervised', MNTP + '-supervised', 'baa8ebf04a1c2500e61288e7dad65e8ae42601a7', 'adapter_model.safetensors', 167829552, '53f8f94ebdf396667ba99dd96e78203edae27bbcdbd1cf5f12b611e1d916b225'),
    ('kimodo_text_supervised_config', 'supervised', MNTP + '-supervised', 'baa8ebf04a1c2500e61288e7dad65e8ae42601a7', 'adapter_config.json', 794, '2a1265e4a7829c16dad38a5175ec8ac41ea1e851654da5044ace2437615a58ba'),
)
LLAMA_FILES = (
    ('kimodo_llama', 'model-00001-of-00004.safetensors', 4976698672, 'd8cf9c4d0dd972e1a2131bfe656235ee98221679711a3beef6d46dadf0f20b5c'),
    ('kimodo_llama_2', 'model-00002-of-00004.safetensors', 4999802720, '8d4782b4a69ef03845159ce1a15e272aadaaf134dc138d68f616098e8531729c'),
    ('kimodo_llama_3', 'model-00003-of-00004.safetensors', 4915916176, '3acdd690e65c24f42a24581b8467af98bd3ca357444580f8012aacd2bd607921'),
    ('kimodo_llama_4', 'model-00004-of-00004.safetensors', 1168138808, '67e9ad31c8c32abf3a55ee7fc7217b3ecb35fd3c74d98a5bd233e0e4d6964f46'),
)
TEXT_KEYS = tuple(row[0] for row in TEXT_FILES)
LLAMA_KEYS = tuple(row[0] for row in LLAMA_FILES)
GROUP_KEYS = KIMODO_KEYS + TEXT_KEYS + LLAMA_KEYS


class KimodoGroupRecordTests(unittest.TestCase):
    def test_three_groups_with_pinned_records(self):
        self.assertEqual(models.GROUPS, {'kimodo': KIMODO_KEYS, 'kimodo_text': TEXT_KEYS, 'kimodo_llama': LLAMA_KEYS})
        self.assertEqual(models.BY_KEY['kimodo_text'].name, 'Kimodo Text Adapters (Experimental, optional)')
        self.assertEqual(models.BY_KEY['kimodo_llama'].name, 'Llama 3 8B Instruct Weights for Descriptions (Experimental, optional)')
        for key, folder, repository, revision, file, size, digest in TEXT_FILES:
            model = models.BY_KEY[key]
            self.assertEqual((model.relative, model.url, model.download_url, model.minimum_bytes, model.expected_sha256),
                             (f'kimodo_text/{folder}/{file}', repository, f'{repository}/resolve/{revision}/{file}', size, digest))
        for key, file, size, digest in LLAMA_FILES:
            model = models.BY_KEY[key]
            self.assertEqual((model.relative, model.url, model.download_url, model.drive_id, model.minimum_bytes, model.expected_sha256),
                             ('kimodo_text/llama/' + file, 'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct', None, None, size, digest))
        self.assertEqual(models.STAGE_REQUIREMENTS['kimodo_text'], TEXT_KEYS + LLAMA_KEYS)
        self.assertEqual(models.STAGE_REQUIREMENTS['kimodo_fill'], KIMODO_KEYS)

    def test_only_the_three_leaders_are_rows_on_gvhmr_and_exmr(self):
        catalog = manager.workflow_catalog()
        for workflow, keys in catalog.items():
            shown = set(keys) & set(GROUP_KEYS)
            self.assertEqual(shown, {'kimodo', 'kimodo_text', 'kimodo_llama'} if workflow in ('gvhmr', 'exmr') else set(), workflow)

    def test_all_twelve_text_files_missing_in_an_empty_folder(self):
        with tempfile.TemporaryDirectory() as name, patch.object(models, 'MODELS', Path(name)):
            self.assertEqual([model.key for model in models.missing('kimodo_text')], list(TEXT_KEYS + LLAMA_KEYS))

    def test_download_all_leaves_the_groups_out(self):
        self.assertTrue(all(manager.download_all_skips(key) for key in GROUP_KEYS))

    def test_llama_instructions_name_the_licence_and_the_import(self):
        text = models.MANUAL_INSTRUCTIONS['kimodo_llama']
        for part in ('Hugging Face', 'licence', 'model-0000N-of-00004.safetensors', 'one folder', 'Import',
                     'The app does not need your account password or access token.'):
            self.assertIn(part, text)
        self.assertIn('https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct', models.manual_instructions_html('kimodo_llama'))


def small_records(keys):
    """The group records with tiny contents, so no gigabytes are written."""
    import dataclasses, hashlib
    return {key: dataclasses.replace(models.BY_KEY[key], minimum_bytes=len(content(key)),
                                     expected_sha256=hashlib.sha256(content(key)).hexdigest()) for key in keys}


def free_space(free):
    """What shutil.disk_usage returns, as far as the free-space check reads it."""
    from types import SimpleNamespace
    return SimpleNamespace(free=free)


def content(key):
    """What the patched network returns for a downloadable record, else a short stand-in."""
    url = models.BY_KEY[key].download_url
    return b'payload of ' + url.encode() if url else b'content of ' + key.encode()


class KimodoGroupTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve(); self.root = base / 'installation'; self.root.mkdir()
        self.folder = base / 'Downloads'; self.folder.mkdir()
        self.urls, self.failing = [], set()
        def get(url, **kw):self.urls.append(url); return Response(url, self.failing)
        paths = dict(HOME=self.root, ROOT=self.root, MODELS=self.root / 'models', UPSTREAM=self.root / 'third_party')
        for target in (patch.multiple(manager, **paths), patch.object(models, 'MODELS', paths['MODELS']),
                       patch.dict(models.BY_KEY, small_records(GROUP_KEYS)), patch('requests.get', get)):
            target.start(); self.addCleanup(target.stop)

    def install(self, key):
        model = models.BY_KEY[key]; model.path.parent.mkdir(parents=True, exist_ok=True)
        model.path.write_bytes(content(key)); stat = model.path.stat()
        atomic_json(manager.receipt_path(model), dict(sha256=model.expected_sha256, stat=[stat.st_size, stat.st_mtime_ns]))

    def downloads(self):
        for key, *_ in LLAMA_FILES:
            (self.folder / models.BY_KEY[key].path.name).write_bytes(content(key))

    def files(self):
        return sorted(path for path in manager.MODELS.rglob('*') if path.is_file()) if manager.MODELS.exists() else []

    def test_a_leader_is_ready_when_every_member_has_its_file_and_receipt(self):
        for key in LLAMA_KEYS:
            self.assertFalse(models.ready(models.BY_KEY['kimodo_llama']))
            self.install(key)
        self.assertTrue(models.ready(models.BY_KEY['kimodo_llama']))
        manager.receipt_path(models.BY_KEY['kimodo_llama_3']).unlink()
        self.assertFalse(models.ready(models.BY_KEY['kimodo_llama']))
        self.assertTrue(models.ready(models.BY_KEY['kimodo_llama_2']))

    def test_text_download_fetches_members_then_the_leader(self):
        path = models.download('kimodo_text')
        self.assertEqual(self.urls, [models.BY_KEY[key].download_url for key in TEXT_KEYS[1:] + TEXT_KEYS[:1]])
        self.assertEqual(path, models.BY_KEY['kimodo_text'].path)
        self.assertTrue(models.ready(models.BY_KEY['kimodo_text']))
        self.urls.clear(); models.download('kimodo_text'); self.assertEqual(self.urls, [])

    def test_ready_members_are_skipped(self):
        models.download('kimodo_text_tokenizer'); self.urls.clear()
        models.download('kimodo_text')
        self.assertNotIn(models.BY_KEY['kimodo_text_tokenizer'].download_url, self.urls)
        self.assertEqual(len(self.urls), 7)

    def test_a_failed_member_stops_before_the_leader(self):
        import requests
        self.failing.add(models.BY_KEY['kimodo_text_supervised'].download_url)
        with self.assertRaises(requests.HTTPError):models.download('kimodo_text')
        self.assertNotIn(models.BY_KEY['kimodo_text'].download_url, self.urls)
        self.assertFalse(models.ready(models.BY_KEY['kimodo_text']))

    def test_group_download_refuses_without_disk_space(self):
        import shutil
        for key in ('kimodo_text', 'kimodo'):
            needed = sum(models.BY_KEY[member].minimum_bytes for member in models.GROUPS[key]) + 256 * 1024**2
            with patch.object(shutil, 'disk_usage', return_value=free_space(needed - 1)):
                with self.assertRaisesRegex(OSError, r'^Not enough free disk space for '
                                            + models.BY_KEY[key].name.replace('(', r'\(').replace(')', r'\)')
                                            + r' \(0\.3 GiB needed\)\.$'):
                    models.download(key)
        self.assertEqual(self.urls, []); self.assertEqual(self.files(), [])

    def test_llama_import_takes_all_four_from_one_pick(self):
        self.downloads()
        path = models.import_model_file('kimodo_llama', self.folder / 'model-00003-of-00004.safetensors')
        self.assertEqual(path, models.BY_KEY['kimodo_llama'].path)
        self.assertTrue(models.ready(models.BY_KEY['kimodo_llama']))
        self.assertEqual(models.missing('kimodo_text')[0].key, 'kimodo_text')
        self.assertEqual([model.key for model in models.missing('kimodo_text')], list(TEXT_KEYS))

    def test_llama_import_prints_progress_lines(self):
        import contextlib, io, itertools, time
        self.downloads(); out = io.StringIO()
        with patch.object(time, 'monotonic', side_effect=itertools.count(0, 5)), contextlib.redirect_stdout(out):
            models.import_model_file('kimodo_llama', self.folder / 'model-00001-of-00004.safetensors')
        self.assertIn('Importing model-00004-of-00004.safetensors: 0.0 of 0.0 GiB', out.getvalue().splitlines())

    def test_llama_import_skips_ready_files(self):
        self.downloads(); self.install('kimodo_llama_2')
        before = models.BY_KEY['kimodo_llama_2'].path.stat().st_mtime_ns
        (self.folder / 'model-00002-of-00004.safetensors').write_bytes(b'x' * models.BY_KEY['kimodo_llama_2'].minimum_bytes)
        models.import_model_file('kimodo_llama', self.folder / 'model-00001-of-00004.safetensors')
        self.assertEqual(models.BY_KEY['kimodo_llama_2'].path.stat().st_mtime_ns, before)
        self.assertTrue(models.ready(models.BY_KEY['kimodo_llama']))

    def test_llama_import_refusals_copy_nothing(self):
        self.downloads()
        sibling = self.folder / 'model-00004-of-00004.safetensors'
        size = models.BY_KEY['kimodo_llama_4'].minimum_bytes
        cases = (
            (lambda: sibling.unlink(), 'model-00004-of-00004.safetensors is missing next to the file you chose.'),
            (lambda: sibling.write_bytes(b'short'), f'model-00004-of-00004.safetensors is not complete (expected {size} bytes).'),
            (lambda: sibling.write_bytes(b'y' * size), 'model-00004-of-00004.safetensors is not the expected Llama 3 8B Instruct file.'),
        )
        for change, sentence in cases:
            self.downloads(); change()
            with self.assertRaises((ValueError, OSError)) as caught:
                models.import_model_file('kimodo_llama', self.folder / 'model-00001-of-00004.safetensors')
            self.assertEqual(str(caught.exception), sentence)
            self.assertEqual(self.files(), [])

    def test_llama_import_refuses_without_disk_space(self):
        import shutil
        self.downloads()
        needed = sum(models.BY_KEY[key].minimum_bytes for key in LLAMA_KEYS) + 256 * 1024**2
        with patch.object(shutil, 'disk_usage', return_value=free_space(needed - 1)):
            with self.assertRaisesRegex(OSError, r'^Not enough free disk space for Llama 3 8B Instruct Weights'):
                models.import_model_file('kimodo_llama', self.folder / 'model-00002-of-00004.safetensors')
        self.assertEqual(self.files(), [])
        with patch.object(shutil, 'disk_usage', return_value=free_space(needed)):
            models.import_model_file('kimodo_llama', self.folder / 'model-00002-of-00004.safetensors')

    def test_llama_import_removes_parts_a_stopped_import_left(self):
        import shutil
        target = models.BY_KEY['kimodo_llama'].path.parent
        needed = sum(models.BY_KEY[key].minimum_bytes for key in LLAMA_KEYS) + 256 * 1024**2
        def stale():
            target.mkdir(parents=True, exist_ok=True)
            parts = (target / 'model-00002-of-00004.safetensors.import.part',
                     target / ('model-00003-of-00004.safetensors.' + '0' * 32 + '.part'))
            for part in parts:part.write_bytes(b'left by a stopped import')
            return parts
        def usage(folder):  # free space as it would be with the stale parts still on disk
            return free_space(needed - sum(part.stat().st_size for part in target.glob('*part')))
        with patch.object(shutil, 'disk_usage', side_effect=usage):
            self.downloads(); (self.folder / 'model-00004-of-00004.safetensors').write_bytes(
                b'y' * models.BY_KEY['kimodo_llama_4'].minimum_bytes)
            parts = stale()
            with self.assertRaisesRegex(ValueError, 'not the expected Llama 3 8B Instruct file'):
                models.import_model_file('kimodo_llama', self.folder / 'model-00001-of-00004.safetensors')
            self.assertFalse(any(part.exists() for part in parts)); self.assertEqual(self.files(), [])
            self.downloads(); parts = stale()
            models.import_model_file('kimodo_llama', self.folder / 'model-00001-of-00004.safetensors')
        self.assertFalse(any(part.exists() for part in parts))
        self.assertTrue(models.ready(models.BY_KEY['kimodo_llama']))
        self.assertEqual(sorted(path.name for path in target.iterdir() if path.suffix == '.part'), [])

    def kimodo_copy(self):
        """A downloaded copy of nvidia/Kimodo-SOMA-RP-v1.2, files at their paths in the repository; returns model.safetensors."""
        folder = self.folder / 'Kimodo-SOMA-RP-v1.2'
        for key in KIMODO_KEYS:
            path = folder / models.BY_KEY[key].relative.removeprefix('kimodo/Kimodo-SOMA-RP-v1.2/')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content(key))
        return folder / 'model.safetensors'

    def text_copy(self, supervised='LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised'):
        """The two adapter repositories downloaded side by side; returns the mntp adapter_model.safetensors."""
        for key in TEXT_KEYS:
            model = models.BY_KEY[key]
            folder = self.folder / (supervised if model.path.parent.name == 'supervised' else 'LLM2Vec-Meta-Llama-3-8B-Instruct-mntp')
            folder.mkdir(parents=True, exist_ok=True); (folder / model.path.name).write_bytes(content(key))
        return self.folder / 'LLM2Vec-Meta-Llama-3-8B-Instruct-mntp' / 'adapter_model.safetensors'

    def test_kimodo_import_takes_all_eight_from_one_pick(self):
        path = models.import_model_file('kimodo', self.kimodo_copy())
        self.assertEqual(path, models.BY_KEY['kimodo'].path); self.assertTrue(models.ready(models.BY_KEY['kimodo']))
        self.assertEqual(models.missing('kimodo_fill'), []); self.assertEqual(self.urls, [])

    def test_text_import_takes_both_adapters_from_one_pick(self):
        import shutil
        for supervised in ('LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised', 'supervised'):
            for folder in (manager.MODELS, self.folder):shutil.rmtree(folder, ignore_errors=True)
            self.folder.mkdir()
            path = models.import_model_file('kimodo_text', self.text_copy(supervised))
            self.assertEqual(path, models.BY_KEY['kimodo_text'].path); self.assertTrue(models.ready(models.BY_KEY['kimodo_text']), supervised)
            self.assertEqual([model.key for model in models.missing('kimodo_text')], list(LLAMA_KEYS))

    def test_group_import_refuses_another_file(self):
        self.kimodo_copy(); self.text_copy()
        for key, picked, sentence in (
                ('kimodo', self.folder / 'Kimodo-SOMA-RP-v1.2' / 'config.yaml', 'Choose model.safetensors of a downloaded copy of nvidia/Kimodo-SOMA-RP-v1.2.'),
                ('kimodo_text', self.folder / 'LLM2Vec-Meta-Llama-3-8B-Instruct-mntp' / 'config.json',
                 'Choose adapter_model.safetensors in the folder of McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp.')):
            with self.assertRaises(ValueError) as caught:models.import_model_file(key, picked)
            self.assertEqual(str(caught.exception), sentence); self.assertEqual(self.files(), [])

    def test_group_import_names_a_missing_member_where_it_is_expected(self):
        picked = self.kimodo_copy(); (picked.parent / 'stats/motion/global_root/std.npy').unlink()
        with self.assertRaises(ValueError) as caught:models.import_model_file('kimodo', picked)
        self.assertEqual(str(caught.exception), 'stats/motion/global_root/std.npy is missing next to the file you chose.')
        picked = self.text_copy(); (picked.parent / 'tokenizer.json').unlink()
        with self.assertRaises(ValueError) as caught:models.import_model_file('kimodo_text', picked)
        self.assertEqual(str(caught.exception), 'tokenizer.json is missing next to the file you chose.')
        import shutil; shutil.rmtree(self.folder / 'LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised'); picked = self.text_copy('elsewhere')
        with self.assertRaises(ValueError) as caught:models.import_model_file('kimodo_text', picked)
        self.assertEqual(str(caught.exception), 'adapter_model.safetensors of the supervised adapter is missing: expected in a folder named '
                         'LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised or supervised next to the folder of the file you chose.')
        self.assertEqual(self.files(), [])

    def test_group_import_with_a_wrong_hash_leaves_nothing_new(self):
        picked = self.kimodo_copy(); stats = picked.parent / 'stats/motion/body/mean.npy'
        stats.write_bytes(b'y' * models.BY_KEY['kimodo_stats_body_mean'].minimum_bytes)
        with self.assertRaises(ValueError) as caught:models.import_model_file('kimodo', picked)
        self.assertEqual(str(caught.exception), 'stats/motion/body/mean.npy is not the expected Kimodo-SOMA-RP-v1.2 file.')
        self.assertEqual(self.files(), [])
        picked = self.text_copy(); supervised = self.folder / 'LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised' / 'adapter_config.json'
        supervised.write_bytes(b'y' * models.BY_KEY['kimodo_text_supervised_config'].minimum_bytes)
        with self.assertRaises(ValueError) as caught:models.import_model_file('kimodo_text', picked)
        self.assertEqual(str(caught.exception), '../LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_config.json is not the expected LLM2Vec adapter file.')
        self.assertEqual(self.files(), [])

    def test_group_import_skips_ready_members_and_clears_stale_parts(self):
        import shutil
        self.install('kimodo'); before = models.BY_KEY['kimodo'].path.stat().st_mtime_ns
        picked = self.kimodo_copy(); picked.unlink()  # the large file need not be downloaded again
        picked.write_bytes(b'')                       # only its name is read when it is in place already
        part = models.BY_KEY['kimodo_config'].path.with_name('config.yaml.import.part'); part.write_bytes(b'left by a stopped import')
        needed = sum(models.BY_KEY[key].minimum_bytes for key in KIMODO_KEYS[1:]) + 256 * 1024**2
        with patch.object(shutil, 'disk_usage', return_value=free_space(needed)):
            models.import_model_file('kimodo', picked)
        self.assertEqual(models.BY_KEY['kimodo'].path.stat().st_mtime_ns, before); self.assertFalse(part.exists())
        self.assertTrue(models.ready(models.BY_KEY['kimodo']))

    def test_group_import_refuses_without_disk_space(self):
        import shutil
        picked = self.text_copy(); needed = sum(models.BY_KEY[key].minimum_bytes for key in TEXT_KEYS) + 256 * 1024**2
        with patch.object(shutil, 'disk_usage', return_value=free_space(needed - 1)):
            with self.assertRaisesRegex(OSError, r'^Not enough free disk space for Kimodo Text Adapters'):
                models.import_model_file('kimodo_text', picked)
        self.assertEqual(self.files(), [])

    def test_group_import_instructions_and_filters(self):
        for key, parts in (('kimodo', ('nvidia/Kimodo-SOMA-RP-v1.2', 'config.yaml', 'stats/motion', 'mean.npy', 'std.npy', 'choose model.safetensors')),
                           ('kimodo_text', ('LLM2Vec-Meta-Llama-3-8B-Instruct-mntp', 'LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised', 'supervised',
                                            'tokenizer.json', 'special_tokens_map.json', 'choose adapter_model.safetensors'))):
            for part in parts:self.assertIn(part, models.MANUAL_INSTRUCTIONS[key], (key, part))

    def test_a_complete_group_downloads_nothing_and_needs_no_space(self):
        import shutil
        for key in TEXT_KEYS:self.install(key)
        with patch.object(shutil, 'disk_usage', return_value=free_space(0)):
            self.assertEqual(models.download('kimodo_text'), models.BY_KEY['kimodo_text'].path)
        self.assertEqual(self.urls, [])

    def test_removing_a_leader_removes_the_whole_group(self):
        for key in LLAMA_KEYS:self.install(key)
        plan = manager.removal_plan(['kimodo_llama'])
        expected = {path for key in LLAMA_KEYS for path in (models.BY_KEY[key].path, manager.receipt_path(models.BY_KEY[key]))}
        self.assertEqual({item.path for item in plan}, expected)
        self.assertEqual(set(manager.recovery_roots(['kimodo_llama'], [])) & set(self.files()), expected)
        manager.remove_models(['kimodo_llama'], plan)
        self.assertEqual(self.files(), [])


try:
    import PySide6  # noqa: F401  the GUI runtime; the shared one skips these
except ImportError:
    PySide6 = None


@unittest.skipIf(PySide6 is None, 'needs the GUI runtime (PySide6)')
class KimodoSetupJobTests(unittest.TestCase):
    """The setup job as Model Manager runs it; the dialog itself needs a main window, so a stand-in hosts its methods."""

    def test_stage_is_named_and_watched(self):
        from mocap import gui
        self.assertEqual(gui.STAGES['kimodo_setup'], 'Setting up automatic fill')
        self.assertEqual(gui.MODEL_JOB_STAGES, {'download', 'import_model', 'kimodo_setup'})

    def host(self, busy=False):
        from types import SimpleNamespace
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from mocap.gui import ModelsDialog
        QApplication.instance() or QApplication([])
        self.jobs = []
        host = SimpleNamespace(window=SimpleNamespace(busy=busy), start_model_jobs=self.jobs.append)
        host.sync_kimodo_panel = lambda: ModelsDialog.sync_kimodo_panel(host)
        host.install_kimodo = lambda: ModelsDialog.install_kimodo(host)
        panel = ModelsDialog.kimodo_panel(host)
        self.addCleanup(panel.close)
        return host, panel, ModelsDialog

    def test_panel_shows_the_runtime_state(self):
        from mocap import kimodo_runtime
        from mocap.kimodo_runtime import SETUP
        from mocap.tooltips import tooltip_text
        with patch.object(kimodo_runtime, 'location', side_effect=RuntimeError(SETUP)):
            host, panel, _ = self.host()
        self.assertEqual(host.kimodo_note.text(), 'Automatic fill of hidden ranges (Experimental): install the runtime here, '
                         'then get the three Kimodo rows below.')
        self.assertEqual((host.kimodo_state.text(), host.kimodo_install.text()), ('Runtime not installed', 'Install Runtime'))
        self.assertTrue(host.kimodo_install.isEnabled())
        with patch.object(kimodo_runtime, 'location', return_value=Path('runtime_kimodo')):
            host, panel, dialog = self.host(busy=True)
        self.assertEqual((host.kimodo_state.text(), host.kimodo_install.text()), ('Runtime installed', 'Reinstall Runtime'))
        self.assertEqual(host.kimodo_install.toolTip(), tooltip_text('Check the installed runtime and install it again if the check fails.'))
        self.assertFalse(host.kimodo_install.isEnabled())                       # no second job while one runs
        host.window.busy = False
        with patch.object(kimodo_runtime, 'location', side_effect=RuntimeError(SETUP)):
            dialog.sync_kimodo_panel(host)                                       # what refresh() does when a job ends
        self.assertEqual((host.kimodo_state.text(), host.kimodo_install.text()), ('Runtime not installed', 'Install Runtime'))
        self.assertEqual(host.kimodo_install.toolTip(), tooltip_text('Download and install the Kimodo runtime as a job. RGB MoCap stays open.'))
        self.assertTrue(host.kimodo_install.isEnabled())

    def window(self, stage, failed=False, cancelled=False):
        """A stand-in main window at the end of its last job, for MainWindow.next_job and handle_event."""
        from unittest.mock import MagicMock
        from mocap.gui import MainWindow
        w = MagicMock(); w.jobs = []; w.failed, w.cancelled, w.current_stage = failed, cancelled, stage
        w._batch_started_at = None; w._people_batch = False; w._batch_has_motion = False; w.project = None
        status = []; w.status.setText.side_effect = status.append; w.status.text.side_effect = lambda: status[-1] if status else ''
        w.people_event.return_value = False; w._kimodo_setup_result = None
        w.handle_event = lambda event: MainWindow.handle_event(w, event)
        w.next_job = lambda: MainWindow.next_job(w)
        return w, status

    def test_setup_job_reports_what_it_did(self):
        for message in ('Automatic fill runtime verified.', 'Automatic fill runtime installed.'):
            w, status = self.window('kimodo_setup')
            w.handle_event(dict(type='complete', stage='kimodo_setup', output='runtime_kimodo', message=message))
            w.next_job()
            self.assertEqual(status[-1], message)                                # Model Manager is open: no pointer to it

    def test_a_model_job_end_rebuilds_the_hidden_ranges(self):
        from mocap.gui import MODEL_JOB_STAGES
        for stage in sorted(MODEL_JOB_STAGES):
            for ending in (dict(), dict(failed=True), dict(cancelled=True)):
                w, _ = self.window(stage, **ending); w.next_job()
                w.refresh_hidden_ranges_after_models.assert_called_once_with()
        for stage in ('gvhmr', 'take_join', 'export', 'correct'):
            w, _ = self.window(stage); w.next_job()
            w.refresh_hidden_ranges_after_models.assert_not_called()

    def test_group_rows_pick_their_leader_file(self):
        from types import SimpleNamespace
        from PySide6.QtWidgets import QFileDialog
        from mocap.gui import ModelsDialog
        host, panel, _ = self.host(); host.models = [models.BY_KEY[key] for key in ('kimodo', 'kimodo_text', 'kimodo_llama')]
        shown = []
        with patch.object(QFileDialog, 'getOpenFileName', lambda *a: shown.append(a[3]) or ('C:/x/' + str(len(shown)), '')):
            for key in ('kimodo', 'kimodo_text', 'kimodo_llama'):ModelsDialog.import_local_model(host, key)
        self.assertEqual(shown, ['Kimodo model (model.safetensors);;All files (*)', 'Kimodo text adapter (adapter_model.safetensors);;All files (*)',
                                 'Llama 3 weights (model-0000*-of-00004.safetensors);;All files (*)'])
        self.assertEqual([job[0][1][:2] for job in self.jobs], [['--model', 'kimodo'], ['--model', 'kimodo_text'], ['--model', 'kimodo_llama']])

    def test_install_asks_once_and_starts_the_job(self):
        from PySide6.QtWidgets import QMessageBox
        host, panel, dialog = self.host()
        asked = []
        def answer(reply):
            return lambda *a, **k: asked.append(a[2]) or reply
        with patch.object(QMessageBox, 'question', answer(QMessageBox.No)):
            host.kimodo_install.click()
        self.assertEqual(self.jobs, [])
        with patch.object(QMessageBox, 'question', answer(QMessageBox.Yes)):
            host.kimodo_install.click()
        self.assertEqual(self.jobs, [[('kimodo_setup', [])]])
        self.assertEqual(asked, ['This downloads NVIDIA Kimodo (Apache-2.0) and Python packages into the app folder '
                                 '(about 140 MB). The Kimodo model has its own licence (NVIDIA Open Model License), '
                                 'shown on its download page. Continue?'] * 2)
        host.window.busy = True                                                  # a job started meanwhile
        with patch.object(QMessageBox, 'question', answer(QMessageBox.Yes)):
            dialog.install_kimodo(host)
        self.assertEqual(self.jobs, [[('kimodo_setup', [])]])


if __name__ == '__main__':
    unittest.main()
