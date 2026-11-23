"""Device selection, MPS out-of-memory detection and the unified-memory budget (no GPU needed)."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mocap import compute_device as cd

GIB = 1024**3


class OutOfMemoryError(RuntimeError):
    pass


def fake_torch(cuda=False, mps=False, total=16 * GIB, driver=0, allocated=0):
    calls = []
    return SimpleNamespace(
        calls=calls,
        cuda=SimpleNamespace(is_available=lambda: cuda, is_initialized=lambda: cuda,
                             get_device_name=lambda: 'NVIDIA GeForce RTX 5090',
                             empty_cache=lambda: calls.append('cuda.empty_cache'),
                             OutOfMemoryError=OutOfMemoryError),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        mps=SimpleNamespace(recommended_max_memory=lambda: total, driver_allocated_memory=lambda: driver,
                            current_allocated_memory=lambda: allocated,
                            empty_cache=lambda: calls.append('mps.empty_cache'),
                            set_per_process_memory_fraction=lambda value: calls.append(('fraction', value))))


def memory(available, total=24 * GIB):
    return SimpleNamespace(available=available, total=total)


class DeviceSelectionTests(unittest.TestCase):
    def test_cuda_wins_then_mps_then_cpu(self):
        self.assertEqual(cd.kind(fake_torch(cuda=True, mps=True)), 'cuda')
        self.assertEqual(cd.kind(fake_torch(mps=True)), 'mps')
        self.assertEqual(cd.kind(fake_torch()), 'cpu')
        self.assertEqual(cd.kind(SimpleNamespace(cuda=fake_torch().cuda, backends=SimpleNamespace())), 'cpu')

    def test_cuda_keeps_its_identity_and_arguments(self):
        torch = fake_torch(cuda=True, mps=True)
        self.assertEqual(cd.device_name(torch), 'NVIDIA GeForce RTX 5090')
        self.assertEqual(cd.model_kwargs('cuda'), {})
        self.assertEqual(cd.yolo_device('cuda'), 0)

    def test_mps_has_a_distinct_identity(self):
        with patch.object(cd, 'chip', return_value='M5'):
            self.assertEqual(cd.device_name(fake_torch(mps=True)), 'Apple MPS M5')
        self.assertEqual(cd.device_name(fake_torch()), 'CPU')
        self.assertEqual(cd.model_kwargs('mps'), {'device': 'mps'})
        self.assertEqual(cd.yolo_device('mps'), 'mps')

    def test_cache_release_follows_the_device(self):
        torch = fake_torch(cuda=True, mps=True);cd.empty_cache(torch)
        self.assertEqual(torch.calls, ['cuda.empty_cache'])
        torch = fake_torch(mps=True);cd.empty_cache(torch)
        self.assertEqual(torch.calls, ['mps.empty_cache'])
        torch = fake_torch();cd.empty_cache(torch)
        self.assertEqual(torch.calls, [])

    def test_macos_message_only_on_macos(self):
        with patch.object(cd, 'MACOS', False):
            self.assertEqual(cd.cuda_required('Needs CUDA.'), 'Needs CUDA.')
        with patch.object(cd, 'MACOS', True):
            text = cd.cuda_required('Needs CUDA.')
        self.assertTrue(text.startswith('Needs CUDA. On macOS, ExMR and person tracking run on the Apple GPU'))
        for name in ('GVHMR', 'FlowHMR', 'MAMMA', 'PromptHMR', 'GEM-X', 'SAM 3.1', 'PEAR'):
            self.assertIn(name, text)


class OutOfMemoryTests(unittest.TestCase):
    def test_cuda_and_mps_out_of_memory_are_recognized(self):
        torch = fake_torch()
        self.assertTrue(cd.is_oom(OutOfMemoryError('CUDA out of memory'), torch))
        self.assertTrue(cd.is_oom(RuntimeError('MPS backend out of memory (MPS allocated: 1 GiB, '
                                               'other allocations: 0 bytes, max allowed: 2 GiB).'), torch))
        self.assertFalse(cd.is_oom(RuntimeError('mat1 and mat2 shapes cannot be multiplied'), torch))
        self.assertFalse(cd.is_oom(ValueError('MPS backend out of memory'), torch))

    def test_real_torch_mps_error_type(self):
        import torch
        self.assertTrue(cd.is_oom(RuntimeError('MPS backend out of memory (MPS allocated: 0 bytes)'), torch))
        self.assertTrue(cd.is_oom(torch.cuda.OutOfMemoryError('CUDA out of memory.'), torch))

    def operation(self, failures, error=None):
        seen = []
        def run(low):
            seen.append(low)
            if len(seen) <= failures:
                raise error or RuntimeError('MPS backend out of memory (MPS allocated: 9 GiB)')
            return 'done'
        return run, seen

    def run_mps(self, operation, requested='auto', available=8 * GIB):
        events = []
        env = {'RGB_MOCAP_COMPUTE_MODE': requested}
        with patch.dict(os.environ, env), patch.object(cd, 'mps_available_bytes', return_value=available), \
                patch('mocap.memory_mode.announce', lambda stage, mode, reason, emit=None: events.append((mode, reason))), \
                patch.dict('sys.modules', {'torch': fake_torch(mps=True)}):
            os.environ.pop('RGB_MOCAP_EFFECTIVE_MODE', None)
            return cd.run('ExMR Sapiens', operation, 6., None, 'mps'), events

    def test_automatic_retries_once_in_low_memory_after_an_mps_oom(self):
        operation, seen = self.operation(1)
        result, events = self.run_mps(operation)
        self.assertEqual(result, 'done')
        self.assertEqual(seen, [False, True])
        self.assertEqual([mode for mode, _ in events], ['auto', 'low_memory'])
        self.assertIn('unified-memory budget 8.00 GiB', events[0][1])
        self.assertIn('retrying once', events[1][1])

    def test_a_second_oom_or_an_explicit_low_memory_oom_is_raised(self):
        operation, seen = self.operation(2)
        with self.assertRaisesRegex(RuntimeError, 'MPS backend out of memory'):
            self.run_mps(operation)
        self.assertEqual(seen, [False, True])
        operation, seen = self.operation(1)
        with self.assertRaisesRegex(RuntimeError, 'MPS backend out of memory'):
            self.run_mps(operation, requested='low_memory')
        self.assertEqual(seen, [True])

    def test_small_budget_selects_low_memory_and_other_errors_propagate(self):
        operation, seen = self.operation(0)
        self.assertEqual(self.run_mps(operation, available=2 * GIB)[0], 'done')
        self.assertEqual(seen, [True])
        operation, seen = self.operation(1, RuntimeError('Invalid buffer size'))
        with self.assertRaisesRegex(RuntimeError, 'Invalid buffer size'):
            self.run_mps(operation)
        self.assertEqual(seen, [False])

    def test_cuda_uses_the_existing_memory_mode_run(self):
        with patch('mocap.memory_mode.run', return_value='cuda path') as existing:
            self.assertEqual(cd.run('ExMR Sapiens', lambda low: None, 6., 'emit', 'cuda'), 'cuda path')
        existing.assert_called_once()
        self.assertEqual(existing.call_args.args[0], 'ExMR Sapiens')
        self.assertEqual(existing.call_args.args[2:], (6., 'emit'))


class UnifiedBudgetTests(unittest.TestCase):
    def test_budget_subtracts_only_the_ram_reserve(self):
        from mocap.compute_policy import ram_reserve
        torch = fake_torch(mps=True, total=int(17.8 * GIB), driver=1 * GIB)
        # RAM limits: 10 GiB available minus the 1.92 GiB reserve of a 24 GiB Mac; no extra GPU reserve.
        free, total = cd.mps_budget(torch, memory(10 * GIB))
        self.assertEqual(total, int(17.8 * GIB))
        self.assertEqual(free, 10 * GIB - ram_reserve(24 * GIB))
        # Metal limits: the recommended working set minus what the driver already holds.
        free, _ = cd.mps_budget(torch, memory(40 * GIB, 64 * GIB))
        self.assertEqual(free, int(17.8 * GIB) - GIB)
        self.assertEqual(cd.mps_budget(torch, memory(GIB))[0], 0)

    def test_fraction_follows_resource_safety(self):
        with patch.dict(os.environ, {'RGB_MOCAP_RESOURCE_SAFETY': '1'}):
            self.assertAlmostEqual(cd.mps_memory_fraction(4 * GIB, 16 * GIB, 0), .25)
            self.assertAlmostEqual(cd.mps_memory_fraction(4 * GIB, 16 * GIB, 4 * GIB), .5)
            self.assertEqual(cd.mps_memory_fraction(0, 16 * GIB), .01)  # 0 would mean unlimited on MPS
            self.assertEqual(cd.mps_memory_fraction(40 * GIB, 16 * GIB), 1.)
        with patch.dict(os.environ, {'RGB_MOCAP_RESOURCE_SAFETY': '0'}):
            self.assertIsNone(cd.mps_memory_fraction(4 * GIB, 16 * GIB))

    def test_set_budget_caps_once_per_session_and_not_with_safety_off(self):
        from mocap.model_session import ModelSession
        torch = fake_torch(mps=True, total=16 * GIB)
        with patch.dict(os.environ, {'RGB_MOCAP_RESOURCE_SAFETY': '1'}), \
                patch('psutil.virtual_memory', return_value=memory(10 * GIB, 24 * GIB)), \
                ModelSession().activate():
            cd.set_mps_budget(torch);cd.set_mps_budget(torch)
        fractions = [call for call in torch.calls if call[0] == 'fraction']
        self.assertEqual(len(fractions), 1)
        self.assertAlmostEqual(fractions[0][1], (10 * GIB - 1.92 * GIB) / (16 * GIB), places=3)
        torch = fake_torch(mps=True, total=16 * GIB)
        with patch.dict(os.environ, {'RGB_MOCAP_RESOURCE_SAFETY': '0'}), \
                patch('psutil.virtual_memory', return_value=memory(10 * GIB, 24 * GIB)):
            cd.set_mps_budget(torch)
            self.assertEqual(torch.calls, [])
            from mocap.compute_policy import ram_reserve
            self.assertEqual(cd.mps_available_bytes(torch, memory(10 * GIB, 24 * GIB)), 10 * GIB - ram_reserve(24 * GIB))

    def test_watermarks_are_set_with_resource_safety_only(self):
        keys = [*cd.MPS_WATERMARKS, 'PYTORCH_ENABLE_MPS_FALLBACK']
        base = {k: v for k, v in os.environ.items() if k not in keys}
        with patch.object(cd, 'MACOS', True):
            with patch.dict(os.environ, {**base, 'RGB_MOCAP_RESOURCE_SAFETY': '1'}, clear=True):
                cd.configure_worker_environment()
                self.assertEqual(os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'], '1.0')
                self.assertEqual(os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'], '0.8')
                self.assertEqual(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'], '1')
            with patch.dict(os.environ, {**base, 'RGB_MOCAP_RESOURCE_SAFETY': '0'}, clear=True):
                cd.configure_worker_environment()
                self.assertNotIn('PYTORCH_MPS_HIGH_WATERMARK_RATIO', os.environ)
            with patch.dict(os.environ, {**base, 'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0'}, clear=True):
                cd.configure_worker_environment()  # a user's own limit is kept as it is
                self.assertEqual(os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'], '0.0')
                self.assertNotIn('PYTORCH_MPS_LOW_WATERMARK_RATIO', os.environ)
        with patch.object(cd, 'MACOS', False), patch.dict(os.environ, base, clear=True):
            cd.configure_worker_environment()
            self.assertFalse(any(key in os.environ for key in keys))


class CudaIdentityTests(unittest.TestCase):
    """The MPS port aliases the bytes it changed, so existing CUDA caches keep their identities.
    A later numerical edit must change these values deliberately."""
    PRE_MPS = {
        'mocap/exmr_backend.py': '77c7162093f5bbf9b4dba46287f6542d5069cd3aef24444f531eac94335bd82f',
        'mocap/engines.py': '74512a9b2fb87f4fafb9f0bc963f5b5887e4069ffc59c24f656f984b9c8066fd',
        'mocap/worker.py': '97df91aacab5e68618e0b61752bce3844e08b47ff7e6a69315b6b8a32f96b0f1',
        'third_party/ExMR/exmr/preprocessing/keypoints/sapiens2.py': 'd3a7364e0e50efca09f83843ee2ad95a06d38a458c34df04498c97b8d691c1b9',
        'third_party/ExMR/third_party/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py':
            '577a570d35889939bd3b0aacef4faa30598ec246b97b8f37b70c247d3fa0aa4f',
        'third_party/ExMR/exmr/geometry/smoothing.py': 'b11435d96fdd7257667f8c46a04e7475813f783b349a6c583c060541cafcb7fa',
    }

    def test_changed_files_keep_their_cuda_cache_identity(self):
        from mocap.paths import ROOT
        from mocap.resource_cache import code_hash
        from mocap.cache_compat import discovery_identity
        for name, digest in self.PRE_MPS.items():
            with self.subTest(file=name):
                self.assertEqual(code_hash(ROOT / name), digest)
        self.assertEqual(discovery_identity(ROOT / 'mocap/person_tracking.py'),
                         '39f6422998f9d77a501cd4ac88c2a62cf312cf2be0f01129e9c0e7b8a0dd2b5f')

    def test_macos_refuses_cuda_only_workers_with_the_model_list(self):
        from mocap import paths
        with patch.object(paths.sys, 'platform', 'darwin'):
            for stage in ('flow_features', 'pva', 'handflow', 'sam31_people', 'mamma_masks', 'gemx_reconstruct'):
                with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, 'needs an NVIDIA CUDA GPU.*ExMR and person tracking'):
                    paths.worker_command(stage)
            for stage in ('exmr_preprocess', 'people', 'track'):
                with self.subTest(stage=stage):
                    try:
                        paths.worker_command(stage)
                    except RuntimeError as error:  # a missing development runtime, never the CUDA refusal
                        self.assertNotIn('NVIDIA CUDA', str(error))


if __name__ == '__main__':
    unittest.main()
