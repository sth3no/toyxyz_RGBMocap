"""Inference device for ExMR and person tracking: CUDA, else Apple MPS, else CPU.

With CUDA the callers keep their previous device strings, dtypes, memory APIs
and cache identities; nothing here changes that path. Apple MPS gets its own
identity ("Apple MPS <chip>") so caches never mix, and its own memory budget:
CPU and GPU share one unified pool, so only the RAM reserve is subtracted
(never a separate GPU reserve as well). The other models remain CUDA-only.
"""
from __future__ import annotations
import functools
import gc
import os
import sys

GIB = 1024**3
MACOS = sys.platform == 'darwin'
# Allocation limits relative to Metal's recommendedMaxWorkingSetSize. PyTorch's
# unified-memory defaults (1.7 / 1.4) let a job swap; 1.0 makes it fail cleanly.
MPS_WATERMARKS = {'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '1.0', 'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.8'}
CUDA_ONLY_MODELS = ('GVHMR with HTD-Refine, FootMR and HandFlow, FlowHMR, MAMMA, PromptHMR, '
                    'GEM-X, SAM 3.1 Find person and PEAR')
MACOS_MESSAGE = ('On macOS, ExMR and person tracking run on the Apple GPU (MPS). '
                 + CUDA_ONLY_MODELS + ' need an NVIDIA CUDA GPU and do not run on macOS.')
_fraction = None  # MPS allocator fraction set by this worker, None for PyTorch's default


def cuda_required(message):
    """The existing CUDA error text, or on macOS which models can run here."""
    return message + ' ' + MACOS_MESSAGE if MACOS else message


def configure_worker_environment():
    """Call before torch first allocates on MPS. Resource Safety Off keeps PyTorch's own limits."""
    from .resource_safety import enabled
    if not MACOS:return
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # unported operators run on the CPU
    if enabled() and not any(key in os.environ for key in MPS_WATERMARKS):
        os.environ.update(MPS_WATERMARKS)


def mps_available(torch=None):
    if torch is None:
        import torch
    backend = getattr(torch.backends, 'mps', None)
    return bool(backend is not None and backend.is_available())


def kind(torch=None):
    """'cuda', 'mps' or 'cpu'; CUDA wins whenever it is available."""
    if torch is None:
        import torch
    if torch.cuda.is_available():return 'cuda'
    return 'mps' if mps_available(torch) else 'cpu'


def accelerated(torch=None):
    return kind(torch) != 'cpu'


@functools.lru_cache(maxsize=1)
def chip():
    import subprocess
    try:
        name = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ''
    name = name or __import__('platform').machine()
    return name[len('Apple '):] if name.startswith('Apple ') else name


def device_name(torch=None):
    """Cache identity of the device: unchanged CUDA name, else 'Apple MPS <chip>' or 'CPU'."""
    if torch is None:
        import torch
    device = kind(torch)
    if device == 'cuda':return torch.cuda.get_device_name()
    return 'Apple MPS ' + chip() if device == 'mps' else 'CPU'


def model_kwargs(device):
    """Constructor device keyword: empty for CUDA so the upstream CUDA defaults stay in force."""
    return {} if device == 'cuda' else {'device': device}


def yolo_device(device):
    """Ultralytics device argument: the existing GPU index 0 on CUDA."""
    return 0 if device == 'cuda' else device


def is_oom(error, torch=None):
    """CUDA OutOfMemoryError, or the plain RuntimeError that the MPS allocator raises."""
    if torch is None:
        import torch
    if isinstance(error, torch.cuda.OutOfMemoryError):return True
    return isinstance(error, RuntimeError) and 'MPS backend out of memory' in str(error)


def empty_cache(torch=None):
    if torch is None:
        import torch
    if torch.cuda.is_initialized():torch.cuda.empty_cache()
    elif mps_available(torch):torch.mps.empty_cache()


def mps_budget(torch=None, memory=None):
    """(free, total) bytes for this process on MPS from one unified pool.

    total is Metal's recommended working set. free is what Metal still allows
    beyond the driver's current allocation, limited by the RAM that is actually
    available above the system RAM reserve. No separate GPU reserve is taken.
    """
    if torch is None:
        import torch
    if memory is None:
        import psutil
        memory = psutil.virtual_memory()
    from .compute_policy import ram_reserve
    total = int(torch.mps.recommended_max_memory())
    used = int(torch.mps.driver_allocated_memory())
    free = max(0, min(total - used, int(memory.available) - ram_reserve(int(memory.total))))
    return free, total


def mps_memory_fraction(free, total, allocated=0):
    """Allocator fraction of the recommended working set; None with Resource Safety off.

    Off removes the app's cap and leaves PyTorch's own limit. MPS treats 0 as
    unlimited, so a fraction never drops below 1%.
    """
    from .resource_safety import enabled
    if not enabled():return None
    return max(.01, min(1., (allocated + free) / total)) if total > 0 else .01


def set_mps_budget(torch=None):
    """MPS counterpart of model_session.set_memory_budget: fix the cap once per phase."""
    global _fraction
    if torch is None:
        import torch
    from .model_session import current
    session = current()
    free, total = mps_budget(torch)
    if session is None or not session.cap_set:
        fraction = mps_memory_fraction(free, total, int(torch.mps.current_allocated_memory()))
        if fraction is not None:torch.mps.set_per_process_memory_fraction(fraction)
        _fraction = fraction
        if session is not None:session.cap_set = True
    return free, total


def mps_available_bytes(torch=None, memory=None):
    """MPS counterpart of memory_mode.available_vram, including a reusable resident model."""
    if torch is None:
        import torch
    free, total = mps_budget(torch, memory)
    allocated = int(torch.mps.current_allocated_memory())
    room = free if _fraction is None else min(free, max(0, int(_fraction * total) - allocated))
    from .model_session import current
    session = current()
    return room + (session.resident_bytes if session is not None else 0)


def acquire(key, factory, device):
    """model_session.acquire; on MPS also record the resident bytes CUDA statistics cannot see."""
    from .model_session import acquire as session_acquire, current
    if device != 'mps':return session_acquire(key, factory)
    import torch
    session = current()
    fresh = session is None or session.value is None or session.key != key
    before = int(torch.mps.current_allocated_memory())
    value = session_acquire(key, factory)
    if session is not None and fresh and session.value is value:
        session.resident_bytes = max(0, int(torch.mps.current_allocated_memory()) - before)
    return value


def run(stage, operation, fast_gib, emit=None, device='cuda'):
    """memory_mode.run on CUDA; on MPS the same once-only Low Memory retry after an MPS OOM."""
    if device == 'cuda':
        from .memory_mode import run as cuda_run
        return cuda_run(stage, operation, fast_gib, emit)
    from .memory_mode import announce, decide, effective, requested_mode
    from .model_session import current
    import torch
    session = current()
    available = mps_available_bytes(torch) if device == 'mps' else 0
    selected = decide('low_memory' if session is not None and session.force_low else requested_mode(),
                      available, fast_gib * GIB)
    reason = ('Selected by the user.' if requested_mode() == 'low_memory' else
              f'Available unified-memory budget {available/GIB:.2f} GiB; fast-path estimate {fast_gib:.2f} GiB.')
    announce(stage, selected, reason, emit)
    failed = False
    try:
        with effective(selected):
            return operation(selected == 'low_memory')
    except RuntimeError as error:
        if selected == 'low_memory' or not is_oom(error, torch):raise
        failed = True
    if failed:
        if session is not None:
            session.discard()
            session.force_low = True
        gc.collect()
        empty_cache(torch)
        announce(stage, 'low_memory', 'Automatic recovery after the Apple GPU ran out of memory; retrying once.', emit)
        with effective('low_memory'):
            return operation(True)


def memory_report(device, torch=None):
    """Receipt fields for MPS (CUDA receipts keep their existing fields)."""
    if device != 'mps':return {}
    if torch is None:
        import torch
    return dict(compute_device=device_name(torch),
                mps_driver_allocated_gib=torch.mps.driver_allocated_memory()/GIB,
                mps_fraction=_fraction)
