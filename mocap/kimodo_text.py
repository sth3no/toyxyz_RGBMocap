"""Description encoder: the 4096-number vector Kimodo conditions on, for one description.

The encoder is Llama 3 8B Instruct with LLM2Vec's two LoRA adapters; it fits neither this GPU nor the free RAM, so its 32
layers are streamed (`experiments/hidden-person-fill/step-5-gate.md`). The model is built empty; each layer's weights are
read from the four Llama files just before it runs, both adapters are added (W + alpha/r B A for each), and they are
dropped after it ran. The Llama files are never mapped: each tensor is read by its header offsets with plain file reads
(a mapping kept open keeps every page it touched in the process, and on Windows is charged to commit in full while
open). Only the embedding rows of the prompt's tokens are read. Tokenizing, the prompt format and the pooling are
LLM2Vec's own code. `normalise`, `files`, `tensors` and `read` are standard-library only;
torch, transformers and Kimodo are imported by `main`, which runs in the ExMR runtime with the Kimodo overlay.
"""
import argparse
import contextlib
import json
import math
import os
import struct
import time
import warnings
from pathlib import Path

from .hidden_ranges import normalise_text as normalise
from .paths import MODELS

STAGE = 'take_join'
VERSION = 1
NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'  # LLM2Vec picks its prompt format by this name
LLAMA_FILES = tuple(f'model-0000{n}-of-00004.safetensors' for n in range(1, 5))
EMBEDDING = 'model.embed_tokens.weight'
NO_CUDA = 'A description requires a CUDA GPU.'
NO_MEMORY = 'Not enough free video memory to read the description. Close other GPU work and try again.'
WRONG_FILES = 'The description files are not the expected version. Get them again in Model Manager.'
SIZES = dict(BF16=2, F16=2, F32=4)  # bytes per value of the safetensors dtypes read here
EMPTY = 'The description is empty.'


def files():
    folder = MODELS/'kimodo_text'
    return {k: folder/k for k in ('llama', 'mntp', 'supervised')}


def tensors(paths):
    """Tensor name -> (file, dtype, shape, absolute offset, byte length), from the safetensors files' own headers: 8 bytes
    of little-endian header length N, N bytes of JSON, then the data (offsets count from byte 8 + N). RuntimeError for a
    name found in two files or a malformed header."""
    where = {}
    for path in paths:
        with open(path, 'rb') as handle:
            try:
                size, = struct.unpack('<Q', handle.read(8));header = json.loads(handle.read(size))
                for name, entry in header.items():
                    if name == '__metadata__':continue
                    begin, end = entry['data_offsets']
                    if name in where or not 0 <= begin <= end:raise ValueError
                    where[name] = (path, entry['dtype'], tuple(entry['shape']), 8+size+begin, end-begin)
            except (struct.error, ValueError, KeyError, TypeError) as error:raise RuntimeError(WRONG_FILES) from error
    return where


def read(handle, entry, dtype, rows=None):
    """The bytes of a tensor from its open file (`rows`, a (first, count) pair, reads rows of its first axis only);
    RuntimeError for another dtype or a byte length that does not fit the shape. Plain reads: a safetensors or torch
    mapping of the whole file is charged to Windows commit in full while it is open."""
    _, kind, shape, start, length = entry
    if kind != dtype or not shape or length != math.prod(shape)*SIZES[dtype]:raise RuntimeError(WRONG_FILES)
    if rows is not None:
        row = length//shape[0];first, count = rows
        if not 0 <= first <= first+count <= shape[0]:raise RuntimeError(WRONG_FILES)
        start, length = start+first*row, count*row
    data = bytearray(length);handle.seek(start)
    if handle.readinto(data) != length:raise RuntimeError(WRONG_FILES)
    return data


def _adapters(folders, device):
    """(tensors as float32, alpha/r, target module names) of each adapter; RuntimeError for one that is not plain LoRA."""
    from safetensors import safe_open
    out = []
    for folder in folders:
        cfg = json.loads((folder/'adapter_config.json').read_text(encoding='utf-8'))
        if cfg.get('peft_type') != 'LORA' or cfg.get('bias', 'none') != 'none' or any(cfg.get(k) for k in (
                'use_rslora', 'use_dora', 'fan_in_fan_out', 'modules_to_save', 'rank_pattern', 'alpha_pattern', 'layers_to_transform')):
            raise RuntimeError(WRONG_FILES)
        with safe_open(str(folder/'adapter_model.safetensors'), framework='pt', device=device) as f:
            tensors = {k: f.get_tensor(k).float() for k in f.keys()}
        out.append((tensors, cfg['lora_alpha']/cfg['r'], frozenset(cfg['target_modules'])))
    return out


def _embeddings(ids, where, device):
    """The embedding rows of the tokens `ids` (1, L) as float32 (1, L, 4096), read row by row: the table stays on disk."""
    import torch
    entry = where.get(EMBEDDING)
    if entry is None:raise RuntimeError(WRONG_FILES)
    distinct, inverse = torch.unique(ids, return_inverse=True)
    with open(entry[0], 'rb') as handle:
        rows = torch.stack([torch.frombuffer(read(handle, entry, 'BF16', (i, 1)), dtype=torch.bfloat16) for i in distinct.tolist()])
    return rows.to(device).float()[inverse]


def _encode(text, emit):
    from .compute_policy import configure_worker_cpu, gpu_memory_fraction
    threads = configure_worker_cpu(STAGE);os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoTokenizer
    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
    if not torch.cuda.is_available():raise RuntimeError(NO_CUDA)
    torch.set_num_threads(threads);torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction(*torch.cuda.mem_get_info()))
    started, folders, device = time.monotonic(), files(), 'cuda'
    config = AutoConfig.from_pretrained(folders['mntp']);config._name_or_path = NAME;layers = config.num_hidden_layers
    emit('progress', stage=STAGE, current=0, total=layers, message='Description: loading the text encoder')
    tok = AutoTokenizer.from_pretrained(folders['mntp']);tok.pad_token = tok.eos_token;tok.padding_side = 'left'
    # Built empty on the meta device, its parameters float32 as in the measured reference: each layer's bfloat16 weights
    # compute in float32. (accelerate's init_empty_weights allocates each parameter on the CPU first, which left 3.2 GiB
    # of commit charge behind.) The rotary embedding has no parameters, only buffers computed from the config.
    with torch.device('meta'):model = LlamaBiModel(config)
    model.rotary_emb = type(model.rotary_emb)(config=config).to(device)
    model.eval();adapters = _adapters((folders['mntp'], folders['supervised']), device)
    where = tensors([folders['llama']/name for name in LLAMA_FILES])

    def put(module, prefix):
        with contextlib.ExitStack() as stack:
            handles = {}
            for name, param in list(module.named_parameters()):
                entry = where.get(f'model.{prefix}{name}')
                if entry is None:raise RuntimeError(WRONG_FILES)
                if entry[0] not in handles:handles[entry[0]] = stack.enter_context(open(entry[0], 'rb'))
                W = torch.frombuffer(read(handles[entry[0]], entry, 'BF16'), dtype=torch.bfloat16)
                if W.numel() != param.numel():raise RuntimeError(WRONG_FILES)
                W = W.view(param.shape).to(device)
                if name.endswith('.weight'):
                    stem = f'base_model.model.{prefix}{name[:-len(".weight")]}'
                    for lora, scale, targets in adapters:
                        if name.split('.')[-2] in targets:
                            A, B = lora.get(stem+'.lora_A.weight'), lora.get(stem+'.lora_B.weight')
                            if A is None or B is None:raise RuntimeError(WRONG_FILES)
                            W = W.float()+scale*(B@A)
                set_module_tensor_to_device(module, name, device, value=W.to(torch.bfloat16))

    def drop(module):
        for name, _ in list(module.named_parameters()):set_module_tensor_to_device(module, name, 'meta')

    def before(i):
        def hook(module, args):
            emit('progress', stage=STAGE, current=i, total=layers, message=f'Description: layer {i+1} of {layers}');put(module, f'layers.{i}.')
        return hook
    put(model.norm, 'norm.')
    for i, layer in enumerate(model.layers):layer.register_forward_pre_hook(before(i));layer.register_forward_hook(lambda m, a, o:drop(m))
    enc = LLM2Vec(model=model, tokenizer=tok, pooling_mode='mean', max_length=512)
    features = enc.tokenize([enc.prepare_for_tokenization(enc._convert_to_str('', text))])
    features = {k: x.to(device) for k, x in features.items()};embeds = _embeddings(features['input_ids'], where, device)
    with torch.no_grad():
        # As LLM2Vec.forward, with the prompt's embedding rows in place of the whole table.
        states = model(inputs_embeds=embeds, attention_mask=features['attention_mask'], use_cache=False).last_hidden_state
        vector = enc.get_pooling(features, states)[0].float().cpu().numpy()
    if vector.shape != (4096,) or not np.isfinite(vector).all():raise RuntimeError('The text encoder returned values that are not numbers.')
    emit('progress', stage=STAGE, current=layers, total=layers, message='Description: read')
    emit('info', stage=STAGE, message=f'Description read in {time.monotonic()-started:.1f} seconds; '
                                      f'peak video memory {torch.cuda.max_memory_allocated()/2**30:.2f} GiB.')
    return vector


def _save(vector, path):
    import numpy as np
    tmp = path.with_name(path.name+'.tmp')
    try:
        with open(tmp, 'wb') as handle:np.save(handle, vector)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True);raise


def main(argv, emit):
    parser = argparse.ArgumentParser(prog='kimodo_worker_boot.py text');parser.add_argument('--prompt-file', required=True);parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        try:text = Path(args.prompt_file).read_text(encoding='utf-8-sig')
        except (OSError, UnicodeDecodeError) as error:raise ValueError(f'Could not read the description. {error}') from error
        text = normalise(text)
        if not text:raise ValueError(EMPTY)
        from .kimodo_runtime import text_issue
        reason = text_issue()
        if reason:raise RuntimeError(reason)
        # PEFT warns on import about transformers older than 5.6; the pinned 5.1.0 is the verified one.
        warnings.filterwarnings('ignore', message='Support for Transformers < 5.6', category=FutureWarning)
        _save(_encode(text, emit), Path(args.out))
    except Exception as error:
        message = str(error) or type(error).__name__
        if type(error).__name__ == 'OutOfMemoryError' or 'out of memory' in message.lower():message = NO_MEMORY
        emit('failed', stage=STAGE, message=message);return 1
    return 0
