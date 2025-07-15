"""
Convert Hugging Face models to GGML format
"""
import argparse
from ast import Dict, Tuple
from collections import OrderedDict
import copy
import json
import struct
import time, os
import io
import pickle
import re
from pathlib import Path
from enum import Enum, IntEnum
from pathlib import Path
from typing import IO, Any, Iterable, List, Optional, Tuple
import numpy as np
import math, gc

import torch
from torch import nn
from torch import _weight_norm

from sentencepiece import SentencePieceProcessor  # type: ignore

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK_K  = 256

GGML_MEM_ALIGN = 16

class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 8
    Q4_K = 12

class ChatModelAP(IntEnum):
    Text            = 0x01
    ImageInput      = 0x02
    ImageOutput     = 0x04
    AudioInput      = 0x08
    AudioOutput     = 0x10
    VideoInput      = 0x20
    VideoOutput     = 0x40

ModelTypeTagChatImageIn = ((ChatModelAP.Text.value + ChatModelAP.ImageInput.value) >> 1) << 24
ModelTypeTagChatAudioIn = ((ChatModelAP.Text.value + ChatModelAP.AudioInput.value) >> 1) << 24
ModelTypeTagChatImageVideoIn = ((ChatModelAP.Text.value + ChatModelAP.ImageInput.value + ChatModelAP.VideoInput.value) >> 1) << 24
ModelTypeTagChatImageVideoAudioInAudioOut = ((ChatModelAP.Text.value + ChatModelAP.ImageInput.value + ChatModelAP.VideoInput.value + ChatModelAP.AudioInput.value + ChatModelAP.AudioOutput.value) >> 1) << 24

class ModelType(Enum):
    CHATGLM = 1
    CHATGLM2 = 2
    CHATGLM3 = 3
    CODEGEEX2 = 4
    CharacterGLM = 5
    CHATGLM4 = 6
    CODEGEEX4 = 7
    GLM4        = 8

    InternLM    = 0x100
    InternLM2   = 0x101
    InternLM2_1 = 0x102
    InternLM3   = 0x103

    LlaMA2      = 0x150
    CodeLlaMA   = 0x151
    WizardCoder = 0x152
    WizardLM    = 0x153
    WizardMath  = 0x154
    TigerBot    = 0x155
    LlaMA2Plus  = 0x156
    Megrez      = 0x157
    Falcon3     = 0x158
    RekaFlash3  = 0x159

    BaiChuanLlama   = 0x200
    BaiChuan        = 0x201
    BaiChuanM1      = 0x202

    DeepSeek            = 0x300
    DeepSeekCoder       = 0x301
    CodeFuseDeepSeek    = 0x302
    NuminaMath          = 0x303
    DeepSeekV2Light     = 0x320
    DeepSeekV2          = 0x321
    DeepSeekV3Light     = 0x322
    DeepSeekV3          = 0x323
    DeepSeekV1          = 0x324
    GigaChat            = 0x325
    BailingMoE          = 0x326
    XVERSEMoE           = 0x327

    Yi = 0x400
    MAP_Neo = 0x401

    Phi2                = 0x500
    Phi2_v2             = 0x501
    DolphinPhi2         = 0x510
    DolphinPhi2_v2      = 0x511
    Phi3                = 0x520
    Phi3_ScalingSU      = 0x521
    Phi3_ScalingSU2     = 0x522
    Phi3_ScalingSU3     = 0x523
    Phi3MoE_ScalingSU   = 0x530
    Phi4                = 0x531
    Phi4_Mini           = 0x532

    Mistral = 0x600
    Mixtral = 0x601
    OpenChat = 0x602
    NeuralBeagle = 0x603
    Starling = 0x604
    WizardLMMoE = 0x605
    Mistral2 = 0x606
    DeepHermes3Mistral = 0x607

    QWen    = 0x700
    QWen2   = 0x710
    QWen2Tie = 0x711
    QWen2MoE = 0x750
    MarcoO1  = 0x751
    QwQ      = 0x752
    ReaderLM2= 0x753
    DeepSeek_R1_Distill_QWen    = 0x754
    QWen3                       = 0x755
    DeepSeek_R1_Distill_QWen3   = 0x756

    BlueLM  = 0x800

    StableLM    = 0x900

    Orion       = 0x1000

    MiniCPM     = 0x1100
    MiniCPM2    = 0x1101   # updated chat template, no tie_word_embeddings=False
    MiniCPM_MoE = 0x1102
    MiniCPM3    = 0x1110
    MiniCPM4    = 0x1111

    Persimmon   = 0x1200
    Fuyu        = 0x1201

    Gemma       = 0x1300
    Gemma2      = 0x1301
    Gemma3      = 0x1302

    CohereCommand       = 0x1400
    CohereAya23         = 0x1401
    CohereCommandR7B    = 0x1402

    Grok1         = 0x1500

    Zhinao        = 0x1600

    LlaMA3        = 0x1700
    SmolLM        = 0x1701
    GroqToolUse   = 0x1702
    LlaMA31       = 0x1703
    LlaMA32       = 0x1704
    Exaone        = 0x1705
    DeepSeek_R1_Distill_LlaMA = 0x1706
    Aquila2       = 0x1707
    ERNIE_DENSE   = 0x1708

    StarCoder2    = 0x1800

    XVERSE        = 0x1900

    Index         = 0x1a00

    OLMoE         = 0x1b00
    OLMo2         = 0x1b01

    AlphaGeometryLM = 0x1c00

    GraniteMoE      = 0x1d00
    Granite         = 0x1d01

    TeleChat2       = 0x1e00

    HunYuanDense    = 0x1f00
    HunYuanMoEV1    = 0x1f01

    MoonLight       = 0x2000

    Instella        = 0x2100

    DeciLM          = 0x2200

    SolarPro        = 0x2300

    Apriel          = 0x2400

    ERNIE_MoE       = 0x2500

    PenguMoE        = 0x2600

    SmolLM3         = 0x2700

    Exaone4         = 0x2800

    BCE_Embedding           = 0x10000100
    BCE_ReRanker            = 0x10000101
    BGE_M3                  = 0x10000102
    BGE_ReRankerM3          = 0x10000103
    MiniCPM_Embedding_Light = 0x10000104
    MiniCPM_ReRanker_Light  = 0x10000105
    OrpheusTTS              = 0x10000106
    OuteTTSLlaMA            = 0x10000107
    OuteTTSQwen3            = 0x10000108
    QWen3_Embedding         = 0x10000109
    QWen3_ReRanker          = 0x1000010A

    LlaMAMulti    = 0x20000001

    LlaMA4                  = ModelTypeTagChatImageIn + 0x0000001
    Gemma3Vis               = ModelTypeTagChatImageIn + 0x0000011

    Qwen2Audio              = ModelTypeTagChatAudioIn + 0x0000001

    Qwen2_5VL               = ModelTypeTagChatImageVideoIn + 0x0000001
    KimiVL                  = ModelTypeTagChatImageVideoIn + 0x0000100
    SmolVLM                 = ModelTypeTagChatImageVideoIn + 0x0000200

    MiniCPM_O               = ModelTypeTagChatImageVideoAudioInAudioOut + 0x0000001

class TokenType(Enum):
    UNDEFINED    = 0
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6
class TokenizerType(Enum):
    BPE1         = 0
    BPE2         = 1
    Unigram      = 2

g_tokenizer_type = TokenizerType.BPE1

g_special_tokens: Dict = {}

def pad_to_len(l: list, to_len: int, v = 0) -> list:
    assert len(l) <= to_len
    n = len(l)
    return l + [v] * (to_len - n)

def quantize_q8_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q8_0 in ggml.c
    assert tensor.shape[tensor.ndim - 1] % GGML_QK8_0 == 0
    tensor = tensor.contiguous().view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    assert tensor.shape[tensor.ndim - 1] % GGML_QK4_0 == 0
    tensor = tensor.contiguous().view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into a int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor

def quantize_q4_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_1 in ggml.c
    assert tensor.shape[tensor.ndim - 1] % GGML_QK4_1 == 0
    tensor = tensor.contiguous().view(-1, GGML_QK4_1)
    abs_max_indices = tensor.max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    abs_min_indices = tensor.min(dim=-1, keepdim=True).indices
    min_values = torch.take_along_dim(tensor, abs_min_indices, dim=-1)
    scale = (max_values - min_values) / 15
    tensor = ((tensor - min_values) / scale).round().clamp(min=0, max=15).char()
    # compress two int4 weights into a int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), min_values.half().view(torch.int8), tensor), dim=-1)
    return tensor

@torch.jit.script
def qkx2_quants(x: torch.Tensor, nmax, rmin, rdelta, nstep: int, use_mad: bool):
    assert x.dim() == 1
    N = x.shape[0]
    av_x = torch.norm(x) / math.sqrt(N)
    weights = av_x + torch.abs(x)
    min_x = torch.min(x)
    max_x = torch.max(x)
    sum_w = torch.sum(weights)
    sum_x = torch.sum(weights * x)
    if min_x > 0: min_x = torch.tensor(0)
    if min_x == max_x:
        L = torch.zeros(N)
        return torch.tensor(0.0), -min_x, L

    iscale = nmax / (max_x - min_x)
    scale = 1 / iscale
    best_mad = 0
    L = (iscale * (x - min_x)).round().clamp(min=0, max=nmax)
    diff = scale * L + min_x - x
    diff = torch.abs(diff) if use_mad else torch.square(diff)
    best_mad = torch.sum(weights * diff)

    if nstep < 1:
        return scale, -min_x, L

    for istep in range(nstep):
        iscale = (rmin + rdelta * istep + nmax)/(max_x - min_x)
        l = (iscale * (x - min_x)).round().clamp(min=0, max=nmax)
        sum_l  = torch.sum(weights * l)
        sum_l2 = torch.sum(weights * l * l)
        sum_xl = torch.sum(weights * l * x)

        D = sum_w * sum_l2 - sum_l * sum_l
        if D > 0:
            this_scale = (sum_w * sum_xl - sum_x * sum_l)/D
            this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D
            if this_min > 0:
                this_min = torch.tensor(0)
                this_scale = sum_xl / sum_l2

            diff = this_scale * l + this_min - x
            diff = torch.abs(diff) if use_mad else torch.square(diff)
            mad = torch.sum(weights * diff)
            if mad < best_mad:
                L = l
                scale = this_scale
                min_x = this_min

    return scale, -min_x, L

@torch.jit.script
def quantize_q4_k_block(tensor: torch.Tensor, GGML_QK_K: int) -> torch.CharTensor:
    assert tensor.shape == (GGML_QK_K, )
    tensor = tensor.view(-1, 32)

    subblocks = [qkx2_quants(tensor[i], torch.tensor(15), torch.tensor(-1.0), torch.tensor(0.1), 20, False) for i in range(tensor.shape[0])]
    scale = torch.stack([x[0] for x in subblocks])
    min_x = torch.stack([x[1] for x in subblocks])

    max_scale = torch.max(scale)
    max_min   = torch.max(min_x)

    inv_scale = torch.tensor(63.0) / max_scale if max_scale > 0 else torch.tensor(0.0)
    inv_min   = torch.tensor(64.0) / max_min   if max_min   > 0 else torch.tensor(0.0)

    ls = (inv_scale * scale).round().clamp(max=63)
    lm = (inv_min   * min_x).round().clamp(max=63)

    scales = [0] * 12
    for j in range(GGML_QK_K // 32):
        ls_j = int(ls[j].item())
        lm_j = int(lm[j].item())
        if j < 4:
            scales[j]   = ls_j
            scales[j+4] = lm_j
        else:
            scales[j+4]  = (ls_j & 0xF) | ((lm_j & 0xF) << 4)
            scales[j-4] |= ((ls_j >> 4) << 6)
            scales[j-0] |= ((lm_j >> 4) << 6)

    scales = torch.tensor(scales, dtype=torch.uint8)

    d    = (max_scale / 63.0).half()
    dmin = (max_min   / 63.0).half()

    recovered_scale = (ls * d   ).view(-1, 1)
    recovered_min   = (lm * dmin).view(-1, 1)

    L = ((1 / recovered_scale) * (tensor + recovered_min)).round().clamp(min=0, max=15).to(torch.uint8)
    L = L.view(-1, 2, 32)
    L = L[:, 0, :] | (L[:, 1, :] << 4)

    r = torch.cat((d.view(1).view(torch.int8), dmin.view(1).view(torch.int8), scales.view(torch.uint8), L.flatten().view(torch.uint8)), dim=-1)

    return r

@torch.jit.script
def quantize_q4_k(tensor: torch.Tensor, GGML_QK_K: int) -> torch.CharTensor:
    # equivalent to dequantize_row_q4_K in ggml-quants.c
    assert tensor.shape[tensor.ndim - 1] % GGML_QK_K == 0
    tensor = tensor.view(-1, GGML_QK_K)
    blocks = [quantize_q4_k_block(tensor[i], GGML_QK_K) for i in range(tensor.shape[0])]
    tensor = torch.cat(blocks, dim=-1)
    return tensor

def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GGMLType):
    assert tensor.dtype == torch.float32

    # tensor name
    f.write(struct.pack("i", len(name.encode())))
    f.write(name.encode())

    # tensor shape & dtype
    f.write(struct.pack("i" * (2 + tensor.ndim), tensor.ndim, *tensor.shape, ggml_type.value))

    # tensor data
    try:
        if ggml_type == GGMLType.F32:
            tensor = tensor.float()
        elif ggml_type == GGMLType.F16:
            tensor = tensor.half()
        elif ggml_type == GGMLType.Q8_0:
            tensor = quantize_q8_0(tensor)
        elif ggml_type == GGMLType.Q4_0:
            tensor = quantize_q4_0(tensor)
        elif ggml_type == GGMLType.Q4_1:
            tensor = quantize_q4_1(tensor)
        elif ggml_type == GGMLType.Q4_K:
            tensor = quantize_q4_k(tensor, GGML_QK_K)
        else:
            raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")
    except Exception as e:
        raise Exception(f"Error dumping tensor {name} of shape {tensor.shape}: {e}")

    # align address
    aligned_pos = (f.tell() + (GGML_MEM_ALIGN - 1)) // GGML_MEM_ALIGN * GGML_MEM_ALIGN
    f.seek(aligned_pos)
    tensor.detach().numpy().tofile(f)

def load_model_file(path: Path) -> Dict:
    print(f"loading {path} ...")
    fp = open(path, 'rb')
    first8 = fp.read(8)
    fp.seek(0)
    if first8[:2] == b'PK':
        content = torch.load(fp, map_location=torch.device('cpu'))
        if sorted(list(content.keys())) == ['metadata', 'state_dict']:
            content = content['state_dict']
        return content.items()
    elif struct.unpack('<Q', first8)[0] < 16 * 1024 * 1024:
        from safetensors import safe_open
        tensors = []
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors.append((key, f.get_tensor(key)))
        return tensors
    else:
        raise ValueError(f"unknown format: {path}")

class AttributeDict(dict):
    def __getattr__(self, key):
        return self.__getitem__(key) if key in self else None

    __setattr__ = dict.__setitem__

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

class LoRAState:
    def __init__(self, path: Path, verbose: bool = False) -> None:
        self.verbose = verbose

        with open(path / 'adapter_config.json', 'r') as fp:
            config = AttributeDict(json.load(fp))
            assert config.peft_type == 'LORA', f"peft_type must be `LORA`"
            self.scaling = config.lora_alpha / config.r
            self.fan_in_fan_out = config.fan_in_fan_out

        self.files = []
        for glob in ["adapter_model.safetensors", "adapter_model.bin"]:
            files = list(path.glob(glob))
            if len(files) > 0:
                files.sort()
                self.files = files
                break

        if len(self.files) < 1:
            raise Exception('can not find model files, or unsupported format.')

        self.tensor_dict = {}
        for f in self.files:
            for k, v in load_model_file(f):
                self.tensor_dict[k] = v

        if self.verbose:
            print(f"LoRA tensors: {self.tensor_dict.keys()}")

    # based on: https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/blob/main/scripts/merge_llama3_with_chinese_lora_low_mem.py
    def merge_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        saved_key = 'base_model.model.' + name
        if saved_key in self.tensor_dict:
            if self.verbose:
                print(f"LoRA replacing: {name}")
            return self.tensor_dict[saved_key]

        lora_key_A = saved_key.replace('.weight','.lora_A.weight')
        if lora_key_A in self.tensor_dict:
            if self.verbose:
                print(f"LoRA merging: {name}")

            lora_key_B = saved_key.replace('.weight','.lora_B.weight')
            return tensor.float() + (
                        transpose(
                            self.tensor_dict[lora_key_B].float()
                          @ self.tensor_dict[lora_key_A].float(), self.fan_in_fan_out) * self.scaling
                    )

        return tensor

g_lora: LoRAState = None
g_model_meta = {}

def load_all_model_files(model_files) -> Dict:
    global g_lora
    for f in model_files:
        r = {}
        data = load_model_file(f)
        for k, v in data:
            if k in r:
                raise Exception(f"tensor {k} already loaded. maybe it is stored cross files?")
            r[k] = g_lora.merge_tensor(k, v) if g_lora is not None else v
        yield r

def tabulate(data, headers = []) -> str:
    all = []
    all.append(headers)
    col_num = len(headers)
    for d in data:
        row = [str(x) for x in d]
        all.append(row)
        col_num = max(col_num, len(row))

    def get_col_width(n: int) -> int:
        nonlocal all
        r = 0
        for i in range(len(all)):
            row = all[i]
            if n < len(row): r = max(r, len(row[n]))
        return r

    widths = [get_col_width(i) for i in range(col_num)]
    sep = '+-' + '-+-'.join(['-' * w for w in widths]) + '-+\n'

    def make_row(i: int) -> int:
        nonlocal all, col_num, widths
        row = all[i]
        str_row = [str(row[n]) if n < len(row) else ' ' for n in range(col_num)]

        return '| ' + ' | '.join([f"{{:<{widths[n]}}}".format(str_row[n]) for n in range(col_num)]) + ' |\n'

    return sep + make_row(0) + sep + ''.join([make_row(i) for i in range(1, len(all))]) + sep

def tqdm(items, desc='') -> Iterable[any]:

    def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 60, fill = '█', printEnd = "\r", auto_nl = True):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        if (iteration == total) and auto_nl:
            print()

    def format_time(t) -> str:
        return "{:.2f}s".format(t)

    total = len(items)
    t = time.perf_counter()
    for i, x in enumerate(items):
        yield x
        used = time.perf_counter() - t
        per_item = used / (i + 1)
        remain = (total - i - 1) * per_item
        print_progress_bar(i + 1, total, prefix=desc, suffix=f"({i}/{total}) {format_time(per_item)}/it rem: {format_time(remain)}")

def dump_state_dict(f, weight_names, model_files, ggml_type, config, state_dict_pp, loader_fun = None):
    tensor_info = []
    converted_names = []

    # Note: incremental loading and converting

    remaining: List = weight_names.copy()

    if loader_fun is None:
        loader_fun = load_all_model_files

    for state_dict in loader_fun(model_files):
        this_round = []
        state_dict = state_dict_pp(config, state_dict)

        for x in state_dict:
            if x in remaining:
                this_round.append(x)
                remaining.remove(x)
            else:
                print(f"Warning: tensor {x} is ignored")

        for name in tqdm(this_round, desc="Dumping ..."):
            tensor: torch.Tensor = state_dict[name]

            tensor = tensor.float()

            if tensor.ndim in {2, 3, 4}:
                if tensor.shape[tensor.ndim - 1] % GGML_QK8_0 == 0:
                    tensor_ggml_type = ggml_type
                else:
                    tensor_ggml_type = GGMLType.F16
            else:
                # 1d weight: convert it to float32
                assert tensor.ndim == 1, f'shape of {name} = {tensor.shape}'
                tensor_ggml_type = GGMLType.F32

            dump_tensor(f, name, tensor, tensor_ggml_type)
            tensor_info.append((name, tensor.shape, tensor_ggml_type.name))

        gc.collect()

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"]))

    if len(tensor_info) != len(weight_names):
        raise Exception(f'not all tensors are converted: {remaining}')

class SentencePieceVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:
        self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens: Dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens, encoding='utf-8'))
        else:
            added_tokens = {}
        vocab_size: int = self.sentencepiece_tokenizer.vocab_size()
        print("vocab_size ", vocab_size)
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            if actual_ids == list(range(len(added_tokens))):
                raise Exception(f"added token IDs ({actual_ids}) are starting from 0. `added_token.json` seems WRONG.\n\nDelete it and try again.")
            else:
                raise Exception(f"Expected added token IDs to be sequential and start at {vocab_size}; got {actual_ids}.\n\nDelete it and try again.")
        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base: int = vocab_size
        self.vocab_size: int = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def sentencepiece_tokens(self) -> Iterable[Tuple[bytes, float]]:
        tokenizer = self.sentencepiece_tokenizer
        for i in range(tokenizer.vocab_size()):
            text: bytes
            if tokenizer.is_unknown(i):
                text = " \u2047 ".encode("utf-8")
            elif tokenizer.is_control(i):
                text = b""
            elif tokenizer.is_byte(i):
                piece = tokenizer.id_to_piece(i)
                if len(piece) != 6:
                    raise Exception(f"Invalid token: {piece}")
                byte_value = int(piece[3:-1], 16)
                text = struct.pack("B", byte_value)
            else:
                text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
            score: float = tokenizer.get_score(i)
            yield text, score

    def added_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def write_vocab(self, fout: io.BufferedWriter) -> None:
        for text, score in self.all_tokens():
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", score))

        # marks the end of vocab
        fout.write(struct.pack("i", -1))

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

class SimpleVocab:
    def __init__(self, fname_tokenizer: Path) -> None:
        vocab = []
        with open(fname_tokenizer, encoding='utf-8') as f:
            for l in f.readlines():
                l = l.split()
                if len(l) != 2:
                    break
                vocab.append((l[0], float(l[1])))

        self.vocab = vocab
        print("vocab_size ", len(vocab))

    def sentencepiece_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for (piece, score) in self.vocab:
            if (len(piece) == 6) and (piece[0] == '<') and (piece[5] == '>'):
                byte_value = int(piece[3:-1], 16)
                text = struct.pack("B", byte_value)
                score = -1000.0
            else:
                text = piece.replace("\u2581", " ").encode("utf-8")
            yield text, score

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.sentencepiece_tokens()

    def write_vocab(self, fout: io.BufferedWriter) -> None:
        for text, score in self.all_tokens():
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", score))

        # marks the end of vocab
        fout.write(struct.pack("i", -1))

    def __repr__(self) -> str:
        return f"<SimpleVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

class UnigramTokenizerJsonVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:
        global g_tokenizer_type
        g_tokenizer_type = TokenizerType.Unigram

        model = json.load(open(fname_tokenizer / "tokenizer.json", encoding='utf-8'))
        if model['model']['type'] != 'Unigram':
            raise Exception(f"Unigram expected, but {model['model']['type']} encountered.")

        if fname_added_tokens is not None:
            raise Exception(f"TODO: fname_added_tokens")

        self.vocal_tokens = model['model']['vocab']
        self.vocab_size: int = len(self.vocal_tokens)

        print("Unigram vocab_size ", self.vocab_size)

    def tokenizer_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for v in self.vocal_tokens:
            tok = v[0]
            score = v[1]

            text: bytes = tok.encode('utf-8')
            if tok == '<unk>':
                text = " \u2047 ".encode("utf-8")
            else:
                text = tok.replace("\u2581", " ").encode("utf-8")

            yield text, score

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.tokenizer_tokens()

    def write_vocab(self, fout: io.BufferedWriter) -> None:
        for text, score in self.all_tokens():
            fout.write(struct.pack("<i", len(text)))
            fout.write(text)
            fout.write(struct.pack("<f", score))

        # marks the end of vocab
        fout.write(struct.pack("i", -1))

    def __repr__(self) -> str:
        return f"<UnigramTokenizerJsonVocab with {self.vocab_size} tokens>"

class FastTokenizerVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:

        global g_special_tokens
        global g_tokenizer_type
        g_tokenizer_type = TokenizerType.BPE2

        model = json.load(open(fname_tokenizer / "tokenizer.json", encoding='utf-8'))
        if model['model']['type'] != 'BPE':
            raise Exception(f"BPE expected, but {model['model']['type']} encountered.")

        all_tokens: Dict[str, int] = {}

        if fname_added_tokens is not None:
            all_tokens = json.load(open(fname_added_tokens))

        for tok, tokidx in sorted([(t['content'], t['id']) for t in model['added_tokens']], key=lambda x: x[1]):
            all_tokens[tok] = tokidx
            g_special_tokens[tok] = tokidx

        for tok in model['model']['vocab'].keys():
            tokidx = model['model']['vocab'][tok]
            all_tokens[tok] = tokidx

        all_ids = sorted(list(all_tokens.values()))

        vocab_size: int = all_ids[-1] + 1
        if vocab_size != len(all_ids):
            raise Exception(f'vacab size is {vocab_size}, but {len(all_ids)} tokens are loaded.')

        items: OrderedDict[str, int] = OrderedDict()
        items = sorted(all_tokens.items(), key=lambda text_idx: text_idx[1])

        self.vocab_size: int = vocab_size
        self.vocal_tokens = items
        self.merges = model['model']['merges']

        print("vocab_size ", self.vocab_size)

    def tokenizer_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for tok, idx in self.vocal_tokens:
            t = TokenType.NORMAL.value
            if tok in g_special_tokens:
                t = TokenType.USER_DEFINED.value
                text = tok.encode("utf-8")
            else:
                matches = re.findall('<0x([0-9a-fA-F]+)>', tok)
                if len(matches) == 1:
                    text: bytes = bytes([int(matches[0], 16)])
                else:
                    text: bytes = tok.replace("\u2581", " ").encode("utf-8")

            yield text, t

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.tokenizer_tokens()

    def write_vocab(self, fout: io.BufferedWriter) -> None:
        for text, tt in self.all_tokens():
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("B", tt))

        # marks the end of vocab
        fout.write(struct.pack("i", -1))

        for s in self.merges:
            if isinstance(s, list):
                s = f"{s[0]} {s[1]}"
            text = s.encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)

        fout.write(struct.pack("i", -1))

    def __repr__(self) -> str:
        return f"<FastTokenizerVocab with {self.vocab_size} tokens>"

class GenericBPETokenizerVocab:
    def __init__(self, vocab_fn: Path, merges_fn: Path) -> None:

        all_tokens: Dict[str, int] = {}

        vocab_dict = json.load(open(vocab_fn, encoding='utf-8'))

        for tok in vocab_dict.keys():
            tokidx = vocab_dict[tok]
            all_tokens[tok] = tokidx

        all_ids = sorted(list(all_tokens.values()))

        vocab_size: int = all_ids[-1] + 1
        if vocab_size != len(all_ids):
            raise Exception(f'vacab size is {vocab_size}, but {len(all_ids)} tokens are loaded.')

        items: OrderedDict[str, int] = OrderedDict()
        items = sorted(all_tokens.items(), key=lambda text_idx: text_idx[1])

        self.vocab_size: int = vocab_size
        self.vocal_tokens = items

        with open(merges_fn, 'r', encoding='utf-8') as f:
            self.merges = [l.strip() for l in f.readlines()]

        if self.merges[0].startswith('#'):
            self.merges.pop(0)

        print("vocab_size ", self.vocab_size)

    def tokenizer_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for tok, idx in self.vocal_tokens:
            text: bytes = tok.encode('utf-8')
            t = TokenType.NORMAL.value
            if tok in g_special_tokens:
                t = TokenType.USER_DEFINED.value

            yield text, t

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.tokenizer_tokens()

    def write_vocab(self, fout: io.BufferedWriter) -> None:
        for text, tt in self.all_tokens():
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("B", tt))

        # marks the end of vocab
        fout.write(struct.pack("i", -1))

        for s in self.merges:
            text = s.encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)

        fout.write(struct.pack("i", -1))

    def __repr__(self) -> str:
        return f"<GenericBPETokenizerVocab with {self.vocab_size} tokens>"

class TikTokenizerVocab:

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def __init__(self, fname_tokenizer: Any, fname_added_tokens: Optional[Path]) -> None:
        global g_tokenizer_type
        g_tokenizer_type = TokenizerType.BPE2

        if isinstance(fname_tokenizer, Path):
            from transformers import AutoTokenizer  # type: ignore[attr-defined]
            tokenizer = AutoTokenizer.from_pretrained(fname_tokenizer, trust_remote_code=True)
            vocab_size = max(tokenizer.get_vocab().values()) + 1
        else:
            tokenizer = fname_tokenizer
            vocab_size = len(tokenizer.mergeable_ranks)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[self.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = TikTokenizerVocab.bpe(mergeable_ranks, token, max_rank=rank)
            if len(merged) != 2:
                continue
            merges.append(' '.join(map(self.token_bytes_to_string, merged)))

        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in vocab.items()}
        added_vocab = tokenizer.special_tokens

        vocab_tokens = []

        for i in range(vocab_size):
            if i not in reverse_vocab:
                pad_token = f"[PAD{i}]".encode("utf-8")
                vocab_tokens.append((bytearray(pad_token), TokenType.USER_DEFINED.value))
            elif reverse_vocab[i] in added_vocab:
                vocab_tokens.append((reverse_vocab[i], TokenType.CONTROL.value))
            else:
                vocab_tokens.append((reverse_vocab[i], TokenType.NORMAL.value))

        self.vocab_size: int = vocab_size
        self.merges = merges
        self.vocab_tokens = vocab_tokens
        print("vocab_size ", self.vocab_size)

    def tokenizer_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for tok, t in self.vocab_tokens:
            yield tok, t

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.tokenizer_tokens()

    def write_vocab(self, fout: io.BufferedWriter) -> None:
        for tok, tt in self.all_tokens():
            text = tok.encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("B", tt))

        # marks the end of vocab
        fout.write(struct.pack("i", -1))

        for s in self.merges:
            text = s.encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)

        # marks the end of additional vocab
        fout.write(struct.pack("i", -1))

    def __repr__(self) -> str:
        return f"<TikTokenizerVocab with {self.vocab_size} tokens>"

def load_vocab_from_tiktok_mergeable_ranks(path9):
    import base64
    PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def _load_tiktoken_bpe(tiktoken_bpe_file: str):
        with open(tiktoken_bpe_file, "rb") as f:
            contents = f.read()
        return {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }

    mergeable_ranks = _load_tiktoken_bpe(path9)

    class TempTokenizer:
        def __init__(self, mergeable_ranks) -> None:
            self.mergeable_ranks = mergeable_ranks
            self.special_tokens = {}

    return TikTokenizerVocab(TempTokenizer(mergeable_ranks), fname_added_tokens = None)

class BaseConverter:
    FILE_VERSION = 1

    @classmethod
    def pp(cls, config, name: str, tensor):
        return tensor

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            tensor = cls.pp(config, name, tensor)
            state_dict[name] = tensor
        return state_dict

    @classmethod
    def convert(cls, config, model_files, vocab: Any, ggml_type, save_path):

        def write_size_to_pos(f, offset):
            size = f.tell()
            f.seek(offset)
            f.write(struct.pack("i", size))
            f.seek(0, 2)

        # convert all weights to fp16
        with open(save_path, "wb") as f:
            f.write(b"ggmm")  # magic
            GGMM_VER = 1
            f.write(struct.pack("i" * 4, GGMM_VER, 0, 0, 0))

            meta_bytes = json.dumps(g_model_meta, ensure_ascii=False).encode()
            rounded_len = ((len(meta_bytes) + 3) // 4) * 4
            meta_bytes += b'\x00' * (rounded_len - len(meta_bytes))
            f.write(meta_bytes)
            write_size_to_pos(f, 8)

            f.write(struct.pack("ii", cls.MODEL_TYPE.value, cls.FILE_VERSION))  # model type & version
            cls.dump_config(f, config, ggml_type)
            write_size_to_pos(f, 12)

            vocab.write_vocab(f)
            write_size_to_pos(f, 16)

            weight_names = cls.get_weight_names(config)
            dump_state_dict(f, weight_names, model_files, ggml_type, config, cls.state_dict_pp)

        print(f"{cls.MODEL_TYPE.name} GGML model saved to {save_path}")

def permute(weights: torch.Tensor, n_head: int) -> torch.Tensor:
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                   .swapaxes(1, 2)
                   .reshape(weights.shape))

def permute_pair(weights: torch.Tensor, n_head: int) -> torch.Tensor:
    return (weights.reshape(n_head, weights.shape[0] // n_head // 2, 2, *weights.shape[1:])
                   .swapaxes(1, 2)
                   .reshape(weights.shape))

class InternLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.InternLM

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight') or name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        rope_theta = 10000
        rope_scaling = 1.0
        num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        if config.rotary is not None:
            assert config.rotary['type'] == 'dynamic', "rotary['type'] must be dynamic"
            rope_theta = config.rotary['base']
            rope_scaling = config.rotary.get("scaling_factor", 1.0)

        if config.bias:
            assert num_key_value_heads == config.num_attention_heads, "num_key_value_heads must equal to num_attention_heads"
            assert rope_theta == 10000, "rotary['base'] must be 10000"
            assert rope_scaling == 1.0, "rotary['base'] must be 10000"

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        if not config.bias:
            config_values = [
                num_key_value_heads,
            ]
            f.write(struct.pack("i" * len(config_values), *config_values))
            f.write(struct.pack("<f", rope_theta))
            f.write(struct.pack("<f", rope_scaling))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            if config.bias:
                weight_names += [
                    f"model.layers.{i}.self_attn.q_proj.weight",
                    f"model.layers.{i}.self_attn.q_proj.bias",
                    f"model.layers.{i}.self_attn.k_proj.weight",
                    f"model.layers.{i}.self_attn.k_proj.bias",
                    f"model.layers.{i}.self_attn.v_proj.weight",
                    f"model.layers.{i}.self_attn.v_proj.bias",
                    f"model.layers.{i}.self_attn.o_proj.weight",
                    f"model.layers.{i}.self_attn.o_proj.bias",
                ]
            else:
                weight_names += [
                    f"model.layers.{i}.self_attn.q_proj.weight",
                    f"model.layers.{i}.self_attn.k_proj.weight",
                    f"model.layers.{i}.self_attn.v_proj.weight",
                    f"model.layers.{i}.self_attn.o_proj.weight",
                ]
            weight_names += [
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.down_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                    f"model.layers.{i}.input_layernorm.weight",
                    f"model.layers.{i}.post_attention_layernorm.weight",
                ]
        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]
        return weight_names

class InternLM2Converter(BaseConverter):
    MODEL_TYPE = ModelType.InternLM2_1

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            if name.endswith('attention.wqkv.weight'):

                kv_groups = config.num_attention_heads // config.num_key_value_heads
                head_dim = config.hidden_size // config.num_attention_heads
                gs = 2 + kv_groups
                h = tensor.shape[0] // (gs * head_dim)

                v = tensor.view(h, gs, head_dim, config.hidden_size)

                wq = v[:, 0:kv_groups, ...].reshape(h * kv_groups * head_dim, config.hidden_size)
                wk = v[:, -2,          ...].reshape(h * 1         * head_dim, config.hidden_size)
                wv = v[:, -1,          ...].reshape(h * 1         * head_dim, config.hidden_size)

                new_dict[name.replace('attention.wqkv.weight', 'self_attn.q_proj.weight')] = permute(wq, config.num_attention_heads)
                new_dict[name.replace('attention.wqkv.weight', 'self_attn.k_proj.weight')] = permute(wk, config.num_key_value_heads)
                new_dict[name.replace('attention.wqkv.weight', 'self_attn.v_proj.weight')] = wv
            else:
                old_name: str = ''

                mapping = {
                    'model.tok_embeddings.weight': 'model.embed_tokens.weight',
                    'model.norm.weight': 'model.norm.weight',
                    'output.weight': 'lm_head.weight',

                    'attention.wo.weight': 'self_attn.o_proj.weight',
                    'feed_forward.w1.weight': 'mlp.gate_proj.weight',
                    'feed_forward.w2.weight': 'mlp.down_proj.weight',
                    'feed_forward.w3.weight': 'mlp.up_proj.weight',
                    'attention_norm.weight': 'input_layernorm.weight',
                    'ffn_norm.weight': 'post_attention_layernorm.weight',
                }

                for k in mapping.keys():
                    if name.endswith(k):
                        old_name = name.replace(k, mapping[k])
                        break

                if old_name == '':
                    raise Exception(f'unhandled tensor {name}')

                new_dict[old_name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.bias == False, "bias must be False"
        if config.rope_scaling is not None:
            assert config.rope_scaling['type'] == 'dynamic', "rope_scaling['type'] must be dynamic"
            rope_scaling = config.rope_scaling.get("scaling_factor", 1.0)
        else:
            rope_scaling = 1.0

        rope_theta = config.rope_theta
        num_key_value_heads = config.num_key_value_heads

        if config.eos_token_id is None:
            eos_token_id = -1
        elif isinstance(config.eos_token_id, list):
            eos_token_id = config.eos_token_id[0]
        else:
            eos_token_id = config.eos_token_id

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            eos_token_id,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<f", rope_scaling))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                    f"model.layers.{i}.self_attn.q_proj.weight",
                    f"model.layers.{i}.self_attn.k_proj.weight",
                    f"model.layers.{i}.self_attn.v_proj.weight",
                    f"model.layers.{i}.self_attn.o_proj.weight",
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.down_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                    f"model.layers.{i}.input_layernorm.weight",
                    f"model.layers.{i}.post_attention_layernorm.weight",
                ]
        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]
        return weight_names

def dump_llama_like_config(f, config, ggml_type):
    assert config.hidden_act == 'silu', "hidden_act must be silu"

    config_values = [
        ggml_type.value,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_hidden_layers,
        config.intermediate_size,
        config.max_position_embeddings,
        config.bos_token_id if config.bos_token_id is not None else -1,
        config.eos_token_id if isinstance(config.eos_token_id, int) else -1,
        config.pad_token_id if config.pad_token_id is not None else -1,
        config.sep_token_id if config.sep_token_id is not None else -1,
    ]
    f.write(struct.pack("i" * len(config_values), *config_values))

class LlamaConverter(BaseConverter):
    MODEL_TYPE = ModelType.LlaMA2

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight') or name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        if (config.num_key_value_heads is not None) and (config.num_key_value_heads != config.num_attention_heads):
            print(f"warning: num_key_value_heads != num_attention_heads, make sure this is intended")
        if config.rope_scaling is not None:
            assert config.rope_scaling == 1.0, 'rope_scaling must equal to 1.0'
        if config.rope_theta is not None:
            assert config.rope_theta == 10000.0, f"rope_theta must equal to 10000.0"

        dump_llama_like_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class Llama3Converter(BaseConverter):
    MODEL_TYPE = ModelType.LlaMA3

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        else:
            return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        if config.rope_scaling is not None:
            assert config.rope_scaling == 1.0, 'rope_scaling must equal to 1.0'

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class ERNIEDenseConverter(BaseConverter):
    MODEL_TYPE = ModelType.ERNIE_DENSE

    @classmethod
    def pp(cls, config, name: str, tensor):
        return Llama3Converter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        if config.rope_scaling is not None:
            assert config.rope_scaling == 1.0, 'rope_scaling must equal to 1.0'

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            config.head_dim,
            1 if config.tie_word_embeddings else 0,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        weight_names = Llama3Converter.get_weight_names(config)
        if (config.tie_word_embeddings is not None) and config.tie_word_embeddings:
            weight_names.remove('lm_head.weight')
        return weight_names

class Llama31Converter(BaseConverter):
    MODEL_TYPE = ModelType.LlaMA31

    @classmethod
    def pp(cls, config, name: str, tensor):
        return Llama3Converter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert isinstance(config.rope_scaling, dict), 'rope_scaling must be a dict'
        assert config.rope_scaling['rope_type'] == 'llama3', 'rope_scaling.rope_type must be `llama3`'

        if isinstance(config.eos_token_id, list):
            config.eos_token_id = config.eos_token_id[0]

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        config_values = [
            config.rope_theta,
            config.rope_scaling['original_max_position_embeddings'],
            config.rope_scaling['factor'],
            config.rope_scaling['low_freq_factor'],
            config.rope_scaling['high_freq_factor'],
        ]
        f.write(struct.pack("<fifff", *config_values))

    @staticmethod
    def get_weight_names(config):
        return Llama3Converter.get_weight_names(config)

class Llama32Converter(BaseConverter):
    MODEL_TYPE = ModelType.LlaMA32

    @classmethod
    def pp(cls, config, name: str, tensor):
        return Llama31Converter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert not config.mlp_bias, '`mlp_bias` must be False'

        Llama31Converter.dump_config(f, config, ggml_type)

        config_values = [
            1 if config.tie_word_embeddings else 0,
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = Llama31Converter.get_weight_names(config)
        if (config.tie_word_embeddings is not None) and config.tie_word_embeddings:
            weight_names.remove('lm_head.weight')
        return weight_names

class AprielConverter(BaseConverter):
    MODEL_TYPE = ModelType.Apriel

    @classmethod
    def pp(cls, config, name: str, tensor):
        return Llama31Converter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert not config.mlp_bias, '`mlp_bias` must be False'
        assert isinstance(config.rope_scaling, dict), 'rope_scaling must be a dict'
        assert config.rope_scaling['rope_type'] == 'yarn', 'rope_scaling.rope_type must be `yarn`'
        assert config.rope_scaling['attention_factor'] is None

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        config_values = [
            config.rope_theta,
            config.head_dim,
            config.rope_scaling['original_max_position_embeddings'],
            config.rope_scaling['beta_fast'],
            config.rope_scaling['beta_slow'],
            config.rope_scaling['factor'],
        ]
        f.write(struct.pack("<fiifff", *config_values))

    @staticmethod
    def get_weight_names(config):
        return Llama32Converter.get_weight_names(config)

class Llama4Converter(BaseConverter):
    MODEL_TYPE = ModelType.LlaMA4

    @classmethod
    def pp(cls, config, name: str, tensor):
        return Llama3Converter.pp(config, name, tensor)

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        config = AttributeDict(config.text_config)
        new_dict = {}
        for k in state_dict:
            kk: str = k
            tensor: torch.Tensor = state_dict[kk]
            if kk.startswith('language_model.'):

                kk = kk[len('language_model.'):]
                kk = kk.replace('.feed_forward.', '.mlp.')

                if kk.endswith('mlp.experts.down_proj'):
                    shape = tensor.shape
                    for j in range(shape[0]):
                        kkk = kk.replace('mlp.experts.down_proj', f'mlp.experts.{j}.down_proj.weight')
                        new_dict[kkk] = tensor[j].T.contiguous()
                elif kk.endswith('mlp.experts.gate_up_proj'):
                    shape = tensor.shape
                    gate = tensor[:, :, : shape[2] // 2]
                    up   = tensor[:, :, shape[2] // 2 : ]
                    for j in range(shape[0]):
                        kkk = kk.replace('mlp.experts.gate_up_proj', f'mlp.experts.{j}.gate_proj.weight')
                        new_dict[kkk] = gate[j].T.contiguous()
                        kkk = kk.replace('mlp.experts.gate_up_proj', f'mlp.experts.{j}.up_proj.weight')
                        new_dict[kkk] = up[j].T.contiguous()
                elif kk.endswith('mlp.router.weight'):
                    new_dict[kk.replace('mlp.router.weight', 'mlp.gate.weight')] = tensor
                else:
                    new_dict[kk] = tensor
            else:
                new_dict[kk] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        config = AttributeDict(config.text_config)
        if config.rope_scaling is not None:
            assert isinstance(config.rope_scaling, dict), 'rope_scaling must be a dict'
            assert config.rope_scaling['rope_type'] == 'llama3', 'rope_scaling.rope_type must be `llama3`'
        assert not config.attention_bias
        assert not config.for_llm_compressor
        if config.no_rope_layers is not None:
            assert config.no_rope_layers == []

        if isinstance(config.eos_token_id, list):
            config.eos_token_id = config.eos_token_id[0]

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            config.attention_chunk_size,
            config.head_dim,
            config.interleave_moe_layer_step,
            config.intermediate_size_mlp,
            config.num_experts_per_tok,
            config.num_local_experts,
            1 if config.use_qk_norm else 0,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        config_values = [
            config.router_aux_loss_coef,
            config.rope_theta,
            config.rope_scaling['original_max_position_embeddings'] if config.rope_scaling is not None else -1,
            config.rope_scaling['factor'] if config.rope_scaling is not None else -1.0,
            config.rope_scaling['low_freq_factor'] if config.rope_scaling is not None else -1.0,
            config.rope_scaling['high_freq_factor'] if config.rope_scaling is not None else -1.0,
        ]
        f.write(struct.pack("<ffifff", *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        config = AttributeDict(config.text_config)
        moe_layers = list(range(config.interleave_moe_layer_step - 1, config.num_hidden_layers, config.interleave_moe_layer_step))
        def is_moe(layer_idx):
            nonlocal moe_layers
            return layer_idx in moe_layers

        for i in range(config.num_hidden_layers):

            if is_moe(i):
                for j in range(config.num_local_experts):
                    weight_names += [
                        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                    ]

                weight_names += [
                    f"model.layers.{i}.mlp.gate.weight",
                    f"model.layers.{i}.mlp.shared_expert.down_proj.weight",
                    f"model.layers.{i}.mlp.shared_expert.gate_proj.weight",
                    f"model.layers.{i}.mlp.shared_expert.up_proj.weight",
                ]
            else:
                weight_names += [
                    f"model.layers.{i}.mlp.down_proj.weight",
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class DeciLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.DeciLM

    @classmethod
    def pp(cls, config, name: str, tensor):

        if name.endswith('k_proj.weight'):
            r = re.findall(r'model\.layers.([0-9]*)\.', name)
            layer_id = int(r[0])
            num_key_value_heads = config.block_configs[layer_id]['attention']['n_heads_in_group']
            return permute(tensor, num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        else:
            return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        MAX_LAYERS = 100

        assert config.num_hidden_layers <= MAX_LAYERS, f'num_hidden_layers must <= {MAX_LAYERS}'
        assert not config.mlp_bias, '`mlp_bias` must be False'
        assert config.num_key_value_heads is None, 'num_key_value_heads must be null'

        config.num_key_value_heads = 0
        config.intermediate_size = 0

        Llama31Converter.dump_config(f, config, ggml_type)

        def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
            def _find_multiple(n: int, k: int) -> int:
                if n % k == 0:
                    return n
                return n + k - (n % k)

            intermediate_size = int(2 * ffn_mult * n_embd / 3)
            return _find_multiple(intermediate_size, 256)

        config_values = []
        for i in range(config.num_hidden_layers):
            cfg = config.block_configs[i]
            att_cfg = AttributeDict(cfg['attention'])
            ffn_cfg = AttributeDict(cfg['ffn'])
            assert att_cfg.num_sink_tokens is None
            assert not att_cfg.replace_with_linear
            assert att_cfg.sparsify is None
            assert not att_cfg.unshifted_sink
            assert not att_cfg.use_prefill_window_in_sink_attention
            assert att_cfg.window_length is None

            if att_cfg.no_op:
                assert att_cfg.n_heads_in_group is None

            assert not ffn_cfg.no_op
            assert not ffn_cfg.replace_with_linear
            assert ffn_cfg.sparsify is None

            config_values.append(0 if att_cfg.no_op else att_cfg.n_heads_in_group)
            config_values.append(_ffn_mult_to_intermediate_size(ffn_cfg.ffn_mult, config.hidden_size))

        for i in range(config.num_hidden_layers, MAX_LAYERS):
            config_values.append(0)
            config_values.append(0)

        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
            ]

            if not config.block_configs[i]['attention']['no_op']:
                weight_names += [
                    f"model.layers.{i}.input_layernorm.weight",
                    f"model.layers.{i}.self_attn.k_proj.weight",
                    f"model.layers.{i}.self_attn.o_proj.weight",
                    f"model.layers.{i}.self_attn.q_proj.weight",
                    f"model.layers.{i}.self_attn.v_proj.weight",
                ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class InternLM3Converter(BaseConverter):
    MODEL_TYPE = ModelType.InternLM3

    @classmethod
    def pp(cls, config, name: str, tensor):
        return Llama32Converter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.rope_scaling['rope_type'] == 'dynamic', "rope_scaling['rope_type'] != 'dynamic'"
        assert not config.qkv_bias, "qkv_bias must be False"
        assert not config.bias, "bias must be False"
        assert config.head_dim == config.hidden_size // config.num_attention_heads, 'head_dim must equal to (hidden_size / num_attention_heads)'

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            config.rope_theta,
            config.rope_scaling['factor'],
            config.max_position_embeddings
        ]
        f.write(struct.pack("<iffi", *config_values))

    @staticmethod
    def get_weight_names(config):
        return Llama32Converter.get_weight_names(config)

class ExaoneConverter(BaseConverter):
    MODEL_TYPE = ModelType.Exaone

    @staticmethod
    def upgrade_name(name: str) -> str:
        if name == 'transformer.ln_f.weight':
            name = 'model.norm.weight'
        elif name == 'transformer.wte.weight':
            name = 'model.embed_tokens.weight'
        else:
            name = name.replace('transformer.h', 'model.layers')
            name = name.replace('attn.attention.k_proj.weight', 'self_attn.k_proj.weight')
            name = name.replace('attn.attention.out_proj.weight', 'self_attn.o_proj.weight')
            name = name.replace('attn.attention.q_proj.weight', 'self_attn.q_proj.weight')
            name = name.replace('attn.attention.v_proj.weight', 'self_attn.v_proj.weight')
            name = name.replace('ln_1.weight', 'input_layernorm.weight')
            name = name.replace('ln_2.weight', 'post_attention_layernorm.weight')
            name = name.replace('mlp.c_fc_0.weight', 'mlp.gate_proj.weight')
            name = name.replace('mlp.c_fc_1.weight', 'mlp.up_proj.weight')
            name = name.replace('mlp.c_proj.weight', 'mlp.down_proj.weight')
        return name

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            new_name = ExaoneConverter.upgrade_name(name)
            new_dict[new_name] = Llama32Converter.pp(config, new_name, tensor)
        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        config.num_hidden_layers = config.num_layers
        config.hidden_act = config.activation_function

        Llama32Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return Llama32Converter.get_weight_names(config)

class TeleChat2Converter(BaseConverter):
    MODEL_TYPE = ModelType.TeleChat2

    @staticmethod
    def upgrade_name(name: str) -> str:
        if name == 'transformer.ln_f.weight':
            name = 'model.norm.weight'
        elif name == 'transformer.word_embeddings.weight':
            name = 'model.embed_tokens.weight'
        else:
            name = name.replace('transformer.h', 'model.layers')
            name = name.replace('.self_attention.', '.self_attn.')
            name = name.replace('.dense.', '.o_proj.')
            name = name.replace('.query.', '.q_proj.')
        return name

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            new_name = TeleChat2Converter.upgrade_name(name)

            if new_name.endswith('key_value.weight'):
                num_heads = config.n_head
                head_dim = config.hidden_size // num_heads
                num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else num_heads
                kv_projection_size = head_dim * num_key_value_heads

                v = tensor.view(num_heads, 2, head_dim, tensor.shape[1])

                wk = v[:, 0, ...].reshape(config.hidden_size, tensor.shape[1])
                wv = v[:, 1, ...].reshape(config.hidden_size, tensor.shape[1])

                k_name = new_name.replace('key_value.weight', 'k_proj.weight')
                v_name = new_name.replace('key_value.weight', 'v_proj.weight')
                new_dict[k_name] = Llama32Converter.pp(config, k_name, wk)
                new_dict[v_name] = Llama32Converter.pp(config, v_name, wv)
            else:
                new_dict[new_name] = Llama32Converter.pp(config, new_name, tensor)
        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act is None, 'hidden_act must be None'
        assert not config.tie_word_embeddings, 'tie_word_embeddings must be False'
        assert config.embed_layernorm is None, 'embed_layernorm must be None'
        assert not config.apply_residual_connection_post_layernorm, 'apply_residual_connection_post_layernorm must be False'
        assert config.training_seqlen == config.base_seqlen, 'training_seqlen must equal to base_seqlen'

        config.hidden_act = 'silu'

        config.num_attention_heads = config.n_head
        config.num_hidden_layers = config.n_layer
        config.intermediate_size = config.ffn_hidden_size
        config.max_position_embeddings = config.seq_length

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.base_seqlen,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.down_proj.bias",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.bias",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class SmolLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.SmolLM

    @classmethod
    def pp(cls, config, name: str, tensor):
        return Llama3Converter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        return Llama3Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        r = Llama3Converter.get_weight_names(config)
        return r[:-1]

class SmolLM3Converter(BaseConverter):
    MODEL_TYPE = ModelType.SmolLM3
    tie_word_embeddings = True

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.rope_scaling is None
        assert (config.layer_types.count('full_attention') == config.num_hidden_layers) or \
               (config.use_sliding_window is None) or (not config.use_sliding_window)
        SmolLM3Converter.tie_word_embeddings = (config.tie_word_embeddings is None) or (config.tie_word_embeddings)

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            config.no_rope_layer_interval,
            1 if SmolLM3Converter.tie_word_embeddings else 0,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        r = Llama3Converter.get_weight_names(config)
        return r[:-1] if SmolLM3Converter.tie_word_embeddings else r

class SmolVLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.SmolVLM

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        r = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.startswith('model.text_model.'):
                name = name.replace('model.text_model.', 'model.')
                r[name] = SmolLMConverter.pp(SmolVLMConverter.txt_config, name, tensor)
            elif name.startswith('model.vision_model'):
                name = name.replace('model.vision_model.', 'vision_model.')

                if 'mlp.fc1.' in name:
                    name = name.replace('.fc1.', '.fc0.')
                elif 'mlp.fc2.' in name:
                    name = name.replace('.fc2.', '.fc1.')
                elif '.out_proj.' in name:
                    name = name.replace('.out_proj.', '.o_proj.')
                elif name.startswith('vision_model.post_layernorm'):
                    name = name.replace('.post_layernorm.', '.final_layernorm.')

                r[name] = tensor
            elif name.startswith('vision_tower.'):
                r[name.replace('vision_tower.', 'vision_model.')] = tensor
            elif name == 'model.connector.modality_projection.proj.weight':
                r["multi_modal_projector.proj.weight"] = tensor
            else:
                r[name] = tensor

        return r

    @staticmethod
    def dump_config(f, config, ggml_type):
        SmolVLMConverter.txt_config = AttributeDict(config.text_config)
        if SmolVLMConverter.txt_config.bos_token_id is None:
            SmolVLMConverter.txt_config.bos_token_id = 128_000
        if SmolVLMConverter.txt_config.eos_token_id is None:
            SmolVLMConverter.txt_config.eos_token_id = 128_001
        if SmolVLMConverter.txt_config.num_attention_heads is None:
            SmolVLMConverter.txt_config.num_attention_heads = 32
        if SmolVLMConverter.txt_config.hidden_act is None:
            SmolVLMConverter.txt_config.hidden_act = 'silu'
        if SmolVLMConverter.txt_config.num_key_value_heads is None:
            SmolVLMConverter.txt_config.num_key_value_heads = SmolVLMConverter.txt_config.num_attention_heads
        if SmolVLMConverter.txt_config.tie_word_embeddings is None:
            SmolVLMConverter.txt_config.tie_word_embeddings = False

        assert not SmolVLMConverter.txt_config.tie_word_embeddings
        assert not SmolVLMConverter.txt_config.qk_layer_norms
        assert not SmolVLMConverter.txt_config.use_resampler
        SmolLMConverter.dump_config(f, SmolVLMConverter.txt_config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        weight_names = Llama3Converter.get_weight_names(SmolVLMConverter.txt_config)

        for i in range(config.vision_config['num_hidden_layers']):
            weight_names += [
                f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias",
                f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight",
                f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias",
                f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight",
                f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias",
                f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight",
                f"vision_model.encoder.layers.{i}.self_attn.o_proj.bias",
                f"vision_model.encoder.layers.{i}.self_attn.o_proj.weight",
                f"vision_model.encoder.layers.{i}.mlp.fc0.bias",
                f"vision_model.encoder.layers.{i}.mlp.fc0.weight",
                f"vision_model.encoder.layers.{i}.mlp.fc1.bias",
                f"vision_model.encoder.layers.{i}.mlp.fc1.weight",
                f"vision_model.encoder.layers.{i}.layer_norm1.bias",
                f"vision_model.encoder.layers.{i}.layer_norm1.weight",
                f"vision_model.encoder.layers.{i}.layer_norm2.bias",
                f"vision_model.encoder.layers.{i}.layer_norm2.weight",
            ]

        weight_names += [
            "multi_modal_projector.proj.weight",
            "vision_model.final_layernorm.bias",
            "vision_model.final_layernorm.weight",
            "vision_model.embeddings.position_embedding.weight",
            "vision_model.embeddings.patch_embedding.bias",
            "vision_model.embeddings.patch_embedding.weight",
        ]

        return weight_names

class LlamaMultiConverter(BaseConverter):
    MODEL_TYPE = ModelType.LlaMAMulti

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            mapping = {
                'tok_embeddings.weight': 'embed_tokens.weight',
                "attention_norm.weight": 'input_layernorm.weight',
                "feed_forward.w1.weight": 'mlp.gate_proj.weight',
                "feed_forward.w2.weight": 'mlp.down_proj.weight',
                "feed_forward.w3.weight": 'mlp.up_proj.weight',
                "ffn_norm.weight": 'post_attention_layernorm.weight',
                "attention.wk.weight": 'self_attn.k_proj.weight',
                "attention.wo.weight": 'self_attn.o_proj.weight',
                "attention.wq.weight": 'self_attn.q_proj.weight',
                "attention.wv.weight": 'self_attn.v_proj.weight',
                "model.output.weight": 'lm_head.weight',
            }

            new_name = 'model.' + name

            for k in mapping.keys():
                if new_name.endswith(k):
                    new_name = new_name.replace(k, mapping[k])
                    break

            new_dict[new_name] = tensor #Llama3Converter.pp(config, new_name, tensor)

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):

        hidden_dim = 4 * config.dim
        intermediate_size = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            intermediate_size = int(config.ffn_dim_multiplier * intermediate_size)
        intermediate_size = config.multiple_of * ((intermediate_size + config.multiple_of - 1) // config.multiple_of)

        config.hidden_size = config.dim
        config.num_attention_heads = config.n_heads
        config.num_hidden_layers = config.n_layers - config.n_future_tokens + 1
        config.intermediate_size = intermediate_size
        config.max_position_embeddings = 4096
        config.num_key_value_heads = config.n_kv_heads
        config.bos_token_id = 1
        config.eos_token_id = 2

        config.hidden_act = 'silu'

        Llama3Converter.dump_config(f, config, ggml_type)
        config_values = [
            config.n_future_tokens,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = Llama3Converter.get_weight_names(config)

        for i in range(config.n_future_tokens - 1):
            weight_names += [
                f"model.extra_heads.{i}.input_layernorm.weight",
                f"model.extra_heads.{i}.mlp.down_proj.weight",
                f"model.extra_heads.{i}.mlp.gate_proj.weight",
                f"model.extra_heads.{i}.mlp.up_proj.weight",
                f"model.extra_heads.{i}.post_attention_layernorm.weight",
                f"model.extra_heads.{i}.self_attn.k_proj.weight",
                f"model.extra_heads.{i}.self_attn.o_proj.weight",
                f"model.extra_heads.{i}.self_attn.q_proj.weight",
                f"model.extra_heads.{i}.self_attn.v_proj.weight",
            ]

        return weight_names

class CodeLlamaConverter(BaseConverter):
    MODEL_TYPE = ModelType.CodeLlaMA

    @classmethod
    def pp(cls, config, name: str, tensor):
        return LlamaConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.rope_theta > 0, "rope_theta must be positive"
        assert config.rope_scaling is None, "rope_scaling must be `null`"
        LlamaConverter.dump_config(f, config, ggml_type)
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class WizardCoderConverter(BaseConverter):
    MODEL_TYPE = ModelType.WizardCoder

    @classmethod
    def pp(cls, config, name: str, tensor):
        return CodeLlamaConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        CodeLlamaConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return CodeLlamaConverter.get_weight_names(config)

class WizardLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.WizardLM

    @classmethod
    def pp(cls, config, name: str, tensor):
        return LlamaConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        LlamaConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class OrionConverter(BaseConverter):
    MODEL_TYPE = ModelType.Orion

    @classmethod
    def pp(cls, config, name: str, tensor):
        return LlamaConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        scaling = config.rope_scaling if config.rope_scaling is not None else 1.0
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_sequence_length if config.max_sequence_length is not None else config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", scaling))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.input_layernorm.bias",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.bias",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "model.norm.bias",
            "lm_head.weight"
        ]

        return weight_names

class MiniCPMConverter(BaseConverter):
    MODEL_TYPE = ModelType.MiniCPM

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name == 'model.embed_tokens.weight':
            return tensor * config.scale_emb
        elif name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        else:
            return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        _attn = config._attn_implementation if config._attn_implementation is not None else 'eager'
        assert (_attn == 'eager') or (_attn == 'flash_attention_2'), '_attn_implementation must be eager or flash_attention_2'

        assert config.hidden_act == 'silu', "hidden_act must be silu"

        if config.rope_scaling is not None:
            assert config.rope_scaling['type'] == 'dynamic', "rope_scaling['type'] != 'dynamic'"
            scaling = 1.0
            print("WARNING: we are not going to support NTK dynamic scaling for this model.")
        else:
            scaling = 1.0

        rope_theta = config.rope_theta if config.rope_theta is not None else 10000

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", scaling))
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<f", config.scale_depth / math.sqrt(config.num_hidden_layers)))

    @staticmethod
    def get_weight_names(config):
        r = LlamaConverter.get_weight_names(config)
        if (config.tie_word_embeddings is None) or config.tie_word_embeddings:
            r.remove('lm_head.weight')
        return r

class MiniCPM4Converter(BaseConverter):
    MODEL_TYPE = ModelType.MiniCPM4

    @classmethod
    def pp(cls, config, name: str, tensor):
        return MiniCPMConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        MAX_FACTOR_LEN = 128

        assert config.hidden_act == 'silu', "hidden_act must be silu"
        if config.tie_word_embeddings is None:
            config.tie_word_embeddings = True
        if config.rope_scaling is not None:
            assert config.rope_scaling['rope_type'] == 'longrope'
            factor_len = len(config.rope_scaling['long_factor'])
            assert factor_len <= MAX_FACTOR_LEN, "config.rope_scaling['long_factor']) must <= MAX_FACTOR_LEN"
            factors = pad_to(config.rope_scaling['short_factor'], MAX_FACTOR_LEN) + pad_to(config.rope_scaling['long_factor'], MAX_FACTOR_LEN)

            if config.max_position_embeddings == 32768:
                print("`longrope` is configured, extend to 32k * 4.")
                config.max_position_embeddings = 32768 * 4
        else:
            factor_len = 0
            factors = pad_to([0.0], MAX_FACTOR_LEN * 2)

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id,
            config.eos_token_id[0],
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.rope_scaling['original_max_position_embeddings'],
            1 if config.tie_word_embeddings else 0,
            factor_len,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        float_values = [
            config.mup_denominator if config.mup_denominator is not None else 0.0,
            config.dim_model_base / config.hidden_size,
            config.rope_theta if config.mup_denominator is not None else 10000.0,
            config.scale_depth / math.sqrt(config.num_hidden_layers),
        ] + factors
        f.write(struct.pack("<" + "f" * len(float_values), *float_values))

    @staticmethod
    def get_weight_names(config):
        r = LlamaConverter.get_weight_names(config)
        if config.tie_word_embeddings:
            r.remove('lm_head.weight')
        return r

class MiniCPMEmbConverter(BaseConverter):
    MODEL_TYPE = ModelType.MiniCPM_Embedding_Light

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            new_name = name
            if ('model.' + name) in MiniCPMEmbConverter.weight_names:
                new_name = 'model.' + name

            if new_name == 'model.embed_tokens.weight':
                tensor = tensor * config.scale_emb

            new_dict[new_name] = tensor
        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.rope_scaling['type'] == 'longrope', "rope_scaling['type'] must be 'longrope'"
        assert len(config.rope_scaling['long_factor']) == 32, f"length of rope_scaling['factor'] {len(config.rope_scaling['long_factor'])} must be 32"

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.rope_scaling['original_max_position_embeddings'],
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta,
            config.scale_depth / math.sqrt(config.num_hidden_layers)
        ]  + config.rope_scaling['short_factor'] + config.rope_scaling['long_factor']
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = MiniCPMConverter.get_weight_names(config)
        weight_names += [
            'head.weight'
        ]
        MiniCPMEmbConverter.weight_names = weight_names
        return weight_names

class MiniCPMReRankerConverter(BaseConverter):
    MODEL_TYPE = ModelType.MiniCPM_ReRanker_Light

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        return MiniCPMEmbConverter.state_dict_pp(config, state_dict)

    @staticmethod
    def dump_config(f, config, ggml_type):
        MiniCPMEmbConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        weight_names = MiniCPMConverter.get_weight_names(config)
        weight_names += [
            'score.weight'
        ]
        MiniCPMEmbConverter.weight_names = weight_names
        return weight_names

class MiniCPMMoEConverter(BaseConverter):
    MODEL_TYPE = ModelType.MiniCPM_MoE

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name == 'model.embed_tokens.weight':
            return tensor * config.scale_emb
        elif name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        else:
            return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        _attn = config._attn_implementation if config._attn_implementation is not None else 'eager'
        assert (_attn == 'eager') or (_attn == 'flash_attention_2'), '_attn_implementation must be eager or flash_attention_2'

        assert config.hidden_act == 'silu', "hidden_act must be silu"

        assert config.rope_scaling is None, "rope_scaling must be null"
        scaling = 1.0
        rope_theta = config.rope_theta if config.rope_theta is not None else 10000

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
            config.num_experts,
            config.num_experts_per_tok,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", scaling))
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<f", config.scale_depth / math.sqrt(config.num_hidden_layers)))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):

            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
            ]

            for j in range(config.num_experts):
                weight_names += [
                    f"model.layers.{i}.mlp.experts.{j}.w1.weight",
                    f"model.layers.{i}.mlp.experts.{j}.w2.weight",
                    f"model.layers.{i}.mlp.experts.{j}.w3.weight",
                ]

            weight_names += [
                f"model.layers.{i}.mlp.gate.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        return weight_names

class MiniCPM3Converter(BaseConverter):
    MODEL_TYPE = ModelType.MiniCPM3

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name == 'model.embed_tokens.weight':
                new_dict[name] = tensor * config.scale_emb
            elif name.endswith('kv_a_proj_with_mqa.weight'):

                w_d_kv, w_k_pe = torch.split(
                    tensor, [config.kv_lora_rank, config.qk_rope_head_dim])

                new_dict[name.replace('kv_a_proj_with_mqa.weight', 'd_kv_proj.weight')] = w_d_kv
                new_dict[name.replace('kv_a_proj_with_mqa.weight', 'k_pe_proj.weight')] = w_k_pe
            elif name.endswith('kv_a_layernorm.weight'):
                new_dict[name.replace('kv_a_layernorm.weight', 'kv_norm.weight')] = tensor
            elif name.endswith('kv_b_proj.weight'):

                v = tensor.reshape(config.num_attention_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank)
                u_k_nope_proj = v[:, :config.qk_nope_head_dim, :].reshape(config.num_attention_heads * config.qk_nope_head_dim, config.kv_lora_rank)
                u_v_proj = v[:, config.qk_nope_head_dim:, :].reshape(config.num_attention_heads * config.v_head_dim, config.kv_lora_rank)

                new_dict[name.replace('kv_b_proj.weight', 'u_k_nope_proj.weight')] = u_k_nope_proj
                new_dict[name.replace('kv_b_proj.weight', 'u_v_proj.weight')] = u_v_proj
            elif name.endswith('q_proj.weight'):
                new_dict[name] = tensor
            elif name.endswith('q_a_proj.weight'):
                new_dict[name.replace('q_a_proj.weight', 'd_q_proj.weight')] = tensor
            elif name.endswith('q_a_layernorm.weight'):
                new_dict[name.replace('q_a_layernorm.weight', 'q_norm.weight')] = tensor
            elif name.endswith('q_b_proj.weight'):
                new_dict[name.replace('q_b_proj.weight', 'u_q_proj.weight')] = tensor
            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.rope_scaling['type'] == 'longrope', "rope_scaling['type'] must be 'longrope'"
        assert len(config.rope_scaling['long_factor']) == 16, f"length of rope_scaling['factor'] {len(config.rope_scaling['long_factor'])} must be 16"

        config.v_head_dim = config.hidden_size // config.num_attention_heads

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id[0],
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
            config.kv_lora_rank,
            config.q_lora_rank,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
            config.rope_scaling['original_max_position_embeddings'],
            config.v_head_dim,
            config.dim_model_base,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        config_values = [
            config.scale_depth / math.sqrt(config.num_hidden_layers),
        ] + config.rope_scaling['short_factor'] + config.rope_scaling['long_factor']

        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight",
                        "model.norm.weight"
                        ]
        for i in range(config.num_hidden_layers):

            weight_names += [
                f"model.layers.{i}.self_attn.d_q_proj.weight",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.u_q_proj.weight",

                f"model.layers.{i}.self_attn.d_kv_proj.weight",
                f"model.layers.{i}.self_attn.k_pe_proj.weight",
                f"model.layers.{i}.self_attn.kv_norm.weight",
                f"model.layers.{i}.self_attn.u_k_nope_proj.weight",
                f"model.layers.{i}.self_attn.u_v_proj.weight",

                f"model.layers.{i}.self_attn.o_proj.weight",

                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight",

                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
            ]

        return weight_names

class MistralConverter(BaseConverter):
    MODEL_TYPE = ModelType.Mistral

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.sliding_window if config.sliding_window is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class Mistral2Converter(BaseConverter):
    MODEL_TYPE = ModelType.Mistral2

    @classmethod
    def pp(cls, config, name: str, tensor):
        return MistralConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert g_tokenizer_type == TokenizerType.BPE2, 'BPE2 must be used'
        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.head_dim if config.head_dim is not None else (config.hidden_size // config.num_attention_heads),
            config.sliding_window if config.sliding_window is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        return MistralConverter.get_weight_names(config)

class MistralSmall31Converter(BaseConverter):
    MODEL_TYPE = ModelType.Mistral2

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        d = {}
        for name in state_dict:
            if name.startswith('vision_encoder.'): continue

            t = state_dict[name]
            if name == 'norm.weight':
                new_name = 'model.norm.weight'
            elif name == 'output.weight':
                new_name = 'lm_head.weight'
            elif name == 'tok_embeddings.weight':
                new_name = 'model.embed_tokens.weight'
            elif name.startswith('layers.'):
                new_name = 'model.' + name
                mapping = {
                    'attention.wk.weight': 'self_attn.k_proj.weight',
                    'attention.wo.weight': 'self_attn.o_proj.weight',
                    'attention.wq.weight': 'self_attn.q_proj.weight',
                    'attention.wv.weight': 'self_attn.v_proj.weight',
                    'attention_norm.weight': 'input_layernorm.weight',
                    'feed_forward.w1.weight': 'mlp.gate_proj.weight',
                    'feed_forward.w2.weight': 'mlp.down_proj.weight',
                    'feed_forward.w3.weight': 'mlp.up_proj.weight',
                    'ffn_norm.weight': 'post_attention_layernorm.weight',
                }
                for x in mapping:
                    if name.endswith(x):
                        new_name = new_name.replace(x, mapping[x])
                        break
            else:
                print(f'ignore tensor: {name}')
                continue
            d[new_name] = t # Mistral2Converter.pp(config, new_name, t)
        return d

    @staticmethod
    def dump_config(f, config, ggml_type):
        Mistral2Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return Mistral2Converter.get_weight_names(config)

def make_experts_id_map(experts):
    r = {}
    for i in range(len(experts)):
        r[experts[i]] = i
    return r

class MixtralConverter(BaseConverter):
    MODEL_TYPE = ModelType.Mixtral

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        experts_id_map = make_experts_id_map(config.experts)
        for name in state_dict:
            tensor = state_dict[name]
            if name.endswith('k_proj.weight'):
                new_dict[name] = permute(tensor, config.num_key_value_heads)
            elif name.endswith('q_proj.weight'):
                new_dict[name] = permute(tensor, config.num_attention_heads)
            elif 'block_sparse_moe.experts' in name:
                id = int(re.findall("block_sparse_moe\\.experts\\.([0-9]+)\\.", name)[0])
                if id in experts_id_map:
                    new_name = name.replace(f"block_sparse_moe.experts.{id}.", f"block_sparse_moe.experts.{experts_id_map[id]}.")
                    new_dict[new_name] = tensor
            elif name.endswith('block_sparse_moe.gate.weight'):
                indices = torch.tensor(config.experts)
                new_dict[name] = torch.index_select(tensor, 0, indices)
            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.output_router_logits == False, "output_router_logits must be False"
        assert len(config.experts) >= config.num_experts_per_tok, f"must select at least {config.num_experts_per_tok} experts"

        MistralConverter.dump_config(f, config, ggml_type)

        config_values = [
            config.num_experts_per_tok,
            config.num_local_experts,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        experts = range(len(config.experts))
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            for j in experts:
                weight_names += [
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight",
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight",
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight",
                ]

            weight_names += [
                f"model.layers.{i}.block_sparse_moe.gate.weight",
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class GraniteMoEConverter(BaseConverter):
    MODEL_TYPE = ModelType.GraniteMoE

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor = state_dict[name]
            if name == 'model.embed_tokens.weight':
                new_dict[name] = tensor * config.embedding_multiplier
            elif name.endswith('k_proj.weight'):
                new_dict[name] = permute(tensor, config.num_key_value_heads)
            elif name.endswith('q_proj.weight'):
                new_dict[name] = permute(tensor, config.num_attention_heads)
            elif name.endswith('block_sparse_moe.input_linear.weight'):
                parts = [t[0] for t in torch.split(tensor, 1)]
                for j in range(len(parts)):
                    dim = parts[j].shape[0] // 2
                    newname = name.replace("block_sparse_moe.input_linear.weight", f"block_sparse_moe.experts.{j}.gate_proj.weight")
                    new_dict[newname] = parts[j][:dim]
                    newname = name.replace("block_sparse_moe.input_linear.weight", f"block_sparse_moe.experts.{j}.up_proj.weight")
                    new_dict[newname] = parts[j][dim:]
            elif name.endswith('block_sparse_moe.output_linear.weight'):
                parts = [t[0] for t in torch.split(tensor, 1)]
                for j in range(len(parts)):
                    newname = name.replace("block_sparse_moe.output_linear.weight", f"block_sparse_moe.experts.{j}.down_proj.weight")
                    new_dict[newname] = parts[j]
            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.output_router_logits == False, "output_router_logits must be False"
        assert config.attention_bias == False, 'attention_bias must be False'
        assert config.rope_scaling is None, 'rope_scaling must be `null`'

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            1 if config.tie_word_embeddings else 0,
            config.num_experts_per_tok,
            config.num_local_experts,
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

        config_values = [
            config.attention_multiplier,
            config.logits_scaling,
            config.residual_multiplier,
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            for j in range(config.num_local_experts):
                weight_names += [
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.down_proj.weight",
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.gate_proj.weight",
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.up_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.block_sparse_moe.router.layer.weight",
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        if not config.tie_word_embeddings:
            weight_names.append("lm_head.weight")

        return weight_names

class GraniteConverter(BaseConverter):
    MODEL_TYPE = ModelType.Granite

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name == 'model.embed_tokens.weight':
            return tensor * config.embedding_multiplier
        elif name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        else:
            return tensor

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor = state_dict[name]
            if name == 'model.embed_tokens.weight':
                new_dict[name] = tensor * config.embedding_multiplier
            elif name.endswith('k_proj.weight'):
                new_dict[name] = permute(tensor, config.num_key_value_heads)
            elif name.endswith('q_proj.weight'):
                new_dict[name] = permute(tensor, config.num_attention_heads)
            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.attention_bias == False, 'attention_bias must be False'
        assert config.rope_scaling is None, 'rope_scaling must be `null`'

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            1 if config.tie_word_embeddings else 0,
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

        config_values = [
            config.attention_multiplier,
            config.logits_scaling,
            config.residual_multiplier,
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        if not config.tie_word_embeddings:
            weight_names.append("lm_head.weight")

        return weight_names

class OLMoEConverter(BaseConverter):
    MODEL_TYPE = ModelType.OLMoE

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.output_router_logits == False, "output_router_logits must be False"

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.num_experts_per_tok,
            config.num_experts,
            1 if config.norm_topk_prob else 0,
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            for j in range(config.num_experts):
                weight_names += [
                    f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                    f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                    f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.mlp.gate.weight",

                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class OLMo2Converter(BaseConverter):
    MODEL_TYPE = ModelType.OLMo2

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.rope_scaling is None, "rope_scaling must be null"
        assert not config.tie_word_embeddings, "tie_word_embeddings must be False"
        assert not config.attention_bias, "attention_bias must be False"

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_feedforward_layernorm.weight",
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class Exaone4Converter(BaseConverter):
    MODEL_TYPE = ModelType.Exaone4

    @staticmethod
    def dump_config(f, config, ggml_type):
        MAX_LAYERS = 128
        assert config.num_hidden_layers < MAX_LAYERS
        assert config.rope_scaling['rope_type'] == 'llama3'
        assert not config.attention_bias, "attention_bias must be False"
        assert config.head_dim == config.hidden_size // config.num_attention_heads

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.sliding_window if config.sliding_window is not None else -1,
            1 if config.tie_word_embeddings else 0,
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta,
            config.rope_scaling['original_max_position_embeddings'],
            config.rope_scaling['factor'],
            config.rope_scaling['low_freq_factor'],
            config.rope_scaling['high_freq_factor'],
        ]
        f.write(struct.pack("<fifff", *config_values))

        def check_is_sliding(config, layer_idx):
            if config.sliding_window is None:
                return False
            if config.layer_types is not None:
                return config.layer_types[layer_idx] == "sliding_attention"
            if isinstance(config.sliding_window_pattern, int):
                return ((layer_idx + 1) % config.sliding_window_pattern) != 0
            elif isinstance(config.sliding_window_pattern, str):
                assert isinstance(config.sliding_window, int), (
                    f"Sliding window must be positive integer, but got {config.sliding_window}"
                )
                return (
                    layer_idx != config.num_hidden_layers - 1
                    and config.sliding_window_pattern[layer_idx % len(config.sliding_window_pattern)] == "L"
                )
            else:
                pass
            return False

        config_values = [0] * MAX_LAYERS
        for i in range(config.num_hidden_layers):
            if check_is_sliding(config, i):
                config_values[i] = 1
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = OLMo2Converter.get_weight_names(config)
        if config.tie_word_embeddings:
            weight_names = weight_names[:-1]

        return weight_names

class InstellaConverter(BaseConverter):
    MODEL_TYPE = ModelType.Instella

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.rope_scaling is None, "rope_scaling must be null"
        assert not config.tie_word_embeddings, "tie_word_embeddings must be False"
        assert not config.attention_bias, "attention_bias must be False"

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.pre_attention_layernorm.weight",
                f"model.layers.{i}.pre_feedforward_layernorm.weight",
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class WizardMathConverter(BaseConverter):
    MODEL_TYPE = ModelType.WizardMath

    @classmethod
    def pp(cls, config, name: str, tensor):
        return MistralConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        MistralConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return MistralConverter.get_weight_names(config)

def part(weights: torch.Tensor, n_part: int, part_no: int = 3) -> torch.Tensor:
    r = weights.shape[0] // part_no
    return weights[r * n_part : r * n_part + r, ...]

class BaiChuanConverter(BaseConverter):
    MODEL_TYPE = ModelType.BaiChuan

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        added = {}
        removed = []
        for name in state_dict:
            if name == 'lm_head.weight':
                tensor: torch.Tensor = state_dict[name]
                state_dict[name] = nn.functional.normalize(tensor)
                continue

            if not name.endswith('W_pack.weight'):
                continue

            tensor: torch.Tensor = state_dict[name]
            removed.append(name)

            wq = permute(part(tensor, 0), config.num_attention_heads)
            wk = permute(part(tensor, 1), config.num_attention_heads)
            wv = part(tensor, 2)

            added[name.replace('W_pack', 'q_proj')] = wq
            added[name.replace('W_pack', 'k_proj')] = wk
            added[name.replace('W_pack', 'v_proj')] = wv

        for s in removed:
            del state_dict[s]
        for name in added:
            state_dict[name] = added[name]
        return state_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        LlamaConverter.dump_config(f, config, ggml_type)
        config_values = [
            config.user_token_id,
            config.assistant_token_id,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class BaiChuanLlamaConverter(BaiChuanConverter):
    MODEL_TYPE = ModelType.BaiChuanLlama

class BaiChuanM1Converter(BaseConverter):
    MODEL_TYPE = ModelType.BaiChuanM1

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        def split_w(name, tensor, hidden_size, num_attention_heads, num_key_value_heads):
            nonlocal new_dict

            head_dim = hidden_size // num_attention_heads
            k_size = head_dim * num_key_value_heads
            wq = tensor[0:hidden_size, :]
            wk = tensor[hidden_size:hidden_size + k_size, :]
            wv = tensor[hidden_size + k_size:, :]

            new_dict[name.replace('W_pack', 'q_proj')] = wq
            new_dict[name.replace('W_pack', 'k_proj')] = wk
            new_dict[name.replace('W_pack', 'v_proj')] = wv

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            if name == 'lm_head.weight':
                new_dict[name] = nn.functional.normalize(tensor)
            elif name.endswith('W_pack.weight'):
                layer_id = int(re.findall('layers.([0-9]*).self_attn', name)[0])
                if layer_id in config.sliding_window_layers:
                    split_w(name, tensor, config.hidden_size, config.num_swa_attention_heads, config.num_swa_key_value_heads)
                else:
                    split_w(name, tensor, config.hidden_size, config.num_attention_heads, config.num_key_value_heads)
            elif name.endswith('.conv_k') or name.endswith('.conv_v'):
                s = tensor.shape
                new_dict[name] = tensor.view(s[2], s[4])
            else:
                new_dict[name] = tensor
        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.conv_window == 2, 'conv_window must be 2'
        assert config.model_max_length == config.max_position_embeddings, 'max_position_embeddings must == model_max_length'

        dump_llama_like_config(f, config, ggml_type)

        sliding_window_pattern = config.sliding_window_layers[1] - config.sliding_window_layers[0]

        sw_layers = list(range(1, config.num_hidden_layers, sliding_window_pattern))
        assert sw_layers == config.sliding_window_layers, 'unsupported sliding_window_layers'

        config_values = [
            config.num_key_value_heads,
            config.conv_window,
            config.num_swa_attention_heads,
            config.num_swa_key_value_heads,
            config.sliding_window,
            sliding_window_pattern
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta
        ]
        f.write(struct.pack("f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.conv_k",
                f"model.layers.{i}.self_attn.conv_v",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class BlueLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.BlueLM

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        added = {}
        removed = []
        for name in state_dict.keys():
            if name == 'model.embed_tokens.weight':

                ln = nn.LayerNorm(config.hidden_size)
                ln.weight.data = state_dict['model.embed_layer_norm.weight']
                ln.bias.data   = state_dict['model.embed_layer_norm.bias']

                state_dict[name] = ln.forward(state_dict[name])

                removed.append('model.embed_layer_norm.bias')
                removed.append('model.embed_layer_norm.weight')
            elif name.endswith('k_proj.weight'):
                state_dict[name] =  permute(state_dict[name], config.num_key_value_heads)
            elif name.endswith('q_proj.weight'):
                state_dict[name] =  permute(state_dict[name], config.num_attention_heads)

        for s in removed:
            del state_dict[s]
        for name in added:
            state_dict[name] = added[name]
        return state_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.use_stable_embedding, "must use_stable_embedding"

        if config.rope_scaling is not None:
            assert config.rope_scaling['type'] == 'ntkmixed', "hidden_act['type'] must be 'ntkmixed'"
            rope_scaling_factor = config.rope_scaling['factor']
            rope_scaling_power  = config.rope_scaling['power']
        else:
            rope_scaling_factor = 1.0
            rope_scaling_power = 0.0
            if config.vocab_size == 100096:
                print(f"Warning: fixing vocab_size for Chat-7B")
                config.vocab_size = 100008

        max_len = int(rope_scaling_factor * config.max_position_embeddings)

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            max_len,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", config.rope_theta))
        f.write(struct.pack("<f", rope_scaling_factor))
        f.write(struct.pack("<f", rope_scaling_power))


    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class YiConverter(BaseConverter):
    MODEL_TYPE = ModelType.Yi

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.rope_theta > 0, "rope_theta must be positive"
        scaling = config.rope_scaling if config.rope_scaling is not None else 1.0

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", scaling))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class TigerBotConverter(BaseConverter):
    MODEL_TYPE = ModelType.TigerBot

    @classmethod
    def pp(cls, config, name: str, tensor):
        return LlamaConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        scaling = config.rope_scaling['factor'] if config.rope_scaling is not None else 1.0
        rope_theta = config.rope_theta if config.rope_theta is not None else 10000.0

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", scaling))
        f.write(struct.pack("<f", rope_theta))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class DeepSeekCoderConverter(BaseConverter):
    MODEL_TYPE = ModelType.DeepSeekCoder

    @classmethod
    def pp(cls, config, name: str, tensor):
        return LlamaConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.rope_scaling['type'] == "linear", "rope_scaling.type must be linear"
        assert config.rope_theta > 0, "rope_theta must be positive"
        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else g_special_tokens['<pad>'],
            config.sep_token_id if config.sep_token_id is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_scaling['factor']))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class CodeFuseDeepSeekCoderConverter(BaseConverter):
    MODEL_TYPE = ModelType.CodeFuseDeepSeek

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        elif name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.rope_scaling['type'] == "linear", "rope_scaling.type must be linear"
        assert config.rope_theta > 0, "rope_theta must be positive"
        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else g_special_tokens['<pad>'],
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_scaling['factor']))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class DeepSeekConverter(BaseConverter):
    MODEL_TYPE = ModelType.DeepSeek

    @classmethod
    def pp(cls, config, name: str, tensor):
        return LlamaConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        LlamaConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class CohereCommandConverter(BaseConverter):
    MODEL_TYPE = ModelType.CohereCommand

    @classmethod
    def pp(cls, config, name: str, tensor):
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", config.rope_theta))
        f.write(struct.pack("<f", config.logit_scale))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        return weight_names

class Cohere2CommandConverter(BaseConverter):
    MODEL_TYPE = ModelType.CohereCommandR7B

    @classmethod
    def pp(cls, config, name: str, tensor):
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.use_embedding_sharing, 'use_embedding_sharing must be True'
        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.sliding_window,
            config.sliding_window_pattern,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", config.rope_theta))
        f.write(struct.pack("<f", config.logit_scale))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        return weight_names

class StableLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.StableLM

    @classmethod
    def pp(cls, config, name: str, tensor):
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.rope_theta > 0, "rope_theta must be positive"

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            int(config.rope_pct * head_dim),
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", config.rope_theta))
        f.write(struct.pack("<f", config.rotary_scaling_factor))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.input_layernorm.bias",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.bias",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "model.norm.bias",
            "lm_head.weight"
        ]

        return weight_names

class ChatGLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.position_encoding_2d, "unimplemented: position_encoding_2d should be True"
        assert (
            config.inner_hidden_size == 4 * config.hidden_size
        ), "unimplemented: inner_hidden_size should be 4 times hidden_size"
        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_layers,
            config.inner_hidden_size,
            config.max_sequence_length,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):

        weight_names = ["transformer.word_embeddings.weight"]
        for i in range(config.num_layers):
            weight_names += [
                f"transformer.layers.{i}.input_layernorm.weight",
                f"transformer.layers.{i}.input_layernorm.bias",
                f"transformer.layers.{i}.attention.query_key_value.weight",
                f"transformer.layers.{i}.attention.query_key_value.bias",
                f"transformer.layers.{i}.attention.dense.weight",
                f"transformer.layers.{i}.attention.dense.bias",
                f"transformer.layers.{i}.post_attention_layernorm.weight",
                f"transformer.layers.{i}.post_attention_layernorm.bias",
                f"transformer.layers.{i}.mlp.dense_h_to_4h.weight",
                f"transformer.layers.{i}.mlp.dense_h_to_4h.bias",
                f"transformer.layers.{i}.mlp.dense_4h_to_h.weight",
                f"transformer.layers.{i}.mlp.dense_4h_to_h.bias",
            ]
        weight_names += [
            "transformer.final_layernorm.weight",
            "transformer.final_layernorm.bias",
        ]

        return weight_names


class ChatGLM2Converter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM2

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.add_bias_linear is False, "unimplemented: add_bias_linear must be false"
        assert config.add_qkv_bias is True, "unimplemented: add_qkv_bias must be true"
        assert (
            config.apply_residual_connection_post_layernorm is False
        ), "unimplemented: apply_residual_connection_post_layernorm must be false"
        assert (
            config.kv_channels * config.num_attention_heads == config.hidden_size
        ), "unimplemented: invalid kv_channels"
        assert config.multi_query_attention is True, "unimplemented: multi_query_attention must be true"
        assert config.original_rope is True, "unimplemented: original_rope must be true"
        assert config.post_layer_norm is True, "unimplemented: post_layer_norm must be true"
        assert config.rmsnorm is True, "unimplemented: rmsnorm must be true"

        config_values = [
            ggml_type.value,
            config.padded_vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_layers,
            config.ffn_hidden_size,
            config.seq_length,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.multi_query_group_num,
        ]

        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["transformer.embedding.word_embeddings.weight"]
        for i in range(config.num_layers):
            weight_names += [
                f"transformer.encoder.layers.{i}.input_layernorm.weight",
                f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight",
                f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias",
                f"transformer.encoder.layers.{i}.self_attention.dense.weight",
                f"transformer.encoder.layers.{i}.post_attention_layernorm.weight",
                f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight",
                f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight",
            ]
        weight_names += [
            "transformer.encoder.final_layernorm.weight",
            "transformer.output_layer.weight",
        ]

        return weight_names

class ChatGLM4Converter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM4

    @staticmethod
    def dump_config(f, config, ggml_type):
        config.eos_token_id = config.eos_token_id[0]

        ChatGLM2Converter.dump_config(f, config, ggml_type)

        config_values = [
            config.rope_ratio
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        return ChatGLM2Converter.get_weight_names(config)

class CharacterGLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.CharacterGLM

    @staticmethod
    def dump_config(f, config, ggml_type):
        ChatGLM2Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return ChatGLM2Converter.get_weight_names(config)

class GLM4Converter(BaseConverter):
    MODEL_TYPE = ModelType.GLM4

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        r = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.endswith('mlp.gate_up_proj.weight'):
                r[name.replace('gate_up_proj.weight', 'gate_proj.weight')] = part(tensor, 0, 2).contiguous()
                r[name.replace('gate_up_proj.weight', 'up_proj.weight')]   = part(tensor, 1, 2).contiguous()
            else:
                r[name] = tensor

        return r

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.head_dim == config.hidden_size // config.num_attention_heads
        if isinstance(config.eos_token_id, list):
            config.eos_token_id = config.eos_token_id[0]

        rope_dim = int(config.partial_rotary_factor * config.head_dim)

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            1 if config.attention_bias else 0,
            rope_dim,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        config_values = [
            config.rope_theta,
        ]
        f.write(struct.pack("<f", *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_mlp_layernorm.weight",
                f"model.layers.{i}.post_self_attn_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
            ]

            if config.attention_bias:
                weight_names += [
                    f"model.layers.{i}.self_attn.v_proj.bias",
                    f"model.layers.{i}.self_attn.k_proj.bias",
                    f"model.layers.{i}.self_attn.q_proj.bias",
                ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class Phi2Converter(BaseConverter):
    MODEL_TYPE = ModelType.Phi2

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        added = {}
        removed = []
        for name in state_dict:

            if name.find('.Wqkv.') < 0:
                continue

            tensor: torch.Tensor = state_dict[name]
            removed.append(name)

            wq = part(tensor, 0)
            wk = part(tensor, 1)
            wv = part(tensor, 2)

            if False: #name.endswith('weight'):
                wq =  permute(wq, config.n_head)
                wk =  permute(wk, config.n_head)

            added[name.replace('Wqkv', 'q_proj')] = wq
            added[name.replace('Wqkv', 'k_proj')] = wk
            added[name.replace('Wqkv', 'v_proj')] = wv

        for s in removed:
            del state_dict[s]
        for name in added:
            state_dict[name] = added[name]
        return state_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        if config.hidden_act is None:
            assert config.activation_function == 'gelu_new', "activation_function must be gelu_new"
            assert config.n_head_kv is None, "n_head_kv must be null"
            assert config.rotary_dim == config.n_head, "rotary_dim must == n_head"
            config_values = [
                ggml_type.value,
                config.vocab_size,
                config.n_embd,
                config.n_head,
                config.n_layer,
                config.n_inner if config.n_inner is not None else 4 * config.n_embd,
                config.n_positions,
                config.bos_token_id if config.bos_token_id is not None else -1,
                config.eos_token_id if config.eos_token_id is not None else -1,
                config.pad_token_id if config.pad_token_id is not None else -1,
                config.sep_token_id if config.sep_token_id is not None else -1,
            ]
            f.write(struct.pack("i" * len(config_values), *config_values))
        else:
            assert config.hidden_act == 'gelu_new', "activation_function must be gelu_new"
            assert config.rope_scaling is None, "rope_scaling must be None"
            assert config.qk_layernorm == False, "qk_layernorm must be False"
            assert config.num_attention_heads == config.num_key_value_heads, "num_attention_heads must equal to num_key_value_heads"

            head_dim = config.hidden_size // config.num_attention_heads
            rope_dim = int(config.partial_rotary_factor * head_dim)

            config_values = [
                ggml_type.value,
                config.vocab_size,
                config.hidden_size,
                config.num_attention_heads,
                config.num_hidden_layers,
                config.intermediate_size,
                config.max_position_embeddings,
                config.bos_token_id if config.bos_token_id is not None else -1,
                config.eos_token_id if config.eos_token_id is not None else -1,
                config.pad_token_id if config.pad_token_id is not None else -1,
                config.sep_token_id if config.sep_token_id is not None else -1,
                rope_dim
            ]
            f.write(struct.pack("i" * len(config_values), *config_values))
            f.write(struct.pack("<f", config.rope_theta))


    @staticmethod
    def get_weight_names_v1(config):
        weight_names = ["transformer.embd.wte.weight"]
        for i in range(config.n_layer):
            weight_names += [
                f"transformer.h.{i}.ln.bias",
                f"transformer.h.{i}.ln.weight",
                f"transformer.h.{i}.mixer.q_proj.bias",
                f"transformer.h.{i}.mixer.q_proj.weight",
                f"transformer.h.{i}.mixer.k_proj.bias",
                f"transformer.h.{i}.mixer.k_proj.weight",
                f"transformer.h.{i}.mixer.v_proj.bias",
                f"transformer.h.{i}.mixer.v_proj.weight",
                f"transformer.h.{i}.mixer.out_proj.bias",
                f"transformer.h.{i}.mixer.out_proj.weight",
                f"transformer.h.{i}.mlp.fc1.bias",
                f"transformer.h.{i}.mlp.fc1.weight",
                f"transformer.h.{i}.mlp.fc2.bias",
                f"transformer.h.{i}.mlp.fc2.weight",
            ]

        weight_names += [
            "lm_head.ln.bias",
            "lm_head.ln.weight",
            "lm_head.linear.bias",
            "lm_head.linear.weight",
        ]

        return weight_names

    @staticmethod
    def get_weight_names_v2(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.bias",
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.dense.bias",
                f"model.layers.{i}.self_attn.dense.weight",
                f"model.layers.{i}.mlp.fc1.bias",
                f"model.layers.{i}.mlp.fc1.weight",
                f"model.layers.{i}.mlp.fc2.bias",
                f"model.layers.{i}.mlp.fc2.weight",
            ]

        weight_names += [
            "model.final_layernorm.bias",
            "model.final_layernorm.weight",
            "lm_head.bias",
            "lm_head.weight",
        ]

        return weight_names

    @staticmethod
    def get_weight_names(config):
        if config.n_layer is not None:
            return Phi2Converter.get_weight_names_v1(config)
        else:
            return Phi2Converter.get_weight_names_v2(config)

def pad_to(l: list, to_len: int, fill = 0) -> list:
    if len(l) < to_len:
        return l + [fill] * (to_len - len(l))
    else:
        return l

class Phi3Converter(BaseConverter):
    MODEL_TYPE = ModelType.Phi3

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.endswith('mlp.gate_up_proj.weight'):

                new_dict[name.replace('gate_up_proj.weight', 'gate_proj.weight')] = part(tensor, 0, 2)
                new_dict[name.replace('gate_up_proj.weight', 'up_proj.weight')]   = part(tensor, 1, 2)
            if name.endswith('.qkv_proj.weight'):

                q_size = config.hidden_size
                kv_size = config.hidden_size // config.num_attention_heads * config.num_key_value_heads

                wq = tensor[: q_size, ...]
                wk = tensor[q_size : q_size + kv_size, ...]
                wv = tensor[q_size + kv_size :, ...]

                if config.partial_rotary_factor is None:
                    new_dict[name.replace('qkv_proj.weight', 'q_proj.weight')] = permute(wq, config.num_attention_heads)
                    new_dict[name.replace('qkv_proj.weight', 'k_proj.weight')] = permute(wk, config.num_key_value_heads)
                else:
                    new_dict[name.replace('qkv_proj.weight', 'q_proj.weight')] = permute2(wq, config.num_attention_heads, config.partial_rotary_factor)
                    new_dict[name.replace('qkv_proj.weight', 'k_proj.weight')] = permute2(wk, config.num_key_value_heads, config.partial_rotary_factor)
                new_dict[name.replace('qkv_proj.weight', 'v_proj.weight')] = wv
            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):

        assert config.rope_scaling is None, "rope_scaling must be null"

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.original_max_position_embeddings,
            config.sliding_window,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        float_values = [
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(float_values), *float_values))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class Phi4Converter(Phi3Converter):
    MODEL_TYPE = ModelType.Phi4

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        return Phi3Converter.state_dict_pp(config, state_dict)

    @staticmethod
    def dump_config(f, config, ggml_type):

        assert config.rope_scaling is None, "rope_scaling must be null"
        assert config.sliding_window is None, "sliding_window must be null"
        assert config.original_max_position_embeddings == config.max_position_embeddings, "original_max_position_embeddings must be equal to max_position_embeddings"

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        float_values = [
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(float_values), *float_values))

    @staticmethod
    def get_weight_names(config):
        return Phi3Converter.get_weight_names(config)
class Phi3SUConverter(BaseConverter):
    MODEL_TYPE = ModelType.Phi3_ScalingSU2

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        return Phi3Converter.state_dict_pp(config, state_dict)

    @staticmethod
    def dump_config(f, config, ggml_type):

        dump_llama_like_config(f, config, ggml_type)

        MAX_FACTOR_LEN = 128

        assert config.rope_scaling['type'] == "longrope", "rope_scaling must be null or `longrope`"
        assert len(config.rope_scaling['long_factor']) <= MAX_FACTOR_LEN, "config.rope_scaling['long_factor']) must <= MAX_FACTOR_LEN"
        rope_scaling = 1
        factors = pad_to(config.rope_scaling['short_factor'], MAX_FACTOR_LEN) + pad_to(config.rope_scaling['long_factor'], MAX_FACTOR_LEN)

        config_values = [
            config.max_position_embeddings,
            config.num_key_value_heads,
            config.original_max_position_embeddings,
            config.sliding_window,
            rope_scaling,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        float_values = [
            config.rope_theta,
        ] + factors
        f.write(struct.pack("<" + "f" * len(float_values), *float_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = Phi3Converter.get_weight_names(config)
        if config.tie_word_embeddings:
            weight_names.remove("lm_head.weight")
        return weight_names

class Phi3SU3Converter(BaseConverter):
    MODEL_TYPE = ModelType.Phi3_ScalingSU3

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        return Phi3SUConverter.state_dict_pp(config, state_dict)

    @staticmethod
    def dump_config(f, config, ggml_type):
        Phi3SUConverter.dump_config(f, config, ggml_type)

        config_values = [
            config.rope_scaling['short_mscale'],
            config.rope_scaling['long_mscale'],
        ]
        f.write(struct.pack('<' + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        return Phi3SUConverter.get_weight_names(config)

class Phi3MoESUConverter(BaseConverter):
    MODEL_TYPE = ModelType.Phi3MoE_ScalingSU

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        else:
            return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.lm_head_bias, 'lm_head_bias must be True'

        Phi3SU3Converter.dump_config(f, config, ggml_type)

        config_values = [
            config.num_experts_per_tok,
            config.num_local_experts,
        ]
        f.write(struct.pack('<' + "i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.input_layernorm.bias",
            ]

            for j in range(config.num_local_experts):
                weight_names += [
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight",
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight",
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight",
                ]

            weight_names += [
                f"model.layers.{i}.block_sparse_moe.gate.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.bias",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.bias",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
            ]

        weight_names += [
            "model.norm.weight",
            "model.norm.bias",
            "lm_head.weight",
            "lm_head.bias"
        ]

        return weight_names

class QWenConverter(BaseConverter):
    MODEL_TYPE = ModelType.QWen
    FILE_VERSION = 2

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        added = {}
        removed = []
        for name in state_dict:

            if name.endswith('attn.c_attn.weight'):
                tensor: torch.Tensor = state_dict[name]
                removed.append(name)

                wq = part(tensor, 0)
                wk = part(tensor, 1)
                wv = part(tensor, 2)

                added[name.replace('c_attn', 'q_proj')] = wq
                added[name.replace('c_attn', 'k_proj')] = wk
                added[name.replace('c_attn', 'v_proj')] = wv

            elif name.endswith('attn.c_attn.bias'):
                tensor: torch.Tensor = state_dict[name]
                removed.append(name)

                wq = part(tensor, 0)
                wk = part(tensor, 1)
                wv = part(tensor, 2)

                added[name.replace('c_attn', 'q_proj')] = wq
                added[name.replace('c_attn', 'k_proj')] = wk
                added[name.replace('c_attn', 'v_proj')] = wv

        for s in removed:
            del state_dict[s]
        for name in added:
            state_dict[name] = added[name]
        return state_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.no_bias == True, "no_bias must be true"
        assert config.scale_attn_weights == True, "scale_attn_weights must be true"
        assert config.seq_length > 0, "seq_length must be positive"
        assert config.kv_channels * config.num_attention_heads == config.hidden_size, "`kv_channels * num_attention_heads = config.hidden_size` must holds"

        rope_dim = int(config.kv_channels * config.rotary_pct)
        flags = 0
        flags |= 1 if config.use_dynamic_ntk else 0
        flags |= 2 if config.use_logn_attn else 0

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size // 2,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,

            config.seq_length,
            rope_dim,
            flags
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rotary_emb_base))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["transformer.wte.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"transformer.h.{i}.attn.k_proj.weight",
                f"transformer.h.{i}.attn.k_proj.bias",
                f"transformer.h.{i}.attn.q_proj.weight",
                f"transformer.h.{i}.attn.q_proj.bias",
                f"transformer.h.{i}.attn.v_proj.weight",
                f"transformer.h.{i}.attn.v_proj.bias",
                f"transformer.h.{i}.attn.c_proj.weight",
                f"transformer.h.{i}.ln_1.weight",
                f"transformer.h.{i}.ln_2.weight",
                f"transformer.h.{i}.mlp.c_proj.weight",
                f"transformer.h.{i}.mlp.w1.weight",
                f"transformer.h.{i}.mlp.w2.weight",
            ]

        weight_names += [
            "transformer.ln_f.weight",
            "lm_head.weight"
        ]

        return weight_names

class QWen2Converter(BaseConverter):
    MODEL_TYPE = ModelType.QWen2

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.use_sliding_window == False, "use_sliding_window must be False"
        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.sliding_window if config.sliding_window is not None else config.max_position_embeddings
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        if (config.tie_word_embeddings is None) or (not config.tie_word_embeddings):
            weight_names += [
                "lm_head.weight"
            ]

        return weight_names

class QWen2AudioConverter(BaseConverter):
    MODEL_TYPE = ModelType.Qwen2Audio

    txt_config = {}

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        r = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            new_name = name
            if new_name.startswith('audio_tower.'):
                new_name = new_name.replace('audio_tower.', 'audio.')
                if '.out_proj.' in new_name:
                    new_name = new_name.replace('.out_proj.', '.o_proj.')
                elif '.fc1.' in new_name:
                    new_name = new_name.replace('.fc1.', '.mlp.fc1.')
                elif '.fc2.' in new_name:
                    new_name = new_name.replace('.fc2.', '.mlp.fc2.')
                new_name = new_name.replace('.self_attn_layer_norm.', '.input_layernorm.')
                new_name = new_name.replace('.final_layer_norm.',     '.post_attention_layernorm.')
            elif new_name.startswith('language_model.'):
                new_name = new_name.replace('language_model.', '')

            r[new_name] = tensor
        return r

    @staticmethod
    def dump_config(f, config, ggml_type):
        txt_config = copy.deepcopy(config.text_config)

        default = {
            'hidden_act': 'silu',
            'hidden_size': 4096,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 32,
            'use_sliding_window': False
        }
        for k, v in default.items():
            if k not in txt_config:
                txt_config[k] = v

        QWen2AudioConverter.txt_config = AttributeDict(txt_config)
        QWen2Converter.dump_config(f, QWen2AudioConverter.txt_config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        weight_names = QWen2Converter.get_weight_names(QWen2AudioConverter.txt_config)

        for i in range(config.audio_config['encoder_layers']):
            weight_names += [
                f"audio.layers.{i}.mlp.fc1.bias",
                f"audio.layers.{i}.mlp.fc1.weight",
                f"audio.layers.{i}.mlp.fc2.bias",
                f"audio.layers.{i}.mlp.fc2.weight",
                f"audio.layers.{i}.post_attention_layernorm.bias",
                f"audio.layers.{i}.post_attention_layernorm.weight",
                f"audio.layers.{i}.self_attn.k_proj.weight",
                f"audio.layers.{i}.self_attn.o_proj.bias",
                f"audio.layers.{i}.self_attn.o_proj.weight",
                f"audio.layers.{i}.self_attn.q_proj.bias",
                f"audio.layers.{i}.self_attn.q_proj.weight",
                f"audio.layers.{i}.self_attn.v_proj.bias",
                f"audio.layers.{i}.self_attn.v_proj.weight",
                f"audio.layers.{i}.input_layernorm.bias",
                f"audio.layers.{i}.input_layernorm.weight",
            ]

        weight_names += [
            "audio.conv1.bias",
            "audio.conv1.weight",
            "audio.conv2.bias",
            "audio.conv2.weight",
            "audio.embed_positions.weight",
            "audio.layer_norm.bias",
            "audio.layer_norm.weight",
            "multi_modal_projector.linear.bias",
            "multi_modal_projector.linear.weight",
        ]

        return weight_names

class QWen2_5VLConverter(BaseConverter):
    MODEL_TYPE = ModelType.Qwen2_5VL

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        r = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            if name == 'visual.patch_embed.proj.weight':
                shape = tensor.shape
                assert len(shape) == 5
                r[name] = tensor.view(shape[0], shape[1] * shape[2] * shape[3] * shape[4])
            elif name.endswith('.attn.qkv.bias') or name.endswith('.attn.qkv.weight'):
                #print(f'shape: {name} = {tensor.shape}')
                num_heads = config.vision_config['hidden_size']
                q, k, v = tensor.split([num_heads, num_heads, num_heads], dim=0)
                r[name.replace('.attn.qkv.', '.attn.q_proj.')] = q
                r[name.replace('.attn.qkv.', '.attn.k_proj.')] = k
                r[name.replace('.attn.qkv.', '.attn.v_proj.')] = v
            else:
                r[name] = tensor

        return r

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.rope_scaling['type'] == 'mrope', 'rope_scaling must be mrope'
        assert config.vision_config['hidden_act'] == 'silu'

        QWen2Converter.dump_config(f, config, ggml_type)

        MROPE_SECTION_MAX = 4

        config_values = [
            config.tie_word_embeddings if config.tie_word_embeddings is not None else 0
        ] + pad_to_len(config.rope_scaling['mrope_section'], MROPE_SECTION_MAX)
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = QWen2Converter.get_weight_names(config)

        for i in range(config.vision_config['depth']):
            weight_names += [
                f"visual.blocks.{i}.attn.proj.bias",
                f"visual.blocks.{i}.attn.proj.weight",
                f"visual.blocks.{i}.attn.q_proj.bias",
                f"visual.blocks.{i}.attn.q_proj.weight",
                f"visual.blocks.{i}.attn.k_proj.bias",
                f"visual.blocks.{i}.attn.k_proj.weight",
                f"visual.blocks.{i}.attn.v_proj.bias",
                f"visual.blocks.{i}.attn.v_proj.weight",
                f"visual.blocks.{i}.mlp.down_proj.bias",
                f"visual.blocks.{i}.mlp.down_proj.weight",
                f"visual.blocks.{i}.mlp.gate_proj.bias",
                f"visual.blocks.{i}.mlp.gate_proj.weight",
                f"visual.blocks.{i}.mlp.up_proj.bias",
                f"visual.blocks.{i}.mlp.up_proj.weight",
                f"visual.blocks.{i}.norm1.weight",
                f"visual.blocks.{i}.norm2.weight",
            ]

        weight_names += [
            "visual.merger.ln_q.weight",
            "visual.merger.mlp.0.bias",
            "visual.merger.mlp.0.weight",
            "visual.merger.mlp.2.bias",
            "visual.merger.mlp.2.weight",
            "visual.patch_embed.proj.weight",
        ]

        return weight_names

class MiniCPMOConverter(BaseConverter):
    MODEL_TYPE = ModelType.MiniCPM_O

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name == 'visual.patch_embed.proj.weight':
            shape = tensor.shape
            return tensor.view(shape[0], shape[1] * shape[2] * shape[3] * shape[4])
        else:
            return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert not config.tie_word_embeddings, 'tie_word_embeddings must be False'
        assert not config.drop_vision_last_layer, 'drop_vision_last_layer must be False'
        assert config.slice_config["model_type"] == "minicpmv", 'slice_config.model_type must be minicpmv'
        assert config.slice_mode, 'slice_mode must be True'
        assert config.use_image_id, 'use_image_id must be True'
        assert config.audio_config['architectures'][0] == "MiniCPMWhisperEncoder", 'audio_config.architectures[0] must be MiniCPMWhisperEncoder'

        QWen2Converter.dump_config(f, config, ggml_type)

        # vision
        config_values = [
            config.image_size,
            config.patch_size,
            config.query_num,
            config.slice_config['max_slice_nums'],
            config.vision_batch_size,
            config.vision_config['hidden_size'],
            config.vision_config['image_size'],
            config.vision_config['intermediate_size'],
            config.vision_config['num_attention_heads'],
            config.vision_config['num_hidden_layers'],
            config.vision_config['patch_size'],
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

        # audio
        config_values = [
            config.audio_chunk_length,
            config.audio_config['d_model'],
            config.audio_config['num_hidden_layers'],
            config.audio_config['decoder_attention_heads'],
            config.audio_config['decoder_ffn_dim'],
            config.audio_config['decoder_layers'],
            config.audio_config['encoder_attention_heads'],
            config.audio_config['encoder_ffn_dim'],
            config.audio_config['encoder_layers'],

            config.audio_config['decoder_start_token_id'],
            config.audio_config['bos_token_id'],
            config.audio_config['eos_token_id'],
            config.audio_config['pad_token_id'],

            config.audio_config['max_length'],

            config.audio_pool_step,
        ]
        f.write(struct.pack("<" + "f" * 1 + "i" * (len(config_values) - 1), *config_values))

        tts_config = {
            "llm_dim": 2560,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 20,
            "max_position_embeddings": 4096,
            "num_audio_tokens": 626,
            "num_text_tokens": 21178,
            "num_mel_bins": 100,
            "num_vq": 4,
            "spk_emb_token_id": 21143,
            "num_spk_embs": 1,
            "audio_bos_token_id": 21132,
            "text_eos_token_id": 21133,
            "streaming_text_chunk_size": 10,
            "streaming_text_reserved_len": 300,
            "streaming_audio_chunk_size": 50,
        }
        tts_config.update(config.tts_config)

        # tts
        config_values = [
            tts_config['llm_dim'],
            tts_config['hidden_size'],
            tts_config['intermediate_size'],
            tts_config['num_attention_heads'],
            tts_config['num_hidden_layers'],
            tts_config['max_position_embeddings'],
            tts_config['num_audio_tokens'],
            tts_config['num_text_tokens'],
            tts_config['num_mel_bins'],
            tts_config['num_vq'],
            tts_config['spk_emb_token_id'],
            tts_config['num_spk_embs'],
            tts_config['audio_bos_token_id'],
            tts_config['text_eos_token_id'],
            tts_config['streaming_text_chunk_size'],
            tts_config['streaming_text_reserved_len'],
            tts_config['streaming_audio_chunk_size'],
        ]
        f.write(struct.pack("<" + "f" * 1 + "i" * (len(config_values) - 1), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ['llm.' + n for n in QWen2Converter.get_weight_names(config)]

        weight_names += [
                "apm.conv1.bias",
                "apm.conv1.weight",
                "apm.conv2.bias",
                "apm.conv2.weight",
                "apm.embed_positions.weight",
                "apm.layer_norm.bias",
                "apm.layer_norm.weight",
                "audio_projection_layer.linear1.bias",
                "audio_projection_layer.linear1.weight",
                "audio_projection_layer.linear2.bias",
                "audio_projection_layer.linear2.weight"
        ]
        for i in range(config.audio_config['encoder_layers']):
            weight_names += [
                f"apm.layers.{i}.fc1.bias",
                f"apm.layers.{i}.fc1.weight",
                f"apm.layers.{i}.fc2.bias",
                f"apm.layers.{i}.fc2.weight",
                f"apm.layers.{i}.final_layer_norm.bias",
                f"apm.layers.{i}.final_layer_norm.weight",
                f"apm.layers.{i}.self_attn.k_proj.weight",
                f"apm.layers.{i}.self_attn.out_proj.bias",
                f"apm.layers.{i}.self_attn.out_proj.weight",
                f"apm.layers.{i}.self_attn.q_proj.bias",
                f"apm.layers.{i}.self_attn.q_proj.weight",
                f"apm.layers.{i}.self_attn.v_proj.bias",
                f"apm.layers.{i}.self_attn.v_proj.weight",
                f"apm.layers.{i}.self_attn_layer_norm.bias",
                f"apm.layers.{i}.self_attn_layer_norm.weight",
            ]

        weight_names += [
                "resampler.attn.in_proj_bias",
                "resampler.attn.in_proj_weight",
                "resampler.attn.out_proj.bias",
                "resampler.attn.out_proj.weight",
                "resampler.kv_proj.weight",
                "resampler.ln_kv.bias",
                "resampler.ln_kv.weight",
                "resampler.ln_post.bias",
                "resampler.ln_post.weight",
                "resampler.ln_q.bias",
                "resampler.ln_q.weight",
                "resampler.proj",
                "resampler.query",
                "tts.dvae.coef",
                "tts.dvae.decoder.conv_in.0.bias",
                "tts.dvae.decoder.conv_in.0.weight",
                "tts.dvae.decoder.conv_in.2.bias",
                "tts.dvae.decoder.conv_in.2.weight",
                "tts.dvae.decoder.conv_out.weight",
        ]

        for i in range(12):
            weight_names += [
                f"tts.dvae.decoder.decoder_block.{i}.coef",
                f"tts.dvae.decoder.decoder_block.{i}.dwconv.bias",
                f"tts.dvae.decoder.decoder_block.{i}.dwconv.weight",
                f"tts.dvae.decoder.decoder_block.{i}.norm.bias",
                f"tts.dvae.decoder.decoder_block.{i}.norm.weight",
                f"tts.dvae.decoder.decoder_block.{i}.pwconv1.bias",
                f"tts.dvae.decoder.decoder_block.{i}.pwconv1.weight",
                f"tts.dvae.decoder.decoder_block.{i}.pwconv2.bias",
                f"tts.dvae.decoder.decoder_block.{i}.pwconv2.weight",
            ]


        weight_names += [
            "tts.dvae.downsample_conv.0.bias",
            "tts.dvae.downsample_conv.0.weight",
            "tts.dvae.downsample_conv.2.bias",
            "tts.dvae.downsample_conv.2.weight",
            "tts.dvae.encoder.conv_in.0.bias",
            "tts.dvae.encoder.conv_in.0.weight",
            "tts.dvae.encoder.conv_in.2.bias",
            "tts.dvae.encoder.conv_in.2.weight",
            "tts.dvae.encoder.conv_out.weight",
        ]

        for i in range(12):
            weight_names += [
                f"tts.dvae.encoder.decoder_block.{i}.coef",
                f"tts.dvae.encoder.decoder_block.{i}.dwconv.bias",
                f"tts.dvae.encoder.decoder_block.{i}.dwconv.weight",
                f"tts.dvae.encoder.decoder_block.{i}.norm.bias",
                f"tts.dvae.encoder.decoder_block.{i}.norm.weight",
                f"tts.dvae.encoder.decoder_block.{i}.pwconv1.bias",
                f"tts.dvae.encoder.decoder_block.{i}.pwconv1.weight",
                f"tts.dvae.encoder.decoder_block.{i}.pwconv2.bias",
                f"tts.dvae.encoder.decoder_block.{i}.pwconv2.weight",
            ]

        weight_names += [
            "tts.dvae.out_conv.weight",
            "tts.dvae.vq_layer.quantizer.rvqs.0.project_in.bias",
            "tts.dvae.vq_layer.quantizer.rvqs.0.project_in.weight",
            "tts.dvae.vq_layer.quantizer.rvqs.0.project_out.bias",
            "tts.dvae.vq_layer.quantizer.rvqs.0.project_out.weight",
            "tts.dvae.vq_layer.quantizer.rvqs.1.project_in.bias",
            "tts.dvae.vq_layer.quantizer.rvqs.1.project_in.weight",
            "tts.dvae.vq_layer.quantizer.rvqs.1.project_out.bias",
            "tts.dvae.vq_layer.quantizer.rvqs.1.project_out.weight",
            "tts.emb_code.0.weight",
            "tts.emb_code.1.weight",
            "tts.emb_code.2.weight",
            "tts.emb_code.3.weight",
            "tts.emb_text.weight",
            "tts.head_code.0.parametrizations.weight.original0",
            "tts.head_code.0.parametrizations.weight.original1",
            "tts.head_code.1.parametrizations.weight.original0",
            "tts.head_code.1.parametrizations.weight.original1",
            "tts.head_code.2.parametrizations.weight.original0",
            "tts.head_code.2.parametrizations.weight.original1",
            "tts.head_code.3.parametrizations.weight.original0",
            "tts.head_code.3.parametrizations.weight.original1",
            "tts.model.embed_tokens.weight",
        ]

        for i in range(config.tts_config['num_hidden_layers'] if 'num_hidden_layers' in config.tts_config else 20):
            weight_names += [
                f"tts.model.layers.{i}.input_layernorm.weight",
                f"tts.model.layers.{i}.mlp.down_proj.weight",
                f"tts.model.layers.{i}.mlp.gate_proj.weight",
                f"tts.model.layers.{i}.mlp.up_proj.weight",
                f"tts.model.layers.{i}.post_attention_layernorm.weight",
                f"tts.model.layers.{i}.self_attn.k_proj.weight",
                f"tts.model.layers.{i}.self_attn.o_proj.weight",
                f"tts.model.layers.{i}.self_attn.q_proj.weight",
                f"tts.model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "tts.model.norm.weight",
            "tts.projector.linear1.bias",
            "tts.projector.linear1.weight",
            "tts.projector.linear2.bias",
            "tts.projector.linear2.weight",
            "vpm.embeddings.patch_embedding.bias",
            "vpm.embeddings.patch_embedding.weight",
            "vpm.embeddings.position_embedding.weight",
        ]

        for i in range(config.vision_config['num_hidden_layers']):
            weight_names += [
                f"vpm.encoder.layers.{i}.layer_norm1.bias",
                f"vpm.encoder.layers.{i}.layer_norm1.weight",
                f"vpm.encoder.layers.{i}.layer_norm2.bias",
                f"vpm.encoder.layers.{i}.layer_norm2.weight",
                f"vpm.encoder.layers.{i}.mlp.fc1.bias",
                f"vpm.encoder.layers.{i}.mlp.fc1.weight",
                f"vpm.encoder.layers.{i}.mlp.fc2.bias",
                f"vpm.encoder.layers.{i}.mlp.fc2.weight",
                f"vpm.encoder.layers.{i}.self_attn.k_proj.bias",
                f"vpm.encoder.layers.{i}.self_attn.k_proj.weight",
                f"vpm.encoder.layers.{i}.self_attn.out_proj.bias",
                f"vpm.encoder.layers.{i}.self_attn.out_proj.weight",
                f"vpm.encoder.layers.{i}.self_attn.q_proj.bias",
                f"vpm.encoder.layers.{i}.self_attn.q_proj.weight",
                f"vpm.encoder.layers.{i}.self_attn.v_proj.bias",
                f"vpm.encoder.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "vpm.post_layernorm.bias",
            "vpm.post_layernorm.weight"
        ]

        return weight_names

class QWen2TieConverter(BaseConverter):
    MODEL_TYPE = ModelType.DeepSeek_R1_Distill_QWen

    @staticmethod
    def dump_config(f, config, ggml_type):
        QWen2Converter.dump_config(f, config, ggml_type)
        config_values = [
            0 if (config.tie_word_embeddings is None) or (not config.tie_word_embeddings) else 1
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        return QWen2Converter.get_weight_names(config)

class QWen2MoEConverter(BaseConverter):
    MODEL_TYPE = ModelType.QWen2MoE

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.use_sliding_window == False, "use_sliding_window must be False"
        assert config.decoder_sparse_step == 1, "decoder_sparse_step must be 1"
        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.moe_intermediate_size,
            config.shared_expert_intermediate_size,
            config.sliding_window,
            config.num_experts_per_tok,
            config.num_experts,
            1 if config.norm_topk_prob else 0,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):

            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
            ]

            for j in range(config.num_experts):
                weight_names += [
                    f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                    f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                    f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.mlp.gate.weight",
                f"model.layers.{i}.mlp.shared_expert.down_proj.weight",
                f"model.layers.{i}.mlp.shared_expert.gate_proj.weight",
                f"model.layers.{i}.mlp.shared_expert.up_proj.weight",
                f"model.layers.{i}.mlp.shared_expert_gate.weight",

                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class PanguMoEConverter(BaseConverter):
    MODEL_TYPE = ModelType.PenguMoE

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert not config.tie_word_embeddings
        assert config.intermediate_size is None
        config.intermediate_size = config.shared_expert_intermediate_size

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.moe_intermediate_size,
            config.num_experts_per_tok,
            config.num_experts,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):

            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
            ]

            for j in range(config.num_experts):
                weight_names += [
                    f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                    f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                    f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.mlp.gate.weight",
                f"model.layers.{i}.mlp.shared_expert.down_proj.weight",
                f"model.layers.{i}.mlp.shared_expert.gate_proj.weight",
                f"model.layers.{i}.mlp.shared_expert.up_proj.weight",
                f"model.layers.{i}.mlp.router_scale",

                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.bias",
            ]

        weight_names += [
            "model.norm.weight",
            "lm_head.weight"
        ]

        return weight_names

class QWen3Converter(BaseConverter):
    MODEL_TYPE = ModelType.QWen3

    layer_is_sparse = []
    has_lm_head = True

    @staticmethod
    def dump_config(f, config, ggml_type):
        MAX_LAYERS = 128
        assert config.use_sliding_window == False, "use_sliding_window must be False"
        assert not config.attention_bias
        assert (config.output_router_logits is None) or (not config.output_router_logits)

        rope_scaling = config.rope_scaling
        if rope_scaling is None:
            rope_scaling = {
                "type": "yarn",
                "factor": -1,
                "original_max_position_embeddings": -1
            }
        else:
            if not 'type' in rope_scaling:
                rope_scaling['type'] = rope_scaling['rope_type']
            assert rope_scaling['type'] == 'yarn', 'rope_scaling must be yarn'
            config.max_position_embeddings = int(rope_scaling['original_max_position_embeddings'] * rope_scaling['factor'])

        if config.moe_intermediate_size is None:
            config.moe_intermediate_size = -1
        if config.num_experts is None:
            config.num_experts = -1
        if config.num_experts_per_tok is None:
            config.num_experts_per_tok = -1
        if config.norm_topk_prob is None:
            config.norm_topk_prob = False

        decoder_sparse_step = config.decoder_sparse_step if config.decoder_sparse_step is not None else 1
        mlp_only_layers = config.mlp_only_layers if config.mlp_only_layers is not None else list(range(config.num_hidden_layers))

        def is_sparse(layer_idx, config):
            nonlocal decoder_sparse_step, mlp_only_layers
            if layer_idx >= config.num_hidden_layers: return 0

            f = (layer_idx not in mlp_only_layers) and (
                    config.num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
                )
            return 1 if f else 0

        QWen3Converter.layer_is_sparse = [is_sparse(i, config) for i in range(MAX_LAYERS)]

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.head_dim,
            config.rope_theta,
            rope_scaling['factor'],
            rope_scaling['original_max_position_embeddings'],
            decoder_sparse_step,
            config.moe_intermediate_size,
            config.num_experts_per_tok,
            config.num_experts,
            1 if config.norm_topk_prob else 0,
            1 if config.tie_word_embeddings else 0,
        ]
        f.write(struct.pack("<iiffiiiiiii", *config_values))
        f.write(struct.pack("<" + "i" * len(QWen3Converter.layer_is_sparse), *QWen3Converter.layer_is_sparse))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):

            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
            ]

            if QWen3Converter.layer_is_sparse[i]:
                for j in range(config.num_experts):
                    weight_names += [
                        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                    ]
                weight_names += [
                    f"model.layers.{i}.mlp.gate.weight",
                ]
            else:
                weight_names += [
                    f"model.layers.{i}.mlp.down_proj.weight",
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
            ]

        weight_names += [
            "model.norm.weight"
        ]

        if QWen3Converter.has_lm_head and (not config.tie_word_embeddings):
            weight_names += [
                "lm_head.weight"
            ]

        return weight_names

class QWen3EmbConverter(BaseConverter):
    MODEL_TYPE = ModelType.QWen3_Embedding

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        r = {}
        for name in state_dict:
            r['model.' + name] = state_dict[name]

        return r

    @staticmethod
    def dump_config(f, config, ggml_type):
        QWen3Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        QWen3Converter.has_lm_head = False
        return QWen3Converter.get_weight_names(config)

def permute2(weights: torch.Tensor, n_head: int, partial_rotary_factor: float) -> torch.Tensor:
    hidden_size = weights.shape[0]
    head_dim = hidden_size // n_head
    rope_dim = int(partial_rotary_factor * head_dim)
    reshaped = weights.reshape(n_head, head_dim, *weights.shape[1:])
    rot = reshaped[:, :rope_dim, ...]
    other = reshaped[:, rope_dim:, ...]
    rot = rot.reshape(n_head, 2, rope_dim // 2, *weights.shape[1:]).swapaxes(1, 2).reshape(rot.shape)
    combined = torch.concat((rot, other), dim = 1)
    return combined.reshape(weights.shape)

def permute_pair_3(weights: torch.Tensor, n_head: int, nope_dim: int) -> torch.Tensor:
    hidden_size = weights.shape[0]
    head_dim = hidden_size // n_head
    rope_dim = head_dim - nope_dim
    reshaped = weights.reshape(n_head, head_dim, *weights.shape[1:])
    rot = reshaped[:, nope_dim:, ...]
    other = reshaped[:, :nope_dim, ...]
    rot = rot.reshape(n_head, rope_dim // 2, 2, *weights.shape[1:]).swapaxes(1, 2).reshape(rot.shape)
    combined = torch.concat((other, rot), dim = 1)
    return combined.reshape(weights.shape)

class PersimmonConverter(BaseConverter):
    MODEL_TYPE = ModelType.Persimmon

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        partial_rotary_factor = config.partial_rotary_factor if config.partial_rotary_factor is not None else 0.5

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.endswith('query_key_value.weight'):
                num_heads = config.num_attention_heads
                head_dim = config.hidden_size // config.num_attention_heads

                v = tensor.view(num_heads, 3, head_dim, tensor.shape[1])

                wq = v[:, 0, ...].reshape(config.hidden_size, tensor.shape[1])
                wk = v[:, 1, ...].reshape(config.hidden_size, tensor.shape[1])
                wv = v[:, 2, ...].reshape(config.hidden_size, tensor.shape[1])

                new_dict[name.replace('query_key_value', 'q_proj')] = wq
                new_dict[name.replace('query_key_value', 'k_proj')] = wk
                new_dict[name.replace('query_key_value', 'v_proj')] = wv

            elif name.endswith('query_key_value.bias'):
                num_heads = config.num_attention_heads
                head_dim = config.hidden_size // config.num_attention_heads

                v = tensor.view(num_heads, 3, head_dim)

                wq = v[:, 0, ...].reshape(config.hidden_size)
                wk = v[:, 1, ...].reshape(config.hidden_size)
                wv = v[:, 2, ...].reshape(config.hidden_size)

                new_dict[name.replace('query_key_value', 'q_proj')] = wq
                new_dict[name.replace('query_key_value', 'k_proj')] = wk
                new_dict[name.replace('query_key_value', 'v_proj')] = wv

            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'relu2', 'hidden_act must be relu2'
        assert config.qk_layernorm, "qk_layernorm must be true"
        assert config.rope_scaling is None, 'rope_scaling must be null'
        if config.num_key_value_heads is not None:
            assert config.num_attention_heads == config.num_key_value_heads, 'num_attention_heads must be equal to num_key_value_heads'

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads

        partial_rotary_factor = config.partial_rotary_factor if config.partial_rotary_factor is not None else 0.5
        rope_dim = int(partial_rotary_factor * head_dim)

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads if config.sep_token_id is not None else config.num_attention_heads,
            rope_dim,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.input_layernorm.bias",
                f"model.layers.{i}.mlp.dense_4h_to_h.weight",
                f"model.layers.{i}.mlp.dense_4h_to_h.bias",
                f"model.layers.{i}.mlp.dense_h_to_4h.weight",
                f"model.layers.{i}.mlp.dense_h_to_4h.bias",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.bias",
                f"model.layers.{i}.self_attn.dense.weight",
                f"model.layers.{i}.self_attn.dense.bias",
                f"model.layers.{i}.self_attn.k_layernorm.weight",
                f"model.layers.{i}.self_attn.k_layernorm.bias",
                f"model.layers.{i}.self_attn.q_layernorm.weight",
                f"model.layers.{i}.self_attn.q_layernorm.bias",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
            ]

        weight_names += [
            "model.final_layernorm.weight",
            "model.final_layernorm.bias",
            "lm_head.weight"
        ]

        return weight_names

class FuyuConverter(BaseConverter):
    MODEL_TYPE = ModelType.Fuyu

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.startswith('language_model.'):
                new_dict[name.replace('language_model.', '')] = tensor
            else:
                new_dict[name] = tensor

        return PersimmonConverter.state_dict_pp(config, new_dict)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.num_channels == 3, 'num_channels must be 3'

        PersimmonConverter.dump_config(f, config, ggml_type)

        config_values = [
            config.patch_size,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = PersimmonConverter.get_weight_names(config)
        weight_names += [
            'vision_embed_tokens.weight',
            'vision_embed_tokens.bias',
        ]

        return weight_names

class XLMRobertaConverter(BaseConverter):
    MODEL_TYPE = ModelType.BCE_Embedding

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        token_type: torch.Tensor = state_dict['embeddings.token_type_embeddings.weight']
        token_pos:  torch.Tensor = state_dict['embeddings.position_embeddings.weight']

        assert token_type.shape[0] == 1, f'shape of token_type_embeddings ({token_type.shape}) must be one row'

        state_dict['embeddings.position_embeddings.weight'] = token_pos + token_type

        return state_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'gelu', 'hidden_act must be gelu'
        assert config.type_vocab_size == 1, 'type_vocab_size must be 1'

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ['embeddings.word_embeddings.weight',
                        'embeddings.position_embeddings.weight',
                        'embeddings.LayerNorm.weight',
                        'embeddings.LayerNorm.bias',
                        ]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f'encoder.layer.{i}.attention.self.query.weight',
                f'encoder.layer.{i}.attention.self.query.bias',
                f'encoder.layer.{i}.attention.self.key.weight',
                f'encoder.layer.{i}.attention.self.key.bias',
                f'encoder.layer.{i}.attention.self.value.weight',
                f'encoder.layer.{i}.attention.self.value.bias',
                f'encoder.layer.{i}.attention.output.dense.weight',
                f'encoder.layer.{i}.attention.output.dense.bias',
                f'encoder.layer.{i}.attention.output.LayerNorm.weight',
                f'encoder.layer.{i}.attention.output.LayerNorm.bias',
                f'encoder.layer.{i}.intermediate.dense.weight',
                f'encoder.layer.{i}.intermediate.dense.bias',
                f'encoder.layer.{i}.output.dense.weight',
                f'encoder.layer.{i}.output.dense.bias',
                f'encoder.layer.{i}.output.LayerNorm.weight',
                f'encoder.layer.{i}.output.LayerNorm.bias',
            ]

        return weight_names

class XLMRobertaClassificationConverter(BaseConverter):
    MODEL_TYPE = ModelType.BCE_ReRanker

    @classmethod
    def state_dict_pp(cls, config, state_dict):

        new_dict = {}
        for k in state_dict:
            kk = k[8:] if k.startswith('roberta.') else k
            new_dict[kk] = state_dict[k]

        return XLMRobertaConverter.state_dict_pp(config, new_dict)

    @staticmethod
    def dump_config(f, config, ggml_type):
        XLMRobertaConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        weight_names = XLMRobertaConverter.get_weight_names(config)

        weight_names += [
            'classifier.dense.weight',
            'classifier.dense.bias',
            'classifier.out_proj.weight',
            'classifier.out_proj.bias',
        ]

        return weight_names


class GemmaConverter(BaseConverter):
    MODEL_TYPE = ModelType.Gemma

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name == 'model.embed_tokens.weight':
            return tensor * (config.hidden_size ** 0.5)
        elif name.endswith('input_layernorm.weight') or name.endswith('post_attention_layernorm.weight') or name == 'model.norm.weight':
            return 1 + tensor
        if name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        if name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        else:
            return LlamaConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.attention_bias == False, 'attention_bias == False'
        if config.hidden_activation is not None:
            assert config.hidden_activation == 'gelu_pytorch_tanh', "hidden_activation == 'gelu_pytorch_tanh'"
        assert config.rope_theta > 0, "rope_theta must be positive"
        assert config.rope_scaling is None, "rope_scaling must be `null`"

        # fake it for LlamaConverter
        config.hidden_act = 'silu'
        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            config.head_dim,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        r = LlamaConverter.get_weight_names(config)
        return r[:-1]

class Gemma2Converter(BaseConverter):
    MODEL_TYPE = ModelType.Gemma2
    FILE_VERSION = 2

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name == 'model.embed_tokens.weight':
            return tensor * (config.hidden_size ** 0.5)
        elif name.endswith('input_layernorm.weight') or \
            name.endswith('post_attention_layernorm.weight') or \
            name.endswith('post_feedforward_layernorm.weight') or \
            name.endswith('pre_feedforward_layernorm.weight') or \
            name == 'model.norm.weight':
            return 1 + tensor
        if name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        if name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        else:
            return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.attention_bias == False, 'attention_bias == False'
        if config.hidden_activation is not None:
            assert config.hidden_activation == 'gelu_pytorch_tanh', "hidden_activation == 'gelu_pytorch_tanh'"
        assert config.rope_theta > 0, "rope_theta must be positive"
        assert config.final_logit_softcapping > 0, "final_logit_softcapping must be positive"

        # fake it for LlamaConverter
        config.hidden_act = 'silu'
        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            config.head_dim,
            config.query_pre_attn_scalar,
            config.sliding_window
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta,
            config.final_logit_softcapping,
            config.attn_logit_softcapping,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_feedforward_layernorm.weight",
                f"model.layers.{i}.pre_feedforward_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        return weight_names

class Gemma3Converter(BaseConverter):
    MODEL_TYPE = ModelType.Gemma3
    FILE_VERSION = 1

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name == 'model.embed_tokens.weight':
            return tensor * (config.hidden_size ** 0.5)
        elif name.endswith('input_layernorm.weight') or \
            name.endswith('post_attention_layernorm.weight') or \
            name.endswith('post_feedforward_layernorm.weight') or \
            name.endswith('pre_feedforward_layernorm.weight') or \
            name.endswith('k_norm.weight') or \
            name.endswith('q_norm.weight') or \
            name == 'model.norm.weight':
            return 1 + tensor
        else:
            return tensor

    @classmethod
    def state_dict_pp(cls, config, state_dict):

        new_dict = {}
        for k in state_dict:
            kk: str = k
            if kk.startswith('language_model.'):
                kk = kk[len('language_model.'):]

            if kk.startswith('model.'):
                new_dict[kk] = Gemma3Converter.pp(config, kk, state_dict[k])
            else:
                kk = kk.replace('vision_tower.', '')
                kk = kk.replace('mm_input_projection_weight', 'mm_input_projection.weight')
                kk = kk.replace('.fc1.', '.fc0.')
                kk = kk.replace('.fc2.', '.fc1.')
                kk = kk.replace('.out_proj.', '.o_proj.')
                kk = kk.replace('.layer_norm1.', '.input_layernorm.')
                kk = kk.replace('.layer_norm2.', '.post_attention_layernorm.')
                if 'mm_input_projection.weight' in kk:
                    new_dict[kk] = torch.transpose(state_dict[k], 0, 1)
                elif 'mm_soft_emb_norm' in kk:
                    new_dict[kk] = 1.0 + state_dict[k]
                else:
                    new_dict[kk] = state_dict[k]

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):

        if config.text_config is not None:
            for k in config.text_config.keys():
                config[k] = config.text_config[k]

        if isinstance(config.eos_token_id, list):
            config.eos_token_id = config.eos_token_id[0]
        if config.bos_token_id is None:
            config.bos_token_id = 2

        if config.vocab_size is None:
            config.vocab_size=262_208
        if config.hidden_size is None:
            config.hidden_size = 2304
        if config.intermediate_size is None:
            config.intermediate_size = 9216
        if config.max_position_embeddings is None:
            config.max_position_embeddings = 131_072
        if config.num_attention_heads is None:
            config.num_attention_heads = 8
        if config.num_key_value_heads is None:
            config.num_key_value_heads = 4
        if config.head_dim is None:
            config.head_dim = 256
        if config.sliding_window_pattern is None:
            config.sliding_window_pattern = 6
        if config.query_pre_attn_scalar is None:
            config.query_pre_attn_scalar = 256
        if config.rope_local_base_freq is None:
            config.rope_local_base_freq = 10000.0
        if config.rope_theta is None:
            config.rope_theta = 1_000_000.0

        if config.vision_config is not None:
            cfg = AttributeDict(config.vision_config)
            if cfg.hidden_act is not None:
                assert cfg.hidden_act == 'gelu_pytorch_tanh'
            assert cfg.model_type == 'siglip_vision_model'

        assert (config.attention_bias is None) or (config.attention_bias == False), 'attention_bias must == False'
        assert (config.hidden_activation is None) or (config.hidden_activation == 'gelu_pytorch_tanh'), "hidden_activation must  == 'gelu_pytorch_tanh'"
        assert config.final_logit_softcapping is None, 'final_logit_softcapping must be null'
        assert config.attn_logit_softcapping is None, 'attn_logit_softcapping must be null'

        rope_scaling_factor = -1
        if config.rope_scaling is not None:
            assert config.rope_scaling['rope_type'] == 'linear'
            rope_scaling_factor = config.rope_scaling['factor']

        # fake it for LlamaConverter
        config.hidden_act = 'silu'
        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            config.head_dim,
            config.query_pre_attn_scalar,
            config.sliding_window,
            config.sliding_window_pattern,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        config_values = [
            config.rope_local_base_freq,
            config.rope_theta,
            rope_scaling_factor,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_feedforward_layernorm.weight",
                f"model.layers.{i}.pre_feedforward_layernorm.weight",
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        if config.vision_config is not None:
            cfg = AttributeDict(config.vision_config)
            weight_names += [
                "multi_modal_projector.mm_input_projection.weight",
                "multi_modal_projector.mm_soft_emb_norm.weight",
                "vision_model.embeddings.patch_embedding.bias",
                "vision_model.embeddings.patch_embedding.weight",
                "vision_model.embeddings.position_embedding.weight",
                "vision_model.post_layernorm.bias",
                "vision_model.post_layernorm.weight",
            ]

            for i in range(cfg.num_hidden_layers):
                weight_names += [
                    f"vision_model.encoder.layers.{i}.input_layernorm.bias",
                    f"vision_model.encoder.layers.{i}.input_layernorm.weight",
                    f"vision_model.encoder.layers.{i}.post_attention_layernorm.bias",
                    f"vision_model.encoder.layers.{i}.post_attention_layernorm.weight",
                    f"vision_model.encoder.layers.{i}.mlp.fc1.bias",
                    f"vision_model.encoder.layers.{i}.mlp.fc1.weight",
                    f"vision_model.encoder.layers.{i}.mlp.fc0.bias",
                    f"vision_model.encoder.layers.{i}.mlp.fc0.weight",
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias",
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight",
                    f"vision_model.encoder.layers.{i}.self_attn.o_proj.bias",
                    f"vision_model.encoder.layers.{i}.self_attn.o_proj.weight",
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias",
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight",
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias",
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight",
                ]

        return weight_names

class Grok1Converter(BaseConverter):
    MODEL_TYPE = ModelType.Grok1
    tensor_map = []
    file_to_name = {}
    experts = []

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            if name.endswith('embed_tokens.weight'):
                new_dict['model.embed_tokens.weight'] = tensor * config.embedding_multiplier_scale
            elif 'multi_head_attention' in name:
                old_name = name.replace('multi_head_attention', 'self_attn')
                if name.endswith('k_proj.weight'):
                    new_dict[old_name] = permute(tensor, config.num_key_value_heads)
                elif name.endswith('q_proj.weight'):
                    new_dict[old_name] = permute(tensor, config.num_attention_heads)
                else:
                    new_dict[old_name] = tensor
            elif 'experts' in name:
                new_dict[name] = tensor
            else:
                old_name = ''
                mapping = {
                    'language_model.norm.weight': 'model.norm.weight',
                    'rms_norm.weight': 'rms_norm.weight',
                    'rms_norm_1.weight': 'rms_norm_1.weight',
                    'rms_norm_2.weight': 'rms_norm_2.weight',
                    'rms_norm_3.weight': 'rms_norm_3.weight',
                    'router.weight': 'router.weight',
                }

                for k in mapping.keys():
                    if name.endswith(k):
                        old_name = name.replace(k, mapping[k])
                        break

                if old_name == '':
                    raise Exception(f'unhandled tensor {name}')

                new_dict[old_name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'gelu', "hidden_act == 'gelu'"

        config.hidden_act = 'silu'
        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            config.num_experts,
            config.num_selected_experts,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))
        f.write(struct.pack("<f", config.output_multiplier_scale))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            for j in range(config.num_experts):
                weight_names += [
                    f"model.layers.{i}.experts.{j}.w1.weight",
                    f"model.layers.{i}.experts.{j}.w2.weight",
                    f"model.layers.{i}.experts.{j}.w3.weight",
                ]

            weight_names += [
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.rms_norm.weight",
                f"model.layers.{i}.rms_norm_1.weight",
                f"model.layers.{i}.rms_norm_2.weight",
                f"model.layers.{i}.rms_norm_3.weight",
                f"model.layers.{i}.router.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        return weight_names

    @staticmethod
    def load_tensor_file(tensor_name, fn) -> Any:
        tensor_dict = {}
        new_dict = {}

        # copied from https://gist.github.com/chu-tianxiang/ec310e15d56949fd0f351cb5f65ee7a1
        def convert_weight(v):
            dtype = torch.float32
            if hasattr(v , 'scales'):
                weight = torch.from_numpy(np.asarray(v.weight).astype(np.float32)).to(dtype)
                scale =torch.from_numpy(np.asarray(v.scales).astype(np.float32)).to(dtype)
                # row parallel layers have sharded scale
                if len(scale.shape) >= 2 and scale.shape[-2] != 1:
                    scale = scale[..., None, :]
                    weight = weight.view(*weight.shape[:-2], 8, -1, weight.shape[-1])
                    weight = (weight * scale).view(*weight.shape[:-3], -1, weight.shape[-1])
                else:
                    weight = weight * scale
            else:
                weight = torch.from_numpy(np.asarray(v).astype(np.float32)).to(dtype)

            # Transpose linear matrix
            if len(weight.shape) >= 2 and 'embed_tokens.weight' not in tensor_name:
                weight = weight.transpose(-1, -2).contiguous()

            if tensor_name.endswith('router.weight'):
                new_dict[tensor_name] = weight[Grok1Converter.experts]
            elif 'experts' not in tensor_name:
                new_dict[tensor_name] = weight
            else:
                # split moe
                for i in range(len(Grok1Converter.experts)):
                    new_key_i = tensor_name.replace('experts', f'experts.{i}')
                    new_dict[new_key_i] = weight[Grok1Converter.experts[i]]

        with open(fn, 'rb') as f:
            r = pickle.load(f)
            tensor_dict[tensor_name] = r

        convert_weight(r)

        return new_dict

    @staticmethod
    def load_tensor_files(tensor_files) -> Dict:
        for (t, f) in tensor_files:
            print(f)
            yield Grok1Converter.load_tensor_file(t, f)

    @classmethod
    def convert(cls, config, model_files_path, vocab: Any, ggml_type, save_path):

        Grok1Converter.experts = config.experts

        map = ['language_model.embed_tokens.weight',
               'language_model.norm.weight']

        # caution: alphabet order must not be changed!
        for i in range(config.num_hidden_layers):
            map += [
                f"model.layers.{i}.experts.w1.weight",
                f"model.layers.{i}.experts.w2.weight",
                f"model.layers.{i}.experts.w3.weight",
                f"model.layers.{i}.multi_head_attention.k_proj.weight",
                f"model.layers.{i}.multi_head_attention.o_proj.weight",
                f"model.layers.{i}.multi_head_attention.q_proj.weight",
                f"model.layers.{i}.multi_head_attention.v_proj.weight",
                f"model.layers.{i}.rms_norm.weight",
                f"model.layers.{i}.rms_norm_1.weight",
                f"model.layers.{i}.rms_norm_2.weight",
                f"model.layers.{i}.rms_norm_3.weight",
                f"model.layers.{i}.router.weight",
            ]

        order = list(range(len(map)))
        order.sort(key=lambda i: map[i])

        for i in range(len(map)):
            idx = order.index(i)
            fn = model_files_path + f'/tensor{idx:05}_000'
            info = (map[i], fn)
            Grok1Converter.tensor_map.append(info)
            Grok1Converter.file_to_name[fn] = map[i]

        # convert all weights to fp16
        with open(save_path, "wb") as f:
            f.write(b"ggml")  # magic
            f.write(struct.pack("ii", cls.MODEL_TYPE.value, cls.FILE_VERSION))
            Grok1Converter.dump_config(f, config, ggml_type)
            vocab.write_vocab(f)

            weight_names = Grok1Converter.get_weight_names(config)
            dump_state_dict(f, weight_names, Grok1Converter.tensor_map, ggml_type, config, Grok1Converter.state_dict_pp, loader_fun=Grok1Converter.load_tensor_files)

        print(f"{Grok1Converter.MODEL_TYPE.name} GGML model saved to {save_path}")

class AlphaGeometryLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.AlphaGeometryLM
    tensor_map = []
    file_to_name = {}
    experts = []

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            if name == 'model.embed.embedding':
                new = 'model.embed_tokens.weight'
            elif name == 'model.final_layernorm.scale':
                new = 'model.norm.weight'
            else:
                new: str = name.replace('model.transformer', 'model.layers.')
                mapping = {
                    'relative_positions.rel_embedding': 'rel_embedding.weight',
                    'tbase._kvq.attention_scale': 'self_attn.attention_scale.weight',
                    'tbase._kvq.keys_layer.kernel': 'self_attn.k_proj.weight',
                    'tbase._kvq.pre_attn_layernorm.scale': 'input_layernorm.weight',
                    'tbase._kvq.queries_layer.kernel': 'self_attn.q_proj.weight',
                    'tbase._kvq.values_layer.kernel': 'self_attn.v_proj.weight',
                    'tbase.ffn.hidden0.kernel': 'mlp.hidden0.weight',
                    'tbase.ffn.output_layer.kernel': 'mlp.output_layer.weight',
                    'tbase.post_attn_mlp.output_layer.kernel': 'self_attn.o_proj.weight',
                    'tbase.pre_ffn_layernorm.scale': 'post_attention_layernorm.weight',
                }
                for k in mapping.keys():
                    if new.endswith(k):
                        new = new.replace(k, mapping[k])
                        break
            new_dict[new] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert ggml_type == GGMLType.F32, 'this model must converted with `-t f32`'

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.window_length,
            config.max_distance,
            config.num_buckets,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.rel_embedding.weight",
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.self_attn.attention_scale.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.mlp.hidden0.weight",
                f"model.layers.{i}.mlp.output_layer.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        return weight_names

    class _MsgpackExtType(IntEnum):
        """Messagepack custom type ids."""
        ndarray = 1
        native_complex = 2
        npscalar = 3

    def _dtype_from_name(name: str):
        """Handle JAX bfloat16 dtype correctly."""
        if name == b'bfloat16':
            return np.dtypes.Float16DType
        else:
            return np.dtype(name)

    _dict_to_tuple = lambda dct: tuple(dct[str(i)] for i in range(len(dct)))

    def _unchunk(data: dict[str, any]):
        """Convert canonical dictionary of chunked arrays back into array."""
        assert '__msgpack_chunked_array__' in data
        _dict_to_tuple = lambda dct: tuple(dct[str(i)] for i in range(len(dct)))
        shape = _dict_to_tuple(data['shape'])
        flatarr = np.concatenate(_dict_to_tuple(data['chunks']))
        return flatarr.reshape(shape)

    def convert_weight(v, tensor_name: str) -> torch.tensor:
        dtype = torch.float32
        weight = torch.from_numpy(np.asarray(v).astype(np.float32)).to(dtype)
        # Transpose linear matrix
        if len(weight.shape) >= 2 and 'embed.embedding' not in tensor_name:
            weight = weight.transpose(-1, -2).contiguous()
        return weight

    def extract_tensor_dict(d) -> dict:
        r = {}

        def visit(o, path: str):
            nonlocal r
            if isinstance(o, dict):
                for k in o.keys():
                    visit(o[k], path + '.' + k)
            elif isinstance(o, np.ndarray):
                r[path] = AlphaGeometryLMConverter.convert_weight(o, path)
            else:
                raise Exception(f'{path}: {type(o)}')

        visit(d, 'model')
        return r

    @staticmethod
    def load_tensor_file(tensor_name) -> Any:
        try:
            import msgpack
        except:
            raise Exception(f"Package `msgpack` is required. \n`pip install msgpack`")

        def _ndarray_from_bytes(data: bytes) -> np.ndarray:
            """Load ndarray from simple msgpack encoding."""
            shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
            return np.frombuffer(
                buffer, dtype=AlphaGeometryLMConverter._dtype_from_name(dtype_name), count=-1, offset=0
            ).reshape(shape, order='C')

        def _msgpack_ext_unpack(code, data):
            """Messagepack decoders for custom types."""
            if code == AlphaGeometryLMConverter._MsgpackExtType.ndarray:
                return _ndarray_from_bytes(data)
            elif code == AlphaGeometryLMConverter._MsgpackExtType.native_complex:
                complex_tuple = msgpack.unpackb(data)
                return complex(complex_tuple[0], complex_tuple[1])
            elif code == AlphaGeometryLMConverter._MsgpackExtType.npscalar:
                ar = _ndarray_from_bytes(data)
                return ar[()]  # unpack ndarray to scalar
            return msgpack.ExtType(code, data)

        def _unchunk_array_leaves_in_place(d):
            """Convert chunked array leaves back into array leaves, in place."""
            if isinstance(d, dict):
                if '__msgpack_chunked_array__' in d:
                    return AlphaGeometryLMConverter._unchunk(d)
                else:
                    for k, v in d.items():
                        if isinstance(v, dict) and '__msgpack_chunked_array__' in v:
                            d[k] = AlphaGeometryLMConverter._unchunk(v)
                        elif isinstance(v, dict):
                            _unchunk_array_leaves_in_place(v)
            return d

        with open(tensor_name, 'rb') as f:
            state_dict = msgpack.unpack(f, ext_hook=_msgpack_ext_unpack, raw=False)
        r = _unchunk_array_leaves_in_place(state_dict)
        return AlphaGeometryLMConverter.extract_tensor_dict(r['optimizer']['target']['decoder'])

    @staticmethod
    def load_tensor_files(tensor_files) -> Dict:
        assert len(tensor_files) == 1
        return [AlphaGeometryLMConverter.load_tensor_file(tensor_files[0])]

    @classmethod
    def convert(cls, config, model_files_path, vocab: Any, ggml_type, save_path):

        path = Path(model_files_path)
        files = list(path.glob('checkpoint_*'))
        model_files = [sorted(files)[-1]]

        # convert all weights to fp16
        with open(save_path, "wb") as f:
            f.write(b"ggml")  # magic
            f.write(struct.pack("ii", cls.MODEL_TYPE.value, cls.FILE_VERSION))
            AlphaGeometryLMConverter.dump_config(f, config, ggml_type)
            vocab.write_vocab(f)

            weight_names = AlphaGeometryLMConverter.get_weight_names(config)
            dump_state_dict(f, weight_names, model_files, ggml_type, config, AlphaGeometryLMConverter.state_dict_pp, loader_fun=AlphaGeometryLMConverter.load_tensor_files)

        print(f"{AlphaGeometryLMConverter.MODEL_TYPE.name} GGML model saved to {save_path}")

class ZhinaoConverter(BaseConverter):
    MODEL_TYPE = ModelType.Zhinao

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            if name.endswith('qkv_proj.weight'):

                kv_groups = config.num_attention_heads // config.num_key_value_heads
                head_dim = config.hidden_size // config.num_attention_heads
                gs = 2 + kv_groups

                v = tensor.view(config.num_key_value_heads, gs * head_dim, config.hidden_size)

                wq, wk, wv = torch.split(v, [kv_groups * head_dim, head_dim, head_dim], dim=1)

                new_dict[name.replace('qkv_proj.weight', 'q_proj.weight')] = wq.reshape(kv_groups * head_dim * config.num_key_value_heads, config.hidden_size)
                new_dict[name.replace('qkv_proj.weight', 'k_proj.weight')] = wk.reshape(head_dim * config.num_key_value_heads, config.hidden_size)
                new_dict[name.replace('qkv_proj.weight', 'v_proj.weight')] = wv.reshape(head_dim * config.num_key_value_heads, config.hidden_size)

            if name.endswith('qkv_proj.bias'):
                kv_groups = config.num_attention_heads // config.num_key_value_heads
                head_dim = config.hidden_size // config.num_attention_heads
                gs = 2 + kv_groups

                v = tensor.view(config.num_key_value_heads, gs * head_dim)

                bq, bk, bv = torch.split(v, [kv_groups * head_dim, head_dim, head_dim], dim=1)

                new_dict[name.replace('qkv_proj.bias', 'q_proj.bias')] = bq.contiguous().view(kv_groups * head_dim * config.num_key_value_heads)
                new_dict[name.replace('qkv_proj.bias', 'k_proj.bias')] = bk.contiguous().view(head_dim * config.num_key_value_heads)
                new_dict[name.replace('qkv_proj.bias', 'v_proj.bias')] = bv.contiguous().view(head_dim * config.num_key_value_heads)
            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.rope_scaling is None, "rope_scaling['type'] must be null"

        rope_theta = config.rope_theta

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", rope_theta))

    @staticmethod
    def get_weight_names(config):
        return QWen2Converter.get_weight_names(config)

class StarCoder2Converter(BaseConverter):
    MODEL_TYPE = ModelType.StarCoder2

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'gelu_pytorch_tanh', "hidden_act must be gelu_pytorch_tanh"
        assert config.use_bias, "hidden_act must be True"
        assert config.mlp_type == 'default', "mlp_type must be 'default'"

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
            config.sliding_window,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.input_layernorm.bias",
                f"model.layers.{i}.mlp.c_fc.weight",
                f"model.layers.{i}.mlp.c_fc.bias",
                f"model.layers.{i}.mlp.c_proj.weight",
                f"model.layers.{i}.mlp.c_proj.bias",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.bias",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.bias",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
            ]

        weight_names += [
            "model.norm.weight",
            "model.norm.bias",
        ]

        return weight_names

class DeepSeekV1Converter(BaseConverter):
    MODEL_TYPE = ModelType.DeepSeekV1

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        elif name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.attention_bias == False, "attention_bias must be False"
        assert config.rope_scaling is None, "rope_scaling must be null"
        assert config.scoring_func == 'softmax', "scoring_func must be 'softmax'"
        assert config.n_routed_experts is not None, "n_routed_experts must not be null"
        if config.head_dim is not None:
            assert config.head_dim == config.hidden_size // config.num_attention_heads

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
            config.first_k_dense_replace,
            config.moe_intermediate_size,
            config.moe_layer_freq,
            config.n_routed_experts,
            config.n_shared_experts,
            1 if config.norm_topk_prob else 0,
            config.num_experts_per_tok,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight",
                        "model.norm.weight",
                        "lm_head.weight"]
        for i in range(config.num_hidden_layers):

            weight_names += [
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
            ]

            if (config.n_routed_experts is not None
                and (i >= config.first_k_dense_replace)
                and (i % config.moe_layer_freq == 0)):
                weight_names += [
                    f"model.layers.{i}.mlp.gate.weight",
                    f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                    f"model.layers.{i}.mlp.shared_experts.up_proj.weight",
                    f"model.layers.{i}.mlp.shared_experts.down_proj.weight",
                ]
                for j in range(config.n_routed_experts):
                    weight_names += [
                        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                    ]
            else:
                weight_names += [
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                    f"model.layers.{i}.mlp.down_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
            ]

        return weight_names

class XverseMoEConverter(BaseConverter):
    MODEL_TYPE = ModelType.XVERSEMoE

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        r = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            if name.endswith('mlp.router.weight'):
                r[name.replace('.router.', '.gate.')] = tensor
            else:
                r[name] = DeepSeekV1Converter.pp(config, name, tensor)
        return r

    @staticmethod
    def dump_config(f, config, ggml_type):
        config.attention_bias = False
        config.scoring_func = 'softmax'
        config.num_key_value_heads = config.num_attention_heads
        config.first_k_dense_replace = 0
        config.moe_intermediate_size = config.intermediate_size
        config.moe_layer_freq = 1
        config.n_routed_experts = config.num_experts
        config.n_shared_experts = config.num_shared_experts
        config.norm_topk_prob = True
        config.num_experts_per_tok = config.moe_top_k

        DeepSeekV1Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return DeepSeekV1Converter.get_weight_names(config)

class BailingMoeConverter(BaseConverter):
    MODEL_TYPE = ModelType.BailingMoE

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        r = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            if name == 'model.word_embeddings.weight':
                r['model.embed_tokens.weight'] = tensor
            elif name == "lm_head.weight":
                tensor = tensor / (torch.norm(tensor, p=2, dim=0, keepdim=True) + 1e-7)
                r[name] = tensor
            elif name.endswith('query_key_value.weight'):
                head_dim = config.head_dim
                num_heads = config.num_attention_heads
                num_key_value_heads = config.num_key_value_heads

                q, k, v = tensor.split([num_heads * head_dim, num_key_value_heads * head_dim, num_key_value_heads * head_dim], dim=-2)

                r[name.replace('attention.query_key_value', 'self_attn.q_proj')] = permute(q, num_heads)
                r[name.replace('attention.query_key_value', 'self_attn.k_proj')] = permute(k, num_key_value_heads)
                r[name.replace('attention.query_key_value', 'self_attn.v_proj')] = v

            elif name.endswith('attention.dense.weight'):
                r[name.replace('attention.dense', 'self_attn.o_proj')] = tensor
            else:
                r[name] = tensor
        return r

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.norm_head
        if config.moe_layer_freq is None: config.moe_layer_freq = 1
        if config.attention_bias is None: config.attention_bias = False
        if config.scoring_func is None: config.scoring_func = 'softmax'
        config.n_routed_experts = config.num_experts
        config.n_shared_experts = config.num_shared_experts
        head_dim = config.head_dim or config.hidden_size // config.num_attention_heads

        # make DeepSeekV1Converter.dump_config happy
        if head_dim != config.hidden_size // config.num_attention_heads:
            config.head_dim = config.hidden_size // config.num_attention_heads
            assert config.moe_layer_freq == 1

        DeepSeekV1Converter.dump_config(f, config, ggml_type)

        config_values = [
            head_dim,
        ]
        f.write(struct.pack("<i", *config_values))

        config.head_dim = head_dim

    @staticmethod
    def get_weight_names(config):
        return DeepSeekV1Converter.get_weight_names(config)

class DeepSeekV2Converter(BaseConverter):
    MODEL_TYPE = ModelType.DeepSeekV2Light

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.endswith('kv_a_proj_with_mqa.weight'):

                w_d_kv, w_k_pe = torch.split(
                    tensor, [config.kv_lora_rank, config.qk_rope_head_dim])

                new_dict[name.replace('kv_a_proj_with_mqa.weight', 'd_kv_proj.weight')] = w_d_kv
                new_dict[name.replace('kv_a_proj_with_mqa.weight', 'k_pe_proj.weight')] = permute_pair(w_k_pe, 1)
            elif name.endswith('kv_a_layernorm.weight'):
                new_dict[name.replace('kv_a_layernorm.weight', 'kv_norm.weight')] = tensor
            elif name.endswith('kv_b_proj.weight'):

                v = tensor.reshape(config.num_attention_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank)
                u_k_nope_proj = v[:, :config.qk_nope_head_dim, :].reshape(config.num_attention_heads * config.qk_nope_head_dim, config.kv_lora_rank)
                u_v_proj = v[:, config.qk_nope_head_dim:, :].reshape(config.num_attention_heads * config.v_head_dim, config.kv_lora_rank)

                new_dict[name.replace('kv_b_proj.weight', 'u_k_nope_proj.weight')] = u_k_nope_proj
                new_dict[name.replace('kv_b_proj.weight', 'u_v_proj.weight')] = u_v_proj
            elif name.endswith('q_proj.weight'):
                new_dict[name] = permute_pair_3(tensor, config.num_attention_heads, config.qk_nope_head_dim)
            elif name.endswith('q_a_proj.weight'):
                new_dict[name.replace('q_a_proj.weight', 'd_q_proj.weight')] = tensor
            elif name.endswith('q_a_layernorm.weight'):
                new_dict[name.replace('q_a_layernorm.weight', 'q_norm.weight')] = tensor
            elif name.endswith('q_b_proj.weight'):
                new_dict[name.replace('q_b_proj.weight', 'u_q_proj.weight')] = permute_pair_3(tensor, config.num_attention_heads, config.qk_nope_head_dim)
            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.attention_bias == False, "attention_bias must be False"
        assert config.rope_scaling['type'] == 'yarn', "rope_scaling['type'] must be 'yarn'"
        assert config.scoring_func == 'softmax', "scoring_func must be 'softmax'"
        assert config.ep_size == 1, "ep_size must be 1"
        if config.q_lora_rank is None:
            assert config.topk_method == 'greedy', "topk_method must be 'greedy'"
        else:
            assert config.topk_method == 'group_limited_greedy', "topk_method must be 'group_limited_greedy'"
        assert config.n_routed_experts is not None, "n_routed_experts must not be null"

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
            config.first_k_dense_replace,
            config.kv_lora_rank,
            config.moe_intermediate_size,
            config.moe_layer_freq,
            config.n_group,
            config.n_routed_experts,
            config.n_shared_experts,
            1 if config.norm_topk_prob else 0,
            config.num_experts_per_tok,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
            config.rope_scaling['original_max_position_embeddings'],
            config.v_head_dim,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        config_values = [
            config.rope_scaling['beta_fast'],
            config.rope_scaling['beta_slow'],
            config.rope_scaling['factor'],
            config.rope_scaling['mscale'],
            config.rope_scaling['mscale_all_dim'],
            config.rope_theta,
            config.routed_scaling_factor,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

        if config.q_lora_rank is not None:
            config_values = [
                config.q_lora_rank,
                config.topk_group if config.topk_group is not None else 1,
            ]
            f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight",
                        "model.norm.weight",
                        "lm_head.weight"]
        for i in range(config.num_hidden_layers):

            if config.q_lora_rank is None:
                weight_names += [
                    f"model.layers.{i}.self_attn.q_proj.weight",
                ]
            else:
                weight_names += [
                    f"model.layers.{i}.self_attn.d_q_proj.weight",
                    f"model.layers.{i}.self_attn.q_norm.weight",
                    f"model.layers.{i}.self_attn.u_q_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.self_attn.d_kv_proj.weight",
                f"model.layers.{i}.self_attn.k_pe_proj.weight",
                f"model.layers.{i}.self_attn.kv_norm.weight",
                f"model.layers.{i}.self_attn.u_k_nope_proj.weight",
                f"model.layers.{i}.self_attn.u_v_proj.weight",

                f"model.layers.{i}.self_attn.o_proj.weight",
            ]

            if (config.n_routed_experts is not None
                and (i >= config.first_k_dense_replace)
                and (i % config.moe_layer_freq == 0)):
                weight_names += [
                    f"model.layers.{i}.mlp.gate.weight",
                    f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                    f"model.layers.{i}.mlp.shared_experts.up_proj.weight",
                    f"model.layers.{i}.mlp.shared_experts.down_proj.weight",
                ]
                for j in range(config.n_routed_experts):
                    weight_names += [
                        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                    ]
            else:
                weight_names += [
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                    f"model.layers.{i}.mlp.down_proj.weight",
                ]

            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
            ]

        return weight_names

class DeepSeekV3Converter(BaseConverter):
    MODEL_TYPE = ModelType.DeepSeekV3

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        return DeepSeekV2Converter.state_dict_pp(config, state_dict)

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == 'silu', "hidden_act must be silu"
        assert config.attention_bias == False, "attention_bias must be False"
        assert config.ep_size == 1, "ep_size must be 1"
        if config.rope_scaling is not None:
            assert config.rope_scaling['type'] == 'yarn', "rope_scaling['type'] must be 'yarn'"
        assert config.scoring_func == 'sigmoid', "scoring_func must be 'sigmoid'"
        assert config.topk_method == 'noaux_tc', "topk_method must be 'noaux_tc'"
        assert config.n_routed_experts is not None, "n_routed_experts must not be null"
        has_scaling = config.rope_scaling is not None

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if isinstance(config.eos_token_id, int) else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.num_key_value_heads,
            config.first_k_dense_replace,
            config.kv_lora_rank,
            config.moe_intermediate_size,
            config.moe_layer_freq,
            config.n_group,
            config.n_routed_experts,
            config.n_shared_experts,
            1 if config.norm_topk_prob else 0,
            config.num_experts_per_tok,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
            config.rope_scaling['original_max_position_embeddings'] if has_scaling else -1,
            config.v_head_dim,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

        config_values = [
            config.rope_scaling['beta_fast'] if has_scaling else 0.0,
            config.rope_scaling['beta_slow'] if has_scaling else 0.0,
            config.rope_scaling['factor'] if has_scaling else 0.0,
            config.rope_scaling['mscale'] if has_scaling else 0.0,
            config.rope_scaling['mscale_all_dim'] if has_scaling else 0.0,
            config.rope_theta,
            config.routed_scaling_factor,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

        if config.q_lora_rank is not None:
            config_values = [
                config.q_lora_rank,
                config.topk_group if config.topk_group is not None else 1,
            ]
            f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = DeepSeekV2Converter.get_weight_names(config)

        for i in range(config.num_hidden_layers):
            if (config.n_routed_experts is not None
                and (i >= config.first_k_dense_replace)
                and (i % config.moe_layer_freq == 0)):
                weight_names += [
                    f"model.layers.{i}.mlp.gate.e_score_correction_bias",
                ]

        return weight_names

class ERNIEMoEConverter(BaseConverter):
    MODEL_TYPE = ModelType.ERNIE_MoE

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert not config.use_bias
        assert len(config.moe_capacity) == 3
        if config.rope_scaling is not None:
            assert config.rope_scaling == 1.0, 'rope_scaling must equal to 1.0'

        dump_llama_like_config(f, config, ggml_type)
        config_values = [
            config.num_key_value_heads,
            1 if config.tie_word_embeddings else 0,
            config.moe_num_experts,
            config.moe_num_shared_experts,
            config.moe_layer_start_index,
            config.moe_intermediate_size,
            config.moe_capacity[0],
            config.moe_capacity[1],
            config.moe_capacity[2],
            config.moe_k,
            config.moe_layer_interval,
            1 if config.moe_use_aux_free else 0,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
            ]

            if (i >= config.moe_layer_start_index) and ((i + 1) % config.moe_layer_interval == 0):
                weight_names += [
                    f"model.layers.{i}.mlp.gate.weight",
                    f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                    f"model.layers.{i}.mlp.shared_experts.up_proj.weight",
                    f"model.layers.{i}.mlp.shared_experts.down_proj.weight",
                ]
                if config.moe_use_aux_free:
                    weight_names += [
                        f"model.layers.{i}.mlp.moe_statics.e_score_correction_bias",
                    ]
                for j in range(config.moe_num_experts):
                    weight_names += [
                        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                    ]
            else:
                weight_names += [
                    f"model.layers.{i}.mlp.down_proj.weight",
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                ]

        weight_names += [
            "model.norm.weight",
        ]

        if not config.tie_word_embeddings:
            weight_names += [
                "lm_head.weight"
            ]
        return weight_names

class KimiVLConverter(BaseConverter):
    MODEL_TYPE = ModelType.KimiVL

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        r = {}
        txt_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.startswith('language_model.'):
                txt_dict[name.replace('language_model.', '')] = tensor
            elif name.startswith('vision_tower.encoder.'):

                name = name.replace('vision_tower.encoder.', 'vision_model.encoder.')
                if '.wo.' in name:
                    r[name.replace('.wo.', '.attn.o_proj.')] = tensor
                elif name.endswith('.wqkv.bias') or name.endswith('.wqkv.weight'):
                    #print(f'shape: {name} = {tensor.shape}')
                    num_heads = config.vision_config['hidden_size']
                    q, k, v = tensor.split([num_heads, num_heads, num_heads], dim=0)
                    r[name.replace('.wqkv.', '.attn.q_proj.')] = q
                    r[name.replace('.wqkv.', '.attn.k_proj.')] = k
                    r[name.replace('.wqkv.', '.attn.v_proj.')] = v
                elif '.final_layernorm.' in name:
                    txt_dict[name.replace('encoder.', '')] = tensor
                else:
                    r[name] = tensor
            elif name.startswith('vision_tower.'):
                txt_dict[name.replace('vision_tower.', 'vision_model.')] = tensor
            else:
                r[name] = tensor

        r.update(DeepSeekV3Converter.state_dict_pp(KimiVLConverter.txt_config, txt_dict))

        return r

    @staticmethod
    def dump_config(f, config, ggml_type):
        KimiVLConverter.txt_config = AttributeDict(config.text_config)
        DeepSeekV3Converter.dump_config(f, KimiVLConverter.txt_config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        weight_names = DeepSeekV3Converter.get_weight_names(KimiVLConverter.txt_config)

        for i in range(config.vision_config['num_hidden_layers']):
            weight_names += [
                f"vision_model.encoder.blocks.{i}.attn.q_proj.bias",
                f"vision_model.encoder.blocks.{i}.attn.q_proj.weight",
                f"vision_model.encoder.blocks.{i}.attn.k_proj.bias",
                f"vision_model.encoder.blocks.{i}.attn.k_proj.weight",
                f"vision_model.encoder.blocks.{i}.attn.v_proj.bias",
                f"vision_model.encoder.blocks.{i}.attn.v_proj.weight",
                f"vision_model.encoder.blocks.{i}.attn.o_proj.bias",
                f"vision_model.encoder.blocks.{i}.attn.o_proj.weight",
                f"vision_model.encoder.blocks.{i}.mlp.fc0.bias",
                f"vision_model.encoder.blocks.{i}.mlp.fc0.weight",
                f"vision_model.encoder.blocks.{i}.mlp.fc1.bias",
                f"vision_model.encoder.blocks.{i}.mlp.fc1.weight",
                f"vision_model.encoder.blocks.{i}.norm0.bias",
                f"vision_model.encoder.blocks.{i}.norm0.weight",
                f"vision_model.encoder.blocks.{i}.norm1.bias",
                f"vision_model.encoder.blocks.{i}.norm1.weight",
            ]

        weight_names += [
            "multi_modal_projector.linear_1.bias",
            "multi_modal_projector.linear_1.weight",
            "multi_modal_projector.linear_2.bias",
            "multi_modal_projector.linear_2.weight",
            "multi_modal_projector.pre_norm.bias",
            "multi_modal_projector.pre_norm.weight",
            "vision_model.final_layernorm.bias",
            "vision_model.final_layernorm.weight",
            "vision_model.patch_embed.pos_emb.weight",
            "vision_model.patch_embed.proj.bias",
            "vision_model.patch_embed.proj.weight",
        ]

        return weight_names

class IndexConverter(BaseConverter):
    MODEL_TYPE = ModelType.Index

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name == 'lm_head.weight':
            return nn.Parameter(nn.functional.normalize(tensor)) if config.norm_head else 0
        else:
            return Llama3Converter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        config.rope_theta = 10000.0
        if config.rope_ratio is not None:
            assert isinstance(config.rope_ratio, int) or isinstance(config.rope_ratio, float), "`rope_ratio` must be a number"
            assert config.rope_scaling is None, "rope_scaling must be null"
            config.rope_theta *= config.rope_ratio
        Llama3Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return Llama3Converter.get_weight_names(config)

class HunYuanDenseConverter(BaseConverter):
    MODEL_TYPE = ModelType.HunYuanDense

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.endswith('mlp.gate_and_up_proj.weight'):

                new_dict[name.replace('gate_and_up_proj.weight', 'gate_proj.weight')] = part(tensor, 1, 2)
                new_dict[name.replace('gate_and_up_proj.weight', 'up_proj.weight')]   = part(tensor, 0, 2)
            if name.endswith('.qkv_proj.weight'):

                kv_groups = config.num_attention_heads // config.num_key_value_heads
                head_dim = config.hidden_size // config.num_attention_heads
                gs = 2 + kv_groups
                h = tensor.shape[0] // (gs * head_dim)

                v = tensor.view(h, gs, head_dim, config.hidden_size)

                wq = v[:, 0:kv_groups, ...].reshape(h * kv_groups * head_dim, config.hidden_size)
                wk = v[:, -2,          ...].reshape(h * 1         * head_dim, config.hidden_size)
                wv = v[:, -1,          ...].reshape(h * 1         * head_dim, config.hidden_size)

                new_dict[name.replace('qkv_proj.weight', 'q_proj.weight')] = wq
                new_dict[name.replace('qkv_proj.weight', 'k_proj.weight')] = wk
                new_dict[name.replace('qkv_proj.weight', 'v_proj.weight')] = wv
            else:
                new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.tie_word_embeddings, "tie_word_embeddings must be True"
        assert config.attention_bias == False, "attention_bias must be False"
        assert config.mlp_bias == False, "mlp_bias must be False"
        assert not config.use_cla, "use_cla must be False"
        assert not config.use_mla, "use_mla must be False"
        assert config.rope_scaling['type'] == 'dynamic', "rope_scaling['type'] must be 'dynamic'"
        assert config.use_qk_norm, "use_qk_norm must be True"
        assert config.rope_scaling['alpha'] > 0, "rope_scaling['alpha'] must be > 0"

        assert not (((isinstance(config.num_experts, int) and config.num_experts > 1) or
            (isinstance(config.num_experts, list) and max(config.num_experts) > 1))), "MoE not allowed"

        dump_llama_like_config(f, config, ggml_type)

        head_dim = config.attention_head_dim
        config.rope_theta = config.rope_theta * config.rope_scaling['alpha'] ** (head_dim / (head_dim - 2))

        config_values = [
            config.num_key_value_heads,
            config.rope_theta,
        ]
        f.write(struct.pack("if", *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.key_layernorm.weight",
                f"model.layers.{i}.self_attn.query_layernorm.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        return weight_names

class HunYuanMoEV1Converter(BaseConverter):
    MODEL_TYPE = ModelType.HunYuanMoEV1

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]
            new_name = name
            new_name = new_name.replace('.mlp.gate.wg.', '.mlp.gate.')
            new_name = new_name.replace('.shared_mlp.', '.shared_expert.')

            new_dict[new_name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.tie_word_embeddings, "tie_word_embeddings must be True"
        assert config.attention_bias == False, "attention_bias must be False"
        assert config.mlp_bias == False, "mlp_bias must be False"
        assert not config.use_cla, "use_cla must be False"
        assert not config.use_mla, "use_mla must be False"
        assert config.rope_scaling['type'] == 'dynamic', "rope_scaling['type'] must be 'dynamic'"
        assert config.use_qk_norm, "use_qk_norm must be True"
        assert config.rope_scaling['alpha'] > 0, "rope_scaling['alpha'] must be > 0"
        assert config.moe_layer_num_skipped == 0
        assert config.use_mixed_mlp_moe
        assert len(set(config.moe_intermediate_size)) == 1
        assert len(set(config.moe_topk)) == 1
        assert len(set(config.num_shared_expert)) == 1
        assert config.attention_head_dim == config.hidden_size / config.num_attention_heads

        head_dim = config.attention_head_dim
        config.rope_theta = config.rope_theta * config.rope_scaling['alpha'] ** (head_dim / (head_dim - 2))

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.num_experts,

            list(set(config.moe_intermediate_size))[0],
            list(set(config.moe_topk))[0],
            list(set(config.num_shared_expert))[0],
        ]
        f.write(struct.pack("<" + "i" * len(config_values), *config_values))

        config_values = [
            config.rope_theta,
        ]
        f.write(struct.pack("<" + "f" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            for j in range(config.num_experts):
                    weight_names += [
                        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                    ]

            weight_names += [
                f"model.layers.{i}.mlp.gate.weight",
                f"model.layers.{i}.mlp.shared_expert.down_proj.weight",
                f"model.layers.{i}.mlp.shared_expert.gate_proj.weight",
                f"model.layers.{i}.mlp.shared_expert.up_proj.weight",
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.key_layernorm.weight",
                f"model.layers.{i}.self_attn.query_layernorm.weight",
            ]

        weight_names += [
            "model.norm.weight",
        ]

        return weight_names

class SolarConverter(BaseConverter):
    MODEL_TYPE = ModelType.SolarPro

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        elif name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):

        MAX_LEN = 20

        assert config.rope_scaling is None, "rope_scaling must be null"
        assert len(config.bskcn_1) == len(config.bskcn_3)
        assert len(config.bskcn_2) == len(config.bskcn_4)
        assert len(config.bskcn_tv) == 2

        forwarding_ids = list(zip(config.bskcn_1, config.bskcn_3)) + list(zip(config.bskcn_2, config.bskcn_4))
        pairs = len(forwarding_ids)
        assert pairs <= MAX_LEN
        forwarding_ids = forwarding_ids + [(-1, -1)] * (MAX_LEN - pairs)
        # flatten(forwarding_ids)
        forwarding_ids = list(sum(forwarding_ids, ()))

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.sliding_window,
            pairs,
        ] + forwarding_ids
        f.write(struct.pack("i" * len(config_values), *config_values))

        float_values = [
            config.rope_theta,
            config.bskcn_tv[1],
        ]
        f.write(struct.pack("<" + "f" * len(float_values), *float_values))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class AquilaConverter(BaseConverter):
    MODEL_TYPE = ModelType.Aquila2

    @classmethod
    def pp(cls, config, name: str, tensor):
        return Llama31Converter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        if config.num_key_value_heads is None:
            config.num_key_value_heads = config.num_attention_heads
        if config.rope_theta is None:
            config.rope_theta = 10000.0
        rope_scaling_factor = -1
        if config.rope_scaling is not None:
            assert config.rope_scaling['rope_type'] == 'linear'
            rope_scaling_factor = config.rope_scaling['factor']

        dump_llama_like_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.rope_theta,
            rope_scaling_factor
        ]
        f.write(struct.pack("<iff", *config_values))

    @staticmethod
    def get_weight_names(config):
        return Llama32Converter.get_weight_names(config)

class OrpheusTTSConverter(BaseConverter):
    MODEL_TYPE = ModelType.OrpheusTTS

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.startswith('encoder'): continue

            if name.startswith('decoder'):
                if name.endswith('.weight_g'): continue

                new_name = 'snac.' + name
                if name.endswith('.alpha'):
                    shape = tensor.shape
                    assert (len(shape) == 3) and (shape[0] == 1) and (shape[2] == 1)
                    new_dict[new_name] = tensor.reshape(shape[1])
                elif name.endswith('.bias'):
                    new_dict[new_name] = tensor
                elif name.endswith('.weight_v'):
                    g = state_dict[name.replace('.weight_v', '.weight_g')]
                    new_name = new_name.replace('.weight_v', '.weight')
                    g = _weight_norm(tensor, g)
                    g = g.transpose(1, 2) # g: [out_dim, in_dim / groups, kernel_size]
                    new_dict[new_name] = g
                else:
                    raise Exception(f"unknown name: {name}")

            elif name.startswith('quantizer'):
                if name.endswith('.weight_g'): continue
                new_name = name.replace('quantizer.quantizers.', 'snac.quantizer.strides.')
                if name.endswith('.weight_v'):
                    g = state_dict[name.replace('.weight_v', '.weight_g')]
                    new_name = new_name.replace('.weight_v', '.weight')
                    g = _weight_norm(tensor, g)
                    g = g.transpose(1, 2) # g: [out_dim, in_dim / groups, kernel_size]
                    new_dict[new_name] = g
                else:
                    new_dict[new_name] = tensor
            else:
                new_dict[name] = Llama32Converter.pp(config, name, tensor)

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.snac_model['sampling_rate'] == 24000, "sampling_rate must be 24000"
        assert config.snac_model['depthwise']
        assert config.snac_model['noise']
        assert config.snac_model['attn_window_size'] is None

        print(f"WARNING: encoder not supported")

        Llama32Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_conv1d(prefix: str, bias: bool = True):
        weights = [
            f"{prefix}.bias",
            f"{prefix}.weight",
        ]
        if not bias: weights.pop(0)
        return weights

    @staticmethod
    def get_snake(prefix: str):
        return [
            f"{prefix}.alpha",
        ]

    @staticmethod
    def get_residual(prefix: str):
        weights = OrpheusTTSConverter.get_snake(prefix   + '.block.layers.0') + \
                  OrpheusTTSConverter.get_conv1d(prefix + '.block.layers.1') + \
                  OrpheusTTSConverter.get_snake(prefix   + '.block.layers.2') + \
                  OrpheusTTSConverter.get_conv1d(prefix + '.block.layers.3')
        return weights

    @staticmethod
    def get_block(prefix: str):
        weights = OrpheusTTSConverter.get_snake(prefix   + '.block.layers.0') + \
                  OrpheusTTSConverter.get_conv1d(prefix + '.block.layers.1') + \
                  OrpheusTTSConverter.get_conv1d(prefix + '.block.layers.2.linear', False)
        for i in range(3):
            weights.extend(OrpheusTTSConverter.get_residual(f"{prefix}.block.layers.{i + 3}"))
        return weights

    @staticmethod
    def get_vq(prefix: str):
        weights = OrpheusTTSConverter.get_conv1d(prefix + '.in_proj') + \
                  OrpheusTTSConverter.get_conv1d(prefix + '.out_proj')
        weights += [f"{prefix}.codebook.weight"]

        return weights

    @staticmethod
    def get_weight_names(config):
        weights = Llama32Converter.get_weight_names(config)

        snac_config = config.snac_model

        snac_weights = []
        layer_id = 1
        if config.snac_model['depthwise']:
            snac_weights += OrpheusTTSConverter.get_conv1d('snac.decoder.model.layers.0')
            snac_weights += OrpheusTTSConverter.get_conv1d('snac.decoder.model.layers.1')
            layer_id = 2
        else:
            snac_weights += OrpheusTTSConverter.get_conv1d('snac.decoder.model.layers.0')

        for i, _ in enumerate(snac_config['decoder_rates']):
            snac_weights += OrpheusTTSConverter.get_block(f"snac.decoder.model.layers.{layer_id}")
            layer_id += 1

        snac_weights += OrpheusTTSConverter.get_snake(f'snac.decoder.model.layers.{layer_id}')
        layer_id += 1
        snac_weights += OrpheusTTSConverter.get_conv1d(f'snac.decoder.model.layers.{layer_id}')
        layer_id += 1

        for i, _stride in enumerate(snac_config['vq_strides']):
            snac_weights += OrpheusTTSConverter.get_vq(f'snac.quantizer.strides.{i}')

        return weights + snac_weights

class OuteTTSConverter(BaseConverter):
    MODEL_TYPE = ModelType.OuteTTSLlaMA
    IsQwen3 = False # a shortcut

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}

        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            if name.startswith('decoder') or name.startswith('encoder'):
                if name.endswith('.weight_g'): continue

                new_name = 'dac.' + name.replace('.model.', '.model.layers.')
                new_name = new_name.replace('.block.', '.block.layers.')
                if name.endswith('.alpha'):
                    shape = tensor.shape
                    assert (len(shape) == 3) and (shape[0] == 1) and (shape[2] == 1)
                    new_dict[new_name] = tensor.reshape(shape[1])
                elif name.endswith('.bias'):
                    new_dict[new_name] = tensor
                elif name.endswith('.weight_v'):
                    g = state_dict[name.replace('.weight_v', '.weight_g')]
                    new_name = new_name.replace('.weight_v', '.weight')
                    g = _weight_norm(tensor, g)
                    new_dict[new_name] = g
                else:
                    raise Exception(f"unknown name: {name}")

            elif name.startswith('quantizer'):
                if name.endswith('.weight_g'): continue
                new_name = name.replace('quantizer.quantizers.', 'dac.quantizer.strides.')
                if name.endswith('.weight_v'):
                    g = state_dict[name.replace('.weight_v', '.weight_g')]
                    new_name = new_name.replace('.weight_v', '.weight')
                    g = _weight_norm(tensor, g)
                    new_dict[new_name] = g
                else:
                    new_dict[new_name] = tensor
            else:
                if not OuteTTSConverter.IsQwen3:
                    new_dict[name] = Llama32Converter.pp(config, name, tensor)
                else:
                    new_dict[name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.dac_model['sampling_rate'] == 24000, "sampling_rate must be 24000"

        print(f"WARNING: encoder not supported")

        if OuteTTSConverter.IsQwen3:
            QWen3Converter.dump_config(f, config, ggml_type)
        else:
            Llama32Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        if OuteTTSConverter.IsQwen3:
            weights = QWen3Converter.get_weight_names(config)
        else:
            weights = Llama32Converter.get_weight_names(config)

        dac_config = config.dac_model

        def get_block(prefix: str):
            weights = OrpheusTTSConverter.get_snake(prefix  + '.block.layers.0') + \
                      OrpheusTTSConverter.get_conv1d(prefix + '.block.layers.1')
            for i in range(3):
                weights.extend(OrpheusTTSConverter.get_residual(f"{prefix}.block.layers.{i + 2}"))
            return weights

        dac_weights = []
        layer_id = 1
        dac_weights += OrpheusTTSConverter.get_conv1d('dac.decoder.model.layers.0')

        for i, _ in enumerate(dac_config['upsampling_ratios']):
            dac_weights += get_block(f"dac.decoder.model.layers.{layer_id}")
            layer_id += 1

        dac_weights += OrpheusTTSConverter.get_snake(f'dac.decoder.model.layers.{layer_id}')
        layer_id += 1
        dac_weights += OrpheusTTSConverter.get_conv1d(f'dac.decoder.model.layers.{layer_id}')
        layer_id += 1

        for i in range(dac_config['n_codebooks']):
            dac_weights += OrpheusTTSConverter.get_vq(f'dac.quantizer.strides.{i}')

        return weights + dac_weights

def convert_grok_1_base(args, vocab, ggml_type):
    def ffn_size(emb_size, widening_factor):
        _ffn_size = int(widening_factor * emb_size) * 2 // 3
        _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
        return _ffn_size

    grok1_config = {
        'vocab_size': 128 * 1024,
        'hidden_act': 'gelu',
        'pad_token_id': 0,
        'eos_token_id': 2,
        'max_position_embeddings': 8192,
        'output_multiplier_scale': 0.5773502691896257,
        'embedding_multiplier_scale': 78.38367176906169,
        'hidden_size': 48 * 128,
        'intermediate_size': -1,
        'num_attention_heads': 48,
        'num_key_value_heads': 8,
        'num_hidden_layers': 64,
        'num_selected_experts': 2,
        'rope_theta': 10000,
        'attn_output_multiplier': 0.08838834764831845,
    }

    grok1_config['intermediate_size'] = ffn_size(grok1_config['hidden_size'], 8)

    grok1_config['experts'] = list(range(8))
    if args.experts != '':
        grok1_config['experts'] = [int(x, 0) for x in args.experts.split(',')]

    grok1_config['num_experts'] = len(grok1_config['experts'])

    if grok1_config['num_experts'] < 2:
        raise Exception(f"at least 2 experts")

    print(f"experts to export: {grok1_config['experts']}")

    Grok1Converter.convert(AttributeDict(grok1_config), args.model_name_or_path, vocab, ggml_type, args.save_path)
    return

def convert_alphageometry_lm(args, vocab, ggml_type):
    alphageo_config = {
        'vocab_size': 1024,
        'hidden_act': 'gelu',
        'pad_token_id': 0,
        'eos_token_id': 1,
        'bos_token_id': 2,
        'hidden_size': 1024,
        'intermediate_size': 4096,
        'num_attention_heads': 8,
        'num_hidden_layers': 12,
        'max_position_embeddings': 1024,
        'window_length': 1024,
        'max_distance': 128,
        'num_buckets': 32,
    }

    AlphaGeometryLMConverter.convert(AttributeDict(alphageo_config), args.model_name_or_path, vocab, ggml_type, args.save_path)
    return

def load_vocab(path: Path, skip_def_model_file: bool = False) -> Any:

    def load_spm(p: Path) -> Any:
        added_tokens_path = p.parent / "added_tokens.json"
        print(f"Loading vocab file {p}")
        return SentencePieceVocab(p, added_tokens_path if added_tokens_path.exists() else None)

    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        if not skip_def_model_file:
            path2 = path / "tokenizer.model"
            # Use `.parent` instead of /.. to handle the symlink case better.
            path3 = path.parent / "tokenizer.model"
            path4 = path / "ice_text.model"

            if path2.exists():
                return load_spm(path2)
            elif path3.exists():
                return load_spm(path3)
            elif path4.exists():
                return load_spm(path4)

        # path20 = path / "sentencepiece.bpe.model"

        path5 = path / "geometry.757.vocab"
        if path5.exists():
            return SimpleVocab(path5)

        path5 = path / "qwen.tiktoken"
        if path5.exists():
            return TikTokenizerVocab(path, None)
        path5 = path / "tokenizer.tiktoken"
        if path5.exists():
            return TikTokenizerVocab(path, None)

        path4 = path / "tokenizer.json"
        if path4.exists():
            print(f"Loading vocab file {path}")
            added_tokens_path = path.parent / "added_tokens.json"
            model = json.load(open(path4, encoding='utf-8'))
            if model['model']['type'] == 'BPE':
                return FastTokenizerVocab(path, added_tokens_path if added_tokens_path.exists() else None)
            elif model['model']['type'] == 'Unigram':
                return UnigramTokenizerJsonVocab(path, added_tokens_path if added_tokens_path.exists() else None)
            else:
                raise Exception(f"unsupported tokenizer model type {model['model']['type']}")

        path6 = path / "tokenizer_config.json"
        path7 = path / 'vocab.json'
        path8 = path / 'merges.txt'
        if path6.exists():
             config = json.load(open(path6, encoding='utf-8'))
             if (config['tokenizer_class'] == 'GPT2Tokenizer') and path7.exists() and path8.exists():
                 return GenericBPETokenizerVocab(path7, path8)

        path9 = path / 'vocab' / "360.tiktoken"
        if path9.exists():
            return load_vocab_from_tiktok_mergeable_ranks(path9)
        path9 = path / "hy.tiktoken"
        if path9.exists():
            return load_vocab_from_tiktok_mergeable_ranks(path9)
        path9 = path / "tiktoken.model"
        if path9.exists():
            return load_vocab_from_tiktok_mergeable_ranks(path9)

    raise FileNotFoundError(
        f"Could not find tokenizer.model in {path} or its parent; "
        "if it's in another directory, pass the directory as --vocab-dir")

def load_config(path: Path, config_fn: str) -> Any:
    with open(path / config_fn, 'r', encoding='utf-8') as fp:
        r = json.load(fp)
    return r

def load_some_info(r: dict, path: Path, prefix: str = '') -> None:
    if path.is_dir():
        globs = ["config.json", "configuration.json", "special_tokens_map.json",
                 "tokenizer_config.json", "preprocessor_config.json"]
        for glob in globs:
            files = list(path.glob(glob))
            for f in files:
                r[prefix + os.path.basename(f.name)] = load_config(path, f.name)

def load_some_model(path: Path, fallback_files: list[Path] = []) -> List[Path]:
    '''Load a model of any supported format.'''
    # Be extra-friendly and accept either a file or a directory:
    if path.is_dir():
        globs = ["model-*-of-*.safetensors", "consolidated.*.pth",
                 "pytorch_model-*-of-*.bin", "pytorch_model_*-of-*.bin",
                 "pytorch_model.bin", "model.safetensors", "*.pt", "consolidated.safetensors", "consolidated.pth"]
        for glob in globs:
            files = list(path.glob(glob))
            if len(files) > 0:
                files.sort()
                return files
        if len(fallback_files) > 0: return fallback_files
        raise Exception('can not find model files, or unsupported format.')
    else:
        return [path]

def main():
    global g_lora

    parser = argparse.ArgumentParser("chatllm-convert")
    parser.add_argument("-i", "--model_name_or_path", type=str)
    parser.add_argument("-a", "--arch", type=str, default='')
    parser.add_argument("-l", "--lora_model_name_or_path", type=str, default=None)
    parser.add_argument("-o", "--save_path", type=Path)
    parser.add_argument("-t", "--type", type=str, default="q8_0", choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q4_k"])
    parser.add_argument("-n", "--name", type=str, required=True, help='model name in English')
    parser.add_argument("--vocab_dir", type=str, default='')
    parser.add_argument("--experts", type=str, default='')
    parser.add_argument("--native_name", type=str, default='', help='model native name')
    parser.add_argument("--snac_model", type=str, default='', help='snac model path (required by Orpheus-TTS, etc)')
    parser.add_argument("--dac_model", type=str, default='', help='dac model path (required by Oute-TTS, etc)')
    args = parser.parse_args()

    arch = args.arch.lower()

    if args.lora_model_name_or_path is not None:
        g_lora = LoRAState(Path(args.lora_model_name_or_path), False)

    ggml_type = GGMLType[args.type.upper()]

    skip_def_vocab_model = False
    if args.arch.lower() == 'codeqwen':
        skip_def_vocab_model = True
        args.arch = ''

    config_fn = 'config.json'
    if args.arch.lower() == 'llama-multi-token-prediction-ckpt':
        config_fn = 'params.json'

    if (Path(args.model_name_or_path) / config_fn).exists():
        config = AttributeDict(load_config(Path(args.model_name_or_path), config_fn))
    else:
        config = AttributeDict({})

    if arch == '':
        if len(config.architectures) != 1:
            raise Exception(f'unknown architectures: {config.architectures}')
        arch = config.architectures[0]

    vocab_dir = Path(args.model_name_or_path) if args.vocab_dir == '' else Path(args.vocab_dir)
    tokenizer_model_file_exists = False

    if config._name_or_path in ['THUDM/glm-4-9b-chat', 'THUDM/glm4-9b-chat', 'THUDM/codegeex4-all-9b']:
        vocab = load_vocab_from_tiktok_mergeable_ranks(vocab_dir / 'tokenizer.model')
    else:
        tokenizer_model_file_exists = (vocab_dir / 'tokenizer.model').exists()
        vocab = load_vocab(vocab_dir, skip_def_vocab_model)

    if args.arch.lower() == 'grok-1-base':
        convert_grok_1_base(args, vocab, ggml_type)
        return
    elif args.arch.lower() == 'alphageometry-lm':
        convert_alphageometry_lm(args, vocab, ggml_type)
        return

    model_files = load_some_model(Path(args.model_name_or_path))

    global g_model_meta
    load_some_info(g_model_meta, Path(args.model_name_or_path))

    if arch == 'orpheus-tts':
        if args.snac_model == '':
            raise Exception('snac_model (`--snac_model`) is required for Orpheus-TTS')
        load_some_info(g_model_meta, Path(args.snac_model), 'snac_')
        model_files += load_some_model(Path(args.snac_model))
        config.snac_model = g_model_meta['snac_config.json']
    elif arch == 'outetts':
        if args.dac_model == '':
            raise Exception('dac_model (`--dac_model`) is required for Oute-TTS')
        load_some_info(g_model_meta, Path(args.dac_model), 'dac_')
        model_files += [Path(args.dac_model) / "weights_24khz_1.5kbps_v1.0.pth"]
        g_model_meta['dac_config.json']['n_codebooks'] = 2
        config.dac_model = g_model_meta['dac_config.json']

    if args.arch != '':
        g_model_meta['model_name'] = args.arch
    if args.name != '':
        g_model_meta['model_name'] = args.name
    if args.native_name != '':
        g_model_meta['model_native_name'] = args.native_name

    #if args.lora_model_name_or_path is not None:
    #    from peft import PeftModel
    #
    #    model = PeftModel.from_pretrained(model, args.lora_model_name_or_path)
    #    model = model.merge_and_unload()

    if arch == 'ChatGLMModel':
        if config.multi_query_attention is not None:
            if config.rope_ratio is not None:
                ChatGLM4Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
            else:
                auto_map = config.auto_map
                if 'AutoModelForCausalLM' in auto_map:
                    name: str = config._name_or_path
                    if name.find('codegeex') > 0:
                        ChatGLM2Converter.MODEL_TYPE = ModelType.CODEGEEX2
                    else:
                        ChatGLM2Converter.MODEL_TYPE = ModelType.CHATGLM3

                ChatGLM2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            ChatGLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'codegeex4':
        ChatGLM4Converter.MODEL_TYPE = ModelType.CODEGEEX4
        ChatGLM4Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'characterglm':
        CharacterGLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Glm4ForCausalLM':
        GLM4Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'InternLMForCausalLM':
        if not config.bias:
            InternLMConverter.MODEL_TYPE = ModelType.InternLM2
        InternLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'InternLM2ForCausalLM':
        InternLM2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'InternLM3ForCausalLM':
        InternLM3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'LlamaForCausalLM':
        if isinstance(config.rope_scaling, dict):
            if config.tie_word_embeddings:
                Llama32Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
            else:
                Llama31Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
            return

        if tokenizer_model_file_exists:
            if ((config.num_key_value_heads is None) or (config.num_key_value_heads == config.num_attention_heads)) \
                and ((config.rope_theta is None) or (config.rope_theta == 10000.0)):
                LlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
            else:
                Llama3Converter.MODEL_TYPE = ModelType.LlaMA2Plus
                Llama3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            Llama3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Llama4ForConditionalGeneration':
        Llama4Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseek-r1-distill-llama':
        assert isinstance(config.rope_scaling, dict), 'rope_scaling must be a dict'
        Llama32Converter.MODEL_TYPE = ModelType.DeepSeek_R1_Distill_LlaMA
        Llama32Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Llama-3-Groq-8B-Tool-Use'.lower():
        Llama3Converter.MODEL_TYPE = ModelType.GroqToolUse
        Llama3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'llama-multi-token-prediction-ckpt':
        LlamaMultiConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'megrez':
        assert config.rope_scaling is None, 'config.rope_scaling must be null'
        assert not config.tie_word_embeddings, 'config.tie_word_embeddings must be false'
        Llama3Converter.MODEL_TYPE = ModelType.Megrez
        Llama3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'falcon3':
        assert config.rope_scaling is None, 'config.rope_scaling must be null'
        assert not config.tie_word_embeddings, 'config.tie_word_embeddings must be false'
        Llama3Converter.MODEL_TYPE = ModelType.Falcon3
        Llama3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'smollm':
        SmolLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'SmolLM3ForCausalLM':
        SmolLM3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'SmolVLMForConditionalGeneration':
        SmolVLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'XverseForCausalLM':
        if config.num_experts is None:
            LlamaConverter.MODEL_TYPE = ModelType.XVERSE
            LlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            XverseMoEConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'codellama':
        CodeLlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseek':
        DeepSeekConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'numinamath':
        DeepSeekConverter.MODEL_TYPE = ModelType.NuminaMath
        DeepSeekConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseekcoder':
        DeepSeekCoderConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'codefusedeepseek':
        CodeFuseDeepSeekCoderConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'BaichuanForCausalLM':
        if config.num_hidden_layers <= 32:
            BaiChuanLlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            config.max_position_embeddings = config.model_max_length
            BaiChuanConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'BaichuanM1ForCausalLM':
        BaiChuanM1Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'yi':
        YiConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'map-neo':
        YiConverter.MODEL_TYPE = ModelType.MAP_Neo
        YiConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'PhiForCausalLM':
        if config.hidden_act is not None:
            Phi2Converter.MODEL_TYPE = ModelType.Phi2_v2
        Phi2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Phi3ForCausalLM':
        if config._name_or_path == 'Phi-4-mini-instruct':
            assert not ('long_mscale' in config.rope_scaling)
            Phi3SUConverter.MODEL_TYPE = ModelType.Phi4_Mini
            Phi3SUConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
            return
        if config.rope_scaling is None:
            if config.sliding_window is None:
                Phi4Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
            else:
                Phi3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            if config.rope_scaling['type'] == 'su':
                Phi3SUConverter.MODEL_TYPE = ModelType.Phi3_ScalingSU
                config.rope_scaling['type'] = 'longrope'

            if config.rope_scaling['type'] == 'longrope':
                if 'long_mscale' in config.rope_scaling:
                    Phi3SU3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
                else:
                    Phi3SUConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
            else:
                raise Exception(config.rope_scaling['type'])
    elif arch == 'PhiMoEForCausalLM':
        Phi3MoESUConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'dolphinphi2':
        Phi2Converter.MODEL_TYPE = ModelType.DolphinPhi2_v2 if config.hidden_act is not None else ModelType.DolphinPhi2
        Phi2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'wizardcoder':
        WizardCoderConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'wizardlm':
        WizardLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'wizardmath':
        WizardMathConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'MistralForCausalLM':
        custom_head_dim = False
        if config.head_dim is not None:
            if config.head_dim != config.hidden_size // config.num_attention_heads:
                custom_head_dim = True

        if (g_tokenizer_type == TokenizerType.BPE2) or custom_head_dim:
            Mistral2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            MistralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'mistral-small-3.1':
        MistralSmall31Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'MixtralForCausalLM':
        if args.experts != '':
            config.experts = sorted([int(x, 0) for x in args.experts.split(',')])
            config.num_local_experts = len(config.experts)
        else:
            config.experts = range(config.num_local_experts)
        MixtralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'wizardlm-2-moe':
        MixtralConverter.MODEL_TYPE = ModelType.WizardLMMoE
        if args.experts != '':
            config.experts = sorted([int(x, 0) for x in args.experts.split(',')])
            config.num_local_experts = len(config.experts)
        else:
            config.experts = range(config.num_local_experts)
        MixtralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'openchat':
        MistralConverter.MODEL_TYPE = ModelType.OpenChat
        MistralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'starling':
        MistralConverter.MODEL_TYPE = ModelType.Starling
        MistralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif (arch == 'QWenLMHeadModel') or (arch == 'qwen-qanything') or (arch == 'qwenqanything'):
        if config.hidden_size is None:
            config.hidden_size = config.n_embd
        if config.num_attention_heads is None:
            config.num_attention_heads = config.n_head
        if config.num_hidden_layers is None:
            config.num_hidden_layers = config.n_layer
        if config.max_position_embeddings is None:
            config.max_position_embeddings = config.n_positions
        if config.intermediate_size is None:
            config.intermediate_size = config.ffn_hidden_size
        QWenConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif (arch == 'Qwen2ForCausalLM') or (arch == 'marco-o1') or (arch == 'qwq') or (arch == 'readerlm-v2') or (arch == 'MiMoForCausalLM'):
        if arch == 'MiMoForCausalLM':
            print("WARNING: MTP is not supported")

        if (config.tie_word_embeddings is not None) and config.tie_word_embeddings:
            QWen2Converter.MODEL_TYPE = ModelType.QWen2Tie
        if arch == 'marco-o1':
            QWen2Converter.MODEL_TYPE = ModelType.MarcoO1
        if arch == 'qwq':
            QWen2Converter.MODEL_TYPE = ModelType.QwQ
        if arch == 'readerlm-v2':
            QWen2Converter.MODEL_TYPE = ModelType.ReaderLM2
            assert config.tie_word_embeddings
        QWen2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Qwen2AudioForConditionalGeneration':
        QWen2AudioConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Qwen2_5_VLForConditionalGeneration':
        QWen2_5VLConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'KimiVLForConditionalGeneration':
        KimiVLConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseek-r1-distill-qwen':
        QWen2TieConverter.MODEL_TYPE = ModelType.DeepSeek_R1_Distill_QWen
        QWen2TieConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Qwen2MoeForCausalLM':
        QWen2MoEConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'tigerbot':
        TigerBotConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'BlueLMForCausalLM':
        BlueLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'StableLMEpochForCausalLM':
        StableLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'XLMRobertaModel':
        XLMRobertaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'bge-m3':
        XLMRobertaConverter.MODEL_TYPE = ModelType.BGE_M3
        XLMRobertaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'XLMRobertaForSequenceClassification':
        XLMRobertaClassificationConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'bge-reranker-m3':
        XLMRobertaClassificationConverter.MODEL_TYPE = ModelType.BGE_ReRankerM3
        XLMRobertaClassificationConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'neuralbeagle':
        MistralConverter.MODEL_TYPE = ModelType.NeuralBeagle
        MistralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'OrionForCausalLM':
        OrionConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'MiniCPMForCausalLM':
        if config.num_experts is None:
            if (config.rope_scaling is not None) and ('rope_type' in config.rope_scaling) and (config.rope_scaling['rope_type'] == 'longrope'):
                MiniCPM4Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
            else:
                if (config.tie_word_embeddings is not None) and (not config.tie_word_embeddings):
                    MiniCPMConverter.MODEL_TYPE = ModelType.MiniCPM2
                MiniCPMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            MiniCPMMoEConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'MiniCPM3ForCausalLM':
        MiniCPM3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'MiniCPMO':
        MiniCPMOConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'MiniCPMModel':
        assert config._name_or_path == "openbmb/UltraRAG-Embedding", "only support openbmb/UltraRAG-Embedding"
        MiniCPMEmbConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'MiniCPMForSequenceClassification':
        assert config._name_or_path == "OpenBMB/UltraRAG-Reranker", "only support OpenBMB/UltraRAG-Reranker"
        MiniCPMReRankerConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'PersimmonForCausalLM':
        PersimmonConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'FuyuForCausalLM':
        FuyuConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'GemmaForCausalLM':
        GemmaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Gemma2ForCausalLM':
        Gemma2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Gemma3ForCausalLM':
        Gemma3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Gemma3ForConditionalGeneration':
        if config.vision_config is not None:
            Gemma3Converter.MODEL_TYPE = ModelType.Gemma3Vis
        Gemma3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'CohereForCausalLM':
        CohereCommandConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Cohere2ForCausalLM':
        Cohere2CommandConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'aya-23':
        CohereCommandConverter.MODEL_TYPE = ModelType.CohereAya23
        CohereCommandConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'ZhinaoForCausalLM':
        ZhinaoConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Starcoder2ForCausalLM':
        StarCoder2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'DeepseekV2ForCausalLM':
        if config.q_lora_rank is not None:
            DeepSeekV2Converter.MODEL_TYPE = ModelType.DeepSeekV2
        DeepSeekV2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'moonlight':
        DeepSeekV3Converter.MODEL_TYPE = ModelType.MoonLight
        DeepSeekV3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'DeepseekForCausalLM':
        DeepSeekV1Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'gigachat':
        DeepSeekV1Converter.MODEL_TYPE = ModelType.GigaChat
        DeepSeekV1Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'IndexForCausalLM':
        IndexConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'OlmoeForCausalLM':
        OLMoEConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Olmo2ForCausalLM':
        OLMo2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'GraniteForCausalLM':
        GraniteConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'GraniteMoeForCausalLM':
        GraniteMoEConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'ExaoneForCausalLM':
        ExaoneConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Exaone4ForCausalLM':
        Exaone4Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'TeleChat2ForCausalLM':
        TeleChat2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'HunYuanForCausalLM':
        if ((isinstance(config.num_experts, int) and config.num_experts > 1) or
            (isinstance(config.num_experts, list) and max(config.num_experts) > 1)):
            raise Exception('HunYuanForCausalLM: only dense model is supported')
        HunYuanDenseConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'HunYuanMoEV1ForCausalLM':
        HunYuanMoEV1Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'InstellaForCausalLM':
        InstellaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'DeciLMForCausalLM':
        DeciLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'SolarForCausalLM':
        SolarConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'AquilaForCausalLM':
        AquilaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'BailingMoeForCausalLM':
        BailingMoeConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'AprielForCausalLM':
        AprielConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch in ['Qwen3MoeForCausalLM', 'Qwen3ForCausalLM']:
        QWen3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Ernie4_5_ForCausalLM':
        ERNIEDenseConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'Ernie4_5_MoeForCausalLM':
        ERNIEMoEConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'PanguProMoEForCausalLM':
        PanguMoEConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseek-r1-distill-qwen3':
        QWen3Converter.MODEL_TYPE = ModelType.DeepSeek_R1_Distill_QWen3
        QWen3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'qwen3-embedding':
        QWen3EmbConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'qwen3-reranker':
        QWen3Converter.MODEL_TYPE = ModelType.QWen3_ReRanker
        QWen3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'reka-flash-3':
        assert config.rope_scaling is None, 'config.rope_scaling must be null'
        assert not config.tie_word_embeddings, 'config.tie_word_embeddings must be false'
        Llama3Converter.MODEL_TYPE = ModelType.RekaFlash3
        Llama3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deephermes-3-mistral':
        Mistral2Converter.MODEL_TYPE = ModelType.DeepHermes3Mistral
        Mistral2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'orpheus-tts':
        OrpheusTTSConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'outetts':
        if config.architectures[0] == 'Qwen3ForCausalLM':
            OuteTTSConverter.MODEL_TYPE = ModelType.OuteTTSQwen3
            OuteTTSConverter.IsQwen3 = True
        OuteTTSConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    else:
        raise Exception(f'unknown model_type: {arch}')

def detect_layer_index(tensor_name: str) -> int:
    import re
    r = re.findall('PhiForCausalLM_transformer_PhiModel_h_ModuleList_([0-9]*)_', tensor_name)
    return int(r[0]) if len(r) == 1 else -1

class DumpModule(nn.Module):

    def __init__(self, dumped_module, dump_path="./dump/"):
        super().__init__()
        self.dumped_module = dumped_module
        self.dump_reg_hooks = []
        self.dump_path = dump_path

    def forward(self, inputs):
        return self.dumped_module(inputs)

    def save_tensor_to_text(self, tensor, file_name):
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        if tensor.device != torch.device("cpu"):
            tensor = tensor.to("cpu")
        tensor_shape_str = '-'.join([str(i) for i in tensor.shape])
        save_name = file_name + "_" + tensor_shape_str + ".npz"

        tensor = tensor.detach().numpy()
        np.savetxt(file_name, tensor.reshape(-1))
        np.savez(save_name, input=tensor)
        # print("dump ", save_name, " success")

    def dump_layer_output(self, module, inputs, outputs):
            filename = self.dump_path + module.module_name
            self.save_tensor_to_text(outputs, filename)

    def reg_dump_hook(self, module_name, module):
        if isinstance(module, nn.Module):
            tmp_name = module_name + '_' + module._get_name()

            layer_index = detect_layer_index(tmp_name)
            if (layer_index >= 0) and (layer_index != 0):
                return

            module.__setattr__("module_name", tmp_name)

            tmp_hook = module.register_forward_hook(self.dump_layer_output)
            self.dump_reg_hooks.append(tmp_hook)

            for name, mm in module.named_children():
                tmp2_name = tmp_name + "_" + name
                self.reg_dump_hook(tmp2_name, mm)

    def init_dump(self):
        for name, mm in self.dumped_module.named_children():
            tmp_name = self.dumped_module._get_name() + "_" + name
            self.reg_dump_hook(tmp_name, mm)

def test(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "hi"
    inputs = tokenizer(prompt, return_tensors="pt")
    print(inputs.input_ids)

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    dumpModule = DumpModule(dumped_module=model)
    dumpModule.init_dump()

    generate_ids = model.generate(inputs.input_ids, max_length=2, do_sample=False)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

if __name__ == "__main__":
    main()
