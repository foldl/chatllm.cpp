"""
Convert Hugging Face models to GGML format
"""
import argparse
from ast import Dict, Tuple
from collections import OrderedDict
import json
import struct
import sys
import io
import pickle
import re
from pathlib import Path
from enum import Enum
from pathlib import Path
from typing import IO, Any, Iterable, List, Optional, Tuple
import numpy as np
import math
from attr import dataclass

import torch
from torch import nn
from tabulate import tabulate
from tqdm import tqdm

from sentencepiece import SentencePieceProcessor  # type: ignore

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32

GGML_MEM_ALIGN = 16


class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 8


class ModelType(Enum):
    CHATGLM = 1
    CHATGLM2 = 2
    CHATGLM3 = 3
    CODEGEEX2 = 4
    CharacterGLM = 5

    InternLM   = 0x100
    InternLM2   = 0x101
    InternLM3   = 0x102

    LlaMA2      = 0x150
    CodeLlaMA   = 0x151
    WizardCoder = 0x152
    WizardLM    = 0x153
    WizardMath  = 0x154
    TigerBot    = 0x155

    BaiChuanLlama = 0x200
    BaiChuan = 0x201

    DeepSeek            = 0x300
    DeepSeekCoder       = 0x301
    CodeFuseDeepSeek    = 0x302
    DeepSeekV2Light     = 0x320
    DeepSeekV2          = 0x321

    Yi = 0x400
    MAP_Neo = 0x401

    Phi2                = 0x500
    Phi2_v2             = 0x501
    DolphinPhi2         = 0x510
    DolphinPhi2_v2      = 0x511
    Phi3                = 0x520
    Phi3_ScalingSU      = 0x521

    Mistral = 0x600
    Mixtral = 0x601
    OpenChat = 0x602
    NeuralBeagle = 0x603
    Starling = 0x604
    WizardLMMoE = 0x605
    Codestral = 0x606

    QWen    = 0x700
    QWen2   = 0x710
    QWen2MoE = 0x750

    BlueLM  = 0x800

    StableLM    = 0x900

    Orion       = 0x1000

    MiniCPM     = 0x1100
    MiniCPM2    = 0x1101   # updated chat template, no tie_word_embeddings=False
    MiniCPM_MoE = 0x1102

    Persimmon   = 0x1200
    Fuyu        = 0x1201

    Gemma       = 0x1300

    CohereCommand = 0x1400
    CohereAya23   = 0x1401

    Grok1         = 0x1500

    Zhinao        = 0x1600

    LlaMA3        = 0x1700

    StarCoder2    = 0x1800

    BCE_Embedding = 0x10000100
    BCE_ReRanker  = 0x10000101
    BGE_M3        = 0x10000102
    BGE_ReRankerM3 = 0x10000103


class TokenType(Enum):
    UNDEFINED    = 0
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6

g_special_tokens: Dict = {}

def quantize_q8_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q8_0 in ggml.c
    assert tensor.shape[1] % GGML_QK8_0 == 0
    tensor = tensor.view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
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
    assert tensor.shape[1] % GGML_QK4_1 == 0
    tensor = tensor.view(-1, GGML_QK4_1)
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


def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GGMLType):
    assert tensor.dtype == torch.float32

    # tensor name
    f.write(struct.pack("i", len(name.encode())))
    f.write(name.encode())

    # tensor shape & dtype
    f.write(struct.pack("i" * (2 + tensor.ndim), tensor.ndim, *tensor.shape, ggml_type.value))

    # tensor data
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
    else:
        raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")

    # align address
    aligned_pos = (f.tell() + (GGML_MEM_ALIGN - 1)) // GGML_MEM_ALIGN * GGML_MEM_ALIGN
    f.seek(aligned_pos)
    tensor.numpy().tofile(f)

def load_model_file(path: Path) -> Dict:
    print(f"loading {path} ...")
    fp = open(path, 'rb')
    first8 = fp.read(8)
    fp.seek(0)
    if first8[:2] == b'PK':
        return torch.load(fp, map_location=torch.device('cpu')).items()
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

def dump_state_dict(f, weight_names, model_files, ggml_type, config, state_dict_pp, loader_fun = None):
    tensor_info = []
    converted_names = []

    # Note: incremental loading and converting

    state_dict_cache = {}
    remaining: List = weight_names.copy()

    if loader_fun is None:
        loader_fun = load_all_model_files

    for state_dict in loader_fun(model_files):
        this_round = {}
        state_dict = state_dict_pp(config, state_dict)

        for x in remaining:
            if x in state_dict:
                this_round[x] = state_dict[x]
                del state_dict[x]
            elif x in state_dict_cache:
                this_round[x] = state_dict_cache[x]
                del state_dict_cache[x]
            else:
                break

        remaining = remaining[len(this_round):]

        for x in state_dict:
            state_dict_cache[x] = state_dict[x]

        for name in tqdm(this_round.keys(), desc="Dumping ..."):
            tensor: torch.Tensor = this_round[name]

            tensor = tensor.float()

            if tensor.ndim == 2:
                if tensor.shape[1] % GGML_QK8_0 == 0:
                    tensor_ggml_type = ggml_type
                else:
                    tensor_ggml_type = GGMLType.F16
            else:
                # 1d weight: convert it to float32
                assert tensor.ndim == 1
                tensor_ggml_type = GGMLType.F32

            dump_tensor(f, name, tensor, tensor_ggml_type)
            tensor_info.append((name, tensor.shape, tensor_ggml_type.name))

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"], tablefmt="psql"))

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

def s_to_bytes(s):
    print([ord(c) for c in s])
    return bytes([ord(c) for c in s])

class UnigramTokenizerJsonVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:
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
        print(len(vocab_tokens))
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

        # convert all weights to fp16
        with open(save_path, "wb") as f:
            f.write(b"ggml")  # magic
            f.write(struct.pack("ii", cls.MODEL_TYPE.value, cls.FILE_VERSION))  # model type & version
            cls.dump_config(f, config, ggml_type)
            vocab.write_vocab(f)

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
    MODEL_TYPE = ModelType.InternLM3

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
                id = int(re.findall("block_sparse_moe\.experts\.([0-9]+)\.", name)[0])
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

class ChatGLM3Converter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM3

    @staticmethod
    def dump_config(f, config, ggml_type):
        ChatGLM2Converter.dump_config(f, config, ggml_type)

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

class CodeGeeX2Converter(BaseConverter):
    MODEL_TYPE = ModelType.CODEGEEX2

    @staticmethod
    def dump_config(f, config, ggml_type):
        ChatGLM2Converter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return ChatGLM2Converter.get_weight_names(config)

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

                new_dict[name.replace('qkv_proj.weight', 'q_proj.weight')] = permute(wq, config.num_attention_heads)
                new_dict[name.replace('qkv_proj.weight', 'k_proj.weight')] = permute(wk, config.num_key_value_heads)
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

class Phi3SUConverter(BaseConverter):
    MODEL_TYPE = ModelType.Phi3_ScalingSU

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        return Phi3Converter.state_dict_pp(config, state_dict)

    @staticmethod
    def dump_config(f, config, ggml_type):

        dump_llama_like_config(f, config, ggml_type)

        MAX_FACTOR_LEN = 128

        assert config.rope_scaling['type'] == "su", "rope_scaling must be null or `su`"
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
        return Phi3Converter.get_weight_names(config)

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
            config.sliding_window,
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
            "lm_head.weight"
        ]

        return weight_names

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

@dataclass
class QuantizedWeight8bit:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        import jnp.array

        self.weight: jnp.array
        self.scales: jnp.array

    @property
    def shape(self):
        return self.weight.shape

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

class CodestralConverter(BaseConverter):
    MODEL_TYPE = ModelType.Codestral

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        return tensor

    @classmethod
    def state_dict_pp(cls, config, state_dict):
        new_dict = {}
        for name in state_dict:
            tensor: torch.Tensor = state_dict[name]

            new_name: str = name

            if new_name.startswith('layers.'):
                new_name = 'model.' + new_name

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

                for k in mapping.keys():
                    if new_name.endswith(k):
                        new_name = new_name.replace(k, mapping[k])
                        break

                new_dict[new_name] = MistralConverter.pp(config, new_name, tensor)

            else:
                mapping = {
                    'tok_embeddings.weight': 'model.embed_tokens.weight',
                    'norm.weight': 'model.norm.weight',
                    'output.weight': 'lm_head.weight',
                }

                new_name = mapping[new_name]
                new_dict[new_name] = tensor

        return new_dict

    @staticmethod
    def dump_config(f, config, ggml_type):
        MistralConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

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
            path20 = path / "tokenizer.model.v3"
            # Use `.parent` instead of /.. to handle the symlink case better.
            path3 = path.parent / "tokenizer.model"

            if path2.exists():
                return load_spm(path2)
            elif path3.exists():
                return load_spm(path3)
            elif path20.exists():
                return load_spm(path20)

        # path20 = path / "sentencepiece.bpe.model"

        path5 = path / "qwen.tiktoken"
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
            import base64
            import tiktoken

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

            #encode = tiktoken.Encoding(
                #"zhinao",
                #pat_str=PAT_STR,
                #mergeable_ranks=mergeable_ranks,
                #special_tokens={}
            #)

            #decoder = {v: k for k, v in mergeable_ranks.items()}
            #vocab = {decoder[i]: i for i in range(len(mergeable_ranks))}
            ##vocab.update(self.added_tokens_encoder)
            #print(vocab)

            return TikTokenizerVocab(TempTokenizer(mergeable_ranks), fname_added_tokens = None)


    raise FileNotFoundError(
        f"Could not find tokenizer.model in {path} or its parent; "
        "if it's in another directory, pass the directory as --vocab-dir")

def load_config(path: Path) -> Any:
    with open(path / 'config.json', 'r') as fp:
        r = json.load(fp)
    if (path / 'generation_config.json').is_file():
        with open(path / 'generation_config.json', 'r') as fp:
            rr = json.load(fp)
            for k in rr:
                r[k] = rr[k]
    return r

def load_some_model(path: Path) -> List[Path]:
    '''Load a model of any supported format.'''
    # Be extra-friendly and accept either a file or a directory:
    if path.is_dir():
        globs = ["model-*-of-*.safetensors", "consolidated.*.pth", "pytorch_model-*-of-*.bin",
                 "pytorch_model.bin", "model.safetensors", "*.pt", "consolidated.safetensors"]
        for glob in globs:
            files = list(path.glob(glob))
            if len(files) > 0:
                files.sort()
                return files
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
    parser.add_argument("-t", "--type", type=str, default="q8_0", choices=["f32", "f16", "q8_0", "q4_0", "q4_1"])
    parser.add_argument("--vocab_dir", type=str, default='')
    parser.add_argument("--experts", type=str, default='')
    args = parser.parse_args()

    arch = args.arch.lower()

    if args.lora_model_name_or_path is not None:
        g_lora = LoRAState(Path(args.lora_model_name_or_path), False)

    ggml_type = GGMLType[args.type.upper()]

    skip_def_vocab_model = False
    if args.arch.lower() == 'codeqwen':
        skip_def_vocab_model = True
        args.arch = ''

    vocab = load_vocab(Path(args.model_name_or_path) if args.vocab_dir == '' else Path(args.vocab_dir), skip_def_vocab_model)

    if args.arch.lower() == 'codestral':
        with open(Path(args.model_name_or_path) / 'params.json', 'r') as fp:
             param = AttributeDict(json.load(fp))
        config = {
            "architectures": ['MistralForCausalLM'],
            "hidden_act": 'silu',
            "vocab_size": param.vocab_size,
            "hidden_size": param.dim,
            "num_attention_heads": param.n_heads,
            "num_hidden_layers": param.n_layers,
            "intermediate_size": param.hidden_dim,
            "max_position_embeddings": 8192,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": -1,
            "sep_token_id": -1,
            "num_key_value_heads": param.n_kv_heads,
            "sliding_window": -1,
            "rope_theta": param.rope_theta,
        }
        config = AttributeDict(config)
    else:
        config = AttributeDict(load_config(Path(args.model_name_or_path)))

    if len(config.architectures) != 1:
        raise Exception(f'unknown architectures: {config.architectures}')

    if arch == '':
        arch = config.architectures[0]

    if args.arch.lower() == 'grok-1-base':
        convert_grok_1_base(args, vocab, ggml_type)
        return

    model_files = load_some_model(Path(args.model_name_or_path))

    #if args.lora_model_name_or_path is not None:
    #    from peft import PeftModel
    #
    #    model = PeftModel.from_pretrained(model, args.lora_model_name_or_path)
    #    model = model.merge_and_unload()

    if arch == 'ChatGLMModel':
        if hasattr(config, "multi_query_attention"):
            auto_map = config.auto_map
            if 'AutoModelForCausalLM' in auto_map:
                name: str = config._name_or_path
                if name.find('codegeex') > 0:
                    CodeGeeX2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
                else:
                    ChatGLM3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
            else:
                ChatGLM2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            ChatGLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'characterglm':
        CharacterGLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'InternLMForCausalLM':
        if not config.bias:
            InternLMConverter.MODEL_TYPE = ModelType.InternLM2
        InternLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'InternLM2ForCausalLM':
        InternLM2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'LlamaForCausalLM':
        if (config.num_key_value_heads is None) or (config.num_key_value_heads == config.num_attention_heads):
            LlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            Llama3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'codellama':
        CodeLlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseek':
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
        if config.rope_scaling is None:
            Phi3Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            if config.rope_scaling['type'] == 'su':
                Phi3SUConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
            else:
                raise Exception(config.rope_scaling['type'])
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
        MistralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
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
    elif arch == 'Qwen2ForCausalLM':
        QWen2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
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
    elif arch == 'codestral':
        CodestralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'OrionForCausalLM':
        OrionConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'MiniCPMForCausalLM':
        if config.num_experts is None:
            if (config.tie_word_embeddings is not None) and (not config.tie_word_embeddings):
                MiniCPMConverter.MODEL_TYPE = ModelType.MiniCPM2
            MiniCPMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            MiniCPMMoEConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'PersimmonForCausalLM':
        PersimmonConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'FuyuForCausalLM':
        FuyuConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'GemmaForCausalLM':
        GemmaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'CohereForCausalLM':
        CohereCommandConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
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
            print("DeelseekV2 is not fully supported yet!!!!")
        DeepSeekV2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
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
