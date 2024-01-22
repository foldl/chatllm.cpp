"""
Convert Hugging Face ChatLLM/ChatLLM2 models to GGML format
"""
import argparse
from ast import Dict, Tuple
from collections import OrderedDict
import json
import struct
import sys
import io
from pathlib import Path
from enum import Enum
from pathlib import Path
from typing import IO, Any, Iterable, List, Optional, Tuple
import numpy as np

import torch
from torch import nn
from tabulate import tabulate
from tqdm import tqdm

from sentencepiece import SentencePieceProcessor  # type: ignore

GGML_QK8_0 = 32
GGML_QK4_0 = 32

GGML_MEM_ALIGN = 16


class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q8_0 = 8


class ModelType(Enum):
    CHATGLM = 1
    CHATGLM2 = 2
    CHATGLM3 = 3
    CODEGEEX2 = 4

    InternLM   = 0x100
    InternLM2   = 0x101

    LlaMA2      = 0x150
    CodeLlaMA   = 0x151
    WizardCoder = 0x152
    WizardLM    = 0x153
    WizardMath  = 0x154
    TigerBot    = 0x155

    BaiChuanLlama = 0x200
    BaiChuan = 0x201

    DeepSeek = 0x300
    DeepSeekCoder = 0x301

    Yi = 0x400

    Phi2                = 0x500
    Phi2_v2             = 0x501
    DolphinPhi2         = 0x510
    DolphinPhi2_v2      = 0x511

    Mistral = 0x600
    Mixtral = 0x601
    OpenChat = 0x602
    NeuralBeagle = 0x603

    QWen    = 0x700

    BlueLM  = 0x800

    StableLM    = 0x900

    BCE_Embedding = 0x10000100


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

def load_all_model_files(model_files) -> Dict:
    for f in model_files:
        r = {}
        data = load_model_file(f)
        for k, v in data:
            if k in r:
                raise Exception(f"tensor {k} already loaded. maybe it is stored cross files?")
            r[k] = v
        yield r

def dump_state_dict(f, weight_names, model_files, ggml_type, config, state_dict_pp):
    tensor_info = []
    converted_names = []

    # Note: incremental loading and converting

    state_dict_cache = {}
    remaining: List = weight_names.copy()

    for state_dict in load_all_model_files(model_files):
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

            if tensor.ndim == 2:
                # 2d weight: should quantize it if needed
                tensor = tensor.float()

                # step 2: quantize it into ggml format
                tensor_ggml_type = ggml_type
            else:
                # 1d weight: convert it to float32
                assert tensor.ndim == 1
                tensor = tensor.float()
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
            added_tokens = json.load(open(fname_added_tokens))
        else:
            added_tokens = {}
        vocab_size: int = self.sentencepiece_tokenizer.vocab_size()
        print("vocab_size ", vocab_size)
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            if actual_ids == list(range(len(added_tokens))):
                raise Exception(f"added token IDs ({actual_ids}) are starting from 0. `added_token.json` seems WRONG. \n\nDelete it and try again.")
            else:
                raise Exception(f"Expected added token IDs to be sequential and start at {len(added_tokens)}; got {actual_ids}")
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
        return f"<FastTokenizerVocab with {self.vocab_size} tokens>"

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

    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:

        from transformers import AutoTokenizer  # type: ignore[attr-defined]
        tokenizer = AutoTokenizer.from_pretrained(fname_tokenizer, trust_remote_code=True)
        vocab_size = max(tokenizer.get_vocab().values()) + 1

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[self.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = TikTokenizerVocab.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
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
            assert num_key_value_heads == config.num_attention_heads == 'dynamic', "num_key_value_heads must equal to num_attention_heads"
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

class LlamaConverter(BaseConverter):
    MODEL_TYPE = ModelType.LlaMA2

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight') or name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
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
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
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
        LlamaConverter.dump_config(f, config, ggml_type)

        config_values = [
            config.num_key_value_heads,
            config.sliding_window if config.sliding_window is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))
        f.write(struct.pack("<f", config.rope_theta))

    @staticmethod
    def get_weight_names(config):
        return LlamaConverter.get_weight_names(config)

class OpenChatConverter(BaseConverter):
    MODEL_TYPE = ModelType.OpenChat

    @classmethod
    def pp(cls, config, name: str, tensor):
        return MistralConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        MistralConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return MistralConverter.get_weight_names(config)

class NeuralBeagleConverter(BaseConverter):
    MODEL_TYPE = ModelType.NeuralBeagle

    @classmethod
    def pp(cls, config, name: str, tensor):
        return MistralConverter.pp(config, name, tensor)

    @staticmethod
    def dump_config(f, config, ggml_type):
        MistralConverter.dump_config(f, config, ggml_type)

    @staticmethod
    def get_weight_names(config):
        return MistralConverter.get_weight_names(config)

class MixtralConverter(BaseConverter):
    MODEL_TYPE = ModelType.Mixtral

    @classmethod
    def pp(cls, config, name: str, tensor):
        if name.endswith('k_proj.weight'):
            return permute(tensor, config.num_key_value_heads)
        elif name.endswith('q_proj.weight'):
            return permute(tensor, config.num_attention_heads)
        return tensor

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.output_router_logits == False, "output_router_logits must be False"

        MistralConverter.dump_config(f, config, ggml_type)

        config_values = [
            config.num_experts_per_tok,
            config.num_local_experts,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def get_weight_names(config):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(config.num_hidden_layers):
            for j in range(config.num_local_experts):
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

def part(weights: torch.Tensor, n_part: int) -> torch.Tensor:
    r = weights.shape[0] // 3
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

        LlamaConverter.dump_config(f, config, ggml_type)

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

        LlamaConverter.dump_config(f, config, ggml_type)

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

        LlamaConverter.dump_config(f, config, ggml_type)

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

class XLMRobertaConverter(BaseConverter):
    MODEL_TYPE = ModelType.BCE_Embedding

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
                        'embeddings.token_type_embeddings.weight',
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

        weight_names += [
            'pooler.dense.weight',
            'pooler.dense.bias',
        ]

        return weight_names

def load_vocab(path: Path) -> Any:
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        path2 = path / "tokenizer.model"
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / "tokenizer.model"
        path20 = path / "sentencepiece.bpe.model"
        path4 = path / "tokenizer.json"
        path5 = path / "qwen.tiktoken"
        if path2.exists():
            path = path2
        elif path20.exists():
            path = path20
        elif path3.exists():
            path = path3
        elif path4.exists():
            added_tokens_path = path.parent / "added_tokens.json"
            print(f"Loading vocab file {path}")
            return FastTokenizerVocab(path, added_tokens_path if added_tokens_path.exists() else None)
        elif path5.exists():
            return TikTokenizerVocab(path, None)
        else:
            raise FileNotFoundError(
                f"Could not find tokenizer.model in {path} or its parent; "
                "if it's in another directory, pass the directory as --vocab-dir")
    added_tokens_path = path.parent / "added_tokens.json"
    print(f"Loading vocab file {path}")
    return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None)

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
        globs = ["model-*-of-*.safetensors", "consolidated.*.pth", "pytorch_model-*-of-*.bin", "*.pt", "pytorch_model.bin"]
        for glob in globs:
            files = list(path.glob(glob))
            if len(files) > 0:
                files.sort()
                return files
        raise Exception('can not find model files, or unsupported format.')
    else:
        return [path]

class AttributeDict(dict):
    def __getattr__(self, key):
        return self.__getitem__(key) if key in self else None

    __setattr__ = dict.__setitem__

def main():
    parser = argparse.ArgumentParser("chatllm-convert")
    parser.add_argument("-i", "--model_name_or_path", type=str)
    parser.add_argument("-a", "--arch", type=str, default='')
    # TODO: LoRA
    #parser.add_argument("-l", "--lora_model_name_or_path", type=str, default=None)
    parser.add_argument("-o", "--save_path", type=Path)
    parser.add_argument("-t", "--type", type=str, default="q8_0", choices=["f32", "f16", "q8_0", "q4_0"])
    parser.add_argument("--vocab_dir", type=str, default='')
    args = parser.parse_args()

    ggml_type = GGMLType[args.type.upper()]

    vocab = load_vocab(Path(args.model_name_or_path) if args.vocab_dir == '' else Path(args.vocab_dir))
    model_files = load_some_model(Path(args.model_name_or_path))

    #if args.lora_model_name_or_path is not None:
    #    from peft import PeftModel
    #
    #    model = PeftModel.from_pretrained(model, args.lora_model_name_or_path)
    #    model = model.merge_and_unload()

    config = AttributeDict(load_config(Path(args.model_name_or_path)))

    if len(config.architectures) != 1:
        raise Exception(f'unknown architectures: {config.architectures}')

    arch = args.arch.lower()
    if arch == '':
        arch = config.architectures[0]

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
    elif arch == 'InternLMForCausalLM':
        if not config.bias:
            InternLMConverter.MODEL_TYPE = ModelType.InternLM2
        InternLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'LlamaForCausalLM':
        LlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'codellama':
        CodeLlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseek':
        DeepSeekConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseekcoder':
        DeepSeekCoderConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'BaichuanForCausalLM':
        if config.num_hidden_layers <= 32:
            BaiChuanLlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            config.max_position_embeddings = config.model_max_length
            BaiChuanConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'yi':
        YiConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'PhiForCausalLM':
        if config.hidden_act is not None:
            Phi2Converter.MODEL_TYPE = ModelType.Phi2_v2
        Phi2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
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
        MixtralConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'openchat':
        OpenChatConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'QWenLMHeadModel':
        QWenConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'tigerbot':
        TigerBotConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'BlueLMForCausalLM':
        BlueLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'StableLMEpochForCausalLM':
        StableLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'XLMRobertaModel':
        XLMRobertaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'neuralbeagle':
        NeuralBeagleConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
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
