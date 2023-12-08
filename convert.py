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
from enum import Enum
from pathlib import Path
from typing import IO, Any, Iterable, List, Optional, Tuple

import torch
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
    InternLM = 0x100
    LlaMA2 = 0x150
    BaiChuan = 0x200
    DeepSeek = 0x300
    DeepSeekCoder = 0x301

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
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    else:
        raise ValueError(f"unknown format: {path}")

def load_all_model_files(model_files) -> Dict:
    r = {}
    for f in model_files:
        for k, v in load_model_file(f):
            if k in r:
                raise Exception(f"tensor {k} already loaded. maybe it is stored cross files?")
            r[k] = v
    return r

def dump_state_dict(f, weight_names, model_files, ggml_type, config, tensor_pp):
    tensor_info = []
    converted_names = []
    state_dict = load_all_model_files(model_files)
    for name in tqdm(weight_names, desc="Dumping model state"):
        tensor: torch.Tensor = state_dict[name]

        tensor = tensor_pp(config, name, tensor)

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
        return f"<FastTokenizerVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

class BaseConverter:

    @classmethod
    def pp(cls, config, name: str, tensor):
        return tensor

    @classmethod
    def convert(cls, config, model_files, vocab: Any, ggml_type, save_path):

        # convert all weights to fp16
        with open(save_path, "wb") as f:
            f.write(b"ggml")  # magic
            f.write(struct.pack("ii", cls.MODEL_TYPE.value, 1))  # model type & version
            cls.dump_config(f, config, ggml_type)
            vocab.write_vocab(f)

            weight_names = cls.get_weight_names(config)
            dump_state_dict(f, weight_names, model_files, ggml_type, config, cls.pp)

        print(f"{cls.MODEL_TYPE.name} GGML model saved to {save_path}")

class InternLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.InternLM

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
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.bias",
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

def permute(weights: torch.Tensor, n_head: int) -> torch.Tensor:
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                   .swapaxes(1, 2)
                   .reshape(weights.shape))

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

class DeepSeekCoderConverter(BaseConverter):
    MODEL_TYPE = ModelType.DeepSeekCoder

    @classmethod
    def pp(cls, config, name: str, tensor):
        return LlamaConverter.pp(config, name, tensor)

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
            config.pad_token_id if config.pad_token_id is not None else g_special_tokens['<pad>'],
            config.sep_token_id if config.sep_token_id is not None else -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

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

def load_vocab(path: Path) -> Any:
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        path2 = path / "tokenizer.model"
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / "tokenizer.model"
        path4 = path / "tokenizer.json"
        if path2.exists():
            path = path2
        elif path3.exists():
            path = path3
        elif path4.exists():
            added_tokens_path = path.parent / "added_tokens.json"
            print(f"Loading vocab file {path}")
            return FastTokenizerVocab(path, added_tokens_path if added_tokens_path.exists() else None)
        else:
            raise FileNotFoundError(
                f"Could not find tokenizer.model in {path} or its parent; "
                "if it's in another directory, pass the directory as --vocab-dir")
    added_tokens_path = path.parent / "added_tokens.json"
    print(f"Loading vocab file {path}")
    return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None)

def load_config(path: Path) -> Any:
    with open(path / 'config.json', 'r') as fp:
        return json.load(fp)

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
            ChatGLM2Converter.convert(config, model_files, vocab, ggml_type, args.save_path)
        else:
            ChatGLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'InternLMForCausalLM':
        InternLMConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'LlamaForCausalLM':
        LlamaConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseek':
        DeepSeekConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    elif arch == 'deepseekcoder':
        DeepSeekCoderConverter.convert(config, model_files, vocab, ggml_type, args.save_path)
    else:
        raise Exception(f'unknown model_type: {config.model_type}')

if __name__ == "__main__":
    main()
