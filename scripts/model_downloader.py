import requests
import os

def model_on_modelscope(proj: str, fn: str) -> dict:
    url = f"https://modelscope.cn/api/v1/models/judd2024/{proj}/repo?Revision=master&FilePath={fn}"
    return { 'fn': fn, 'url': url }

all_models = {
    'qwen2': {
        'default': '1.5b',
        'brief': 'Qwen2 is a new series of large language models from Alibaba group.',
        'variants': {
            '7b': {
                'default': 'q8',
                'quantized': {
                    'q8': model_on_modelscope('chatllm_quantized_qwen2', 'qwen2-7b.bin')
                }
            },
            '1.5b': {
                'default': 'q8',
                'quantized': {
                    'q8': model_on_modelscope('chatllm_quantized_qwen2', 'qwen2-1.5b.bin')
                }
            },
            '0.5b': {
                'default': 'q8',
                'quantized': {
                    'q8': model_on_modelscope('chatllm_quantized_qwen2', 'qwen2-0.5b.bin')
                }
            },
        }
    },
    'gemma': {
        'default': '2b',
        'brief': 'Gemma is a family of lightweight, state-of-the-art open models built by Google DeepMind. Updated to version 1.1.',
        'variants': {
            '2b': {
                'default': 'q8',
                'quantized': {
                    'q8': model_on_modelscope('chatllm_quantized_models', 'gemma-1.1-2b.bin')
                }
            },
        }
    },
    'llama3': {
        'default': '8b',
        'brief': 'Meta Llama 3: The most capable openly available LLM to date.',
        'variants': {
            '8b': {
                'default': 'q4_1',
                'quantized': {
                    'q4_1': model_on_modelscope('chatllm_quantized_models', 'llama3-8b-q4_1.bin')
                }
            },
        }
    },
    'minicpm': {
        'default': '2b-sft',
        'brief': 'Meta Llama 3: The most capable openly available LLM to date.',
        'variants': {
            '2b-sft': {
                'default': 'q8',
                'quantized': {
                    'q8': model_on_modelscope('chatllm_quantized_models', 'minicpm_sft_q8.bin')
                }
            },
            '2b-dpo': {
                'default': 'q4_1',
                'quantized': {
                    'q4_1': model_on_modelscope('chatllm_quantized_models', 'minicpm-dpo-q4_1.bin')
                }
            },
        }
    },
    'qwen1.5': {
        'default': 'moe',
        'brief': 'Qwen1.5 is the beta version of Qwen2 from Alibaba group.',
        'variants': {
            '1.8b': {
                'default': 'q8',
                'quantized': {
                    'q8': model_on_modelscope('chatllm_quantized_models', 'qwen1.5-1.8b.bin')
                }
            },
            'moe': {
                'default': 'q4_1',
                'quantized': {
                    'q4_1': model_on_modelscope('chatllm_quantized_models', 'qwen1.5-moe-q4_1.bin')
                }
            },
        }
    },
    'qanything': {
        'default': '7b',
        'brief': 'QAnything is a local knowledge base question-answering system based on QwenLM.',
        'variants': {
            '7b': {
                'default': 'q4_1',
                'quantized': {
                    'q4_1': model_on_modelscope('chatllm_quantized_models', 'qwen-qany-7b-q4_1.bin')
                }
            },
        }
    },
    'starling-lm': {
        'default': '7b',
        'brief': 'Starling is a large language model trained by reinforcement learning from AI feedback focused on improving chatbot helpfulness.',
        'variants': {
            '7b': {
                'default': 'q4_1',
                'quantized': {
                    'q4_1': model_on_modelscope('chatllm_quantized_models', 'starling-7b-q4_1.bin')
                }
            },
        }
    },
    'yi-1': {
        'default': '34b',
        'brief': 'Yi (v1) is a high-performing, bilingual language model.',
        'variants': {
            '34b': {
                'default': 'q4_1',
                'quantized': {
                    'q4_1': model_on_modelscope('chatllm_quantized_models', 'yi-34b-q4.bin')
                }
            },
        }
    },
}

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 60, fill = '█', printEnd = "\r", auto_nl = True):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if (iteration == total) and auto_nl:
        print()

def download_file(url: str, fn: str, prefix: str):
    flag = False
    print(f"downloading {prefix}")
    with open(fn, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            progress = 0

            for chunk in r.iter_content(chunk_size=8192):
                progress += len(chunk)
                f.write(chunk)
                print_progress_bar(progress, total)

            flag = progress == total
    return flag

def show():
    def show_variants(info, default):
        sizes = [s for s in info.keys()]
        variants = [m + ":" + s for s in sizes]
        all_var = ', '.join(variants)
        print(f"Available: {all_var}")
        if len(variants) > 1:
            print(f"Default  : {m + ':' + default}")

    def show_model(m):
        info = all_models[m]
        print(f"**{m}**: {info['brief']}")
        show_variants(info['variants'], info['default'])
        print()

    for m in all_models.keys():
        show_model(m)

def parse_model_id(model_id: str):
    parts = model_id.split(':')
    model = all_models[parts[0]]
    variants = model['variants']
    var = variants[parts[1]] if len(parts) >= 2 else variants['default']
    return var['quantized'][var['default']]

def get_model(model_id, storage_dir):
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)
    assert os.path.isdir(storage_dir), f"{storage_dir} is invalid"

    info = parse_model_id(model_id)
    fn = os.path.join(storage_dir, info['fn'])
    if os.path.isfile(fn):
        return fn

    assert download_file(info['url'], fn, model_id), f"failed to download {model_id}"

    return fn

def find_index(l: list, x) -> int:
    if x in l:
        return l.index(x)
    else:
        return -1

def preprocess_args(args: list[str], storage_dir) -> list[str]:
    i = find_index(args, '-m')
    if i < 0:
        i = find_index(args, '--model')
    if i < 0:
        return args
    if args[i + 1].startswith(':'):
        args[i + 1] = get_model(args[i + 1][1:], storage_dir)

    return args

if __name__ == '__main__':
    show()