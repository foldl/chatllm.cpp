# Tutorial on RAG

## Prepare Models

Get the models for text embedding and QA ranking that are supported, for example,

* Text Embedding (`XLMRobertaModel`)
    * [x] [BCE-Embedding](https://huggingface.co/maidalun1020/bce-embedding-base_v1)

* QA Ranking (`XLMRobertaForSequenceClassification`) (_Optional_)
    * [x] [BCE-ReRanker](https://huggingface.co/maidalun1020/bce-reranker-base_v1)

Use `convert.py` to convert the models.

## Prepare Vector Store

`ChatLLM.cpp` converts a raw data file into a vector store file.

1. Generate raw data files

    The format of the raw data file is very simple: every two lines is a record, where
    the first line is Base64 encoded text content, and the second one is Base64 encoded meta
    data. Meta data is also a string which may contain anything, such as serialized JSON data.

    Here is an example. `fruits.dat` contains three records.

    ```python
    def to_b64(data) -> str:
        s = json.dumps(data) if isinstance(data, dict) else data
        return base64.b64encode(s.encode('utf-8')).decode()

    def gen_test_data():
        texts = [
            {'page_content': 'the apple is black', 'metadata': {'file': 'a.txt'}},
            {'page_content': 'the orange is green', 'metadata': {'file': '2.txt'}},
            {'page_content': 'the banana is red', 'metadata': {'file': '3.txt'}},
        ]

        with open('fruits.dat', 'w') as f:
            for x in texts:
                f.write(to_b64(x['page_content']))
                f.write('\n')
                f.write(to_b64(x['metadata']))
                f.write('\n')
    ```

2. Convert raw data file into vector store

    ```
    ./bin/main --embedding_model ../quantized/bce_em.bin --init_vs /path/to/fruits.dat
    ```

    Note that we must specify the text embedding model.
    The vector store file will be save to `fruits.dat.vsdb`.

## Chat with RAG

Now let's chat with RAG. You can select any support LLM as backend and compare their performance.

### Using MiniCPM

Here we are using [MiniCPM DPO-2B](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16),
and QA ranking model is also used.

```
./bin/main -i -m ../quantized/minicpm_dpo_f16.bin --embedding_model ../quantized/bce_em.bin --reranker_model ../quantized/bce_rank.bin --vector_store /path/to/fruits.dat.vsdb
    ________          __  __    __    __  ___
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by MiniCPM,                    /_/   /_/
with 2724880896 (2.7B) parameters.
Augmented by BCE-Embedding (0.2B) and BCE-ReRanker (0.2B).

You  > what color is the banana?
A.I. >  Based on the given information, the color of the banana is red.

References:
1. {"file": "3.txt"}
You  > write a quick sort function in python
A.I. >  Sure, here's a simple implementation of the QuickSort algorithm in Python:

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less = [i for i in arr[1:] if i <= pivot]
        greater = [i for i in arr[1:] if i > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)
...
```

Note that, no information from the vector store is used when answering the second question.

### Using Qwen-QAnything

Here we are using [Qwen-QAnything-7B](https://huggingface.co/netease-youdao/Qwen-7B-QAnything),
which is a bilingual instruction-tuned model of [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B) for [QAnything](https://github.com/netease-youdao/QAnything),
and QA ranking model is also used.

```
./bin/main -i --temp 0 -m ../quantized/qwen-qany-7b.bin --embedding_model ../quantized/bce_em.bin --reranker_model ../quantized/bce_rank.bin --vector_store /path/to/fruits.dat.vsdb --rag_template "参考信息：\n{context}\n---\n我的问题或指令：\n{question}\n---\n请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复."
    ________          __  __    __    __  ___ (通义千问)
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by QWen,                       /_/   /_/
with 7721324544 (7.7B) parameters.
Augmented by BCE-Embedding (0.2B) and BCE-ReRanker (0.2B).

You  > what color is the banana?
A.I. > the banana is red.

References:
1. {"file": "3.txt"}
You  > write a quick sort function in python
A.I. > Sure, here's a quick sort function in Python:

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
...
```

Note that, we are using the [default prompt template](https://github.com/netease-youdao/QAnything/blob/master/etc/qanything/prompt.conf) provided by [QAnything](https://github.com/netease-youdao/QAnything).
