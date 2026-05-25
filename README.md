![GWen Architecture](./GWen.png)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Tokenizer](https://img.shields.io/badge/Tokenizer-8K%20BPE-6f42c1)
![Training](https://img.shields.io/badge/Training-DDP%20%7C%20BF16%20%7C%20SFT-brightgreen)
![Status](https://img.shields.io/badge/Status-Research%20Preview-orange)

# GWen（格温）

> GWen（格温）是一个面向中文小模型实验的纯 PyTorch LLM 项目：小词表、轻量参数、清晰代码、完整训练链路。
GWen 参考了现代 decoder-only LLM 的常见设计，包括 RMSNorm、SwiGLU、GQA、QK Norm、Partial RoPE、权重共享、Gate Attention，以及 Gated DeltaNet / Full Attention 混合路线，并把它们压缩到适合从零训练、阅读和改造的小模型规模。

中文名：**格温**  
默认身份：**GWen（格温），由 Chengjun 开发的小型中文语言模型。**

![GWen Architecture](./GWen%20Architecture.png)

## 目录
- [项目结构](#项目结构)
- [重要参数](#重要参数)
- [测试](#测试)
- [项目定位](#项目定位)
- [核心特性](#核心特性)
- [模型配置](#模型配置)
- [模型结构](#模型结构)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [数据格式](#数据格式)
- [训练命令](#训练命令)
- [实验记录](#实验记录)
- [Checkpoint 说明](#checkpoint-说明)
- [常见问题](#常见问题)
- [路线图](#路线图)
- [致谢](#致谢)
- [许可证](#许可证)

## 重要参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 可选 `gwen8k_hybrid`、`gwen8k_hybrid_128m`、`gwen8k_hybrid_256m` |
| `--tokenizer_path` | tokenizer 路径，默认通常为 `model/tokenizer_mini8k` |
| `--dataset_mode` | `lazy` 按样本 tokenize；`packed` 会拼接 token 并切块，预训练推荐 `packed` |
| `--data_cache_dir` | packed 预训练缓存目录 |
| `--linear_attention_backend` | `gdn` 为默认主线，按 `3×GDN + 1×Full` 混合层训练；`full` 用于全 Full Attention 对照，训练速度更快 |
| `--gdn_kernel_backend` | `auto/fla/torch`，仅在 `--linear_attention_backend gdn` 时生效；有 FLA 时可用 `fla` 提速 |
| `--gated_attention` | `none/headwise/elementwise/sigmoid`，默认和主线推荐为 `sigmoid` |
| `--dropout` | pretrain 推荐 `0.0`，SFT 推荐 `0.05` |
| `--eval_data_path` | 可选验证集 |
| `--eval_interval` | 每隔多少 optimizer step 做 eval；`0` 表示关闭 |
| `--max_steps` | smoke test 或短跑调试时很有用 |

训练日志包含：

```text
loss、 avg_loss、 lr、 eta、 tokens/s、 optimizer_step、 effective_tokens_per_step
```

## 项目结构

```text
.
├── dataset/                  # 数据集读取与预处理
├── model/
│   ├── model_gwen.py          # GWen config、模型、注意力、生成逻辑
│   ├── model_lora.py          # 轻量 LoRA 工具
│   └── tokenizer_mini8k/      # 默认 8K tokenizer
├── out/
│   ├── pretrain.log           # pretrain 训练日志

├── scripts/
│   ├── train_tokenizer.py     # tokenizer 训练
│   ├── eval_llm.py            # 命令行推理
│   ├── convert_model.py       # pth / safetensors 转换
│   └── smoke_gdn_stability.py # GDN 稳定性 smoke
├── trainer/
│   ├── 1-pretrain.py
│   ├── 2-full_sft.py
│   ├── 3-lora_sft.py
│   ├── 4-dpo_train.py
│   ├── 5-grpo_train.py
│   └── common.py
└── README.md

```


模型训练数据已经上传至modelscope：[模型训练数据](https://modelscope.cn/datasets/chengjun0178/Gwen-Train-DataSet)
- Gwen-PreTrain-DataSet    为预训练数据集
- Gwen-Train-DataSet       为SFT数据集

模型参数（gwen8k_hybrid的预训练和SFT的版本）已经上传至modelscope：
- [pretrain_final.pth](https://www.modelscope.cn/models/chengjun0178/pretrain_final.pth)   "pretrain_final.pth"为预训练后的模型参数
- [sft_final.pth](https://www.modelscope.cn/models/chengjun0178/sft_final.pth)        "sft_final.pth"为SFT后的模型参数


## 测试

模型和 GDN smoke：

```bash
python model/model_gwen.py
python scripts/smoke_gdn_stability.py
```

## 项目定位

GWen 的目标不是堆参数，而是把中文小模型从零训练这件事做清楚、做稳、做得容易复现。

很多小模型项目会遇到三个问题：

- 词表太大，embedding 吃掉大量参数，backbone 学习能力不够。
- 训练脚本、tokenizer、checkpoint、config 容易不一致，SFT 后输出混乱。
- 模型代码抽象太重，不方便读懂、调试和改结构。

GWen 选择 `8K tokenizer + 轻量 decoder-only backbone + 完整训练链路` 作为主线，让 80M、128M、256M 级别模型更适合中文实验和结构研究。
当然，各位佬可以自行选择调整参数大小，得到不同的模型配置。

## 核心特性

- **中文优先**：默认 8K 中文 tokenizer，适合中文小模型实验。
- **小词表主线**：`8192` vocab，降低 embedding 参数占比，把更多参数留给 transformer backbone。
- **纯 PyTorch 训练**：主训练链路不依赖 DeepSpeed、TRL、PEFT。
- **完整训练流程**：支持 Pretrain、Full SFT、LoRA SFT、DPO、实验性 GRPO。
- **清晰模型实现**：核心模型在 [model/model_gwen.py](./model/model_gwen.py)。
- **Gate Attention 默认开启**：默认模式为 `sigmoid`。
- **Hybrid 主线**：默认采用 `3 层 Gated DeltaNet + 1 层 Full Attention` 的混合层结构，兼顾长程建模实验和标准注意力稳定性。
- **严格加载检查**：推理和 SFT 会检查 tokenizer vocab、checkpoint config、缺失权重和 shape mismatch。

## 模型配置

| 配置名 | Vocab | Hidden | 层数 | 注意力头 | 层结构 | 参数量 | 推荐用途 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `gwen8k_hybrid` | 8192 | 768 | 8 | 8 / 4 KV | 3 GDN + 1 Full | ~80.8M | 快速实验 |
| `gwen8k_hybrid_128m` | 8192 | 768 | 12 | 8 / 4 KV | 3 GDN + 1 Full | ~118.0M | 中等规模实验 |
| `gwen8k_hybrid_256m` | 8192 | 1024 | 16 | 8 / 4 KV | 3 GDN + 1 Full | ~252.1M | 更强质量实验 |

查看真实参数量：

```bash
python model/model_gwen.py
```

`gwen8k_hybrid` 是当前主线配置。预训练脚本默认使用 `linear_attention_backend=gdn`，并按 `full_attention_interval=4` 形成 `3×GDN + 1×Full` 的周期结构。

如果要做全 Full Attention 对照实验，可以显式传入：

```bash
--linear_attention_backend full
```

## 模型结构

GWen 是 decoder-only causal language model，主要结构如下：

- Token Embedding
- 多层 PreNorm Decoder Layer
- RMSNorm
- QK Norm
- Partial RoPE
- Grouped Query Attention
- SwiGLU FFN
- 默认 sigmoid Gate Attention
- Gated DeltaNet / Full Attention 混合层
- Tied LM Head
- 自回归生成 cache

当 `linear_attention_backend=gdn` 时，层结构周期为：

```text
GDN -> GDN -> GDN -> Full -> GDN -> GDN -> GDN -> Full -> ...
```

当 `linear_attention_backend=full` 时，所有 decoder layer 都使用标准 full attention。

## 环境安装

基础安装：

```bash
git clone https://github.com/juncheng0178-del/GWen.git
cd GWen
pip install -r requirements.txt
```

GDN fused kernel 可选依赖：

```bash
pip install causal-conv1d
pip install flash-linear-attention
```

推荐环境：

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU，推荐支持 BF16
- 多卡训练建议使用 Linux + torchrun

## 快速开始

### 1. 检查模型

```bash
python model/model_gwen.py
```

你应该能看到类似输出：

```text
gwen8k_hybrid: params=80.8M
gwen8k_hybrid_128m: params=118.0M
gwen8k_hybrid_256m: params=252.1M
```

### 2. 训练或重建 tokenizer

默认 tokenizer 路径：

```text
model/tokenizer_mini8k
```

快速训练 8K tokenizer(本项目提供了tokenizer_mini8K的版本，不推荐重新训练)：

```bash
python scripts/train_tokenizer.py \
  --input dataset/gwen_sft_dataset.jsonl \
  --output model/tokenizer_mini8k \
  --vocab_size 8192 \
  --special_tokens_num 36 \
  --max_samples 100000
```

检查 tokenizer：

```bash
python -c "from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('model/tokenizer_mini8k'); print(len(tok), tok.encode('你好，GWen！'))"
```

### 3. 跑一个最小预训练 smoke

```bash
python trainer/1-pretrain.py \
  --config gwen8k_hybrid \
  --tokenizer_path model/tokenizer_mini8k \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --out_dir out/smoke_pretrain \
  --dataset_mode lazy \
  --max_seq_len 64 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 2 \
  --dtype fp32
```

## 数据格式

### Pretrain

每行一个 JSON，包含 `text` 字段：

```json
{"text": "GWen 使用普通文本做 next-token prediction 预训练。"}
```

### SFT

推荐格式：

```json
{"conversations": [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": "我是 GWen（格温），由 Chengjun 开发的小型中文语言模型。"}]}
```

也支持 `messages` 字段。支持的 role：

```text
system, user, assistant, tool
```

SFT 默认只训练 assistant 内容：system/user/tool 会被 mask 为 `-100`，assistant 内容和 assistant 的 `<|im_end|>` 会参与 loss。

## 训练命令

下面都以 `gwen8k_hybrid` 为例。这个配置默认就是 `3×GDN + 1×Full`，Gate Attention 默认是 `sigmoid`，所以命令里不需要重复传 `--linear_attention_backend gdn` 和 `--gated_attention sigmoid`。

不同硬件主要调整 `batch_size`、`gradient_accumulation_steps` 和 `max_seq_len`。如果显存足够，优先增大 `batch_size`；如果显存不够，再减小`batch_size`，增大 `gradient_accumulation_steps` 保持有效 batch。

### 单卡预训练

```bash
python trainer/1-pretrain.py \
  --config gwen8k_hybrid \
  --tokenizer_path model/tokenizer_mini8k \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --out_dir out/pretrain_gwen8k_hybrid \
  --max_seq_len 340 \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --epochs 5 \
```

### 多卡预训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 trainer/1-pretrain.py \
  --config gwen8k_hybrid \
  --tokenizer_path model/tokenizer_mini8k \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --out_dir out/pretrain_gwen8k_hybrid \
  --max_seq_len 1024 \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --epochs 10 \
  --learning_rate 3e-4 \
```

### 单卡 SFT

```bash
python trainer/2-full_sft.py \
  --config gwen8k_hybrid \
  --tokenizer_path model/tokenizer_mini8k \
  --pretrain_path out/pretrain_gwen8k_hybrid/pretrain_final.pth \
  --data_path dataset/sft_t2t_mini.jsonl \
  --out_dir out/sft_gwen8k_hybrid \
  --max_seq_len 768 \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --epochs 3 \
```

### 多卡 SFT

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 trainer/2-full_sft.py \
  --config gwen8k_hybrid \
  --tokenizer_path model/tokenizer_mini8k \
  --pretrain_path out/pretrain_final.pth \
  --data_path dataset/sft_t2t_mini.jsonl \
  --out_dir out/sft_gwen8k_hybrid \
  --max_seq_len 1024 \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --epochs 3 \
```

### 常用可选参数

- `--gdn_kernel_backend fla`：安装 `flash-linear-attention` 后可启用 fused GDN kernel，通常训练更快。
- `--gdn_kernel_backend torch`：纯 PyTorch GDN 路径，速度较慢，适合排查数值问题。
- `--linear_attention_backend full`：把所有层切成标准 Full Attention，用于对照实验。
- `--warmup_steps 500 --weight_decay 0.1`：预训练常用优化参数。
- `--dropout 0.05`：SFT 常用 dropout；预训练通常保持默认 `0.0`。
- `--num_workers 8 --log_interval 10 --use_compile`：吞吐优化和日志相关参数，可按环境开启。

## 实验记录

下面记录来自一次多卡实验记录，主要用于展示 `gwen8k_hybrid` 系列在 Pretrain 和 Full SFT 后的真实 CLI 表现。完整原始记录见 [实验配置.txt](./实验配置.txt) 和 [Cli推理.txt](./Cli推理.txt)。
（PS：在下为了快速实验验证，这里是用8张A800训练的）

### 实验配置

Pretrain 使用 `dataset/pretrain_t2t_mini.jsonl`，SFT 使用 `dataset/sft_t2t_mini.jsonl`。三组模型均开启 `sigmoid` Gate Attention。

| 标记 | 配置 | 记录参数量 | Backend | GDN Kernel | Seq Len | Batch | Accum | GPU | 训练步数 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | `gwen8k_hybrid` | 80.8M | `gdn` | `fla` | 340 | 32 | 8 | 8 | 3100 |
| B | `gwen8k_hybrid_128m` | 118.0M | `gdn` | `fla` | 1024 | 32 | 1 | 8 | 24805 |
| C | `gwen8k_hybrid_256m` | 252.1M | `gdn` | `fla` | 1024 | 32 | 2 | 8 | 7441 |

SFT 加载预训练 checkpoint 时三组均为：

```text
missing=0 unexpected=0 skipped_shape=0
```

### CLI 推理

评估命令示例：

```bash
python scripts/eval_llm.py \
  --tokenizer_path model/tokenizer_mini8k \
  --model_path out/pretrain_final.pth \
  --max_new_tokens 512 \
  --config gwen8k_hybrid
```

其中 A/B/C 分别对应 `gwen8k_hybrid`、`gwen8k_hybrid_128m`、`gwen8k_hybrid_256m`。

| Prompt | Pretrain 现象 | SFT 现象 |
| --- | --- | --- |
| `你好，请问你是谁？` | 三组都会受语料影响回答 Qwen、通义千问或阿里云模型身份。 | 三组均能回答 GWen / Gwen / 格温，并说明由 Chengjun 开发。 |
| `什么是人工智能？` | 已能生成基本定义和应用场景，但容易带出“请写一段简介”等数据格式痕迹。 | 回答更像助手，结构更完整；大模型版本内容更长，但仍有少量重复。 |
| `你有什么能力？` | 能列出问答、翻译、写作、编程等能力，但存在重复和泛化表达。 | 身份与助手语气更稳定，能列出文本生成、摘要、问答、代码等能力。 |
| `用 Python 写一个斐波那契数列的函数。` | 能生成代码外形，但格式和逻辑不稳定。 | 仍有代码正确性问题，说明后续需要补充高质量代码 SFT 数据。 |

身份问题的 SFT 输出示例：

```text
You: 你好，请问你是谁？

[A]: 您好！我是由 Chengjun 开发的智能助手 Gwen。
[B]: 您好！我是Gwen，中文名是格温。由 Chengjun 开发。
[C]: 我叫 GWen，中文名格温，由 Chengjun 开发。很高兴和你聊天。
```

这组实验说明：Pretrain 阶段模型已经具备基础中文续写和知识表达能力，但身份会明显继承预训练语料；Full SFT 可以有效修正身份和对话风格。代码生成仍是当前短板，需要更干净的代码数据和专门 SFT。


正常加载 checkpoint 时应该看到：

```text
[Load] missing=0 unexpected=0 skipped_shape=0
```
如果看到 `missing/unexpected/skipped_shape` 不为 0，说明 checkpoint 和当前模型配置或 tokenizer 不匹配，继续使用可能会导致训练或推理异常。
## Checkpoint 说明

训练 checkpoint 会保存：

- `model_state_dict`
- `config`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `scaler_state_dict`
- `step`
- `epoch`
- `train_args`

SFT 和推理会优先读取 checkpoint 内的真实 config，并检查 tokenizer vocab 是否匹配。如果 `missing/unexpected/skipped_shape` 不为 0，除非你明确在做实验，否则不要继续使用这个 checkpoint。

为了减少磁盘压力，训练脚本不会再按 step 周期保存 checkpoint，只保留 epoch checkpoint 和 final checkpoint。

## 常见问题

**GWen 是 Qwen 官方模型吗？**  
不是。GWen 是独立研究项目，只参考了一些现代 LLM 结构设计。

**为什么使用 8K tokenizer？**  
小模型参数很紧张，大词表会让 embedding 占掉过多参数。8K tokenizer 能把更多参数留给 backbone，更适合 80M 到 256M 级别实验。

**为什么模型有时会说自己是通义千问？**  
这通常来自预训练语料分布。如果语料里有大量 Qwen/通义千问身份表达，模型会学到这些模式。建议在 SFT 阶段加入高质量身份样本，并在评估时使用 system prompt 固定身份。

**训练应该用 full 还是 gdn？**  
README 主线默认用 `gdn`，也就是 `3×GDN + 1×Full`。如果你想做 dense baseline、排查 GDN kernel 或复现实验对照，可以显式传 `--linear_attention_backend full`。此外，实验结果显示，当`--linear_attention_backend full` 模型训练速度更快。

**`packed` 和 `lazy` 有什么区别？**  
`lazy` 每条样本单独 tokenize，简单直观；`packed` 会把文本 token 拼接后切成固定长度块，padding 更少，预训练吞吐通常更好。这里默认`lazy`。

**旧 checkpoint 能加载吗？**  
如果旧 checkpoint 的 config 名、tokenizer 路径或 vocab size 和当前 `gwen8k_*` 配置不一致，可能需要单独迁移。

## 路线图

- 完善 `gwen8k_hybrid_128m` 的公开训练 recipe
- 增加中文 QA / 指令跟随评测脚本
- 增加 Hugging Face 格式导出
- 增强身份 SFT 数据构造工具
- 系统比较 full attention 与 GDN 的效果差异

## 致谢

GWen 参考和受启发于：

- [Qwen](https://github.com/QwenLM/Qwen)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [Gated Attention](https://github.com/qiuzh20/gated_attention)
- [MiniMind](https://github.com/jingyaogong/minimind)

## 许可证

本项目用于研究和学习。正式开源前，请添加明确的 `LICENSE` 文件，并确认训练数据、tokenizer 和外部参考项目的许可证约束。
