
# MiniQwen模型使用教程

MiniQwen是一个迷你版的因果语言建模模型，参考了Qwen2的架构设计。该模型具有以下特点：
- 仅有6个transformer层
- 模型总大小不超过200MB
- 上下文窗口为1024
- 使用BPE分词器并采用BERT的原始词表

这个教程将指导你如何使用MiniQwen模型进行文本生成和模型训练。

## 目录

1. [环境准备](#1-环境准备)
2. [模型架构概述](#2-模型架构概述)
3. [模型推理](#3-模型推理)
4. [模型训练](#4-模型训练)
5. [模型自定义](#5-模型自定义)
6. [常见问题](#6-常见问题)

## 1. 环境准备

首先，你需要安装必要的Python包：

```bash
pip install torch transformers tqdm regex numpy
```

然后，你需要下载以下文件：
- `modeling_miniqwen.py`: 模型架构实现
- `tokenization_miniqwen.py`: 分词器实现
- `inference_miniqwen.py`: 推理脚本
- `train_miniqwen.py`: 训练脚本

## 2. 模型架构概述

MiniQwen模型基于Transformer架构，并采用了一些现代LLM的关键技术：

- **Group Query Attention (GQA)**: 一种优化的注意力机制，通过将多个查询头共享一个键值头来减少计算量和内存占用。
- **Rotary Positional Embeddings (RoPE)**: 一种位置编码方法，通过应用旋转矩阵到查询和键来编码位置信息。
- **SwiGLU激活函数**: 一种结合了Swish和GLU的激活函数，性能优于传统的ReLU或GELU。
- **RMSNorm**: 比LayerNorm更高效的归一化方法。

模型结构：
- 输入层: 将token ID转换为嵌入向量
- Transformer层 (x6): 每层包含自注意力机制和前馈网络
- 输出层: 将最终的隐藏状态映射到词汇表，预测下一个token

模型规模:
- 隐藏层大小: 768
- 注意力头数: 12
- KV头数: 4 (GQA)
- 词汇表大小: 30522 (BERT词表)
- 上下文窗口: 1024
- 参数量: 约169MB (满足不超过200MB的要求)

## 3. 模型推理

### 从头开始进行推理

如果你想使用新初始化的模型进行推理（虽然未经训练的模型会产生随机文本）：

```bash
python inference_miniqwen.py --prompt "你好，请介绍一下自己" --max_new_tokens 50
```

### 使用预训练模型进行推理

如果你已经有了训练好的模型：

```bash
python inference_miniqwen.py --model_path ./miniqwen_checkpoints/final_model --tokenizer_path ./miniqwen_checkpoints/final_model --prompt "你好，请介绍一下自己" --max_new_tokens 100 --temperature 0.7 --do_sample --streaming
```

参数说明：
- `--model_path`: 模型路径
- `--tokenizer_path`: 分词器路径
- `--prompt`: 提示文本
- `--max_new_tokens`: 最大生成token数
- `--temperature`: 生成温度（越高越随机）
- `--top_p`: 核采样参数
- `--top_k`: top-k采样参数
- `--do_sample`: 启用采样（否则使用贪婪解码）
- `--streaming`: 启用流式输出

## 4. 模型训练

### 准备训练数据

训练数据应为纯文本文件，每行一个样本。例如：

```
这是第一个训练样本。
这是第二个训练样本。
...
```

### 从头开始训练

```bash
python train_miniqwen.py --train_file ./data/train.txt --output_dir ./miniqwen_checkpoints --batch_size 8 --learning_rate 5e-5 --epochs 3
```

### 从现有检查点继续训练

```bash
python train_miniqwen.py --model_path ./miniqwen_checkpoints/checkpoint-1000 --tokenizer_path ./miniqwen_checkpoints/checkpoint-1000 --train_file ./data/train.txt --output_dir ./miniqwen_checkpoints --batch_size 8 --learning_rate 2e-5 --epochs 2
```

参数说明：
- `--model_path`: 预训练模型路径（可选）
- `--tokenizer_path`: 分词器路径（可选）
- `--train_file`: 训练数据文件
- `--output_dir`: 模型保存目录
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--weight_decay`: 权重衰减
- `--epochs`: 训练轮数
- `--gradient_accumulation_steps`: 梯度累积步数
- `--max_seq_length`: 最大序列长度
- `--save_steps`: 每多少步保存一次模型
- `--warmup_steps`: 预热步数
- `--logging_steps`: 每多少步记录一次日志

## 5. 模型自定义

### 修改模型大小

如果你想调整模型大小，可以修改`create_fresh_model`函数中的配置：

```python
config = MiniQwenConfig(
    vocab_size=30522,  # 可以调整词汇表大小
    hidden_size=768,   # 可以调整隐藏层大小
    intermediate_size=2048,  # 可以调整前馈网络大小
    num_hidden_layers=6,  # 可以调整层数
    num_attention_heads=12,  # 可以调整注意力头数
    num_key_value_heads=4,  # 可以调整KV头数
    # 其他参数...
)
```

### 自定义分词器

如果你想使用自己的词汇表，可以：

1. 准备自己的词汇表文件（每行一个token）
2. 修改分词器初始化：

```python
tokenizer = MiniQwenTokenizer(
    vocab_file="path/to/your/vocab.txt",
    bos_token="<s>",
    eos_token="</s>",
    # 其他参数...
)
```

## 6. 常见问题

### Q: 模型训练需要多少显存？
A: 使用默认设置（6层，隐藏层大小768），训练需要约2-3GB的显存（batch_size=8）。如果显存不足，可以减小batch_size或使用梯度累积。

### Q: 如何保存和加载自定义模型？
A: 模型和分词器兼容HuggingFace的保存和加载接口。使用`model.save_pretrained()`保存，使用`MiniQwenForCausalLM.from_pretrained()`加载。

### Q: 如何在模型推理时使用GPU加速？
A: 推理脚本会自动检测GPU并使用。如果有多个GPU，默认使用第一个。可以通过设置环境变量`CUDA_VISIBLE_DEVICES`来选择特定GPU。

### Q: 分词器不满足我的需求怎么办？
A: 当前分词器实现比较基础。如果需要更复杂的分词功能，可以考虑：
1. 使用HuggingFace的tokenizers库自定义一个BPE分词器
2. 扩展当前分词器的`_tokenize`方法，添加更高级的分词逻辑

### Q: 如何评估模型性能？
A: 可以使用困惑度(perplexity)来评估语言模型性能。添加评估脚本，在验证集上计算模型的困惑度：

```python
def calculate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()
    
    return math.exp(total_loss / total_tokens)
```

## 后续改进方向

1. **优化分词器**: 实现完整的BPE分词算法，使分词更高效和准确
2. **添加更多训练选项**: 支持LoRA等参数高效微调方法
3. **模型优化**: 添加更多性能优化，如FlashAttention
4. **模型量化**: 添加INT8/INT4量化支持，以减小模型大小
5. **多语言支持**: 扩展分词器和模型以更好地支持多语言处理
