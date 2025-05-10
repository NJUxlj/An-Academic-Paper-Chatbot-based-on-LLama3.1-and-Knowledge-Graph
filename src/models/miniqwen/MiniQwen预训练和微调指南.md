
# MiniQwen预训练和微调指南

本指南详细介绍了如何使用提供的脚本对MiniQwen模型进行继续预训练和垂直领域微调。整个流程包括数据准备、继续预训练、领域微调和模型评估四个主要步骤。

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [继续预训练](#3-继续预训练)
4. [垂直领域微调](#4-垂直领域微调)
5. [模型评估](#5-模型评估)
6. [一键运行](#6-一键运行)
7. [常见问题](#7-常见问题)

## 1. 环境准备

首先，确保你已经安装了所有必需的依赖项：

```bash
pip install torch transformers datasets tqdm regex numpy sacrebleu scikit-learn
```

然后，将以下文件放在同一目录下：

- `modeling_miniqwen.py`：MiniQwen模型实现
- `tokenization_miniqwen.py`：MiniQwen分词器实现
- `continue_pretraining.py`：继续预训练脚本
- `finetune_qa_datasets.py`：QA数据集微调脚本
- `evaluate_qa_models.py`：模型评估脚本
- `prepare_datasets.py`：数据准备脚本
- `run_training.sh`：一键运行脚本

## 2. 数据准备

使用`prepare_datasets.py`脚本准备预训练和微调所需的数据集：

```bash
# 准备所有数据集
python prepare_datasets.py --all

# 或者单独准备特定数据集
python prepare_datasets.py --wikipedia --arxiv
python prepare_datasets.py --sciq --commonsenseqa
```

参数说明：
- `--wikipedia`：准备Wikipedia数据集
- `--arxiv`：准备arXiv论文数据集
- `--sciq`：准备SciQ数据集
- `--commonsenseqa`：准备CommonsenseQA数据集
- `--all`：准备所有以上数据集
- `--streaming`：使用流式模式加载大型数据集（推荐用于大型预训练数据集）
- `--wikipedia_version`：指定Wikipedia数据集版本（默认为"20220301.en"）
- `--cache_dir`：指定数据集缓存目录

准备过程中，脚本会验证数据集的可访问性并提供样本示例。对于预训练数据集（Wikipedia和arXiv），建议使用流式模式以节省内存。

## 3. 继续预训练

使用`continue_pretraining.py`脚本在Wikipedia和arXiv数据集上对MiniQwen模型进行继续预训练：

```bash
python continue_pretraining.py \
  --model_path ./miniqwen_checkpoints/final_model \
  --dataset_name both \
  --output_dir ./miniqwen/pretrained \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --max_steps 10000 \
  --warmup_steps 1000 \
  --logging_steps 100 \
  --save_steps 1000 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --load_best_model_at_end \
  --streaming True \
  --max_seq_length 1024 \
  --block_size 1024 \
  --preprocessing_num_workers 4 \
  --overwrite_output_dir
```

关键参数说明：
- `--model_path`：初始MiniQwen模型的路径
- `--dataset_name`：要使用的数据集，可选值为"wikipedia"、"arxiv"或"both"
- `--output_dir`：模型保存目录
- `--streaming`：是否使用流式模式加载数据集（推荐用于大型数据集）
- `--max_steps`：训练的最大步数
- `--warmup_steps`：预热步数
- `--gradient_accumulation_steps`：梯度累积步数，可用于增加有效批次大小
- `--per_device_train_batch_size`：每个设备的训练批次大小
- `--max_seq_length`和`--block_size`：处理的最大序列长度

可以根据你的计算资源调整批次大小和梯度累积步数。如果显存有限，可以减少`per_device_train_batch_size`并增加`gradient_accumulation_steps`。

## 4. 垂直领域微调

使用`finetune_qa_datasets.py`脚本在特定任务数据集上微调模型：

### SciQ数据集微调

```bash
python finetune_qa_datasets.py \
  --model_path ./miniqwen/pretrained \
  --dataset_name sciq \
  --output_dir ./miniqwen/sciq_finetuned \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 5 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --load_best_model_at_end \
  --do_qa_format True \
  --max_seq_length 512 \
  --overwrite_output_dir
```

### CommonsenseQA数据集微调

```bash
python finetune_qa_datasets.py \
  --model_path ./miniqwen/pretrained \
  --dataset_name commonsenseqa \
  --output_dir ./miniqwen/commonsenseqa_finetuned \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 5 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --load_best_model_at_end \
  --do_qa_format True \
  --max_seq_length 512 \
  --overwrite_output_dir
```

关键参数说明：
- `--model_path`：预训练MiniQwen模型的路径
- `--dataset_name`：要使用的数据集，可选值为"sciq"或"commonsenseqa"
- `--do_qa_format`：是否将数据格式化为问答格式（推荐用于因果LM微调）
- `--num_train_epochs`：训练的轮数
- `--warmup_ratio`：预热步数占总步数的比例
- `--learning_rate`：学习率（微调通常使用比预训练更小的学习率）

## 5. 模型评估

使用`evaluate_qa_models.py`脚本评估微调后的模型在测试集上的性能：

```bash
# 评估SciQ模型
python evaluate_qa_models.py \
  --model_path ./miniqwen/sciq_finetuned \
  --dataset_name sciq \
  --output_dir ./miniqwen/evaluation/sciq \
  --max_eval_samples 100

# 评估CommonsenseQA模型
python evaluate_qa_models.py \
  --model_path ./miniqwen/commonsenseqa_finetuned \
  --dataset_name commonsenseqa \
  --output_dir ./miniqwen/evaluation/commonsenseqa \
  --max_eval_samples 100

# 综合评估
python evaluate_qa_models.py \
  --model_path ./miniqwen/pretrained \
  --dataset_name both \
  --output_dir ./miniqwen/evaluation/combined \
  --max_eval_samples 100
```

参数说明：
- `--model_path`：要评估的模型路径
- `--dataset_name`：要评估的数据集，可选值为"sciq"、"commonsenseqa"或"both"
- `--output_dir`：评估结果保存目录
- `--max_eval_samples`：用于评估的最大样本数（用较小的值进行快速测试）

评估结果将以JSON格式保存在指定目录中，包括准确率、每个样本的预测和正确答案等信息。

## 6. 一键运行

为了方便使用，我们提供了`run_training.sh`脚本，可以一键执行整个流程：

```bash
# 首先赋予脚本执行权限
chmod +x run_training.sh

# 运行脚本
./run_training.sh
```

这个脚本会自动执行以下步骤：
1. 继续预训练（使用Wikipedia和arXiv数据集）
2. SciQ数据集微调
3. CommonsenseQA数据集微调
4. 模型评估

如果你想修改训练参数，可以直接编辑脚本中的相应命令。

## 7. 常见问题

### Q: 如何在低内存环境中运行预训练？
A: 对于大型预训练数据集，建议使用流式模式（`--streaming True`）以节省内存。此外，可以减少批次大小（`--per_device_train_batch_size`）并增加梯度累积步数（`--gradient_accumulation_steps`）。

### Q: 如何在自定义数据集上微调？
A: 你可以参考`finetune_qa_datasets.py`脚本中的数据处理部分，为你的自定义数据集创建类似的处理函数。关键是将数据格式化为适合因果语言模型训练的形式。

### Q: 如何节省磁盘空间？
A: 可以在训练命令中添加`--fp16 True`参数启用混合精度训练，这不仅可以加速训练，还能减少模型检查点的大小。对于大型数据集，使用流式模式可以避免将整个数据集下载到磁盘。

### Q: 如何监控训练进度？
A: 可以使用TensorBoard来可视化训练过程。在训练命令中添加`--report_to tensorboard`参数，然后在另一个终端运行`tensorboard --logdir ./miniqwen`。

### Q: 如何调整提示模板？
A: 在`finetune_qa_datasets.py`脚本中，可以通过`--prompt_template`和`--choices_template`参数自定义提示模板的格式。

### Q: 继续预训练和微调应该训练多久？
A: 这取决于你的具体需求和计算资源。一般来说，继续预训练通常需要较长时间（数千到数万步），而微调则相对较短（几个epoch）。可以通过监控验证集上的性能来决定何时停止训练。

---

希望这个指南能帮助你顺利完成MiniQwen模型的预训练和微调。如果遇到任何问题，可以参考相关脚本的源代码或查阅Hugging Face Transformers的文档。
