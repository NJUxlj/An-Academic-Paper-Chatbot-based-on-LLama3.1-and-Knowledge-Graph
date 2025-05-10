
"""
MiniQwen 垂直领域微调脚本
用于在SciQ和CommonsenseQA数据集上微调MiniQwen模型
使用Hugging Face Trainer类进行训练
"""

import os
import json
import logging
import math
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from modeling_miniqwen import MiniQwenForCausalLM, MiniQwenConfig
from tokenization_miniqwen import MiniQwenTokenizer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    模型相关参数
    """
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "预训练MiniQwen模型的路径"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "分词器的路径，如果与模型不同的话"}
    )

@dataclass
class DataTrainingArguments:
    """
    数据相关参数
    """
    dataset_name: str = field(
        default=None, metadata={"help": "要使用的数据集名称，可选值为 'sciq' 或 'commonsenseqa'"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "处理的最大序列长度"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "要用于数据预处理的进程数"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "用于训练的最大样本数"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "用于评估的最大样本数"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的预处理数据集"}
    )
    do_qa_format: bool = field(
        default=True,
        metadata={"help": "是否将数据格式化为问答格式（推荐用于因果LM微调）"}
    )
    prompt_template: str = field(
        default="问题: {question}\n选项: {choices}\n答案是:",
        metadata={"help": "用于格式化输入的提示模板"}
    )
    choices_template: str = field(
        default="({label}) {text}",
        metadata={"help": "用于格式化选项的模板"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "是否将所有样本填充到max_seq_length"}
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "最大序列长度，默认为模型的上下文窗口大小"}
    )

@dataclass
class DataCollatorForMultipleChoice:
    """
    用于多选题的数据整理器
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        
        flattened_features = [
            {k: v[i] for k, v in feature.items() if k != "label"}
            for feature in features
            for i in range(num_choices)
        ]
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor([feature["label"] for feature in features], dtype=torch.int64)
        
        return batch

@dataclass
class DataCollatorForCausalQA:
    """
    用于因果语言模型问答格式的数据整理器
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        # 将数据重新格式化为因果语言模型的输入格式
        for feature in features:
            # 确保我们有输入和标签
            if "labels" not in feature:
                feature["labels"] = feature["input_ids"].copy()
            
            # 找到分隔符的位置（答案的开始）
            sep_pos = feature["input_ids"].index(self.tokenizer.sep_token_id) if self.tokenizer.sep_token_id in feature["input_ids"] else -1
            
            if sep_pos > 0:
                # 将分隔符之前的token的标签设置为-100（不计算损失）
                feature["labels"][:sep_pos+1] = [-100] * (sep_pos+1)
        
        # 使用tokenizer的填充功能
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # 确保标签中的填充token也是-100
        if "labels" in batch:
            pad_token_id = self.tokenizer.pad_token_id
            batch["labels"][batch["labels"] == pad_token_id] = -100
            
        return batch

def load_and_process_sciq(args, tokenizer):
    """
    加载和处理SciQ数据集
    """
    logger.info("加载SciQ数据集...")
    
    # 加载数据集
    sciq_dataset = load_dataset("sciq")
    
    if args.do_qa_format:
        # 将数据集格式化为问答格式，适用于因果语言模型
        logger.info("将SciQ数据集格式化为问答格式...")
        
        def format_as_qa(examples):
            formatted_examples = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for question, support, correct_answer, distractor1, distractor2, distractor3 in zip(
                examples["question"],
                examples["support"],
                examples["correct_answer"],
                examples["distractor1"],
                examples["distractor2"],
                examples["distractor3"]
            ):
                # 创建选项列表
                choices = [correct_answer, distractor1, distractor2, distractor3]
                choice_labels = ["A", "B", "C", "D"]
                formatted_choices = "\n".join([
                    args.choices_template.format(label=label, text=choice)
                    for label, choice in zip(choice_labels, choices)
                ])
                
                # 创建提示和答案
                if support and support.strip():
                    prompt = f"上下文: {support}\n" + args.prompt_template.format(
                        question=question, choices=formatted_choices
                    )
                else:
                    prompt = args.prompt_template.format(
                        question=question, choices=formatted_choices
                    )
                answer = " A"  # 正确答案总是A
                
                # 将提示和答案连接起来，并用特殊token分隔
                if tokenizer.sep_token:
                    full_text = prompt + tokenizer.sep_token + answer
                else:
                    full_text = prompt + " " + answer
                
                # 令牌化
                encoded = tokenizer(
                    full_text,
                    max_length=args.max_length,
                    padding="max_length" if args.pad_to_max_length else False,
                    truncation=True,
                )
                
                formatted_examples["input_ids"].append(encoded["input_ids"])
                formatted_examples["attention_mask"].append(encoded["attention_mask"])
                
                # 创建标签
                labels = encoded["input_ids"].copy()
                
                formatted_examples["labels"].append(labels)
            
            return formatted_examples
        
        # 转换数据集
        processed_datasets = {}
        for split in sciq_dataset:
            processed_datasets[split] = sciq_dataset[split].map(
                format_as_qa,
                batched=True,
                remove_columns=sciq_dataset[split].column_names,
                desc=f"处理SciQ {split}集",
                num_proc=args.preprocessing_num_workers,
            )
    else:
        # 将数据集格式化为多选题格式
        logger.info("将SciQ数据集格式化为多选题格式...")
        
        def format_as_multiple_choice(examples):
            first_sentences = []
            second_sentences = []
            labels = []
            
            for question, support, correct_answer, distractor1, distractor2, distractor3 in zip(
                examples["question"],
                examples["support"],
                examples["correct_answer"],
                examples["distractor1"],
                examples["distractor2"],
                examples["distractor3"]
            ):
                if support and support.strip():
                    context = f"上下文: {support}\n问题: {question}"
                else:
                    context = f"问题: {question}"
                
                choices = [correct_answer, distractor1, distractor2, distractor3]
                first_sentences.extend([context] * 4)
                second_sentences.extend(choices)
                labels.append(0)  # 正确答案总是位于第一位
            
            return {
                "first_sentences": first_sentences,
                "second_sentences": second_sentences,
                "labels": labels
            }
        
        # 转换数据集
        processed_datasets = {}
        for split in sciq_dataset:
            formatted = sciq_dataset[split].map(
                format_as_multiple_choice,
                batched=True,
                remove_columns=sciq_dataset[split].column_names,
                desc=f"处理SciQ {split}集",
                num_proc=args.preprocessing_num_workers,
            )
            
            # 令牌化
            def tokenize_multiple_choice(examples):
                num_examples = len(examples["first_sentences"]) // 4
                result = {
                    "input_ids": [],
                    "attention_mask": [],
                    "label": []
                }
                
                for i in range(num_examples):
                    first_sentences = examples["first_sentences"][i*4:(i+1)*4]
                    second_sentences = examples["second_sentences"][i*4:(i+1)*4]
                    
                    inputs = tokenizer(
                        first_sentences,
                        second_sentences,
                        max_length=args.max_seq_length,
                        padding="max_length" if args.pad_to_max_length else False,
                        truncation=True,
                    )
                    
                    item = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "label": examples["labels"][i]
                    }
                    
                    for k, v in item.items():
                        result[k].append(v)
                
                return result
            
            processed_datasets[split] = formatted.map(
                tokenize_multiple_choice,
                batched=True,
                remove_columns=formatted.column_names,
                desc=f"令牌化SciQ {split}集",
                num_proc=args.preprocessing_num_workers,
            )
    
    return processed_datasets

def load_and_process_commonsenseqa(args, tokenizer):
    """
    加载和处理CommonsenseQA数据集
    """
    logger.info("加载CommonsenseQA数据集...")
    
    # 加载数据集
    commonsenseqa_dataset = load_dataset("commonsenseqa")
    
    if args.do_qa_format:
        # 将数据集格式化为问答格式，适用于因果语言模型
        logger.info("将CommonsenseQA数据集格式化为问答格式...")
        
        def format_as_qa(examples):
            formatted_examples = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for question, choices, answer_key in zip(
                examples["question"],
                examples["choices"],
                examples["answerKey"]
            ):
                # 创建选项列表
                choice_texts = [choice["text"] for choice in choices["text"]]
                choice_labels = [choice["label"] for choice in choices["label"]]
                formatted_choices = "\n".join([
                    args.choices_template.format(label=label, text=text)
                    for label, text in zip(choice_labels, choice_texts)
                ])
                
                # 创建提示和答案
                prompt = args.prompt_template.format(
                    question=question, choices=formatted_choices
                )
                
                # 查找正确答案的索引
                answer_idx = choice_labels.index(answer_key)
                answer = f" {choice_labels[answer_idx]}"
                
                # 将提示和答案连接起来，并用特殊token分隔
                if tokenizer.sep_token:
                    full_text = prompt + tokenizer.sep_token + answer
                else:
                    full_text = prompt + " " + answer
                
                # 令牌化
                encoded = tokenizer(
                    full_text,
                    max_length=args.max_length,
                    padding="max_length" if args.pad_to_max_length else False,
                    truncation=True,
                )
                
                formatted_examples["input_ids"].append(encoded["input_ids"])
                formatted_examples["attention_mask"].append(encoded["attention_mask"])
                
                # 创建标签
                labels = encoded["input_ids"].copy()
                
                formatted_examples["labels"].append(labels)
            
            return formatted_examples
        
        # 转换数据集
        processed_datasets = {}
        for split in commonsenseqa_dataset:
            processed_datasets[split] = commonsenseqa_dataset[split].map(
                format_as_qa,
                batched=True,
                remove_columns=commonsenseqa_dataset[split].column_names,
                desc=f"处理CommonsenseQA {split}集",
                num_proc=args.preprocessing_num_workers,
            )
    else:
        # 将数据集格式化为多选题格式
        logger.info("将CommonsenseQA数据集格式化为多选题格式...")
        
        def format_as_multiple_choice(examples):
            first_sentences = []
            second_sentences = []
            labels = []
            
            for question, choices, answer_key in zip(
                examples["question"],
                examples["choices"],
                examples["answerKey"]
            ):
                context = f"问题: {question}"
                
                choice_texts = [choice["text"] for choice in choices["text"]]
                choice_labels = [choice["label"] for choice in choices["label"]]
                answer_idx = choice_labels.index(answer_key)
                
                first_sentences.extend([context] * len(choice_texts))
                second_sentences.extend(choice_texts)
                labels.append(answer_idx)
            
            return {
                "first_sentences": first_sentences,
                "second_sentences": second_sentences,
                "labels": labels
            }
        
        # 转换数据集
        processed_datasets = {}
        for split in commonsenseqa_dataset:
            formatted = commonsenseqa_dataset[split].map(
                format_as_multiple_choice,
                batched=True,
                remove_columns=commonsenseqa_dataset[split].column_names,
                desc=f"处理CommonsenseQA {split}集",
                num_proc=args.preprocessing_num_workers,
            )
            
            # 令牌化
            def tokenize_multiple_choice(examples):
                num_choices = 5  # CommonsenseQA有5个选项
                num_examples = len(examples["first_sentences"]) // num_choices
                result = {
                    "input_ids": [],
                    "attention_mask": [],
                    "label": []
                }
                
                for i in range(num_examples):
                    first_sentences = examples["first_sentences"][i*num_choices:(i+1)*num_choices]
                    second_sentences = examples["second_sentences"][i*num_choices:(i+1)*num_choices]
                    
                    inputs = tokenizer(
                        first_sentences,
                        second_sentences,
                        max_length=args.max_seq_length,
                        padding="max_length" if args.pad_to_max_length else False,
                        truncation=True,
                    )
                    
                    item = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "label": examples["labels"][i]
                    }
                    
                    for k, v in item.items():
                        result[k].append(v)
                
                return result
            
            processed_datasets[split] = formatted.map(
                tokenize_multiple_choice,
                batched=True,
                remove_columns=formatted.column_names,
                desc=f"令牌化CommonsenseQA {split}集",
                num_proc=args.preprocessing_num_workers,
            )
    
    return processed_datasets

def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 加载分词器
    if model_args.tokenizer_path:
        tokenizer = MiniQwenTokenizer.from_pretrained(model_args.tokenizer_path)
    else:
        tokenizer = MiniQwenTokenizer.from_pretrained(model_args.model_path)
    
    # 加载模型
    if model_args.model_path:
        logger.info(f"从{model_args.model_path}加载模型...")
        model = MiniQwenForCausalLM.from_pretrained(model_args.model_path)
    else:
        raise ValueError("必须提供预训练模型路径进行微调")
    
    # 根据数据集名称加载和处理数据
    if data_args.dataset_name.lower() == "sciq":
        processed_datasets = load_and_process_sciq(data_args, tokenizer)
    elif data_args.dataset_name.lower() == "commonsenseqa":
        processed_datasets = load_and_process_commonsenseqa(data_args, tokenizer)
    else:
        raise ValueError(f"不支持的数据集名称: {data_args.dataset_name}，请选择 'sciq' 或 'commonsenseqa'")
    
    # 获取训练和评估数据集
    train_dataset = processed_datasets["train"]
    if "validation" in processed_datasets:
        eval_dataset = processed_datasets["validation"]
    elif "dev" in processed_datasets:
        eval_dataset = processed_datasets["dev"]
    else:
        eval_dataset = None
    
    # 限制训练和评估样本数量（如果指定）
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
    
    if eval_dataset and data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
    
    # 创建数据整理器
    if data_args.do_qa_format:
        data_collator = DataCollatorForCausalQA(
            tokenizer=tokenizer,
            padding="max_length" if data_args.pad_to_max_length else True,
            max_length=data_args.max_length,
        )
    else:
        data_collator = DataCollatorForMultipleChoice(
            tokenizer=tokenizer,
            padding="max_length" if data_args.pad_to_max_length else True,
            max_length=data_args.max_seq_length,
        )
    
    # 定义计算指标的函数
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        if data_args.do_qa_format:
            # 对于因果语言模型，我们需要从生成结果中提取预测的答案
            # 简化起见，我们只计算损失和困惑度
            return {"perplexity": math.exp(preds[0])}
        else:
            # 对于多选，我们可以计算准确率
            preds = preds.argmax(dim=1)
            accuracy = (preds == labels).sum() / len(labels)
            return {"accuracy": float(accuracy)}
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if not data_args.do_qa_format else None,
    )
    
    # 训练
    if training_args.do_train:
        logger.info("*** 开始训练 ***")
        
        # 检查是否有上次的检查点
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint := get_last_checkpoint(training_args.output_dir):
            checkpoint = last_checkpoint
            logger.info(f"恢复训练，从检查点 {checkpoint} 继续")
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # 保存模型
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # 评估
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** 开始评估 ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        
        # 如果使用因果语言模型格式，计算困惑度
        if data_args.do_qa_format:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # 保存最终模型和分词器
    if training_args.output_dir:
        logger.info(f"保存最终模型和分词器到 {training_args.output_dir}")
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info(f"微调 {data_args.dataset_name} 数据集完成!")

if __name__ == "__main__":
    main()
