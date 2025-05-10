
"""
MiniQwen 评估脚本
用于评估在SciQ和CommonsenseQA数据集上微调的MiniQwen模型
"""

import os
import logging
import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from modeling_miniqwen import MiniQwenForCausalLM
from tokenization_miniqwen import MiniQwenTokenizer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    模型相关参数
    """
    model_path: str = field(
        metadata={"help": "评估模型的路径"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "分词器的路径，如果与模型不同"}
    )

@dataclass
class DataArguments:
    """
    数据相关参数
    """
    dataset_name: str = field(
        default=None, metadata={"help": "要评估的数据集名称，可选值为 'sciq' 或 'commonsenseqa'"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的预处理数据集"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于数据预处理的进程数"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "用于评估的最大样本数"}
    )
    output_dir: str = field(
        default="./evaluation_results",
        metadata={"help": "保存评估结果的目录"}
    )

def evaluate_sciq(model, tokenizer, args):
    """
    评估模型在SciQ数据集上的性能
    """
    logger.info("加载SciQ数据集...")
    sciq_dataset = load_dataset("sciq")
    
    # 获取评估集
    if "validation" in sciq_dataset:
        eval_dataset = sciq_dataset["validation"]
    elif "dev" in sciq_dataset:
        eval_dataset = sciq_dataset["dev"]
    else:
        eval_dataset = sciq_dataset["test"]
    
    # 限制评估样本数量
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))
    
    logger.info(f"评估样本数量: {len(eval_dataset)}")
    
    # 设置模型为评估模式
    model.eval()
    
    correct = 0
    total = 0
    results = []
    
    for example in eval_dataset:
        question = example["question"]
        support = example["support"]
        correct_answer = example["correct_answer"]
        distractors = [example["distractor1"], example["distractor2"], example["distractor3"]]
        
        # 创建选项列表
        choices = [correct_answer] + distractors
        choice_labels = ["A", "B", "C", "D"]
        
        # 创建提示
        if support and support.strip():
            prompt = f"上下文: {support}\n问题: {question}\n选项:\n"
        else:
            prompt = f"问题: {question}\n选项:\n"
        
        for i, (label, choice) in enumerate(zip(choice_labels, choices)):
            prompt += f"({label}) {choice}\n"
        
        prompt += "答案是:"
        
        # 令牌化
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                num_return_sequences=1,
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取预测的答案
        prediction = generated_text[len(prompt):].strip().upper()
        
        # 检查预测是否匹配正确的选择标签
        if prediction.startswith("A"):  # 正确答案总是A
            correct += 1
        
        total += 1
        
        # 保存结果
        result = {
            "question": question,
            "support": support,
            "choices": {label: choice for label, choice in zip(choice_labels, choices)},
            "correct_answer": "A",
            "prediction": prediction,
            "is_correct": prediction.startswith("A")
        }
        results.append(result)
        
        if total % 10 == 0:
            logger.info(f"已评估 {total} 个样本，当前准确率: {correct / total:.4f}")
    
    # 计算总体准确率
    accuracy = correct / total
    logger.info(f"最终准确率: {accuracy:.4f}")
    
    # 保存评估结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "sciq_evaluation_results.json")
    
    with open(output_file, "w") as f:
        json.dump({
            "dataset": "SciQ",
            "accuracy": accuracy,
            "total_samples": total,
            "correct_samples": correct,
            "results": results
        }, f, indent=2)
    
    logger.info(f"评估结果已保存到 {output_file}")
    
    return accuracy

def evaluate_commonsenseqa(model, tokenizer, args):
    """
    评估模型在CommonsenseQA数据集上的性能
    """
    logger.info("加载CommonsenseQA数据集...")
    commonsenseqa_dataset = load_dataset("commonsenseqa")
    
    # 获取评估集
    if "validation" in commonsenseqa_dataset:
        eval_dataset = commonsenseqa_dataset["validation"]
    elif "dev" in commonsenseqa_dataset:
        eval_dataset = commonsenseqa_dataset["dev"]
    else:
        eval_dataset = commonsenseqa_dataset["test"]
    
    # 限制评估样本数量
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))
    
    logger.info(f"评估样本数量: {len(eval_dataset)}")
    
    # 设置模型为评估模式
    model.eval()
    
    correct = 0
    total = 0
    results = []
    
    for example in eval_dataset:
        question = example["question"]
        choices = example["choices"]
        answer_key = example["answerKey"]
        
        # 提取选项
        choice_texts = [choice["text"] for choice in choices["text"]]
        choice_labels = [choice["label"] for choice in choices["label"]]
        
        # 创建提示
        prompt = f"问题: {question}\n选项:\n"
        
        for label, text in zip(choice_labels, choice_texts):
            prompt += f"({label}) {text}\n"
        
        prompt += "答案是:"
        
        # 令牌化
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                num_return_sequences=1,
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取预测的答案
        prediction = generated_text[len(prompt):].strip().upper()
        
        # 检查预测是否匹配正确的答案
        if any(label.upper() in prediction for label in [answer_key]):
            correct += 1
        
        total += 1
        
        # 保存结果
        result = {
            "question": question,
            "choices": {label: text for label, text in zip(choice_labels, choice_texts)},
            "correct_answer": answer_key,
            "prediction": prediction,
            "is_correct": any(label.upper() in prediction for label in [answer_key])
        }
        results.append(result)
        
        if total % 10 == 0:
            logger.info(f"已评估 {total} 个样本，当前准确率: {correct / total:.4f}")
    
    # 计算总体准确率
    accuracy = correct / total
    logger.info(f"最终准确率: {accuracy:.4f}")
    
    # 保存评估结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "commonsenseqa_evaluation_results.json")
    
    with open(output_file, "w") as f:
        json.dump({
            "dataset": "CommonsenseQA",
            "accuracy": accuracy,
            "total_samples": total,
            "correct_samples": correct,
            "results": results
        }, f, indent=2)
    
    logger.info(f"评估结果已保存到 {output_file}")
    
    return accuracy

def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    
    # 设置随机种子
    set_seed(42)
    
    # 加载模型
    logger.info(f"从 {model_args.model_path} 加载模型...")
    model = MiniQwenForCausalLM.from_pretrained(model_args.model_path)
    
    # 加载分词器
    tokenizer_path = model_args.tokenizer_path or model_args.model_path
    logger.info(f"从 {tokenizer_path} 加载分词器...")
    tokenizer = MiniQwenTokenizer.from_pretrained(tokenizer_path)
    
    # 将模型移至设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"使用设备: {device}")
    
    # 根据数据集名称评估模型
    if data_args.dataset_name.lower() == "sciq":
        accuracy = evaluate_sciq(model, tokenizer, data_args)
    elif data_args.dataset_name.lower() == "commonsenseqa":
        accuracy = evaluate_commonsenseqa(model, tokenizer, data_args)
    elif data_args.dataset_name.lower() == "both":
        sciq_accuracy = evaluate_sciq(model, tokenizer, data_args)
        commonsenseqa_accuracy = evaluate_commonsenseqa(model, tokenizer, data_args)
        
        # 输出综合结果
        logger.info(f"SciQ 准确率: {sciq_accuracy:.4f}")
        logger.info(f"CommonsenseQA 准确率: {commonsenseqa_accuracy:.4f}")
        logger.info(f"平均准确率: {(sciq_accuracy + commonsenseqa_accuracy) / 2:.4f}")
        
        # 保存综合结果
        summary_file = os.path.join(data_args.output_dir, "evaluation_summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "sciq_accuracy": sciq_accuracy,
                "commonsenseqa_accuracy": commonsenseqa_accuracy,
                "average_accuracy": (sciq_accuracy + commonsenseqa_accuracy) / 2
            }, f, indent=2)
        
        logger.info(f"评估摘要已保存到 {summary_file}")
    else:
        raise ValueError(f"不支持的数据集名称: {data_args.dataset_name}，请选择 'sciq'、'commonsenseqa' 或 'both'")

if __name__ == "__main__":
    main()
