
"""
MiniQwen 继续预训练脚本
使用Wikipedia和arXiv数据集进行继续预训练
使用Hugging Face Trainer类进行训练
"""

import os
import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

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
        metadata={"help": "MiniQwen模型的路径。如果不提供，将从头初始化一个新模型。"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "分词器的路径。如果不提供，将基于BERT词表创建一个新分词器。"}
    )

@dataclass
class DataTrainingArguments:
    """
    数据相关参数
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集名称，可选值为 'wikipedia', 'arxiv', 'both'"}
    )
    wikipedia_version: Optional[str] = field(
        default="20220301.en", metadata={"help": "Wikipedia数据集版本"}
    )
    arxiv_subset: Optional[str] = field(
        default="arxiv:2007.08259", metadata={"help": "arXiv数据集子集"}
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={"help": "预处理输入序列的最大长度"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "要用于数据预处理的进程数"}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "掩码语言建模的掩码概率"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "是否将每一行视为一个序列"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "用于训练的最大样本数"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "用于评估的最大样本数"}
    )
    streaming: bool = field(
        default=False, metadata={"help": "启用数据集流式加载以节省内存"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "用于训练的块大小。如果为None，将默认为模型的最大上下文大小"
        },
    )

def create_fresh_model():
    """
    创建一个全新的MiniQwen模型
    """
    logger.info("创建新的MiniQwen模型...")
    # 配置模型参数
    config = MiniQwenConfig(
        vocab_size=30522,  # BERT词表大小
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=4,
        hidden_act="swiglu",
        max_position_embeddings=1024,
    )
    
    # 初始化模型
    model = MiniQwenForCausalLM(config)
    
    # 计算模型大小
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    logger.info(f"模型大小: {model_size:.2f} MB")
    
    return model

def create_tokenizer(tokenizer_path=None):
    """
    创建或加载MiniQwen分词器
    """
    if tokenizer_path:
        logger.info(f"从{tokenizer_path}加载分词器...")
        tokenizer = MiniQwenTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info("创建基于BERT的分词器...")
        # 初始化我们的分词器
        tokenizer = MiniQwenTokenizer(
            vocab_file=None,  # 会自动从BERT加载词表
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
    
    return tokenizer

def load_datasets(args, tokenizer):
    """
    加载并预处理数据集
    """
    datasets_dict = {}
    
    if args.dataset_name in ["wikipedia", "both"]:
        logger.info(f"加载Wikipedia数据集 (版本: {args.wikipedia_version})...")
        wiki_dataset = load_dataset("wikipedia", args.wikipedia_version, streaming=args.streaming)
        
        # 提取并清理文本
        def extract_wiki_text(examples):
            return {"text": [doc for doc in examples["text"]]}
        
        if args.streaming:
            wiki_dataset = wiki_dataset.map(extract_wiki_text, batched=True)
        else:
            wiki_dataset = wiki_dataset.map(
                extract_wiki_text, 
                batched=True, 
                num_proc=args.preprocessing_num_workers
            )
        
        datasets_dict["wikipedia"] = wiki_dataset
    
    if args.dataset_name in ["arxiv", "both"]:
        logger.info(f"加载arXiv论文数据集...")
        arxiv_dataset = load_dataset("arxiv_dataset", streaming=args.streaming)
        
        # 提取论文摘要和标题
        def extract_arxiv_text(examples):
            texts = []
            for abstract, title in zip(examples["abstract"], examples["title"]):
                text = f"Title: {title}\nAbstract: {abstract}"
                texts.append(text)
            return {"text": texts}
        
        if args.streaming:
            arxiv_dataset = arxiv_dataset.map(extract_arxiv_text, batched=True)
        else:
            arxiv_dataset = arxiv_dataset.map(
                extract_arxiv_text, 
                batched=True, 
                num_proc=args.preprocessing_num_workers
            )
        
        datasets_dict["arxiv"] = arxiv_dataset
    
    # 合并数据集（如果需要）
    if args.dataset_name == "both":
        if args.streaming:
            # 对于流式数据集，我们可以交替采样
            combined_dataset = datasets.interleave_datasets([
                datasets_dict["wikipedia"]["train"],
                datasets_dict["arxiv"]["train"]
            ])
            datasets_dict["combined"] = {"train": combined_dataset}
        else:
            # 对于内存中的数据集，我们可以连接它们
            combined_train = datasets.concatenate_datasets([
                datasets_dict["wikipedia"]["train"],
                datasets_dict["arxiv"]["train"]
            ])
            
            # 如果有验证集的话
            if "validation" in datasets_dict["wikipedia"] and "validation" in datasets_dict["arxiv"]:
                combined_valid = datasets.concatenate_datasets([
                    datasets_dict["wikipedia"]["validation"],
                    datasets_dict["arxiv"]["validation"]
                ])
                datasets_dict["combined"] = {
                    "train": combined_train,
                    "validation": combined_valid
                }
            else:
                # 如果没有验证集，我们可以从训练集中划分一部分
                train_test_split = combined_train.train_test_split(test_size=0.05)
                datasets_dict["combined"] = {
                    "train": train_test_split["train"],
                    "validation": train_test_split["test"]
                }
    
    # 确定使用哪个数据集
    if args.dataset_name == "both":
        dataset_to_use = datasets_dict["combined"]
    elif args.dataset_name == "wikipedia":
        dataset_to_use = datasets_dict["wikipedia"]
    else:  # arxiv
        dataset_to_use = datasets_dict["arxiv"]
    
    # 限制训练样本数量（如果指定）
    if args.max_train_samples is not None and not args.streaming:
        dataset_to_use["train"] = dataset_to_use["train"].select(range(args.max_train_samples))
    
    # 限制评估样本数量（如果指定）
    if "validation" in dataset_to_use and args.max_eval_samples is not None and not args.streaming:
        dataset_to_use["validation"] = dataset_to_use["validation"].select(range(args.max_eval_samples))
    
    # 令牌化数据集
    block_size = args.block_size
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"tokenizer的最大长度是{tokenizer.model_max_length}，但我们将使用block_size={block_size}"
            )
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=True)
    
    def group_texts(examples):
        # 连接所有文本
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # 我们丢弃小批量余数，以便可以将数据整齐地分块
        total_length = (total_length // block_size) * block_size
        
        # 分块并创建 input_ids 和 attention_mask
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        result["labels"] = result["input_ids"].copy()
        return result
    
    # 令牌化数据集
    logger.info("令牌化和分块数据集...")
    if args.streaming:
        tokenized_dataset = dataset_to_use.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True
        )
    else:
        tokenized_dataset = dataset_to_use.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=["text"],
            load_from_cache_file=not args.streaming,
            desc="使用tokenizer处理数据集",
        )
        
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.streaming,
            desc=f"将令牌分组为长度为{block_size}的样本",
        )
    
    return lm_dataset

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
    
    # 加载或创建模型
    if model_args.model_path:
        logger.info(f"从{model_args.model_path}加载模型...")
        model = MiniQwenForCausalLM.from_pretrained(model_args.model_path)
    else:
        model = create_fresh_model()
    
    # 创建或加载分词器
    tokenizer = create_tokenizer(model_args.tokenizer_path)
    
    # 加载数据集
    train_dataset = load_datasets(data_args, tokenizer)
    
    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 我们用于因果语言建模而不是掩码语言建模
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"] if not data_args.streaming else train_dataset,
        eval_dataset=train_dataset["validation"] if "validation" in train_dataset and not data_args.streaming else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
    if training_args.do_eval:
        logger.info("*** 开始评估 ***")
        metrics = trainer.evaluate()
        
        # 计算困惑度
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # 保存模型和分词器
    if training_args.output_dir:
        logger.info(f"保存模型和分词器到 {training_args.output_dir}")
        
        # 保存模型
        model.save_pretrained(training_args.output_dir)
        
        # 保存分词器
        tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("预训练完成!")

if __name__ == "__main__":
    main()
