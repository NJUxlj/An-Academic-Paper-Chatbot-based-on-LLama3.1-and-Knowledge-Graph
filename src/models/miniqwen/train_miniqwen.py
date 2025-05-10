
"""
MiniQwen 训练脚本示例
"""

import argparse
import os
import math
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import get_scheduler, set_seed

from modeling_miniqwen import MiniQwenForCausalLM, MiniQwenConfig
from tokenization_miniqwen import MiniQwenTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="训练MiniQwen语言模型")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="如果提供，从该路径加载模型"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None, 
        help="如果提供，从该路径加载分词器"
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        required=True, 
        help="训练数据文件路径（文本文件）"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./miniqwen_checkpoints", 
        help="模型和分词器保存路径"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="训练批次大小"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="权重衰减"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="训练轮数"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1, 
        help="梯度累积步数"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=1024, 
        help="最大序列长度"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=500, 
        help="每多少步保存一次模型"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=500, 
        help="预热步数"
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=100, 
        help="每多少步记录一次日志"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子"
    )
    return parser.parse_args()

def create_fresh_model():
    """
    创建一个全新的MiniQwen模型
    """
    print("创建新的MiniQwen模型...")
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
    print(f"模型大小: {model_size:.2f} MB")
    
    return model

def create_tokenizer(tokenizer_path=None):
    """
    创建或加载MiniQwen分词器
    """
    if tokenizer_path:
        print(f"从{tokenizer_path}加载分词器...")
        tokenizer = MiniQwenTokenizer.from_pretrained(tokenizer_path)
    else:
        print("创建基于BERT的分词器...")
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

def create_dataset(file_path, tokenizer, max_seq_length):
    """
    创建用于训练的数据集
    """
    print(f"从{file_path}创建数据集...")
    
    # 创建TextDataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=max_seq_length,
    )
    
    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用掩码语言建模，而是因果语言建模
    )
    
    return dataset, data_collator

def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建或加载分词器
    tokenizer = create_tokenizer(args.tokenizer_path)
    
    # 创建数据集
    dataset, data_collator = create_dataset(args.train_file, tokenizer, args.max_seq_length)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    
    # 创建或加载模型
    if args.model_path:
        print(f"从{args.model_path}加载模型...")
        model = MiniQwenForCausalLM.from_pretrained(args.model_path)
    else:
        model = create_fresh_model()
    
    model.to(device)
    
    # 准备优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # 计算训练总步数
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    
    # 创建学习率调度器
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # 准备训练
    print(f"开始训练，总步数: {total_steps}")
    progress_bar = tqdm(range(total_steps))
    
    # 训练参数
    global_step = 0
    tr_loss = 0.0
    model.train()
    
    # 训练循环
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            # 将batch移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            
            # 损失缩放（梯度累积）
            loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            tr_loss += loss.item()
            
            # 每累积一定步数后更新参数
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # 参数更新
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
                
                # 记录日志
                if global_step % args.logging_steps == 0:
                    avg_loss = tr_loss / args.logging_steps
                    print(f"Step {global_step}, Loss: {avg_loss:.4f}")
                    tr_loss = 0.0
                
                # 保存模型
                if global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    print(f"保存模型到 {checkpoint_dir}")
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
        
        # 计算每轮训练时间
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} 完成，用时: {epoch_time:.2f}秒")
    
    # 保存最终模型
    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    print(f"保存最终模型到 {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print("训练完成!")

if __name__ == "__main__":
    main()
