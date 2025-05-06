# src/models/trainer.py
import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from typing import Dict, List, Any, Optional, Tuple
from src.configs.config import Config
from src.models.entity_extractor.bert_bilstm_crf import BERT_BiLSTM_CRF

logger = logging.getLogger(__name__)

class NERDataset(Dataset):
    """
    NER数据集类
    """
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer: BertTokenizer, max_length: int = 128):
        """
        初始化数据集
        
        Args:
            data: 数据列表
            tokenizer: BERT分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
        
        # 标签映射
        self.label_map = {
            "O": 0,
            "B-MODEL": 1, "I-MODEL": 2,
            "B-METRIC": 3, "I-METRIC": 4,
            "B-DATASET": 5, "I-DATASET": 6,
            "B-METHOD": 7, "I-METHOD": 8,
            "B-TASK": 9, "I-TASK": 10,
            "B-FRAMEWORK": 11, "I-FRAMEWORK": 12
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item["labels"]
        
        # 转换为BERT输入
        bert_tokens = []
        bert_labels = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # 使用tokenizer处理token
            subwords = self.tokenizer.tokenize(token)
            
            if not subwords:
                subwords = [self.tokenizer.unk_token]
            
            bert_tokens.extend(subwords)
            
            # 对于分成多个子词的token，第一个子词保持原标签，其余子词使用相同的标签
            bert_labels.extend([self.label_map[label]] * len(subwords))
        
        # 截断
        if len(bert_tokens) > self.max_length - 2:  # 考虑[CLS]和[SEP]
            bert_tokens = bert_tokens[:self.max_length - 2]
            bert_labels = bert_labels[:self.max_length - 2]
        
        # 添加[CLS]和[SEP]
        bert_tokens = [self.tokenizer.cls_token] + bert_tokens + [self.tokenizer.sep_token]
        bert_labels = [0] + bert_labels + [0]  # CLS和SEP的标签设为O
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        
        # 填充
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        bert_labels += [0] * padding_length  # 填充标签为O
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(bert_labels)
        }

class NERTrainer:
    """
    NER模型训练器
    """
    
    def __init__(self, 
                 model_path: str = Config.MODEL_CONFIG["bert_ner"]["model_path"],
                 device: str = Config.MODEL_CONFIG["qwen"]["device"],
                 batch_size: int = Config.MODEL_CONFIG["bert_ner"]["batch_size"],
                 max_length: int = Config.MODEL_CONFIG["bert_ner"]["max_length"]):
        """
        初始化训练器
        
        Args:
            model_path: 模型路径
            device: 设备
            batch_size: 批次大小
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 初始化分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        
        # 标签数量
        self.num_labels = 13  # O + 6个实体类型的B/I
        
        # 初始化模型
        self.model = BERT_BiLSTM_CRF(
            bert_model=self.model_path,
            num_tags=self.num_labels
        )
        self.model.to(self.device)
        
        logger.info(f"初始化NER训练器: model_path={model_path}, device={self.device}")
    
    def train(self, 
              train_data_path: str, 
              val_data_path: Optional[str] = None,
              epochs: int = 5,
              learning_rate: float = 5e-5,
              warmup_steps: int = 0,
              weight_decay: float = 0.01,
              save_dir: Optional[str] = None):
        """
        训练模型
        
        Args:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径（可选）
            epochs: 训练轮数
            learning_rate: 学习率
            warmup_steps: 预热步数
            weight_decay: 权重衰减
            save_dir: 模型保存目录
        """
        # 加载训练数据
        with open(train_data_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        
        # 创建数据集
        train_dataset = NERDataset(train_data, self.tokenizer, self.max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 验证数据
        val_dataloader = None
        if val_data_path:
            with open(val_data_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            
            val_dataset = NERDataset(val_data, self.tokenizer, self.max_length)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 优化器
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练循环
        logger.info(f"开始训练: epochs={epochs}, total_steps={total_steps}")
        
        best_val_loss = float("inf")
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0
            
            # 进度条
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # 将数据移到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                loss = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 更新参数
                optimizer.step()
                
                # 更新学习率
                scheduler.step()
                
                # 累计损失
                train_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({"loss": loss.item()})
            
            # 平均训练损失
            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Avg. Train Loss: {avg_train_loss:.4f}")
            
            # 验证
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_dir:
                        self.save_model(os.path.join(save_dir, "best_model"))
                        logger.info(f"保存最佳模型: val_loss={val_loss:.4f}")
            
            # 每个epoch后保存模型
            if save_dir:
                self.save_model(os.path.join(save_dir, f"epoch_{epoch+1}"))
        
        # 保存最终模型
        if save_dir:
            self.save_model(os.path.join(save_dir, "final_model"))
            logger.info("训练完成，保存最终模型")
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            平均损失
        """
        # 评估模式
        self.model.eval()
        eval_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 将数据移到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                loss = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # 累计损失
                eval_loss += loss.item()
        
        # 平均损失
        avg_eval_loss = eval_loss / len(dataloader)
        
        return avg_eval_loss
    
    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型
        torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # 保存配置
        self.model.bert.config.save_pretrained(save_path)
        
        # 保存分词器
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"模型保存到: {save_path}")
    
    def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        # 加载模型
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        logger.info(f"从 {model_path} 加载模型")