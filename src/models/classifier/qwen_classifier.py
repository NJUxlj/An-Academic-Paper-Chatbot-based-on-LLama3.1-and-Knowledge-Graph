
# src/models/classifier/qwen_classifier.py
import os
import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Any, Optional, Union
import json
from src.configs.config import Config

logger = logging.getLogger(__name__)

class PaperFrameworkClassifier:
    """
    使用Qwen2.5模型对论文框架进行分类
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化分类器
        
        Args:
            model_path: 模型路径，默认使用配置文件中的路径
            device: 设备，默认使用配置文件中的设备
        """
        self.model_path = model_path or Config.MODEL_CONFIG["qwen"]["model_path"]
        self.device = device or Config.MODEL_CONFIG["qwen"]["device"]
        
        if not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"
            logger.warning("CUDA不可用，使用CPU代替")
        
        self.categories = Config.PAPER_CATEGORIES
        self.num_categories = len(self.categories)
        
        # 加载模型和分词器
        self._load_model()
    
    def _load_model(self):
        """
        加载分类模型和分词器
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 检查是否有微调后的分类模型
            fine_tuned_path = os.path.join(self.model_path, "classifier")
            if os.path.exists(fine_tuned_path):
                model_path = fine_tuned_path
            else:
                model_path = self.model_path
                logger.warning(f"未找到微调后的分类模型，使用基础模型: {model_path}")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_categories
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"成功加载论文分类模型: {model_path}")
        except Exception as e:
            logger.error(f"加载分类模型时出错: {e}")
            raise
    
    def classify(self, paper_framework: Dict[str, str]) -> Dict[str, Any]:
        """
        对论文框架进行分类
        
        Args:
            paper_framework: 论文框架字典
            
        Returns:
            分类结果和概率
        """
        try:
            # 合并框架内容
            text = self._prepare_input_text(paper_framework)
            
            # 分词
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=Config.MODEL_CONFIG["qwen"]["max_length"]
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0]
            
            # 获取最高概率的类别
            top_prob, top_class_idx = torch.max(probabilities, dim=0)
            predicted_category = self.categories[top_class_idx.item()]
            
            # 获取所有类别的概率
            all_probs = {
                category: prob.item()
                for category, prob in zip(self.categories, probabilities)
            }
            
            # 按概率排序
            sorted_categories = sorted(
                all_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return {
                "predicted_category": predicted_category,
                "confidence": top_prob.item(),
                "all_probabilities": all_probs,
                "top_categories": sorted_categories[:3]  # 返回前3个最可能的类别
            }
        except Exception as e:
            logger.error(f"分类论文框架时出错: {e}")
            return {
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "all_probabilities": {},
                "top_categories": []
            }
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        直接对文本进行分类
        
        Args:
            text: 输入文本
            
        Returns:
            分类结果和概率
        """
        try:
            # 分词
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=Config.MODEL_CONFIG["qwen"]["max_length"]
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0]
            
            # 获取最高概率的类别
            top_prob, top_class_idx = torch.max(probabilities, dim=0)
            predicted_category = self.categories[top_class_idx.item()]
            
            # 获取所有类别的概率
            all_probs = {
                category: prob.item()
                for category, prob in zip(self.categories, probabilities)
            }
            
            # 按概率排序
            sorted_categories = sorted(
                all_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return {
                "predicted_category": predicted_category,
                "confidence": top_prob.item(),
                "all_probabilities": all_probs,
                "top_categories": sorted_categories[:3]  # 返回前3个最可能的类别
            }
        except Exception as e:
            logger.error(f"分类文本时出错: {e}")
            return {
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "all_probabilities": {},
                "top_categories": []
            }
    
    def _prepare_input_text(self, paper_framework: Dict[str, str]) -> str:
        """
        准备用于分类的输入文本
        
        Args:
            paper_framework: 论文框架字典
            
        Returns:
            合并后的文本
        """
        # 合并框架中的各个部分
        parts = []
        for section, content in paper_framework.items():
            if content:  # 跳过空内容
                parts.append(f"{section}: {content}")
        
        return "\n\n".join(parts)
    
    def fine_tune(self, train_data: List[Dict[str, Union[str, Dict[str, str]]]], 
                 epochs: int = 3, batch_size: int = 8, 
                 learning_rate: float = 5e-5) -> None:
        """
        微调分类模型
        
        Args:
            train_data: 训练数据列表，每项包含框架和标签
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        try:
            from transformers import Trainer, TrainingArguments
            from torch.utils.data import Dataset
            
            # 准备数据集
            class PaperDataset(Dataset):
                def __init__(self, data, tokenizer, max_length):
                    self.data = data
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                    self.categories = Config.PAPER_CATEGORIES
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    item = self.data[idx]
                    
                    if isinstance(item["framework"], dict):
                        text = "\n\n".join([
                            f"{section}: {content}" 
                            for section, content in item["framework"].items() 
                            if content
                        ])
                    else:
                        text = item["framework"]
                    
                    inputs = self.tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    # 移除批次维度
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    
                    # 标签
                    label = self.categories.index(item["label"])
                    
                    return {
                        **inputs,
                        "labels": torch.tensor(label)
                    }
            
            # 创建训练数据集
            train_dataset = PaperDataset(
                train_data,
                self.tokenizer,
                Config.MODEL_CONFIG["qwen"]["max_length"]
            )
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=os.path.join(self.model_path, "classifier"),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                save_strategy="epoch",
                save_total_limit=2,
                logging_dir=os.path.join(self.model_path, "logs"),
                logging_steps=10,
            )
            
            # 创建训练器
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
            )
            
            # 开始训练
            trainer.train()
            
            # 保存模型
            self.model.save_pretrained(os.path.join(self.model_path, "classifier"))
            self.tokenizer.save_pretrained(os.path.join(self.model_path, "classifier"))
            
            logger.info("分类器微调完成")
        except Exception as e:
            logger.error(f"微调分类器时出错: {e}")
            raise
        
        
        
        
        
    