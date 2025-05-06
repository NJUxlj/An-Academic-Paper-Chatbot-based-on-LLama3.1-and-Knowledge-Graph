# src/knowledge_base/faq_manager.py
import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from src.configs.config import Config

logger = logging.getLogger(__name__)

class FAQManager:
    """
    FAQ管理器，用于管理FAQ库和进行相似问题检索
    """
    
    def __init__(self, faq_path: Optional[str] = None):
        """
        初始化FAQ管理器
        
        Args:
            faq_path: FAQ数据文件路径（可选）
        """
        self.faq_path = faq_path
        
        # 默认FAQ路径
        if not self.faq_path:
            self.faq_path = os.path.join(Config.PROCESSED_DATA_DIR, "faq_data.json")
        
        # 加载模型
        model_path = Config.MODEL_CONFIG["qwen"]["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        # 如果有GPU，将模型迁移到GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 加载FAQ数据
        self.faqs = []
        self.load_faqs()
        
        # 生成和缓存标准问题的嵌入
        self.std_question_embeddings = self._generate_question_embeddings()
    
    def load_faqs(self) -> None:
        """
        加载FAQ数据
        """
        try:
            if os.path.exists(self.faq_path):
                with open(self.faq_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 确保similar_query是集合
                for item in data:
                    if isinstance(item["similar_query"], list):
                        item["similar_query"] = set(item["similar_query"])
                    
                self.faqs = data
                logger.info(f"成功加载 {len(self.faqs)} 条FAQ数据")
            else:
                logger.warning(f"FAQ文件不存在: {self.faq_path}")
                self.faqs = []
        except Exception as e:
            logger.error(f"加载FAQ数据时出错: {e}")
            self.faqs = []
    
    def _generate_question_embeddings(self) -> np.ndarray:
        """
        为标准问题生成嵌入向量
        
        Returns:
            嵌入向量数组
        """
        if not self.faqs:
            return np.array([])
        
        std_questions = [faq["stand_query"] for faq in self.faqs]
        embeddings = []
        
        with torch.no_grad():
            for question in std_questions:
                inputs = self.tokenizer(
                    question,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    def find_similar_questions(self, query: str, threshold: float = 0.7, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        查找与查询最相似的问题
        
        Args:
            query: 查询文本
            threshold: 相似度阈值
            top_k: 返回的最大结果数
            
        Returns:
            相似问题列表，包含问题、答案和相似度
        """
        if not self.faqs:
            return []
        
        # 获取查询嵌入
        query_embedding = self._get_embedding(query)
        
        # 计算相似度
        similarities = cosine_similarity([query_embedding], self.std_question_embeddings)[0]
        
        # 获取最相似的问题
        similar_questions = []
        
        # 先检查是否与相似问题完全匹配
        for i, faq in enumerate(self.faqs):
            if query in faq["similar_query"] or query == faq["stand_query"]:
                similar_questions.append({
                    "question": faq["stand_query"],
                    "answer": faq["answer"],
                    "similarity": 1.0
                })
                break
        
        # 如果没有完全匹配，使用嵌入相似度
        if not similar_questions:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                similarity = similarities[idx]
                
                if similarity >= threshold:
                    similar_questions.append({
                        "question": self.faqs[idx]["stand_query"],
                        "answer": self.faqs[idx]["answer"],
                        "similarity": float(similarity)
                    })
        
        return similar_questions
    
    def add_faq(self, std_question: str, similar_questions: List[str], answer: str) -> bool:
        """
        添加新的FAQ
        
        Args:
            std_question: 标准问题
            similar_questions: 相似问题列表
            answer: 答案
            
        Returns:
            是否添加成功
        """
        try:
            # 检查是否已存在
            for faq in self.faqs:
                if faq["stand_query"] == std_question:
                    # 更新现有FAQ
                    faq["similar_query"].update(similar_questions)
                    faq["answer"] = answer
                    
                    # 更新缓存的嵌入
                    self.std_question_embeddings = self._generate_question_embeddings()
                    
                    # 保存FAQ数据
                    self._save_faqs()
                    
                    logger.info(f"更新FAQ: {std_question}")
                    return True
            
            # 添加新FAQ
            new_faq = {
                "stand_query": std_question,
                "similar_query": set(similar_questions),
                "answer": answer
            }
            
            self.faqs.append(new_faq)
            
            # 更新缓存的嵌入
            self.std_question_embeddings = self._generate_question_embeddings()
            
            # 保存FAQ数据
            self._save_faqs()
            
            logger.info(f"添加新FAQ: {std_question}")
            return True
        except Exception as e:
            logger.error(f"添加FAQ时出错: {e}")
            return False
    
    def delete_faq(self, std_question: str) -> bool:
        """
        删除FAQ
        
        Args:
            std_question: 标准问题
            
        Returns:
            是否删除成功
        """
        try:
            # 查找要删除的FAQ
            for i, faq in enumerate(self.faqs):
                if faq["stand_query"] == std_question:
                    del self.faqs[i]
                    
                    # 更新缓存的嵌入
                    self.std_question_embeddings = self._generate_question_embeddings()
                    
                    # 保存FAQ数据
                    self._save_faqs()
                    
                    logger.info(f"删除FAQ: {std_question}")
                    return True
            
            logger.warning(f"未找到要删除的FAQ: {std_question}")
            return False
        except Exception as e:
            logger.error(f"删除FAQ时出错: {e}")
            return False
    
    def update_faq(self, std_question: str, new_data: Dict[str, Any]) -> bool:
        """
        更新FAQ
        
        Args:
            std_question: 标准问题
            new_data: 新数据
            
        Returns:
            是否更新成功
        """
        try:
            # 查找要更新的FAQ
            for faq in self.faqs:
                if faq["stand_query"] == std_question:
                    # 更新数据
                    if "stand_query" in new_data:
                        faq["stand_query"] = new_data["stand_query"]
                    
                    if "similar_query" in new_data:
                        if isinstance(new_data["similar_query"], list):
                            faq["similar_query"] = set(new_data["similar_query"])
                        else:
                            faq["similar_query"] = new_data["similar_query"]
                    
                    if "answer" in new_data:
                        faq["answer"] = new_data["answer"]
                    
                    # 更新缓存的嵌入
                    self.std_question_embeddings = self._generate_question_embeddings()
                    
                    # 保存FAQ数据
                    self._save_faqs()
                    
                    logger.info(f"更新FAQ: {std_question}")
                    return True
            
            logger.warning(f"未找到要更新的FAQ: {std_question}")
            return False
        except Exception as e:
            logger.error(f"更新FAQ时出错: {e}")
            return False
    
    def _save_faqs(self) -> None:
        """
        保存FAQ数据
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.faq_path), exist_ok=True)
            
            # 保存FAQ数据
            with open(self.faq_path, "w", encoding="utf-8") as f:
                json.dump(self.faqs, f, ensure_ascii=False, indent=2, default=lambda obj: list(obj) if isinstance(obj, set) else obj)
            
            logger.info(f"成功保存 {len(self.faqs)} 条FAQ数据")
        except Exception as e:
            logger.error(f"保存FAQ数据时出错: {e}")
    
    def get_all_faqs(self) -> List[Dict[str, Any]]:
        """
        获取所有FAQ
        
        Returns:
            FAQ列表
        """
        # 转换set为list
        formatted_faqs = []
        for faq in self.faqs:
            formatted_faq = {
                "stand_query": faq["stand_query"],
                "similar_query": list(faq["similar_query"]),
                "answer": faq["answer"]
            }
            formatted_faqs.append(formatted_faq)
        
        return formatted_faqs
    
    def get_faq_by_question(self, std_question: str) -> Optional[Dict[str, Any]]:
        """
        根据标准问题获取FAQ
        
        Args:
            std_question: 标准问题
            
        Returns:
            FAQ或None
        """
        for faq in self.faqs:
            if faq["stand_query"] == std_question:
                return {
                    "stand_query": faq["stand_query"],
                    "similar_query": list(faq["similar_query"]),
                    "answer": faq["answer"]
                }
        
        return None