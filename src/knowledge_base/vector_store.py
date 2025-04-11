
# src/knowledge_base/vector_store.py
import logging
from typing import Dict, List, Any, Optional, Union
import os
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from transformers import AutoTokenizer, AutoModel
import torch
from src.configs.config import Config

logger = logging.getLogger(__name__)

class VectorStore:
    """
    向量存储模块，使用Milvus数据库存储文档片段的向量表示
    """
    
    def __init__(self):
        """
        初始化向量存储
        """
        self.host = Config.VECTOR_DB_CONFIG["host"]
        self.port = Config.VECTOR_DB_CONFIG["port"]
        self.collection_name = Config.VECTOR_DB_CONFIG["collection"]
        self.dim = Config.VECTOR_DB_CONFIG["dim"]
        
        # 加载Qwen模型用于向量嵌入
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CONFIG["qwen"]["model_path"])
        self.model = AutoModel.from_pretrained(Config.MODEL_CONFIG["qwen"]["model_path"])
        
        # 如果有GPU，将模型迁移到GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 连接Milvus
        self._connect()
        
        # 确保集合存在
        self._ensure_collection()
    
    def _connect(self):
        """
        连接到Milvus服务器
        """
        try:
            connections.connect(
                alias="default", 
                host=self.host, 
                port=self.port
            )
            logger.info(f"成功连接到Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def _ensure_collection(self):
        """
        确保集合存在，如果不存在则创建
        """
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                logger.info(f"集合已存在: {self.collection_name}")
                return
            
            # 定义集合字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields, "向量化论文片段")
            
            # 创建集合
            collection = Collection(self.collection_name, schema)
            
            # 创建索引
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index("embedding", index_params)
            
            logger.info(f"成功创建集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"确保集合存在时出错: {e}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            向量嵌入
        """
        try:
            # 分词
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]令牌表示整个序列
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"获取文本嵌入时出错: {e}")
            raise
    
    def add_chunks(self, paper_id: str, chunks: List[Dict[str, str]]) -> List[int]:
        """
        将论文文本块添加到向量存储
        
        Args:
            paper_id: 论文ID
            chunks: 文本块列表，每个块包含text和section
            
        Returns:
            插入的ID列表
        """
        try:
            # 获取集合
            collection = Collection(self.collection_name)
            
            # 准备数据
            paper_ids = []
            chunk_ids = []
            texts = []
            sections = []
            embeddings = []
            
            for i, chunk in enumerate(chunks):
                text = chunk["text"]
                section = chunk.get("section", "unknown")
                chunk_id = f"{paper_id}_{i}"
                
                # 获取嵌入
                embedding = self._get_embedding(text)
                
                paper_ids.append(paper_id)
                chunk_ids.append(chunk_id)
                texts.append(text)
                sections.append(section)
                embeddings.append(embedding)
            
            # 插入数据
            data = [
                paper_ids,
                chunk_ids,
                texts,
                sections,
                embeddings
            ]
            
            ids = collection.insert(data)
            collection.flush()
            
            logger.info(f"成功添加 {len(chunks)} 个文本块到向量存储")
            return ids
        except Exception as e:
            logger.error(f"添加文本块到向量存储时出错: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        搜索与查询最相似的文本块
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            filter_expr: 过滤表达式
            
        Returns:
            相似文本块列表
        """
        try:
            # 获取查询嵌入
            query_embedding = self._get_embedding(query)
            
            # 获取集合
            collection = Collection(self.collection_name)
            collection.load()
            
            # 准备搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }
            
            # 执行搜索
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["paper_id", "chunk_id", "text", "section"]
            )
            
            # 解析结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "paper_id": hit.entity.get("paper_id"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "text": hit.entity.get("text"),
                        "section": hit.entity.get("section"),
                        "score": hit.distance
                    }
                    search_results.append(result)
            
            collection.release()
            return search_results
        except Exception as e:
            logger.error(f"搜索向量存储时出错: {e}")
            return []
    
    def search_by_paper_id(self, query: str, paper_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在特定论文中搜索与查询最相似的文本块
        
        Args:
            query: 查询文本
            paper_id: 论文ID
            top_k: 返回的最大结果数
            
        Returns:
            相似文本块列表
        """
        filter_expr = f'paper_id == "{paper_id}"'
        return self.search(query, top_k, filter_expr)
    
    def delete_by_paper_id(self, paper_id: str) -> int:
        """
        删除特定论文的所有文本块
        
        Args:
            paper_id: 论文ID
            
        Returns:
            删除的记录数
        """
        try:
            collection = Collection(self.collection_name)
            expr = f'paper_id == "{paper_id}"'
            
            # 计算删除数量
            count = collection.query(expr=expr, output_fields=["count(*)"])[0]["count(*)"]
            
            # 执行删除
            collection.delete(expr)
            collection.flush()
            
            logger.info(f"成功删除论文 {paper_id} 的 {count} 个文本块")
            return count
        except Exception as e:
            logger.error(f"删除论文文本块时出错: {e}")
            return 0
    
    def close(self):
        """
        关闭连接
        """
        try:
            connections.disconnect("default")
            logger.info("关闭Milvus连接")
        except Exception as e:
            logger.error(f"关闭Milvus连接时出错: {e}")
