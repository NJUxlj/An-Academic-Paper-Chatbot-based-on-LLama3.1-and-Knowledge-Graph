# tests/test_vector_store.py
import os
import sys
import unittest
from pathlib import Path

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base.vector_store import VectorStore

@unittest.skip("需要Milvus服务器")
class TestVectorStore(unittest.TestCase):
    """
    测试向量存储模块
    注意：这个测试需要一个运行中的Milvus服务器
    """
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.vector_store = VectorStore()
        
        # 测试数据
        cls.test_paper_id = "test_paper_001"
        cls.test_chunks = [
            {
                "text": "BERT is a transformer-based language model pre-trained on large amounts of text.",
                "section": "introduction"
            },
            {
                "text": "GPT models use decoder-only transformer architecture for generative tasks.",
                "section": "related_work"
            },
            {
                "text": "Fine-tuning LLMs on domain-specific data improves performance on specialized tasks.",
                "section": "methodology"
            }
        ]
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 删除测试数据
        cls.vector_store.delete_by_paper_id(cls.test_paper_id)
        
        # 关闭连接
        cls.vector_store.close()
    
    def test_add_chunks(self):
        """测试添加文本块"""
        # 添加文本块
        ids = self.vector_store.add_chunks(self.test_paper_id, self.test_chunks)
        
        # 验证结果
        self.assertIsInstance(ids, list)
        self.assertEqual(len(ids), len(self.test_chunks))
    
    def test_search(self):
        """测试搜索"""
        # 确保测试数据存在
        self.vector_store.add_chunks(self.test_paper_id, self.test_chunks)
        
        # 测试查询
        query = "transformer models pre-training"
        results = self.vector_store.search(query, top_k=2)
        
        # 验证结果
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        if results:
            # 验证结果结构
            result = results[0]
            self.assertIn("paper_id", result)
            self.assertIn("chunk_id", result)
            self.assertIn("text", result)
            self.assertIn("section", result)
            self.assertIn("score", result)
    
    def test_search_by_paper_id(self):
        """测试按论文ID搜索"""
        # 确保测试数据存在
        self.vector_store.add_chunks(self.test_paper_id, self.test_chunks)
        
        # 测试查询
        query = "transformer architecture"
        results = self.vector_store.search_by_paper_id(query, self.test_paper_id, top_k=1)
        
        # 验证结果
        self.assertIsInstance(results, list)
        
        if results:
            # 验证结果是否来自正确的论文
            self.assertEqual(results[0]["paper_id"], self.test_paper_id)
    
    def test_delete_by_paper_id(self):
        """测试按论文ID删除"""
        # 确保测试数据存在
        self.vector_store.add_chunks(self.test_paper_id, self.test_chunks)
        
        # 删除数据
        count = self.vector_store.delete_by_paper_id(self.test_paper_id)
        
        # 验证结果
        self.assertEqual(count, len(self.test_chunks))
        
        # 验证数据确实被删除
        results = self.vector_store.search_by_paper_id("transformer", self.test_paper_id)
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()