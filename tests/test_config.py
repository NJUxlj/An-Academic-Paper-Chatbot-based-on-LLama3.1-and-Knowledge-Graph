# tests/test_config.py
import os
import sys
import unittest
from pathlib import Path

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configs.config import Config

class TestConfig(unittest.TestCase):
    """
    测试配置模块
    """
    
    def test_config_structure(self):
        """测试配置结构"""
        # 检查必要的配置项
        self.assertIn("MODEL_CONFIG", dir(Config))
        self.assertIn("NEO4J_CONFIG", dir(Config))
        self.assertIn("VECTOR_DB_CONFIG", dir(Config))
        self.assertIn("API_CONFIG", dir(Config))
        
        # 检查模型配置
        self.assertIn("qwen", Config.MODEL_CONFIG)
        self.assertIn("glm", Config.MODEL_CONFIG)
        self.assertIn("bert_ner", Config.MODEL_CONFIG)
        
        # 检查Neo4j配置
        self.assertIn("uri", Config.NEO4J_CONFIG)
        self.assertIn("auth", Config.NEO4J_CONFIG)
        self.assertIn("database", Config.NEO4J_CONFIG)
        
        # 检查向量数据库配置
        self.assertIn("host", Config.VECTOR_DB_CONFIG)
        self.assertIn("port", Config.VECTOR_DB_CONFIG)
        self.assertIn("collection", Config.VECTOR_DB_CONFIG)
        
        # 检查API配置
        self.assertIn("host", Config.API_CONFIG)
        self.assertIn("port", Config.API_CONFIG)
    
    def test_directories(self):
        """测试目录配置"""
        # 检查目录路径
        self.assertTrue(hasattr(Config, "ROOT_DIR"))
        self.assertTrue(hasattr(Config, "DATA_DIR"))
        self.assertTrue(hasattr(Config, "PROCESSED_DATA_DIR"))
        self.assertTrue(hasattr(Config, "RAW_DATA_DIR"))
        self.assertTrue(hasattr(Config, "CACHE_DIR"))
        
        # 检查目录存在
        self.assertTrue(os.path.exists(Config.DATA_DIR))
        self.assertTrue(os.path.exists(Config.PROCESSED_DATA_DIR))
        self.assertTrue(os.path.exists(Config.RAW_DATA_DIR))
        self.assertTrue(os.path.exists(Config.CACHE_DIR))
    
    def test_paper_categories(self):
        """测试论文类别配置"""
        self.assertTrue(hasattr(Config, "PAPER_CATEGORIES"))
        self.assertIsInstance(Config.PAPER_CATEGORIES, list)
        self.assertGreater(len(Config.PAPER_CATEGORIES), 0)

if __name__ == "__main__":
    unittest.main()
