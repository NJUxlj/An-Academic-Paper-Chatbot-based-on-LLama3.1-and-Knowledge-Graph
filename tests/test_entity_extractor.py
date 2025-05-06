# tests/test_entity_extractor.py
import os
import sys
import unittest
from pathlib import Path
import torch

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.entity_extractor.bert_bilstm_crf import EntityTripleExtractor
from src.configs.config import Config

class TestEntityExtractor(unittest.TestCase):
    """
    测试实体抽取器
    """
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 检查GPU可用性
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        # 初始化实体抽取器
        cls.extractor = EntityTripleExtractor(
            model_path=Config.MODEL_CONFIG["bert_ner"]["model_path"],
            device=device
        )
    
    def test_predict_entities(self):
        """测试实体预测"""
        # 测试文本
        test_text = "BERT is a popular model that outperforms GPT on GLUE benchmark tasks like SQuAD and achieves state-of-the-art results."
        
        # 预测实体
        entities = self.extractor.predict_entities(test_text)
        
        # 验证结果
        self.assertIsInstance(entities, list)
        
        # 即使没有找到实体，也应该返回一个空列表而不是None
        self.assertIsNotNone(entities)
        
        if entities:
            # 验证实体结构
            for entity in entities:
                self.assertIn("type", entity)
                self.assertIn("start", entity)
                self.assertIn("end", entity)
                self.assertIn("text", entity)
    
    def test_extract_triples(self):
        """测试三元组提取"""
        # 测试文本
        test_text = "BERT model uses Transformer architecture and achieves excellent results on GLUE benchmark."
        
        # 提取三元组
        triples = self.extractor.extract_triples(test_text)
        
        # 验证结果
        self.assertIsInstance(triples, list)
        
        # 即使没有找到三元组，也应该返回一个空列表而不是None
        self.assertIsNotNone(triples)
        
        if triples:
            # 验证三元组结构
            for triple in triples:
                self.assertIn("head", triple)
                self.assertIn("head_type", triple)
                self.assertIn("relation", triple)
                self.assertIn("tail", triple)
                self.assertIn("tail_type", triple)

if __name__ == "__main__":
    unittest.main()