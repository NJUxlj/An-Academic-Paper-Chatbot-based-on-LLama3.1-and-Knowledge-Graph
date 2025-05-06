# tests/test_pdf_parser.py
import os
import sys
import unittest
from pathlib import Path
import tempfile

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.pdf_parser import PDFParser
from src.configs.config import Config

class TestPDFParser(unittest.TestCase):
    """
    测试PDF解析器
    """
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.parser = PDFParser()
        
        # 创建测试PDF文件
        cls.test_pdf_path = cls._create_test_pdf()
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 删除测试PDF文件
        if os.path.exists(cls.test_pdf_path):
            os.remove(cls.test_pdf_path)
    
    @classmethod
    def _create_test_pdf(cls):
        """创建测试PDF文件"""
        # 查找测试数据目录中的PDF文件
        test_data_dir = os.path.join(Config.ROOT_DIR, "tests", "test_data")
        
        # 如果有测试PDF，直接使用
        for file in os.listdir(test_data_dir):
            if file.endswith(".pdf"):
                return os.path.join(test_data_dir, file)
        
        # 如果没有测试PDF，创建一个空的临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_file.close()
        
        return temp_file.name
    
    def test_extract_text(self):
        """测试文本提取"""
        # 跳过测试如果文件大小为0
        if os.path.getsize(self.test_pdf_path) == 0:
            self.skipTest("测试PDF文件为空")
        
        text = self.parser.extract_text(self.test_pdf_path)
        self.assertIsInstance(text, str)
    
    def test_extract_metadata(self):
        """测试元数据提取"""
        metadata = self.parser.extract_metadata(self.test_pdf_path)
        
        self.assertIsInstance(metadata, dict)
        self.assertIn("title", metadata)
        self.assertIn("author", metadata)
        self.assertIn("filename", metadata)
    
    def test_chunk_text(self):
        """测试文本分块"""
        # 测试文本
        test_text = "This is the first sentence. This is the second sentence. " * 20
        
        # 使用默认参数分块
        chunks = self.parser.chunk_text(test_text)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # 检查第一个块的长度
        self.assertLessEqual(len(chunks[0]), 1000 + 200)  # 考虑最后一个句子可能超出chunk_size
        
        # 使用自定义参数分块
        custom_chunks = self.parser.chunk_text(test_text, chunk_size=500, overlap=100)
        
        self.assertIsInstance(custom_chunks, list)
        self.assertGreater(len(custom_chunks), len(chunks))  # 更小的块大小应该产生更多的块
    
    def test_process_pdf(self):
        """测试PDF处理"""
        # 跳过测试如果文件大小为0
        if os.path.getsize(self.test_pdf_path) == 0:
            self.skipTest("测试PDF文件为空")
        
        result = self.parser.process_pdf(self.test_pdf_path)
        
        self.assertIsInstance(result, dict)
        self.assertIn("metadata", result)
        self.assertIn("sections", result)
        self.assertIn("full_text", result)
        self.assertIn("chunks", result)
        self.assertIn("sectioned_chunks", result)

if __name__ == "__main__":
    unittest.main()