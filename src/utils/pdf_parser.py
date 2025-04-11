
# src/utils/pdf_parser.py
import fitz  # PyMuPDF
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PDFParser:
    """
    PDF解析模块，用于从PDF文件中提取文本内容
    """
    
    def __init__(self):
        self.sections = [
            "abstract", "introduction", "related work", "methodology", 
            "method", "experiment", "results", "discussion", 
            "conclusion", "references"
        ]
    
    def extract_text(self, pdf_path: str) -> str:
        """
        从PDF提取所有文本内容
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text") + "\n\n"
                
            doc.close()
            return text
        except Exception as e:
            logger.error(f"从PDF提取文本时出错: {e}")
            raise
    
    def extract_sections(self, pdf_path: str) -> Dict[str, str]:
        """
        从PDF中提取关键章节内容
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            章节内容的字典
        """
        full_text = self.extract_text(pdf_path)
        sections_content = {}
        
        # 使用正则表达式查找各个章节
        for i, section in enumerate(self.sections):
            pattern = re.compile(f"(?i)\\b{section}\\b.*?\\n", re.DOTALL)
            matches = pattern.finditer(full_text)
            
            for match in matches:
                start_idx = match.start()
                
                # 查找下一个章节的位置
                next_section_idx = float('inf')
                for next_section in self.sections[i+1:]:
                    next_pattern = re.compile(f"(?i)\\b{next_section}\\b.*?\\n", re.DOTALL)
                    next_match = next_pattern.search(full_text[start_idx:])
                    if next_match:
                        curr_next_idx = start_idx + next_match.start()
                        next_section_idx = min(next_section_idx, curr_next_idx)
                
                # 如果没有找到下一个章节，使用文档结尾
                if next_section_idx == float('inf'):
                    next_section_idx = len(full_text)
                
                # 提取章节内容
                section_content = full_text[start_idx:next_section_idx].strip()
                sections_content[section.lower()] = section_content
                break  # 只提取第一个匹配项
        
        return sections_content
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        提取PDF文件的元数据
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            元数据字典
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            # 格式化元数据
            formatted_metadata = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "filename": Path(pdf_path).name,
            }
            
            return formatted_metadata
        except Exception as e:
            logger.error(f"提取PDF元数据时出错: {e}")
            return {
                "title": "",
                "author": "",
                "subject": "",
                "keywords": "",
                "creation_date": "",
                "modification_date": "",
                "filename": Path(pdf_path).name,
            }
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        将长文本分割成重叠的块
        
        Args:
            text: 要分割的文本
            chunk_size: 每个块的最大长度
            overlap: 块之间的重叠字符数
            
        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # 如果不是最后一块，尝试在句子边界处截断
            if end < text_len:
                # 查找最后一个句号、问号或感叹号及其后面的空格
                last_period = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end)
                )
                
                if last_period != -1:
                    end = last_period + 2  # 包含句号和空格
            
            chunks.append(text[start:end])
            start = end - overlap  # 重叠部分
            
        return chunks
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000) -> Dict:
        """
        处理PDF文件，提取文本、元数据和分块内容
        
        Args:
            pdf_path: PDF文件路径
            chunk_size: 分块大小
            
        Returns:
            包含处理结果的字典
        """
        metadata = self.extract_metadata(pdf_path)
        sections = self.extract_sections(pdf_path)
        full_text = self.extract_text(pdf_path)
        
        # 将全文分块
        chunks = self.chunk_text(full_text, chunk_size)
        
        # 将每个章节分块
        sectioned_chunks = {}
        for section, content in sections.items():
            sectioned_chunks[section] = self.chunk_text(content, chunk_size)
        
        return {
            "metadata": metadata,
            "sections": sections,
            "full_text": full_text,
            "chunks": chunks,
            "sectioned_chunks": sectioned_chunks
        }

