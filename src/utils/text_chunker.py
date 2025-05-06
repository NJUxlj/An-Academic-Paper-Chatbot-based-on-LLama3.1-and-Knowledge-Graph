# src/utils/text_chunker.py
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TextChunker:
    """
    文本分块器，用于将长文本分割成适合处理的块
    """
    
    def __init__(self, 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200,
                respect_sections: bool = True):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠（字符数）
            respect_sections: 是否尊重章节边界
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sections = respect_sections
    
    def chunk_text(self, text: str) -> List[str]:
        """
        将文本分成重叠的块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # 如果不是最后一块，尝试在句子边界处截断
            if end < text_len:
                # 查找最后一个句号、问号或感叹号及其后面的空格
                last_period = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind(".\n", start, end),
                    text.rfind("?\n", start, end),
                    text.rfind("!\n", start, end)
                )
                
                if last_period != -1:
                    end = last_period + 2  # 包含句号和空格/换行符
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap  # 重叠部分
            
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将文档分成块，保留元数据
        
        Args:
            document: 包含文本和元数据的文档
            
        Returns:
            文档块列表
        """
        text = document.get("text", "")
        if not text:
            return []
        
        metadata = {k: v for k, v in document.items() if k != "text"}
        
        text_chunks = self.chunk_text(text)
        document_chunks = []
        
        for i, chunk in enumerate(text_chunks):
            document_chunks.append({
                "text": chunk,
                "chunk_id": f"{i}",
                **metadata
            })
        
        return document_chunks
    
    def chunk_by_section(self, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        按章节分块
        
        Args:
            sections: 章节字典，键为章节名，值为章节内容
            
        Returns:
            章节块列表
        """
        section_chunks = []
        
        for section_name, section_content in sections.items():
            if not section_content:
                continue
            
            chunks = self.chunk_text(section_content)
            
            for i, chunk in enumerate(chunks):
                section_chunks.append({
                    "text": chunk,
                    "section": section_name,
                    "chunk_id": f"{section_name}_{i}"
                })
        
        return section_chunks
    
    def chunk_with_sliding_window(self, text: str, window_size: int = 5) -> List[str]:
        """
        使用滑动窗口进行分块
        
        Args:
            text: 输入文本
            window_size: 窗口大小（句子数）
            
        Returns:
            文本块列表
        """
        # 分割成句子
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= window_size:
            return [" ".join(sentences)]
        
        chunks = []
        
        for i in range(0, len(sentences) - window_size + 1, max(1, window_size - self.chunk_overlap // 50)):
            window = sentences[i:i + window_size]
            chunks.append(" ".join(window))
        
        return chunks
    
    def adaptive_chunking(self, text: str, min_chunk_size: int = 500) -> List[str]:
        """
        自适应分块，根据文本结构动态调整块大小
        
        Args:
            text: 输入文本
            min_chunk_size: 最小块大小
            
        Returns:
            文本块列表
        """
        # 检测文本中的结构，如标题、段落等
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 判断是否为标题（简单启发式）
            is_title = len(para) < 100 and len(para.split()) < 15 and not para.endswith('.')
            
            # 如果当前块加上这个段落超过大小限制，且不是标题，则完成当前块
            if len(current_chunk) + len(para) > self.chunk_size and not is_title and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # 添加分隔符
            if current_chunk:
                current_chunk += "\n\n"
            
            current_chunk += para
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def semantic_chunking(self, text: str, split_markers: List[str] = None) -> List[str]:
        """
        语义分块，根据语义边界分割文本
        
        Args:
            text: 输入文本
            split_markers: 分割标记列表
            
        Returns:
            文本块列表
        """
        if split_markers is None:
            split_markers = [
                "In conclusion", 
                "To summarize", 
                "Furthermore", 
                "In contrast", 
                "However", 
                "Moreover", 
                "In addition",
                "First", 
                "Second", 
                "Third", 
                "Finally", 
                "Lastly"
            ]
        
        # 创建分割模式
        pattern = r'(\b' + r'\b|\b'.join(split_markers) + r'\b)'
        
        # 根据模式分割文本
        segments = re.split(pattern, text)
        
        # 重新组合分割标记和后续文本
        markers_and_text = []
        for i in range(0, len(segments), 2):
            if i + 1 < len(segments):
                markers_and_text.append(segments[i] + segments[i+1])
            else:
                markers_and_text.append(segments[i])
        
        # 基于大小合并块
        chunks = []
        current_chunk = ""
        
        for segment in markers_and_text:
            if len(current_chunk) + len(segment) > self.chunk_size and len(current_chunk) > 0:
                chunks.append(current_chunk.strip())
                current_chunk = segment
            else:
                current_chunk += segment
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks