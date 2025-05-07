
# src/dialogue_system/dialog_manager.py
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import zhipuai
from zhipuai import ZhipuAI
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.configs.config import Config
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

class DialogManager:
    """
    对话管理器，负责处理用户问题和生成回答
    """
    
    def __init__(self):
        """
        初始化对话管理器
        """
        # 加载Qwen模型
        model_path = Config.MODEL_CONFIG["qwen"]["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 如果有GPU，将模型迁移到GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        
        self.client = OpenAI(api_key=Config.MODEL_CONFIG["deepseek"]["api_key"], base_url=Config.MODEL_CONFIG["deepseek"]["base_url"])
        
        # 创建向量存储和知识图谱实例
        self.vector_store = VectorStore()
        self.kg_builder = KnowledgeGraphBuilder()
        
        # 对话历史
        self.history = []
    
    def process_query(self, query: str, paper_id: Optional[str] = None) -> str:
        """
        处理用户查询并生成回答
        
        Args:
            query: 用户查询
            paper_id: 论文ID（可选）
            
        Returns:
            生成的回答
        """
        try:
            # 检查是否是重听请求
            if self._is_rehear_request(query):
                return self._handle_rehear_request()
            
            # 添加到历史
            self.history.append({"role": "user", "content": query})
            
            # 步骤1：召回相关文档片段
            if paper_id:
                # 特定论文中搜索
                relevant_chunks = self.vector_store.search_by_paper_id(query, paper_id, top_k=3)
            else:
                # 全库搜索
                relevant_chunks = self.vector_store.search(query, top_k=3)
            
            # 步骤2：使用Qwen2.5生成初始回答
            raw_answer = self._generate_raw_answer(query, relevant_chunks)
            
            # 步骤3：从答案中提取实体
            entities = self._extract_entities_from_text(raw_answer)
            
            # 步骤4：查询知识图谱获取相关实体信息
            kg_context = self._query_knowledge_graph(entities)
            
            # 步骤5：使用GLM-4生成最终答案
            final_answer = self._generate_final_answer(query, raw_answer, kg_context, relevant_chunks)
            
            # 添加到历史
            self.history.append({"role": "assistant", "content": final_answer})
            
            return final_answer
        except Exception as e:
            logger.error(f"处理查询时出错: {e}")
            error_msg = f"抱歉，处理您的问题时出现了错误: {str(e)}"
            self.history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def _is_rehear_request(self, query: str) -> bool:
        """
        检查是否是重听请求
        
        Args:
            query: 用户查询
            
        Returns:
            是否是重听请求
        """
        rehear_patterns = ["我没听清楚", "没听清", "再说一遍", "重复一遍"]
        for pattern in rehear_patterns:
            if pattern in query:
                return True
        return False
    
    def _handle_rehear_request(self) -> str:
        """
        处理重听请求
        
        Returns:
            上一个问题或默认回复
        """
        # 查找历史中上一个助手回复
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "assistant":
                last_response = self.history[i]["content"]
                self.history.append({"role": "assistant", "content": last_response})
                return last_response
        
        return "抱歉，我没有找到之前的回复。请问您有什么问题吗？"
    
    def _generate_raw_answer(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        使用Qwen2.5生成初始回答
        
        Args:
            query: 用户查询
            relevant_chunks: 相关文档片段
            
        Returns:
            生成的初始回答
        """
        # 准备上下文
        context = ""
        for chunk in relevant_chunks:
            context += f"{chunk['text']}\n\n"
        
        # 构建提示词
        prompt = f"""以下是一些与用户问题相关的文档片段:

{context}

用户问题: {query}

基于上述文档片段回答用户问题。如果文档片段中没有足够的信息，请诚实地表明。
"""
        
        # 使用Qwen2.5生成回答
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=Config.MODEL_CONFIG["qwen"]["temperature"],
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体列表
        """
        # 使用Qwen2.5模型提取实体
        prompt = f"""请从以下文本中提取关键实体（如模型名称、方法、任务、指标等），以逗号分隔列出：

{text}

实体列表："""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 解析实体
        entities = [entity.strip() for entity in response.split(",") if entity.strip()]
        return entities
    
    def _query_knowledge_graph(self, entities: List[str]) -> str:
        """
        查询知识图谱获取实体信息
        
        Args:
            entities: 实体列表
            
        Returns:
            知识图谱中的相关信息
        """
        kg_info = []
        
        for entity in entities:
            # 查询实体
            entity_results = self.kg_builder.query_entity(entity)
            
            if entity_results:
                kg_info.append(f"实体 '{entity}' 的类型: {entity_results[0]['types']}")
                
                # 查询相关实体
                related = self.kg_builder.query_related_entities(entity, max_depth=1)
                if related:
                    relationships = []
                    for item in related[:5]:  # 限制数量
                        rel_type = item["relations"][0] if item["relations"] else "相关"
                        relationships.append(f"{entity} {rel_type} {item['name']}")
                    
                    kg_info.append("关系: " + "; ".join(relationships))
                
                # 查询相关论文
                papers = self.kg_builder.query_papers_by_entity(entity)
                if papers:
                    paper_titles = [p["title"] for p in papers[:3]]
                    kg_info.append(f"相关论文: {', '.join(paper_titles)}")
        
        return "\n".join(kg_info)
    
    def _generate_final_answer(self, query: str, raw_answer: str, 
                             kg_context: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        使用GLM-4生成最终答案
        
        Args:
            query: 用户查询
            raw_answer: 初始回答
            kg_context: 知识图谱上下文
            relevant_chunks: 相关文档片段
            
        Returns:
            生成的最终答案
        """
        # 创建知识库上下文
        kb_context = ""
        for chunk in relevant_chunks:
            kb_context += f"文档片段({chunk['paper_id']}): {chunk['text']}\n\n"
        
        # 创建提示词
        prompt = f"""你是一个学术论文助手，帮助研究生和教授回答关于AI论文的问题。

                用户问题: {query}

                初始回答: {raw_answer}

                知识图谱信息:
                {kg_context}

                相关文档片段:
                {kb_context}

                基于以上所有信息，请提供一个全面、准确且连贯的最终回答。特别注意整合知识图谱提供的关系信息，确保回答学术严谨且信息丰富。
                """
        
        try:
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=Config.MODEL_CONFIG["deepseek"]["model"],
                messages={"role": "user", "content": prompt},
                temperature=0.3,
                top_p=0.7,
                max_tokens=1500
            )
            
            if response.get("code") == 200:
                final_answer = response["data"]["choices"][0]["content"]
                return final_answer
            else:
                logger.error(f"DeepSeek API调用失败: {response}")
                return raw_answer
        except Exception as e:
            logger.error(f"生成最终答案时出错: {e}")
            return raw_answer
    
    def clear_history(self):
        """
        清除对话历史
        """
        self.history = []
