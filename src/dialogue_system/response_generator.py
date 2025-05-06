# src/dialogue_system/response_generator.py
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
import torch
import zhipuai
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.configs.config import Config
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder
from src.knowledge_base.faq_manager import FAQManager

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    响应生成器，负责生成对话回复
    """
    
    def __init__(self):
        """
        初始化响应生成器
        """
        # 加载Qwen模型
        model_path = Config.MODEL_CONFIG["qwen"]["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 如果有GPU，将模型迁移到GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 设置GLM API密钥
        zhipuai.api_key = Config.MODEL_CONFIG["glm"]["api_key"]
        
        # 创建FAQ管理器
        self.faq_manager = FAQManager()
        
        # 模板
        self.templates = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict[str, str]:
        """
        加载响应模板
        
        Returns:
            模板字典
        """
        templates = {
            "greeting": "您好，我是学术论文助手，可以帮您解答关于AI论文的问题。您可以询问特定论文内容、研究方法、模型架构等问题。请问有什么可以帮您的？",
            
            "not_understand": "抱歉，我没有理解您的问题。您可以尝试更清晰地描述您的需求，或者换一种方式提问。",
            
            "no_info": "抱歉，我没有找到相关信息。您可以尝试重新表述问题，或者提供更多详细信息。",
            
            "faq": "{answer}\n\n希望这个回答对您有帮助。如果您有更多问题，欢迎继续提问。",
            
            "paper_info": "论文《{title}》由{author}撰写，属于{category}类别。\n\n摘要：{abstract}\n\n这篇论文主要研究了{focus}。",
            
            "entity_info": "关于{entity}：\n\n{description}\n\n相关实体：{related}\n\n在相关论文中的应用：{applications}",
            
            "method_comparison": "{method1}和{method2}的主要区别在于：\n\n{comparison}\n\n在{context}场景下，{recommendation}可能更适合。",
            
            "recommendation": "基于您的兴趣，我推荐以下论文：\n\n{recommendations}\n\n这些论文都与{topic}相关，希望对您的研究有所帮助。",
            
            "clarification": "您想了解的是{options}中的哪一个？或者您可以提供更多细节，帮助我更好地理解您的问题。",
            
            "error": "抱歉，处理您的请求时出现了错误：{error_message}。请稍后再试或联系系统管理员。"
        }
        
        return templates
    
    def generate_faq_response(self, query: str) -> Optional[str]:
        """
        生成FAQ回答
        
        Args:
            query: 用户查询
            
        Returns:
            FAQ回答，如果没有匹配的FAQ则返回None
        """
        similar_questions = self.faq_manager.find_similar_questions(query, threshold=0.75)
        
        if similar_questions:
            best_match = similar_questions[0]
            
            if best_match["similarity"] >= 0.9:
                logger.info(f"找到高匹配度FAQ: {best_match['question']}, 相似度={best_match['similarity']:.4f}")
                return self.templates["faq"].format(answer=best_match["answer"])
            elif best_match["similarity"] >= 0.75:
                logger.info(f"找到可能的FAQ匹配: {best_match['question']}, 相似度={best_match['similarity']:.4f}")
                return self.templates["faq"].format(answer=best_match["answer"])
        
        return None
    
    def generate_raw_response(self, query: str, context: str) -> str:
        """
        使用Qwen2.5生成原始回答
        
        Args:
            query: 用户查询
            context: 上下文信息
            
        Returns:
            生成的回答
        """
        # 构建提示词
        prompt = f"""以下是一些与用户问题相关的上下文信息:

{context}

用户问题: {query}

请根据上述上下文信息回答用户问题。如果上下文中没有足够的信息，请诚实表明。回答要简洁明了、逻辑清晰、学术严谨。
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
    
    def generate_enhanced_response(self, query: str, raw_response: str, 
                                 kg_context: str, 
                                 relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        使用GLM-4生成增强回答
        
        Args:
            query: 用户查询
            raw_response: 原始回答
            kg_context: 知识图谱上下文
            relevant_chunks: 相关文档片段
            
        Returns:
            增强后的回答
        """
        # 创建知识库上下文
        kb_context = ""
        for chunk in relevant_chunks:
            kb_context += f"文档片段({chunk['paper_id']}): {chunk['text']}\n\n"
        
        # 创建提示词
        prompt = f"""你是一个学术论文助手，帮助研究生和教授回答关于AI论文的问题。

用户问题: {query}

初始回答: {raw_response}

知识图谱信息:
{kg_context}

相关文档片段:
{kb_context}

基于以上所有信息，请提供一个全面、准确且连贯的最终回答。特别注意整合知识图谱提供的关系信息，确保回答学术严谨且信息丰富。回答应具有指导性，并且在必要时引用具体论文或研究成果。
"""
        
        try:
            # 调用GLM-4 API
            response = zhipuai.model_api.invoke(
                model=Config.MODEL_CONFIG["glm"]["model"],
                prompt=prompt,
                temperature=0.3,
                top_p=0.7,
                max_tokens=1500
            )
            
            if response.get("code") == 200:
                final_answer = response["data"]["choices"][0]["content"]
                return final_answer
            else:
                logger.error(f"GLM-4 API调用失败: {response}")
                return raw_response
        except Exception as e:
            logger.error(f"生成增强回答时出错: {e}")
            return raw_response
    
    def format_entity_description(self, entity_info: Dict[str, Any]) -> str:
        """
        格式化实体描述
        
        Args:
            entity_info: 实体信息
            
        Returns:
            格式化的描述
        """
        if not entity_info or not isinstance(entity_info, dict):
            return "没有找到相关实体信息。"
        
        entity_name = entity_info.get("name", "未知实体")
        entity_type = entity_info.get("type", "")
        description = entity_info.get("description", "暂无描述。")
        related = entity_info.get("related", [])
        papers = entity_info.get("papers", [])
        
        # 格式化相关实体
        related_str = "暂无相关实体。"
        if related:
            related_items = [f"{r.get('name', '')}（{r.get('relation', '')}）" for r in related[:5]]
            related_str = "、".join(related_items)
        
        # 格式化论文
        papers_str = "暂无相关论文。"
        if papers:
            paper_items = [f"《{p.get('title', '')}》" for p in papers[:3]]
            papers_str = "、".join(paper_items)
        
        # 构建描述
        formatted = f"{entity_name}"
        if entity_type:
            formatted += f"（{entity_type}）"
        
        formatted += f"\n\n{description}\n\n"
        
        if related_str != "暂无相关实体。":
            formatted += f"相关实体：{related_str}\n\n"
        
        if papers_str != "暂无相关论文。":
            formatted += f"相关论文：{papers_str}"
        
        return formatted
    
    def format_recommendations(self, recommendations: List[Dict[str, Any]], topic: str) -> str:
        """
        格式化论文推荐
        
        Args:
            recommendations: 推荐论文列表
            topic: 主题
            
        Returns:
            格式化的推荐
        """
        if not recommendations:
            return f"抱歉，没有找到与{topic}相关的论文推荐。"
        
        formatted = ""
        
        for i, paper in enumerate(recommendations):
            title = paper.get("title", "未知标题")
            author = paper.get("author", "未知作者")
            category = paper.get("category", "")
            
            formatted += f"{i+1}. 《{title}》 - {author}"
            
            if category:
                formatted += f" [{category}]"
            
            formatted += "\n"
        
        return formatted
    
    def generate_error_response(self, error_message: str) -> str:
        """
        生成错误响应
        
        Args:
            error_message: 错误消息
            
        Returns:
            错误响应
        """
        return self.templates["error"].format(error_message=error_message)
    
    def generate_greeting(self) -> str:
        """
        生成问候语
        
        Returns:
            问候语
        """
        return self.templates["greeting"]
    
    def generate_not_understand(self) -> str:
        """
        生成不理解响应
        
        Returns:
            不理解响应
        """
        return self.templates["not_understand"]
    
    def generate_no_info(self) -> str:
        """
        生成无信息响应
        
        Returns:
            无信息响应
        """
        return self.templates["no_info"]
    
    def generate_clarification(self, options: List[str]) -> str:
        """
        生成澄清请求
        
        Args:
            options: 选项列表
            
        Returns:
            澄清请求
        """
        options_str = "、".join([f"{option}" for option in options])
        return self.templates["clarification"].format(options=options_str)