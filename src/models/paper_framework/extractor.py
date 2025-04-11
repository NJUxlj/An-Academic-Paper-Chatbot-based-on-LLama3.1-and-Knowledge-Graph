

# src/models/paper_framework/extractor.py
import json
import logging
from typing import Dict, List, Any, Optional
import os
import zhipuai
from src.configs.config import Config

logger = logging.getLogger(__name__)

class PaperFrameworkExtractor:
    """
    使用GLM-4模型抽取论文框架
    """
    
    def __init__(self):
        """
        初始化论文框架抽取器
        """
        self.api_key = Config.MODEL_CONFIG["glm"]["api_key"]
        self.model = Config.MODEL_CONFIG["glm"]["model"]
        
        # 初始化智谱AI客户端
        zhipuai.api_key = self.api_key
    
    def extract_framework(self, paper_text: str) -> Dict[str, str]:
        """
        从论文文本中抽取论文框架
        
        Args:
            paper_text: 论文文本
            
        Returns:
            论文框架字典，包含Abstract、Introduction、Methodology、Experiment、Results等字段
        """
        try:
            # 准备提示词
            prompt = self._create_extraction_prompt(paper_text)
            
            # 调用GLM-4 API
            response = zhipuai.model_api.invoke(
                model=self.model,
                prompt=prompt,
                temperature=0.1,
                top_p=0.7,
                max_tokens=4000
            )
            
            # 解析响应
            if response.get("code") == 200:
                framework_text = response["data"]["choices"][0]["content"]
                
                # 检查是否是JSON格式的响应
                try:
                    framework = json.loads(framework_text)
                except json.JSONDecodeError:
                    # 如果不是JSON，尝试提取JSON部分
                    framework = self._extract_json_from_text(framework_text)
                
                return framework
            else:
                logger.error(f"API调用失败: {response}")
                return self._create_empty_framework()
        except Exception as e:
            logger.error(f"抽取论文框架时出错: {e}")
            return self._create_empty_framework()
    
    def _create_extraction_prompt(self, paper_text: str) -> str:
        """
        创建用于抽取论文框架的提示词
        
        Args:
            paper_text: 论文文本
            
        Returns:
            提示词
        """
        return f"""你是一个专业的论文分析助手，请从下面的论文文本中提取出关键的框架信息，并按照以下结构组织：

```json
{{
  "Abstract": "论文摘要",
  "Introduction": "介绍部分的要点和主要目标",
  "Methodology": "研究方法的详细描述",
  "Experiment": "实验设置和步骤",
  "Results": "主要结果和发现"
}}
```

请确保提取的内容准确反映论文的各个部分，并保持简洁。仅返回JSON格式的结果，不要添加任何其他解释。

论文文本:
{paper_text[:8000]}  # 限制输入长度
"""
    
    def _extract_json_from_text(self, text: str) -> Dict[str, str]:
        """
        从文本中提取JSON部分
        
        Args:
            text: 包含JSON的文本
            
        Returns:
            解析后的JSON字典
        """
        try:
            # 查找JSON开始和结束位置
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._create_empty_framework()
        except Exception as e:
            logger.error(f"从文本中提取JSON时出错: {e}")
            return self._create_empty_framework()
    
    def _create_empty_framework(self) -> Dict[str, str]:
        """
        创建空的论文框架模板
        
        Returns:
            空的论文框架字典
        """
        return {
            "Abstract": "",
            "Introduction": "",
            "Methodology": "",
            "Experiment": "",
            "Results": ""
        }

