# src/data/data_preprocess.py
import os
import json
import logging
import pandas as pd
import re
import random
from typing import Dict, List, Any, Optional, Set, Tuple
import torch
from tqdm import tqdm
from pathlib import Path
from src.configs.config import Config
from src.utils.pdf_parser import PDFParser

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    数据预处理模块，用于处理原始论文数据，生成训练所需的数据集
    """
    
    def __init__(self):
        """
        初始化数据预处理器
        """
        self.pdf_parser = PDFParser()
        self.raw_data_dir = Config.RAW_DATA_DIR
        self.processed_data_dir = Config.PROCESSED_DATA_DIR
        
        # 确保目录存在
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def process_raw_papers(self, target_count: int = 20) -> List[Dict[str, Any]]:
        """
        批量处理原始论文PDF文件
        
        Args:
            target_count: 目标处理数量
            
        Returns:
            处理后的论文数据列表
        """
        logger.info(f"开始处理原始论文，目标数量: {target_count}")
        
        # 获取所有PDF文件
        pdf_files = list(Path(self.raw_data_dir).glob("*.pdf"))
        random.shuffle(pdf_files)  # 随机打乱顺序
        
        if len(pdf_files) < target_count:
            logger.warning(f"可用PDF文件数量 ({len(pdf_files)}) 少于目标数量 ({target_count})")
        
        # 限制处理数量
        pdf_files = pdf_files[:target_count]
        
        processed_papers = []
        
        for pdf_file in tqdm(pdf_files, desc="处理论文"):
            try:
                logger.info(f"处理文件: {pdf_file.name}")
                
                # 解析PDF
                paper_data = self.pdf_parser.process_pdf(str(pdf_file))
                
                # 提取元数据
                metadata = paper_data["metadata"]
                
                # 保存处理结果
                output_file = Path(self.processed_data_dir) / f"{pdf_file.stem}.json"
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(paper_data, f, ensure_ascii=False, indent=2)
                
                processed_papers.append({
                    "file_name": pdf_file.name,
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "processed_path": str(output_file)
                })
                
                logger.info(f"成功处理论文: {metadata.get('title', 'Unknown')}")
            except Exception as e:
                logger.error(f"处理论文 {pdf_file.name} 时出错: {e}")
        
        # 保存处理日志
        log_file = Path(self.processed_data_dir) / "processing_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(processed_papers, f, ensure_ascii=False, indent=2)
        
        logger.info(f"成功处理 {len(processed_papers)} 篇论文")
        return processed_papers
    
    def prepare_ner_training_data(self, manual_annotations_path: Optional[str] = None) -> str:
        """
        准备命名实体识别的训练数据
        
        Args:
            manual_annotations_path: 手动标注的NER数据路径（可选）
            
        Returns:
            训练数据的输出路径
        """
        output_file = Path(self.processed_data_dir) / "ner_training_data.json"
        
        # 如果提供了手动标注的数据
        if manual_annotations_path and os.path.exists(manual_annotations_path):
            logger.info(f"使用手动标注的NER数据: {manual_annotations_path}")
            
            # 导入手动标注的数据
            with open(manual_annotations_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            
            # 保存为训练数据格式
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"NER训练数据准备完成，共 {len(annotations)} 条数据")
            return str(output_file)
        
        # 否则，尝试生成合成数据（注意：实际应用中最好使用真实标注数据）
        logger.info("生成合成NER训练数据")
        
        # 加载处理过的论文数据
        processed_papers = []
        for json_file in Path(self.processed_data_dir).glob("*.json"):
            if json_file.name != "processing_log.json" and json_file.name != "ner_training_data.json":
                with open(json_file, "r", encoding="utf-8") as f:
                    paper_data = json.load(f)
                    processed_papers.append(paper_data)
        
        # 定义一些模型、方法、任务和指标的示例
        models = ["BERT", "GPT", "T5", "LSTM", "Transformer", "RoBERTa", "XLNet", "ALBERT", "ELECTRA", "ViT"]
        methods = ["Fine-tuning", "Pre-training", "Transfer Learning", "Multi-task Learning", "Prompt-tuning"]
        tasks = ["Classification", "Generation", "Translation", "Summarization", "QA", "NER", "Sentiment Analysis"]
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "BLEU", "ROUGE", "METEOR", "Perplexity"]
        
        # 生成合成的NER数据
        synthetic_data = []
        
        for paper in processed_papers:
            text = paper.get("full_text", "")
            
            # 分割成句子
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # 选择部分句子生成标注
            for sentence in random.sample(sentences, min(20, len(sentences))):
                sentence = sentence.strip()
                if len(sentence) < 20 or len(sentence) > 200:
                    continue
                
                tokens = []
                labels = []
                
                # 简单的分词和标注
                for word in sentence.split():
                    tokens.append(word)
                    
                    # 检查是否是特定实体类型
                    if word in models:
                        labels.append("B-MODEL")
                    elif word in methods:
                        labels.append("B-METHOD")
                    elif word in tasks:
                        labels.append("B-TASK")
                    elif word in metrics:
                        labels.append("B-METRIC")
                    else:
                        labels.append("O")
                
                if any(label != "O" for label in labels):
                    synthetic_data.append({
                        "tokens": tokens,
                        "labels": labels,
                        "sentence": sentence
                    })
        
        # 保存合成数据
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"合成NER训练数据准备完成，共 {len(synthetic_data)} 条数据")
        return str(output_file)
    
    def prepare_classification_data(self, manual_labels_path: Optional[str] = None) -> str:
        """
        准备论文分类的训练数据
        
        Args:
            manual_labels_path: 手动标注的分类数据路径（可选）
            
        Returns:
            训练数据的输出路径
        """
        output_file = Path(self.processed_data_dir) / "classification_training_data.json"
        
        # 如果提供了手动标注的数据
        if manual_labels_path and os.path.exists(manual_labels_path):
            logger.info(f"使用手动标注的分类数据: {manual_labels_path}")
            
            # 导入手动标注的数据
            with open(manual_labels_path, "r", encoding="utf-8") as f:
                labeled_data = json.load(f)
            
            # 保存为训练数据格式
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(labeled_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分类训练数据准备完成，共 {len(labeled_data)} 条数据")
            return str(output_file)
        
        # 否则，生成合成数据
        logger.info("生成合成分类训练数据")
        
        # 获取论文框架数据
        framework_data = []
        for json_file in Path(self.processed_data_dir).glob("*.json"):
            if json_file.name != "processing_log.json" and json_file.name != "classification_training_data.json":
                with open(json_file, "r", encoding="utf-8") as f:
                    paper_data = json.load(f)
                    
                    # 提取论文框架（如果存在）或者从章节构建框架
                    if "framework" in paper_data:
                        framework = paper_data["framework"]
                    else:
                        sections = paper_data.get("sections", {})
                        framework = {
                            "Abstract": sections.get("abstract", ""),
                            "Introduction": sections.get("introduction", ""),
                            "Methodology": sections.get("methodology", "") or sections.get("method", ""),
                            "Experiment": sections.get("experiment", "") or sections.get("experiments", ""),
                            "Results": sections.get("results", "")
                        }
                    
                    # 随机分配一个类别（在实际应用中应使用真实标注）
                    category = random.choice(Config.PAPER_CATEGORIES)
                    
                    framework_data.append({
                        "framework": framework,
                        "label": category,
                        "file_name": json_file.name
                    })
        
        # 保存合成数据
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(framework_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"合成分类训练数据准备完成，共 {len(framework_data)} 条数据")
        return str(output_file)
    
    def prepare_faq_data(self) -> str:
        """
        准备FAQ数据，用于问答系统
        
        Returns:
            FAQ数据的输出路径
        """
        output_file = Path(self.processed_data_dir) / "faq_data.json"
        
        # 定义一些常见的AI/ML相关问题
        standard_questions = [
            "什么是注意力机制？",
            "BERT模型的主要创新是什么？",
            "LoRA微调技术的原理是什么？",
            "Chain-of-Thought推理的工作原理是怎样的？",
            "大语言模型如何处理长上下文？",
            "什么是RLHF？",
            "RAG技术是如何工作的？",
            "Prompt-Tuning和Fine-Tuning有什么区别？",
            "常见的大模型评测基准有哪些？",
            "Transformer模型结构的关键组件有哪些？"
        ]
        
        # 为每个标准问题生成一些相似问法
        faq_data = []
        
        for std_q in standard_questions:
            # 生成几个相似问法
            similar_questions = self._generate_similar_questions(std_q)
            
            # 生成一个示例答案
            answer = self._generate_sample_answer(std_q)
            
            faq_entry = {
                "stand_query": std_q,
                "similar_query": similar_questions,
                "answer": answer
            }
            
            faq_data.append(faq_entry)
        
        # 保存FAQ数据
        with open(output_file, "w", encoding="utf-8") as f:
            # 需要自定义序列化处理set类型
            json.dump(faq_data, f, ensure_ascii=False, indent=2, default=lambda obj: list(obj) if isinstance(obj, set) else obj)
        
        logger.info(f"FAQ数据准备完成，共 {len(faq_data)} 条数据")
        return str(output_file)
    
    def _generate_similar_questions(self, standard_question: str) -> Set[str]:
        """
        为标准问题生成相似问法
        
        Args:
            standard_question: 标准问题
            
        Returns:
            相似问法集合
        """
        similar_questions = set()
        
        # 添加原始问题
        similar_questions.add(standard_question)
        
        # 生成变体
        prefixes = ["请问", "能否解释", "我想了解", "说说", "讲解一下"]
        suffixes = ["是什么？", "？", "的概念", "的原理", ""]
        
        # 从问题中提取核心内容
        core = standard_question.rstrip("？?。.!")
        if "什么是" in core:
            core = core.replace("什么是", "")
        elif "是什么" in core:
            core = core.replace("是什么", "")
        
        # 生成变体
        for prefix in prefixes:
            for suffix in suffixes:
                variant = f"{prefix}{core}{suffix}"
                if variant != standard_question:
                    similar_questions.add(variant)
        
        # 只保留部分变体，避免过多
        if len(similar_questions) > 5:
            return set(list(similar_questions)[:5])
        
        return similar_questions
    
    def _generate_sample_answer(self, question: str) -> str:
        """
        为问题生成示例答案
        
        Args:
            question: 问题
            
        Returns:
            示例答案
        """
        # 针对不同问题生成示例答案
        if "注意力机制" in question:
            return """注意力机制（Attention Mechanism）是深度学习中的一种技术，它允许模型根据输入的相关性选择性地关注输入数据的特定部分。
            
在自然语言处理中，注意力机制使模型能够在处理序列时"关注"序列中的特定位置，而不是同等对待所有位置。这在翻译、摘要等任务中尤为重要，让模型可以对齐和关联输入序列的不同部分。

Transformer模型的核心创新之一就是引入了自注意力（Self-Attention）机制，计算序列中每个位置与所有其他位置的关系，从而捕获长距离依赖关系。注意力机制通常通过计算查询（Query）、键（Key）和值（Value）之间的相似度来实现，这种设计使模型能够有选择地加权信息。"""
        
        elif "BERT" in question:
            return """BERT (Bidirectional Encoder Representations from Transformers) 模型的主要创新点包括：

1. 双向语言模型：BERT突破了传统语言模型的单向限制，引入了双向上下文表示，通过masked language modeling (MLM)任务，模型可以同时考虑左右两侧的上下文信息。

2. 预训练+微调范式：BERT采用了两阶段训练方法，先通过大规模无标注数据进行预训练，再针对下游任务进行微调，极大提高了模型的泛化能力。

3. 统一架构：BERT提供了一个统一的架构，可以处理多种NLP任务，无需针对每个任务设计特定的网络结构。

4. 深层Transformer结构：BERT使用了多层Transformer编码器，能够捕获文本的深层语义信息。

5. 创新的预训练任务：除了MLM，BERT还引入了Next Sentence Prediction (NSP)任务，帮助模型学习句子间的关系。

BERT的这些创新为后续的GPT、RoBERTa、ALBERT等模型奠定了基础，彻底改变了NLP领域的技术路线。"""
        
        elif "LoRA" in question:
            return """LoRA (Low-Rank Adaptation) 是一种高效的微调技术，它的核心原理是：

1. 参数高效微调：LoRA冻结预训练的模型权重，仅引入少量可训练参数，大大减少了微调的计算和存储需求。

2. 低秩矩阵分解：LoRA在原模型的权重矩阵W旁边添加一个低秩分解矩阵ΔW=AB，其中A∈R^(d×r)，B∈R^(r×k)，r是LoRA的秩，通常远小于d和k。

3. 数学表达：原本的映射y=Wx被替换为y=(W+ΔW)x=Wx+ABx，只有A和B是可训练的。

4. 优势：
   - 显著减少了可训练参数数量，通常减少95%以上
   - 适用于大型模型，如GPT-3、LLaMA等
   - 多个LoRA适配器可以轻松合并或切换
   - 微调后的模型推理速度与原模型相当

LoRA是PEFT（Parameter-Efficient Fine-Tuning）家族中的重要方法，已成为大模型微调的主流技术之一。"""
        
        elif "Chain-of-Thought" in question:
            return """Chain-of-Thought (CoT) 推理是一种改进大语言模型推理能力的技术，其工作原理如下：

1. 分步思考：CoT鼓励模型产生中间推理步骤，而不是直接跳到最终答案，类似人类解决复杂问题时的思考过程。

2. 提示技术：通过在提示中添加"让我们一步一步思考"这样的引导语，或提供包含推理过程的示例（少样本示例），激发模型进行逐步推理。

3. 改进推理能力：研究表明，CoT显著提高了模型在算术、常识和符号推理任务上的性能。

4. 自洽性：CoT通过系统地探索解题过程，增加了推理的自洽性，减少了直接推断的错误。

5. 可解释性：CoT提供了模型如何到达结论的详细解释，使推理过程更加透明和可理解。

6. 相关变体：存在多种CoT变体，如自我一致性CoT和零样本CoT，进一步提升了推理表现。

CoT推理已成为大型语言模型的核心能力，特别是在需要复杂推理的任务中表现突出。"""
        
        else:
            # 对于其他问题，生成一个通用的回答
            return f"""关于"{question.strip('？?')}",这是人工智能领域的一个重要概念。

它主要涉及深度学习和自然语言处理技术的应用，在现代AI系统中发挥着关键作用。这一技术通过创新的算法实现了更高效的计算和更准确的结果，为研究人员和开发者提供了新的工具和方法。

在实际应用中，这一技术已经展示出了显著的性能提升，并影响了多个相关领域的发展方向。未来研究可能会进一步改进其效率和适用性，拓展其应用场景。"""
        
    def convert_to_spacy_format(self, ner_data_path: str) -> str:
        """
        将NER数据转换为spaCy训练格式
        
        Args:
            ner_data_path: NER数据路径
            
        Returns:
            spaCy格式数据的输出路径
        """
        output_file = Path(self.processed_data_dir) / "ner_spacy_format.json"
        
        # 加载NER数据
        with open(ner_data_path, "r", encoding="utf-8") as f:
            ner_data = json.load(f)
        
        # 转换为spaCy格式
        spacy_data = []
        
        for item in ner_data:
            tokens = item["tokens"]
            labels = item["labels"]
            
            # 创建实体标注
            entities = []
            start_idx = 0
            i = 0
            
            while i < len(labels):
                if labels[i].startswith("B-"):
                    entity_type = labels[i][2:]  # 去掉"B-"前缀
                    start_char_idx = start_idx
                    
                    # 计算实体文本和长度
                    entity_text = tokens[i]
                    end_char_idx = start_char_idx + len(entity_text)
                    i += 1
                    
                    # 检查是否有连续的I-标签
                    while i < len(labels) and labels[i] == f"I-{entity_type}":
                        entity_text += " " + tokens[i]
                        i += 1
                    
                    # 添加实体
                    entities.append((start_char_idx, end_char_idx, entity_type))
                else:
                    i += 1
                
                # 更新起始索引
                start_idx += len(tokens[i-1]) + 1  # +1 for space
            
            # 创建spaCy格式数据
            text = " ".join(tokens)
            spacy_item = {
                "text": text,
                "entities": entities
            }
            
            spacy_data.append(spacy_item)
        
        # 保存spaCy格式数据
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(spacy_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"spaCy格式NER数据准备完成，共 {len(spacy_data)} 条数据")
        return str(output_file)
    
    def prepare_all_data(self) -> Dict[str, str]:
        """
        准备所有类型的数据
        
        Returns:
            各类数据路径的字典
        """
        logger.info("开始准备所有数据")
        
        # 处理原始论文
        self.process_raw_papers()
        
        # 准备NER数据
        ner_data_path = self.prepare_ner_training_data()
        
        # 准备分类数据
        classification_data_path = self.prepare_classification_data()
        
        # 准备FAQ数据
        faq_data_path = self.prepare_faq_data()
        
        # 准备spaCy格式的NER数据
        spacy_ner_data_path = self.convert_to_spacy_format(ner_data_path)
        
        return {
            "ner_data": ner_data_path,
            "classification_data": classification_data_path,
            "faq_data": faq_data_path,
            "spacy_ner_data": spacy_ner_data_path
        }