# src/evaluation/evaluate.py
import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.configs.config import Config
from src.dialogue_system.dialog_manager import DialogManager
from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder
from src.knowledge_base.vector_store import VectorStore
from src.models.classifier.qwen_classifier import PaperFrameworkClassifier
from src.models.entity_extractor.bert_bilstm_crf import EntityTripleExtractor

logger = logging.getLogger(__name__)

class SystemEvaluator:
    """
    系统评估模块，用于评估不同组件的性能
    """
    
    def __init__(self):
        """
        初始化评估模块
        """
        self.dialog_manager = DialogManager()
        self.kg_builder = KnowledgeGraphBuilder()
        self.vector_store = VectorStore()
        self.classifier = PaperFrameworkClassifier()
        self.entity_extractor = EntityTripleExtractor(
            model_path=Config.MODEL_CONFIG["bert_ner"]["model_path"],
            device=Config.MODEL_CONFIG["qwen"]["device"]
        )
        
        # 评估结果
        self.results = {}
        
        # 评估数据目录
        self.eval_data_dir = os.path.join(Config.PROCESSED_DATA_DIR, "eval")
        os.makedirs(self.eval_data_dir, exist_ok=True)
    
    def evaluate_classification(self, test_data_path: str) -> Dict[str, Any]:
        """
        评估论文分类性能
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估论文分类: {test_data_path}")
        
        # 加载测试数据
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # 预测结果
        true_labels = []
        pred_labels = []
        confidences = []
        
        for item in tqdm(test_data, desc="论文分类评估"):
            # 获取真实标签
            true_label = item["label"]
            true_labels.append(true_label)
            
            # 预测
            if isinstance(item["framework"], dict):
                prediction = self.classifier.classify(item["framework"])
            else:
                prediction = self.classifier.classify_text(item["framework"])
            
            # 保存预测结果
            pred_label = prediction["predicted_category"]
            pred_labels.append(pred_label)
            confidences.append(prediction["confidence"])
        
        # 计算性能指标
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, 
            pred_labels, 
            average="weighted"
        )
        
        # 类别级别的指标
        class_metrics = {}
        for category in Config.PAPER_CATEGORIES:
            class_true = [1 if label == category else 0 for label in true_labels]
            class_pred = [1 if label == category else 0 for label in pred_labels]
            
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                class_true, 
                class_pred, 
                average="binary"
            )
            
            class_metrics[category] = {
                "precision": class_precision,
                "recall": class_recall,
                "f1": class_f1
            }
        
        # 结果统计
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_confidence": np.mean(confidences),
            "class_metrics": class_metrics
        }
        
        # 保存评估结果
        self.results["classification"] = results
        
        # 输出结果
        logger.info(f"论文分类评估结果: accuracy={accuracy:.4f}, f1={f1:.4f}")
        
        return results
    
    def evaluate_entity_extraction(self, test_data_path: str) -> Dict[str, Any]:
        """
        评估实体提取性能
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估实体提取: {test_data_path}")
        
        # 加载测试数据
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # 预测结果
        all_true_labels = []
        all_pred_labels = []
        
        # 实体类型
        entity_types = ["MODEL", "METRIC", "DATASET", "METHOD", "TASK", "FRAMEWORK"]
        
        for item in tqdm(test_data, desc="实体提取评估"):
            # 获取文本
            tokens = item["tokens"]
            sentence = " ".join(tokens)
            
            # 获取真实标签
            true_labels = item["labels"]
            all_true_labels.extend(true_labels)
            
            # 预测
            entities = self.entity_extractor.predict_entities(sentence)
            
            # 转换预测结果为标签序列
            pred_labels = ["O"] * len(tokens)
            
            for entity in entities:
                entity_type = entity["type"]
                start = entity["start"]
                end = entity["end"]
                
                # 确保索引在范围内
                if start < len(tokens):
                    pred_labels[start] = f"B-{entity_type}"
                
                for i in range(start + 1, min(end + 1, len(tokens))):
                    pred_labels[i] = f"I-{entity_type}"
            
            all_pred_labels.extend(pred_labels)
        
        # 计算总体指标
        true_entities = self._extract_entities_from_labels(all_true_labels)
        pred_entities = self._extract_entities_from_labels(all_pred_labels)
        
        # 计算性能指标
        precision, recall, f1 = self._calculate_ner_metrics(true_entities, pred_entities)
        
        # 按实体类型计算指标
        type_metrics = {}
        
        for entity_type in entity_types:
            type_true = [e for e in true_entities if e[2] == entity_type]
            type_pred = [e for e in pred_entities if e[2] == entity_type]
            
            type_precision, type_recall, type_f1 = self._calculate_ner_metrics(type_true, type_pred)
            
            type_metrics[entity_type] = {
                "precision": type_precision,
                "recall": type_recall,
                "f1": type_f1,
                "count": len(type_true)
            }
        
        # 结果统计
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "type_metrics": type_metrics
        }
        
        # 保存评估结果
        self.results["entity_extraction"] = results
        
        # 输出结果
        logger.info(f"实体提取评估结果: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        
        return results
    
    def _extract_entities_from_labels(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        """
        从标签序列中提取实体
        
        Args:
            labels: 标签序列
            
        Returns:
            实体列表 (start, end, type)
        """
        entities = []
        current_entity = None
        
        for i, label in enumerate(labels):
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = (i, i, entity_type)
            
            elif label.startswith("I-"):
                entity_type = label[2:]
                
                if current_entity and current_entity[2] == entity_type:
                    current_entity = (current_entity[0], i, entity_type)
            
            elif label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _calculate_ner_metrics(self, true_entities: List[Tuple[int, int, str]], 
                             pred_entities: List[Tuple[int, int, str]]) -> Tuple[float, float, float]:
        """
        计算NER性能指标
        
        Args:
            true_entities: 真实实体列表
            pred_entities: 预测实体列表
            
        Returns:
            (precision, recall, f1)
        """
        if not pred_entities:
            return 0.0, 0.0, 0.0
        
        if not true_entities:
            return 0.0, 0.0, 0.0
        
        # 计算TP, FP, FN
        tp = len([e for e in pred_entities if e in true_entities])
        fp = len(pred_entities) - tp
        fn = len(true_entities) - tp
        
        # 计算精确率、召回率和F1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return precision, recall, f1
    
    def evaluate_kg_query(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估知识图谱查询性能
        
        Args:
            test_queries: 测试查询列表
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估知识图谱查询: {len(test_queries)} 个查询")
        
        # 查询结果
        query_results = []
        
        for query in tqdm(test_queries, desc="知识图谱查询评估"):
            query_type = query["type"]
            query_input = query["input"]
            expected = query.get("expected", [])
            
            # 根据查询类型进行不同的查询
            result = None
            
            if query_type == "entity":
                result = self.kg_builder.query_entity(query_input)
            elif query_type == "related_entities":
                entity_name = query_input["entity_name"]
                entity_type = query_input.get("entity_type")
                result = self.kg_builder.query_related_entities(entity_name, entity_type)
            elif query_type == "papers_by_entity":
                entity_name = query_input["entity_name"]
                entity_type = query_input.get("entity_type")
                result = self.kg_builder.query_papers_by_entity(entity_name, entity_type)
            elif query_type == "papers_by_category":
                result = self.kg_builder.query_papers_by_category(query_input)
            elif query_type == "path":
                entity1 = query_input["entity1"]
                entity2 = query_input["entity2"]
                result = self.kg_builder.find_path_between_entities(entity1, entity2)
            
            # 计算查询质量
            quality = self._evaluate_query_quality(result, expected)
            
            # 保存查询结果
            query_results.append({
                "query": query,
                "result": result,
                "quality": quality
            })
        
        # 计算平均质量
        avg_quality = np.mean([r["quality"] for r in query_results])
        
        # 结果统计
        results = {
            "avg_quality": avg_quality,
            "query_results": query_results
        }
        
        # 保存评估结果
        self.results["kg_query"] = results
        
        # 输出结果
        logger.info(f"知识图谱查询评估结果: avg_quality={avg_quality:.4f}")
        
        return results
    
    def _evaluate_query_quality(self, result: List[Dict[str, Any]], 
                               expected: List[Dict[str, Any]]) -> float:
        """
        评估查询质量
        
        Args:
            result: 查询结果
            expected: 期望结果
            
        Returns:
            查询质量 [0, 1]
        """
        # 如果没有期望结果，无法评估
        if not expected:
            return 0.5  # 默认中等质量
        
        # 如果结果为空但期望非空
        if not result and expected:
            return 0.0
        
        # 如果结果非空但期望为空
        if result and not expected:
            return 0.0
        
        # 计算结果与期望的重叠度
        # 这里使用一个简单的启发式方法
        result_names = {str(item) for item in result}
        expected_names = {str(item) for item in expected}
        
        intersection = result_names.intersection(expected_names)
        union = result_names.union(expected_names)
        
        # Jaccard相似度
        similarity = len(intersection) / len(union) if union else 1.0
        
        return similarity
    
    def evaluate_vector_search(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估向量搜索性能
        
        Args:
            test_queries: 测试查询列表
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估向量搜索: {len(test_queries)} 个查询")
        
        # 搜索结果
        search_results = []
        
        for query in tqdm(test_queries, desc="向量搜索评估"):
            query_text = query["query"]
            paper_id = query.get("paper_id")
            relevancy_judgments = query.get("relevancy", {})
            
            # 执行搜索
            if paper_id:
                results = self.vector_store.search_by_paper_id(query_text, paper_id, top_k=10)
            else:
                results = self.vector_store.search(query_text, top_k=10)
            
            # 计算搜索质量指标
            metrics = self._evaluate_search_quality(results, relevancy_judgments)
            
            # 保存搜索结果
            search_results.append({
                "query": query,
                "results": results,
                "metrics": metrics
            })
        
        # 计算平均指标
        avg_metrics = {
            "precision@3": np.mean([r["metrics"]["precision@3"] for r in search_results]),
            "precision@5": np.mean([r["metrics"]["precision@5"] for r in search_results]),
            "precision@10": np.mean([r["metrics"]["precision@10"] for r in search_results]),
            "recall@10": np.mean([r["metrics"]["recall@10"] for r in search_results]),
            "mrr": np.mean([r["metrics"]["mrr"] for r in search_results]),
            "ndcg@10": np.mean([r["metrics"]["ndcg@10"] for r in search_results])
        }
        
        # 结果统计
        results = {
            "avg_metrics": avg_metrics,
            "search_results": search_results
        }
        
        # 保存评估结果
        self.results["vector_search"] = results
        
        # 输出结果
        logger.info(f"向量搜索评估结果: precision@5={avg_metrics['precision@5']:.4f}, ndcg@10={avg_metrics['ndcg@10']:.4f}")
        
        return results
    
    def _evaluate_search_quality(self, results: List[Dict[str, Any]], 
                                relevancy_judgments: Dict[str, int]) -> Dict[str, float]:
        """
        评估搜索质量
        
        Args:
            results: 搜索结果
            relevancy_judgments: 相关性判断 {chunk_id: 相关性}
            
        Returns:
            评估指标
        """
        # 如果没有相关性判断，无法评估
        if not relevancy_judgments:
            return {
                "precision@3": 0.0,
                "precision@5": 0.0,
                "precision@10": 0.0,
                "recall@10": 0.0,
                "mrr": 0.0,
                "ndcg@10": 0.0
            }
        
        # 提取结果ID和得分
        result_ids = [r["chunk_id"] for r in results]
        
        # 计算精确率
        precision_3 = self._calculate_precision_at_k(result_ids, relevancy_judgments, 3)
        precision_5 = self._calculate_precision_at_k(result_ids, relevancy_judgments, 5)
        precision_10 = self._calculate_precision_at_k(result_ids, relevancy_judgments, 10)
        
        # 计算召回率
        recall_10 = self._calculate_recall_at_k(result_ids, relevancy_judgments, 10)
        
        # 计算MRR
        mrr = self._calculate_mrr(result_ids, relevancy_judgments)
        
        # 计算NDCG
        ndcg_10 = self._calculate_ndcg_at_k(result_ids, relevancy_judgments, 10)
        
        return {
            "precision@3": precision_3,
            "precision@5": precision_5,
            "precision@10": precision_10,
            "recall@10": recall_10,
            "mrr": mrr,
            "ndcg@10": ndcg_10
        }
    
    def _calculate_precision_at_k(self, result_ids: List[str], 
                                relevancy_judgments: Dict[str, int], 
                                k: int) -> float:
        """
        计算@K精确率
        
        Args:
            result_ids: 结果ID列表
            relevancy_judgments: 相关性判断
            k: 截断位置
            
        Returns:
            @K精确率
        """
        if not result_ids or k <= 0:
            return 0.0
        
        k = min(k, len(result_ids))
        
        # 计算相关结果数
        relevant = sum(1 for i in range(k) if i < len(result_ids) and 
                     result_ids[i] in relevancy_judgments and 
                     relevancy_judgments[result_ids[i]] > 0)
        
        return relevant / k
    
    def _calculate_recall_at_k(self, result_ids: List[str], 
                             relevancy_judgments: Dict[str, int], 
                             k: int) -> float:
        """
        计算@K召回率
        
        Args:
            result_ids: 结果ID列表
            relevancy_judgments: 相关性判断
            k: 截断位置
            
        Returns:
            @K召回率
        """
        if not result_ids or k <= 0:
            return 0.0
        
        k = min(k, len(result_ids))
        
        # 计算相关结果数
        relevant = sum(1 for i in range(k) if i < len(result_ids) and 
                     result_ids[i] in relevancy_judgments and 
                     relevancy_judgments[result_ids[i]] > 0)
        
        # 总的相关结果数
        total_relevant = sum(1 for _, rel in relevancy_judgments.items() if rel > 0)
        
        return relevant / total_relevant if total_relevant > 0 else 0.0
    
    def _calculate_mrr(self, result_ids: List[str], 
                     relevancy_judgments: Dict[str, int]) -> float:
        """
        计算MRR (Mean Reciprocal Rank)
        
        Args:
            result_ids: 结果ID列表
            relevancy_judgments: 相关性判断
            
        Returns:
            MRR
        """
        if not result_ids:
            return 0.0
        
        # 找到第一个相关结果的位置
        for i, result_id in enumerate(result_ids):
            if result_id in relevancy_judgments and relevancy_judgments[result_id] > 0:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_ndcg_at_k(self, result_ids: List[str], 
                           relevancy_judgments: Dict[str, int], 
                           k: int) -> float:
        """
        计算@K NDCG (Normalized Discounted Cumulative Gain)
        
        Args:
            result_ids: 结果ID列表
            relevancy_judgments: 相关性判断
            k: 截断位置
            
        Returns:
            @K NDCG
        """
        if not result_ids or k <= 0:
            return 0.0
        
        k = min(k, len(result_ids))
        
        # 计算DCG
        dcg = 0.0
        for i in range(k):
            if i < len(result_ids) and result_ids[i] in relevancy_judgments:
                rel = relevancy_judgments[result_ids[i]]
                dcg += (2 ** rel - 1) / np.log2(i + 2)
        
        # 计算理想DCG
        ideal_rankings = sorted(relevancy_judgments.values(), reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rankings))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_dialogues(self, test_dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估对话系统性能
        
        Args:
            test_dialogues: 测试对话列表
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估对话系统: {len(test_dialogues)} 个对话")
        
        # 对话结果
        dialogue_results = []
        
        for dialogue in tqdm(test_dialogues, desc="对话系统评估"):
            # 清除历史
            self.dialog_manager.clear_history()
            
            # 对话ID和论文ID
            dialogue_id = dialogue.get("id", "unknown")
            paper_id = dialogue.get("paper_id")
            
            # 对话轮次
            turns = dialogue["turns"]
            turn_results = []
            
            for turn in turns:
                user_query = turn["user"]
                expected_response = turn.get("expected", "")
                
                # 生成回答
                response = self.dialog_manager.process_query(user_query, paper_id)
                
                # 评估回答质量
                quality = self._evaluate_response_quality(response, expected_response)
                
                # 保存轮次结果
                turn_results.append({
                    "user": user_query,
                    "response": response,
                    "expected": expected_response,
                    "quality": quality
                })
            
            # 计算对话平均质量
            avg_quality = np.mean([t["quality"] for t in turn_results])
            
            # 保存对话结果
            dialogue_results.append({
                "dialogue_id": dialogue_id,
                "paper_id": paper_id,
                "turns": turn_results,
                "avg_quality": avg_quality
            })
        
        # 计算总体平均质量
        overall_avg_quality = np.mean([d["avg_quality"] for d in dialogue_results])
        
        # 结果统计
        results = {
            "overall_avg_quality": overall_avg_quality,
            "dialogue_results": dialogue_results
        }
        
        # 保存评估结果
        self.results["dialogues"] = results
        
        # 输出结果
        logger.info(f"对话系统评估结果: overall_avg_quality={overall_avg_quality:.4f}")
        
        return results
    
    def _evaluate_response_quality(self, response: str, expected: str) -> float:
        """
        评估回答质量
        
        Args:
            response: 系统回答
            expected: 期望回答
            
        Returns:
            回答质量 [0, 1]
        """
        # 如果没有期望回答，无法评估
        if not expected:
            return 0.5  # 默认中等质量
        
        # 这里应该使用更复杂的评估方法，如BLEU、ROUGE或BERT相似度
        # 这里使用一个简单的词重叠率作为示例
        
        # 分词
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        
        # 计算重叠率
        intersection = response_words.intersection(expected_words)
        union = response_words.union(expected_words)
        
        # Jaccard相似度
        similarity = len(intersection) / len(union) if union else 0.0
        
        return similarity
    
    def run_all_evaluations(self, eval_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行所有评估
        
        Args:
            eval_config: 评估配置
            
        Returns:
            评估结果
        """
        logger.info("开始全面系统评估")
        
        # 评估论文分类
        if "classification" in eval_config:
            self.evaluate_classification(eval_config["classification"])
        
        # 评估实体提取
        if "entity_extraction" in eval_config:
            self.evaluate_entity_extraction(eval_config["entity_extraction"])
        
        # 评估知识图谱查询
        if "kg_query" in eval_config:
            self.evaluate_kg_query(eval_config["kg_query"])
        
        # 评估向量搜索
        if "vector_search" in eval_config:
            self.evaluate_vector_search(eval_config["vector_search"])
        
        # 评估对话系统
        if "dialogues" in eval_config:
            self.evaluate_dialogues(eval_config["dialogues"])
        
        # 输出汇总结果
        self._print_summary()
        
        # 保存评估结果
        results_path = os.path.join(self.eval_data_dir, "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存到: {results_path}")
        
        return self.results
    
    def _print_summary(self):
        """
        打印评估汇总
        """
        summary = {
            "classification": {
                "accuracy": self.results.get("classification", {}).get("accuracy", 0),
                "f1": self.results.get("classification", {}).get("f1", 0)
            },
            "entity_extraction": {
                "precision": self.results.get("entity_extraction", {}).get("precision", 0),
                "recall": self.results.get("entity_extraction", {}).get("recall", 0),
                "f1": self.results.get("entity_extraction", {}).get("f1", 0)
            },
            "kg_query": {
                "avg_quality": self.results.get("kg_query", {}).get("avg_quality", 0)
            },
            "vector_search": {
                "precision@5": self.results.get("vector_search", {}).get("avg_metrics", {}).get("precision@5", 0),
                "ndcg@10": self.results.get("vector_search", {}).get("avg_metrics", {}).get("ndcg@10", 0)
            },
            "dialogues": {
                "overall_avg_quality": self.results.get("dialogues", {}).get("overall_avg_quality", 0)
            }
        }
        
        logger.info("======== 评估汇总 ========")
        logger.info(f"论文分类: 准确率={summary['classification']['accuracy']:.4f}, F1={summary['classification']['f1']:.4f}")
        logger.info(f"实体提取: 精确率={summary['entity_extraction']['precision']:.4f}, 召回率={summary['entity_extraction']['recall']:.4f}, F1={summary['entity_extraction']['f1']:.4f}")
        logger.info(f"知识图谱查询: 平均质量={summary['kg_query']['avg_quality']:.4f}")
        logger.info(f"向量搜索: P@5={summary['vector_search']['precision@5']:.4f}, NDCG@10={summary['vector_search']['ndcg@10']:.4f}")
        logger.info(f"对话系统: 平均质量={summary['dialogues']['overall_avg_quality']:.4f}")
        logger.info("==========================")