
# src/recommendation/paper_recommender.py
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

class PaperRecommender:
    """
    论文推荐系统，根据用户上传的论文或问题推荐相关论文
    """
    
    def __init__(self):
        """
        初始化推荐系统
        """
        self.vector_store = VectorStore()
        self.kg_builder = KnowledgeGraphBuilder()
    
    def recommend_by_paper(self, paper_id: str, max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        根据用户上传的论文推荐相关论文
        
        Args:
            paper_id: 论文ID
            max_recommendations: 最大推荐数量
            
        Returns:
            推荐论文列表
        """
        try:
            recommendations = []
            
            # 步骤1: 从知识图谱获取论文实体和类别
            paper_entities = self._get_paper_entities(paper_id)
            paper_category = self._get_paper_category(paper_id)
            
            # 步骤2: 基于实体寻找相关论文
            entity_based_papers = self._recommend_by_entities(paper_entities, paper_id)
            
            # 步骤3: 基于类别寻找相关论文
            category_based_papers = self._recommend_by_category(paper_category, paper_id)
            
            # 步骤4: 合并推荐结果，去重
            seen_papers = set()
            
            # 先添加基于实体的推荐（可能更相关）
            for paper in entity_based_papers:
                if paper["paper_id"] not in seen_papers:
                    recommendations.append(paper)
                    seen_papers.add(paper["paper_id"])
                    
                    if len(recommendations) >= max_recommendations:
                        break
            
            # 然后添加基于类别的推荐
            if len(recommendations) < max_recommendations:
                for paper in category_based_papers:
                    if paper["paper_id"] not in seen_papers:
                        recommendations.append(paper)
                        seen_papers.add(paper["paper_id"])
                        
                        if len(recommendations) >= max_recommendations:
                            break
            
            return recommendations
        except Exception as e:
            logger.error(f"根据论文推荐相关论文时出错: {e}")
            return []
    
    def recommend_by_query(self, query: str, max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        根据用户问题推荐相关论文
        
        Args:
            query: 用户问题
            max_recommendations: 最大推荐数量
            
        Returns:
            推荐论文列表
        """
        try:
            # 步骤1: 从查询提取实体
            entities = self._extract_entities_from_query(query)
            
            # 步骤2: 使用向量数据库搜索相关文本块
            relevant_chunks = self.vector_store.search(query, top_k=10)
            
            # 步骤3: 统计文档频率并汇总
            paper_scores = {}
            
            # 基于文档块的相似度评分
            for chunk in relevant_chunks:
                paper_id = chunk["paper_id"]
                score = chunk["score"]
                
                if paper_id not in paper_scores:
                    paper_scores[paper_id] = {"score": 0, "chunks": 0}
                
                paper_scores[paper_id]["score"] += score
                paper_scores[paper_id]["chunks"] += 1
            
            # 计算平均分
            for paper_id in paper_scores:
                paper_scores[paper_id]["avg_score"] = (
                    paper_scores[paper_id]["score"] / paper_scores[paper_id]["chunks"]
                )
            
            # 步骤4: 基于实体添加额外权重
            entity_papers = self._get_papers_by_entities(entities)
            for paper in entity_papers:
                paper_id = paper["paper_id"]
                if paper_id in paper_scores:
                    # 为包含查询实体的论文增加权重
                    paper_scores[paper_id]["avg_score"] += 0.2
                else:
                    # 添加新论文
                    paper_scores[paper_id] = {
                        "score": 0.6,  # 基础分
                        "chunks": 1,
                        "avg_score": 0.6
                    }
            
            # 步骤5: 排序并选择前N篇
            sorted_papers = sorted(
                paper_scores.items(),
                key=lambda x: x[1]["avg_score"],
                reverse=True
            )
            
            # 步骤6: 获取论文详细信息
            recommendations = []
            seen_papers = set()
            
            for paper_id, _ in sorted_papers[:max_recommendations]:
                if paper_id not in seen_papers:
                    paper_info = self._get_paper_info(paper_id)
                    if paper_info:
                        recommendations.append(paper_info)
                        seen_papers.add(paper_id)
            
            return recommendations
        except Exception as e:
            logger.error(f"根据查询推荐相关论文时出错: {e}")
            return []
    
    def _get_paper_entities(self, paper_id: str) -> List[str]:
        """
        从知识图谱获取论文相关实体
        
        Args:
            paper_id: 论文ID
            
        Returns:
            实体列表
        """
        entities = []
        
        try:
            with self.kg_builder.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Paper)-[:MENTIONS]->(e)
                    WHERE id(p) = $paper_id
                    RETURN e.name as entity
                    """,
                    paper_id=paper_id
                )
                
                for record in result:
                    entities.append(record["entity"])
        except Exception as e:
            logger.error(f"获取论文实体时出错: {e}")
        
        return entities
    
    def _get_paper_category(self, paper_id: str) -> str:
        """
        获取论文类别
        
        Args:
            paper_id: 论文ID
            
        Returns:
            论文类别
        """
        try:
            with self.kg_builder.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    WHERE id(p) = $paper_id
                    RETURN p.category as category
                    """,
                    paper_id=paper_id
                )
                
                record = result.single()
                if record:
                    return record["category"]
        except Exception as e:
            logger.error(f"获取论文类别时出错: {e}")
        
        return "Unknown"
    
    def _recommend_by_entities(self, entities: List[str], exclude_paper_id: str) -> List[Dict[str, Any]]:
        """
        基于实体推荐论文
        
        Args:
            entities: 实体列表
            exclude_paper_id: 要排除的论文ID
            
        Returns:
            推荐论文列表
        """
        recommendations = []
        
        try:
            with self.kg_builder.driver.session() as session:
                # 在查询中使用IN运算符
                result = session.run(
                    """
                    MATCH (p:Paper)-[:MENTIONS]->(e)
                    WHERE e.name IN $entities AND id(p) <> $exclude_paper_id
                    WITH p, count(DISTINCT e) as commonEntities
                    ORDER BY commonEntities DESC
                    RETURN id(p) as paper_id, p.title as title, p.author as author, 
                           p.category as category, commonEntities
                    LIMIT 10
                    """,
                    entities=entities,
                    exclude_paper_id=exclude_paper_id
                )
                
                for record in result:
                    recommendations.append({
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "author": record["author"],
                        "category": record["category"],
                        "relevance_score": record["commonEntities"] / len(entities) if entities else 0
                    })
        except Exception as e:
            logger.error(f"基于实体推荐论文时出错: {e}")
        
        return recommendations
    
    def _recommend_by_category(self, category: str, exclude_paper_id: str) -> List[Dict[str, Any]]:
        """
        基于类别推荐论文
        
        Args:
            category: 论文类别
            exclude_paper_id: 要排除的论文ID
            
        Returns:
            推荐论文列表
        """
        recommendations = []
        
        try:
            with self.kg_builder.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    WHERE p.category = $category AND id(p) <> $exclude_paper_id
                    RETURN id(p) as paper_id, p.title as title, p.author as author, 
                           p.category as category
                    LIMIT 10
                    """,
                    category=category,
                    exclude_paper_id=exclude_paper_id
                )
                
                for record in result:
                    recommendations.append({
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "author": record["author"],
                        "category": record["category"],
                        "relevance_score": 0.5  # 基础相关性分数
                    })
        except Exception as e:
            logger.error(f"基于类别推荐论文时出错: {e}")
        
        return recommendations
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        从查询中提取实体
        
        Args:
            query: 用户查询
            
        Returns:
            提取的实体列表
        """
        # 简单实现：从查询中提取名词短语
        # 实际系统中应使用NER模型
        
        # 使用知识图谱中的实体进行匹配
        entities = []
        try:
            with self.kg_builder.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e)
                    WHERE NOT e:Paper
                    RETURN e.name as entity
                    """
                )
                
                all_entities = [record["entity"] for record in result]
                
                # 简单匹配
                for entity in all_entities:
                    if entity and entity.lower() in query.lower():
                        entities.append(entity)
        except Exception as e:
            logger.error(f"从查询提取实体时出错: {e}")
        
        return entities
    
    def _get_papers_by_entities(self, entities: List[str]) -> List[Dict[str, Any]]:
        """
        根据实体获取论文
        
        Args:
            entities: 实体列表
            
        Returns:
            论文列表
        """
        papers = []
        
        if not entities:
            return papers
        
        try:
            with self.kg_builder.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Paper)-[:MENTIONS]->(e)
                    WHERE e.name IN $entities
                    WITH p, count(DISTINCT e) as commonEntities
                    ORDER BY commonEntities DESC
                    RETURN id(p) as paper_id, p.title as title, p.author as author, 
                           p.category as category, commonEntities
                    LIMIT 10
                    """,
                    entities=entities
                )
                
                for record in result:
                    papers.append({
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "author": record["author"],
                        "category": record["category"],
                        "relevance_score": record["commonEntities"] / len(entities)
                    })
        except Exception as e:
            logger.error(f"根据实体获取论文时出错: {e}")
        
        return papers
    
    def _get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        获取论文详细信息
        
        Args:
            paper_id: 论文ID
            
        Returns:
            论文信息
        """
        try:
            with self.kg_builder.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    WHERE id(p) = $paper_id
                    RETURN p.title as title, p.author as author, p.category as category,
                           p.filename as filename
                    """,
                    paper_id=paper_id
                )
                
                record = result.single()
                if record:
                    return {
                        "paper_id": paper_id,
                        "title": record["title"],
                        "author": record["author"],
                        "category": record["category"],
                        "filename": record["filename"]
                    }
        except Exception as e:
            logger.error(f"获取论文信息时出错: {e}")
        
        return None
