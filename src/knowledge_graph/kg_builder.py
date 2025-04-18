
# src/knowledge_graph/kg_builder.py
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from neo4j import GraphDatabase
from src.configs.config import Config
from src.models.entity_extractor.bert_bilstm_crf import EntityTripleExtractor

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """
    知识图谱构建模块，负责将抽取的三元组插入Neo4j数据库
    """
    
    def __init__(self):
        """
        初始化知识图谱构建器
        """
        self.uri = Config.NEO4J_CONFIG["uri"]
        self.auth = Config.NEO4J_CONFIG["auth"]
        self.database = Config.NEO4J_CONFIG["database"]
        
        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=self.auth,
            database=self.database
        )
        
        # 初始化实体抽取器
        self.extractor = EntityTripleExtractor(
            model_path=Config.MODEL_CONFIG["bert_ner"]["model_path"],
            device=Config.MODEL_CONFIG["qwen"]["device"]
        )
    
    def close(self):
        """
        关闭数据库连接
        """
        self.driver.close()
    
    def build_graph_from_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从论文数据构建知识图谱
        
        Args:
            paper_data: 论文数据，包含文本、元数据和分类结果
            
        Returns:
            图谱构建结果统计
        """
        try:
            metadata = paper_data.get("metadata", {})
            text = paper_data.get("full_text", "")
            paper_framework = paper_data.get("framework", {})
            category = paper_data.get("category", "Unknown")
            
            # 创建论文节点
            paper_id = self._create_paper_node(metadata, category)
            
            # 抽取三元组
            triples = self.extractor.extract_triples(text)
            
            # 将三元组插入图谱
            stats = self._insert_triples(triples, paper_id)
            
            # 保存论文框架
            if paper_framework:
                self._save_paper_framework(paper_id, paper_framework)
            
            logger.info(f"论文 {metadata.get('title', 'Unknown')} 知识图谱构建完成")
            return stats
        except Exception as e:
            logger.error(f"构建知识图谱时出错: {e}")
            return {"error": str(e)}
    
    def _create_paper_node(self, metadata: Dict[str, str], category: str) -> str:
        """
        创建论文节点
        
        Args:
            metadata: 论文元数据
            category: 论文类别
            
        Returns:
            论文节点ID
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (p:Paper {title: $title}) 
                ON CREATE SET 
                    p.author = $author,
                    p.date = $date,
                    p.category = $category,
                    p.filename = $filename
                RETURN id(p) as paper_id
                """,
                title=metadata.get("title", "Unknown Title"),
                author=metadata.get("author", "Unknown Author"),
                date=metadata.get("creation_date", ""),
                category=category,
                filename=metadata.get("filename", "")
            )
            return result.single()["paper_id"]
    
    def _insert_triples(self, triples: List[Dict[str, str]], paper_id: str) -> Dict[str, int]:
        """
        将三元组插入图谱
        
        Args:
            triples: 三元组列表
            paper_id: 论文节点ID
            
        Returns:
            插入统计
        """
        stats = {
            "entities_created": 0,
            "relationships_created": 0
        }
        
        with self.driver.session() as session:
            for triple in triples:
                head = triple["head"]
                head_type = triple["head_type"]
                relation = triple["relation"]
                tail = triple["tail"]
                tail_type = triple["tail_type"]
                
                # 创建头实体
                head_result = session.run(
                    f"""
                    MERGE (h:{head_type} {{name: $name}})
                    ON CREATE SET h.created_at = datetime()
                    RETURN id(h) as entity_id, h.created_at as created_at
                    """,
                    name=head
                )
                head_data = head_result.single()
                if head_data.get("created_at") is not None:
                    stats["entities_created"] += 1
                
                # 创建尾实体
                tail_result = session.run(
                    f"""
                    MERGE (t:{tail_type} {{name: $name}})
                    ON CREATE SET t.created_at = datetime()
                    RETURN id(t) as entity_id, t.created_at as created_at
                    """,
                    name=tail
                )
                tail_data = tail_result.single()
                if tail_data.get("created_at") is not None:
                    stats["entities_created"] += 1
                
                # 创建三元组关系
                rel_result = session.run(
                    f"""
                    MATCH (h:{head_type} {{name: $head_name}})
                    MATCH (t:{tail_type} {{name: $tail_name}})
                    MERGE (h)-[r:{relation}]->(t)
                    ON CREATE SET r.created_at = datetime()
                    RETURN id(r) as rel_id, r.created_at as created_at
                    """,
                    head_name=head,
                    tail_name=tail
                )
                rel_data = rel_result.single()
                if rel_data.get("created_at") is not None:
                    stats["relationships_created"] += 1
                
                # 将实体与论文关联
                session.run(
                    f"""
                    MATCH (p:Paper) WHERE id(p) = $paper_id
                    MATCH (e:{head_type} {{name: $entity_name}})
                    MERGE (p)-[r:MENTIONS]->(e)
                    """,
                    paper_id=paper_id,
                    entity_name=head
                )
                
                session.run(
                    f"""
                    MATCH (p:Paper) WHERE id(p) = $paper_id
                    MATCH (e:{tail_type} {{name: $entity_name}})
                    MERGE (p)-[r:MENTIONS]->(e)
                    """,
                    paper_id=paper_id,
                    entity_name=tail
                )
        
        return stats
    
    def _save_paper_framework(self, paper_id: str, framework: Dict[str, str]) -> None:
        """
        保存论文框架
        
        Args:
            paper_id: 论文节点ID
            framework: 论文框架
        """
        with self.driver.session() as session:
            for section, content in framework.items():
                if content:  # 跳过空内容
                    session.run(
                        """
                        MATCH (p:Paper) WHERE id(p) = $paper_id
                        SET p.`framework_` + $section = $content
                        """,
                        paper_id=paper_id,
                        section=section,
                        content=content
                    )
    
    def query_entity(self, entity_name: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询实体
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）
            
        Returns:
            查询结果列表
        """
        with self.driver.session() as session:
            if entity_type:
                result = session.run(
                    f"""
                    MATCH (e:{entity_type})
                    WHERE e.name CONTAINS $name
                    RETURN e.name as name, labels(e) as types
                    """,
                    name=entity_name
                )
            else:
                result = session.run(
                    """
                    MATCH (e)
                    WHERE e.name CONTAINS $name AND NOT e:Paper
                    RETURN e.name as name, labels(e) as types
                    """,
                    name=entity_name
                )
            
            return [record.data() for record in result]
    
    def query_related_entities(self, entity_name: str, 
                              entity_type: Optional[str] = None, 
                              max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        查询与实体相关的其他实体
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）
            max_depth: 最大关系深度
            
        Returns:
            相关实体列表
        """
        with self.driver.session() as session:
            if entity_type:
                query = f"""
                MATCH path = (e:{entity_type})-[*1..{max_depth}]-(related)
                WHERE e.name = $name AND NOT related:Paper
                RETURN related.name as name, labels(related) as types, 
                       [rel in relationships(path) | type(rel)] as relations,
                       length(path) as depth
                ORDER BY depth
                LIMIT 100
                """
            else:
                query = f"""
                MATCH path = (e)-[*1..{max_depth}]-(related)
                WHERE e.name = $name AND NOT e:Paper AND NOT related:Paper
                RETURN related.name as name, labels(related) as types, 
                       [rel in relationships(path) | type(rel)] as relations,
                       length(path) as depth
                ORDER BY depth
                LIMIT 100
                """
            
            result = session.run(query, name=entity_name)
            return [record.data() for record in result]
    
    def query_papers_by_entity(self, entity_name: str, 
                              entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询提到指定实体的论文
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）
            
        Returns:
            论文列表
        """
        with self.driver.session() as session:
            if entity_type:
                query = f"""
                MATCH (p:Paper)-[:MENTIONS]->(e:{entity_type} {{name: $name}})
                RETURN p.title as title, p.author as author, p.category as category
                """
            else:
                query = """
                MATCH (p:Paper)-[:MENTIONS]->(e {name: $name})
                RETURN p.title as title, p.author as author, p.category as category
                """
            
            result = session.run(query, name=entity_name)
            return [record.data() for record in result]
    
    def query_papers_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        查询指定类别的论文
        
        Args:
            category: 论文类别
            
        Returns:
            论文列表
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper)
                WHERE p.category = $category
                RETURN p.title as title, p.author as author, p.category as category
                """,
                category=category
            )
            return [record.data() for record in result]
    
    def find_path_between_entities(self, 
                                 entity1_name: str, 
                                 entity2_name: str,
                                 max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        查找两个实体之间的路径
        
        Args:
            entity1_name: 第一个实体名称
            entity2_name: 第二个实体名称
            max_depth: 最大路径深度
            
        Returns:
            路径列表
        """
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (e1)-[*1..{max_depth}]-(e2)
                WHERE e1.name = $name1 AND e2.name = $name2
                RETURN [node in nodes(path) | node.name] as entities,
                       [rel in relationships(path) | type(rel)] as relations,
                       length(path) as length
                ORDER BY length
                LIMIT 10
                """,
                name1=entity1_name,
                name2=entity2_name
            )
            return [record.data() for record in result]
