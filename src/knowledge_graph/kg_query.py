# src/knowledge_graph/kg_query.py
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from neo4j import GraphDatabase
import json
from src.configs.config import Config

logger = logging.getLogger(__name__)

class KnowledgeGraphQuerier:
    """
    知识图谱查询模块，提供高级查询功能
    """
    
    def __init__(self):
        """
        初始化知识图谱查询器
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
    
    def close(self):
        """
        关闭数据库连接
        """
        self.driver.close()
    
    def query_entity_neighbors(self, entity_name: str, 
                             entity_type: Optional[str] = None, 
                             depth: int = 1) -> Dict[str, Any]:
        """
        查询实体邻居节点和关系
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）
            depth: 查询深度
            
        Returns:
            邻居节点和关系
        """
        nodes = []
        relationships = []
        
        try:
            with self.driver.session() as session:
                # 构建查询语句
                if entity_type:
                    query = f"""
                    MATCH (e:{entity_type} {{name: $name}})
                    CALL apoc.path.expand(e, "", "", 1, {depth}) YIELD path
                    WITH path, [n IN nodes(path) | n] as nodes, relationships(path) AS rels
                    UNWIND nodes AS node
                    WITH collect({{id: id(node), labels: labels(node), properties: node}}) AS nodes, rels
                    UNWIND rels AS rel
                    RETURN nodes, collect({{
                        id: id(rel),
                        type: type(rel),
                        source: id(startNode(rel)),
                        target: id(endNode(rel)),
                        properties: rel
                    }}) AS relationships
                    """
                else:
                    query = f"""
                    MATCH (e {{name: $name}})
                    CALL apoc.path.expand(e, "", "", 1, {depth}) YIELD path
                    WITH path, [n IN nodes(path) | n] as nodes, relationships(path) AS rels
                    UNWIND nodes AS node
                    WITH collect({{id: id(node), labels: labels(node), properties: node}}) AS nodes, rels
                    UNWIND rels AS rel
                    RETURN nodes, collect({{
                        id: id(rel),
                        type: type(rel),
                        source: id(startNode(rel)),
                        target: id(endNode(rel)),
                        properties: rel
                    }}) AS relationships
                    """
                
                # 执行查询
                result = session.run(query, name=entity_name)
                record = result.single()
                
                if record:
                    nodes = record["nodes"]
                    relationships = record["relationships"]
        except Exception as e:
            logger.error(f"查询实体邻居时出错: {e}")
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }
    
    def query_entity_details(self, entity_name: str, entity_type: Optional[str] = None) -> Dict[str, Any]:
        """
        查询实体详细信息
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）
            
        Returns:
            实体详细信息
        """
        entity_details = {
            "name": entity_name,
            "type": entity_type,
            "properties": {},
            "relations": [],
            "papers": []
        }
        
        try:
            with self.driver.session() as session:
                # 查询实体属性
                if entity_type:
                    props_query = f"""
                    MATCH (e:{entity_type} {{name: $name}})
                    RETURN properties(e) AS props
                    """
                else:
                    props_query = """
                    MATCH (e {name: $name})
                    RETURN properties(e) AS props, labels(e) AS types
                    """
                
                props_result = session.run(props_query, name=entity_name)
                props_record = props_result.single()
                
                if props_record:
                    entity_details["properties"] = props_record["props"]
                    
                    # 如果未提供实体类型，从查询结果获取
                    if not entity_type and "types" in props_record and props_record["types"]:
                        entity_details["type"] = props_record["types"][0]
                
                # 查询实体关系
                query = """
                MATCH (e {name: $name})-[r]-(other)
                RETURN type(r) AS relation, other.name AS related_entity, labels(other) AS types,
                       startNode(r).name AS source, endNode(r).name AS target
                """
                
                result = session.run(query, name=entity_name)
                
                for record in result:
                    relation_type = record["relation"]
                    related_entity = record["related_entity"]
                    entity_type = record["types"][0] if record["types"] else "Unknown"
                    
                    # 确定关系方向
                    direction = "outgoing" if record["source"] == entity_name else "incoming"
                    
                    entity_details["relations"].append({
                        "relation": relation_type,
                        "entity": related_entity,
                        "entity_type": entity_type,
                        "direction": direction
                    })
                
                # 查询相关论文
                papers_query = """
                MATCH (p:Paper)-[:MENTIONS]->(e {name: $name})
                RETURN p.title AS title, p.author AS author, p.category AS category, id(p) AS paper_id
                LIMIT 10
                """
                
                papers_result = session.run(papers_query, name=entity_name)
                
                for paper in papers_result:
                    entity_details["papers"].append({
                        "title": paper["title"],
                        "author": paper["author"],
                        "category": paper["category"],
                        "paper_id": paper["paper_id"]
                    })
        except Exception as e:
            logger.error(f"查询实体详细信息时出错: {e}")
        
        return entity_details
    
    def query_relation_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        查询知识图谱中的关系模式
        
        Args:
            limit: 返回的最大结果数
            
        Returns:
            关系模式列表
        """
        patterns = []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (s)-[r]->(t)
                WHERE NOT s:Paper AND NOT t:Paper
                WITH labels(s)[0] AS source_type, type(r) AS relation, labels(t)[0] AS target_type,
                     count(*) AS frequency
                RETURN source_type, relation, target_type, frequency
                ORDER BY frequency DESC
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                
                for record in result:
                    patterns.append({
                        "source_type": record["source_type"],
                        "relation": record["relation"],
                        "target_type": record["target_type"],
                        "frequency": record["frequency"]
                    })
        except Exception as e:
            logger.error(f"查询关系模式时出错: {e}")
        
        return patterns
    
    def query_latest_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        查询最新添加的论文
        
        Args:
            limit: 返回的最大结果数
            
        Returns:
            论文列表
        """
        papers = []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:Paper)
                RETURN id(p) AS paper_id, p.title AS title, p.author AS author,
                       p.category AS category, p.filename AS filename
                ORDER BY id(p) DESC
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                
                for record in result:
                    papers.append({
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "author": record["author"],
                        "category": record["category"],
                        "filename": record["filename"]
                    })
        except Exception as e:
            logger.error(f"查询最新论文时出错: {e}")
        
        return papers
    
    def query_related_entities(self, entity1: str, entity2: str) -> Dict[str, Any]:
        """
        查询两个实体之间的关系
        
        Args:
            entity1: 第一个实体名称
            entity2: 第二个实体名称
            
        Returns:
            关系信息
        """
        results = {
            "direct_relation": None,
            "paths": [],
            "common_papers": []
        }
        
        try:
            with self.driver.session() as session:
                # 查询直接关系
                direct_query = """
                MATCH (e1 {name: $name1})-[r]-(e2 {name: $name2})
                RETURN type(r) AS relation, 
                       startNode(r).name AS source, 
                       endNode(r).name AS target
                LIMIT 1
                """
                
                direct_result = session.run(direct_query, name1=entity1, name2=entity2)
                direct_record = direct_result.single()
                
                if direct_record:
                    # 确定关系方向
                    direction = "outgoing" if direct_record["source"] == entity1 else "incoming"
                    
                    results["direct_relation"] = {
                        "relation": direct_record["relation"],
                        "direction": direction
                    }
                
                # 查询路径
                paths_query = """
                MATCH path = shortestPath((e1 {name: $name1})-[*1..3]-(e2 {name: $name2}))
                WHERE NOT e1:Paper AND NOT e2:Paper
                RETURN [node in nodes(path) | node.name] AS entities,
                       [rel in relationships(path) | type(rel)] AS relations
                LIMIT 3
                """
                
                paths_result = session.run(paths_query, name1=entity1, name2=entity2)
                
                for record in paths_result:
                    results["paths"].append({
                        "entities": record["entities"],
                        "relations": record["relations"]
                    })
                
                # 查询共同论文
                papers_query = """
                MATCH (p:Paper)-[:MENTIONS]->(e1 {name: $name1})
                MATCH (p)-[:MENTIONS]->(e2 {name: $name2})
                RETURN p.title AS title, p.author AS author, p.category AS category, id(p) AS paper_id
                LIMIT 5
                """
                
                papers_result = session.run(papers_query, name1=entity1, name2=entity2)
                
                for paper in papers_result:
                    results["common_papers"].append({
                        "title": paper["title"],
                        "author": paper["author"],
                        "category": paper["category"],
                        "paper_id": paper["paper_id"]
                    })
        except Exception as e:
            logger.error(f"查询实体关系时出错: {e}")
        
        return results
    
    def query_entity_statistics(self) -> Dict[str, Any]:
        """
        查询实体统计信息
        
        Returns:
            统计信息
        """
        stats = {
            "entity_counts": {},
            "relation_counts": {},
            "paper_count": 0,
            "total_triples": 0
        }
        
        try:
            with self.driver.session() as session:
                # 查询实体数量
                entity_query = """
                MATCH (n)
                WHERE NOT n:Paper
                RETURN labels(n)[0] AS type, count(*) AS count
                """
                
                entity_result = session.run(entity_query)
                
                for record in entity_result:
                    stats["entity_counts"][record["type"]] = record["count"]
                
                # 查询关系数量
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                """
                
                relation_result = session.run(relation_query)
                
                for record in relation_result:
                    stats["relation_counts"][record["type"]] = record["count"]
                
                # 查询论文数量
                paper_query = """
                MATCH (p:Paper)
                RETURN count(*) AS count
                """
                
                paper_result = session.run(paper_query)
                paper_record = paper_result.single()
                
                if paper_record:
                    stats["paper_count"] = paper_record["count"]
                
                # 计算总三元组数量
                stats["total_triples"] = sum(stats["relation_counts"].values())
        except Exception as e:
            logger.error(f"查询实体统计信息时出错: {e}")
        
        return stats
    
    def query_knowledge_graph_for_query(self, query: str) -> Dict[str, Any]:
        """
        根据用户查询从知识图谱获取相关信息
        
        Args:
            query: 用户查询
            
        Returns:
            查询结果
        """
        # 提取查询中的关键实体（简单实现，实际应使用NER）
        entities = self._extract_entities_from_query(query)
        
        results = {
            "entities": [],
            "relations": [],
            "papers": []
        }
        
        # 查询每个实体的信息
        for entity in entities:
            entity_details = self.query_entity_details(entity)
            
            if entity_details["properties"]:
                results["entities"].append(entity_details)
                
                # 添加关系
                for relation in entity_details["relations"]:
                    if relation not in results["relations"]:
                        results["relations"].append(relation)
                
                # 添加论文
                for paper in entity_details["papers"]:
                    if paper not in results["papers"]:
                        results["papers"].append(paper)
        
        # 检查实体之间的关系
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                
                relation_info = self.query_related_entities(entity1, entity2)
                
                if relation_info["direct_relation"] or relation_info["paths"]:
                    results["entity_relations"] = results.get("entity_relations", [])
                    results["entity_relations"].append({
                        "entity1": entity1,
                        "entity2": entity2,
                        "info": relation_info
                    })
        
        return results
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        从查询中提取实体（简单实现）
        
        Args:
            query: 用户查询
            
        Returns:
            实体列表
        """
        # 在实际系统中应使用NER模型
        # 这里简单实现为从知识图谱中匹配可能的实体
        
        entities = []
        
        try:
            with self.driver.session() as session:
                # 获取所有实体
                result = session.run(
                    """
                    MATCH (e) 
                    WHERE NOT e:Paper
                    RETURN e.name AS name 
                    """
                )
                
                all_entities = [record["name"] for record in result if record["name"]]
                
                # 匹配查询中的实体
                query_lower = query.lower()
                for entity in all_entities:
                    if entity and entity.lower() in query_lower:
                        entities.append(entity)
        except Exception as e:
            logger.error(f"从查询提取实体时出错: {e}")
        
        return entities
    
    def search_entities(self, keyword: str, entity_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索匹配关键词的实体
        
        Args:
            keyword: 搜索关键词
            entity_type: 实体类型（可选）
            limit: 返回的最大结果数
            
        Returns:
            实体列表
        """
        entities = []
        
        try:
            with self.driver.session() as session:
                # 构建查询
                if entity_type:
                    query = f"""
                    MATCH (e:{entity_type})
                    WHERE toLower(e.name) CONTAINS toLower($keyword)
                    RETURN e.name AS name, labels(e)[0] AS type
                    LIMIT $limit
                    """
                else:
                    query = """
                    MATCH (e)
                    WHERE NOT e:Paper AND toLower(e.name) CONTAINS toLower($keyword)
                    RETURN e.name AS name, labels(e)[0] AS type
                    LIMIT $limit
                    """
                
                # 执行查询
                result = session.run(query, keyword=keyword, limit=limit)
                
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"]
                    })
        except Exception as e:
            logger.error(f"搜索实体时出错: {e}")
        
        return entities
    
    def get_entity_hierarchy(self, entity_name: str, entity_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取实体层次结构
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）
            
        Returns:
            层次结构
        """
        hierarchy = {
            "root": {"name": entity_name, "type": entity_type},
            "children": []
        }
        
        try:
            with self.driver.session() as session:
                # 查询下层实体（part_of关系）
                if entity_type:
                    query = f"""
                    MATCH (e:{entity_type} {{name: $name}})<-[:part_of]-(child)
                    RETURN child.name AS name, labels(child)[0] AS type
                    """
                else:
                    query = """
                    MATCH (e {name: $name})<-[:part_of]-(child)
                    RETURN child.name AS name, labels(child)[0] AS type
                    """
                
                # 执行查询
                result = session.run(query, name=entity_name)
                
                for record in result:
                    child = {
                        "name": record["name"],
                        "type": record["type"],
                        "children": []
                    }
                    
                    # 递归查询子实体的子实体
                    child_children = self._get_children(record["name"], record["type"])
                    if child_children:
                        child["children"] = child_children
                    
                    hierarchy["children"].append(child)
                
                # 查询父实体
                parent_query = """
                MATCH (e {name: $name})-[:part_of]->(parent)
                RETURN parent.name AS name, labels(parent)[0] AS type
                """
                
                parent_result = session.run(parent_query, name=entity_name)
                parent_record = parent_result.single()
                
                if parent_record:
                    hierarchy["parent"] = {
                        "name": parent_record["name"],
                        "type": parent_record["type"]
                    }
        except Exception as e:
            logger.error(f"获取实体层次结构时出错: {e}")
        
        return hierarchy
    
    def _get_children(self, entity_name: str, entity_type: str) -> List[Dict[str, Any]]:
        """
        递归获取子实体
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型
            
        Returns:
            子实体列表
        """
        children = []
        
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (e:{entity_type} {{name: $name}})<-[:part_of]-(child)
                RETURN child.name AS name, labels(child)[0] AS type
                """
                
                result = session.run(query, name=entity_name)
                
                for record in result:
                    child = {
                        "name": record["name"],
                        "type": record["type"],
                        "children": []
                    }
                    
                    # 递归深度限制为2级
                    children.append(child)
        except Exception as e:
            logger.error(f"获取子实体时出错: {e}")
        
        return children
    
    def get_knowledge_graph_summary(self) -> Dict[str, Any]:
        """
        获取知识图谱概要信息
        
        Returns:
            概要信息
        """
        summary = {
            "entity_counts": {},
            "relation_counts": {},
            "paper_count": 0,
            "top_entities": {},
            "top_relations": []
        }
        
        try:
            with self.driver.session() as session:
                # 查询实体类型数量
                entity_query = """
                MATCH (n)
                WHERE NOT n:Paper
                RETURN labels(n)[0] AS type, count(*) AS count
                """
                
                entity_result = session.run(entity_query)
                
                for record in entity_result:
                    summary["entity_counts"][record["type"]] = record["count"]
                
                # 查询关系类型数量
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY count DESC
                """
                
                relation_result = session.run(relation_query)
                
                for record in relation_result:
                    summary["relation_counts"][record["type"]] = record["count"]
                
                # 查询论文数量
                paper_query = """
                MATCH (p:Paper)
                RETURN count(*) AS count
                """
                
                paper_result = session.run(paper_query)
                paper_record = paper_result.single()
                
                if paper_record:
                    summary["paper_count"] = paper_record["count"]
                
                # 查询每种类型的热门实体
                for entity_type in summary["entity_counts"].keys():
                    top_query = f"""
                    MATCH (e:{entity_type})-[r]-()
                    RETURN e.name AS name, count(r) AS count
                    ORDER BY count DESC
                    LIMIT 5
                    """
                    
                    top_result = session.run(top_query)
                    summary["top_entities"][entity_type] = [
                        {"name": record["name"], "count": record["count"]}
                        for record in top_result
                    ]
                
                # 查询最常见的关系模式
                pattern_query = """
                MATCH (s)-[r]->(t)
                WHERE NOT s:Paper AND NOT t:Paper
                WITH labels(s)[0] AS source_type, type(r) AS relation, labels(t)[0] AS target_type,
                     count(*) AS frequency
                RETURN source_type, relation, target_type, frequency
                ORDER BY frequency DESC
                LIMIT 10
                """
                
                pattern_result = session.run(pattern_query)
                
                for record in pattern_result:
                    summary["top_relations"].append({
                        "source_type": record["source_type"],
                        "relation": record["relation"],
                        "target_type": record["target_type"],
                        "frequency": record["frequency"]
                    })
        except Exception as e:
            logger.error(f"获取知识图谱概要信息时出错: {e}")
        
        return summary