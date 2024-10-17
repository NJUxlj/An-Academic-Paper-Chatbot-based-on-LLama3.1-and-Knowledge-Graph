# knowledge_graph.py  

import rdflib  

def build_knowledge_graph(entities, relations):  
    g = rdflib.Graph()  

    for entity in entities:  
        # 添加实体到图中  
        pass  # 实现具体的逻辑  

    for relation in relations:  
        # 添加关系到图中  
        pass  # 实现具体的逻辑  

    return g  

def query_knowledge_graph(graph, query_intent):  
    # 根据意图在知识图谱中查询  
    pass  # 实现具体的查询逻辑