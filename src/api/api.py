# src/api/api.py
import os
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.configs.config import Config
from src.dialogue_system.dialog_manager import DialogManager
from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder
from src.knowledge_graph.kg_query import KnowledgeGraphQuerier
from src.knowledge_base.vector_store import VectorStore
from src.utils.pdf_parser import PDFParser
from src.models.classifier.qwen_classifier import PaperFrameworkClassifier
from src.models.paper_framework.extractor import PaperFrameworkExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="学术论文对话系统API",
    description="基于Qwen2.5和知识图谱的学术论文对话系统API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名跨域访问（开发环境常用）
    allow_credentials=True,   # 允许携带凭据（如cookies）
    allow_methods=["*"],    # 允许所有HTTP方法
    allow_headers=["*"],   # 允许所有请求头
)

# 全局组件
dialog_manager = DialogManager()
kg_builder = KnowledgeGraphBuilder()
kg_querier = KnowledgeGraphQuerier()
vector_store = VectorStore()
pdf_parser = PDFParser()
classifier = PaperFrameworkClassifier()
extractor = PaperFrameworkExtractor()

# 请求模型
class QuestionRequest(BaseModel):
    question: str
    paper_id: Optional[str] = None

class RecommendationRequest(BaseModel):
    question: Optional[str] = None
    paper_id: Optional[str] = None
    max_recommendations: int = Field(default=5, ge=1, le=20)

class EntityRequest(BaseModel):
    name: str
    type: Optional[str] = None

# 启动事件
@app.on_event("startup")
async def startup_event():
    logger.info("API服务启动")

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API服务关闭")
    kg_querier.close()
    vector_store.close()

# 健康检查端点
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

# 问答端点
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # 处理问题
        response = dialog_manager.process_query(request.question, request.paper_id)
        
        # 获取推荐
        recommendations = get_paper_recommendations(request.question, request.paper_id)
        
        return {
            "answer": response,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"处理问题时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 获取论文列表端点
@app.get("/papers")
async def get_papers(
    category: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    query: Optional[str] = None
):
    try:
        # 计算偏移量
        offset = (page - 1) * limit
        
        # 查询论文
        if query:
            # 搜索论文
            papers = kg_builder.search_papers(query, category, limit, offset)
            total = kg_builder.count_papers(query, category)
        else:
            # 获取所有论文
            papers = kg_builder.get_all_papers(category, limit, offset)
            total = kg_builder.count_papers(None, category)
        
        return {
            "papers": papers,
            "total": total,
            "page": page,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"获取论文列表时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 获取论文分类端点
@app.get("/categories")
async def get_categories():
    try:
        # 获取所有分类
        categories = Config.PAPER_CATEGORIES
        return categories
    except Exception as e:
        logger.error(f"获取论文分类时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 获取论文详情端点
@app.get("/paper/{paper_id}")
async def get_paper_details(paper_id: str):
    try:
        # 查询论文
        paper = kg_builder.get_paper_details(paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="论文不存在")
        
        return paper
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取论文详情时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 删除论文端点
@app.delete("/paper/{paper_id}")
async def delete_paper(paper_id: str):
    try:
        # 删除论文
        success = kg_builder.delete_paper(paper_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="论文不存在")
        
        # 删除向量存储中的论文
        vector_store.delete_by_paper_id(paper_id)
        
        return {"status": "success", "message": "论文已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除论文时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 上传论文端点
@app.post("/upload_paper")
async def upload_paper(file: UploadFile = File(...)):
    try:
        # 检查文件类型
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="仅支持PDF文件")
        
        # 保存临时文件
        temp_file_path = os.path.join(Config.CACHE_DIR, file.filename)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 处理PDF
        paper_data = pdf_parser.process_pdf(temp_file_path)
        
        # 分类论文
        if paper_data["full_text"]:
            classification_result = classifier.classify_text(paper_data["full_text"])
            paper_data["category"] = classification_result["predicted_category"]
            paper_data["category_confidence"] = classification_result["confidence_scores"]
        
        # 提取框架
        framework = extractor.extract_framework(paper_data["full_text"])
        paper_data["framework"] = framework
        
        # 添加到知识图谱
        paper_id = kg_builder.add_paper(paper_data)
        
        # 添加到向量存储
        chunks = paper_data["chunks"]
        vector_store.add_chunks(paper_id, chunks)
        
        # 构建统计信息
        statistics = {
            "chunks": len(chunks),
            "entities": kg_builder.count_paper_entities(paper_id),
            "relations": kg_builder.count_paper_relations(paper_id)
        }
        
        # 返回结果
        result = {
            "paper_id": paper_id,
            "title": paper_data["metadata"]["title"],
            "author": paper_data["metadata"]["author"],
            "category": paper_data["category"],
            "abstract": framework.get("abstract", ""),
            "category_confidence": paper_data["category_confidence"],
            "statistics": statistics
        }
        
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传论文时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 论文推荐端点
@app.post("/recommend")
async def recommend_papers(request: RecommendationRequest):
    try:
        recommendations = get_paper_recommendations(
            request.question, 
            request.paper_id, 
            request.max_recommendations
        )
        
        return recommendations
    except Exception as e:
        logger.error(f"推荐论文时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 知识图谱统计端点
@app.get("/kg/stats")
async def get_kg_stats():
    try:
        stats = kg_querier.query_entity_statistics()
        return stats
    except Exception as e:
        logger.error(f"获取知识图谱统计时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 搜索实体端点
@app.get("/kg/entities/search")
async def search_entities(
    keyword: str,
    entity_type: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100)
):
    try:
        entities = kg_querier.search_entities(keyword, entity_type, limit)
        return entities
    except Exception as e:
        logger.error(f"搜索实体时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 获取实体详情端点
@app.get("/kg/entity")
async def get_entity_details(name: str, type: Optional[str] = None):
    try:
        entity = kg_querier.query_entity_details(name, type)
        
        if not entity["properties"]:
            raise HTTPException(status_code=404, detail="实体不存在")
        
        return entity
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实体详情时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 获取知识图谱数据端点
@app.get("/kg/data")
async def get_kg_data(entity_type: Optional[str] = None, limit: int = Query(200, ge=1, le=1000)):
    try:
        # 根据是否提供实体类型来决定查询方式
        if entity_type:
            # 查询特定类型的实体及其关系
            entity_nodes = kg_builder.get_entities_by_type(entity_type, limit)
            
            # 获取所有实体ID
            entity_ids = [node["id"] for node in entity_nodes]
            
            # 查询这些实体之间的关系
            relationships = kg_builder.get_relationships_between_entities(entity_ids)
        else:
            # 获取一个合理大小的子图
            entity_nodes = kg_builder.get_entities(limit)
            entity_ids = [node["id"] for node in entity_nodes]
            relationships = kg_builder.get_relationships_between_entities(entity_ids)
        
        return {
            "nodes": entity_nodes,
            "relationships": relationships
        }
    except Exception as e:
        logger.error(f"获取知识图谱数据时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 辅助函数：获取论文推荐
def get_paper_recommendations(question: Optional[str] = None, paper_id: Optional[str] = None, max_recommendations: int = 5) -> List[Dict[str, Any]]:
    """
    获取论文推荐
    
    Args:
        question: 用户问题
        paper_id: 当前论文ID
        max_recommendations: 最大推荐数量
        
    Returns:
        推荐论文列表
    """
    recommendations = []
    
    try:
        if question and paper_id:
            # 基于问题和当前论文的推荐
            results = vector_store.search_by_paper_id(question, paper_id, top_k=max_recommendations)
            
            # 提取论文ID
            paper_ids = set()
            for result in results:
                if result["paper_id"] != paper_id:
                    paper_ids.add(result["paper_id"])
                
                if len(paper_ids) >= max_recommendations:
                    break
            
            # 查询论文详情
            for paper_id in paper_ids:
                paper = kg_builder.get_paper_basic_info(paper_id)
                if paper:
                    recommendations.append(paper)
        
        elif question:
            # 仅基于问题的推荐
            results = vector_store.search(question, top_k=max_recommendations * 2)
            
            # 提取论文ID（去重）
            paper_ids = set()
            for result in results:
                paper_ids.add(result["paper_id"])
                
                if len(paper_ids) >= max_recommendations:
                    break
            
            # 查询论文详情
            for paper_id in paper_ids:
                paper = kg_builder.get_paper_basic_info(paper_id)
                if paper:
                    recommendations.append(paper)
        
        elif paper_id:
            # 仅基于当前论文的推荐
            similar_papers = kg_builder.find_similar_papers(paper_id, max_count=max_recommendations)
            recommendations = similar_papers
        
        else:
            # 无条件推荐：返回最新的论文
            recommendations = kg_builder.get_latest_papers(max_recommendations)
    
    except Exception as e:
        logger.error(f"获取论文推荐时出错: {e}")
        # 出错时返回空列表
    
    return recommendations[:max_recommendations]

# 主函数
def start_api_server():
    """启动API服务器"""
    host = Config.API_CONFIG["host"]
    port = Config.API_CONFIG["port"]
    
    logger.info(f"启动API服务器: {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api_server()

