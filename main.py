
# main.py
import os
import logging
import tempfile
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch

from src.configs.config import Config
from src.utils.pdf_parser import PDFParser
from src.models.paper_framework.extractor import PaperFrameworkExtractor
from src.models.classifier.qwen_classifier import PaperFrameworkClassifier
from src.models.entity_extractor.bert_bilstm_crf import EntityTripleExtractor
from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder
from src.knowledge_base.vector_store import VectorStore
from src.dialogue_system.dialog_manager import DialogManager
from src.recommendation.paper_recommender import PaperRecommender

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Academic Paper Chatbot API", 
              description="基于知识图谱和大模型的学术论文对话系统")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型和组件
pdf_parser = PDFParser()
framework_extractor = PaperFrameworkExtractor()
classifier = PaperFrameworkClassifier()
entity_extractor = EntityTripleExtractor(
    model_path=Config.MODEL_CONFIG["bert_ner"]["model_path"],
    device=Config.MODEL_CONFIG["qwen"]["device"]
)
kg_builder = KnowledgeGraphBuilder()
vector_store = VectorStore()
dialog_manager = DialogManager()
paper_recommender = PaperRecommender()

# 定义API模型
class QuestionRequest(BaseModel):
    question: str
    paper_id: Optional[str] = None

class RecommendationRequest(BaseModel):
    paper_id: Optional[str] = None
    question: Optional[str] = None
    max_recommendations: int = 5

@app.get("/")
async def root():
    """API根路径，返回基本信息"""
    return {
        "status": "ok",
        "message": "学术论文对话系统API服务已启动",
        "info": "基于知识图谱和大模型的对话系统",
        "version": "1.0.0"
    }

@app.post("/upload_paper")
async def upload_paper(file: UploadFile = File(...)):
    """
    上传论文PDF文件
    
    Args:
        file: PDF文件
    
    Returns:
        处理结果
    """
    try:
        # 检查文件类型
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="仅支持PDF文件")
        
        # 保存上传的文件
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # 解析PDF文件
        paper_data = pdf_parser.process_pdf(file_path)
        
        # 提取论文框架
        framework = framework_extractor.extract_framework(paper_data["full_text"])
        
        # 分类论文
        classification = classifier.classify(framework)
        
        # 将处理后的数据添加到原始数据中
        paper_data["framework"] = framework
        paper_data["category"] = classification["predicted_category"]
        paper_data["classification"] = classification
        
        # 构建知识图谱
        kg_stats = kg_builder.build_graph_from_paper(paper_data)
        
        # 将文本块添加到向量存储
        chunks = []
        for section, section_chunks in paper_data["sectioned_chunks"].items():
            for chunk_text in section_chunks:
                chunks.append({
                    "text": chunk_text,
                    "section": section
                })
        
        vector_ids = vector_store.add_chunks(str(kg_stats.get("paper_id", "unknown")), chunks)
        
        # 返回处理结果
        return {
            "status": "success",
            "paper_id": kg_stats.get("paper_id", "unknown"),
            "title": paper_data["metadata"].get("title", "Unknown Title"),
            "author": paper_data["metadata"].get("author", "Unknown Author"),
            "category": classification["predicted_category"],
            "confidence": classification["confidence"],
            "top_categories": classification["top_categories"],
            "framework": framework,
            "num_chunks": len(chunks),
            "kg_stats": kg_stats
        }
    except Exception as e:
        logger.error(f"上传论文时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理论文时出错: {str(e)}")
    finally:
        # 清理临时文件
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    提问问题
    
    Args:
        request: 问题请求
    
    Returns:
        回答
    """
    try:
        question = request.question
        paper_id = request.paper_id
        
        # 处理问题
        answer = dialog_manager.process_query(question, paper_id)
        
        # 推荐相关论文
        recommendations = []
        if paper_id:
            recommendations = paper_recommender.recommend_by_paper(paper_id, 3)
        else:
            recommendations = paper_recommender.recommend_by_query(question, 3)
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "paper_id": paper_id,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"回答问题时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理问题时出错: {str(e)}")

@app.post("/recommend")
async def recommend_papers(request: RecommendationRequest):
    """
    推荐相关论文
    
    Args:
        request: 推荐请求
    
    Returns:
        推荐论文列表
    """
    try:
        paper_id = request.paper_id
        question = request.question
        max_count = request.max_recommendations
        
        recommendations = []
        
        if paper_id:
            # 基于论文ID推荐
            recommendations = paper_recommender.recommend_by_paper(paper_id, max_count)
        elif question:
            # 基于问题推荐
            recommendations = paper_recommender.recommend_by_query(question, max_count)
        else:
            raise HTTPException(status_code=400, detail="必须提供论文ID或问题")
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"推荐论文时出错: {e}")
        raise HTTPException(status_code=500, detail=f"推荐论文时出错: {str(e)}")

@app.get("/categories")
async def get_categories():
    """
    获取所有论文类别
    
    Returns:
        论文类别列表
    """
    return {
        "status": "success",
        "categories": Config.PAPER_CATEGORIES
    }

@app.get("/papers")
async def get_papers(category: Optional[str] = None, limit: int = Query(10, ge=1, le=100)):
    """
    获取论文列表
    
    Args:
        category: 论文类别（可选）
        limit: 最大返回数量
    
    Returns:
        论文列表
    """
    try:
        papers = []
        
        with kg_builder.driver.session() as session:
            if category:
                # 获取特定类别的论文
                result = session.run(
                    """
                    MATCH (p:Paper)
                    WHERE p.category = $category
                    RETURN id(p) as paper_id, p.title as title, p.author as author, 
                           p.category as category, p.filename as filename
                    LIMIT $limit
                    """,
                    category=category,
                    limit=limit
                )
            else:
                # 获取所有论文
                result = session.run(
                    """
                    MATCH (p:Paper)
                    RETURN id(p) as paper_id, p.title as title, p.author as author, 
                           p.category as category, p.filename as filename
                    LIMIT $limit
                    """,
                    limit=limit
                )
            
            for record in result:
                papers.append({
                    "paper_id": record["paper_id"],
                    "title": record["title"],
                    "author": record["author"],
                    "category": record["category"],
                    "filename": record["filename"]
                })
        
        return {
            "status": "success",
            "papers": papers,
            "count": len(papers)
        }
    except Exception as e:
        logger.error(f"获取论文列表时出错: {e}")
        raise HTTPException(status_code=500, detail=f"获取论文列表时出错: {str(e)}")

@app.delete("/paper/{paper_id}")
async def delete_paper(paper_id: str):
    """
    删除论文
    
    Args:
        paper_id: 论文ID
    
    Returns:
        删除结果
    """
    try:
        # 从知识图谱中删除
        with kg_builder.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper)
                WHERE id(p) = $paper_id
                OPTIONAL MATCH (p)-[r]-()
                DELETE r, p
                RETURN count(p) as deleted
                """,
                paper_id=paper_id
            )
            deleted = result.single()["deleted"]
        
        # 从向量存储中删除
        vector_deleted = vector_store.delete_by_paper_id(paper_id)
        
        return {
            "status": "success",
            "paper_id": paper_id,
            "deleted_from_kg": deleted > 0,
            "deleted_from_vector_store": vector_deleted
        }
    except Exception as e:
        logger.error(f"删除论文时出错: {e}")
        raise HTTPException(status_code=500, detail=f"删除论文时出错: {str(e)}")

@app.get("/health")
async def health_check():
    """
    健康检查
    
    Returns:
        系统状态
    """
    gpu_available = torch.cuda.is_available()
    gpu_info = {
        "count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0) if gpu_available else None
    } if gpu_available else {}
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "components": {
            "pdf_parser": "loaded",
            "framework_extractor": "loaded",
            "classifier": "loaded",
            "entity_extractor": "loaded",
            "kg_builder": "loaded",
            "vector_store": "loaded",
            "dialog_manager": "loaded",
            "paper_recommender": "loaded"
        }
    }

@app.on_event("shutdown")
def shutdown_event():
    """关闭资源"""
    try:
        kg_builder.close()
        vector_store.close()
        logger.info("成功关闭资源")
    except Exception as e:
        logger.error(f"关闭资源时出错: {e}")

if __name__ == "__main__":
    host = Config.API_CONFIG["host"]
    port = int(Config.API_CONFIG["port"])
    debug = Config.API_CONFIG["debug"]
    
    uvicorn.run("main:app", host=host, port=port, reload=debug)

