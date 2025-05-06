
# main.py
import os
import sys
import time
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



import argparse
from pathlib import Path
from multiprocessing import Process

# 添加项目根目录到路径，以便导入模块
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.api import start_api_server
from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder
from src.knowledge_base.vector_store import VectorStore
from src.utils.monitor import start_monitoring, stop_monitoring
from src.configs.config import Config
from src.utils.pdf_parser import PDFParser


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


"""
学术论文对话系统主入口脚本

使用方法：
    python main.py [command] [options]
"""

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
        
        





def setup_logging(verbose=False):
    """
    设置日志
    
    Args:
        verbose: 是否启用详细日志
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # 确保日志目录存在
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.LOG_DIR, "paperchat.log")),
            logging.StreamHandler()
        ]
    )
    
    # 设置第三方库的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)


def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description="学术论文对话系统")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 服务器命令
    server_parser = subparsers.add_parser("server", help="启动API服务器")
    server_parser.add_argument("-h", "--host", help="主机地址", default=Config.API_CONFIG["host"])
    server_parser.add_argument("-p", "--port", type=int, help="端口号", default=Config.API_CONFIG["port"])
    server_parser.add_argument("-m", "--monitor", action="store_true", help="启用系统监控")
    
    # 客户端命令
    client_parser = subparsers.add_parser("client", help="启动Web客户端")
    client_parser.add_argument("-p", "--port", type=int, help="Web服务器端口号", default=8080)
    
    # 初始化命令
    init_parser = subparsers.add_parser("init", help="初始化系统")
    init_parser.add_argument("-d", "--db", action="store_true", help="初始化数据库")
    init_parser.add_argument("-v", "--vector", action="store_true", help="初始化向量存储")
    init_parser.add_argument("--all", action="store_true", help="初始化所有组件")
    
    # 导入论文命令
    import_parser = subparsers.add_parser("import", help="导入论文")
    import_parser.add_argument("path", help="PDF文件或目录路径")
    import_parser.add_argument("-r", "--recursive", action="store_true", help="递归处理目录")
    
    # 知识图谱命令
    kg_parser = subparsers.add_parser("kg", help="知识图谱操作")
    kg_parser.add_argument("action", choices=["build", "clear", "stats"], help="知识图谱操作类型")
    kg_parser.add_argument("-f", "--force", action="store_true", help="强制执行操作")
    
    # 全局参数
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细日志")
    parser.add_argument("--version", action="store_true", help="显示版本信息")
    
    return parser.parse_args()


def start_server(host, port, enable_monitor=False):
    """
    启动API服务器
    
    Args:
        host: 主机地址
        port: 端口号
        enable_monitor: 是否启用监控
    """
    logging.info(f"正在启动API服务器: {host}:{port}")
    
    # 更新配置
    Config.API_CONFIG["host"] = host
    Config.API_CONFIG["port"] = port
    
    # 启动监控
    if enable_monitor:
        logging.info("启用系统监控")
        start_monitoring()
    
    try:
        # 启动服务器
        start_api_server()
    except KeyboardInterrupt:
        logging.info("收到中断信号，正在停止服务器")
        
        # 停止监控
        if enable_monitor:
            stop_monitoring()
    except Exception as e:
        logging.error(f"服务器启动失败: {e}")
        
        # 停止监控
        if enable_monitor:
            stop_monitoring()
        
        sys.exit(1)


def start_client(port):
    """
    启动Web客户端
    
    Args:
        port: Web服务器端口号
    """
    import http.server
    import socketserver
    
    # 前端目录
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    
    # 检查前端目录是否存在
    if not os.path.exists(frontend_dir):
        logging.error(f"前端目录不存在: {frontend_dir}")
        sys.exit(1)
    
    # 修改工作目录
    os.chdir(frontend_dir)
    
    # 创建HTTP服务器
    handler = http.server.SimpleHTTPRequestHandler
    
    logging.info(f"正在启动Web客户端: http://localhost:{port}")
    
    try:
        # 启动服务器
        with socketserver.TCPServer(("", port), handler) as httpd:
            logging.info(f"Web服务器已启动，访问 http://localhost:{port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("收到中断信号，正在停止Web服务器")
    except Exception as e:
        logging.error(f"Web服务器启动失败: {e}")
        sys.exit(1)


def init_system(init_db=False, init_vector=False, init_all=False):
    """
    初始化系统
    
    Args:
        init_db: 是否初始化数据库
        init_vector: 是否初始化向量存储
        init_all: 是否初始化所有组件
    """
    if init_all:
        init_db = init_vector = True
    
    if not (init_db or init_vector):
        logging.error("请至少指定一个组件进行初始化")
        sys.exit(1)
    
    # 初始化数据库
    if init_db:
        logging.info("正在初始化Neo4j数据库")
        kg_builder = KnowledgeGraphBuilder()
        kg_builder.init_database()
        logging.info("Neo4j数据库初始化完成")
    
    # 初始化向量存储
    if init_vector:
        logging.info("正在初始化向量存储")
        vector_store = VectorStore()
        vector_store.init_collection()
        vector_store.close()
        logging.info("向量存储初始化完成")
    
    logging.info("系统初始化完成")


def import_papers(path, recursive=False):
    """
    导入论文
    
    Args:
        path: PDF文件或目录路径
        recursive: 是否递归处理目录
    """
    # 检查路径是否存在
    if not os.path.exists(path):
        logging.error(f"路径不存在: {path}")
        sys.exit(1)
    
    # 初始化组件
    pdf_parser = PDFParser()
    kg_builder = KnowledgeGraphBuilder()
    vector_store = VectorStore()
    
    # 导入论文
    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        # 导入单个文件
        _import_single_paper(path, pdf_parser, kg_builder, vector_store)
    elif os.path.isdir(path):
        # 导入目录
        _import_directory(path, recursive, pdf_parser, kg_builder, vector_store)
    else:
        logging.error(f"不支持的文件类型: {path}")
        sys.exit(1)
    
    # 关闭连接
    vector_store.close()
    
    logging.info("论文导入完成")


def _import_single_paper(file_path, pdf_parser, kg_builder, vector_store):
    """
    导入单个论文
    
    Args:
        file_path: PDF文件路径
        pdf_parser: PDF解析器
        kg_builder: 知识图谱构建器
        vector_store: 向量存储
    """
    try:
        logging.info(f"正在处理: {file_path}")
        
        # 解析PDF
        paper_data = pdf_parser.process_pdf(file_path)
        
        # 添加到知识图谱
        paper_id = kg_builder.add_paper(paper_data)
        
        # 添加到向量存储
        chunks = paper_data["chunks"]
        vector_store.add_chunks(paper_id, chunks)
        
        logging.info(f"处理完成: {file_path}")
        
        return True
    except Exception as e:
        logging.error(f"处理失败 {file_path}: {e}")
        return False


def _import_directory(dir_path, recursive, pdf_parser, kg_builder, vector_store):
    """
    导入目录中的论文
    
    Args:
        dir_path: 目录路径
        recursive: 是否递归处理子目录
        pdf_parser: PDF解析器
        kg_builder: 知识图谱构建器
        vector_store: 向量存储
    """
    success_count = 0
    fail_count = 0
    
    # 遍历目录
    for root, dirs, files in os.walk(dir_path):
        # 处理当前目录中的PDF文件
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                if _import_single_paper(file_path, pdf_parser, kg_builder, vector_store):
                    success_count += 1
                else:
                    fail_count += 1
        
        # 如果不递归，则中断
        if not recursive:
            break
    
    logging.info(f"导入统计: 成功 {success_count}, 失败 {fail_count}")


def manage_knowledge_graph(action, force=False):
    """
    管理知识图谱
    
    Args:
        action: 操作类型
        force: 是否强制执行
    """
    kg_builder = KnowledgeGraphBuilder()
    
    if action == "build":
        logging.info("正在构建知识图谱")
        kg_builder.build_knowledge_graph()
        logging.info("知识图谱构建完成")
    
    elif action == "clear":
        if not force:
            logging.warning("清空知识图谱是危险操作，请使用 --force 参数确认")
            sys.exit(1)
        
        logging.info("正在清空知识图谱")
        kg_builder.clear_database()
        logging.info("知识图谱已清空")
    
    elif action == "stats":
        logging.info("正在获取知识图谱统计信息")
        stats = kg_builder.get_statistics()
        
        # 打印统计信息
        print("\n知识图谱统计信息:")
        print(f"论文数量: {stats.get('paper_count', 0)}")
        print(f"实体数量: {stats.get('entity_count', 0)}")
        print(f"关系数量: {stats.get('relation_count', 0)}")
        print(f"三元组数量: {stats.get('triple_count', 0)}")
        
        # 打印实体类型分布
        if "entity_types" in stats:
            print("\n实体类型分布:")
            for type_name, count in stats["entity_types"].items():
                print(f"  {type_name}: {count}")
        
        # 打印关系类型分布
        if "relation_types" in stats:
            print("\n关系类型分布:")
            for type_name, count in stats["relation_types"].items():
                print(f"  {type_name}: {count}")


def show_version():
    """
    显示版本信息
    """
    print("学术论文对话系统 v1.0.0")
    print("基于Qwen2.5和知识图谱的AI驱动学术论文分析与对话")
    print("Copyright (c) 2023")


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 显示版本信息
    if args.version:
        show_version()
        return
    
    # 根据命令执行相应操作
    if args.command == "server":
        start_server(args.host, args.port, args.monitor)
    
    elif args.command == "client":
        start_client(args.port)
    
    elif args.command == "init":
        init_system(args.db, args.vector, args.all)
    
    elif args.command == "import":
        import_papers(args.path, args.recursive)
    
    elif args.command == "kg":
        manage_knowledge_graph(args.action, args.force)
    
    else:
        # 显示帮助信息
        print(__doc__)
        print("使用 -h 或 --help 查看详细帮助")


if __name__ == "__main__":
    main()
        
        

# if __name__ == "__main__":
#     host = Config.API_CONFIG["host"]
#     port = int(Config.API_CONFIG["port"])
#     debug = Config.API_CONFIG["debug"]
    
#     uvicorn.run("main:app", host=host, port=port, reload=debug)

