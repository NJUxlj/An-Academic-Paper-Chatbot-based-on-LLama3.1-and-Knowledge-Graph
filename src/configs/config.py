# config.py  

import os  
import torch
from pathlib import Path
from dotenv import load_dotenv
# === 路径配置 ===  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(BASE_DIR, 'data')  
MODEL_DIR = os.path.join(BASE_DIR, 'model')  

# === 模型配置 ===  
# LLAMA_MODEL_PATH = os.path.join(MODEL_DIR, 'llama3-8b')  # 请将模型路径更新为实际路径  
# LLAMA_TOKENIZER_PATH = os.path.join(MODEL_DIR, 'llama3-8b-tokenizer')
LLAMA_ADAPTER_PATH = os.path.join(MODEL_DIR, 'llama3-8b-adapter')
LLAMA_TRAINED_PATH = os.path.join(MODEL_DIR, 'llama3-8b-trained')
# BERT_MODEL_PATH = 'bert-base-uncased'  # 或根据实际情况更改  



load_dotenv()

class Config:
    # === 其他配置 ===  
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # 项目根目录
    ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
    
    LOG_DIR = ROOT_DIR / "logs"
    
    QWEN_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
    
    BERT_MODEL_PATH = "/root/autodl-tmp/models/bert-base-chinese"
    
    # 模型配置
    MODEL_CONFIG = {
        "qwen": {
            "model_path": "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
            "device": "cuda",
            "max_length": 2048,
            "temperature": 0.7,
        },
        "glm": {
            "api_key": os.getenv("ZHIPU_API_KEY", ""),
            "model": "glm-4-flash",  # 智谱API的GLM-4模型
        },
        "bert_ner": {
            "model_path": "/root/autodl-tmp/models/bert-base-chinese",
            "max_length": 512,
            "batch_size": 16,
        },
        "deepseek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "model": "deepseek-chat",  # DeepSeek API的模型
            "max_length": 2048,
            "base_url": os.getenv("BASE_URL", "")
        }
    }
    
    # 知识图谱配置
    NEO4J_CONFIG = {
        "uri": "neo4j://localhost:7687",
        "auth": ("neo4j", os.getenv("NEO4J_PASSWORD", "password")),
        "database": "academic_papers",
    }
    
    # 向量数据库配置
    VECTOR_DB_CONFIG = {
        "host": "localhost",
        "port": "19530",
        "collection": "paper_chunks",
        "dim": 768,  # 向量维度
    }
    
    # API配置
    API_CONFIG = {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
    }
    
    # 数据目录
    DATA_DIR = ROOT_DIR / "src" / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "src" / "processed"
    RAW_DATA_DIR = DATA_DIR / "src" / "raw"
    
    # 缓存目录
    CACHE_DIR = DATA_DIR / "cache"
    
    # 论文分类类别
    NUM_CLASSES = 12  
    
    PAPER_CATEGORIES = [
        "Attention & Model Architecture", 
        "Benchmarks", 
        "BERT", 
        "Chain-of-Thought", 
        "Fine-Tuning", 
        "Long-Context", 
        "LoRA", 
        "Instruction&Prompt-Tuning", 
        "RAG", 
        "RL", 
        "RLHF", 
        "Reasoning"
    ]
    
    # === 超参数 ===  
    NUM_EPOCHS = 10  
    BATCH_SIZE = 16  
    LEARNING_RATE = 2e-5  
    MAX_SEQ_LENGTH = 512  
    
    # 创建必要的目录
    @classmethod
    def create_directories(cls):
        directories = [
            cls.DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.CACHE_DIR,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# 初始化创建目录
Config.create_directories()

