# config.py  

import os  

# === 路径配置 ===  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(BASE_DIR, 'data')  
MODEL_DIR = os.path.join(BASE_DIR, 'model')  

# === 模型配置 ===  
LLAMA_MODEL_PATH = os.path.join(MODEL_DIR, 'llama3-8b')  # 请将模型路径更新为实际路径  
BERT_MODEL_PATH = 'bert-base-uncased'  # 或根据实际情况更改  

# === 超参数 ===  
NUM_EPOCHS = 10  
BATCH_SIZE = 16  
LEARNING_RATE = 2e-5  
MAX_SEQ_LENGTH = 512  

# === 分类类别 ===  
NUM_CLASSES = 12  
CATEGORY_NAMES = [  
    "Attention & Model Architecture",  
    "Benchmarks",  
    "BERT",  
    "Chain-of-Thought",  
    "Fine-Tuning",  
    "Long-Context",  
    "LoRA",  
    "Instruction & Prompt-Tuning",  
    "RAG",  
    "RL",  
    "RLHF",  
    "Reasoning"  
]  

# === 其他配置 ===  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'