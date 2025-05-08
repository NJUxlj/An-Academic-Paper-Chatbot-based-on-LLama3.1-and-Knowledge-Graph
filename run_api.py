# run_api.py
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent))

from src.api.api import start_api_server
from src.configs.config import Config

if __name__ == "__main__":
    # python  run_api.py
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.LOG_DIR, "api.log")),
            logging.StreamHandler()
        ]
    )
    
    # 启动API服务器
    start_api_server()
