
# src/utils/monitor.py
import os
import logging
import time
import json
import threading
import psutil
import platform
import socket
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import requests
from src.configs.config import Config

# 配置日志
logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    系统监控模块，负责收集系统运行状态和性能数据
    """
    
    def __init__(self, interval: int = 60):
        """
        初始化监控模块
        
        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        self.stats_file = os.path.join(Config.LOG_DIR, "system_stats.json")
        self.api_url = f"http://{Config.API_CONFIG['host']}:{Config.API_CONFIG['port']}/health"
        
        # 确保日志目录存在
        os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    def start(self):
        """
        启动监控
        """
        if self.running:
            logger.warning("监控已在运行中")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"系统监控已启动，间隔：{self.interval}秒")
    
    def stop(self):
        """
        停止监控
        """
        if not self.running:
            logger.warning("监控未运行")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """
        监控循环
        """
        while self.running:
            try:
                # 收集系统状态
                stats = self._collect_stats()
                
                # 保存状态
                self._save_stats(stats)
                
                # 检查异常
                self._check_anomalies(stats)
                
            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
            
            # 等待下一次监控
            time.sleep(self.interval)
    
    def _collect_stats(self) -> Dict[str, Any]:
        """
        收集系统状态
        
        Returns:
            系统状态数据
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "system": self._get_system_info(),
            "resources": self._get_resource_usage(),
            "api": self._check_api_health(),
            "database": self._check_database_health(),
            "vector_store": self._check_vector_store_health()
        }
        
        return stats
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息
        """
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "cpu_count": psutil.cpu_count(),
            "uptime": time.time() - psutil.boot_time()
        }
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """
        获取资源使用情况
        
        Returns:
            资源使用数据
        """
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_used_percent = memory.percent
        memory_used = memory.used
        memory_total = memory.total
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_used_percent = disk.percent
        disk_used = disk.used
        disk_total = disk.total
        
        # 获取当前进程的资源使用
        process = psutil.Process(os.getpid())
        process_cpu_percent = process.cpu_percent(interval=1)
        process_memory = process.memory_info().rss
        
        return {
            "cpu": {
                "percent": cpu_percent
            },
            "memory": {
                "percent": memory_used_percent,
                "used": memory_used,
                "total": memory_total
            },
            "disk": {
                "percent": disk_used_percent,
                "used": disk_used,
                "total": disk_total
            },
            "process": {
                "cpu_percent": process_cpu_percent,
                "memory": process_memory
            }
        }
    
    def _check_api_health(self) -> Dict[str, Any]:
        """
        检查API健康状态
        
        Returns:
            API状态
        """
        result = {
            "status": "unknown",
            "response_time": 0,
            "error": None
        }
        
        try:
            start_time = time.time()
            response = requests.get(self.api_url, timeout=5)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result["status"] = "healthy"
                result["response_time"] = elapsed
                result["response"] = response.json()
            else:
                result["status"] = "unhealthy"
                result["error"] = f"状态码: {response.status_code}"
        except requests.RequestException as e:
            result["status"] = "unhealthy"
            result["error"] = str(e)
        
        return result
    
    def _check_database_health(self) -> Dict[str, Any]:
        """
        检查数据库健康状态
        
        Returns:
            数据库状态
        """
        # 这里简单返回一个占位符
        # 实际实现应连接Neo4j并检查状态
        return {
            "status": "unknown"
        }
    
    def _check_vector_store_health(self) -> Dict[str, Any]:
        """
        检查向量存储健康状态
        
        Returns:
            向量存储状态
        """
        # 这里简单返回一个占位符
        # 实际实现应连接Milvus并检查状态
        return {
            "status": "unknown"
        }
    
    def _save_stats(self, stats: Dict[str, Any]):
        """
        保存状态数据
        
        Args:
            stats: 状态数据
        """
        try:
            # 读取现有数据
            existing_stats = []
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    existing_stats = json.load(f)
            
            # 限制历史记录数量，保留最近100条
            if len(existing_stats) >= 100:
                existing_stats = existing_stats[-99:]
            
            # 添加新数据
            existing_stats.append(stats)
            
            # 保存数据
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(existing_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存状态数据时出错: {e}")
    
    def _check_anomalies(self, stats: Dict[str, Any]):
        """
        检查异常情况
        
        Args:
            stats: 状态数据
        """
        # 检查CPU使用率是否过高
        cpu_percent = stats["resources"]["cpu"]["percent"]
        if cpu_percent > 90:
            logger.warning(f"CPU使用率过高: {cpu_percent}%")
        
        # 检查内存使用率是否过高
        memory_percent = stats["resources"]["memory"]["percent"]
        if memory_percent > 90:
            logger.warning(f"内存使用率过高: {memory_percent}%")
        
        # 检查磁盘使用率是否过高
        disk_percent = stats["resources"]["disk"]["percent"]
        if disk_percent > 90:
            logger.warning(f"磁盘使用率过高: {disk_percent}%")
        
        # 检查API状态
        if stats["api"]["status"] != "healthy":
            logger.error(f"API不健康: {stats['api'].get('error', '未知错误')}")
    
    def get_latest_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的状态数据
        
        Returns:
            最新的状态数据，如果没有则返回None
        """
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    if stats:
                        return stats[-1]
            return None
        except Exception as e:
            logger.error(f"获取最新状态数据时出错: {e}")
            return None
    
    def get_stats_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取状态历史数据
        
        Args:
            limit: 最大返回记录数
            
        Returns:
            状态历史数据
        """
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    return stats[-limit:] if stats else []
            return []
        except Exception as e:
            logger.error(f"获取状态历史数据时出错: {e}")
            return []

# 单例实例
monitor = SystemMonitor()

def start_monitoring():
    """
    启动系统监控
    """
    monitor.start()

def stop_monitoring():
    """
    停止系统监控
    """
    monitor.stop()

def get_latest_stats() -> Optional[Dict[str, Any]]:
    """
    获取最新的状态数据
    
    Returns:
        最新的状态数据
    """
    return monitor.get_latest_stats()

def get_stats_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    获取状态历史数据
    
    Args:
        limit: 最大返回记录数
        
    Returns:
        状态历史数据
    """
    return monitor.get_stats_history(limit)

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 启动监控
    start_monitoring()
    
    try:
        # 保持程序运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 停止监控
        stop_monitoring()

