
# 使用NVIDIA CUDA基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 避免安装过程中的交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装Python和必要的依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential libpq-dev \
    wget curl git \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# 安装Neo4j (确保添加适当的仓库)
RUN curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - \
    && echo 'deb https://debian.neo4j.com stable latest' | tee /etc/apt/sources.list.d/neo4j.list \
    && apt-get update \
    && apt-get install -y neo4j=1:4.4.18 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装项目依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 下载必要的模型和数据
RUN mkdir -p /root/autodl-tmp/models/ \
    && python3 -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', cache_dir='/root/autodl-tmp/models/Qwen2.5-1.5B'); \
    AutoModel.from_pretrained('Qwen/Qwen2.5-1.5B', cache_dir='/root/autodl-tmp/models/Qwen2.5-1.5B')"

# 配置Neo4j
RUN echo 'dbms.security.auth.enabled=true' >> /etc/neo4j/neo4j.conf \
    && echo 'dbms.security.auth.username=neo4j' >> /etc/neo4j/neo4j.conf \
    && echo 'dbms.security.auth.password=password' >> /etc/neo4j/neo4j.conf \
    && echo 'dbms.default_database=academic_papers' >> /etc/neo4j/neo4j.conf

# 创建启动脚本
RUN echo '#!/bin/bash\n\
service neo4j start\n\
milvus_server &\n\
cd /app\n\
python3 main.py\n' > /app/start.sh \
    && chmod +x /app/start.sh

# 暴露API端口
EXPOSE 8000

# 启动服务
CMD ["/app/start.sh"]

