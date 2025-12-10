FROM python:3.6-slim

# 1. 基础工具
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Python 环境配置
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. 科学计算库
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    scikit-learn

# 4. 安装 PyTorch CPU 版
RUN pip install --no-cache-dir \
    torch==1.9.0 \
    torchvision==0.10.0

# 5. 工作目录
WORKDIR /workspace

# 6. 创建空目录
RUN mkdir -p /workspace/logs

# 7. 日志目录声明为可挂载卷
VOLUME ["/workspace/logs"]

# 8. 环境变量
ENV PYTHONPATH=/workspace \
    LOG_DIR=/workspace/logs

# 9. 默认命令
CMD ["/bin/bash"]