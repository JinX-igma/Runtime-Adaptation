#!/usr/bin/env bash
set -euo pipefail

# 脚本所在目录 = 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="wesad"

SRC_DIR="$SCRIPT_DIR/src"
LOG_DIR="$SCRIPT_DIR/logs"

# 确保目录存在
mkdir -p "$SRC_DIR" "$LOG_DIR"

# echo "[INFO] Project dir: $SCRIPT_DIR"
# echo "[INFO] Src dir     : $SRC_DIR"
# echo "[INFO] Log dir     : $LOG_DIR"
# echo "[INFO] Image       : $IMAGE_NAME"

# 进入容器，同时挂载 src 和 logs
docker run -it --rm \
  -v "$SRC_DIR":/workspace/src \
  -v "$LOG_DIR":/workspace/logs \
  "$IMAGE_NAME" \
  /bin/bash