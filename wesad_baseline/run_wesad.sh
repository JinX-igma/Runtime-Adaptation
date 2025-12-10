#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Resolve project root (script location)
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="wesad"

SRC_DIR="$SCRIPT_DIR/src"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$SRC_DIR" "$LOG_DIR"

# ------------------------------------------------------------
# Search for WESAD dataset automatically in parent directories
# ------------------------------------------------------------
search_paths=(
  "$SCRIPT_DIR/WESAD"
  "$SCRIPT_DIR/../WESAD"
  "$SCRIPT_DIR/../../WESAD"
  "$SCRIPT_DIR/../../../WESAD"
)

DATA_DIR=""

for p in "${search_paths[@]}"; do
  if [[ -d "$p" ]]; then
    DATA_DIR="$(cd "$p" && pwd)"
    break
  fi
done

if [[ -z "$DATA_DIR" ]]; then
  echo "--------------------------------------------------"
  echo "[ERROR] Could not locate WESAD dataset."
  echo "Searched paths:"
  for p in "${search_paths[@]}"; do
    echo "  $p"
  done
  echo "--------------------------------------------------"
  echo "Please move WESAD folder near project, e.g.:"
  echo "  ../WESAD"
  echo "or edit this script to specify dataset path manually."
  echo "--------------------------------------------------"
  exit 1
fi

echo "[INFO] Project dir : $SCRIPT_DIR"
echo "[INFO] Src dir     : $SRC_DIR"
echo "[INFO] Log dir     : $LOG_DIR"
echo "[INFO] Data dir    : $DATA_DIR"
echo "[INFO] Image       : $IMAGE_NAME"

# ------------------------------------------------------------
# Run container with mounted dirs
# ------------------------------------------------------------
docker run -it --rm \
  -v "$SRC_DIR":/workspace/src \
  -v "$LOG_DIR":/workspace/logs \
  -v "$DATA_DIR":/workspace/data/WESAD \
  "$IMAGE_NAME" \
  /bin/bash