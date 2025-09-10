#!/bin/bash
# chmod +x infer_obb_track.sh
set -e  # 遇到错误立即退出

# 日志目录（自动创建）
LOGDIR="./logs_obb_track"
OUTPUT_DIR_OBB="./output_obb_track"
mkdir -p "$LOGDIR"
mkdir -p "$OUTPUT_DIR_OBB"

# 日志文件名（自动加日期）
LOGFILE="$LOGDIR/run_$(date +'%Y-%m-%d_%H-%M-%S').log"

echo "======================================" > "$LOGFILE"
echo " Run started at $(date)" >> "$LOGFILE"
echo "======================================" >> "$LOGFILE"

# 只写入日志文件，不在终端显示
exec >> "$LOGFILE" 2>&1

# 推理阶段
echo "[INFO] 开始推理..."
cd ./build
./yolo_obb_track /home/HwHiAiUser/gp/DATASETS/test0909/imgs_640
echo "[INFO] 推理完成."