#!/bin/bash
# chmod +x infer_obb_track.sh
set -e  # 遇到错误立即退出

# 日志目录（自动创建）
LOGDIR="./logs_obb_track"
OUTPUT_DIR="./results"
mkdir -p "$LOGDIR"
mkdir -p "$OUTPUT_DIR"

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
# ./yolo_obb_track detect \
#     --model ../model/YOLO11s_obb_video_base_640.om \
#     --input  /home/HwHiAiUser/gp/DATASETS/test0909/imgs_640/000001.jpg \
#     --image_out  ../results/imgs \
#     --label_out  ../results/txts

# ./yolo_obb_track detect \
#     --model ../model/YOLO11s_obb_video_base_640.om \
#     --input  /home/HwHiAiUser/gp/DATASETS/test0909/imgs_640 \
#     --image_out  ../results/imgs \
#     --label_out  ../results/txts

./yolo_obb_track track \
    --model ../model/YOLO11s_obb_video_base_640.om \
    --input  /home/HwHiAiUser/gp/DATASETS/test0909/imgs_640 \
    --image_out  ../results/imgs \
    --label_out  ../results/txts \
    --track_image_out ../results/imgs_track 

echo "[INFO] 推理完成."