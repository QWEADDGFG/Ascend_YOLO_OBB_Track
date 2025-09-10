# YOLO_OBB_Track
YOLO_OBB_Track

## 数据集路径
/home/HwHiAiUser/gp/DATASETS/test0909/imgs_640

./yolo_obb_track detect \
    --model ../model/YOLO11s_obb_video_base_640.om \
    --input  /home/HwHiAiUser/gp/DATASETS/test0909/imgs_640/000001.jpg \
    --image_out  ../results/imgs \
    --label_out  ../results/txts