# YOLO_OBB_Track
YOLO_OBB_Track

## 数据集
[路径：]/home/HwHiAiUser/gp/DATASETS/test0909/imgs_640
[预处理：]/home/HwHiAiUser/gp/Ascend_YOLO_OBB_Track/resize.py
图像预处理工具，其核心功能是将指定源目录 (SRC_DIR) 中的所有支持的图像文件（如JPG, PNG等）进行批量处理。
对于每张图像，它会先将其转换为RGB格式，然后等比例缩放，使其最长边不超过目标尺寸（默认为640x640）。
接着，它会将缩放后的图像居中放置在一个新的640x640的画布上，画布的空白区域则用YOLO模型常用的中性灰边 (114, 114, 114) 进行填充。
最终，处理后的图像将以JPEG格式（质量95）保存到指定的输出目录 (DST_DIR)。

目的：将指定目录中的图像批量处理为YOLO_OBB_Track模型所需的输入格式（640*640尺寸与Baseline JPEG格式）。
某些库（如老旧 OpenCV、嵌入式系统）不支持Progressive JPEG 。

## 编译：
```bash
mkdir build
./run_build.sh
```
```shell
[INFO] 开始编译源代码...
-- The C compiler identification is GNU 11.3.0
-- The CXX compiler identification is GNU 11.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found OpenCV: /usr/local (found version "4.5.4") 
-- set INC_PATH: /usr/local/Ascend/ascend-toolkit/latest
-- set LIB_PATH: /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub
-- set THIRDPART: /usr/local/Ascend/ascend-toolkit/latest/thirdpart
-- Configuring done
-- Generating done
-- Build files have been written to: /home/HwHiAiUser/gp/Ascend_YOLO_OBB_Track/build
[ 11%] Building CXX object CMakeFiles/yolo_obb_track.dir/main_all.cpp.o
[ 22%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/BYTETracker.cpp.o
[ 33%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/STrack.cpp.o
[ 44%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/kalmanFilter.cpp.o
[ 55%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/lapjv.cpp.o
[ 66%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/utils.cpp.o
[ 77%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/yolo_obb.cpp.o
[ 88%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/yolov8.cpp.o
[100%] Linking CXX executable yolo_obb_track
[100%] Built target yolo_obb_track
[INFO] 编译完成.
```


## 运行推理：

1. 
```bash
./infer_obb_track.sh 
```

2. 
```bash

cd build
# 单张图片检测
./yolo_obb_track detect \
    --model ../model/YOLO11s_obb_video_base_640.om \
    --input  /home/HwHiAiUser/gp/DATASETS/test0909/imgs_640/000002.jpg \
    --image_out  ../results/imgs_01 \
    --label_out  ../results/txts_01

# 批量图片检测
./yolo_obb_track detect \
    --model ../model/YOLO11s_obb_video_base_640.om \
    --input  /home/HwHiAiUser/gp/DATASETS/test0909/imgs_640 \
    --image_out  ../results/imgs \
    --label_out  ../results/txts

# 批量图片检测+跟踪
./yolo_obb_track track \
    --model ../model/YOLO11s_obb_video_base_640.om \
    --input  /home/HwHiAiUser/gp/DATASETS/test0909/imgs_640 \
    --image_out  ../results/imgs \
    --label_out  ../results/txts \
    --track_image_out ../results/imgs_track 

```
