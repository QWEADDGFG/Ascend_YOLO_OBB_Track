# YOLO_OBB_Track

项目使用 C++17 和 C11 标准，启用了 DVPP 接口支持，并设置了发布版本的编译优化选项。

该配置文件依赖 Eigen3 数学库和 OpenCV 计算机视觉库，并通过环境变量（DDK_PATH、NPU_HOST_LIB、THIRDPART_PATH）自动检测昇腾开发套件的安装路径，如果环境变量未设置则使用默认路径。项目包含了昇腾运行时库的头文件和链接库路径，最终生成的可执行文件由 main_all.cpp 和 src 目录下的所有 cpp 源文件编译而成，链接了 Eigen、OpenCV 以及昇腾相关的核心库（ascendcl、acl_dvpp、acllite 等），用于在昇腾 NPU 上运行 YOLO 算法进行旋转框目标检测和跟踪任务。

## 1. 环境准备、数据集与模型准备

### 1.1 环境准备
```bash
npu-smi info
+--------------------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc3                                 Version: 23.0.rc3                                     |
+-------------------------------+-----------------+------------------------------------------------------+
| NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 0       310B1                 | OK              | 8.9          54                15    / 15            |
| 0       0                     | NA              | 0            4974 / 11577                            |
+===============================+=================+======================================================+
```
### 1.2 数据集准备
1. [路径：]/home/HwHiAiUser/gp/DATASETS/test0909/imgs_640

2. [预处理：]/home/HwHiAiUser/gp/Ascend_YOLO_OBB_Track/resize.py
图像预处理工具，其核心功能是将指定源目录 (SRC_DIR) 中的所有支持的图像文件（如JPG, PNG等）进行批量处理。
对于每张图像，它会先将其转换为RGB格式，然后等比例缩放，使其最长边不超过目标尺寸（默认为640x640）。
接着，它会将缩放后的图像居中放置在一个新的640x640的画布上，画布的空白区域则用YOLO模型常用的中性灰边 (114, 114, 114) 进行填充。
最终，处理后的图像将以JPEG格式（质量95）保存到指定的输出目录 (DST_DIR)。

3. 目的：将指定目录中的图像批量处理为YOLO_OBB_Track模型所需的输入格式（640*640尺寸与Baseline JPEG格式）。
某些库（如老旧 OpenCV、嵌入式系统）不支持Progressive JPEG 。

### 1.3 模型准备
1. onnx获取

```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = YOLO('/data/gy/gp/Huawei/yolo11obb/runs/train/yolo11sobb_MVRSD2/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=11, dynamic=True, imgsz=640, nms=False)
```

2. onnx转换为om

```shell
atc --model=YOLO11n_p2_hbb_IRSTD_1K_512.onnx --framework=5 --output=YOLO11n_p2_hbb_IRSTD_1K_512 --input_shape="images:1,3,512,512"  --soc_version=Ascend310B1  --insert_op_conf=aipp512.cfg
```
```shell
atc --model=YOLO11s_base_obb_MVRSD_640.onnx --framework=5 --output=YOLO11s_base_obb_MVRSD_640 --input_shape="images:1,3,640,640"  --soc_version=Ascend310B1  --insert_op_conf=aipp640.cfg
```
```shell
atc --model=yolov8n.onnx --framework=5 --output=yolov8n --input_shape="images:1,3,640,640"  --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg
```
```shell
atc --model=test0909.onnx --framework=5 --output=test0909 --input_shape="images:1,3,640,640"  --soc_version=Ascend310B1  --insert_op_conf=aipp640.cfg
```
aipp.cfg做了三件事：
1. YUV420SP → RGB 颜色空间转换；
2. 裁剪成 640×640；
3. 归一化到 [0,1]（通过除以 255）

## 2.编译：

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
## 3.代码

### 3.1 主函数
main_all.cpp
基于YOLO OBB检测和BYTETracker跟踪的C++应用程序。

该程序提供两种工作模式：
1. **纯检测模式**：对单张图片或图片序列进行目标检测
2. **检测+跟踪模式**：在检测基础上增加多目标跟踪功能

#### 1. YoloOBBWrapper类
- 封装了YOLO OBB推理引擎
- 提供`Detect`方法进行目标检测
- 返回OBB格式的检测结果和AABB格式的目标对象
- 支持旋转边界框的检测和可视化

#### 2. 检测流程
- 图像预处理
- 模型推理
- 后处理（NMS非极大值抑制）
- 将OBB转换为AABB格式供跟踪器使用

#### 3. 跟踪功能
- 使用BYTETracker进行多目标跟踪
- 通过IoU匹配将检测结果与跟踪轨迹关联-**(使用OBB的最小外接矩形作为匹配依据)**
- 为每个目标分配唯一的track_id

#### 4. 注意事项

##### 输入内容
- 单张图片文件
- 按6位数字编号的图片序列（如000001.jpg, 000002.jpg等）

##### 输出内容
- **检测结果图**：带有旋转边界框标注的图片
- **标签文件**：包含类别、置信度和OBB四个角点坐标
- **跟踪结果图**（可选）：带有track_id标注的图片

##### 配置参数
支持多种可配置参数：
- 模型路径、输入输出路径
- 模型尺寸（宽高，与输入图像尺寸一致）
- 类别数量、置信度阈值、NMS阈值
- 模型输出框数量

### 3.2 YOLO OBB实现
```c++
#ifndef YOLO_OBB_H
#define YOLO_OBB_H

//#include ***

using namespace std;
using namespace cv;

extern const string label[];

#if __cplusplus < 201402L
namespace std
{
    template <typename T, typename... Args>
    unique_ptr<T> make_unique(Args &&...args)
    {
        return unique_ptr<T>(new T(forward<Args>(args)...));
    }
}
#endif

#ifndef MODEL_ANGLE_MODE
#define MODEL_ANGLE_MODE 0
#endif

struct OBBBoundingBox // 旋转框结构体-包含了OBB中心点坐标、宽高、角度、置信度、类别索引、索引
{
    float cx, cy, width, height;
    float angle;
    float confidence;
    size_t classIndex;
    size_t index;

    vector<cv::Point2f> getCornerPoints() const;
    OBBBoundingBox();
};

class InferenceConfig // 模型推理配置参数
{
public:
    string modelPath;
    string inputDir;
    string outputImgDir;
    string outputTxtDir;
    int32_t modelWidth;
    int32_t modelHeight;
    float confidenceThreshold;
    float nmsThreshold;
    size_t modelOutputBoxNum;
    size_t classNum;

    InferenceConfig();
};

class Utils // 工具类
{
public:
    static bool sortByConfidence(const OBBBoundingBox &a, const OBBBoundingBox &b);
    static void createDirectory(const string &path);
    static vector<string> getImagePaths(const string &dirPath);
    static string getFileNameWithoutExt(const string &path);
    static float normalizeAngle(float angle);
};

class OBBPostProcessor // OBB后处理器
{
private:
    size_t modelOutputBoxNum_;
    size_t classNum_;

public:
    OBBPostProcessor(size_t boxNum, size_t classNum);
    vector<OBBBoundingBox> parseOutput(float *outputData, size_t outputSize,
                                       int srcWidth, int srcHeight,
                                       int modelWidth, int modelHeight,
                                       float confidenceThreshold);
};

class OBBNMSProcessor // OBB NMS 处理器
{
private:
    static std::tuple<float, float, float> getCovarianceMatrix(const OBBBoundingBox &box);

public:
    static float calculateProbIOU(const OBBBoundingBox &box1, const OBBBoundingBox &box2,
                                  bool useCIoU = false, float eps = 1e-7f);
    static float calculateOBBIOU(const OBBBoundingBox &box1, const OBBBoundingBox &box2);
    static vector<OBBBoundingBox> applyNMS(vector<OBBBoundingBox> &boxes,
                                           float nmsThreshold);
};

class OBBResultSaver // OBB 结果保存器
{
public:
    static void saveResults(const vector<OBBBoundingBox> &boxes,
                            const string &imagePath,
                            const string &outputImgDir,
                            const string &outputTxtDir,
                            int srcWidth, int srcHeight);

private:
    static void saveTxtFile(const vector<OBBBoundingBox> &boxes,
                            const string &txtPath, int srcWidth, int srcHeight);
    static void saveVisualization(const vector<OBBBoundingBox> &boxes,
                                  const string &imagePath,
                                  const string &outputPath);
};


class YOLOOBBInference // YOLO OBB 推理类
{
public:
    YOLOOBBInference(const InferenceConfig &config);
    ~YOLOOBBInference();

    bool initialize();
    void runInference();

    // 公开接口（顺序： preprocess/preprocessImage -> runModelInference -> postprocessResults）
    bool processImage(const string &imagePath);
    bool preprocessImage(const string &imagePath);
    bool runModelInference(vector<InferenceOutput> &inferOutputs);
    bool postprocessResults(vector<InferenceOutput> &inferOutputs, const string &imagePath);

private:
    InferenceConfig config_;
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    unique_ptr<OBBPostProcessor> postProcessor_;

    void releaseResources();
};
#endif // YOLO_OBB_H
```

```c++
/*
算法将每个OBB表示为一个二维高斯分布，其中边界框的中心点作为均值，协方差矩阵由边界框的尺寸和旋转角度确定。两个高斯分布之间的相似度通过Bhattacharyya距离来衡量，然后转换为IoU值。
*/
/*
代码首先提取两个边界框的中心坐标，这些坐标将作为高斯分布的均值。
通过调用getCovarianceMatrix函数计算每个OBB对应的协方差矩阵。协方差矩阵编码了边界框的形状信息，包括长宽比例和旋转角度。
接着计算两个高斯分布之间的Bhattacharyya距离
将Bhattacharyya距离转换为Hellinger距离，再进一步转换为IoU值。这种转换确保了结果在[0,1]区间内，符合IoU的定义。
*/
/*
当useCIoU为true时，算法会额外考虑两个边界框的长宽比差异：
通过自适应权重alpha来平衡IoU和长宽比差异的影响，确保在IoU较高时长宽比的影响更为显著
*/
float OBBNMSProcessor::calculateProbIOU(const OBBBoundingBox &box1, const OBBBoundingBox &box2,
                                        bool useCIoU, float eps)
{
    float x1 = box1.cx;
    float y1 = box1.cy;
    float x2 = box2.cx;
    float y2 = box2.cy;

    std::tuple<float, float, float> cov1 = getCovarianceMatrix(box1);
    std::tuple<float, float, float> cov2 = getCovarianceMatrix(box2);

    float a1 = std::get<0>(cov1);
    float b1 = std::get<1>(cov1);
    float c1 = std::get<2>(cov1);

    float a2 = std::get<0>(cov2);
    float b2 = std::get<1>(cov2);
    float c2 = std::get<2>(cov2);

    float dx = x2 - x1;
    float dy = y2 - y1;

    float denom_a = (a1 + a2);
    float denom_b = (b1 + b2);
    float denom_c = (c1 + c2);
    float denominator = denom_a * denom_b - denom_c * denom_c + eps;

    float t1 = ((denom_a * dy * dy + denom_b * dx * dx) / denominator) * 0.25f;
    float t2 = ((denom_c * dx * dy) / denominator) * 0.5f;

    float det1 = std::max(a1 * b1 - c1 * c1, 0.0f);
    float det2 = std::max(a2 * b2 - c2 * c2, 0.0f);
    float sqrt_dets = std::sqrt(det1 * det2) + eps;
    float t3_inner = denominator / (4.0f * sqrt_dets) + eps;
    float t3 = 0.5f * std::log(t3_inner);

    float bd = t1 + t2 + t3;
    bd = std::max(eps, std::min(bd, 100.0f));

    float hd = std::sqrt(1.0f - std::exp(-bd) + eps);
    float iou = 1.0f - hd;
    if (iou < 0.0f)
        iou = 0.0f;
    if (iou > 1.0f)
        iou = 1.0f;

    if (useCIoU)
    {
        float w1 = box1.width;
        float h1 = box1.height;
        float w2 = box2.width;
        float h2 = box2.height;
        float aspect1 = std::atan2(w1, h1);
        float aspect2 = std::atan2(w2, h2);
        float v = (4.0f / (M_PI * M_PI)) * (aspect2 - aspect1) * (aspect2 - aspect1);
        float alpha = v / (v + 1.0f - iou + eps);
        return iou - alpha * v;
    }

    return iou;
}
```
### 3.3 BYTETracker实现

```c++
#pragma once

#include "STrack.h"

struct Object// 跟踪对象结构体-包含了边界框、类别、置信度
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class BYTETracker// BYTETracker类
{
public:
	BYTETracker(int frame_rate = 30, int track_buffer = 30);
	~BYTETracker();

	std::vector<STrack> update(const std::vector<Object>& objects);
	cv::Scalar get_color(int idx);

private:
	std::vector<STrack*> joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb);
	std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

	std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
	void remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);

	void linear_assignment(std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
	std::vector<std::vector<float> > ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs);

	double lapjv(const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;

	std::vector<STrack> tracked_stracks;
	std::vector<STrack> lost_stracks;
	std::vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};

```