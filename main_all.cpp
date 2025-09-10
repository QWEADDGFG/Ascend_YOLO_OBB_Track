#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <map>
#include <memory>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <float.h>
#include <stdio.h>

#include "BYTETracker.h"
// 用 yolo_obb.h 替换 yolov8.h
#include "yolo_obb.h"

// 简单包装 YOLOOBBInference 以便初始化与单张图片推理
class YoloOBBWrapper {
public:
    YoloOBBWrapper(const InferenceConfig &cfg) : cfg_(cfg), inference_(cfg_) {}

    bool Init() {
        return inference_.initialize();
    }

    // 对单张图片进行推理，返回 OBB 列表（filter 后、NMS 后）以及同时生成的 Object 列表（AABB）
    bool Detect(const std::string &imagePath,
                std::vector<OBBBoundingBox> &outOBBs,
                std::vector<Object> &outObjects)
    {
        // 使用 YOLOOBBInference 的 preprocess/run/postprocess 分步接口（按 yolo_obb.h 假定提供）
        if (!inference_.preprocessImage(imagePath)) {
            ACLLITE_LOG_ERROR("yolo_obb preprocessImage failed for %s", imagePath.c_str());
            return false;
        }

        std::vector<InferenceOutput> inferOutputs;
        if (!inference_.runModelInference(inferOutputs)) {
            ACLLITE_LOG_ERROR("yolo_obb runModelInference failed for %s", imagePath.c_str());
            return false;
        }

        // 读取源图像以获得尺寸
        cv::Mat srcImage = cv::imread(imagePath);
        if (srcImage.empty()) {
            ACLLITE_LOG_ERROR("Cannot read source image: %s", imagePath.c_str());
            return false;
        }

        float *outputData = static_cast<float *>(inferOutputs[0].data.get());

        // 使用推理器内部的 postProcessor_：为了避免修改原库，这里使用本地的 OBBPostProcessor（假设构造签名匹配）
        OBBPostProcessor localPost(cfg_.modelOutputBoxNum, cfg_.classNum);

        std::vector<OBBBoundingBox> boxes = localPost.parseOutput(
            outputData, inferOutputs[0].size,
            srcImage.cols, srcImage.rows,
            cfg_.modelWidth, cfg_.modelHeight,
            cfg_.confidenceThreshold);

        std::vector<OBBBoundingBox> finalBoxes = OBBNMSProcessor::applyNMS(boxes, cfg_.nmsThreshold);

        // 返回 finalBoxes，同时转换为 Object (AABB)
        outOBBs = finalBoxes;
        outObjects.clear();
        for (const auto &obb : finalBoxes) {
            // convert to AABB (axis-aligned bounding box)
            std::vector<cv::Point2f> pts = obb.getCornerPoints();
            float xmin = pts[0].x, ymin = pts[0].y, xmax = pts[0].x, ymax = pts[0].y;
            for (int k = 1; k < 4; ++k) {
                xmin = std::min(xmin, pts[k].x);
                ymin = std::min(ymin, pts[k].y);
                xmax = std::max(xmax, pts[k].x);
                ymax = std::max(ymax, pts[k].y);
            }
            Object obj;
            obj.label = static_cast<int>(obb.classIndex);
            obj.prob = obb.confidence;
            obj.rect.x = xmin;
            obj.rect.y = ymin;
            obj.rect.width = xmax - xmin;
            obj.rect.height = ymax - ymin;
            outObjects.push_back(obj);
        }

        return true;
    }

private:
    InferenceConfig cfg_;
    YOLOOBBInference inference_;
};

// 格式化字符串工具（保留你的 str_format）
template<typename ... Args>
static std::string str_format(const std::string& format, Args ... args)
{
    auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    std::unique_ptr<char[]> buf(new(std::nothrow) char[size_buf]);

    if (!buf)
        return std::string("");

    std::snprintf(buf.get(), size_buf, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size_buf - 1);
}

// 将单帧 OBB 转换为 ByteTrack 的 Object 列表（AABB），如果你更喜欢直接调用 wrapper.Detect 已经返回 this 了。
// 这里保留以备复用
static void convertOBBsToObjects(const std::vector<OBBBoundingBox> &obbs, std::vector<Object> &objects)
{
    objects.clear();
    for (const auto &obb : obbs)
    {
        std::vector<cv::Point2f> pts = obb.getCornerPoints();
        float xmin = pts[0].x, ymin = pts[0].y, xmax = pts[0].x, ymax = pts[0].y;
        for (int k = 1; k < 4; ++k) {
            xmin = std::min(xmin, pts[k].x);
            ymin = std::min(ymin, pts[k].y);
            xmax = std::max(xmax, pts[k].x);
            ymax = std::max(ymax, pts[k].y);
        }
        Object obj;
        obj.label = static_cast<int>(obb.classIndex);
        obj.prob = obb.confidence;
        obj.rect.x = xmin;
        obj.rect.y = ymin;
        obj.rect.width = xmax - xmin;
        obj.rect.height = ymax - ymin;
        objects.push_back(obj);
    }
}

// 用于绘制 OBB（仅绘制旋转矩形的四个顶点连线与中心，不绘制最小外接矩形）
static void drawOBB(cv::Mat &img, const OBBBoundingBox &obb, const cv::Scalar &color, const std::string &text = "")
{
    std::vector<cv::Point2f> pts_f = obb.getCornerPoints();
    std::vector<cv::Point> pts;
    for (auto &p : pts_f) pts.emplace_back(cv::Point(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y))));
    const cv::Point *pts_ptr = pts.data();
    int n = static_cast<int>(pts.size());
    // 只绘制多边形（OBB 的四条边）
    polylines(img, &pts_ptr, &n, 1, true, color, 2);

    // center
    cv::circle(img, cv::Point(static_cast<int>(std::round(obb.cx)), static_cast<int>(std::round(obb.cy))), 3, color, -1);

    // text: label + conf 或自定义 text（文本背景或边框略微调整，避免与最小外接矩形相关绘制）
    std::string labelText = text;
    if (labelText.empty()) {
        if (obb.classIndex >= 0 && obb.classIndex < static_cast<int>(label->size())) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s:%.3f", label[obb.classIndex].c_str(), obb.confidence);
            labelText = std::string(buf);
        } else {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "id:%ld, conf:%.3f", obb.classIndex, obb.confidence);
            labelText = std::string(buf);
        }
    }
    int baseline = 0;
    cv::Size tsize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int tx = std::max(0, static_cast<int>(std::round(obb.cx - tsize.width/2)));
    int ty = std::max(0, static_cast<int>(std::round(obb.cy - obb.height/2 - 6)));
    cv::putText(img, labelText, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagedir]\n", argv[0]);
        return -1;
    }

    const char* imageDir = argv[1];

    // 配置 inference
    InferenceConfig cfg;
    cfg.modelPath = "../model/YOLO11s_obb_video_base_640.om"; // 根据需要修改
    cfg.modelWidth = 640;
    cfg.modelHeight = 640;
    cfg.modelOutputBoxNum = 8400;
    cfg.classNum = 1;
    cfg.confidenceThreshold = 0.25f;
    cfg.nmsThreshold = 0.45f;
    cfg.inputDir = ""; // unused here

    // 初始化检测器（YOLO OBB）
    YoloOBBWrapper detector(cfg);
    if (!detector.Init()) {
        std::cerr << "Failed to initialize YOLO OBB detector." << std::endl;
        return -1;
    }

    BYTETracker tracker(30, 30);

    int num_frames = 0;
    int64_t total_us = 1; // microseconds to avoid div0
    int64_t one_us = 0;

    // track_id -> last associated OBB (用于绘制 OBB 时引用)
    std::map<int, OBBBoundingBox> trackid2obb;

    // // ensure results dir
    // std::filesystem::create_directories("../results");

    // 遍历图像序列：从 000001.jpg 开始，向上查找直到没有文件（连续编号或中断停止）
    // 支持任意数量图片（例如 000001.jpg .. 000413.jpg 甚至更多）
    int no = 1;
    while (true) {
        std::string imagePath = str_format("%s/%06d.jpg", imageDir, no);
        if (!std::filesystem::exists(imagePath)) {
            break;
        }

        ++num_frames;

        std::vector<OBBBoundingBox> obbs;
        std::vector<Object> objects;
        bool ok = detector.Detect(imagePath, obbs, objects);
        if (!ok) {
            ACLLITE_LOG_ERROR("Detect failed for %s", imagePath.c_str());
            ++no;
            continue;
        }

        // 调用 tracker
        auto t0 = std::chrono::steady_clock::now();
        std::vector<STrack> output_stracks = tracker.update(objects);
        auto t1 = std::chrono::steady_clock::now();
        one_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::cout << "Frame " << no << " Detect objects " << objects.size() << ", Tracking update takes " << one_us << " us" << std::endl;
        total_us += one_us;

        // 将输出 stracks 的 track_id 与检测到的 obb 做关联：
        // 对于每个 track，找与其 AABB（tlwh） IoU 最大的 obb（AABB）并绑定（如果 IoU > 0）
        trackid2obb.clear();
        for (const auto &trk : output_stracks) {
            float bestIou = 0.0f;
            int bestIdx = -1;
            std::vector<float> tlwh = trk.tlwh;
            cv::Rect2f aabb_t(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
            for (size_t k = 0; k < obbs.size(); ++k) {
                std::vector<cv::Point2f> pts = obbs[k].getCornerPoints();
                float xmin = pts[0].x, ymin = pts[0].y, xmax = pts[0].x, ymax = pts[0].y;
                for (int p=1; p<4; ++p) { xmin = std::min(xmin, pts[p].x); ymin = std::min(ymin, pts[p].y); xmax = std::max(xmax, pts[p].x); ymax = std::max(ymax, pts[p].y); }
                cv::Rect2f aabb_o(xmin, ymin, xmax - xmin, ymax - ymin);
                // IoU
                float interW = std::max(0.0f, std::min(aabb_t.x + aabb_t.width, aabb_o.x + aabb_o.width) - std::max(aabb_t.x, aabb_o.x));
                float interH = std::max(0.0f, std::min(aabb_t.y + aabb_t.height, aabb_o.y + aabb_o.height) - std::max(aabb_t.y, aabb_o.y));
                float interArea = interW * interH;
                float unionArea = aabb_t.width * aabb_t.height + aabb_o.width * aabb_o.height - interArea + 1e-6f;
                float iou = interArea / unionArea;
                if (iou > bestIou) {
                    bestIou = iou;
                    bestIdx = static_cast<int>(k);
                }
            }
            if (bestIdx >= 0 && bestIou > 0.0f) {
                trackid2obb[trk.track_id] = obbs[bestIdx];
            }
        }

        // 读取图片用于绘制
        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            ACLLITE_LOG_ERROR("Read image failed: %s", imagePath.c_str());
            ++no;
            continue;
        }

        // 绘制检测到的 OBB（仅 OBB，不绘制最小外接矩形）
        const std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)
        };

        for (size_t k = 0; k < obbs.size(); ++k) {
            cv::Scalar c = colors[obbs[k].classIndex % colors.size()];
            // draw OBB only
            drawOBB(img, obbs[k], c);
        }

        // 绘制跟踪结果：只绘制 track id 文本 与 （若有）对应 OBB（绘制 OBB 更醒目）
        // 不绘制 AABB（最小外接矩形）
        for (size_t i = 0; i < output_stracks.size(); ++i) {
            int tid = output_stracks[i].track_id;
            cv::Scalar color = tracker.get_color(tid);

            // 仅绘制 ID 文本；若需要在 ID 附近放置文本，则使用 track 的 tlwh 的中心
            std::vector<float> tlwh = output_stracks[i].tlwh;
            int cx = static_cast<int>(tlwh[0] + tlwh[2] / 2.0f);
            int cy = static_cast<int>(tlwh[1] + tlwh[3] / 2.0f);
            cv::putText(img, cv::format("%d", tid), cv::Point(std::max(0, cx - 10), std::max(0, cy)), 0, 0.6, color, 2, cv::LINE_AA);

            auto it = trackid2obb.find(tid);
            if (it != trackid2obb.end()) {
                // draw OBB with thicker line and label
                drawOBB(img, it->second, color, std::string("ID:") + std::to_string(tid));
            }
        }

        // 绘制帧信息和 FPS
        int fps = static_cast<int>(num_frames * 1'000'000LL / total_us);
        cv::putText(img, cv::format("frame: %d fps: %d num: %d", num_frames, fps, (int)output_stracks.size()),
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        // 保存结果图像
        std::string outPath = str_format("../output_obb_track/%06d.jpg", no);
        cv::imwrite(outPath, img);

        ++no;
    }

    std::cout << "Processed frames: " << num_frames << std::endl;
    std::cout << "FPS: " << (num_frames * 1000000LL / total_us) << std::endl;

    return 0;
}
