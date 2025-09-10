// main_all.cpp
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

// ========== 1. 外部头文件 ==========
#include "BYTETracker.h"
#include "yolo_obb.h"
// ==================================

namespace fs = std::filesystem;
using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

/* ----------------------------------------------------------
 * 通用
 * ---------------------------------------------------------- */
template<typename... Args>
static std::string str_format(const std::string& fmt, Args... args)
{
    int sz = std::snprintf(nullptr, 0, fmt.c_str(), args...) + 1;
    std::unique_ptr<char[]> buf(new (std::nothrow) char[sz]);
    if (!buf) return "";
    std::snprintf(buf.get(), sz, fmt.c_str(), args...);
    return std::string(buf.get(), buf.get() + sz - 1);
}

/* ----------------------------------------------------------
 * YoloOBB 封装
 * ---------------------------------------------------------- */
class YoloOBBWrapper
{
public:
    YoloOBBWrapper(const InferenceConfig& cfg) : cfg_(cfg), inference_(cfg_) {}
    bool Init() { return inference_.initialize(); }

    // 纯检测接口：返回 OBB 和 AABB（Object）
    bool Detect(const std::string& imagePath,
                std::vector<OBBBoundingBox>& outOBBs,
                std::vector<Object>& outObjects)
    {
        if (!inference_.preprocessImage(imagePath)) return false;
        std::vector<InferenceOutput> outs;
        if (!inference_.runModelInference(outs)) return false;

        cv::Mat src = cv::imread(imagePath);
        if (src.empty()) return false;

        float* ptr = static_cast<float*>(outs[0].data.get());
        OBBPostProcessor post(cfg_.modelOutputBoxNum, cfg_.classNum);
        auto boxes = post.parseOutput(ptr, outs[0].size,
                                      src.cols, src.rows,
                                      cfg_.modelWidth, cfg_.modelHeight,
                                      cfg_.confidenceThreshold);
        outOBBs = OBBNMSProcessor::applyNMS(boxes, cfg_.nmsThreshold);

        outObjects.clear();
        for (const auto& obb : outOBBs)
        {
            auto pts = obb.getCornerPoints();
            float xmi = pts[0].x, ymi = pts[0].y, xma = pts[0].x, yma = pts[0].y;
            for (int k = 1; k < 4; ++k)
            {
                xmi = std::min(xmi, pts[k].x); ymi = std::min(ymi, pts[k].y);
                xma = std::max(xma, pts[k].x); yma = std::max(yma, pts[k].y);
            }
            Object obj;
            obj.label = static_cast<int>(obb.classIndex);
            obj.prob  = obb.confidence;
            obj.rect.x = xmi; obj.rect.y = ymi;
            obj.rect.width  = xma - xmi;
            obj.rect.height = yma - ymi;
            outObjects.push_back(obj);
        }
        return true;
    }

private:
    InferenceConfig cfg_;
    YOLOOBBInference inference_;
};

/* ----------------------------------------------------------
 * 绘制 OBB
 * ---------------------------------------------------------- */
static void drawOBB(cv::Mat& img, const OBBBoundingBox& obb,
                    const cv::Scalar& color, const std::string& text = "")
{
    auto pts_f = obb.getCornerPoints();
    std::vector<cv::Point> pts;
    for (auto& p : pts_f) pts.emplace_back(cv::Point(int(std::round(p.x)), int(std::round(p.y))));
    const cv::Point* ppt = pts.data();
    int n = int(pts.size());
    cv::polylines(img, &ppt, &n, 1, true, color, 2);
    cv::circle(img, cv::Point(int(std::round(obb.cx)), int(std::round(obb.cy))),
               3, color, -1);
    std::string txt = text.empty()
        ? cv::format("cls:%d %.2f", int(obb.classIndex), obb.confidence)
        : text;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int tx0 = std::max(0, int(std::round(obb.cx - ts.width / 2.f)));
    int ty0 = std::max(0, int(std::round(obb.cy - obb.height / 2 - 6)));
    cv::putText(img, txt, cv::Point(tx0, ty0), cv::FONT_HERSHEY_SIMPLEX,
                0.5, color, 1);
}

/* ----------------------------------------------------------
 * 命令行参数解析结构
 * ---------------------------------------------------------- */
struct Args
{
    std::string task;               // "detect" 或 "track"
    std::string modelPath;
    std::string imageIn;            // 单张图或图目录
    std::string imgOut;             // 检测/跟踪结果图保存路径（文件或目录）
    std::string labelOut;           // 检测标签保存路径（文件或目录）
    std::string trackImgOut;        // 仅跟踪任务时有效，可空
    int         classNum      = 1;
    float       confThresh    = 0.25f;
    float       nmsThresh     = 0.45f;
    int         modelW        = 640;
    int         modelH        = 640;
    int         modelBoxNum   = 8400;
};

static void printUsage(const char* prog)
{
    std::cout <<
    "\nUsage:\n"
    "  (1) 纯检测：\n"
    "    " << prog << " detect \\\n"
    "        --model  ../model/YOLO11s_obb_video_base_640.om \\\n"
    "        --input  /path/to/xxx.jpg \\\n"
    "        --image_out  /path/to/res.jpg \\\n"
    "        --label_out  /path/to/res.txt\n\n"
    "  (2) 检测+跟踪：\n"
    "    " << prog << " track \\\n"
    "        --model  ../model/YOLO11s_obb_video_base_640.om \\\n"
    "        --input  /path/to/image_dir \\\n"
    "        --image_out  /path/to/det_dir \\\n"
    "        --label_out  /path/to/det_label_dir \\\n"
    "        --track_image_out /path/to/track_dir\n\n"
    "  可选参数：\n"
    "    --class_num, --conf, --nms, --model_w, --model_h, --model_box_num\n";
}

static bool parseArgs(int argc, char** argv, Args& a)
{
    if (argc < 2) return false;
    a.task = argv[1];
    if (a.task != "detect" && a.task != "track") return false;

    for (int i = 2; i < argc; ++i)
    {
        std::string key = argv[i];
        if (key == "--model"          && i + 1 < argc) a.modelPath     = argv[++i];
        else if (key == "--input"     && i + 1 < argc) a.imageIn       = argv[++i];
        else if (key == "--image_out" && i + 1 < argc) a.imgOut        = argv[++i];
        else if (key == "--label_out" && i + 1 < argc) a.labelOut      = argv[++i];
        else if (key == "--track_image_out" && i + 1 < argc) a.trackImgOut = argv[++i];
        else if (key == "--class_num" && i + 1 < argc) a.classNum      = std::stoi(argv[++i]);
        else if (key == "--conf"      && i + 1 < argc) a.confThresh    = std::stof(argv[++i]);
        else if (key == "--nms"       && i + 1 < argc) a.nmsThresh     = std::stof(argv[++i]);
        else if (key == "--model_w"   && i + 1 < argc) a.modelW        = std::stoi(argv[++i]);
        else if (key == "--model_h"   && i + 1 < argc) a.modelH        = std::stoi(argv[++i]);
        else if (key == "--model_box_num" && i + 1 < argc) a.modelBoxNum = std::stoi(argv[++i]);
    }
    return !a.modelPath.empty() && !a.imageIn.empty() &&
           !a.imgOut.empty()   && !a.labelOut.empty();
}

/* ----------------------------------------------------------
 * 纯检测任务（单张图 或 连续编号文件夹）
 * ---------------------------------------------------------- */
static void runDetect(const Args& a)
{
    InferenceConfig cfg;
    cfg.modelPath      = a.modelPath;
    cfg.modelWidth     = a.modelW;
    cfg.modelHeight    = a.modelH;
    cfg.modelOutputBoxNum = a.modelBoxNum;
    cfg.classNum       = a.classNum;
    cfg.confidenceThreshold = a.confThresh;
    cfg.nmsThreshold   = a.nmsThresh;

    YoloOBBWrapper detector(cfg);
    if (!detector.Init()) { std::cerr << "Init detector failed.\n"; return; }

    /* 如果是单张图，列表只有 1 项；如果是目录，按 6 位编号扫描 */
    std::vector<fs::path> inFiles;
    if (fs::is_regular_file(a.imageIn))
        inFiles.emplace_back(a.imageIn);
    else
    {
        for (int no = 1; ; ++no)
        {
            fs::path p = fs::path(a.imageIn) / str_format("%06d.jpg", no);
            if (!fs::exists(p)) break;
            inFiles.push_back(p);
        }
    }
    if (inFiles.empty()) { std::cerr << "No input images.\n"; return; }

    fs::create_directories(a.imgOut);
    fs::create_directories(a.labelOut);

    const cv::Scalar colors[] = {
        {255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255}
    };

    int64_t totalUs = 0;
    int     done    = 0;

    for (const auto& inPath : inFiles)
    {
        cv::Mat img = cv::imread(inPath.string());
        if (img.empty()) continue;

        std::vector<OBBBoundingBox> obbs;
        std::vector<Object> objects;
        auto t0 = steady_clock::now();
        bool ok = detector.Detect(inPath.string(), obbs, objects);
        auto t1 = steady_clock::now();
        if (!ok) continue;

        int64_t us = duration_cast<microseconds>(t1 - t0).count();
        totalUs += us;
        ++done;

        /* 画框 */
        cv::Mat res = img.clone();
        for (const auto& o : obbs)
            drawOBB(res, o, colors[o.classIndex % 6]);

        /* 输出文件：与输入同名 */
        fs::path outImg   = fs::path(a.imgOut)   / (inPath.stem().string() + ".jpg");
        fs::path outLabel = fs::path(a.labelOut) / (inPath.stem().string() + ".txt");

        cv::imwrite(outImg.string(), res);

        std::ofstream fo(outLabel.string());
        for (const auto& o : obbs)
        {
            auto pts = o.getCornerPoints();
            fo << o.classIndex << ' ' << o.confidence;
            for (const auto& p : pts) fo << ' ' << p.x << ' ' << p.y;
            fo << '\n';
        }
    }

    if (done == 0) { std::cerr << "No frame processed.\n"; return; }

    double fps = done * 1'000'000.0 / totalUs;
    std::cout << "[Detect] 平均 FPS = " << fps
              << "  (共 " << done << " 张，总耗时 "
              << totalUs / 1000 << " ms)\n";
}

/* ----------------------------------------------------------
 * 检测+跟踪任务（连续编号图像序列）
 * ---------------------------------------------------------- */
static void runTrack(const Args& a)
{
    InferenceConfig cfg;
    cfg.modelPath = a.modelPath;
    cfg.modelWidth = a.modelW; cfg.modelHeight = a.modelH;
    cfg.modelOutputBoxNum = a.modelBoxNum;
    cfg.classNum = a.classNum;
    cfg.confidenceThreshold = a.confThresh;
    cfg.nmsThreshold = a.nmsThresh;

    YoloOBBWrapper detector(cfg);
    if (!detector.Init()) { std::cerr << "Init detector failed.\n"; return; }

    BYTETracker tracker(30, 30);

    fs::create_directories(a.imgOut);
    fs::create_directories(a.labelOut);
    if (!a.trackImgOut.empty()) fs::create_directories(a.trackImgOut);

    int64_t totalDetUs = 0, totalTrkUs = 0;
    int frameCnt = 0;

    for (int no = 1; ; ++no)
    {
        std::string imgPath = str_format("%s/%06d.jpg", a.imageIn.c_str(), no);
        if (!fs::exists(imgPath)) break;
        ++frameCnt;

        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) continue;

        /* ---- detect ---- */
        std::vector<OBBBoundingBox> obbs; std::vector<Object> objects;
        auto t0 = steady_clock::now();
        bool ok = detector.Detect(imgPath, obbs, objects);
        auto t1 = steady_clock::now();
        if (!ok) continue;
        int64_t detUs = duration_cast<microseconds>(t1 - t0).count();
        totalDetUs += detUs;

        /* ---- track ---- */
        t0 = steady_clock::now();
        std::vector<STrack> stracks = tracker.update(objects);
        t1 = steady_clock::now();
        int64_t trkUs = duration_cast<microseconds>(t1 - t0).count();
        totalTrkUs += trkUs;

        /* 关联 obb -> track_id */
        std::map<int, OBBBoundingBox> tid2obb;
        for (const auto& trk : stracks)
        {
            float bestIou = 0; int bestIdx = -1;
            cv::Rect2f r1(trk.tlwh[0], trk.tlwh[1], trk.tlwh[2], trk.tlwh[3]);
            for (size_t k = 0; k < obbs.size(); ++k)
            {
                auto pts = obbs[k].getCornerPoints();
                float xmi = pts[0].x, ymi = pts[0].y, xma = pts[0].x, yma = pts[0].y;
                for (int p = 1; p < 4; ++p)
                { xmi = std::min(xmi, pts[p].x); ymi = std::min(ymi, pts[p].y);
                  xma = std::max(xma, pts[p].x); yma = std::max(yma, pts[p].y); }
                cv::Rect2f r2(xmi, ymi, xma - xmi, yma - ymi);
                float inter = (r1 & r2).area();
                float u = r1.area() + r2.area() - inter + 1e-6f;
                float iou = inter / u;
                if (iou > bestIou) { bestIou = iou; bestIdx = int(k); }
            }
            if (bestIdx >= 0 && bestIou > 0) tid2obb[trk.track_id] = obbs[bestIdx];
        }

        /* 保存检测图/标签 */
        const cv::Scalar colors[] = {
            {255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255}
        };
        cv::Mat detImg = img.clone();
        for (const auto& o : obbs) drawOBB(detImg, o, colors[o.classIndex % 6]);
        cv::imwrite(str_format("%s/%06d.jpg", a.imgOut.c_str(), no), detImg);

        std::ofstream folab(str_format("%s/%06d.txt", a.labelOut.c_str(), no));
        for (const auto& o : obbs)
        {
            auto pts = o.getCornerPoints();
            folab << o.classIndex << ' ' << o.confidence;
            for (const auto& p : pts) folab << ' ' << p.x << ' ' << p.y;
            folab << '\n';
        }

        /* 保存跟踪图（可选） */
        if (!a.trackImgOut.empty())
        {
            cv::Mat trkImg = img.clone();
            for (const auto& trk : stracks)
            {
                cv::Scalar color = tracker.get_color(trk.track_id);
                int cx = int(trk.tlwh[0] + trk.tlwh[2] / 2);
                int cy = int(trk.tlwh[1] + trk.tlwh[3] / 2);
                cv::putText(trkImg, cv::format("%d", trk.track_id),
                            cv::Point(cx - 10, cy), 0, 0.6, color, 2, cv::LINE_AA);
                auto it = tid2obb.find(trk.track_id);
                if (it != tid2obb.end())
                    drawOBB(trkImg, it->second, color, cv::format("ID:%d", trk.track_id));
            }
            cv::imwrite(str_format("%s/%06d.jpg", a.trackImgOut.c_str(), no), trkImg);
        }
    }

    if (frameCnt == 0) { std::cerr << "No frame processed.\n"; return; }

    double fpsDet = frameCnt * 1'000'000.0 / totalDetUs;
    double fpsTrk = frameCnt * 1'000'000.0 / totalTrkUs;
    std::cout << "[Track] 检测模块平均 FPS = " << fpsDet
              << "  |  跟踪模块平均 FPS = " << fpsTrk << '\n';
}

/* ----------------------------------------------------------
 * main
 * ---------------------------------------------------------- */
int main(int argc, char** argv)
{
    Args a;
    if (!parseArgs(argc, argv, a)) { printUsage(argv[0]); return -1; }

    if (a.task == "detect") runDetect(a);
    else                    runTrack(a);
    return 0;
}