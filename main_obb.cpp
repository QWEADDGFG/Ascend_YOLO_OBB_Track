#include "yolo_obb.h"
#include <iostream>
#include <string.h>

static void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -i, --input_dir <path>       Input image directory\n"
              << "  -m, --model <path>           Model file path (.om)\n"
              << "  -o, --output_img_dir <path>  Output image directory\n"
              << "  -l, --output_txt_dir <path>  Output txt/label directory\n"
              << "  --conf <float>               Confidence threshold (default 0.25)\n"
              << "  --nms <float>                NMS threshold (default 0.45)\n"
              << "  --help                       Show this help\n";
}

int main(int argc, char** argv)
{
    InferenceConfig config;

    // 默认配置（可被命令行选项覆盖）
    config.modelPath = "/home/HwHiAiUser/gp/yolov8_bytetrack/build/YOLO11s_base_obb_MVRSD_640.om";
    config.inputDir = "/home/HwHiAiUser/gp/DATASETS/MVRSD/test";
    config.outputImgDir = "../output_obb/images";
    config.outputTxtDir = "../output_obb/labels";

    config.modelWidth = 640;
    config.modelHeight = 640;
    config.modelOutputBoxNum = 8400;
    config.classNum = 1;

    config.confidenceThreshold = 0.25f;
    config.nmsThreshold = 0.45f;

    // 解析命令行参数（简单解析）
    for (int i = 1; i < argc; ++i)
    {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input_dir") == 0) && i + 1 < argc)
        {
            config.inputDir = argv[++i];
        }
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc)
        {
            config.modelPath = argv[++i];
        }
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output_img_dir") == 0) && i + 1 < argc)
        {
            config.outputImgDir = argv[++i];
        }
        else if ((strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--output_txt_dir") == 0) && i + 1 < argc)
        {
            config.outputTxtDir = argv[++i];
        }
        else if (strcmp(argv[i], "--conf") == 0 && i + 1 < argc)
        {
            config.confidenceThreshold = static_cast<float>(atof(argv[++i]));
        }
        else if (strcmp(argv[i], "--nms") == 0 && i + 1 < argc)
        {
            config.nmsThreshold = static_cast<float>(atof(argv[++i]));
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown option or missing argument: " << argv[i] << "\n";
            printUsage(argv[0]);
            return -1;
        }
    }

    // 打印最终使用的配置（便于调试）
    ACLLITE_LOG_INFO("Using config:");
    ACLLITE_LOG_INFO("  modelPath: %s", config.modelPath.c_str());
    ACLLITE_LOG_INFO("  inputDir: %s", config.inputDir.c_str());
    ACLLITE_LOG_INFO("  outputImgDir: %s", config.outputImgDir.c_str());
    ACLLITE_LOG_INFO("  outputTxtDir: %s", config.outputTxtDir.c_str());
    ACLLITE_LOG_INFO("  confidenceThreshold: %f", config.confidenceThreshold);
    ACLLITE_LOG_INFO("  nmsThreshold: %f", config.nmsThreshold);

    YOLOOBBInference inference(config);

    if (inference.initialize())
    {
        ACLLITE_LOG_INFO("Starting OBB inference with ProbIoU...");
        inference.runInference();
        ACLLITE_LOG_INFO("OBB inference completed");
    }
    else
    {
        ACLLITE_LOG_ERROR("Failed to initialize OBB inference engine");
        return -1;
    }

    return 0;
}