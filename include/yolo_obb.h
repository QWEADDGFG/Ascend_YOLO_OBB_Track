#ifndef YOLO_OBB_H
#define YOLO_OBB_H

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <cmath>
#include <tuple>
#include <numeric>

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

struct OBBBoundingBox
{
    float cx, cy, width, height;
    float angle;
    float confidence;
    size_t classIndex;
    size_t index;

    vector<cv::Point2f> getCornerPoints() const;
    OBBBoundingBox();
};

class InferenceConfig
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

class Utils
{
public:
    static bool sortByConfidence(const OBBBoundingBox &a, const OBBBoundingBox &b);
    static void createDirectory(const string &path);
    static vector<string> getImagePaths(const string &dirPath);
    static string getFileNameWithoutExt(const string &path);
    static float normalizeAngle(float angle);
};

class OBBPostProcessor
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

class OBBNMSProcessor
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

class OBBResultSaver
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


class YOLOOBBInference
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