#include "yolo_obb.h"

// const string label[] = {"AFV", "CV", "LMV", "MCV", "SMV"};
const string label[] = {"car"};

OBBBoundingBox::OBBBoundingBox() : cx(0), cy(0), width(0), height(0), angle(0),
                                   confidence(0), classIndex(0), index(0) {}

vector<cv::Point2f> OBBBoundingBox::getCornerPoints() const
{
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float w_half = width / 2.0f;
    float h_half = height / 2.0f;

    vector<cv::Point2f> points(4);

    float dx1 = -w_half * cos_a + h_half * sin_a;
    float dy1 = -w_half * sin_a - h_half * cos_a;

    float dx2 = w_half * cos_a + h_half * sin_a;
    float dy2 = w_half * sin_a - h_half * cos_a;

    float dx3 = w_half * cos_a - h_half * sin_a;
    float dy3 = w_half * sin_a + h_half * cos_a;

    float dx4 = -w_half * cos_a - h_half * sin_a;
    float dy4 = -w_half * sin_a + h_half * cos_a;

    points[0] = cv::Point2f(cx + dx1, cy + dy1);
    points[1] = cv::Point2f(cx + dx2, cy + dy2);
    points[2] = cv::Point2f(cx + dx3, cy + dy3);
    points[3] = cv::Point2f(cx + dx4, cy + dy4);

    return points;
}

InferenceConfig::InferenceConfig() : modelWidth(640), modelHeight(640),
                                     confidenceThreshold(0.001), nmsThreshold(0.45),
                                     modelOutputBoxNum(8400), classNum(5) {}

bool Utils::sortByConfidence(const OBBBoundingBox &a, const OBBBoundingBox &b)
{
    return a.confidence > b.confidence;
}

void Utils::createDirectory(const string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        mkdir(path.c_str(), 0777);
        ACLLITE_LOG_INFO("Created directory: %s", path.c_str());
    }
}

vector<string> Utils::getImagePaths(const string &dirPath)
{
    vector<string> imagePaths;
    DIR *dir = opendir(dirPath.c_str());
    if (!dir)
    {
        ACLLITE_LOG_ERROR("Cannot open directory: %s", dirPath.c_str());
        return imagePaths;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        string name = entry->d_name;
        if (name != "." && name != ".." && name != ".keep")
        {
            string fullPath = dirPath + "/" + name;
            size_t dot = name.find_last_of(".");
            if (dot == string::npos)
                continue;
            string ext = name.substr(dot + 1);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp")
            {
                imagePaths.push_back(fullPath);
            }
        }
    }
    closedir(dir);
    return imagePaths;
}

string Utils::getFileNameWithoutExt(const string &path)
{
    size_t lastSlash = path.find_last_of("/");
    size_t lastDot = path.find_last_of(".");
    return path.substr(lastSlash + 1, lastDot - lastSlash - 1);
}

float Utils::normalizeAngle(float angle)
{
    angle = std::fmod(angle, (float)M_PI);
    if (angle < -M_PI)
        angle += 2.0f * M_PI;
    if (angle >= M_PI)
        angle -= 2.0f * M_PI;

    if (angle >= M_PI / 2.0f)
        angle -= M_PI;
    if (angle < -M_PI / 2.0f)
        angle += M_PI;

    return angle;
}

OBBPostProcessor::OBBPostProcessor(size_t boxNum, size_t classNum)
    : modelOutputBoxNum_(boxNum), classNum_(classNum) {}

vector<OBBBoundingBox> OBBPostProcessor::parseOutput(float *outputData, size_t outputSize,
                                                     int srcWidth, int srcHeight,
                                                     int modelWidth, int modelHeight,
                                                     float confidenceThreshold)
{
    vector<OBBBoundingBox> boxes;

    for (size_t i = 0; i < modelOutputBoxNum_; ++i)
    {
        float maxValue = 0.0f;
        size_t maxIndex = 0;

        for (size_t j = 0; j < classNum_; ++j)
        {
            float value = outputData[(4 + j) * modelOutputBoxNum_ + i];
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = j;
            }
        }

        if (maxValue > confidenceThreshold)
        {
            OBBBoundingBox box;

            float cx = outputData[0 * modelOutputBoxNum_ + i] * srcWidth / modelWidth;
            float cy = outputData[1 * modelOutputBoxNum_ + i] * srcHeight / modelHeight;
            float w = outputData[2 * modelOutputBoxNum_ + i] * srcWidth / modelWidth;
            float h = outputData[3 * modelOutputBoxNum_ + i] * srcHeight / modelHeight;

            float raw_angle = 0.0f;
#if MODEL_ANGLE_MODE == 0
            raw_angle = outputData[(4 + classNum_) * modelOutputBoxNum_ + i];
#elif MODEL_ANGLE_MODE == 1
            {
                float sin_v = outputData[(4 + classNum_) * modelOutputBoxNum_ + i];
                float cos_v = outputData[(4 + classNum_ + 1) * modelOutputBoxNum_ + i];
                raw_angle = atan2f(sin_v, cos_v);
            }
#else
#error "Unsupported MODEL_ANGLE_MODE"
#endif

            float angle = Utils::normalizeAngle(raw_angle);

            box.cx = cx;
            box.cy = cy;
            box.width = w;
            box.height = h;
            box.angle = angle;
            box.confidence = maxValue;
            box.classIndex = maxIndex;
            box.index = i;

            boxes.push_back(box);
        }
    }
    return boxes;
}

std::tuple<float, float, float> OBBNMSProcessor::getCovarianceMatrix(const OBBBoundingBox &box)
{
    float w = box.width;
    float h = box.height;
    float r = box.angle;

    float cos_r = std::cos(r);
    float sin_r = std::sin(r);

    float w_sq = (w * w) / 12.0f;
    float h_sq = (h * h) / 12.0f;

    float a = w_sq * cos_r * cos_r + h_sq * sin_r * sin_r;
    float b = w_sq * sin_r * sin_r + h_sq * cos_r * cos_r;
    float c = (w_sq - h_sq) * cos_r * sin_r;

    return std::make_tuple(a, b, c);
}

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

float OBBNMSProcessor::calculateOBBIOU(const OBBBoundingBox &box1, const OBBBoundingBox &box2)
{
    return calculateProbIOU(box1, box2, false);
}

vector<OBBBoundingBox> OBBNMSProcessor::applyNMS(vector<OBBBoundingBox> &boxes,
                                                 float nmsThreshold)
{
    vector<OBBBoundingBox> result;
    sort(boxes.begin(), boxes.end(), Utils::sortByConfidence);

    while (!boxes.empty())
    {
        OBBBoundingBox best = boxes[0];
        result.push_back(best);

        vector<OBBBoundingBox> remaining;
        for (size_t i = 1; i < boxes.size(); ++i)
        {
            if (boxes[i].classIndex != best.classIndex)
            {
                remaining.push_back(boxes[i]);
                continue;
            }
            float iou = calculateOBBIOU(best, boxes[i]);
            if (iou <= nmsThreshold)
            {
                remaining.push_back(boxes[i]);
            }
            else
            {
                // suppressed
            }
        }
        boxes.swap(remaining);
    }
    return result;
}

void OBBResultSaver::saveResults(const vector<OBBBoundingBox> &boxes,
                                 const string &imagePath,
                                 const string &outputImgDir,
                                 const string &outputTxtDir,
                                 int srcWidth, int srcHeight)
{
    string fileName = Utils::getFileNameWithoutExt(imagePath);
    string outputImagePath = outputImgDir + "/" + fileName + ".jpg";
    string outputTxtPath = outputTxtDir + "/" + fileName + ".txt";

    saveTxtFile(boxes, outputTxtPath, srcWidth, srcHeight);
    saveVisualization(boxes, imagePath, outputImagePath);
}

void OBBResultSaver::saveTxtFile(const vector<OBBBoundingBox> &boxes,
                                 const string &txtPath, int srcWidth, int srcHeight)
{
    ofstream txtFile(txtPath);
    if (!txtFile.is_open())
    {
        ACLLITE_LOG_ERROR("Cannot open output TXT file: %s", txtPath.c_str());
        return;
    }

    for (const auto &box : boxes)
    {
        vector<cv::Point2f> points = box.getCornerPoints();

        txtFile << points[0].x << " " << points[0].y << " "
                << points[1].x << " " << points[1].y << " "
                << points[2].x << " " << points[2].y << " "
                << points[3].x << " " << points[3].y << " "
                << label[box.classIndex] << " "
                << box.confidence << endl;
    }
    txtFile.close();
}

void OBBResultSaver::saveVisualization(const vector<OBBBoundingBox> &boxes,
                                       const string &imagePath,
                                       const string &outputPath)
{
    cv::Mat srcImage = cv::imread(imagePath);
    if (srcImage.empty())
    {
        ACLLITE_LOG_ERROR("Cannot read image: %s", imagePath.c_str());
        return;
    }

    const vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
        cv::Scalar(128, 0, 0), cv::Scalar(0, 128, 0), cv::Scalar(0, 0, 128),
        cv::Scalar(128, 128, 0), cv::Scalar(128, 0, 128), cv::Scalar(0, 128, 128),
        cv::Scalar(64, 64, 64), cv::Scalar(192, 192, 192), cv::Scalar(255, 128, 0)};

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const auto &box = boxes[i];
        vector<cv::Point2f> points = box.getCornerPoints();

        cv::Scalar color = colors[box.classIndex % colors.size()];
        vector<cv::Point> intPoints;
        for (const auto &p : points)
        {
            intPoints.push_back(cv::Point(static_cast<int>(round(p.x)), static_cast<int>(round(p.y))));
        }

        const cv::Point *pts = intPoints.data();
        int npts = static_cast<int>(intPoints.size());
        polylines(srcImage, &pts, &npts, 1, true, color, 2);

        circle(srcImage, Point(static_cast<int>(round(box.cx)), static_cast<int>(round(box.cy))), 3, color, -1);

        string className = (box.classIndex < 5) ? label[box.classIndex] : "unknown";
        char confBuf[32];
        snprintf(confBuf, sizeof(confBuf), "%.3f", box.confidence);
        string markString = className + ":" + confBuf;

        putText(srcImage, markString,
                Point(static_cast<int>(round(box.cx - box.width / 4)),
                      static_cast<int>(round(box.cy - box.height / 4 - 10))),
                FONT_HERSHEY_COMPLEX, 0.5, color, 1);
    }

    imwrite(outputPath, srcImage);
}

YOLOOBBInference::YOLOOBBInference(const InferenceConfig &config) : config_(config)
{
    postProcessor_ = std::make_unique<OBBPostProcessor>(config_.modelOutputBoxNum, config_.classNum);
}

YOLOOBBInference::~YOLOOBBInference()
{
    releaseResources();
}

bool YOLOOBBInference::initialize()
{
    if (aclResource_.Init() != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("ACL resource initialization failed");
        return false;
    }

    if (aclrtGetRunMode(&runMode_) != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Get run mode failed");
        return false;
    }

    if (imageProcess_.Init() != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Image process initialization failed");
        return false;
    }

    if (model_.Init(config_.modelPath.c_str()) != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Model initialization failed");
        return false;
    }

    ACLLITE_LOG_INFO("Model initialized successfully");
    ACLLITE_LOG_INFO("Input size: %dx%d", config_.modelWidth, config_.modelHeight);
    ACLLITE_LOG_INFO("Output boxes: %zu", config_.modelOutputBoxNum);
    ACLLITE_LOG_INFO("Classes: %zu", config_.classNum);
    ACLLITE_LOG_INFO("Using ProbIoU for NMS processing");

    return true;
}

void YOLOOBBInference::runInference()
{
    Utils::createDirectory(config_.outputImgDir);
    Utils::createDirectory(config_.outputTxtDir);

    vector<string> imagePaths = Utils::getImagePaths(config_.inputDir);
    if (imagePaths.empty())
    {
        ACLLITE_LOG_ERROR("No images found in directory: %s", config_.inputDir.c_str());
        return;
    }

    ACLLITE_LOG_INFO("Found %zu images to process", imagePaths.size());

    double totalTime = 0.0;
    size_t processedImages = 0;

    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        auto start = chrono::steady_clock::now();

        if (processImage(imagePaths[i]))
        {
            auto end = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(end - start).count();

            if (i == 0)
            {
                ACLLITE_LOG_INFO("Warmup image processed in %f s, fps: %f", elapsed, 1.0 / elapsed);
            }
            else
            {
                totalTime += elapsed;
                processedImages++;
                ACLLITE_LOG_INFO("Image %zu processed in %f s, fps: %f", i, elapsed, 1.0 / elapsed);
            }
        }
        else
        {
            ACLLITE_LOG_ERROR("Failed to process image: %s", imagePaths[i].c_str());
        }
    }

    if (processedImages > 0)
    {
        double avgFps = processedImages / totalTime;
        ACLLITE_LOG_INFO("Processed %zu images, average FPS: %f", processedImages, avgFps);
    }
}

bool YOLOOBBInference::processImage(const string &imagePath)
{
    if (!preprocessImage(imagePath))
    {
        return false;
    }

    vector<InferenceOutput> inferOutputs;
    if (!runModelInference(inferOutputs))
    {
        return false;
    }

    return postprocessResults(inferOutputs, imagePath);
}

bool YOLOOBBInference::preprocessImage(const string &imagePath)
{
    ImageData image;
    if (ReadJpeg(image, imagePath) != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Failed to read image: %s", imagePath.c_str());
        return false;
    }

    ImageData imageDevice;
    if (CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP) != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Failed to copy image to device");
        return false;
    }

    ImageData yuvImage;
    if (imageProcess_.JpegD(yuvImage, imageDevice) != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Failed to decode JPEG");
        return false;
    }

    if (imageProcess_.Resize(resizedImage_, yuvImage, config_.modelWidth, config_.modelHeight) != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Failed to resize image");
        return false;
    }

    return true;
}

bool YOLOOBBInference::runModelInference(vector<InferenceOutput> &inferOutputs)
{
    if (model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size) != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Failed to create model input");
        return false;
    }

    if (model_.Execute(inferOutputs) != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Failed to execute model");
        return false;
    }

    return true;
}

bool YOLOOBBInference::postprocessResults(vector<InferenceOutput> &inferOutputs, const string &imagePath)
{
    if (inferOutputs.empty())
    {
        ACLLITE_LOG_ERROR("No inference output");
        return false;
    }

    cv::Mat srcImage = cv::imread(imagePath);
    if (srcImage.empty())
    {
        ACLLITE_LOG_ERROR("Cannot read source image: %s", imagePath.c_str());
        return false;
    }

    float *outputData = static_cast<float *>(inferOutputs[0].data.get());

    vector<OBBBoundingBox> boxes = postProcessor_->parseOutput(
        outputData, inferOutputs[0].size,
        srcImage.cols, srcImage.rows,
        config_.modelWidth, config_.modelHeight,
        config_.confidenceThreshold);

    ACLLITE_LOG_INFO("Filtered %zu OBB boxes by confidence threshold", boxes.size());

    vector<OBBBoundingBox> finalBoxes = OBBNMSProcessor::applyNMS(boxes, config_.nmsThreshold);

    ACLLITE_LOG_INFO("Final result: %zu OBB boxes after ProbIoU NMS", finalBoxes.size());

    OBBResultSaver::saveResults(finalBoxes, imagePath, config_.outputImgDir, config_.outputTxtDir,
                                srcImage.cols, srcImage.rows);

    // for (size_t i = 0; i < std::min<size_t>(finalBoxes.size(), 5); ++i)
    // {
    //     ACLLITE_LOG_INFO("Box %zu: cx=%f cy=%f w=%f h=%f angle(rad)=%f angle(deg)=%f conf=%f",
    //                      i, finalBoxes[i].cx, finalBoxes[i].cy, finalBoxes[i].width, finalBoxes[i].height,
    //                      finalBoxes[i].angle, finalBoxes[i].angle * 180.0f / M_PI, finalBoxes[i].confidence);
    // }

    return true;
}

void YOLOOBBInference::releaseResources()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}