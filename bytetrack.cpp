#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <float.h>
#include <stdio.h>

#include "BYTETracker.h"
#include "yolov8.h"

YOLOV8* InitDetector()
{
	const char *modelPath = "yolov8n.om";
    const int32_t modelWidth = 640;
    const int32_t modelHeight = 640;
	YOLOV8* detector = new YOLOV8(modelPath, modelWidth, modelHeight);
    Result ret = detector->InitResource();
	if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return NULL;
    }
	return detector;
}

Result performDetector(YOLOV8*& detector, std::string fileName, std::vector<Object>& objects, bool release=false) {
    std::vector<InferenceOutput> inferOutputs;
	std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
	Result ret = detector->ProcessInput(fileName);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
        return FAILED;
    }
        
	ret = detector->Inference(inferOutputs);
	if (ret == FAILED)
	{
		ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
		return FAILED;
	}
	
	std::vector<BoundBox> result;
	ret = detector->GetResult(inferOutputs, result, fileName, release);
	if (ret == FAILED)
	{
		ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
		return FAILED;
	}
	
	objects.clear();
	for (size_t i = 0; i < result.size(); ++i)
	{
		Object obj;
		obj.label = result[i].classIndex;
		obj.prob = result[i].score;
		obj.rect.x = result[i].x - result[i].width / 2;
		obj.rect.y = result[i].y - result[i].height / 2;
		obj.rect.width = result[i].width;
		obj.rect.height = result[i].height;
		objects.push_back(obj);
	}
	
	std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	ACLLITE_LOG_INFO("Inference elapsed time : %f s , fps is %f", elapsed.count(), 1 / elapsed.count());
  return SUCCESS;
}

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

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [videopath]\n", argv[0]);
        return -1;
    }

    const char* videopath = argv[1];
    YOLOV8* yoloDetector = InitDetector();
  
    BYTETracker tracker(30, 30);
    int num_frames = 0;
    int total_ms = 1;
    int one_ms = 0;
    std::string imagePath;
	bool release;
	for (int no =0; no < 100; no++)
    {
        imagePath = str_format("%s/%06d.jpg", videopath, no);
		num_frames ++;
		if(no == 100)
			release = true;
		else
			release = false;
        
        std::vector<Object> objects;
        //auto start = std::chrono::system_clock::now();
        performDetector(yoloDetector, imagePath, objects, release);
        //std::cout << "Detect objects " << objects.size() << std::endl;

        auto start = std::chrono::system_clock::now();
        std::vector<STrack> output_stracks = tracker.update(objects);
        auto end = std::chrono::system_clock::now();
        one_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Detect objects " << objects.size() << ", Processing one frame takes " << one_ms << " ms" << std::endl;
        total_ms = total_ms + one_ms;
        
        cv::Mat img = cv::imread(imagePath);
        for (int i = 0; i < output_stracks.size(); i++)
		{
			std::vector<float> tlwh = output_stracks[i].tlwh;
			{
                cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
				cv::putText(img, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			}
		}
        cv::putText(img, cv::format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, (int)output_stracks.size()), 
                cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        imagePath = str_format("%s/%06d.jpg", "results", no);
        img = cv::imwrite(imagePath, img);
    }
	delete yoloDetector;
	yoloDetector = NULL;
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;

    return 0;
}
