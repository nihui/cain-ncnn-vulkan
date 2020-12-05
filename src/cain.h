// cain implemented with ncnn library

#ifndef CAIN_H
#define CAIN_H

#include <string>

// ncnn
#include "net.h"

class CAIN
{
public:
    CAIN(int gpuid);
    ~CAIN();

#if _WIN32
    int load(const std::wstring& modeldir);
#else
    int load(const std::string& modeldir);
#endif

    int process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net cainnet;
    ncnn::Pipeline* cain_preproc;
    ncnn::Pipeline* cain_postproc;
};

#endif // CAIN_H
