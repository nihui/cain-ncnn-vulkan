// cain implemented with ncnn library

#include "cain.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "cain_preproc.comp.hex.h"
#include "cain_postproc.comp.hex.h"

CAIN::CAIN(int gpuid)
{
    vkdev = ncnn::get_gpu_device(gpuid);
    cain_preproc = 0;
    cain_postproc = 0;
}

CAIN::~CAIN()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete cain_preproc;
        delete cain_postproc;
    }
}

int CAIN::load()
{
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;

    cainnet.opt = opt;

    cainnet.set_vulkan_device(vkdev);

    cainnet.load_param("cain.param");
    cainnet.load_model("cain.bin");

    // initialize preprocess and postprocess pipeline
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    compile_spirv_module(cain_preproc_comp_data, sizeof(cain_preproc_comp_data), opt, spirv);
                }
            }

            cain_preproc = new ncnn::Pipeline(vkdev);
            cain_preproc->set_optimal_local_size_xyz(8, 8, 3);
            cain_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    compile_spirv_module(cain_postproc_comp_data, sizeof(cain_postproc_comp_data), opt, spirv);
                }
            }

            cain_postproc = new ncnn::Pipeline(vkdev);
            cain_postproc->set_optimal_local_size_xyz(8, 8, 3);
            cain_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

static void image_mean(const ncnn::Mat& image, float mean_rgb[3])
{
    const unsigned char* pixeldata = (const unsigned char*)image.data;
    const int w = image.w;
    const int h = image.h;
    const int size = w * h;

    float mean_r = 0.f;
    float mean_g = 0.f;
    float mean_b = 0.f;
    for (int i = 0; i < size; i++)
    {
        mean_r += pixeldata[0];
        mean_g += pixeldata[1];
        mean_b += pixeldata[2];

        pixeldata += 3;
    }

    mean_rgb[0] = mean_r / size / 255.f;
    mean_rgb[1] = mean_g / size / 255.f;
    mean_rgb[2] = mean_b / size / 255.f;
}

int CAIN::process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

    float mean_rgb0[3];
    float mean_rgb1[3];
    image_mean(in0image, mean_rgb0);
    image_mean(in1image, mean_rgb1);

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = cainnet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    if (opt.use_fp16_storage && opt.use_int8_storage)
    {
        in0 = ncnn::Mat(w, h, (unsigned char*)pixel0data, (size_t)channels, 1);
        in1 = ncnn::Mat(w, h, (unsigned char*)pixel1data, (size_t)channels, 1);
    }
    else
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
    }

    // preproc
    ncnn::VkMat in0_gpu_padded;
    ncnn::VkMat in1_gpu_padded;
    {
        in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = in0_gpu;
        bindings[1] = in0_gpu_padded;

        std::vector<ncnn::vk_constant_type> constants(9);
        constants[0].i = in0_gpu.w;
        constants[1].i = in0_gpu.h;
        constants[2].i = in0_gpu.cstep;
        constants[3].i = in0_gpu_padded.w;
        constants[4].i = in0_gpu_padded.h;
        constants[5].i = in0_gpu_padded.cstep;
#if _WIN32
        constants[6].f = mean_rgb0[2];
        constants[7].f = mean_rgb0[1];
        constants[8].f = mean_rgb0[0];
#else
        constants[6].f = mean_rgb0[0];
        constants[7].f = mean_rgb0[1];
        constants[8].f = mean_rgb0[2];
#endif

        cmd.record_pipeline(cain_preproc, bindings, constants, in0_gpu_padded);
    }
    {
        in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = in1_gpu;
        bindings[1] = in1_gpu_padded;

        std::vector<ncnn::vk_constant_type> constants(9);
        constants[0].i = in1_gpu.w;
        constants[1].i = in1_gpu.h;
        constants[2].i = in1_gpu.cstep;
        constants[3].i = in1_gpu_padded.w;
        constants[4].i = in1_gpu_padded.h;
        constants[5].i = in1_gpu_padded.cstep;
#if _WIN32
        constants[6].f = mean_rgb1[2];
        constants[7].f = mean_rgb1[1];
        constants[8].f = mean_rgb1[0];
#else
        constants[6].f = mean_rgb1[0];
        constants[7].f = mean_rgb1[1];
        constants[8].f = mean_rgb1[2];
#endif

        cmd.record_pipeline(cain_preproc, bindings, constants, in1_gpu_padded);
    }

    // cainnet
    ncnn::VkMat out_gpu_padded;
    {
        ncnn::Extractor ex = cainnet.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        ex.input("x.1", in0_gpu_padded);
        ex.input("x.3", in1_gpu_padded);

        // save some memory
        in0_gpu_padded.release();
        in1_gpu_padded.release();

        ex.extract("4070", out_gpu_padded, cmd);
    }

    ncnn::VkMat out_gpu;
    if (opt.use_fp16_storage && opt.use_int8_storage)
    {
        out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
    }
    else
    {
        out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
    }

    // postproc
    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = out_gpu_padded;
        bindings[1] = out_gpu;

        std::vector<ncnn::vk_constant_type> constants(9);
        constants[0].i = out_gpu_padded.w;
        constants[1].i = out_gpu_padded.h;
        constants[2].i = out_gpu_padded.cstep;
        constants[3].i = out_gpu.w;
        constants[4].i = out_gpu.h;
        constants[5].i = out_gpu.cstep;
#if _WIN32
        constants[6].f = (mean_rgb0[2] + mean_rgb1[2]) / 2.f;
        constants[7].f = (mean_rgb0[1] + mean_rgb1[1]) / 2.f;
        constants[8].f = (mean_rgb0[0] + mean_rgb1[0]) / 2.f;
#else
        constants[6].f = (mean_rgb0[0] + mean_rgb1[0]) / 2.f;
        constants[7].f = (mean_rgb0[1] + mean_rgb1[1]) / 2.f;
        constants[8].f = (mean_rgb0[2] + mean_rgb1[2]) / 2.f;
#endif

        cmd.record_pipeline(cain_postproc, bindings, constants, out_gpu);
    }

    // download
    {
        ncnn::Mat out;

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data, (size_t)channels, 1);
        }

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        if (!(opt.use_fp16_storage && opt.use_int8_storage))
        {
#if _WIN32
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
