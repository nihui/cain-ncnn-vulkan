// cain implemented with ncnn library

#include "cain.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "cain_preproc.comp.hex.h"
#include "cain_postproc.comp.hex.h"

CAIN::CAIN(int gpuid)
{
    tilesize = 512;
    prepadding = 64;

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

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

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

    // each tile 100x100
    const int xtiles = (w_padded + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h_padded + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

//     fprintf(stderr, "tiles %d %d\n", xtiles, ytiles);

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);

//         fprintf(stderr, "in_tile_y0 %d %d\n", in_tile_y0, in_tile_y1);

        ncnn::Mat in0;
        ncnn::Mat in1;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            in0 = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixel0data + in_tile_y0 * w * channels, (size_t)channels, 1);
            in1 = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixel1data + in_tile_y0 * w * channels, (size_t)channels, 1);
        }
        else
        {
#if _WIN32
            in0 = ncnn::Mat::from_pixels(pixel0data + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
            in1 = ncnn::Mat::from_pixels(pixel1data + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
#else
            in0 = ncnn::Mat::from_pixels(pixel0data + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
            in1 = ncnn::Mat::from_pixels(pixel1data + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
#endif
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in0_gpu;
        ncnn::VkMat in1_gpu;
        {
            cmd.record_clone(in0, in0_gpu, opt);
            cmd.record_clone(in1, in1_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = yi * TILE_SIZE_Y;
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, (out_tile_y1 - out_tile_y0), (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, (out_tile_y1 - out_tile_y0), channels, (size_t)4u, 1, blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++)
        {
            // preproc
            ncnn::VkMat in0_tile_gpu;
            ncnn::VkMat in1_tile_gpu;
            {
                // crop tile
                int tile_x0 = xi * TILE_SIZE_X - prepadding;
                int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w_padded) + prepadding;
                int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h_padded) + prepadding;

                in0_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = in0_gpu;
                bindings[1] = in0_tile_gpu;

                std::vector<ncnn::vk_constant_type> constants(12);
                constants[0].i = in0_gpu.w;
                constants[1].i = in0_gpu.h;
                constants[2].i = in0_gpu.cstep;
                constants[3].i = in0_tile_gpu.w;
                constants[4].i = in0_tile_gpu.h;
                constants[5].i = in0_tile_gpu.cstep;
#if _WIN32
                constants[6].f = mean_rgb0[2];
                constants[7].f = mean_rgb0[1];
                constants[8].f = mean_rgb0[0];
#else
                constants[6].f = mean_rgb0[0];
                constants[7].f = mean_rgb0[1];
                constants[8].f = mean_rgb0[2];
#endif
                constants[9].i = prepadding;
                constants[10].i = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                constants[11].i = xi * TILE_SIZE_X;

                cmd.record_pipeline(cain_preproc, bindings, constants, in0_tile_gpu);
            }
            {
                // crop tile
                int tile_x0 = xi * TILE_SIZE_X - prepadding;
                int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w_padded) + prepadding;
                int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h_padded) + prepadding;

                in1_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = in1_gpu;
                bindings[1] = in1_tile_gpu;

                std::vector<ncnn::vk_constant_type> constants(12);
                constants[0].i = in1_gpu.w;
                constants[1].i = in1_gpu.h;
                constants[2].i = in1_gpu.cstep;
                constants[3].i = in1_tile_gpu.w;
                constants[4].i = in1_tile_gpu.h;
                constants[5].i = in1_tile_gpu.cstep;
#if _WIN32
                constants[6].f = mean_rgb1[2];
                constants[7].f = mean_rgb1[1];
                constants[8].f = mean_rgb1[0];
#else
                constants[6].f = mean_rgb1[0];
                constants[7].f = mean_rgb1[1];
                constants[8].f = mean_rgb1[2];
#endif
                constants[9].i = prepadding;
                constants[10].i = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                constants[11].i = xi * TILE_SIZE_X;

                cmd.record_pipeline(cain_preproc, bindings, constants, in1_tile_gpu);
            }

//             fprintf(stderr, "in0_tile_gpu %d %d\n", in0_tile_gpu.w, in0_tile_gpu.h);

            // cainnet
            ncnn::VkMat out_gpu_padded;
            {
                ncnn::Extractor ex = cainnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("x.1", in0_tile_gpu);
                ex.input("x.3", in1_tile_gpu);

                // save some memory
                in0_tile_gpu.release();
                in1_tile_gpu.release();

                ex.extract("4070", out_gpu_padded, cmd);
            }

            // postproc
            {
                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = out_gpu_padded;
                bindings[1] = out_gpu;

                std::vector<ncnn::vk_constant_type> constants(12);
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
                constants[9].i = prepadding;
                constants[10].i = prepadding;
                constants[11].i = xi * TILE_SIZE_X;

                ncnn::VkMat dispatcher;
                dispatcher.w = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
                dispatcher.h = out_gpu.h;
                dispatcher.c = 3;

                cmd.record_pipeline(cain_postproc, bindings, constants, dispatcher);
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }

//             fprintf(stderr, "%.2f%%\n", (float)(yi * xtiles + xi) / (ytiles * xtiles) * 100);
        }

        // download
        {
            ncnn::Mat out;

            if (opt.use_fp16_storage && opt.use_int8_storage)
            {
                out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data + out_tile_y0 * w * channels, (size_t)channels, 1);
            }

            cmd.record_clone(out_gpu, out, opt);

            cmd.submit_and_wait();

            if (!(opt.use_fp16_storage && opt.use_int8_storage))
            {
#if _WIN32
                out.to_pixels((unsigned char*)outimage.data + out_tile_y0 * w * channels, ncnn::Mat::PIXEL_RGB2BGR);
#else
                out.to_pixels((unsigned char*)outimage.data + out_tile_y0 * w * channels, ncnn::Mat::PIXEL_RGB);
#endif
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
