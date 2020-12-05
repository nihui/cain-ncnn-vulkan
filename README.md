# CAIN ncnn Vulkan

:exclamation: :exclamation: :exclamation: This software is in the early development stage, it may bite your cat

ncnn implementation of CAIN, Channel Attention Is All You Need for Video Frame Interpolation.

cain-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## [Download]

Download Windows/Linux/MacOS Executable for Intel/AMD/Nvidia GPU

**https://github.com/nihui/cain-ncnn-vulkan/actions**

This package includes all the binaries and models required. It is portable, so no CUDA or PyTorch runtime environment is needed :)

## About CAIN

CAIN (Channel Attention Is All You Need for Video Frame Interpolation) (AAAI 2020)

https://github.com/myungsub/CAIN

Myungsub Choi, Heewon Kim, Bohyung Han, Ning Xu, Kyoung Mu Lee

2nd place in [[AIM 2019 ICCV Workshop](http://www.vision.ee.ethz.ch/aim19/)] - Video Temporal Super-Resolution Challenge

[Project](https://myungsub.github.io/CAIN) | [Paper-AAAI](https://aaai.org/Papers/AAAI/2020GB/AAAI-ChoiM.4773.pdf) (Download the paper [[here](https://www.dropbox.com/s/b62wnroqdd5lhfc/AAAI-ChoiM.4773.pdf?dl=0)] in case the AAAI link is broken) | [Poster](https://www.dropbox.com/s/7lxwka16qkuacvh/AAAI-ChoiM.4773.pdf)

## Usages

Input two frame images, output one interpolated frame image.

### Example Command

```shell
./cain-ncnn-vulkan -0 0.jpg -1 1.jpg -o 01.jpg
./cain-ncnn-vulkan -i input_frames/ -o output_frames/
```

### Video Interpolation with FFmpeg

```shell
mkdir input_frames
mkdir output_frames

# find the source fps and format with ffprobe, for example 24fps, AAC
ffprobe input.mp4

# extract audio
ffmpeg -i input.mp4 -vn -acodec copy audio.m4a

# decode all frames
ffmpeg -i input.mp4 input_frames/frame_%06d.png

# interpolate 2x frame count
./cain-ncnn-vulkan -i input_frames -o output_frames

# encode interpolated frames in 48fps with audio
ffmpeg -framerate 48 -i output_frames/%06d.png -i audio.m4a -c:a copy -crf 20 -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Full Usages

```console
Usage: cain-ncnn-vulkan -0 infile -1 infile1 -o outfile [options]...
       cain-ncnn-vulkan -i indir -o outdir [options]...

  -h                   show this help
  -v                   verbose output
  -0 input0-path       input image0 path (jpg/png/webp)
  -1 input1-path       input image1 path (jpg/png/webp)
  -i input-path        input image directory (jpg/png/webp)
  -o output-path       output image path (jpg/png/webp) or directory
  -m model-path        dain model path (default=cain)
  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu
  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu
  -f pattern-format    output image filename pattern format (%08d.jpg/png/webp, default=ext/%08d.png)
```

- `input0-path`, `input1-path` and `output-path` accept file path
- `input-path` and `output-path` accept file directory
- `load:proc:save` = thread count for the three stages (image decoding + cain interpolation + image encoding), using larger values may increase GPU usage and consume more GPU memory. You can tune this configuration with "4:4:4" for many small-size images, and "2:2:2" for large-size images. The default setting usually works fine for most situations. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.
- `format` = the format of the image to be output, png is better supported, however webp generally yields smaller file sizes, both are losslessly encoded

If you encounter a crash or error, try upgrading your GPU driver:

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Build from Source

1. Download and setup the Vulkan SDK from https://vulkan.lunarg.com/
  - For Linux distributions, you can either get the essential build requirements from package manager
```shell
dnf install vulkan-headers vulkan-loader-devel
```
```shell
apt-get install libvulkan-dev
```
```shell
pacman -S vulkan-headers vulkan-icd-loader
```

2. Clone this project with all submodules

```shell
git clone https://github.com/nihui/cain-ncnn-vulkan.git
cd cain-ncnn-vulkan
git submodule update --init --recursive
```

3. Build with CMake
  - You can pass -DUSE_STATIC_MOLTENVK=ON option to avoid linking the vulkan loader library on MacOS

```shell
mkdir build
cd build
cmake ../src
cmake --build . -j 4
```

### TODO

* test-time sptial augmentation aka TTA-s
* test-time temporal augmentation aka TTA-t

## Original CAIN Project

- https://github.com/myungsub/CAIN

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
