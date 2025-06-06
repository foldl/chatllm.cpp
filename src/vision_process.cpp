#include "vision_process.h"
#include <regex>
#include <sstream>
#include <cmath>

#if defined(_MSC_VER)
#define popen  _popen
#define pclose _pclose
#endif

namespace vision
{
    struct Params
    {
        bool do_resize;
        int resize_width;
        int resize_height;

        int merge_kernel_size[2];

        int max_grid_width;
        int max_grid_height;
        int max_patch_num;
    };

    static Params params {
        .do_resize      = false,
        .resize_width   = 0,
        .resize_height  = 0,
        .merge_kernel_size = {0, 0},

        .max_grid_width     = 0,
        .max_grid_height    = 0,
        .max_patch_num      = 0,
    };

    MaxGridWidth::MaxGridWidth(int value)
    {
        params.max_grid_width = value;
    }

    MaxGridWidth::~MaxGridWidth()
    {
        params.max_grid_width = 0;
    }

    MaxGridHeight::MaxGridHeight(int value)
    {
        params.max_grid_height = value;
    }

    MaxGridHeight::~MaxGridHeight()
    {
        params.max_grid_height = 0;
    }

    MaxPatchNum::MaxPatchNum(int value)
    {
        params.max_patch_num = value;
    }

    MaxPatchNum::~MaxPatchNum()
    {
        params.max_patch_num = 0;
    }

    Resize::Resize(int width, int height)
    {
        params.do_resize = true;
        params.resize_height = height;
        params.resize_width  = width;
    }

    Resize::~Resize(void)
    {
        params.do_resize = false;
    }

    MergeKernel::MergeKernel(int size0, int size1)
    {
        params.merge_kernel_size[0] = size0;
        params.merge_kernel_size[1] = size1;
    }

    MergeKernel::MergeKernel(const int *sizes)
        : MergeKernel(sizes[0], sizes[1])
    {

    }

    MergeKernel::~MergeKernel()
    {
        params.merge_kernel_size[0] = 0;
        params.merge_kernel_size[1] = 0;
    }

    void image_dimension(const char *fn, int &width, int &height)
    {
        width  = -1;
        height = -1;
        const static std::regex rgx(R""( ([0-9]+)x([0-9]+) )"");
        std::smatch matches;

        std::string cmd = "magick identify \"" + std::string(fn) + "\"";

        FILE* pp = popen(cmd.c_str(), "r");

        while (!feof(pp))
        {
            char buffer[1024];
            if (NULL == fgets(buffer, sizeof(buffer) - 1, pp)) continue;

            std::string s(buffer);

            if (std::regex_search(s, matches, rgx) && (matches.size() == 3))
            {
                width  = std::stoi(matches[1].str());
                height = std::stoi(matches[2].str());
                break;
            }
        }

        pclose(pp);
    }

    void image_load(const char *fn, std::vector<uint8_t> &rgb_pixels, int &width, int &height, int patch_size, PaddingMode pad)
    {
        // magick -depth 8 demo.jpeg -resize 100x100 rgb:"aaa.raw"
        rgb_pixels.clear();
        width  = -1;
        height = -1;
        image_dimension(fn, width, height);
        if (width <= 0) return;

        std::ostringstream oss;
        oss << "magick -depth 8 \"" << std::string(fn) << "\"";

        if (params.do_resize)
        {
            width = params.resize_width;
            height = params.resize_height;
        }
        else
        {
            if (params.max_grid_width > 0)
            {
                int new_w = std::min(width, patch_size * params.max_grid_width);
                double ratio = (double)new_w / width;
                height = (int)(height * ratio);
                width  = new_w;
            }

            if (params.max_grid_height > 0)
            {
                int new_h = std::min(height, patch_size * params.max_grid_height);
                double ratio = (double)new_h / height;
                width  = (int)(width * ratio);
                height = new_h;
            }

            if (params.max_patch_num > 0)
            {
                int patch_num = (width / patch_size) * (height / patch_size);
                if (patch_num > params.max_patch_num)
                {
                    double ratio = std::sqrt((double)params.max_patch_num / patch_num);
                    height = patch_size * (int)floor(((height / patch_size) * ratio));
                    width  = patch_size * (int)floor(((width  / patch_size) * ratio));
                }
            }
        }

        oss << " -resize " << width << "x" << height << "!";

        int aligned_width  = 0;
        int aligned_height = 0;
        const int patch_size_w = params.merge_kernel_size[0] > 0 ? patch_size * params.merge_kernel_size[0] : patch_size;
        const int patch_size_h = params.merge_kernel_size[1] > 0 ? patch_size * params.merge_kernel_size[1] : patch_size;

        if (pad != PaddingMode::No)
        {
            aligned_width  = ((width  + patch_size_w - 1) / patch_size_w) * patch_size_w;
            aligned_height = ((height + patch_size_h - 1) / patch_size_h) * patch_size_h;

            oss << " -background ";
            switch (pad)
            {
            case PaddingMode::Black:
                oss << "black";
                break;

            default:
                oss << "white";
                break;
            }
            oss << " -compose Copy -gravity northwest -extent " << aligned_width << "x" << aligned_height;
        }
        else
        {
            aligned_width  = (width  / patch_size_w) * patch_size_w;
            aligned_height = (height / patch_size_h) * patch_size_h;

            oss << " -crop " << aligned_width << "x" << aligned_height << "+" << (width - aligned_width) / 2 << "+" << (height - aligned_height) / 2;
        }

        oss << " rgb:-";
        // oss << " rgb:c:\\tmp\\img.raw";

        //printf("%s\n", oss.str().c_str());

        FILE* pp = popen(oss.str().c_str(), "rb");

        while (!feof(pp))
        {
            uint8_t buffer[1024];
            size_t cnt = fread(buffer, 1, sizeof(buffer), pp);
            if (cnt > 0)
                rgb_pixels.insert(rgb_pixels.end(), buffer, buffer + cnt);
        }

        pclose(pp);

        width  = aligned_width;
        height = aligned_height;
    }

    void image_rescale(const std::vector<uint8_t> &rgb_pixels, std::vector<float> &scaled_rgb_pixels, float scale_factor)
    {
        scaled_rgb_pixels.resize(rgb_pixels.size());
        for (size_t i = 0; i < rgb_pixels.size(); i++)
            scaled_rgb_pixels[i] = scale_factor * rgb_pixels[i];
    }

    void image_normalize(std::vector<float> &rgb_pixels, const float *mean, const float *std_d)
    {
        const size_t pixels_num = rgb_pixels.size() / 3;
        for (int i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < pixels_num; j++)
            {
                rgb_pixels[j * 3 + i] = (rgb_pixels[j * 3 + i] - mean[i]) / std_d[i];
            }
        }
    }

    void image_arrange(const std::vector<float> &rgb_pixels, const int width, const int patch_size,
        std::vector<float> &arranged, const PatchesFormat fmt)
    {
        const size_t pixels_num = rgb_pixels.size() / 3;
        const int height = (int)(pixels_num / width);
        arranged.resize(rgb_pixels.size(), 0);
        switch (fmt)
        {
        case PatchesFormat::PatchesLeftRightDown_ChannelsRGB_PixelsLeftRightDown:
            {
                int64_t idx = 0;
                for (int i = 0; i < height - patch_size + 1; i += patch_size)
                {
                    for (int j = 0; j < width - patch_size + 1; j += patch_size)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            for (int k = 0; k < patch_size; k++)
                            {
                                for (int l = 0; l < patch_size; l++)
                                {
                                    int pixel_id = (i + k) * width + (j + l);
                                    arranged[idx++] = rgb_pixels[pixel_id * 3 + c];
                                }
                            }
                        }
                    }
                }
            }
            break;
        case PatchesLeftRightDown_PixelsLeftRightDown_ChannelsRGB:
            {
                int64_t idx = 0;
                for (int i = 0; i < height - patch_size + 1; i += patch_size)
                {
                    for (int j = 0; j < width - patch_size + 1; j += patch_size)
                    {
                        for (int k = 0; k < patch_size; k++)
                        {
                            for (int l = 0; l < patch_size; l++)
                            {
                                const int pixel_id = (i + k) * width + (j + l);

                                for (int c = 0; c < 3; c++)
                                {
                                    arranged[idx++] = rgb_pixels[pixel_id * 3 + c];
                                }
                            }
                        }
                    }
                }
            }
            break;
        case ChannelsRGB_PatchesLeftRightDown_PixelsLeftRightDown:
            {
                int64_t idx = 0;
                for (int c = 0; c < 3; c++)
                {
                    for (int i = 0; i < height - patch_size + 1; i += patch_size)
                    {
                        for (int j = 0; j < width - patch_size + 1; j += patch_size)
                        {
                            for (int k = 0; k < patch_size; k++)
                            {
                                for (int l = 0; l < patch_size; l++)
                                {
                                    const int pixel_id = (i + k) * width + (j + l);
                                    arranged[idx++] = rgb_pixels[pixel_id * 3 + c];
                                }
                            }
                        }
                    }
                }
            }
            break;
        case ChannelsRGB_PixelsLeftRightDown:
            {
                int64_t idx = 0;
                for (int c = 0; c < 3; c++)
                {
                    for (int i = 0; i < height; i++)
                    {
                        for (int j = 0; j < width; j++)
                        {
                            const int pixel_id = i * width + j;
                            arranged[idx++] = rgb_pixels[pixel_id * 3 + c];
                        }
                    }
                }
            }
            break;
        default:
            throw new std::invalid_argument("invalid format");
        }
    }

    static void print_data(const std::vector<float> &pixels, const int group_size = 10, int max_elem = 100)
    {
        const size_t cnt = pixels.size();
        for (size_t j = 0; (j < cnt) && (j < (size_t)max_elem); j++)
        {
            if ((j % group_size) == 0)
            {
                printf("\n%3zd: ", j);
            }
            printf("%+2.4f ", pixels[j]);
        }

        printf("\n");
    }

    static void print_rgb_pixels(const std::vector<float> &pixels)
    {
        const size_t cnt = pixels.size() / 3;
        for (size_t j = 0; j < cnt; j++)
        {
            printf("%3zd: (%.4f, %.4f, %.4f)\n", j, pixels[j * 3 + 0], pixels[j * 3 + 1], pixels[j * 3 + 2]);
            if (j > 10) break;
        }
        printf("\n");
    }

    void test0(const char *fn)
    {
        int w, h;
        std::vector<uint8_t> pixels;

        MaxGridWidth param1(16);

        vision::image_load(fn, pixels, w, h, 14);
        printf("%dx%d = %zd, %d\n", w, h, pixels.size() / 3, w * h == pixels.size() / 3);

        std::vector<float> scaled;
        image_rescale(pixels, scaled);

        print_rgb_pixels(scaled);

        const float means[] = {0.48145466f,
            0.4578275f,
            0.40821073f};
        const float std_d[] = {0.26862954f,
            0.26130258f,
            0.27577711f};

        image_normalize(scaled, means, std_d);

        print_rgb_pixels(scaled);
    }

    void test(const char *fn)
    {
        int w, h;
        std::vector<uint8_t> pixels;
        const int merge_size[] = {2, 2};
        const int patch_size = 14;
        MergeKernel merge(merge_size[0], merge_size[1]);
        MaxPatchNum param(4096);

        vision::image_load(fn, pixels, w, h, patch_size, PaddingMode::Black);

        printf("%dx%d = %zd, %d\n", w, h, pixels.size() / 3, w * h == (int)(pixels.size() / 3));

        std::vector<float> scaled;
        image_rescale(pixels, scaled);

        const float means[] = {0.5f, 0.5f, 0.5f};
        const float std_d[] = {0.5f, 0.5f, 0.5f};

        image_normalize(scaled, means, std_d);

        print_rgb_pixels(scaled);

        std::vector<float> arranged;

        image_arrange(scaled, w, patch_size, arranged, PatchesFormat::PatchesLeftRightDown_ChannelsRGB_PixelsLeftRightDown);

        print_data(arranged, patch_size, patch_size * patch_size * 4);
    }
}
