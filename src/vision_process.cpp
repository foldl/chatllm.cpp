#include "vision_process.h"
#include <regex>
#include <sstream>
#include <cmath>
#include <filesystem>
#include "basics.h"

namespace vision
{
    namespace fs = std::filesystem;

    struct Params
    {
        int pre_max_width;
        int pre_max_height;

        bool do_resize;
        int resize_width;
        int resize_height;

        int merge_kernel_size[2];

        int max_grid_width;
        int max_grid_height;
        int max_patch_num;

        int64_t min_pixels;
        int64_t max_pixels;
    };

    static Params params {
        .pre_max_width  = -1,
        .pre_max_height = -1,
        .do_resize      = false,
        .resize_width   = 0,
        .resize_height  = 0,
        .merge_kernel_size = {0, 0},

        .max_grid_width     = 0,
        .max_grid_height    = 0,
        .max_patch_num      = 0,

        .min_pixels         = 0,
        .max_pixels         = 0,
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

    PreMaxImageSize::PreMaxImageSize(int width, int height)
    {
        params.pre_max_width  = width;
        params.pre_max_height = height;
    }

    PreMaxImageSize::~PreMaxImageSize()
    {
        params.pre_max_width  = -1;
        params.pre_max_height = -1;
    }

    MinMaxPixels::MinMaxPixels(int64_t min_pixels, int64_t max_pixels)
    {
        params.min_pixels  = min_pixels;
        params.max_pixels  = max_pixels;
    }

    MinMaxPixels::~MinMaxPixels()
    {
        params.min_pixels  = 0;
        params.max_pixels  = 0;
    }

    bool PreMaxImageSize::PreScale(int &width, int &height)
    {
        double ratio = (double)width / height;
        int target_w = width;
        int target_h = height;
        if ((params.pre_max_width > 0) && (width > params.pre_max_width))
            target_w = params.pre_max_width;
        if ((params.pre_max_height > 0) && (height > params.pre_max_height))
            target_h = params.pre_max_height;
        if ((target_h != height) || (target_w != width)) return false;

        if ((double)width / target_w > (double)height / target_h)
        {
            target_h = (int)(target_w  / ratio);
        }
        else
        {
            target_w = (int)(target_h * ratio);
        }

        bool r = (target_h != height) || (target_w != width);
        height = target_h;
        width  = target_w;
        return r;
    }

#if defined(_WIN64)
    #define MAGICK_CONVERT_CMD      "magick"
    #define MAGICK_IDENTIFY_CMD     "magick identify"
    #define POPEN_MODE_READ         "rb"
    #define POPEN_MODE_WRITE        "wb"
#else
    #define MAGICK_CONVERT_CMD      "convert"
    #define MAGICK_IDENTIFY_CMD     "identify"
    #define MAGICK_DISPLAY_CMD      "display"
    #define POPEN_MODE_READ         "r"
    #define POPEN_MODE_WRITE        "w"
#endif

    void image_dimension(const char *fn, int &width, int &height)
    {
        width  = -1;
        height = -1;
        const static std::regex rgx(R""( ([0-9]+)x([0-9]+) )"");
        std::smatch matches;

        std::string cmd = MAGICK_IDENTIFY_CMD " \"" + std::string(fn) + "\"";

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

    static void run_cmd(std::ostringstream &oss, image_pixels_t &image)
    {
        oss << " rgb:-";
        // oss << " rgb:c:\\tmp\\img.raw";

        //printf("%s\n", oss.str().c_str());

        FILE* pp = popen(oss.str().c_str(), POPEN_MODE_READ);

        while (!feof(pp))
        {
            uint8_t buffer[1024];
            size_t cnt = fread(buffer, 1, sizeof(buffer), pp);
            if (cnt > 0)
                image.insert(image.end(), buffer, buffer + cnt);
        }

        pclose(pp);
    }

    void image_load_split(const char *fn, std::vector<image_pixels_t> &splits, bool do_split, const int split_width, const int split_height, int &splits_cols_num, int &splits_rows_num)
    {
        splits.clear();
        splits_cols_num = 0;
        splits_rows_num = 0;

        int width = -1;
        int height = -1;
        image_dimension(fn, width, height);
        if (width <= 0) return;

        std::ostringstream base;
        base << MAGICK_CONVERT_CMD << " -depth 8 \"" << std::string(fn) << "\"";
        if (PreMaxImageSize::PreScale(width, height))
            base << " -resize " << width << "x" << height << "!";

        splits_rows_num =(height + split_height - 1) / split_height;
        splits_cols_num =(width  + split_width  - 1) / split_width;
        if (do_split && ((splits_rows_num > 1) || (splits_cols_num > 1)))
        {
            const int optimal_height = (height + splits_rows_num - 1) / splits_rows_num;
            const int optimal_width  = (width  + splits_cols_num - 1) / splits_cols_num;

            for (int r = 0; r < splits_rows_num; r++)
            {
                for (int c = 0; c < splits_cols_num; c++)
                {
                    const int start_x = c * optimal_width;
                    const int start_y = r * optimal_height;
                    const int end_x   = std::min(start_x + optimal_width,  width);
                    const int end_y   = std::min(start_y + optimal_height, height);

                    std::ostringstream oss;
                    oss << base.str();
                    oss << " -crop " << optimal_width << "x" << optimal_height << "+" << start_x << "+" << start_y;

                    splits.emplace_back(image_pixels_t());
                    auto &image = splits.back();
                    run_cmd(oss, image);
                }
            }
        }
        else
        {
            splits_rows_num = 0;
            splits_cols_num = 0;
        }

        // resize the global one
        std::ostringstream oss;
        oss << MAGICK_CONVERT_CMD << " -depth 8 \"" << std::string(fn) << "\"";
        oss << " -resize " << split_width << "x" << split_height << "!";
        splits.emplace_back(image_pixels_t());
        auto &image = splits.back();
        run_cmd(oss, image);
    }

    void image_load_pan_and_scan(const char *fn, std::vector<image_pixels_t> &crops, bool do_pas,
        const int min_crop_size, const int max_num_crops, float min_ratio_to_activate,
        const int crop_width, const int crop_height,
        PanScanDir &dir)
    {
        crops.clear();

        int width = -1;
        int height = -1;
        image_dimension(fn, width, height);
        if (width <= 0) return;

        // whole image
        {
            std::ostringstream oss;
            oss << MAGICK_CONVERT_CMD << " -depth 8 \"" << std::string(fn) << "\"";
            oss << " -resize " << crop_width << "x" << crop_height << "!";
            crops.emplace_back(image_pixels_t());
            auto &image = crops.back();
            run_cmd(oss, image);
        }

        int num_crops_w = 1;
        int num_crops_h = 1;

        if (width >= height)
        {
            auto ratio = (float)width / height;
            if (ratio < min_ratio_to_activate)
                return;
            dir = PanScanDir::Horizontal;

            num_crops_w = int(width / height + 0.5);
            num_crops_w = std::min((width / min_crop_size), num_crops_w);

            num_crops_w = std::max(2, num_crops_w);
            num_crops_w = std::min(max_num_crops, num_crops_w);
        }
        else
        {
            auto ratio = (float)height / width;
            if (ratio < min_ratio_to_activate)
                return;
            dir = PanScanDir::Vertical;

            num_crops_h = int(height / width  + 0.5);
            num_crops_h = std::min((height / min_crop_size), num_crops_h);

            num_crops_h = std::max(2, num_crops_h);
            num_crops_h = std::min(max_num_crops, num_crops_h);
        }

        const int crop_size_w = (width  + (num_crops_w - 1) / num_crops_w);
        const int crop_size_h = (height + (num_crops_h - 1) / num_crops_h);

        if (std::min(crop_size_w, crop_size_h) < min_crop_size)
            return;

        for (int r = 0; r < num_crops_h; r++)
        {
            const int start_y = r * crop_size_h;

            for (int c = 0; c < num_crops_w; c++)
            {
                const int start_x = c * crop_size_w;

                std::ostringstream oss;
                oss << MAGICK_CONVERT_CMD << " -depth 8 \"" << std::string(fn) << "\"";
                oss << " -crop " << crop_size_w << "x" << crop_size_h << "+" << start_x << "+" << start_y;
                oss << " -resize " << crop_width << "x" << crop_height << "!";

                crops.emplace_back(image_pixels_t());
                auto &image = crops.back();
                run_cmd(oss, image);
            }
        }
    }

    void image_load_pan_and_scan(const char *fn, std::vector<image_pixels_t> &crops,
        const int image_size, int crop_size, int &crops_per_row)
    {
        const int MAX_IMAGE_SIZE = 3024;
        crops.clear();
        crops_per_row = 0;

        int width = -1;
        int height = -1;
        image_dimension(fn, width, height);
        if (width <= 0) return;

        auto get_image_size_for_padding = [](int img_width, int img_height)
        {
            const double ratio = (double)img_width / img_height;
            if ((std::min(img_height, img_width) < 32) && ((ratio > 4.0 || (ratio < 1.0 / 4))))
            {
                const int new_size = std::max(img_height, img_width);
                return std::pair<int, int>(new_size, new_size);
            }
            return std::pair<int, int>(img_width, img_height);
        };

        auto get_image_size_for_preprocess = [](int img_width, int img_height)
        {
            if (std::max(img_height, img_width) > MAX_IMAGE_SIZE)
            {
                double scale_factor = (double)MAX_IMAGE_SIZE / std::max(img_height, img_width);
                img_width = int(img_width * scale_factor);
                img_height = int(img_height * scale_factor);

            }
            return std::pair<int, int>(img_width, img_height);
        };

        auto determine_window_size = [](int long_side, int short_side)
        {
            if (long_side <= 728)
                return (double)long_side / short_side > 1.5 ? short_side : 0;
            return (double)long_side / short_side > 4 ? std::min(short_side, 504) : 504;
        };

        auto get_image_size_for_crop = [](int img_width, int img_height, int window_size)
        {
            double w_ratio = (double)img_width / window_size;
            double h_ratio = (double)img_height / window_size;

            int width_new = img_width;
            int height_new = img_height;

            if (w_ratio >= 1)
            {
                const double decimal_w = w_ratio - img_width / window_size;
                w_ratio = decimal_w > 0.2 ? int(w_ratio) + 1 : int(w_ratio);
                width_new = int(window_size * w_ratio);
            }

            if (h_ratio >= 1)
            {
                const double decimal_h = h_ratio - img_height / window_size;
                h_ratio = decimal_h > 0.2 ? int(h_ratio) + 1 : int(h_ratio);
                height_new = int(window_size * h_ratio);
            }
            return std::pair<int, int>(width_new, height_new);
        };

        struct bbox
        {
            int x, y;
            int w, h;
        };

        auto slide_window = [](const int width, const int height,
            const int size_w, const int size_h,
            const int step_w, const int step_h,
            int &crops_per_row,
            std::vector<bbox> &windows,
            const float img_rate_thr = 0.6)
        {
            crops_per_row   = width  <= size_w ? 1 : (int)std::ceil((double)(width  - size_w) / step_w + 1);
            const int y_num = height <= size_h ? 1 : (int)std::ceil((double)(height - size_h) / step_h + 1);

            for (int j = 0; j < y_num; j++)
            {
                const int y_start = std::min(step_h * j, height - size_h);
                for (int i = 0; i < crops_per_row; i++)
                {
                    const int x_start = std::min(step_w * i, width - size_w);

                    bbox b;
                    b.x = x_start; b.y = y_start;
                    b.w = size_w;  b.h = size_h;
                    windows.push_back(b);
                }
            }
        };

        const auto padded_size = get_image_size_for_padding(width, height);
        int square_padded_size = -1;
        if ((padded_size.first != width) || (padded_size.second != height))
        {
            square_padded_size = std::max(width, height);
            width = square_padded_size;
            height = square_padded_size;
        }

        const auto resized_size = get_image_size_for_preprocess(width, height);
        int resized_width = -1;
        int resized_height = -1;
        if ((resized_size.first != width) || (resized_size.second != height))
        {
            width = resized_width = resized_size.first;
            height = resized_height = resized_size.second;
        }

        const int window_size = determine_window_size(std::max(width, height), std::min(width, height));

        std::vector<bbox> windows;

        if (window_size > 1)
        {
            const auto size_for_crop = get_image_size_for_crop(width, height, window_size);
            if ((size_for_crop.first != width) || (size_for_crop.second != height))
            {
                width = resized_width = size_for_crop.first;
                height = resized_height = size_for_crop.second;
            }

            slide_window(width, height, crop_size, crop_size, crop_size, crop_size, crops_per_row, windows);
        }

        // whole image
        {
            std::ostringstream oss;
            oss << MAGICK_CONVERT_CMD << " -depth 8 \"" << std::string(fn) << "\"";
            if (square_padded_size > 0)
            {
                oss << " -compose Copy -gravity northwest";
                oss << " -background back";
                oss << " -extent " << square_padded_size << "x" << square_padded_size;
            }
            oss << " -resize " << image_size << "x" << image_size << "!";

            crops.emplace_back(image_pixels_t());
            auto &image = crops.back();
            run_cmd(oss, image);
        }

        for (auto b : windows)
        {
            std::ostringstream oss;
            oss << MAGICK_CONVERT_CMD << " -depth 8 \"" << std::string(fn) << "\"";
            if (square_padded_size > 0)
            {
                oss << " -compose Copy -gravity northwest";
                oss << " -background back";
                oss << " -extent " << square_padded_size << "x" << square_padded_size;
            }
            if (resized_width > 0)
            {
                oss << " -resize " << resized_width << "x" << resized_height << "!";
            }
            oss << " -crop " << b.w << "x" << b.h << "+" << b.x << "+" << b.y;
            oss << " -resize " << crop_size << "x" << crop_size << "!";

            crops.emplace_back(image_pixels_t());
            auto &image = crops.back();
            run_cmd(oss, image);
        }
    }

    void image_load(const char *fn, std::vector<uint8_t> &rgb_pixels, int &width, int &height, int patch_size, const std::string &pad)
    {
        // magick -depth 8 demo.jpeg -resize 100x100 rgb:"aaa.raw"
        rgb_pixels.clear();
        width  = -1;
        height = -1;
        image_dimension(fn, width, height);
        if (width <= 0) return;

        std::ostringstream oss;
        oss << MAGICK_CONVERT_CMD << " -depth 8 \"" << std::string(fn) << "\"";

        if (params.do_resize)
        {
            width = params.resize_width;
            height = params.resize_height;
        }
        else
        {
            if ((params.min_pixels > 0) || (params.max_pixels > 0))
            {
                int64_t total = (int64_t)width * height;
                if ((params.min_pixels > 0) && (total < params.min_pixels))
                {
                    height = width = 0;
                    return;
                }
                if ((params.max_pixels > 0) && (total > params.max_pixels))
                {
                    double ratio = std::sqrt((double)params.max_pixels / total);
                    width  = (int)std::floor(width * ratio);
                    height = (int)std::floor(height * ratio);
                }
            }

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
        }

        oss << " -resize \"" << width << "x" << height << ">\"";

        int aligned_width  = 0;
        int aligned_height = 0;
        const int patch_size_w = params.merge_kernel_size[0] > 0 ? patch_size * params.merge_kernel_size[0] : patch_size;
        const int patch_size_h = params.merge_kernel_size[1] > 0 ? patch_size * params.merge_kernel_size[1] : patch_size;

        auto apply_max_patch_num = [&aligned_width, &aligned_height, patch_size_w, patch_size_h, patch_size]() {
            if (params.max_patch_num > 0)
            {
                int patch_num = (aligned_width / patch_size) * (aligned_height / patch_size);
                if (patch_num > params.max_patch_num)
                {
                    double ratio = std::sqrt((double)params.max_patch_num / patch_num);

                    auto height = patch_size * (int)floor(((aligned_height / patch_size) * ratio));
                    auto width  = patch_size * (int)floor(((aligned_width  / patch_size) * ratio));

                    aligned_width  = (width  / patch_size_w) * patch_size_w;
                    aligned_height = (height / patch_size_h) * patch_size_h;
                }
            }
        };

        if (params.do_resize)
        {
            aligned_width  = width;
            aligned_height = height;

            oss << " -compose Copy -gravity northwest";
            oss << " -background " << (pad != PaddingMode::No ? pad : "white");
            oss << " -extent " << aligned_width << "x" << aligned_height;
        }
        else if (pad != PaddingMode::No)
        {
            aligned_width  = ((width  + patch_size_w - 1) / patch_size_w) * patch_size_w;
            aligned_height = ((height + patch_size_h - 1) / patch_size_h) * patch_size_h;

            apply_max_patch_num();

            oss << " -compose Copy -gravity northwest";
            oss << " -background " << pad;
            oss << " -extent " << aligned_width << "x" << aligned_height;
        }
        else
        {
            aligned_width  = (width  / patch_size_w) * patch_size_w;
            aligned_height = (height / patch_size_h) * patch_size_h;

            apply_max_patch_num();

            oss << " -crop " << aligned_width << "x" << aligned_height << "+" << (width - aligned_width) / 2 << "+" << (height - aligned_height) / 2;
        }

        run_cmd(oss, rgb_pixels);

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
        case PatchesFormat::PatchesLeftRightDown_MergeN_ChannelsRGB_PixelsLeftRightDown:
            {
                CHATLLM_CHECK((params.merge_kernel_size[0] > 1) || (params.merge_kernel_size[1] > 1));
                int64_t idx = 0;
                for (int i = 0; i <= height - patch_size * params.merge_kernel_size[1]; i += patch_size * params.merge_kernel_size[1])
                {
                    for (int j = 0; j <= width - patch_size * params.merge_kernel_size[0]; j += patch_size * params.merge_kernel_size[0])
                    {
                        for (int ii = 0; ii < params.merge_kernel_size[1]; ii++)
                        {
                            const int i_m = i + ii * patch_size;
                            for (int jj = 0; jj < params.merge_kernel_size[0]; jj++)
                            {
                                const int j_m = j + jj * patch_size;
                                for (int c = 0; c < 3; c++)
                                {
                                    for (int k = 0; k < patch_size; k++)
                                    {
                                        for (int l = 0; l < patch_size; l++)
                                        {
                                            int pixel_id = (i_m + k) * width + (j_m + l);
                                            arranged[idx++] = rgb_pixels[pixel_id * 3 + c];
                                        }
                                    }
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

    VideoLoader::VideoLoader(const char *fn, float fps, const int max_frames, const int resize_width, const int resize_height)
    {
        if (!fs::exists(fn))
            return;

        tmp_dir = utils::tmpname();
        fs::create_directories(tmp_dir);

        char cmd[1024];
        if ((resize_height <= 0) || (resize_height <= 0))
        {
            sprintf(cmd, "ffmpeg -loglevel error -i \"%s\" -vf \"fps=%f\" -frames:v %d \"%s\"",
                fn, fps, max_frames, (fs::path(tmp_dir) / "%04d.jpg").string().c_str());
        }
        else
        {
            sprintf(cmd, "ffmpeg -loglevel error -i \"%s\" -vf \"scale=w=%d:h=%d:force_original_aspect_ratio=1,pad=%d:%d:(ow-iw)/2:(oh-ih)/2,fps=%f\" -frames:v %d \"%s\"",
                fn, resize_width, resize_height, resize_width, resize_height,
                fps, max_frames, (fs::path(tmp_dir) / "%04d.jpg").string().c_str());
        }

        int ret = std::system(cmd);

        for (const auto &entry : fs::directory_iterator(tmp_dir))
        {
            if (entry.path().extension() == ".jpg") {
                frames.push_back(entry.path().string());
            }
        }

        std::sort(frames.begin(), frames.end());
    }

    VideoLoader::~VideoLoader()
    {
        fs::remove_all(tmp_dir);
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

    void image_view(const float *data, size_t data_len, const int image_width, const ImageDataFormat fmt)
    {
        std::vector<uint8_t> data_int;
        data_int.resize(data_len);
        for (size_t i = 0; i < data_len; i++)
        {
            float f = std::roundf(data[i] * 255);
            int v = (int)f;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            data_int[i] = (uint8_t)v;
        }
        image_view(data_int, image_width, fmt);
    }

    void image_view(const std::vector<uint8_t> data, const int image_width, const ImageDataFormat fmt)
    {
        std::ostringstream oss;
        int64_t data_cnt = 0;

#if defined(_WIN64)
        oss << MAGICK_CONVERT_CMD;
        const std::string fn = "\"" + utils::tmpname() + ".png\"";
#else
        oss << MAGICK_DISPLAY_CMD;
#endif

        switch (fmt)
        {
        case ImageDataFormat::PixelsLeftRightDown_ChannelRGB:
            {
                const int height = (int)(data.size() / 3 / image_width);
                data_cnt = (int64_t)height * image_width * 3;

                oss << " -size " << image_width << "x" << height;
                oss << " -depth 8 rgb:-";
            }
            break;

        default:
            CHATLLM_CHECK(false);
            break;
        }

#if defined(_WIN64)
        oss << " " << fn;
#endif

        FILE* pp = popen(oss.str().c_str(), POPEN_MODE_WRITE);
        fwrite(data.data(), 1, data_cnt, pp);
        pclose(pp);

#if defined(_WIN64)
        std::string cmd = std::string("start \"\" ") + fn;
        system(cmd.c_str());
#endif
    }
}
