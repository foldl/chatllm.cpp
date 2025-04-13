#pragma once

#include <stdint.h>
#include <vector>
#include <string>

namespace vision
{
    void image_dimension(const char *fn, int &width, int &height);
    void image_load(const char *fn, std::vector<uint8_t> &rgb_pixels, int &width, int &height, int patch_size, int max_grid_width = -1, int max_patch_num = -1, bool pad = false, const int *merge_kernel_size = nullptr);
    void image_rescale(const std::vector<uint8_t> &rgb_pixels, std::vector<float> &scaled_rgb_pixels, float scale_factor = 1/255.0f);
    void image_normalize(std::vector<float> &rgb_pixels, const float *mean, const float *std_d);

    // ASSUMPTION: already properly aligned to `patch_size`
    void image_arrange(const std::vector<float> &rgb_pixels, const int width, const int patch_size,
        std::vector<float> &arranged, const std::string &fmt);

    void test(const char *fn);
}