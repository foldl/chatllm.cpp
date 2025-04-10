#pragma once

#include <stdint.h>
#include <vector>

namespace vision
{
    void image_dimension(const char *fn, int &width, int &height);
    void image_load(const char *fn, std::vector<uint8_t> &rgb_pixels, int &width, int &height, int patch_size, int max_grid_width = -1);
    void image_rescale(const std::vector<uint8_t> &rgb_pixels, std::vector<float> &scaled_rgb_pixels, float scale_factor = 1/255.0f);
    void image_normalize(std::vector<float> &rgb_pixels, const float *mean, const float *std_d);

    void test(const char *fn);
}