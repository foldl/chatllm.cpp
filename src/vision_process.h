#pragma once

#include <stdint.h>
#include <vector>
#include <string>

namespace vision
{
    enum PaddingMode
    {
        No,
        Black,
        White,
    };

    enum PatchesFormat
    {
        // patches-left-right-down channels-rgb pixels-left-right-down
        // [patch0, patch1, ...,    // row0: left --> right
        //  ....................    // row1: left --> right
        //  ....................]
        // patch   ::= [ch_r, ch_g, ch_b]
        // channel ::= [value0, value1, // row0 within a patch: left --> right
        //              ...]            // row1 ....
        PatchesLeftRightDown_ChannelsRGB_PixelsLeftRightDown,

        // patches-left-right-down pixels-left-right-down channels-rgb
        // [patch0, patch1, ...,    // row0: left --> right
        //  ....................    // row1: left --> right
        //  ....................]
        // patch   ::= [p0, p11, ...,   // row0 within a patch: left --> right
        //              ...]            // row1 ....
        // pixel   ::= [r, g, b]
        PatchesLeftRightDown_PixelsLeftRightDown_ChannelsRGB,
    };

    void image_dimension(const char *fn, int &width, int &height);
    void image_load(const char *fn, std::vector<uint8_t> &rgb_pixels, int &width, int &height, int patch_size, int max_grid_width = -1, int max_patch_num = -1, PaddingMode pad = PaddingMode::No, const int *merge_kernel_size = nullptr);
    void image_rescale(const std::vector<uint8_t> &rgb_pixels, std::vector<float> &scaled_rgb_pixels, float scale_factor = 1/255.0f);
    void image_normalize(std::vector<float> &rgb_pixels, const float *mean, const float *std_d);

    // ASSUMPTION: already properly aligned to `patch_size`
    void image_arrange(const std::vector<float> &rgb_pixels, const int width, const int patch_size,
        std::vector<float> &arranged, const PatchesFormat fmt);

    void test(const char *fn);
}