#pragma once

#include <stdint.h>
#include <vector>
#include <string>

namespace vision
{
    namespace PaddingMode
    {
        const std::string No = "no";
        const std::string Black = "black";
        const std::string White = "white";
        const std::string Gray  = "#7f7f7f";
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

        PatchesLeftRightDown_MergeN_ChannelsRGB_PixelsLeftRightDown,

        // patches-left-right-down pixels-left-right-down channels-rgb
        // [patch0, patch1, ...,    // row0: left --> right
        //  ....................    // row1: left --> right
        //  ....................]
        // patch   ::= [p0, p1, ...,    // row0 within a patch: left --> right
        //              ...]            // row1 ....
        // pixel   ::= [r, g, b]
        PatchesLeftRightDown_PixelsLeftRightDown_ChannelsRGB,

        // channels-rgb patches-left-right-down pixels-left-right-down
        // [ch_r, ch_g, ch_b]
        // channel ::= [patch0, patch1, ...,    // row0: left --> right
        //              ....................    // row1: left --> right
        //              ....................]
        // patch   ::= [v0, v1, ...,    // row0 within a patch: left --> right
        //              ...]            // row1 ....
        ChannelsRGB_PatchesLeftRightDown_PixelsLeftRightDown,

        // channels-rgb pixels-left-right-down
        // [ch_r, ch_g, ch_b]
        // channel   ::= [v0, v1, ...,    // row0 within the picture: left --> right
        //                ...]            // row1 ....
        ChannelsRGB_PixelsLeftRightDown,
    };

    class Resize
    {
    public:
        Resize(int width, int height);
        ~Resize(void);
    };

    class MergeKernel
    {
    public:
        MergeKernel(int size0, int size1);
        MergeKernel(const int *sizes);
        ~MergeKernel(void);
    };

    class MaxGridWidth
    {
    public:
        MaxGridWidth(int value);
        ~MaxGridWidth();
    };

    class MaxGridHeight
    {
    public:
        MaxGridHeight(int value);
        ~MaxGridHeight();
    };

    class MinMaxPixels
    {
    public:
        MinMaxPixels(int64_t min_pixels, int64_t max_pixels);
        ~MinMaxPixels();
    };

    class MaxPatchNum
    {
    public:
        MaxPatchNum(int value);
        ~MaxPatchNum();
    };

    class PreMaxImageSize
    {
    public:
        PreMaxImageSize(int width, int height = -1);
        ~PreMaxImageSize();
        static bool PreScale(int &width, int &height);
    };

    enum PanScanDir
    {
        Horizontal,
        Vertical,
    };

    typedef std::vector<uint8_t> image_pixels_t; // natural sequence of RGB pixels

    void image_dimension(const char *fn, int &width, int &height);
    void image_load(const char *fn, std::vector<uint8_t> &rgb_pixels, int &width, int &height, int patch_size, const std::string &pad = PaddingMode::No);
    void image_load_split(const char *fn, std::vector<image_pixels_t> &splits, bool do_split, const int split_width, const int split_height, int &splits_cols_num, int &splits_rows_num); // splits are in natural order
    void image_load_pan_and_scan(const char *fn, std::vector<image_pixels_t> &crops, bool do_pas,
        const int min_crop_size, const int max_num_crops, float min_ratio_to_activate,
        const int crop_width, const int crop_height,
        PanScanDir &dir);

    // pan and scan for step-vl
    void image_load_pan_and_scan(const char *fn, std::vector<image_pixels_t> &crops,
        const int image_size, int crop_size, int &crops_per_row);

    void image_rescale(const std::vector<uint8_t> &rgb_pixels, std::vector<float> &scaled_rgb_pixels, float scale_factor = 1/255.0f);
    void image_normalize(std::vector<float> &rgb_pixels, const float *mean, const float *std_d);

    class VideoLoader
    {
    public:
        VideoLoader(const char *fn, float fps = 1.0f, const int max_frames = 10, const int resize_width = -1, const int resize_height = -1);
        ~VideoLoader();
    public:
        std::vector<std::string> frames;
    private:
        std::string tmp_dir;
    };

    // ASSUMPTION: already properly aligned to `patch_size`
    void image_arrange(const std::vector<float> &rgb_pixels, const int width, const int patch_size,
        std::vector<float> &arranged, const PatchesFormat fmt);

    enum ImageDataFormat
    {
        // 1) uint8_t for each component;
        // 2) float: [0, 1] for each component;
        PixelsLeftRightDown_ChannelRGB,
    };

    void image_view(const float *data, size_t data_len, const int image_width, const ImageDataFormat fmt);
    void image_view(const std::vector<uint8_t> data, const int image_width, const ImageDataFormat fmt);
}