#include "vision_process.h"
#include <regex>
#include <sstream>

namespace vision
{
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
            fgets(buffer, sizeof(buffer) - 1, pp);

            std::string s(buffer);

            if (std::regex_search(s, matches, rgx) && (matches.size() == 3))
            {
                width  = std::stoi(matches[1].str());
                height = std::stoi(matches[2].str());
                break;
            }
        }

        fclose(pp);
    }

    void image_load(const char *fn, std::vector<uint8_t> &rgb_pixels, int &width, int &height, int patch_size, int max_grid_width)
    {
        // magick -depth 8 demo.jpeg -resize 100x100 rgb:"aaa.raw"
        rgb_pixels.clear();
        width  = -1;
        height = -1;
        image_dimension(fn, width, height);
        if (width < 0) return;

        std::ostringstream oss;
        oss << "magick -depth 8 \"" << std::string(fn) << "\"";

        if (max_grid_width > 0)
        {
            int new_w = std::min(width, patch_size * max_grid_width);
            double ratio = (double)new_w / width;
            height = (int)(height * ratio);
            width  = new_w;
        }

        oss << " -resize " << width << "x" << height << "!";

        int aligned_width  = (width  / patch_size) * patch_size;
        int aligned_height = (height / patch_size) * patch_size;

        oss << " -crop " << aligned_width << "x" << aligned_height << "+" << (width - aligned_width) / 2 << "+" << (height - aligned_height) / 2 << " ";
        oss << " rgb:-";
        // oss << " rgb:c:\\tmp\\img.raw";

        printf("%s\n", oss.str().c_str());

        FILE* pp = popen(oss.str().c_str(), "rb");

        while (!feof(pp))
        {
            uint8_t buffer[1024];
            size_t cnt = fread(buffer, 1, sizeof(buffer), pp);
            if (cnt > 0)
                rgb_pixels.insert(rgb_pixels.end(), buffer, buffer + cnt);
        }

        fclose(pp);

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

    void print_image_chw(const std::vector<float> &pixels)
    {
        const size_t cnt = pixels.size() / 3;
        //for (int i = 0; i < 3)
        for (size_t j = 0; j < cnt; j++)
        {
            printf("%3zd: (%.4f, %.4f, %.4f)\n", j, pixels[j * 3 + 0], pixels[j * 3 + 1], pixels[j * 3 + 2]);
            if (j > 10) break;
        }
    }

    void print_image(const std::vector<float> &pixels)
    {
        const size_t cnt = pixels.size() / 3;
        for (size_t j = 0; j < cnt; j++)
        {
            printf("%3zd: (%.4f, %.4f, %.4f)\n", j, pixels[j * 3 + 0], pixels[j * 3 + 1], pixels[j * 3 + 2]);
            if (j > 10) break;
        }
    }

    void test(const char *fn)
    {
        int w, h;
        std::vector<uint8_t> pixels;
        vision::image_load(fn, pixels, w, h, 14, 16);
        printf("%dx%d = %zd, %d\n", w, h, pixels.size() / 3, w * h == pixels.size() / 3);

        std::vector<float> scaled;
        image_rescale(pixels, scaled);

        print_image(scaled);

        const float means[] = {0.48145466f,
            0.4578275f,
            0.40821073f};
        const float std_d[] = {0.26862954f,
            0.26130258f,
            0.27577711f};

        image_normalize(scaled, means, std_d);

        print_image(scaled);
    }
}
