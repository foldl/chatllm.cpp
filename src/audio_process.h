#pragma once

#include <stdint.h>
#include <vector>
#include <string>

namespace audio
{
    struct mel
    {
        int64_t n_len;
        int     n_mel;

        std::vector<float> data;
    };

    class Pad
    {
    public:
        Pad(int pad_to);
        ~Pad();
    public:
        static int length;
    };

    // load as mono with re-sampling
    bool load(const char *fn, std::vector<float> &samples, int sample_rate);

    bool mel_spectrogram(const float *samples,
        const int64_t n_samples,
        const int64_t pad_to_samples,
        const int sample_rate,
        const int mel_feature_size,
        const int fft_size,
        const int hop_length,
        std::vector<mel> & output,
        int64_t frames_per_chunk = -1);

    void test(void);
}