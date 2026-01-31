#include "audio_process.h"
#define _USE_MATH_DEFINES // for M_PI
#include <cmath>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <array>
#include <thread>
#include "basics.h"

namespace audio
{
    struct mel_filter
    {
        int offset;
        std::vector<float> weights;
    };

    struct mel_filter_bank
    {
        std::vector<mel_filter> filters;
    };

    int Pad::length = 0;

    Pad::Pad(int pad_to)
    {
        Pad::length = pad_to;
    }

    Pad::~Pad()
    {
        Pad::length = 0;
    }

    bool load(const char *fn, std::vector<float> &samples, int sample_rate)
    {
        std::ostringstream oss;
        oss << "ffmpeg"
            << " -nostdin"
            << " -threads 0"
            << " -i \"" << fn << "\""
            << " -f f32le -ac 1 -ar " << sample_rate
            << " -hide_banner -loglevel error -";

        // Open pipe to ffmpeg process
        std::unique_ptr<FILE, decltype(&pclose)> pipe(
            popen(oss.str().c_str(), "rb"),
            pclose
        );

        if (!pipe)
            return false;

        constexpr size_t BUFFER_SIZE = 1024;
        std::array<float, BUFFER_SIZE * sizeof(float)> buffer;

        while (true)
        {
            size_t num_samples = fread(buffer.data(), sizeof(float), BUFFER_SIZE, pipe.get());

            if (num_samples == 0)
            {
                if (feof(pipe.get()))
                    break;
            }

            samples.insert(samples.end(), buffer.begin(), buffer.begin() + num_samples);
        }

        if (Pad::length > 0)
            samples.resize(Pad::length);

        return samples.size() > 0;
    }

    double hz_to_mel(double freq, bool formula_htk = false)
    {
        if (formula_htk)
        {
            return 2595.0 * std::log10(1.0 + freq / 700.0);
        }
        else
        {
            const double f_min = 0.0;
            const double f_sp  = 200.0 / 3;
            const double min_log_hz = 1000.0;
            const double min_log_mel = (min_log_hz - f_min) / f_sp;
            const double logstep     = std::log(6.4) / 27.0;
            double mel = (freq - f_min) / f_sp;
            if (freq >= min_log_hz)
                mel = min_log_mel + std::log(freq / min_log_hz) / logstep;
            return mel;
        }
    }

    double mel_to_hz(double mel, bool formula_htk = false)
    {
        if (formula_htk)
        {
            return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
        }
        else
        {
            const double f_min = 0.0;
            const double f_sp  = 200.0 / 3;
            double freq = f_min + f_sp * mel;

            const double min_log_hz = 1000.0;
            const double min_log_mel = (min_log_hz - f_min) / f_sp;
            const double logstep     = std::log(6.4) / 27.0;

            if (mel >= min_log_mel)
                freq = min_log_hz * std::exp(logstep * (mel - min_log_mel));

            return freq;
        }
    }

    void fft_frequencies(int sample_rate, int n_fft, std::vector<double> &freqs)
    {
        freqs.resize(n_fft / 2 + 1);
        for (int i = 0; i < n_fft / 2 + 1; i++)
            freqs[i] = 1.0 * i * sample_rate / n_fft;
    }

    void frequencies_tuned_to_mel_scale(std::vector<double> &freqs,
        int n_mels, double f_min,
        double f_max, bool formula_htk = false)
    {
        const double min_mel = hz_to_mel(f_min);
        const double max_mel = hz_to_mel(f_max);
        const double step = (max_mel - min_mel) / (n_mels - 1);
        freqs.resize(n_mels);
        double mel = min_mel;
        for (int i = 0; i < n_mels - 1; i++, mel += step)
            freqs[i] = mel_to_hz(mel);
        freqs[n_mels - 1] = mel_to_hz(max_mel);
    }

    // Note: `slaney` normalization is used
    void build_mel_filter_bank(mel_filter_bank &bank,
        const int sample_rate, const int n_fft,
        const int n_mels,
        const double f_min = 0.0, double f_max = -1.0,
        bool formula_htk = false)
    {
        if (f_max <= f_min) f_max = (double)sample_rate / 2;

        // center freqs of each FFT bin
        std::vector<double> fft_freqs;
        fft_frequencies(sample_rate, n_fft, fft_freqs);

        // 'Center freqs' of mel bands - uniformly spaced between limits
        std::vector<double> mel_f;
        frequencies_tuned_to_mel_scale(mel_f, n_mels + 2, f_min, f_max, formula_htk);

        for (int i = 0; i < n_mels; i++)
        {
            mel_filter f;
            f.offset = -1;
            double enorm = 2.0 / (mel_f[2 + i] - mel_f[i]);
            for (int j = 0; j < (int)fft_freqs.size(); j++)
            {
                if (fft_freqs[j] <= mel_f[i]) continue;
                if (fft_freqs[j] >= mel_f[i + 2]) break;

                if (f.offset < 0) f.offset = j;

                double w;
                if (fft_freqs[j] <= mel_f[i + 1])
                    w = (fft_freqs[j] - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
                else
                    w = 1.0 - (fft_freqs[j] - mel_f[i + 1]) / (mel_f[i + 2] - mel_f[i + 1]);
                f.weights.push_back((float)(w * enorm));
            }
            bank.filters.push_back(std::move(f));
        }
    }

    struct triangle_cache
    {
        std::vector<float> sin_vals;
        std::vector<float> cos_vals;

        triangle_cache(int size)
        {
            sin_vals.resize(size);
            cos_vals.resize(size);
            for (int i = 0; i < size; i++)
            {
                float theta = static_cast<float>((2 * M_PI * i) / size);
                sin_vals[i] = sinf(theta);
                cos_vals[i] = cosf(theta);
            }
        }
    };

    // https://en.wikipedia.org/wiki/Hann_function
    struct hann_window
    {
        std::vector<float> window;

        hann_window(int length, bool periodic = true)
        {
            window.resize(length);
            int offset = periodic ? 0 : -1;
            for (int i = 0; i < length; i++)
            {
                window[i] = (float)(0.5 * (1.0 - cos((2.0 * M_PI * i) / (length + offset))));
            }
        }
    };

    // poor man's short-time fourier transform implementation
    // input is real-valued
    // output is complex-valued (IQIQ...)
    class stft
    {
    public:
        stft(int length)
            : cache(length), window(length)
        {
            out_scratch.resize(length * 2 * 2);
        }

        void transform(const float* in, float* out) const
        {
            std::vector<float> window_in;
            const int length = (int)cache.sin_vals.size();

            window_in.resize(length);
            for (int i = 0; i < length; i++)
                window_in[i] = in[i] * window.window[i];

            fft(window_in.data(), length, 1, out,
                (float *)out_scratch.data(), cache.sin_vals.data(), cache.cos_vals.data(), length);
        }
    protected:
        // naive N-point Discrete Fourier Transform
        // input is real-valued
        // output is complex-valued (IQIQ...)
        void dft(const float* in, int N, const int stride, float* out, const float *sin_vals, const float *cos_vals, const int SIN_COS_N_COUNT) const
        {
            const int sin_cos_step = SIN_COS_N_COUNT / N;

            for (int k = 0; k < N; k++)
            {
                float re = 0.0f;
                float im = 0.0f;

                for (int n = 0; n < N; n++)
                {
                    int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
                    re += in[n * stride] * cos_vals[idx]; // cos(t)
                    im -= in[n * stride] * sin_vals[idx]; // sin(t)
                }

                out[k*2 + 0] = re;
                out[k*2 + 1] = im;
            }
        }

        void fft(const float* in, int N, int stride, float* out, float *out_scratch, const float *sin_vals, const float *cos_vals, const int SIN_COS_N_COUNT) const
        {
            if (N == 1)
            {
                out[0] = in[0];
                out[1] = 0;
                return;
            }

            const int half_N = N / 2;
            if (N - half_N * 2 == 1)
            {
                dft(in, N, stride, out, sin_vals, cos_vals, SIN_COS_N_COUNT);
                return;
            }

            float *even_fft = out_scratch;
            float * odd_fft = even_fft + (half_N * 2);
            float *scratch  =  odd_fft + (half_N * 2);
            fft(in,          half_N, stride * 2, even_fft, scratch, sin_vals, cos_vals, SIN_COS_N_COUNT);
            fft(in + stride, half_N, stride * 2,  odd_fft, scratch, sin_vals, cos_vals, SIN_COS_N_COUNT);

            const int sin_cos_step = SIN_COS_N_COUNT / N;
            for (int k = 0; k < half_N; k++)
            {
                int idx = k * sin_cos_step; // t = 2*M_PI*k/N
                float re =  cos_vals[idx]; // cos(t)
                float im = -sin_vals[idx]; // sin(t)

                float re_odd = odd_fft[2*k + 0];
                float im_odd = odd_fft[2*k + 1];

                out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
                out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

                out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
                out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
            }
        }
    protected:
        triangle_cache cache;
        hann_window    window;
        std::vector<float> out_scratch;
    };

    static void log_mel_spectrogram_worker_thread(int ith,
        const std::vector<float> & samples,
        const int64_t n_samples,
        const int frame_size,
        const int frame_step,
        const int n_threads,
        const mel_filter_bank & filters, mel & mel,
        const stft &fft)
    {
        std::vector<float> fft_in(frame_size, 0.0);
        std::vector<float> fft_out(frame_size * 2);

        const int n_fft = frame_size;

        // calculate FFT only when fft_in are not all zero
        int i = ith;
        for (; i < std::min((n_samples + frame_step - 1) / frame_step, mel.n_len); i += n_threads)
        {
            const int offset = i * frame_step;
            const int len = (int)(n_samples - offset >= frame_size ? frame_size : n_samples - offset);
            std::copy(samples.data() + offset, samples.data() + offset + len, fft_in.data());

            // fill the rest with zeros
            if (len < frame_size)
            {
                std::fill(fft_in.begin() + len, fft_in.end(), 0.0f);
            }

            fft.transform(fft_in.data(), fft_out.data());

            // power spectrum
            for (int j = 0; j < n_fft; j++)
            {
                fft_out[j] = fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
            }

            // mel spectrogram
            for (int j = 0; j < mel.n_mel; j++)
            {
                const auto &f = filters.filters[j];
                const int off = f.offset;
                size_t k = 0;
                float sum = 0.0f;
                for (; k + 3 < f.weights.size(); k += 4)
                {
                    sum +=  f.weights[k + 0] * fft_out[off + k + 0]
                          + f.weights[k + 1] * fft_out[off + k + 1]
                          + f.weights[k + 2] * fft_out[off + k + 2]
                          + f.weights[k + 3] * fft_out[off + k + 3];
                }
                for (; k < f.weights.size(); k++)
                {
                    sum +=  f.weights[k + 0] * fft_out[off + k + 0];
                }

                mel.data[j * mel.n_len + i] = log10f(std::max(1e-10f, sum));
            }
        }

        for (; i < mel.n_len; i += n_threads)
        {
            for (int j = 0; j < mel.n_mel; j++)
            {
                mel.data[j * mel.n_len + i] = -10.0f;
            }
        }
    }

    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
    static bool log_mel_spectrogram(
            const float * samples,
            int64_t n_samples,
            const int   sample_rate,
            const int   frame_size,
            const int   frame_step,
            const int   n_threads,
            const mel_filter_bank & filters,
            mel & mel,
            int64_t pad_to_samples,
            int reverse_prefix_length
        )
    {
        stft fft(frame_size);

        if (pad_to_samples < 1) pad_to_samples = n_samples;

        // Calculate the length of padding
        int64_t stage_2_pad = reverse_prefix_length;
        n_samples = std::min(n_samples, pad_to_samples);

        // Initialize a vector and copy data from C array to it.
        std::vector<float> samples_padded;
        samples_padded.resize(pad_to_samples + stage_2_pad);
        std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

        std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.end(), 0.0f);
        if (stage_2_pad > 0)
            std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

        mel.n_mel     = (int)filters.filters.size();
        mel.n_len     = pad_to_samples / frame_step;
        mel.data.resize(mel.n_mel * mel.n_len, 0.0f);

        {
            std::vector<std::thread> workers(n_threads - 1);
            for (int iw = 0; iw < n_threads - 1; ++iw)
            {
                workers[iw] = std::thread(
                        log_mel_spectrogram_worker_thread, iw + 1, std::cref(samples_padded),
                        n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                        std::cref(filters), std::ref(mel),
                        fft
                        );
            }

            // main thread
            log_mel_spectrogram_worker_thread(0, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel,
                fft);

            for (int iw = 0; iw < n_threads - 1; ++iw)
            {
                workers[iw].join();
            }
        }

        // clamping and normalization
        float mmax = std::accumulate(mel.data.begin(), mel.data.end(), -1e20f, [](auto a, auto b) { return std::max(a, b); });
        mmax -= 8.0f;
        for (auto &d : mel.data)
        {
            if (d < mmax)
                d = mmax;

            d = (d + 4.0f)/4.0f;
        }

        return true;
    }

    bool mel_spectrogram(const float *samples,
        const int64_t n_samples,
        const int64_t pad_to_samples,
        const int sample_rate,
        const int mel_feature_size,
        const int fft_size,
        const int hop_length,
        std::vector<mel> & output,
        int64_t frames_per_chunk)
    {
        mel_filter_bank filters;

        build_mel_filter_bank(filters, sample_rate, fft_size, mel_feature_size);

        const int N_FFT = fft_size;
        const int HOP_LENGTH = hop_length;

        mel out_full;
        bool r = log_mel_spectrogram(
                    samples,
                    n_samples,
                    sample_rate,
                    N_FFT,
                    HOP_LENGTH,
                    4, // n_threads
                    filters,
                    out_full,
                    pad_to_samples,
                    N_FFT / 2
        );
        if (!r) return false;

        if (frames_per_chunk <= 0)
        {
            output.push_back(std::move(out_full));
            return true;
        }

        for (int64_t off = 0; off < out_full.n_len; off += frames_per_chunk)
        {
            int64_t n_len = std::min(frames_per_chunk, out_full.n_len - off);

            mel out_chunk;
            out_chunk.n_len     = n_len;
            out_chunk.n_mel     = out_full.n_mel;
            out_chunk.data.reserve(out_chunk.n_mel * out_chunk.n_len);

            for (int i = 0; i < out_full.n_mel; i++)
            {
                auto src = out_full.data.begin() + i*out_full.n_len + off;
                out_chunk.data.insert(out_chunk.data.end(), src, src + n_len);
            }

            output.push_back(std::move(out_chunk));
        }

        return true;
    }

    int64_t mel_len(const int64_t n_samples, const int hop_length)
    {
        return n_samples / hop_length;
    }

    int64_t sample_len_for_mel_len(const int64_t n_mel, const int hop_length)
    {
        return n_mel * hop_length;
    }

    void test(void)
    {
        std::vector<float> in(128, 0.0f);
        std::vector<float> out(in.size() * 2, 0.0f);
        in[0] = 1;
        in[1] = 2;
        in[2] = 3;
        stft fft((int)in.size());
        fft.transform(in.data(), out.data());

        for (size_t i = 0; i < 128; i++)
            printf("%3d = %+10.8f + I  %+10.8f\n", (int)i, out[2 * i], out[2 * i + 1]);
    }

    void test_mel(void)
    {
        mel_filter_bank filters;
        build_mel_filter_bank(filters, 16000, 400, 128);

        for (int i = 0; i < (int)filters.filters.size(); i++)
        {
            const auto &f = filters.filters[i];
            printf("\n==== %d: %d =====", i, f.offset);

            for (size_t j = 0; j < f.weights.size(); j++)
            {
                if ((j % 4) == 0) printf("\n");
                printf("%20.18f, ", f.weights[j]);
            }
        }
    }

    void load_text_data_file(const char *fn, std::vector<float> &data)
    {
        FILE *f = fopen(fn, "r");
        char buf[200];
        for (size_t i = 0; i < data.size(); i++)
        {
            const char *l = fgets(buf, sizeof(buf) - 1, f);
            sscanf(l, "%f", &data[i]);
        }
    }
}