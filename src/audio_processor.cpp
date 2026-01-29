#include "audio_processor.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace godot;

AudioProcessor::AudioProcessor() {
    // Initialize with default values
    init_window();
    init_fft();
    init_mel_filter_bank();
    reset();
}

AudioProcessor::~AudioProcessor() {
}

void AudioProcessor::set_sample_rate(int p_rate) {
    sample_rate = p_rate;
    init_mel_filter_bank(); // Re-calc filter bank as it depends on SR
}

void AudioProcessor::set_fft_size(int p_size) {
    n_fft = p_size;
    init_fft();
    init_mel_filter_bank(); // Filter bank size depends on n_fft
}

void AudioProcessor::set_hop_length(int p_length) {
    hop_length = p_length;
    reset(); // Reset buffer state as dimensions change
}

void AudioProcessor::set_window_length(int p_length) {
    window_length = p_length;
    init_window();
    reset();
}

void AudioProcessor::set_mel_bands(int p_bands) {
    n_mels = p_bands;
    init_mel_filter_bank();
}

void AudioProcessor::set_frequency_range(float p_min, float p_max) {
    f_min = p_min;
    f_max = p_max;
    init_mel_filter_bank();
}

void AudioProcessor::init_window() {
    window.resize(window_length);
    window_buffer.resize(window_length);
    
    // Hann window
    for (int i = 0; i < window_length; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (window_length - 1)));
    }
}

void AudioProcessor::reset() {
    // Resize overlap buffer: window_length - hop_length
    int overlap_len = window_length - hop_length;
    if (overlap_len < 0) overlap_len = 0;
    
    previous_samples.assign(overlap_len, 0.0f);
}

// Simple bit reversal and trig table pre-computation
void AudioProcessor::init_fft() {
    // Assumes n_fft is power of 2
    int levels = 0;
    int temp = n_fft;
    while (temp > 1) {
        temp >>= 1;
        levels++;
    }

    bit_reverse_table.resize(n_fft);
    for (int i = 0; i < n_fft; i++) {
        int rev = 0;
        int curr = i;
        for (int j = 0; j < levels; j++) {
            rev = (rev << 1) | (curr & 1);
            curr >>= 1;
        }
        bit_reverse_table[i] = rev;
    }

    trig_tables.resize(n_fft / 2);
    for (int i = 0; i < n_fft / 2; i++) {
        float angle = -2.0f * M_PI * i / n_fft;
        trig_tables[i] = std::complex<float>(std::cos(angle), std::sin(angle));
    }
    
    fft_input.resize(n_fft);
    fft_output.resize(n_fft);
}

// In-place iterative Cooley-Tukey FFT
void AudioProcessor::perform_fft(std::vector<std::complex<float>>& data) {
    int n = data.size();
    
    // Bit-reverse permutation
    for (int i = 0; i < n; i++) {
        if (i < bit_reverse_table[i]) {
            std::swap(data[i], data[bit_reverse_table[i]]);
        }
    }

    // Butterfly operations
    for (int len = 2; len <= n; len <<= 1) {
        int half_len = len >> 1;
        int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half_len; j++) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + half_len] * trig_tables[j * step];
                data[i + j] = u + v;
                data[i + j + half_len] = u - v;
            }
        }
    }
}

float AudioProcessor::hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float AudioProcessor::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void AudioProcessor::init_mel_filter_bank() {
    int num_spectra = n_fft / 2 + 1;
    mel_filter_bank.assign(n_mels * num_spectra, 0.0f);

    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);

    std::vector<float> mel_points(n_mels + 2);
    std::vector<float> hz_points(n_mels + 2);
    std::vector<float> bin_points(n_mels + 2);

    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
        // Matching C# implementation exactly: (_nFft + 1) * hz / sample_rate
        bin_points[i] = (float)(n_fft + 1) * hz_points[i] / sample_rate;
    }

    for (int i = 0; i < n_mels; i++) {
        float left = bin_points[i];
        float center = bin_points[i + 1];
        float right = bin_points[i + 2];

        for (int j = 0; j < num_spectra; j++) {
            float weight = 0.0f;
            if (j >= left && j <= center) {
                weight = (j - left) / (center - left);
            } else if (j > center && j <= right) {
                weight = (right - j) / (right - center);
            }
            
            mel_filter_bank[i * num_spectra + j] = weight;
        }
    }
}

PackedFloat32Array AudioProcessor::process_frame(const PackedFloat32Array &p_samples) {
    if (p_samples.size() != hop_length) {
        UtilityFunctions::printerr("AudioProcessor: Expected ", hop_length, " samples, got ", p_samples.size());
        return PackedFloat32Array();
    }

    // 1. Update window buffer (shift overlapping part)
    // window_buffer structure: [Overlap (prev)] [New Hop]
    // previous_samples stores the LAST (window_length - hop_length) samples of the current buffer for the NEXT iteration.
    
    // Copy overlap from previous
    // Actually, the C# implementation:
    // _previousSamples.AsSpan().CopyTo(_windowBuffer.AsSpan(0, _previousSamples.Length));
    // hopSamples.CopyTo(_windowBuffer.AsSpan(_previousSamples.Length, _hopLength));
    // _windowBuffer.AsSpan(_hopLength, _previousSamples.Length).CopyTo(_previousSamples.AsSpan());
    
    int overlap_len = previous_samples.size();
    
    // Fill window buffer
    // Part 1: Previous overlap
    for(int i=0; i<overlap_len; i++) {
        window_buffer[i] = previous_samples[i];
    }
    
    // Part 2: New samples
    const float* new_ptr = p_samples.ptr();
    for(int i=0; i<hop_length; i++) {
        if(overlap_len + i < window_length) {
             window_buffer[overlap_len + i] = new_ptr[i];
        }
    }
    
    // Update previous_samples for NEXT call
    // We take the slice starting at `hop_length`
    for(int i=0; i<overlap_len; i++) {
        if (hop_length + i < window_length) {
            previous_samples[i] = window_buffer[hop_length + i];
        } else {
             previous_samples[i] = 0.0f;
        }
    }

    // 2. Apply Window & Prepare FFT Input
    // Zero pad if window_length < n_fft
    std::fill(fft_input.begin(), fft_input.end(), std::complex<float>(0, 0));
    
    for (int i = 0; i < window_length; i++) {
        fft_input[i] = std::complex<float>(window_buffer[i] * window[i], 0);
    }

    // 3. FFT
    perform_fft(fft_input);

    // 4. Power Spectrum & Mel Filtering
    // We only need the first n_fft/2 + 1 bins (nyquist)
    int num_spectra = n_fft / 2 + 1;
    
    PackedFloat32Array mel_features;
    mel_features.resize(n_mels);
    float* mel_ptr = mel_features.ptrw();

    for (int i = 0; i < n_mels; i++) {
        float sum = 0.0f;
        for (int j = 0; j < num_spectra; j++) {
            float mag = std::abs(fft_input[j]); // This is expensive (sqrt), optimize? 
            // C# used Magnitude * Magnitude (Power Spectrum)
            // std::norm returns squared magnitude!
            float power = std::norm(fft_input[j]); 
            
            sum += power * mel_filter_bank[i * num_spectra + j];
        }
        
        // 5. Log Scale (dB)
        // 10 * log10(max(val, 1e-10))
        mel_ptr[i] = 10.0f * std::log10(std::max(sum, 1e-10f));
    }

    // 6. Normalize (Per Utterance - approximated per frame here? or does C# do it per hop?)
    // C# implementation has `NormalizePerUtterance(Span<float> features)`.
    // It normalizes the *output mel bands* for that frame.
    // "Calculate mean and standard deviation... for features.Length".
    // features.Length is _nMels.
    // So it normalizes *across frequency bands* for the current frame?
    // "NormalizePerUtterance" usually means over time, but the C# method takes a Span (frame) and normalizes it.
    // Let's re-read C# code carefully.
    // `public float[] ProcessHop(...) { ... NormalizePerUtterance(_melSpectrum); return _melSpectrum; }`
    // Yes, it normalizes the 80 mel values *relative to each other* in that single frame.
    // This is "Instance Normalization" or similar.
    
    float sum = 0.0f;
    for(int i=0; i<n_mels; i++) sum += mel_ptr[i];
    float mean = sum / n_mels;
    
    float sum_sq = 0.0f;
    for(int i=0; i<n_mels; i++) {
        float diff = mel_ptr[i] - mean;
        sum_sq += diff * diff;
    }
    float std = std::sqrt(sum_sq / n_mels);
    if (std < 1e-8f) std = 1e-8f;
    
    for(int i=0; i<n_mels; i++) {
        mel_ptr[i] = (mel_ptr[i] - mean) / std;
    }

    return mel_features;
}

void AudioProcessor::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_sample_rate", "rate"), &AudioProcessor::set_sample_rate);
    ClassDB::bind_method(D_METHOD("set_hop_length", "length"), &AudioProcessor::set_hop_length);
    ClassDB::bind_method(D_METHOD("set_window_length", "length"), &AudioProcessor::set_window_length);
    ClassDB::bind_method(D_METHOD("set_fft_size", "size"), &AudioProcessor::set_fft_size);
    ClassDB::bind_method(D_METHOD("set_mel_bands", "bands"), &AudioProcessor::set_mel_bands);
    ClassDB::bind_method(D_METHOD("set_frequency_range", "min", "max"), &AudioProcessor::set_frequency_range);
    
    ClassDB::bind_method(D_METHOD("process_frame", "samples"), &AudioProcessor::process_frame);
    ClassDB::bind_method(D_METHOD("reset"), &AudioProcessor::reset);
}
