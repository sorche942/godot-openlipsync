#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <vector>
#include <complex>

namespace godot {

class AudioProcessor : public RefCounted {
    GDCLASS(AudioProcessor, RefCounted)

private:
    // Config
    int sample_rate = 16000;
    int hop_length = 160;   // 10ms at 16kHz
    int window_length = 400; // 25ms at 16kHz
    int n_fft = 1024;
    int n_mels = 80;
    float f_min = 50.0f;
    float f_max = 8000.0f;

    // Buffers
    std::vector<float> window;
    std::vector<float> window_buffer;
    std::vector<float> previous_samples;
    std::vector<float> fft_buffer; // For FFT input/output logic if needed
    
    // Mel Filter Bank: flattened [n_mels * (n_fft/2 + 1)]
    // Stored as row-major: filter_bank[mel_idx * (n_fft/2 + 1) + bin_idx]
    std::vector<float> mel_filter_bank;

    // FFT State
    // Using std::complex for internal FFT implementation
    std::vector<std::complex<float>> fft_input;
    std::vector<std::complex<float>> fft_output;
    std::vector<int> bit_reverse_table;
    std::vector<std::complex<float>> trig_tables;

    // Internal methods
    void init_window();
    void init_mel_filter_bank();
    void init_fft();
    void perform_fft(std::vector<std::complex<float>>& data);
    
    float hz_to_mel(float hz);
    float mel_to_hz(float mel);

protected:
    static void _bind_methods();

public:
    AudioProcessor();
    ~AudioProcessor();

    // Configuration methods
    void set_sample_rate(int p_rate);
    void set_fft_size(int p_size);
    void set_hop_length(int p_length);
    void set_window_length(int p_length);
    void set_mel_bands(int p_bands);
    void set_frequency_range(float p_min, float p_max);
    
    // Main processing
    // Takes exactly hop_length samples. 
    // Maintains internal state for overlapping windows.
    PackedFloat32Array process_frame(const PackedFloat32Array &p_samples);
    
    // Reset internal state (overlap buffer)
    void reset();
};

} // namespace godot

#endif
