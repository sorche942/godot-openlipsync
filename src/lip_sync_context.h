#ifndef LIP_SYNC_CONTEXT_H
#define LIP_SYNC_CONTEXT_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include "audio_processor.h"
#include "onnx_model.h"
#include <vector>
#include <deque>

namespace godot {

class LipSyncContext : public RefCounted {
    GDCLASS(LipSyncContext, RefCounted)

private:
    Ref<AudioProcessor> processor;
    Ref<OnnxModel> model;
    
    // Audio buffering
    std::vector<float> audio_buffer; // Accumulates mono samples at target rate (16kHz)
    
    // Feature buffering (Sliding window for model input)
    // Stored as flat floats. Size = context_size * n_mels
    // We use a deque of "frames" (vectors of floats) for easier management
    std::deque<std::vector<float>> feature_buffer;
    
    int context_size = 100; // Number of frames to keep for model context (e.g. 1s at 100fps)
    int target_sample_rate = 16000;
    
    // Resampling state
    float resample_ratio = 1.0f;
    float resample_fraction = 0.0f; // Fractional part for linear interpolation

    void _resample_and_push(const float* input, int count, int source_rate);

protected:
    static void _bind_methods();

public:
    LipSyncContext();
    ~LipSyncContext();

    // Setup
    bool load_model(const String &p_path);
    void set_context_size(int p_frames);
    
    // Main loop
    // Consumes audio, returns the latest viseme prediction (or empty if no new prediction)
    PackedFloat32Array process(const PackedVector2Array &p_audio_data, int p_source_sample_rate);
    
    // Helpers
    Ref<AudioProcessor> get_processor() const { return processor; }
    void reset();
};

} // namespace godot

#endif
