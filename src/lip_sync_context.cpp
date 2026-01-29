#include "lip_sync_context.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <cmath>

using namespace godot;

LipSyncContext::LipSyncContext() {
    processor.instantiate();
    // Configure processor defaults (match typical TCN config)
    processor->set_sample_rate(16000);
    processor->set_hop_length(160); // 10ms
    processor->set_window_length(400); // 25ms
    processor->set_mel_bands(80);
}

LipSyncContext::~LipSyncContext() {
}

bool LipSyncContext::load_model(const String &p_path) {
    if (model.is_null()) {
        model.instantiate();
    }
    bool success = model->load_model(p_path);
    if (success) {
        reset();
    }
    return success;
}

void LipSyncContext::set_context_size(int p_frames) {
    context_size = p_frames;
    // Trim buffer if needed
    while (feature_buffer.size() > context_size) {
        feature_buffer.pop_front();
    }
}

void LipSyncContext::reset() {
    audio_buffer.clear();
    feature_buffer.clear();
    if (processor.is_valid()) {
        processor->reset();
    }
    resample_fraction = 0.0f;
}

void LipSyncContext::_resample_and_push(const float* input, int count, int source_rate) {
    if (count == 0) return;

    if (source_rate != target_sample_rate) {
        // Simple Linear Interpolation Resampling
        float ratio = (float)source_rate / (float)target_sample_rate;
        
        // Number of output samples to generate from this chunk
        // Note: this is a streaming resampler, state is kept in resample_fraction
        
        // input index comes from: index * ratio + resample_fraction
        // We iterate output samples until we run out of input
        
        // We assume input is contiguous mono floats
        
        float current_pos = resample_fraction;
        
        while (current_pos < count - 1) {
            int idx = (int)current_pos;
            float t = current_pos - idx;
            
            // Lerp
            float s0 = input[idx];
            float s1 = input[idx + 1];
            float val = s0 + (s1 - s0) * t;
            
            audio_buffer.push_back(val);
            
            current_pos += ratio;
        }
        
        // Save fractional part for next chunk (relative to end of this chunk)
        // current_pos is now >= count - 1. 
        // We want the new fraction relative to the START of the next chunk.
        // The start of the next chunk is at index 'count'.
        resample_fraction = current_pos - count; 
        
        // If we processed exactly to the end (unlikely with floats), handle it
        // The loop above stops when we can't interpolate (need idx+1)
        // Ideally we keep the last sample as "previous sample" for the next chunk?
        // For simplicity, we drop < 1 sample at boundaries or rely on frequent updates.
        // A robust implementation would buffer the last sample.
        // Given this is for visualization, dropping 1 sample per frame is OK (1/735 error).
        // Let's just wrap resample_fraction.
        if (resample_fraction < 0) resample_fraction += ratio; // Should not happen with above math
        
    } else {
        // Direct copy
        audio_buffer.insert(audio_buffer.end(), input, input + count);
    }
}

PackedFloat32Array LipSyncContext::process(const PackedVector2Array &p_audio_data, int p_source_sample_rate) {
    if (model.is_null()) {
        // UtilityFunctions::printerr("LipSyncContext: Model not loaded.");
        return PackedFloat32Array();
    }

    int sample_count = p_audio_data.size();
    if (sample_count == 0) return PackedFloat32Array();

    // 1. Downmix to Mono and convert to float array for internal processing
    // p_audio_data is Vector2 (L, R)
    // We could optimize by doing mixing + resampling in one pass, but let's separate for clarity.
    
    // Use stack or temporary vector
    std::vector<float> mono_input(sample_count);
    const Vector2* ptr = p_audio_data.ptr();
    for (int i = 0; i < sample_count; i++) {
        mono_input[i] = (ptr[i].x + ptr[i].y) * 0.5f;
    }

    // 2. Resample and Buffer
    _resample_and_push(mono_input.data(), sample_count, p_source_sample_rate);

    // 3. Process Hops
    bool new_features_added = false;
    int hop_length = 160; // Hardcoded default for now, should get from processor
    
    // We need 'hop_length' samples to process a frame.
    // AudioProcessor consumes 'hop_length' samples per call (conceptually shifting the window).
    
    while (audio_buffer.size() >= hop_length) {
        // Prepare input for processor
        PackedFloat32Array chunk;
        chunk.resize(hop_length);
        float* chunk_ptr = chunk.ptrw();
        
        for (int i = 0; i < hop_length; i++) {
            chunk_ptr[i] = audio_buffer[i];
        }
        
        // Process
        PackedFloat32Array features = processor->process_frame(chunk);
        
        // Append to feature buffer
        // features is a flat array of 80 floats
        if (features.size() > 0) {
             std::vector<float> f_vec(features.size());
             const float* f_ptr = features.ptr();
             for(int i=0; i<features.size(); i++) f_vec[i] = f_ptr[i];
             
             feature_buffer.push_back(f_vec);
             new_features_added = true;
        }
        
        // Remove processed samples from audio buffer
        // Note: This is inefficient (erase from beginning of vector). 
        // Use deque or index offset for production, but vector::erase is fine for small chunks (160 floats).
        audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + hop_length);
        
        // Enforce max context size
        if (feature_buffer.size() > context_size) {
            feature_buffer.pop_front();
        }
    }
    
    // 4. Run Inference (if we have new data)
    if (new_features_added && !feature_buffer.empty()) {
        // Prepare flat input for ONNX
        // Model expects (Batch, Time, Channels) -> (1, T, 80)
        // Flattened: [Frame0(80), Frame1(80)...]
        
        int n_frames = feature_buffer.size();
        int n_mels = 80; // Should get from processor
        
        PackedFloat32Array flat_input;
        flat_input.resize(n_frames * n_mels);
        float* dest = flat_input.ptrw();
        
        for (int i = 0; i < n_frames; i++) {
            const std::vector<float>& frame = feature_buffer[i];
            for (int j = 0; j < n_mels; j++) {
                dest[i * n_mels + j] = frame[j];
            }
        }
        
        // Run inference
        PackedFloat32Array output = model->run_inference(flat_input);
        
        // Output shape: (1, T, Visemes) flattened
        // We want the LAST frame's prediction
        // output size = T * num_visemes
        // num_visemes = output.size() / T
        
        if (output.size() > 0) {
            int num_visemes = output.size() / n_frames;
            
            // Extract last frame
            PackedFloat32Array result;
            result.resize(num_visemes);
            float* res_ptr = result.ptrw();
            const float* out_ptr = output.ptr();
            
            int start_idx = (n_frames - 1) * num_visemes;
            for (int i = 0; i < num_visemes; i++) {
                res_ptr[i] = out_ptr[start_idx + i];
            }
            
            return result;
        }
    }
    
    return PackedFloat32Array(); // No new prediction
}

void LipSyncContext::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "path"), &LipSyncContext::load_model);
    ClassDB::bind_method(D_METHOD("set_context_size", "frames"), &LipSyncContext::set_context_size);
    ClassDB::bind_method(D_METHOD("process", "audio_data", "sample_rate"), &LipSyncContext::process);
    ClassDB::bind_method(D_METHOD("reset"), &LipSyncContext::reset);
}
