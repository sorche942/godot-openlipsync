#include "onnx_model.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <vector>

using namespace godot;

OnnxModel::OnnxModel() : env(ORT_LOGGING_LEVEL_WARNING, "GodotOnnx") {
}

OnnxModel::~OnnxModel() {
    if (session) {
        delete session;
    }
}

bool OnnxModel::load_model(const String &p_path) {
    if (session) {
        delete session;
        session = nullptr;
    }

    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        String global_path = ProjectSettings::get_singleton()->globalize_path(p_path);
        session = new Ort::Session(env, global_path.utf8().get_data(), session_options);
        return true;
    } catch (const Ort::Exception &e) {
        UtilityFunctions::printerr("ONNX Runtime Error: ", e.what());
        return false;
    }
}

PackedFloat32Array OnnxModel::run_inference(const PackedFloat32Array &p_input) {
    if (!session) {
        UtilityFunctions::printerr("Model not loaded.");
        return PackedFloat32Array();
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_ptr = session->GetInputNameAllocated(0, allocator);
        const char* input_names[] = { input_name_ptr.get() };
        
        auto type_info = session->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto input_shape = tensor_info.GetShape();
        
        // Resolve dynamic shapes
        // Expected shape usually: [Batch, Time, Channels] or [Batch, Channels, Time]
        // OpenLipSync TCN export: [Batch=1, Time=Dynamic, Channels=80]
        
        int64_t known_size = 1;
        int dynamic_dim_index = -1;
        
        for (size_t i = 0; i < input_shape.size(); i++) {
            if (input_shape[i] < 0) {
                if (dynamic_dim_index != -1) {
                    // More than one dynamic dimension? Default others to 1 to be safe, 
                    // but usually only Time is dynamic for us.
                    input_shape[i] = 1; 
                } else {
                    dynamic_dim_index = i;
                }
            } else {
                known_size *= input_shape[i];
            }
        }
        
        if (dynamic_dim_index != -1) {
            // Calculate missing dimension
            if (p_input.size() % known_size != 0) {
                 UtilityFunctions::printerr("Input size ", p_input.size(), " not divisible by known dimensions size ", known_size);
                 return PackedFloat32Array();
            }
            input_shape[dynamic_dim_index] = p_input.size() / known_size;
        } else {
            // No dynamic dims, strict check
            if (p_input.size() != known_size) {
                UtilityFunctions::printerr("Input size mismatch. Expected ", known_size, ", got ", p_input.size());
                return PackedFloat32Array();
            }
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Data copy (PackedFloat32Array to std::vector<float>)
        std::vector<float> input_tensor_values(p_input.size());
        const float* src = p_input.ptr();
        for(int i=0; i<p_input.size(); i++) {
            input_tensor_values[i] = src[i];
        }
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        auto output_name_ptr = session->GetOutputNameAllocated(0, allocator);
        const char* output_names[] = { output_name_ptr.get() };

        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        PackedFloat32Array result;
        result.resize(output_count);
        float* dst = result.ptrw();
        for (size_t i = 0; i < output_count; i++) {
            dst[i] = floatarr[i];
        }
        
        return result;

    } catch (const Ort::Exception &e) {
        UtilityFunctions::printerr("Inference Error: ", e.what());
        return PackedFloat32Array();
    }
}

void OnnxModel::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "path"), &OnnxModel::load_model);
    ClassDB::bind_method(D_METHOD("run_inference", "input"), &OnnxModel::run_inference);
}
