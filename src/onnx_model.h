#ifndef ONNX_MODEL_H
#define ONNX_MODEL_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <onnxruntime_cxx_api.h>

namespace godot {

class OnnxModel : public RefCounted {
    GDCLASS(OnnxModel, RefCounted)

private:
    Ort::Env env;
    Ort::Session *session = nullptr;

protected:
    static void _bind_methods();

public:
    OnnxModel();
    ~OnnxModel();

    bool load_model(const String &p_path);
    PackedFloat32Array run_inference(const PackedFloat32Array &p_input);
};

} // namespace godot

#endif
