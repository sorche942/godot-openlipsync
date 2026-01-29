extends Node
class_name LipSyncMicController

# --- Configuration ---
@export_group("Model")
@export_file("*.onnx") var model_path: String = "res://addons/godot_openlipsync/model.onnx"
@export var context_window_size: int = 100

@export_group("Audio")
@export var audio_bus_name: String = "Record" # Default for Mic
@export var use_microphone: bool = true

@export_group("Target")
@export var mesh_instance: MeshInstance3D
@export var sensitivity: float = 1.0
@export_range(0.0, 1.0) var smoothing: float = 0.5
@export_range(0.0, 1.0) var noise_gate: float = 0.05 # Threshold to ignore silence/noise

# Mapping: Viseme Index (Model) -> Blend Shape Name (Mesh)
# --------------------------------------------------------
# 0: silence
# 1: PP (p, b, m)
# 2: FF (f, v)
# 3: TH (th)
# 4: DD (t, d)
# 5: kk (k, g)
# 6: CH (ch, j, sh)
# 7: SS (s, z)
# 8: nn (n, l)
# 9: RR (r)
# 10: aa (a)
# 11: E  (e)
# 12: ih (i)
# 13: oh (o)
# 14: ou (u)
#
# EXAMPLE VRoid (Internal Naming) Mapping:
# 10: "Fcl_MTH_A", 11: "Fcl_MTH_E", 12: "Fcl_MTH_I", 13: "Fcl_MTH_O", 14: "Fcl_MTH_U"
#
# EXAMPLE VRM Mapping:
# 10: "vrm_a", 11: "vrm_e", 12: "vrm_i", 13: "vrm_o", 14: "vrm_u"

@export var viseme_mapping: Dictionary = {
	1: "viseme_PP",
	2: "viseme_FF",
	3: "viseme_TH",
	4: "viseme_DD",
	5: "viseme_kk",
	6: "viseme_CH",
	7: "viseme_SS",
	8: "viseme_nn",
	9: "viseme_RR",
	10: "viseme_aa",
	11: "viseme_E",
	12: "viseme_ih",
	13: "viseme_oh",
	14: "viseme_ou"
}

# --- Internals ---
var context: LipSyncContext
var capture_effect: AudioEffectCapture
var bus_index: int
var current_blend_weights: Dictionary = {}

func _ready():
	if not mesh_instance:
		printerr("LipSyncMicController: No MeshInstance3D assigned!")
		set_process(false)
		return

	# 1. Initialize Context
	context = LipSyncContext.new()
	if not FileAccess.file_exists(model_path):
		printerr("LipSyncMicController: Model not found at ", model_path)
		set_process(false)
		return
		
	if not context.load_model(model_path):
		printerr("LipSyncMicController: Failed to load ONNX model.")
		set_process(false)
		return
		
	context.set_context_size(context_window_size)
	print("LipSync: Model loaded.")

	# 2. Setup Audio
	_setup_audio_capture()
	
	# Initialize current weights
	for v_idx in viseme_mapping:
		current_blend_weights[v_idx] = 0.0

func _setup_audio_capture():
	bus_index = AudioServer.get_bus_index(audio_bus_name)
	if bus_index == -1:
		# Auto-create bus if missing (useful for 'Record')
		AudioServer.add_bus()
		AudioServer.set_bus_name(AudioServer.get_bus_count() - 1, audio_bus_name)
		bus_index = AudioServer.get_bus_index(audio_bus_name)
		# Mute mic bus to avoid feedback
		if use_microphone:
			AudioServer.set_bus_mute(bus_index, true)
		print("LipSync: Created audio bus '", audio_bus_name, "'")

	# Find or add Capture effect
	var effect_count = AudioServer.get_bus_effect_count(bus_index)
	for i in range(effect_count):
		var effect = AudioServer.get_bus_effect(bus_index, i)
		if effect is AudioEffectCapture:
			capture_effect = effect
			break
			
	if not capture_effect:
		capture_effect = AudioEffectCapture.new()
		AudioServer.add_bus_effect(bus_index, capture_effect)
		print("LipSync: Added AudioEffectCapture to bus '", audio_bus_name, "'")
	
	capture_effect.clear_buffer()
	
	# Check for Mic input settings
	if use_microphone:
		var devices = AudioServer.get_input_device_list()
		print("LipSync: Available input devices: ", devices)
		
		# Set to 'Default' or first available
		AudioServer.set_input_device("Default")
		print("LipSync: Set input device to: Default")
			
		print("LipSync: Current audio input device: ", AudioServer.get_input_device())
		if not ProjectSettings.get_setting("audio/driver/enable_input"):
			printerr("LipSync: Microphone input is DISABLED in Project Settings! Enable 'audio/driver/enable_input'.")

var device_check_timer: float = 0.0
var device_checked: bool = false
var silence_timer: float = 0.0

func _process(delta):
	if use_microphone and not device_checked:
		device_check_timer += delta
		if device_check_timer > 1.0:
			print("LipSync: Audio input device after 1s: ", AudioServer.get_input_device())
			device_checked = true

	if not capture_effect: return
	
	var frames_available = capture_effect.get_frames_available()
	
	if frames_available == 0:
		silence_timer += delta
		if silence_timer > 3.0:
			printerr("LipSync: Warning - No audio frames received for 3+ seconds. Check Audio Input settings.")
			silence_timer = 0.0 # Reset to warn again later
	else:
		silence_timer = 0.0
	
	if frames_available > 0:
		var audio_frames = capture_effect.get_buffer(frames_available)
		var source_rate = AudioServer.get_mix_rate()
		
		# Noise Gate Check
		var max_amp = 0.0
		for sample in audio_frames:
			max_amp = max(max_amp, abs(sample.x), abs(sample.y))
			
		if max_amp < noise_gate:
			# Signal silence to smooth towards zero
			var silence = PackedFloat32Array()
			silence.resize(viseme_mapping.size() + 5) # Enough for any index
			silence.fill(0.0)
			_apply_visemes(silence, delta)
			return

		var prediction = context.process(audio_frames, source_rate)
		
		if prediction.size() > 0:
			_apply_visemes(prediction, delta)

func _apply_visemes(prediction: PackedFloat32Array, delta: float):
	# 1. Aggregate weights for each unique blend shape
	var shape_weights = {} 
	
	for v_idx in viseme_mapping:
		if v_idx >= prediction.size(): continue
		
		var blend_name = viseme_mapping[v_idx]
		var raw_weight = prediction[v_idx] * sensitivity
		
		# If multiple visemes map to same shape, take the MAX weight
		# (e.g. if 'ou' and 'u' both map to 'mouth_u', don't sum them)
		if blend_name in shape_weights:
			shape_weights[blend_name] = max(shape_weights[blend_name], raw_weight)
		else:
			shape_weights[blend_name] = raw_weight

	# 2. Smooth and Apply
	for blend_name in shape_weights:
		var target_weight = shape_weights[blend_name]
		var prev_weight = current_blend_weights.get(blend_name, 0.0)
		
		# Frame-rate independent smoothing
		var new_weight = lerp(prev_weight, target_weight, 1.0 - smoothing)
		
		# Clamp to avoid comic distortion
		new_weight = clamp(new_weight, 0.0, 1.0)
		
		current_blend_weights[blend_name] = new_weight
		
		# Apply to mesh
		var bs_idx = mesh_instance.find_blend_shape_by_name(blend_name)
		if bs_idx != -1:
			mesh_instance.set_blend_shape_value(bs_idx, new_weight)
