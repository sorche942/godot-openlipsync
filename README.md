# Godot OpenLipSync

**A Godot 4.x GDExtension integration of the OpenLipSync architecture.**

> [!NOTE]
> The core **Temporal Convolutional Network (TCN)** architecture and the reference C# implementation utilized in this project were designed and developed by **[KyuubiYoru](https://github.com/KyuubiYoru)**, originally for the **Resonite** platform. This project serves as a Godot bridge to their excellent work!

OpenLipSync is a high-performance, real-time lip synchronization GDExtension for Godot 4.x. It uses a TCN to map audio features (mel spectrograms) to viseme probabilities, providing realistic facial expressions based on audio input.

## Features

*   **GDExtension (C++):** Native performance for audio feature extraction (FFT, Mel-scaling) and model execution.
*   **Easy Integration:** Drag-and-drop addon structure with a robust GDScript controller.
*   **Customizable Mapping:** Easily map visemes to any 3D model blend shapes (compatible with VRM, VRoid, etc.).
*   **Streaming Support:** Handles live microphone input or pre-recorded audio streams.

## Project Structure

*   `godot-openlipsync/`: The main Godot project and GDExtension source.
    *   `src/`: C++ source code for the GDExtension.
    *   `project/addons/godot_openlipsync/`: The Godot addon folder.
        *   `model.onnx`: The core trained TCN model.
        *   `examples/`: Sample scripts and scenes for VRM/VRoid characters.
*   `resources/OpenLipSync/`: Original training pipeline and inference references.

## Getting Started

### Prerequisites

*   Godot 4.x (tested with 4.6).
*   For building from source: SCons and a C++ compiler (GCC/Clang for Linux, MSVC for Windows).
*   ONNX Runtime libraries (provided in `resources/onnxruntime-linux-x64-1.23.2`).

### Installation

1.  Copy the `addons/godot_openlipsync` folder from the `project` directory into your own Godot project.
2.  Ensure the `libonnxruntime.so` (Linux) or `onnxruntime.dll` (Windows) is in the same directory as the GDExtension binary.

### Setup for Microphone Input

1.  **Project Settings:** Go to `Project > Project Settings > Audio > Driver` and enable **Enable Input**. Restart Godot.
2.  **Audio Bus:** Open the **Audio** tab at the bottom of the editor. Create a bus named `Record` (or use the provided sample script which can auto-create it).
3.  **Microphone Node:** Add an `AudioStreamPlayer` to your scene.
    *   Set **Stream** to `AudioStreamMicrophone`.
    *   Set **Bus** to `Record`.
    *   Enable **Autoplay**.
4.  **Controller:** Add a node to your scene and attach `res://addons/godot_openlipsync/examples/lipsync_mic_demo.gd`.
5.  **Assign Mesh:** Drag your character's `MeshInstance3D` into the **Mesh Instance** property in the Inspector.
6.  **Map Visemes:** In the Inspector, configure the `Viseme Mapping` dictionary to match your mesh's blend shape names. (See the script comments for a reference list of visemes).

## Development

### Building from Source

To build the GDExtension:

```bash
cd godot-openlipsync
scons platform=linux target=template_debug
```

The build script automatically handles include paths for the bundled ONNX Runtime and sets the RPATH so the extension can find the shared libraries.

## License

*   **Godot OpenLipSync (This Project):** Licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*   **OpenLipSync (Original Architecture):** Licensed under the Apache License 2.0.
*   **ONNX Runtime:** Licensed under the MIT License by Microsoft Corporation.
