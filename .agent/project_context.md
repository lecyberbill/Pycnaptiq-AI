# Project Context: Pycnaptiq-AI

This file serves as a memory and context bridge for AI agents working on this project.

## Project Description
Pycnaptiq-AI is a Gradio-based web interface for image generation using Stable Diffusion XL (SDXL), SD3.5, FLUX, Sana Sprint, and other modern diffusion models. It supports features like ControlNet, IP-Adapter, Inpainting, and Model Management (VRAM optimization, FP8, LoRA).

## Core Architecture
- **`Pycnaptiq-AI.py`**: Main entry point. Initializes `ModelManager`, `PresetManager`, and `GestionModule`.
- **`Utils/model_manager.py`**: Centralizes model loading, unloading, VRAM management, and component registration.
- **`Utils/utils.py`**: Contains `GestionModule`, the engine for the dynamic module system.
- **`core/sdxl_logic.py`**: Contains the specific generation logic for SDXL models.
- **`core/pipeline_executor.py`**: Handles asynchronous execution of diffusion pipelines in separate threads with progress monitoring.
- **`core/model_loaders.py`**: UI wrappers for selecting and loading models.

## Technical Choices
- **Framework**: Python 3.10+, Gradio.
- **Diffusion Library**: `diffusers` (Hugging Face).
- **Hardware**: Optimized for NVIDIA GPUs (RTX 4070 Ti in user's current environment).
- **Quantization**: Supports FP8 (Quanto, BitsAndBytes) and GGUF.
- **Memory**: Uses recursive memory management and `empty_working_set` to optimize VRAM usage.

## Module System
The application uses a dynamic module system located in the `modules/` directory. Each module allows extending the interface with new models or specialized tools.

### Structure of a Module
- **`Module_mod.py`**: The logic and UI of the module.
- **`Module_mod.json`**: Metadata, dependencies, and translations specific to the module.

### Initialization Workflow
1. **Detection**: `GestionModule` scans the `modules/` directory for `.py` files.
2. **Metadata**: It reads the corresponding `.json` for dependencies and basic info.
3. **Loading**: It uses `importlib` to load the module.
4. **Initialization**: It calls the `initialize()` function of the module, passing `global_translations`, `model_manager`, and `gestionnaire`.
5. **UI Creation**: It calls `module_instance.create_tab(translations)` to add a new tab to the Gradio interface.

## Major Recent Changes (Jan 2026)
- **IP-Adapter Fix**: Fixed a `ValueError` when disabling IP-Adapter by implementing a proper unloading logic in `ModelManager.prepare_ip_adapter_pipeline` and `sdxl_logic.py`.
- **Session History**: Added an automatic history gallery monitoring the output directory.
- **Z-Image-Turbo**: Integrated support for the Z-Image-Turbo model.

## Known Patterns
- **Translations**: Uses a centralized `locales/` directory with `fr.json` and `en.json`. Functions use a `translate()` helper.
- **Logging**: Dedicated logger in `core/logger.py`.
- **Progress Tracking**: Uses a custom `preview_queue` and `progress_queue` to update the Gradio UI in real-time during inference.
