# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [B.0.1] - 2025-02-20
I'm moving the app to beta because a lot of work has been done to integrate new features, console feedback improvements, and bug fixes.
### Added
- We can set in the config.json a default model that will be loaded before launching the interface. See readme for more information
- Added Lora (Low-Rank Adaptation) support, only SDXL 1.0 versions are supported. Ability to load from a list to set a weight and unload after use
- Added an additional parameter to locate the folder that contains the Loras in the config.json file
- Added the possibility to share the Gradio interface for browse on the a other location

### Changed
- StableDiffusionXLPipeline is now passed on parameter to  AutoPipelineForText2Image for the gestion of loras
- Documentation update
- Rearrangement of buttons on the front end

### Fixed
- error message when Loras formats are not compatible with SDXL 1.0
- and a lot of bugs !

## [A.0.4] - 2025-02-11
### Added
- possibility when launching the application to download a model if no model found
- loading of models before launching the gradio interface

### Changed
- colors support and clear feed back in the consol
- support for CUDA 12.6 newer than CUDA 11.8

## [A.0.3] - 2025-02-09

### Added

- An entry for Author in config.json
- Author support for HTML report
- Adding additional samplers see documentation
- Feedback in Gradio for sampler loading


### Fixed

- py was used instead of python se which prevented pip from being updated
- Image generation problem on low vram level configurations: addition of an additional argument max_split_size_mb:38 to try to resolve this problem

### Changed

- Updated the image recognition model to the latest version of MiaoshouAI Florence 2 based PromptGen instead of version 1.5
https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0

### Removed
