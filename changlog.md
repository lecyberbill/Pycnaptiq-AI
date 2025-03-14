# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [B.1.5] - 2025-03-14
### Added
- **Preview of latent images during inference**: Users can now see latent images while the inference process is running.
- **New photo editing filters**:
  - *Vibrance*: Enhances color intensity selectively without oversaturating skin tones or natural elements.
  - *Curves*: Allows precise control over brightness and contrast by modifying the tonal curve.
  - *Adaptive sharpness*: Dynamically adjusts the sharpness based on image regions to enhance clarity.
  - *Noise*: Adds artistic noise or grain effects to give the images a textured look.
  - *Color gradient*: Applies smooth color transitions for more stylized effects.
  - *Color shift*: Shifts the color palette for creative or dramatic changes in tone and mood.
- **New image formats**:
  - *Portrait*: Optimized for vertical-oriented images, ideal for portraits or posters.
  - *Landscape*: Optimized for horizontal-oriented images, suitable for landscapes or widescreen visuals.

### Changed
- **Interface reorganization**: The layout has been adjusted to accommodate the newly added features.
- **Addition of a `module` folder**: Prepares for the integration of future functionalities by organizing code into modular components.
- **Ongoing code documentation**: Improved inline comments and explanations within the codebase for easier understanding and maintenance.

### Fixed
- **Resolved numerous bugs**: Fixed issues affecting both new and existing features for a smoother user experience.

## [B.1.0] - 2025-03-11
### Added
- Support for English and French: the application can now be used in either language. Modify the `config` file and set `language` to `en` for English or `fr` for French.
- Language selection during installation: it's now possible to change the language during the installation process.
- Predefined style management: users can now select a predefined style.
- Addition of the `style.json` file in the `config` directory for style management.
- Console messages now feature colorization for important information.

### Changed
- Transitioned to using **compel** for precomputing embeddings, removing the 77-token limit for prompts.
- Token counter removed as it's no longer necessary.
- HTML reports are now generated at the end of image generation to improve speed.
- Image writing and report generation tasks are now handled by dedicated threads.

### Fixed
Lot of new buggs !

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
