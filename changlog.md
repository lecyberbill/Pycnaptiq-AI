# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
