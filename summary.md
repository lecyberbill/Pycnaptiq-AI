# Pycnaptiq-AI Project

## General Description

*Pycnaptiq-AI ** is a comprehensive and user-friendly desktop application designed for high-quality image generation and manipulation. Primarily leveraging the powerful **Stable Diffusion XL (SDXL)** artificial intelligence model, it provides users with a vast array of tools for artistic creation, photo editing, and creative exploration. Its user interface, developed with Gradio, is designed to be intuitive, organized into thematic tabs for simplified access to its multiple features. One of the project's major strengths lies in its modular architecture, allowing for easy extensibility by adding new capabilities via dynamic modules.

## Key Project Features

*   **Text-to-Image Generation (SDXL):**
    *   **Intuitive Prompts:** Input text descriptions to generate images.
    *   **Image-to-Prompt Generation:** Ability to upload an image for the system to automatically generate a corresponding text description (prompt), which can then be used or modified for image generation.
    *   **AI Prompt Enhancement:** Utilizes a local language model (Qwen) to automatically enrich and detail user's initial ideas, transforming simple prompts into richer descriptions.
    *   **Automatic Translation:** Optional conversion of prompts to English for better performance with SDXL.
    *   **Style System:** Apply and combine predefined styles (cinematic, photo, anime, etc.) to influence the image's aesthetics.
    *   **Negative Prompts:** Specify what NOT to include, with intelligent merging of style negatives and a base negative.
    *   **PAG (Perturbed Attention Guidance):** Advanced option for better prompt alignment and potentially increased image quality.

*   **Inpainting (Image Retouching):**
    *   Modify specific areas of an existing image using a user-drawn mask and a descriptive prompt.

*   **Image-to-Image (SDXL):**
    *   Generate new images based on an input image and a text prompt (often provided via a dedicated module like `image_to_image_mod`).

*   **Model and LoRA Management:**
    *   **SDXL Model Selection:** Easy selection from available `.safetensors` checkpoints.
    *   **VAE Support:** Use external VAEs or the model's built-in VAE.
    *   **Multi-LoRA:** Simultaneously apply multiple LoRA (Low-Rank Adaptation) models with individual adjustment of their influence (weights).
    *   **LoRA Training:** Comprehensive module for creating custom LoRA adapters for SDXL models, with features like automatic captioning, sequential data renaming, and modern PEFT configuration. (Documentation available in `/modules/modules_utils/lora_train_mod_doc/`)

*   **Advanced Generation Control:**
    *   **Samplers:** Wide choice of diffusion algorithms (Euler, DPM++, etc.) to vary the generation process.
    *   **Standard Parameters:** Adjust steps, guidance scale, format/resolution, and seed.
    *   **Multiple Generation:** Create several images at once.

*   **Preset Management:**
    *   **Save:** Save complete generation configurations (prompt, model, LoRAs, sampler, etc.) as reusable presets.
    *   **Navigation:** Interface to list, search, filter, load, rate, and delete presets, with an associated image preview.

*   **Batch Processing:**
    *   Automated execution of multiple generation tasks from a JSON configuration file.

*   **User Interface (UI) and User Experience (UX):**
    *   **Gradio Interface:** Local web application, accessible via browser, organized into clear tabs (Generation, Inpainting, Presets, Modules).
    *   **Real-Time Previews:** Display of the image being generated at regular intervals.
    *   **Detailed Progress Bars:** Visual tracking of task progress.
    *   **Internationalization:** Multilingual interface (initially French/English) thanks to translation files.
    *   **Integrated Memory Management:** UI accordion for monitoring system resources (RAM, CPU, VRAM, GPU) and unloading models.
    *   **User Feedback:** Clear informational, warning, and error messages.
    *   **Global Configuration:** `config.json` file for paths, default prompts, etc.

*   **Output and Metadata:**
    *   **Image Saving:** In PNG, JPG, or WEBP formats.
    *   **Full Metadata Integration:** Generation parameters (prompt, model, seed, styles, LoRAs, etc.) are embedded in image files (XMP, EXIF, PNG info) for easy traceability and reproducibility.
    *   **HTML Reports:** Automatic generation of an HTML report per session, listing produced images and their metadata.

## Architecture and Key Technical Components

*   **`ModelManager`:**
    *   Core of resource management: intelligent loading/unloading of SDXL models, VAEs, and LoRAs to optimize VRAM usage.
    *   Manages different types of pipelines (`StableDiffusionXLPipeline`, `StableDiffusionXLInpaintPipeline`, `StableDiffusionXLImg2ImgPipeline`, and potentially `SanaSprintPipeline`).
    *   Integrates `Compel` for advanced prompt interpretation (keyword weighting). Enhanced to handle new model types and improved LoRA application.

*   **`PresetManager`:**
    *   Responsible for the persistence (saving/loading) of user presets and module states.

*   **`GestionModule` (Modular System):**
    *   Allows extending the application's functionality by adding independent modules.
    *   Each module can have its own UI tab, Python dependencies (installed automatically), and translations.
    *   Users can enable/disable modules.

*   **Generation Logic (`cyberbill_SDXL.py`, `pipeline_executor.py`):**
    *   Orchestration of the generation process, including prompt preparation, style application, `diffusers` pipeline configuration, and callback invocation.
    *   Use of `ThreadPoolExecutor` for asynchronous operations (image saving, HTML report generation).

*   **`image_prompter` (in `core/`):**
    *   Module responsible for analyzing a user-provided image to extract a text description (prompt).
*   **`llm_prompter_util` (in `Utils/`):**
    *   Utility for AI prompt enhancement using various Hugging Face Language Models.
    *   Optimized for CPU usage to conserve VRAM and improved parsing for broader model compatibility.
*   **`gest_mem` (in `Utils/`):** Utility for system resource monitoring (RAM, CPU, VRAM, GPU) displayed in the UI.
## Technologies Used

*   **Language:** Python
*   **AI/Deep Learning Framework:** PyTorch
*   **Diffusion Models:** Hugging Face `diffusers` (for Stable Diffusion XL)
*   **Prompt Processing:** `Compel`
*   **User Interface:** Gradio
*   **Image Manipulation:** Pillow (PIL)
*   **Language Models (Prompt Enhancement & Image-to-Prompt):** Hugging Face `transformers` (for Qwen and the image-to-prompt model).

## Project Strengths

*   **Comprehensive Solution:** Covers a wide range of SDXL image generation needs.
*   **User-Friendliness:** Accessible Gradio interface, even for less technical users.
*   **Extensive Customization:** Great control over models, LoRAs, samplers, and styles.
*   **Extensibility:** Modular architecture allowing for easy addition of new features.
*   **Efficient Resource Management:** `ModelManager` optimized for VRAM.
*   **Reproducibility:** Exhaustive saving of generation metadata.
*   **Unique Features:** Integrated LLM prompt enhancement, image-to-prompt generation, advanced preset management.

## Module Management

The application's core can be enriched by external modules. Each module can bring new functionalities, often presented in its own tab. The system manages loading, dependencies (with attempted automatic installation), and module-specific translations. Users can enable or disable modules according to their needs, and these preferences are saved.

## Available Modules

Here is a list of modules currently integrated into the project:

---

### 1. RemBG (Background Removal)

*   **Module Name (internal):** `RemBG_mod`
*   **Description:** This module allows for automatic background removal from an image.
*   **Key Features:**
    *   Uses the `rembg` library for background detection and removal.
    *   Optimizes VRAM usage by unloading the main SDXL model (`ModelManager`) before processing.
    *   Saves the resulting image with a transparent background in PNG format.
    *   Simple interface with a drop zone for the input image and a display area for the processed image.

---

### 2. Civitai Browser

*   **Module Name (internal):** `civitai_browser_mod`
*   **Description:** Integrates a browser to explore images and models available on Civitai.com directly from the application.
*   **Key Features:**
    *   Search for images on Civitai with various filters (number of results, NSFW content type, sort, period).
    *   Pagination to navigate through search results.
    *   Displays images in an interactive HTML gallery.
    *   Ability to view and copy metadata associated with each image (prompt, model used, etc.).

---

### 3. Sana Sprint Generator

*   **Module Name (internal):** `sana_sprint_mod`
*   **Description:** Module dedicated to ultra-fast image generation using the Sana Sprint model. This model is smaller and faster than SDXL, optimized for 1024x1024 pixel output.
*   **Key Features:**
    *   Image generation from text prompts.
    *   Support for predefined styles to influence aesthetics.
    *   Image-to-Prompt generation option.
    *   Ability to generate multiple images in a single session.
    *   Uses a specific `SanaSprintPipeline`, managed via `ModelManager`.
    *   *Note:* This module has different customization capabilities than SDXL (e.g., more limited LoRA or negative prompt management).

---

### 4. Batch Generator

*   **Module Name (internal):** `batch_generator_mod`
*   **Description:** Tool to create and configure JSON files for batch processing. These files are then used by the "Batch Processing" feature in the main SDXL generation tab.
*   **Key Features:**
    *   User interface to define parameters for each generation task (model, VAE, prompt, negative prompt, styles, sampler, steps, CFG, seed, dimensions, LoRAs, output filename, etc.).
    *   Add multiple configured tasks to a batch list.
    *   Automatic generation and saving of a JSON file (e.g., `batch_001.json`) in the `Output/json_batch_files` folder (configurable).
    *   Displays the generated JSON in the interface.

---

### 5. Image Enhancement

*   **Module Name (internal):** `ImageEnhancement_mod`
*   **Description:** Groups several AI-based tools to improve the quality and appearance of existing images.
*   **Key Features:**
    *   **Colorization:** Automatically adds color to black and white images (uses `damo/cv_ddcolor_image-colorization` from ModelScope).
    *   **Upscale:** Increases image resolution (fixed 4x factor, uses `CompVis/ldm-super-resolution-4x-openimages` from Diffusers).
    *   **Restoration (OneRestore):** Attempts to restore degraded images (old photos, visual artifacts, etc.) using the OneRestore model (local models).
    *   **Auto Retouch:** Applies simple automatic adjustments like contrast, sharpness, and saturation enhancement.
    *   Unloads the main SDXL model (`ModelManager`) before executing these tasks for better resource management.

---

### 6. ImageToText

*   **Module Name (internal):** `ImageToText_mod`
*   **Description:** Utility module to generate text descriptions or tags from images using the Florence-2 model.
*   **Key Features:**
    *   Selection of specific Florence-2 tasks (detailed caption, tags, object detection, etc.).
    *   Recursive directory scanning for images.
    *   Filename filtering (e.g., `*.png`).
    *   Option to overwrite existing text files.
    *   "Unload Model" button to free VRAM.
    *   Generates a detailed JSON report of its operations.

---

### 7. LoRA Training

*   **Module Name (internal):** `LoRATraining_mod`
*   **Description:** A comprehensive module for training LoRA (Low-Rank Adaptation) adapters for SDXL models.
*   **Key Features:**
    *   Separate UI for data preparation (including optional automatic captioning with Florence-2, or copying existing `.txt` files, and sequential file renaming) and training.
    *   Supports SDXL-specific training logic like `add_time_ids`, VAE encoding considerations, and gradient clipping.
    *   Modern PEFT configuration with `unet.add_adapter()` and `text_encoder.add_adapter()`.
    *   Saves final LoRA as a single `.safetensors` file.
    *   User-friendly UI with dropdowns for learning rate, base model, optimizer, scheduler, and mixed precision.
    *   Detailed documentation available in `/modules/modules_utils/lora_train_mod_doc/`.

---

### 8. CogView3-Plus Generator

*   **Module Name (internal):** `CogView3Plus_mod`
*   **Description:** Dedicated tab for image generation using the `THUDM/CogView3-Plus-3B` model.
*   **Key Features:**
    *   Text-to-image generation with style mixing and prompt translation.
    *   Asynchronous generation for a responsive UI.
    *   Explicit memory cleanup after each batch.
    *   Model configurations (offload, slicing, tiling) are managed by the central ModelManager.
    *   Saves images with specific CogView3-Plus metadata.

---

### 9. CogView4 Generator

*   **Module Name (internal):** `CogView4_mod`
*   **Description:** Dedicated tab for image generation using the `THUDM/CogView4-6B` model.
*   **Key Features:**
    *   Similar to CogView3-Plus, offers text-to-image generation with styles.
    *   Uses asynchronous generation.
    *   Specific model configurations (CPU offload, VAE slicing/tiling) are applied after the pipeline is loaded.
    *   Saves images with specific CogView4 metadata.

---

### 6. Image Watermark
### 10. Image Watermark

*   **Module Name (internal):** `ImageWatermark_mod`
*   **Description:** Allows adding text or graphic watermarks to one or more images.
*   **Key Features:**
    *   Support for text watermarks with options for content, font size (basic), and color.
    *   Support for image watermarks by uploading an image to use as a watermark, with control over its scale.
    *   Common parameters: opacity, position (corners, center, or tiled), margin from edges, and watermark rotation.
    *   Two operating modes: single image processing or batch processing of an image folder.

---

### 11. Civitai Downloader

*   **Module Name (internal):** `civitai_downloader_mod`
*   **Description:** Facilitates searching, viewing, and downloading models (checkpoints, LoRAs, VAEs, Textual Inversions, etc.) directly from Civitai.com.
*   **Key Features:**
    *   Search interface with various filters: text query, model type, sort method (popularity, date), period, NSFW content level, and SDXL model-specific filter.
    *   Optional use of a Civitai API key (read from `config.json`).
    *   Displays results as clickable cards.
    *   When selecting a model, displays its description, different versions, and downloadable files for each version.
    *   Direct download of model files into the appropriate application directories (`models/Stable-diffusion`, `models/Lora`, `models/VAE`, etc.), managed by `ModelManager` for paths.

---

### 12. Image to Image

*   **Module Name (internal):** `image_to_image_mod`
*   **Description:** Allows generating new images based on an input image and a text prompt, using an SDXL Image-to-Image pipeline.
*   **Key Features:**
    *   Uses a standard SDXL model (not specific to inpainting for this task), loaded via `ModelManager`.
    *   Takes a source image and a text description (prompt) as input.
    *   "Strength" parameter to control the influence of the original image on the final result.
    *   Support for predefined styles, automatic prompt translation, and VAE selection.
    *   Two operating modes: single image processing or batch processing of an image folder.
    *   In batch mode, a preview of the image being processed is displayed.

---

### 13. Photo Editing (Basic)

*   **Module Name (internal):** `photo_editing_mod`
*   **Description:** Provides an interface for basic photo adjustments and filters, primarily operated by the Pillow library.
*   **Key Features:**
    *   **Transformations:** Rotation (90° increments), mirror (horizontal, vertical).
    *   **Adjustments:** Contrast, saturation, color intensity, Gaussian blur, sharpness, convert to black and white.
    *   **Effects:** Vibrance, hue modification.
    *   **Special Filters:** Sepia, contour, negative, posterize, solarize, emboss, pixelate, vignette, mosaic.
    *   **Other Tools:** Basic curves adjustment, Unsharp Mask, add noise, apply color gradient, RGB channel shift.

---

### 14. Test Module

*   **Module Name (internal):** `test_module_mod`
*   **Description:** A skeleton module designed as an example or starting point for developing new custom modules. It illustrates the basic structure of a module, its initialization, and how it can interact with the main system.
*   **Key Features:**
    *   **Basic Structure:** Includes a Python file (`test_module_mod.py`) and a JSON metadata file (`test_module_mod.json`).
    *   **Initialization:** The `initialize` function is called by `GestionModule` and returns an instance of the `TestModule` class.
    *   **Access to Global Components:** The `TestModule` constructor receives and stores:
        *   `initial_module_translations`: Merged translations (global + module-specific) for the current language.
        *   `model_manager_instance`: The global `ModelManager` instance, allowing interaction with loaded models.
        *   `gestionnaire`: The `GestionModule` instance itself, for potential advanced interactions.
        *   `global_config`: The application's global configuration dictionary.
    *   **User Interface (UI):** The `create_tab` method generates a simple tab in Gradio with:
        *   A translated title and description.
        *   A text input field.
        *   A display field for processed text.
        *   A button to trigger simple processing.
    *   **Example Logic:** The button's callback function (`process_text`) shows how to:
        *   Access global configuration (e.g., `self.global_config.get("SAVE_DIR")`).
        *   Check the availability of the main pipeline via `self.model_manager.get_current_pipe()`.
        *   Perform basic processing on the input text.
    *   **Translations:** Uses provided translations to display interface labels (e.g., `translate("test_module_tab_name", self.module_translations)`). Translation keys specific to this module should be defined in `test_module_mod.json`.
    *   **Educational Purpose:** Primarily serves to demonstrate how to create a new module, how it's integrated, and how it can access the application's shared resources. It does not perform complex image processing or heavy operations itself.

### 15. Re-Lighting (Image Re-illumination)

*   **Module Name (internal):** `reLighting_mod`
*   **Description:** This module allows users to adjust or completely change the lighting conditions of an existing image. It can be used to simulate different light sources, times of day, or artistic lighting effects.
*   **Key Features:**
    *   **Light Source Control:** Ability to define virtual light sources (e.g., point, directional, spot) with parameters like position, color, and intensity.
    *   **Shadow Manipulation:** Tools to adjust existing shadows or generate new ones consistent with the new lighting setup.
    *   **Ambient Light Adjustment:** Control over the overall ambient lighting of the scene.
    *   **AI-Powered Options (Potentially):** May leverage AI models to predict realistic light interactions and reflections for more natural results.
    *   **Preview:** Real-time or quick preview of lighting changes.
    *   Resource management, potentially unloading SDXL if using a dedicated re-lighting model or heavy CPU/GPU processing.

### 16. FLUX.1-Schnell Generator

*   **Module Name (internal):** `FluxSchnell_mod`
*   **Description:** Module for ultra-fast image generation using FLUX.1-Schnell models (e.g., `black-forest-labs/FLUX.1-schnell`).
*   **Key Features:**
    *   Supports both **Text-to-Image** and **Image-to-Image** generation.
    *   Utilizes `FluxPipeline` and `FluxImg2ImgPipeline`.
    *   Offers specific FLUX-optimized resolutions.
    *   Integrates LoRA support (up to 2), style selection, Image-to-Prompt (Florence-2), and LLM Prompt Enhancement.
    *   Managed by `ModelManager` for efficient resource handling.
    *   Saves images with detailed metadata.
---

This overview should now be much more complete and accurately reflect the full scope of **Cyberbill Image Generator**'s capabilities!





# Pycnaptiq-AI

## Description Générale

**Pycnaptiq-AI** est une application de bureau complète et conviviale, conçue pour la génération et la manipulation d'images de haute qualité. S'appuyant principalement sur le puissant modèle d'intelligence artificielle **Stable Diffusion XL (SDXL)**, elle met à disposition des utilisateurs une vaste gamme d'outils pour la création artistique, la retouche photo et l'exploration créative. Son interface utilisateur, développée avec Gradio, est pensée pour être intuitive, organisée en onglets thématiques pour un accès simplifié aux multiples fonctionnalités. Un des atouts majeurs du projet réside dans son architecture modulaire, permettant une extensibilité aisée par l'ajout de nouvelles capacités via des modules dynamiques.

## Fonctionnalités Clés du Projet

*   **Génération d'Images Text-to-Image (SDXL) :**
    *   **Prompts Intuitifs :** Saisie de descriptions textuelles pour générer des images.
    *   **Génération de Prompt à partir d'une Image :** Possibilité de télécharger une image pour que le système génère automatiquement une description textuelle (prompt) correspondante, qui peut ensuite être utilisée ou modifiée pour la génération d'images.
    *   **Amélioration IA des Prompts :** Utilisation d'un modèle de langage local (Qwen) pour enrichir et détailler automatiquement les idées de l'utilisateur, transformant des prompts simples en descriptions plus riches.
    *   **Traduction Automatique :** Conversion optionnelle des prompts en anglais pour une meilleure performance avec SDXL.
    *   **Système de Styles :** Application et combinaison de styles prédéfinis (cinématique, photo, anime, etc.) pour influencer l'esthétique de l'image.
    *   **Prompts Négatifs :** Spécification de ce qu'il ne faut PAS inclure, avec fusion intelligente des négatifs des styles et d'un négatif de base.
    *   **PAG (Perturbed Attention Guidance) :** Option avancée pour un meilleur alignement du prompt et une qualité d'image potentiellement accrue.

*   **Inpainting (Retouche d'Image) :**
    *   Modification de zones spécifiques d'une image existante à l'aide d'un masque dessiné par l'utilisateur et d'un prompt descriptif.

*   **Image-to-Image (SDXL) :**
    *   Génération de nouvelles images en se basant sur une image d'entrée et un prompt textuel (fonctionnalité souvent fournie via un module dédié comme `image_to_image_mod`).

*   **Gestion des Modèles et LoRAs :**
    *   **Choix de Modèles SDXL :** Sélection facile parmi les checkpoints `.safetensors` disponibles.
    *   **Support VAE :** Utilisation de VAEs externes ou du VAE intégré au modèle.
    *   **Multi-LoRA :** Application simultanée de plusieurs modèles LoRA (Low-Rank Adaptation) avec ajustement individuel de leur influence (poids).
    *   **Entraînement LoRA :** Module complet pour la création d'adaptateurs LoRA personnalisés pour les modèles SDXL, avec des fonctionnalités telles que le *captioning* automatique, le renommage séquentiel des données et une configuration PEFT moderne. (Documentation disponible dans `/modules/modules_utils/lora_train_mod_doc/`)

*   **Contrôle Avancé de la Génération :**
    *   **Samplers :** Large choix d'algorithmes de diffusion (Euler, DPM++, etc.) pour varier le processus de génération.
    *   **Paramètres Standards :** Ajustement du nombre d'étapes (steps), de l'échelle de guidage (guidance scale), du format/résolution et du seed.
    *   **Génération Multiple :** Création de plusieurs images en une seule fois.

*   **Gestion des Presets :**
    *   **Sauvegarde :** Enregistrement des configurations complètes de génération (prompt, modèle, LoRAs, sampler, etc.) comme presets réutilisables.
    *   **Navigation :** Interface pour lister, rechercher, filtrer, charger, noter et supprimer des presets, avec prévisualisation de l'image associée.

*   **Traitement par Lots (Batch Processing) :**
    *   Exécution automatisée de multiples tâches de génération à partir d'un fichier de configuration JSON.

*   **Interface Utilisateur (UI) et Expérience Utilisateur (UX) :**
    *   **Interface Gradio :** Application web locale, accessible via navigateur, organisée en onglets clairs (Génération, Inpainting, Presets, Modules).
    *   **Prévisualisations en Temps Réel :** Affichage de l'image en cours de génération à intervalles réguliers.
    *   **Barres de Progression Détaillées :** Suivi visuel de l'avancement des tâches.
    *   **Internationalisation :** Interface multilingue (Français/Anglais initialement) grâce à des fichiers de traduction.
    *   **Feedback Utilisateur :** Messages d'information, d'avertissement et d'erreur clairs.
    *   **Gestion de la Mémoire Intégrée :** Accordéon UI pour surveiller les ressources système (RAM, CPU, VRAM, GPU) et décharger les modèles.
    *   **Configuration Globale :** Fichier `config.json` pour les chemins, prompts par défaut, etc.

*   **Sortie et Métadonnées :**
    *   **Sauvegarde des Images :** Aux formats PNG, JPG, ou WEBP.
    *   **Intégration Complète des Métadonnées :** Les paramètres de génération (prompt, modèle, seed, styles, LoRAs, etc.) sont intégrés dans les fichiers images (XMP, EXIF, PNG info) pour une traçabilité et une reproductibilité aisées.
    *   **Rapports HTML :** Génération automatique d'un rapport HTML par session, listant les images produites et leurs métadonnées.

## Architecture et Composants Techniques Clés

*   **`ModelManager` :**
    *   Cœur de la gestion des ressources : chargement/déchargement intelligent des modèles SDXL, VAEs, et LoRAs pour optimiser l'utilisation de la VRAM.
    *   Gère différents types de pipelines (`StableDiffusionXLPipeline`, `StableDiffusionXLInpaintPipeline`, `StableDiffusionXLImg2ImgPipeline` et potentiellement `SanaSprintPipeline`).
    *   Intègre `Compel` pour une interprétation avancée des prompts (pondération des mots-clés). Amélioré pour gérer de nouveaux types de modèles et une application LoRA plus robuste.

*   **`PresetManager` :**
    *   Responsable de la persistance (sauvegarde/chargement) des presets utilisateur et de l'état des modules.

*   **`GestionModule` (Système Modulaire) :**
    *   Permet d'étendre les fonctionnalités de l'application par l'ajout de modules indépendants.
    *   Chaque module peut avoir son propre onglet UI, ses dépendances Python (installées automatiquement) et ses traductions.
    *   Les utilisateurs peuvent activer/désactiver les modules.

*   **Logique de Génération (`cyberbill_SDXL.py`, `pipeline_executor.py`) :**
    *   Orchestration du processus de génération, incluant la préparation des prompts, l'application des styles, la configuration du pipeline `diffusers`, et l'appel aux callbacks.
    *   Utilisation de `ThreadPoolExecutor` pour les opérations asynchrones (sauvegarde d'images, génération de rapports HTML).

*   **`image_prompter` (dans `core/`) :**
    *   Module responsable de l'analyse d'une image fournie par l'utilisateur pour en extraire une description textuelle (prompt).
*   **`llm_prompter_util` (dans `Utils/`) :**
    *   Utilitaire pour l'amélioration des prompts par IA utilisant divers modèles de langage Hugging Face.
    *   Optimisé pour une utilisation CPU afin de préserver la VRAM et amélioration du parsing pour une compatibilité plus large des modèles.
*   **`gest_mem` (dans `Utils/`) :** Utilitaire pour la surveillance des ressources système (RAM, CPU, VRAM, GPU) affichées dans l'interface utilisateur.
## Technologies Utilisées

*   **Langage :** Python
*   **Framework IA/Deep Learning :** PyTorch
*   **Diffusion Models :** Hugging Face `diffusers` (pour Stable Diffusion XL)
*   **Prompt Processing :** `Compel`
*   **Interface Utilisateur :** Gradio
*   **Manipulation d'Images :** Pillow (PIL)
*   **Modèles de Langage (Prompt Enhancement & Image-to-Prompt) :** Hugging Face `transformers` (pour Qwen et le modèle d'image-to-prompt).

## Points Forts du Projet

*   **Solution Complète :** Couvre un large éventail de besoins en génération d'images SDXL.
*   **Convivialité :** Interface Gradio accessible, même pour les utilisateurs moins techniques.
*   **Personnalisation Poussée :** Grand contrôle sur les modèles, LoRAs, samplers, et styles.
*   **Extensibilité :** Architecture modulaire permettant d'ajouter facilement de nouvelles fonctionnalités.
*   **Gestion Efficace des Ressources :** `ModelManager` optimisé pour la VRAM.
*   **Reproductibilité :** Sauvegarde exhaustive des métadonnées de génération.
*   **Fonctionnalités Uniques :** Amélioration de prompt par LLM intégré, génération de prompt depuis une image, gestion avancée des presets.

## Gestion des Modules

Le cœur de l'application peut être enrichi par des modules externes. Chaque module peut apporter de nouvelles fonctionnalités, souvent présentées dans leur propre onglet. Le système gère le chargement, les dépendances (avec tentative d'installation automatique) et les traductions spécifiques à chaque module. Les utilisateurs peuvent activer ou désactiver les modules selon leurs besoins, et ces préférences sont sauvegardées.

## Modules Disponibles

Voici une liste des modules actuellement intégrés au projet :

---

### 1. RemBG (Suppression d'Arrière-plan)

*   **Nom du module (interne) :** `RemBG_mod`
*   **Description :** Ce module permet de supprimer automatiquement l'arrière-plan d'une image.
*   **Fonctionnalités Clés :**
    *   Utilise la bibliothèque `rembg` pour la détection et la suppression de l'arrière-plan.
    *   Optimise l'utilisation de la mémoire VRAM en déchargeant le modèle SDXL principal (`ModelManager`) avant le traitement.
    *   Sauvegarde l'image résultante avec un fond transparent au format PNG.
    *   Interface simple avec une zone de dépôt pour l'image d'entrée et une zone d'affichage pour l'image traitée.

---

### 2. Civitai Browser (Navigateur Civitai)

*   **Nom du module (interne) :** `civitai_browser_mod`
*   **Description :** Intègre un navigateur pour explorer les images et modèles disponibles sur le site Civitai.com directement depuis l'application.
*   **Fonctionnalités Clés :**
    *   Recherche d'images sur Civitai avec divers filtres (nombre de résultats, type de contenu NSFW, tri, période).
    *   Pagination pour naviguer à travers les résultats de recherche.
    *   Affichage des images dans une galerie interactive HTML.
    *   Possibilité de visualiser les métadonnées associées à chaque image (prompt, modèle utilisé, etc.) et de les copier.

---

### 3. Sana Sprint Generator (Générateur Sana Sprint)

*   **Nom du module (interne) :** `sana_sprint_mod`
*   **Description :** Module dédié à la génération d'images ultra-rapide utilisant le modèle Sana Sprint. Ce modèle est plus petit et plus véloce que SDXL, optimisé pour une sortie en 1024x1024 pixels.
*   **Fonctionnalités Clés :**
    *   Génération d'images à partir de prompts textuels.
    *   Support des styles prédéfinis pour influencer l'esthétique.
    *   Option de génération de prompt à partir d'une image (Image-to-Prompt).
    *   Capacité de générer plusieurs images en une seule session.
    *   Utilise un pipeline `SanaSprintPipeline` spécifique, géré via `ModelManager`.
    *   *Note :* Ce module a des capacités de personnalisation différentes de SDXL (par exemple, gestion des LoRAs ou des prompts négatifs plus limitée).

---

### 4. Batch Generator (Générateur de Lots)

*   **Nom du module (interne) :** `batch_generator_mod`
*   **Description :** Outil pour créer et configurer des fichiers JSON destinés au traitement par lots. Ces fichiers sont ensuite utilisés par la fonctionnalité de "Traitement par Lots" de l'onglet principal de génération SDXL.
*   **Fonctionnalités Clés :**
    *   Interface utilisateur pour définir les paramètres de chaque tâche de génération (modèle, VAE, prompt, prompt négatif, styles, sampler, étapes, CFG, seed, dimensions, LoRAs, nom de fichier de sortie, etc.).
    *   Ajout de multiples tâches configurées à une liste de lots.
    *   Génération et sauvegarde automatique d'un fichier JSON (ex: `batch_001.json`) dans le dossier `Output/json_batch_files` (configurable).
    *   Affichage du JSON généré dans l'interface.

---

### 5. Image Enhancement (Amélioration d'Image)

*   **Nom du module (interne) :** `ImageEnhancement_mod`
*   **Description :** Regroupe plusieurs outils basés sur l'IA pour améliorer la qualité et l'apparence des images existantes.
*   **Fonctionnalités Clés :**
    *   **Colorisation :** Ajoute automatiquement de la couleur à des images en noir et blanc (utilise `damo/cv_ddcolor_image-colorization` de ModelScope).
    *   **Upscale (Agrandissement) :** Augmente la résolution des images (facteur 4x fixe, utilise `CompVis/ldm-super-resolution-4x-openimages` de Diffusers).
    *   **Restauration (OneRestore) :** Tente de restaurer des images dégradées (anciennes photos, artefacts visuels, etc.) en utilisant le modèle OneRestore (modèles locaux).
    *   **Retouche Auto :** Applique des ajustements automatiques simples comme l'amélioration du contraste, de la netteté et de la saturation.
    *   Décharge le modèle SDXL principal (`ModelManager`) avant d'exécuter ces tâches pour une meilleure gestion des ressources.

---

### 6. ImageToText

*   **Nom du module (interne) :** `ImageToText_mod`
*   **Description :** Module utilitaire pour générer des descriptions textuelles ou des mots-clés à partir d'images en utilisant le modèle Florence-2.
*   **Fonctionnalités Clés :**
    *   Sélection de tâches spécifiques de Florence-2 (description détaillée, mots-clés, détection d'objets, etc.).
    *   Scan récursif de répertoires pour les images.
    *   Filtrage par nom de fichier (ex: `*.png`).
    *   Option d'écrasement des fichiers texte existants.
    *   Bouton "Décharger le modèle" pour libérer la VRAM.
    *   Génère un rapport JSON détaillé de ses opérations.

---

### 7. Entraînement LoRA

*   **Nom du module (interne) :** `LoRATraining_mod`
*   **Description :** Module complet pour l'entraînement d'adaptateurs LoRA (Low-Rank Adaptation) pour les modèles SDXL.
*   **Fonctionnalités Clés :**
    *   Interface utilisateur séparée pour la préparation des données (incluant le *captioning* automatique optionnel avec Florence-2, ou la copie de fichiers `.txt` existants, et le renommage séquentiel des fichiers) et l'entraînement.
    *   Supporte la logique d'entraînement spécifique à SDXL comme les `add_time_ids`, les considérations d'encodage VAE, et le *gradient clipping*.
    *   Configuration PEFT moderne avec `unet.add_adapter()` et `text_encoder.add_adapter()`.
    *   Sauvegarde le LoRA final en un unique fichier `.safetensors`.
    *   Interface utilisateur conviviale avec des menus déroulants pour le taux d'apprentissage, le modèle de base, l'optimiseur, le planificateur et la précision mixte.
    *   Documentation détaillée disponible dans `/modules/modules_utils/lora_train_mod_doc/`.

---

### 8. Générateur CogView3-Plus

*   **Nom du module (interne) :** `CogView3Plus_mod`
*   **Description :** Onglet dédié pour la génération d'images avec le modèle `THUDM/CogView3-Plus-3B`.
*   **Fonctionnalités Clés :**
    *   Génération texte-vers-image avec mélange de styles et traduction de prompt.
    *   Génération asynchrone pour une interface utilisateur réactive.
    *   Nettoyage explicite de la mémoire après chaque lot.
    *   Les configurations du modèle (déchargement, découpage, tuilage) sont gérées par le ModelManager central.
    *   Sauvegarde les images avec des métadonnées spécifiques à CogView3-Plus.

---

### 9. Générateur CogView4

*   **Nom du module (interne) :** `CogView4_mod`
*   **Description :** Onglet dédié pour la génération d'images avec le modèle `THUDM/CogView4-6B`.
*   **Fonctionnalités Clés :**
    *   Similaire à CogView3-Plus, offre la génération texte-vers-image avec styles.
    *   Utilise la génération asynchrone.
    *   Des configurations spécifiques au modèle (déchargement CPU, découpage/tuilage VAE) sont appliquées après le chargement du pipeline.
    *   Sauvegarde les images avec des métadonnées spécifiques à CogView4.

---

### 6. Image Watermark (Filigrane d'Image)
### 10. Image Watermark (Filigrane d'Image)

*   **Nom du module (interne) :** `ImageWatermark_mod`
*   **Description :** Permet d'ajouter des filigranes (watermarks) textuels ou graphiques sur une ou plusieurs images.
*   **Fonctionnalités Clés :**
    *   Support des filigranes textuels avec options de contenu, taille de police (basique), et couleur.
    *   Support des filigranes image par le téléchargement d'une image à utiliser comme filigrane, avec contrôle de son échelle.
    *   Paramètres communs : opacité, position (coins, centre, ou en mosaïque), marge par rapport aux bords, et rotation du filigrane.
    *   Deux modes de fonctionnement : traitement d'une image unique ou traitement par lot d'un dossier d'images.

---

### 11. Civitai Downloader (Téléchargeur Civitai)

*   **Nom du module (interne) :** `civitai_downloader_mod`
*   **Description :** Facilite la recherche, la visualisation et le téléchargement de modèles (checkpoints, LoRAs, VAEs, Textual Inversions, etc.) directement depuis le site Civitai.com.
*   **Fonctionnalités Clés :**
    *   Interface de recherche avec filtres variés : requête textuelle, type de modèle, méthode de tri (popularité, date), période, niveau de contenu NSFW, et filtre spécifique pour les modèles SDXL.
    *   Utilisation optionnelle d'une clé API Civitai (lue depuis `config.json`).
    *   Affichage des résultats sous forme de cartes cliquables.
    *   Lors de la sélection d'un modèle, affichage de sa description, de ses différentes versions et des fichiers téléchargeables pour chaque version.
    *   Téléchargement direct des fichiers de modèles dans les répertoires appropriés de l'application (`models/Stable-diffusion`, `models/Lora`, `models/VAE`, etc.), gérés par `ModelManager` pour les chemins.

---

### 12. Image to Image (Image vers Image)

*   **Nom du module (interne) :** `image_to_image_mod`
*   **Description :** Permet de générer de nouvelles images en se basant sur une image d'entrée et un prompt textuel, en utilisant un pipeline SDXL Image-to-Image.
*   **Fonctionnalités Clés :**
    *   Utilise un modèle SDXL standard (non spécifique à l'inpainting pour cette tâche), chargé via `ModelManager`.
    *   Prend en entrée une image source et une description textuelle (prompt).
    *   Paramètre de "force" (strength) pour contrôler l'influence de l'image d'origine sur le résultat final.
    *   Support des styles prédéfinis, de la traduction automatique du prompt, et de la sélection d'un VAE.
    *   Deux modes de fonctionnement : traitement d'une image unique ou traitement par lot d'un dossier d'images.
    *   En mode batch, un aperçu de l'image en cours de traitement est affiché.

---

### 13. Photo Editing (Retouche Photo Basique)

*   **Nom du module (interne) :** `photo_editing_mod`
*   **Description :** Fournit une interface pour des ajustements et des filtres photo basiques, opérés principalement par la bibliothèque Pillow.
*   **Fonctionnalités Clés :**
    *   **Transformations :** Rotation (par paliers de 90°), effet miroir (horizontal, vertical).
    *   **Ajustements :** Contraste, saturation, intensité des couleurs, flou gaussien, netteté, conversion en noir et blanc.
    *   **Effets :** Vibrance, modification de la teinte.
    *   **Filtres Spéciaux :** Sépia, contour, négatif, postérisation, solarisation, emboss (relief), pixelisation, vignette, mosaïque.
    *   **Autres Outils :** Ajustement des courbes (basique), netteté adaptative (Unsharp Mask), ajout de bruit, application d'un dégradé de couleur, décalage des canaux de couleur (RGB).

---

### 14. Test Module (Module de Test)

*   **Nom du module (interne) :** `test_module_mod`
*   **Description :** Un module squelette conçu comme exemple ou point de départ pour le développement de nouveaux modules personnalisés. Il illustre la structure de base d'un module, son initialisation, et comment il peut interagir avec le système principal.
*   **Fonctionnalités Clés :**
    *   **Structure de Base :** Comprend un fichier Python (`test_module_mod.py`) et un fichier JSON de métadonnées (`test_module_mod.json`).
    *   **Initialisation :** La fonction `initialize` est appelée par `GestionModule` et retourne une instance de la classe `TestModule`.
    *   **Accès aux Composants Globaux :** Le constructeur de `TestModule` reçoit et stocke :
        *   `initial_module_translations`: Les traductions fusionnées (globales + spécifiques au module) pour la langue courante.
        *   `model_manager_instance`: L'instance globale de `ModelManager`, permettant d'interagir avec les modèles chargés.
        *   `gestionnaire`: L'instance de `GestionModule` elle-même, pour d'éventuelles interactions avancées.
        *   `global_config`: Le dictionnaire de configuration globale de l'application.
    *   **Interface Utilisateur (UI) :** La méthode `create_tab` génère un onglet simple dans Gradio avec :
        *   Un titre et une description traduits.
        *   Un champ de saisie de texte.
        *   Un champ d'affichage du texte traité.
        *   Un bouton pour déclencher un traitement simple.
    *   **Exemple de Logique :** La fonction de rappel du bouton (`process_text`) montre comment :
        *   Accéder à la configuration globale (ex: `self.global_config.get("SAVE_DIR")`).
        *   Vérifier la disponibilité du pipeline principal via `self.model_manager.get_current_pipe()`.
        *   Effectuer un traitement basique sur le texte d'entrée.
    *   **Traductions :** Utilise les traductions fournies pour afficher les libellés de l'interface (ex: `translate("test_module_tab_name", self.module_translations)`). Les clés de traduction spécifiques à ce module doivent être définies dans `test_module_mod.json`.
    *   **Objectif Pédagogique :** Sert principalement à démontrer comment créer un nouveau module, comment il est intégré, et comment il peut accéder aux ressources partagées de l'application. Il n'effectue pas de traitement d'image complexe par lui-même.



### 15. Re-Lighting (Ré-éclairage d'Image)

*   **Nom du module (interne) :** `reLighting_mod`
*   **Description :** Ce module permet aux utilisateurs d'ajuster ou de modifier complètement les conditions d'éclairage d'une image existante. Il peut être utilisé pour simuler différentes sources de lumière, moments de la journée, ou effets d'éclairage artistiques.
*   **Fonctionnalités Clés :**
    *   **Contrôle des Sources Lumineuses :** Capacité à définir des sources de lumière virtuelles (ex: ponctuelle, directionnelle, spot) avec des paramètres tels que la position, la couleur et l'intensité.
    *   **Manipulation des Ombres :** Outils pour ajuster les ombres existantes ou en générer de nouvelles, cohérentes avec la nouvelle configuration d'éclairage.
    *   **Ajustement de la Lumière Ambiante :** Contrôle de l'éclairage ambiant global de la scène.
    *   **Options Basées sur l'IA (Potentiellement) :** Pourrait exploiter des modèles d'IA pour prédire des interactions lumineuses et des reflets réalistes pour des résultats plus naturels.
    *   **Aperçu :** Prévisualisation en temps réel ou rapide des changements d'éclairage.
    *   Gestion des ressources, potentiellement en déchargeant SDXL si un modèle de ré-éclairage dédié est utilisé ou si le traitement est lourd pour le CPU/GPU.

---



Cette vue d'ensemble devrait maintenant être beaucoup plus complète et refléter fidèlement l'étendue des capacités de **Cyberbill Image Generator** !
