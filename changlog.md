# Changelog

## 🐓 Beta 2.0.0 The fearless young rooster 🐓 - 2025-06-13

### ✨ New Features and Improvements ✨

*   **ImageToText (`ImageToText_mod.py`)**:
    *   Improved log management: the log list and the maximum number of entries are now managed at the instance level for better encapsulation.
    *   Added a "Unload Model" button to free up Florence-2 model resources.
    *   Generation of a detailed JSON report at the end of processing, including the status of each image, the method used, processing time, etc.
    *   Direct use of `FLORENCE2_TASKS` and module translations for task mapping, enhancing robustness.

*   **LoRA Training (`LoRATraining_mod.py`)**:
    *   **Major UI and Logic Overhaul**:
        *   Clear separation of data preparation and training steps with dedicated buttons ("Prepare Data", "Start Training").
        *   Internal logic divided into `_actual_preparation_logic` and `_actual_training_logic` for better organization.
        *   Use of `queue.Queue` for non-blocking log communication from background threads to the user interface.
    *   **Enhanced Data Preparation**:
        *   Optional automatic *captioning* toggle via a checkbox.
        *   If automatic *captioning* is disabled, existing `.txt` files in the source folder are copied.
        *   Sequential renaming of images and associated `.txt` files in the prepared data folder (e.g., `concept_0001.png`, `concept_0001.txt`).
    *   **Dataset Management (`DreamBoothDataset`)**:
        *   Now stores and returns the original image sizes (`original_size_hw`) and crop coordinates (`crop_coords_top_left_yx`).
    *   **Training Logic (SDXL)**:
        *   Calculation and passing of `add_time_ids` (including original size and crop coordinates) to the UNet model, as required by SDXL.
        *   VAE encoding is performed outside the `autocast` context if the VAE is in fp32 and training is in fp16/bf16.
        *   Added *gradient clipping* during training for better convergence stability.
        *   Final saving of LoRA as a single `.safetensors` file containing UNet weights and optionally text encoders.
        *   Use of `unet.add_adapter()` and `text_encoder.add_adapter()` for modern PEFT configuration.
        *   Use of `cast_training_params()` to convert LoRA parameters to fp32 when training in fp16.
    *   **User Interface (UI)**:
        *   Learning rate selection is now a dropdown menu with descriptions for each value.
        *   Base model selection is a dropdown menu.
        *   Optimizer, learning rate scheduler, and mixed precision options are dropdown menus.
        *   Advanced network and optimizer settings are grouped into accordion sections.

*   **Memory Management (`Utils/gest_mem.py`)**:
    *   New utility module for monitoring system resource usage (RAM, CPU, VRAM, GPU Usage).
    *   Uses `psutil` for RAM and CPU statistics.
    *   Uses `pynvml` (if available) for detailed VRAM statistics and GPU usage for NVIDIA cards.
    *   Falls back to `torch.cuda` for basic VRAM information if `pynvml` is unavailable.
    *   Displays statistics via circular progress bars in the UI.
    *   Provides a "Memory Management" accordion in the UI with:
        *   Optional live display of statistics.
        *   A "Unload All Models" button interacting with `ModelManager`.
        *   Explicit memory cleanup (`gc.collect()`, `torch.cuda.empty_cache()`) after unloading.

*   **CogView3-Plus (`CogView3Plus_mod.py`)**:
    *   Improved logic for determining allowed resolutions, falling back to `FORMATS` if `COGVIEW3PLUS_ALLOWED_RESOLUTIONS` is undefined.
    *   Model loading now indicates that configurations (offload, slicing, tiling) are handled by `ModelManager`.
    *   Uses `execute_pipeline_task_async` for image generation, enabling a more responsive UI.
    *   Explicit memory cleanup (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) after generating each batch.
    *   Saved image metadata includes `Module: "CogView3-Plus"` and `Model: THUDM/CogView3-Plus-3B`.

*   **CogView4 (`CogView4_mod.py`)**:
    *   Similar resolution determination logic as CogView3Plus, using `COGVIEW4_ALLOWED_RESOLUTIONS`.
    *   Model loading applies specific configurations (CPU offload, VAE slicing/tiling) *after* loading the `CogView4Pipeline`.
    *   Uses `execute_pipeline_task_async` for generation.
    *   Saved image metadata includes `Module: "CogView4"` and `Model: THUDM/CogView4-6B`.

### 🐛 Bug Fixes 🐛

*   **LoRA Training (`LoRATraining_mod.py`)**:
    *   Fixed a potential bug where `is_preparing` was not properly used, replaced with `self.is_preparing`.
    *   Ensures `is_preparing` is set to `False` when training starts.

### ⚙️ Technical and Refactoring ⚙️

*   **General**:
    *   Modules now use `self.module_translations`, initialized with merged translations (global + module-specific) during module initialization via `GestionModule`.

*   **Model Manager (`Utils/model_manager.py`)**:
    *   The `load_model` method now handles `sana_sprint`, `cogview4`, `cogview3plus` model types and applies specific configurations (dtype, offload) for these models.
    *   Improved `unload_model` method for explicit pipeline component removal.
    *   The `apply_loras` method has been revised to use `unload_lora_weights` and `set_adapters` more robustly.

*   **LLM Prompter (`Utils/llm_prompter_util.py`)**:
    *   Uses `AutoModelForCausalLM` and `AutoTokenizer` for broader Hugging Face model compatibility.
    *   Model is loaded on CPU (`device_map="cpu"`) to avoid VRAM conflicts.
    *   Tokenizer `pad_token` is set to `eos_token` if absent, necessary for models like Qwen.
    *   Improved parsing of LLM output to extract the prompt, handling `<think>` tags and common preambles.


## 🐓 béta 2.0.0 The fearless young rooster 🐓 - 2025-06-13

### ✨ Nouvelles Fonctionnalités et Améliorations ✨

*   **ImageToText (`ImageToText_mod.py`)**:
    *   Amélioration de la gestion des logs : la liste des logs et le nombre maximum d'entrées sont désormais gérés au niveau de l'instance pour une meilleure encapsulation.
    *   Ajout d'un bouton "Décharger le modèle" permettant de libérer les ressources du modèle Florence-2.
    *   Génération d'un rapport JSON détaillé à la fin du traitement, incluant le statut de chaque image, la méthode utilisée, le temps de traitement, etc.
    *   Utilisation directe de `FLORENCE2_TASKS` et des traductions du module pour le mappage des tâches, améliorant la robustesse.

*   **Entraînement LoRA (`LoRATraining_mod.py`)**:
    *   **Refonte Majeure de l'Interface et de la Logique**:
        *   Séparation claire des étapes de préparation des données et d'entraînement avec des boutons dédiés ("Préparer les Données", "Démarrer l'Entraînement").
        *   La logique interne a été divisée en `_actual_preparation_logic` et `_actual_training_logic` pour une meilleure organisation.
        *   Utilisation de `queue.Queue` pour une communication non bloquante des logs depuis les threads d'arrière-plan vers l'interface utilisateur.
    *   **Préparation des Données Améliorée**:
        *   Option de *captioning* automatique désactivable via une case à cocher.
        *   Si le *captioning* automatique est désactivé, les fichiers `.txt` existants dans le dossier source sont copiés.
        *   Renommage séquentiel des images et des fichiers `.txt` associés dans le dossier de données préparé (ex: `concept_0001.png`, `concept_0001.txt`).
    *   **Gestion du Dataset (`DreamBoothDataset`)**:
        *   Stocke et retourne maintenant les tailles originales des images (`original_size_hw`) et les coordonnées de recadrage (`crop_coords_top_left_yx`).
    *   **Logique d'Entraînement (SDXL)**:
        *   Calcul et passage des `add_time_ids` (incluant taille originale et coordonnées de recadrage) au modèle UNet, conformément aux exigences de SDXL.
        *   L'encodage VAE est effectué en dehors du contexte `autocast` si le VAE est en fp32 et l'entraînement en fp16/bf16.
        *   Ajout du *gradient clipping* pendant l'entraînement pour stabiliser la convergence.
        *   Sauvegarde finale du LoRA en un unique fichier `.safetensors` contenant les poids de l'UNet et optionnellement des encodeurs de texte.
        *   Utilisation de `unet.add_adapter()` et `text_encoder.add_adapter()` pour une configuration PEFT plus moderne.
        *   Utilisation de `cast_training_params()` pour convertir les paramètres LoRA en fp32 lors de l'entraînement en fp16.
    *   **Interface Utilisateur (UI)**:
        *   Le taux d'apprentissage (learning rate) est maintenant sélectionnable via un menu déroulant avec des descriptions pour chaque valeur.
        *   La sélection du modèle de base est un menu déroulant.
        *   Les options d'optimiseur, de planificateur de taux d'apprentissage et de précision mixte sont des menus déroulants.
        *   Les options avancées de réseau et d'optimiseur sont groupées dans des accordéons.

*   **Gestion de la Mémoire (`Utils/gest_mem.py`)**:
    *   Nouveau module utilitaire pour surveiller l'utilisation des ressources système (RAM, CPU, VRAM, Utilisation GPU).
    *   Utilise `psutil` pour les statistiques RAM et CPU.
    *   Utilise `pynvml` (si disponible) pour des statistiques VRAM détaillées et l'utilisation GPU pour les cartes NVIDIA.
    *   Fallback sur `torch.cuda` pour les informations VRAM de base si `pynvml` n'est pas disponible.
    *   Affiche les statistiques via des barres de progression circulaires SVG dans l'interface utilisateur.
    *   Fournit un accordéon "Gestion de la Mémoire" dans l'interface utilisateur avec :
        *   Affichage en direct (optionnel) des statistiques.
        *   Un bouton "Décharger Tous les Modèles" qui interagit avec `ModelManager`.
        *   Nettoyage explicite de la mémoire (`gc.collect()`, `torch.cuda.empty_cache()`) après le déchargement.

*   **CogView3-Plus (`CogView3Plus_mod.py`)**:
    *   Logique de détermination des résolutions autorisées améliorée, avec fallback sur la configuration `FORMATS` si `COGVIEW3PLUS_ALLOWED_RESOLUTIONS` n'est pas définie.
    *   Le chargement du modèle indique maintenant que les configurations (offload, slicing, tiling) sont gérées par le `ModelManager`.
    *   Utilisation de `execute_pipeline_task_async` pour la génération d'images, permettant une interface utilisateur plus réactive.
    *   Nettoyage explicite de la mémoire (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) après la génération de chaque image dans un lot.
    *   Les métadonnées des images sauvegardées incluent `Module: "CogView3-Plus"` et `Model: THUDM/CogView3-Plus-3B`.

*   **CogView4 (`CogView4_mod.py`)**:
    *   Logique de détermination des résolutions autorisées similaire à CogView3Plus, utilisant `COGVIEW4_ALLOWED_RESOLUTIONS`.
    *   Le chargement du modèle applique les configurations spécifiques (CPU offload, VAE slicing/tiling) *après* le chargement du pipeline `CogView4Pipeline`.
    *   Utilisation de `execute_pipeline_task_async` pour la génération.
    *   Les métadonnées des images sauvegardées incluent `Module: "CogView4"` et `Model: THUDM/CogView4-6B`.

### 🐛 Corrections de Bugs 🐛

*   **LoRATraining (`LoRATraining_mod.py`)**:
    *   Correction d'un bug potentiel où `is_preparing` n'était pas correctement utilisé, remplacé par `self.is_preparing`.
    *   Assure que `is_preparing` est mis à `False` lorsque l'entraînement démarre.

### ⚙️ Technique et Refactoring ⚙️

*   **Général**:
    *   Les modules utilisent maintenant `self.module_translations` qui est initialisé avec les traductions fusionnées (globales + spécifiques au module) lors de l'initialisation du module par `GestionModule`.
*   **ModelManager (`Utils/model_manager.py`)**:
    *   La méthode `load_model` gère maintenant les types de modèles `sana_sprint`, `cogview4`, `cogview3plus` et applique des configurations spécifiques (dtype, offload) pour ces modèles.
    *   La méthode `unload_model` a été améliorée pour une suppression plus explicite des composants du pipeline.
    *   La méthode `apply_loras` a été revue pour utiliser `unload_lora_weights` et `set_adapters` de manière plus robuste.
*   **LLM Prompter (`Utils/llm_prompter_util.py`)**:
    *   Utilise `AutoModelForCausalLM` et `AutoTokenizer` pour une compatibilité plus large avec les modèles Hugging Face.
    *   Le modèle est chargé sur CPU (`device_map="cpu"`) pour éviter les conflits de VRAM.
    *   Le `pad_token` du tokenizer est défini sur `eos_token` si non présent, ce qui est nécessaire pour certains modèles comme Qwen.
    *   Amélioration du parsing de la sortie du LLM pour extraire le prompt, en gérant les balises `<think>` et les préambules courants.

---


## Beta 1.9.0 🐔The Chicken Arrives🐔

*Date: 2025-05-29*

### ✨ New Features / Nouvelles Fonctionnalités

*   **New Module: Image ReLighting (`reLighting_mod.py`)**
    *   Introduced a new tab for advanced image relighting using IC-Light models. This module is based on the excellent work by lllyasviel/IC-Light.
    *   Supports two main modes:
        *   **FC (Foreground Conditioned):** Relights a subject based on the foreground image and a chosen light direction (e.g., left, right, top, bottom, or none).
        *   **FBC (Foreground-Background Conditioned):** Relights a subject considering both a foreground image and a background. The background can be uploaded, flipped, or generated as a directional light source or ambient grey.
    *   Integrates automatic background removal for the foreground subject using BriaRMBG.
    *   Offers comprehensive controls: prompt, negative prompt, seed, steps, CFG scale, high-resolution upscaling with denoising, and mode-specific parameters.
    *   Saves relighted images with detailed generation metadata.
*   **Nouveau Module : Re-Éclairage d'Image (`reLighting_mod.py`)**
    *   Introduction d'un nouvel onglet pour le re-éclairage avancé d'images utilisant les modèles IC-Light. Ce module est basé sur l'excellent travail de lllyasviel/IC-Light.
    *   Supporte deux modes principaux :
        *   **FC (Conditionné par l'Avant-plan) :** Ré-éclaire un sujet en se basant sur l'image d'avant-plan et une direction de lumière choisie (ex: gauche, droite, haut, bas, ou aucune).
        *   **FBC (Conditionné par l'Avant-plan et l'Arrière-plan) :** Ré-éclaire un sujet en considérant à la fois une image d'avant-plan et un arrière-plan. L'arrière-plan peut être téléversé, inversé, ou généré comme une source de lumière directionnelle ou un gris ambiant.
    *   Intègre la suppression automatique de l'arrière-plan pour le sujet d'avant-plan en utilisant BriaRMBG.
    *   Offre des contrôles complets : prompt, prompt négatif, seed, étapes, échelle CFG, mise à l'échelle haute résolution avec débruitage, et paramètres spécifiques au mode.
    *   Sauvegarde les images ré-éclairées avec des métadonnées de génération détaillées.

---

## Beta 1.8.9 🐣The Chick, Future Chicken🐔

### ✨ New Features / Nouvelles Fonctionnalités

*   **AI Prompt Enhancement (LLM):** Added an optional feature to automatically enrich user prompts using a local Language Model (default: `Qwen/Qwen3-0.6B`). The LLM generates more detailed and imaginative prompts in English, optimized for image generators. This feature is configurable via the `LLM_PROMPTER_MODEL_PATH` key in `config.json` and runs on the CPU to preserve GPU resources.
    *   **Amélioration des Prompts par IA (LLM) :** Ajout d'une fonctionnalité optionnelle pour enrichir automatiquement les prompts utilisateurs en utilisant un Modèle de Langage local (par défaut : `Qwen/Qwen3-0.6B`). Le LLM génère des prompts plus détaillés et imaginatifs en anglais, optimisés pour les générateurs d'images. Cette fonctionnalité est configurable via la clé `LLM_PROMPTER_MODEL_PATH` dans `config.json` et s'exécute sur le CPU pour préserver les ressources GPU.

### 🛠️ Fixes / Corrections

*   **Module Translation:** Fixed a major bug where the selected language in `config.json` (e.g., English) was not correctly passed during module initialization, leading to UI translation issues within modules. `GestionModule` now correctly receives and applies the global language and translations.
    *   **Traduction des Modules :** Correction d'un bug majeur où la langue sélectionnée dans `config.json` (par exemple, l'anglais) n'était pas correctement transmise lors de l'initialisation des modules, entraînant des problèmes de traduction de l'interface des modules. `GestionModule` reçoit et applique maintenant correctement la langue et les traductions globales.

---

## Beta 1.8.8 🐥Crazy Happy Chick🐥

*Date: 2025-05-14*

### 🔧 Changes

*   **UI/UX - LoRA Loading:** LoRA dropdown menus are now populated with available LoRAs upon application startup, improving initial usability by removing the need to manually refresh the list.
*   **Gradio Update:** The application has been updated to be compatible with Gradio `5.29.1`.


### 🛠️ Fixes

*   **Preset Loading - VAE:** Corrected an issue where the VAE specified in a loaded preset was not properly selected in the VAE dropdown menu on the image generation interface.

---

### 🔧 Changements

*   **UI/UX - Chargement des LoRAs :** Les menus déroulants des LoRAs sont désormais remplis avec les LoRAs disponibles dès le démarrage de l'application, améliorant l'utilisabilité initiale en supprimant le besoin de rafraîchir manuellement la liste.
*   **Mise à Jour Gradio :** L'application a été mise à jour pour être compatible avec Gradio `5.29.1`.


### 🛠️ Corrections

*   **Chargement des Presets - VAE :** Correction d'un problème où le VAE spécifié dans un preset chargé n'était pas correctement sélectionné dans le menu déroulant VAE de l'interface de génération d'images.

---

## Beta 1.8.7 🐥Crazy Happy Chick🐥

*Date: 2025-05-13*

### ✨ New Features

*   **New Module: Civitai Downloader (`civitai_downloader_mod.py`)**
    *   Added a dedicated tab to search and download models, LoRAs, VAEs, etc., directly from Civitai.
    *   Supports filtering by model type, sort order, period, and NSFW content.
    *   Includes an interface to view model details, select specific versions and files for download.
    *   Option to use a Civitai API key for extended access.
*   **New Module: Image Watermark (`ImageWatermark_mod.py`)**
    *   Added a new tab for applying text or image watermarks to your generated images.
    *   Supports single image processing and batch processing of images from a folder.
    *   Customizable options for watermark content (text/image), font, size, color, scale, opacity, position (including tiling), margin, and rotation.

### 🔧 Changes

*   **Gradio Update:** The application has been updated to be compatible with Gradio `5.29.0`. 

### 🛠️ Fixes

*   **HTML Report Generation:** Improved HTML report generation to ensure it's correctly created or updated even if the image generation process is stopped prematurely.
*   **General Bug Fixes:** Addressed various minor bugs and improved overall stability.

---

### ✨ Nouvelles Fonctionnalités (French)

*   **Nouveau Module : Téléchargeur Civitai (`civitai_downloader_mod.py`)**
    *   Ajout d'un onglet dédié pour rechercher et télécharger des modèles, LoRAs, VAEs, etc., directement depuis Civitai.
    *   Supporte le filtrage par type de modèle, ordre de tri, période et contenu NSFW.
    *   Inclut une interface pour voir les détails du modèle, sélectionner des versions spécifiques et des fichiers à télécharger.
    *   Option d'utiliser une clé API Civitai pour un accès étendu.
*   **Nouveau Module : Filigrane d'Image (`ImageWatermark_mod.py`)**
    *   Ajout d'un nouvel onglet pour appliquer des filigranes textuels ou graphiques sur vos images générées.
    *   Supporte le traitement d'image unique et le traitement par lot d'images depuis un dossier.
    *   Options personnalisables pour le contenu du filigrane (texte/image), police, taille, couleur, échelle, opacité, position (y compris en mosaïque), marge et rotation.

### 🔧 Changements (French)

*   **Mise à Jour Gradio :** L'application a été mise à jour pour être compatible avec Gradio `5.29.0`. 

### 🛠️ Corrections (French)

*   **Génération du Rapport HTML :** Amélioration de la génération du rapport HTML pour s'assurer qu'il est correctement créé ou mis à jour même si le processus de génération d'images est arrêté prématurément.
*   **Corrections de Bugs Générales :** Résolution de divers bugs mineurs et amélioration de la stabilité générale.

---

## Beta 1.8.6 🐥Crazy Happy Chick🐥

*Date: 2025-05-02*

### ✨ New Features

*   **New Module: Sana Sprint (`sana_sprint_mod.py`)**
    *   Added a dedicated tab for image generation using the `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` model.
    *   Supports text-to-image generation with style mixing.
    *   Includes image-to-prompt functionality: generate a prompt directly from an uploaded image within the Sana Sprint tab.
    *   Optimized for speed with fixed steps (2) and output size (1024x1024).
*   **Image-to-Prompt Refactoring:**
    *   Isolated the image-to-prompt generation logic (using `MiaoshouAI/Florence-2-base-PromptGen-v2.0`) into a reusable module: `core/image_prompter.py`.
    *   This functionality is now used by both the main generation tab and the Sana Sprint module.
    *   Model loading is handled centrally and initialized at application startup.

### 🛠️ Fixes

*   **Gradio Dropdown Warnings:** Resolved persistent `UserWarning: The value passed into gr.Dropdown() is not in the list of choices...` by:
    *   Adding `allow_custom_value=True` to relevant dropdown components (`model`, `VAE`, `sampler`, `format`, `LoRA`, `preset filters`, etc.) across the application.
    *   Improving the logic in `ModelManager.list_models` to filter out the placeholder `your_default_modele.safetensors` from the choices list.
*   **Module Stop Button:** Corrected `UserWarning` related to argument mismatch when calling the `stop_generation` method in modules like `image_to_image_mod.py` by passing necessary arguments (like translations) via `gr.State`.

---

### ✨ Nouvelles Fonctionnalités (French)

*   **Nouveau Module : Sana Sprint (`sana_sprint_mod.py`)**
    *   Ajout d'un onglet dédié pour la génération d'images avec le modèle `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`.
    *   Supporte la génération texte-vers-image avec mélange de styles.
    *   Inclut la fonctionnalité image-vers-prompt : générer un prompt directement depuis une image téléversée dans l'onglet Sana Sprint.
    *   Optimisé pour la vitesse avec des étapes fixes (2) et une taille de sortie fixe (1024x1024).
*   **Refactorisation Image-vers-Prompt :**
    *   Isolation de la logique de génération de prompt depuis une image (utilisant `MiaoshouAI/Florence-2-base-PromptGen-v2.0`) dans un module réutilisable : `core/image_prompter.py`.
    *   Cette fonctionnalité est maintenant utilisée par l'onglet de génération principal et le module Sana Sprint.
    *   Le chargement du modèle est géré de manière centralisée et initialisé au démarrage de l'application.

### 🛠️ Corrections (French)

*   **Avertissements Dropdown Gradio :** Résolution des avertissements persistants `UserWarning: The value passed into gr.Dropdown() is not in the list of choices...` en :
    *   Ajoutant `allow_custom_value=True` aux composants dropdown concernés (`modèle`, `VAE`, `sampler`, `format`, `LoRA`, `filtres presets`, etc.) dans toute l'application.
    *   Améliorant la logique dans `ModelManager.list_models` pour filtrer la valeur placeholder `your_default_modele.safetensors` de la liste des choix.
*   **Bouton Stop des Modules :** Correction du `UserWarning` lié à une incohérence d'arguments lors de l'appel de la méthode `stop_generation` dans les modules (ex: `image_to_image_mod.py`) en passant les arguments nécessaires (comme les traductions) via `gr.State`.

---

## Beta 1.8.5 🐥Crazy Happy Chick🐥

*Date: 2025-05-02*

### ✨ New Features

*   **New Module: Image Enhancement (`ImageEnhancement_mod.py`)**
    *   Added a dedicated tab replacing previous Upscaling (SDXL) and Enhancement (AuraSR) functionalities.
    *   **Colorization:** Integrated ModelScope's `damo/cv_ddcolor_image-colorization` model to colorize black and white images.
    *   **Upscale (4x):** Integrated Diffusers' `CompVis/ldm-super-resolution-4x-openimages` model for 4x image upscaling.
    *   **Restoration:** Integrated the OneRestore model (`onerestore_real.tar` + `embedder_model.tar`) for automatic image degradation detection and restoration (e.g., fixing blur, noise). Models are included in `modules/ImageEnhancement_models/`.
    *   **Auto Retouch:** Added a simple automatic retouching option using PIL enhancements (Contrast, Sharpness, Saturation).
*   **Model Management:** Enhancement models are loaded on demand and unloaded after use to save VRAM, including automatic unloading of the main generation model if loaded.
*   **Helper Functions:** Created `ImageEnhancement_helper.py` for loading OneRestore checkpoints.

### 🔧 Changes

*   **Auto Retouch Enhancement:** Added saturation adjustment to the existing "Auto Retouch" feature within the Image Enhancement module.
*   **Dependencies:** Updated and pinned requirements in `requirements.txt` for better stability and reproducibility. Switched `diffusers` from a Git commit to the stable PyPI version (`0.33.1`) to support future features (like SANA). Corrected `opencv-python-headless` version. Removed unnecessary `futures` package.
*   **Module Cleanup:** Removed obsolete AuraSR enhancement and SDXL Upscaling modules/features, now superseded by the new Image Enhancement module.
*   **Code Quality:** Minor internal code adjustments and cleanup in various modules.

### ➕ Added Features

*   **XMP Metadata:** Images are now saved with XMP metadata, enriching PNG, JPEG, and WEBP files with comprehensive information about their generation. For example:
    *   **PNG:** Metadata is stored using `pnginfo`.
    *   **JPEG:** Metadata is stored in `exif.UserComment`.
    *   **WEBP:** Metadata is stored in `xmp` format.
* **Image to Image Batch Mode:** Image to Image module now allows you to select a folder containing multiple images for processing in batch mode.

---

### ✨ Nouvelles Fonctionnalités (French)

*   **Nouveau Module : Amélioration d'Image (`ImageEnhancement_mod.py`)**
    *   Ajout d'un onglet dédié remplaçant les fonctionnalités précédentes d'Upscaling (SDXL) et d'Amélioration (AuraSR).
    *   **Colorisation :** Intégration du modèle ModelScope `damo/cv_ddcolor_image-colorization` pour coloriser les images en noir et blanc.
    *   **Upscale (4x) :** Intégration du modèle Diffusers `CompVis/ldm-super-resolution-4x-openimages` pour l'agrandissement d'image 4x.
    *   **Restauration :** Intégration du modèle OneRestore (`onerestore_real.tar` + `embedder_model.tar`) pour la détection automatique de la dégradation et la restauration d'image (ex: correction du flou, bruit). Modèles inclus dans `modules/ImageEnhancement_models/`.
    *   **Retouche Auto :** Ajout d'une option de retouche automatique simple utilisant les améliorations PIL (Contraste, Netteté, Saturation).
*   **Gestion des Modèles :** Les modèles d'amélioration sont chargés à la demande et déchargés après utilisation pour économiser la VRAM, incluant le déchargement automatique du modèle de génération principal s'il est chargé.
*   **Fonctions Utilitaires :** Création de `ImageEnhancement_helper.py` pour le chargement des checkpoints OneRestore.

### 🔧 Changements (French)

*   **Amélioration Retouche Auto :** Ajout de l'ajustement de la saturation à la fonctionnalité "Retouche Auto" existante dans le module d'Amélioration d'Image.
*   **Dépendances :** Mise à jour et épinglage des dépendances dans `requirements.txt` pour une meilleure stabilité et reproductibilité. Remplacement de `diffusers` d'un commit Git vers la version stable PyPI (`0.33.1`) pour supporter les fonctionnalités futures (comme SANA). Correction de la version de `opencv-python-headless`. Suppression du paquet `futures` inutile.
*   **Nettoyage Modules :** Suppression des modules/fonctionnalités obsolètes d'amélioration AuraSR et d'Upscaling SDXL, désormais remplacés par le nouveau module d'Amélioration d'Image.
*   **Qualité du Code :** Ajustements mineurs du code interne et nettoyage dans divers modules.

### ➕ Fonctionnalités Ajoutées

*   **Métadonnées XMP :** Les images sont désormais sauvegardées avec des métadonnées XMP, enrichissant les fichiers PNG, JPEG et WEBP avec des informations complètes sur leur génération. Par exemple :
    *   **PNG :** Les métadonnées sont stockées en utilisant `pnginfo`.
    *   **JPEG :** Les métadonnées sont stockées dans `exif.UserComment`.
    *   **WEBP :** Les métadonnées sont stockées au format `xmp`.
*   **Mode Batch Image to Image :** Le module Image to Image vous permet maintenant de sélectionner un dossier contenant plusieurs images pour un traitement en mode batch.
---


## Beta 1.8 🚀

*Date: 2025-04-29*

### 🌟 Summary of Version [Beta 1.8]

#### ✨ New Features

*   **Batch Generator Tab:**
    *   Added a new dedicated tab to easily create lists of image generation tasks (batches).
    *   Configure model, VAE, prompts, negative prompts, styles, sampler, steps, guidance, seed, dimensions, LoRAs (up to 4), and optional output filename for each task.
    *   Includes an option to automatically translate the positive prompt to English before adding it to the task.
    *   Displays the list of tasks in a table for review.
    *   Generates a JSON file containing the batch definition.
    *   **Automatic Saving:** The generated JSON is automatically saved to a predefined directory (`Output\json_batch_files` by default, configurable in `config.json` via `SAVE_BATCH_JSON_PATH`) with an incremental filename (e.g., `batch_001.json`, `batch_002.json`).
*   **Batch Runner Integration:**
    *   Added functionality within the main generation tab (under an accordion) to load and execute batch tasks defined in a JSON file.
    *   Loads the JSON file (ideally from the configured save path, e.g., using `gr.FileExplorer` if applicable).
    *   Processes tasks sequentially, automatically handling model/VAE/sampler loading and unloading to optimize performance.
    *   Applies LoRAs specified for each task.
    *   Displays overall batch progress, individual task progress, and generated images in a gallery.
    *   Includes a button to stop the entire batch process.
    *   Saves generated images and creates the HTML report similar to single image generation.

#### 🔧 Changes

*   **Configuration:** Added `SAVE_BATCH_JSON_PATH` to `config.json` to define the default save location for batch JSON files.
*   **UI:** Integrated batch generation creation into its own tab and batch execution controls into the main generation tab.
*   **Core Logic:** Implemented `batch_runner.py` to handle the execution logic for batch files, including model management and task processing. Standardized module JSON structure for translations (`language` key).
*   **Pipeline Execution Refactoring:** Refactored the core pipeline execution logic into `pipeline_executor.py` for better separation of concerns and asynchronous handling, improving UI responsiveness during generation.
*   **Dependency Update:** Tested and confirmed compatibility with **PyTorch 2.7** and **CUDA 12.8**, potentially offering performance improvements. The installation script (`install.bat`) has been updated accordingly.

#### 🛠️ Fixes

*   **Preset Loading:** Corrected an issue where loading a preset with the default VAE ("Défaut VAE") would incorrectly display a "VAE not found" warning.
*   **Translation Loading:** Fixed issues related to loading translations for module-specific UI elements by standardizing the JSON structure (`language` key) and correcting translation function calls (`.format()` usage).
*   **Callback Errors:** Resolved `'NoneType' object has no attribute 'append'` errors in the progress callback when running in batch mode.
*   **JSON Parsing:** Corrected `TypeError` when handling `styles` and `loras` data already parsed from JSON in the batch runner.

### 🌟 Résumé de la Version [Beta 1.8]

#### ✨ Nouvelles Fonctionnalités

*   **Onglet Générateur de Batch:**
    *   Ajout d'un nouvel onglet dédié pour créer facilement des listes de tâches de génération d'images (batches).
    *   Permet de configurer le modèle, VAE, prompts, prompt négatif, styles, sampler, étapes, guidage, seed, dimensions, LoRAs (jusqu'à 4), et un nom de fichier de sortie optionnel pour chaque tâche.
    *   Inclut une option pour traduire automatiquement le prompt positif en anglais avant de l'ajouter à la tâche.
    *   Affiche la liste des tâches dans un tableau pour vérification.
    *   Génère un fichier JSON contenant la définition du batch.
    *   **Sauvegarde Automatique:** Le JSON généré est automatiquement sauvegardé dans un répertoire prédéfini (`Output\json_batch_files` par défaut, configurable dans `config.json` via `SAVE_BATCH_JSON_PATH`) avec un nom de fichier incrémentiel (ex: `batch_001.json`, `batch_002.json`).
*   **Intégration de l'Exécuteur de Batch:**
    *   Ajout d'une fonctionnalité dans l'onglet de génération principal (sous un accordéon) pour charger et exécuter les tâches de batch définies dans un fichier JSON.
    *   Charge le fichier JSON (idéalement depuis le chemin de sauvegarde configuré).
    *   Traite les tâches séquentiellement, gérant automatiquement le chargement/déchargement des modèles/VAE/samplers pour optimiser les performances.
    *   Applique les LoRAs spécifiés pour chaque tâche.
    *   Affiche la progression globale du batch, la progression de la tâche individuelle, et les images générées dans une galerie.
    *   Inclut un bouton pour arrêter l'ensemble du processus de batch.
    *   Sauvegarde les images générées et crée le rapport HTML de manière similaire à la génération d'image unique.

#### 🔧 Changements

*   **Configuration:** Ajout de `SAVE_BATCH_JSON_PATH` dans `config.json` pour définir l'emplacement de sauvegarde par défaut des fichiers JSON de batch.
*   **UI:** Intégration de la création de batch dans son propre onglet et des contrôles d'exécution de batch dans l'onglet de génération principal.
*   **Logique Principale:** Implémentation de `batch_runner.py` pour gérer la logique d'exécution des fichiers batch, y compris la gestion des modèles et le traitement des tâches. Standardisation de la structure JSON des modules pour les traductions (clé `language`).
*   **Refactorisation Exécution Pipeline:** Refactorisation de la logique principale d'exécution du pipeline dans `pipeline_executor.py` pour une meilleure séparation des responsabilités et une gestion asynchrone, améliorant la réactivité de l'interface pendant la génération.
*   **Mise à Jour Dépendances:** Testé et confirmé compatible avec **PyTorch 2.7** et **CUDA 12.8**, offrant potentiellement des améliorations de performance. Le script d'installation (`install.bat`) a été mis à jour en conséquence.

#### 🛠️ Corrections

*   **Chargement des Presets:** Correction d'un problème où le chargement d'un preset avec le VAE par défaut ("Défaut VAE") affichait incorrectement un avertissement "VAE introuvable".
*   **Chargement des Traductions:** Correction de problèmes liés au chargement des traductions pour les éléments d'interface spécifiques aux modules en standardisant la structure JSON (clé `language`) et en corrigeant les appels à la fonction de traduction (usage de `.format()`).
*   **Erreurs Callback:** Résolution des erreurs `'NoneType' object has no attribute 'append'` dans le callback de progression lors de l'exécution en mode batch.
*   **Analyse JSON:** Correction du `TypeError` lors de la manipulation des données `styles` et `loras` déjà analysées depuis le JSON dans l'exécuteur de batch.
