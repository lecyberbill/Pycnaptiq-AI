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
