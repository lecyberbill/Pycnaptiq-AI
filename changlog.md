# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
    *   Loads the JSON file (ideally from the configured save path).
    *   Processes tasks sequentially, automatically handling model/VAE/sampler loading and unloading to optimize performance.
    *   Applies LoRAs specified for each task.
    *   Displays overall batch progress, individual task progress, and generated images in a gallery.
    *   Includes a button to stop the entire batch process.
    *   Saves generated images and creates the HTML report similar to single image generation.

#### 🔧 Changes

*   **Configuration:** Added `SAVE_BATCH_JSON_PATH` to `config.json` to define the default save location for batch JSON files.
*   **UI:** Integrated batch generation creation into its own tab and batch execution controls into the main generation tab.
*   **Core Logic:** Implemented `batch_runner.py` to handle the execution logic for batch files, including model management and task processing.

#### 🛠️ Fixes

*   **Preset Loading:** Corrected an issue where loading a preset with the default VAE ("Défaut VAE") would incorrectly display a "VAE not found" warning.
*   **Translation Loading:** Fixed issues related to loading translations for module-specific UI elements by standardizing the JSON structure (`language` key).
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
*   **Logique Principale:** Implémentation de `batch_runner.py` pour gérer la logique d'exécution des fichiers batch, y compris la gestion des modèles et le traitement des tâches.

#### 🛠️ Corrections

*   **Chargement des Presets:** Correction d'un problème où le chargement d'un preset avec le VAE par défaut ("Défaut VAE") affichait incorrectement un avertissement "VAE introuvable".
*   **Chargement des Traductions:** Correction de problèmes liés au chargement des traductions pour les éléments d'interface spécifiques aux modules en standardisant la structure JSON (clé `language`).
*   **Erreurs Callback:** Résolution des erreurs `'NoneType' object has no attribute 'append'` dans le callback de progression lors de l'exécution en mode batch.
*   **Analyse JSON:** Correction du `TypeError` lors de la manipulation des données `styles` et `loras` déjà analysées depuis le JSON dans l'exécuteur de batch.

---

## Beta 1.7.3 🚀

*Date: 2025-04-25*

### 🌟 Summary of Version [Beta 1.7.3]

#### ✨ Added

*   **XMP/EXIF Metadata Integration:** Generation parameters (model, VAE, prompts, steps, guidance, styles, LoRAs, etc.) are now embedded directly into the metadata of generated images (PNG: `parameters` chunk, JPEG: `EXIF UserComment`, WebP: `XMP dc:description/exif:UserComment`).
*   **Image-to-Image Batch Mode:** Added a batch processing mode to the Image-to-Image module, allowing processing of all images within a selected folder. Includes a preview of the currently processed image.

#### 🛠️ Fixed

*   **Startup Model Loading:** Corrected an issue where the default model specified in `config.json` might fail to load correctly at application startup under certain conditions.
*   **Metadata Encoding:** Fixed encoding issues for JPEG EXIF `UserComment` (using `unicode`) and ensured proper XML escaping for WebP XMP data.
*   **JPEG Save Format:** Corrected `KeyError: 'JPG'` by ensuring Pillow uses the 'JPEG' identifier.
*   **Gradio Warning:** Resolved `Unexpected argument. Filling with None.` warning related to the sampler dropdown change event.

### 🌟 Résumé de la Version [Beta 1.7.3]

#### ✨ Ajouts

*   **Intégration des Métadonnées XMP/EXIF:** Les paramètres de génération (modèle, VAE, prompts, étapes, guidage, styles, LoRAs, etc.) sont maintenant intégrés directement dans les métadonnées des images générées (PNG: chunk `parameters`, JPEG: `EXIF UserComment`, WebP: `XMP dc:description/exif:UserComment`).
*   **Mode Batch pour Image-to-Image:** Ajout d'un mode de traitement par lot au module Image-to-Image, permettant de traiter toutes les images d'un dossier sélectionné. Inclut un aperçu de l'image en cours de traitement.

#### 🛠️ Corrections

*   **Chargement du Modèle au Démarrage:** Correction d'un problème où le modèle par défaut spécifié dans `config.json` pouvait échouer à se charger correctement au démarrage de l'application dans certaines conditions.
*   **Encodage des Métadonnées:** Correction des problèmes d'encodage pour le champ `UserComment` EXIF des JPEG (utilisation de `unicode`) et assurance d'un échappement XML correct pour les données XMP des WebP.
*   **Format Sauvegarde JPEG:** Correction de l'erreur `KeyError: 'JPG'` en s'assurant que Pillow utilise l'identifiant 'JPEG'.
*   **Avertissement Gradio:** Résolution de l'avertissement `Unexpected argument. Filling with None.` lié à l'événement de changement du dropdown des samplers.

---

## Beta 1.7 🐥Chick🐥 Latest

*Date: 2025-04-24*

### 🌟 Summary of Version [Beta 1.7]

#### ✨ New Features

*   **Preset Management System:**
    *   **Save Your Creations:** Easily save all parameters of a successful image generation (model, VAE, prompts, negative prompt, styles, guidance, steps, sampler, seed, dimensions, active LoRAs with weights) along with a preview image, a custom name, rating, and notes.
    *   **Dedicated Preset Tab:** Browse, search, sort, and filter your saved presets in a new dedicated tab.
    *   **Gallery View:** Presets are displayed in a gallery with their preview image, name, rating, and technical details (model, sampler).
    *   **Advanced Filtering & Sorting:** Find presets quickly by searching names, filtering by model, sampler, or used LoRAs, and sorting by creation date, name, last used date, or rating.
    *   **Load Presets:** Instantly load all settings from a saved preset back into the main generation interface with a single click.
    *   **Manage:** Delete unwanted ones, and assign ratings (1-5 stars).
    *   **SQLite Backend:** Uses a local SQLite database (`presets.db`) for efficient storage and retrieval, managed by `presets_Manager.py`.

#### 🔧 Changes

*   **New "Preset" Tab:** Added a dedicated tab for managing saved generation settings.
*   **"Save Preset" Section:** Integrated an accordion section in the main generation tab to name, add notes, and save the current generation settings as a preset.
*   **Integration of `presets_Manager.py`:** Core logic for preset database operations is now handled by this dedicated module.
*   **UI State Management:** Enhanced state handling in `cyberbill_SDXL.py` to manage preset data, preview images, and UI updates related to preset actions.
*   **Generation Workflow Update:** The `generate_image` function now prepares and outputs the necessary data for saving presets.

#### 🛠️ Fixes

*   **Improved Data Consistency:** Refined how generation parameters are captured and stored for presets.
*   **UI Responsiveness:** Addressed potential UI update issues after performing preset actions like deleting or renaming.
*   **Error Handling:** Improved error handling during the saving and loading of presets to prevent issues with invalid data or database access.
*   **Filter Logic:** Corrected and optimized the logic for filtering presets by sampler and LoRAs.

### 🌟 Résumé de la Version [Beta 1.7]

#### ✨ Nouvelles Fonctionnalités

*   **Système de Gestion des Presets:**
    *   **Sauvegardez Vos Créations:** Enregistrez facilement tous les paramètres d'une génération d'image réussie (modèle, VAE, prompts, prompt négatif, styles, guidage, étapes, sampler, seed, dimensions, LoRAs actifs avec leurs poids) ainsi qu'une image d'aperçu, un nom personnalisé, une note et des commentaires.
    *   **Onglet Preset Dédié:** Parcourez, recherchez, triez et filtrez vos presets sauvegardés dans un nouvel onglet dédié.
    *   **Vue Galerie:** Les presets sont affichés dans une galerie avec leur image d'aperçu, nom, note et détails techniques (modèle, sampler).
    *   **Filtrage & Tri Avancés:** Retrouvez rapidement des presets en cherchant par nom, en filtrant par modèle, sampler, ou LoRAs utilisés, et en triant par date de création, nom, date de dernière utilisation ou note.
    *   **Charger les Presets:** Rechargez instantanément tous les réglages d'un preset sauvegardé dans l'interface de génération principale en un seul clic.
    *   **Gérer:** Supprimez ceux qui ne sont plus utiles et attribuez des notes (1-5 étoiles).
    *   **Backend SQLite:** Utilise une base de données locale SQLite (`presets.db`) pour un stockage et une récupération efficaces, gérée par `presets_Manager.py`.

#### 🔧 Changements

*   **Nouvel Onglet "Preset":** Ajout d'un onglet dédié pour la gestion des paramètres de génération sauvegardés.
*   **Section "Sauvegarder Preset":** Intégration d'une section (accordéon) dans l'onglet de génération principal pour nommer, ajouter des notes et sauvegarder les paramètres de la génération actuelle en tant que preset.
*   **Intégration de `presets_Manager.py`:** La logique principale pour les opérations sur la base de données des presets est désormais gérée par ce module dédié.
*   **Gestion de l'État de l'UI:** Amélioration de la gestion de l'état dans `cyberbill_SDXL.py` pour gérer les données des presets, les images d'aperçu et les mises à jour de l'interface liées aux actions sur les presets.
*   **Mise à Jour du Flux de Génération:** La fonction `generate_image` prépare et retourne maintenant les données nécessaires à la sauvegarde des presets.

#### 🛠️ Corrections

*   **Cohérence des Données Améliorée:** Affinement de la manière dont les paramètres de génération sont capturés et stockés pour les presets.
*   **Réactivité de l'UI:** Correction de problèmes potentiels de mise à jour de l'interface après des actions sur les presets comme la suppression ou le renommage.
*   **Gestion des Erreurs:** Amélioration de la gestion des erreurs lors de la sauvegarde et du chargement des presets pour prévenir les problèmes liés à des données invalides ou à l'accès à la base de données.
*   **Logique de Filtrage:** Correction et optimisation de la logique de filtrage des presets par sampler et LoRAs.

**Full Changelog**: BETA1.6.5...BETA1.7

---

## Beta 1.6.5

*Date: 10 Apr*
*Commit: 1703cc0*

### 🌟 Summary of Version [Beta 1.6.5]

#### ✨ New Features

*   **Addition of an Image-to-Image (i2i) module:**
    *   Transforms an input image into a new creation while retaining certain elements of the original.
    *   Ideal for enhancing, stylizing, or altering existing visuals while maintaining consistency.
*   **Expanded style catalog:**
    *   Includes new styles to broaden the possibilities for creating personalized images.

#### 🔧 Changes

*   **Redistribution of common functions:**
    *   Functions have been reorganized into separate files to simplify code maintenance.
*   **General interface reorganization:**
    *   Improved coherence and navigation for a more intuitive user experience.
*   **Removal of SDXL upscaler:**
    *   Removed due to not meeting quality expectations.
*   **Use of Gradio pop-ups:**
    *   Information messages are now managed via pop-up windows for better user interaction.
*   **Update to Gradio 5.23.3.**
*   **Bundling redistributable Python:**
    *   Removes the need for additional downloads.

#### 🛠️ Fixes

*   **Improved error handling:**
    *   Fixed crashes caused by poorly managed errors.
*   **Addition of missing translations:**
    *   Enhanced localization for better accessibility.
*   **Bug fixes:**
    *   Resolved various minor issues for a smoother experience.

### 🌟 Résumé de la Version [Beta 1.6.5]

#### ✨ Nouvelles Fonctionnalités

*   **Ajout d'un module Image to Image (i2i) :**
    *   Permet la transformation d'une image d'entrée en une nouvelle image tout en conservant certains aspects de la première.
    *   Idéal pour améliorer, styliser ou altérer des visuels existants tout en garantissant une cohérence avec l'original.
*   **Enrichissement du catalogue de styles :**
    *   Ajout de nouveaux styles pour étendre les possibilités de création d'images personnalisées.

#### 🔧 Changements

*   **Redistribution des fonctions communes :**
    *   Simplifie la maintenance du code en répartissant les fonctionnalités dans plusieurs fichiers.
*   **Réorganisation générale de l'interface :**
    *   Amélioration de la cohérence et de la navigation.
*   **Suppression de l'upscaler SDXL :**
    *   Retiré pour ne pas répondre aux attentes de qualité.
*   **Utilisation des pop-ups Gradio :**
    *   Messages d'information désormais gérés via des fenêtres pop-up pour une meilleure interaction utilisateur.
*   **Mise à jour vers Gradio 5.23.3.**
*   **Python redistribuable inclus :**
    *   Évite un téléchargement supplémentaire nécessaire.

#### 🛠️ Corrections

*   **Gestion améliorée des erreurs :**
    *   Correction de crashs liés à certaines erreurs mal gérées.
*   **Ajout de traductions manquantes :**
    *   Renforcement de la localisation pour une meilleure accessibilité.
*   **Correction de bugs divers :**
    *   Résolution de plusieurs problèmes mineurs pour une expérience plus fluide.

---

## Beta 1.6

*Date: 01 Apr*
*Commit: 25f21aa*

### 🌟 Summary of Version B.1.6

#### ✨ New Features

*   **Major code refactoring:** The application has been restructured for better scalability and maintainability, with functionalities split into separate modules.
*   **Core module:** Contains the heart of the application, integrating the image generator and inpainting capabilities (still powered by SDXL 😉).
*   **Modular design:** Offers the possibility of creating custom modules. Modules provided with the application include:
    *   Upscaler module: Based on SDXL for enhancing image resolution.
    *   Image enhancement module: Powered by AuraSR (source) for improved image quality.
    *   Background removal module: Utilizing remBG (source) to eliminate image backgrounds.
    *   CivitAI prompt explorer module: Browse the fantastic library of prompts and images from CIVITAI, and easily copy prompts to generate stunning images with your chosen models.
    *   Custom module template: A module to assist users in creating their own modules. This can be disabled by renaming it.
*   **Style management for image generation:** Personalize styles using a JSON file (`config/styles.json`).

#### 🔧 Changes

*   Improved application structure to support future expansions and custom functionalities.

#### 🛠️ Fixes

*   Resolved numerous bugs, though some new ones might have been introduced due to significant changes and revamps.

**Full Changelog**: BETA1.5.1...BETA1.6

---

## Beta 1.5.1

*Date: 16 Mar*
*Commit: 99ef837*

### 🌟 Summary of Version B.1.5.1

#### 🔧 Changes

*   **Revamped the user interface** for improved clarity and user experience.
    *   Image generation and image editing are now accessible through two separate tabs.
    *   It is now possible to copy a generated image to the clipboard using a right-click, with instructions for different browsers:
        *   **Windows and Mac:**
            *   Firefox, Chrome, Edge, and Safari: right-click on the generated image > select "Copy image".
    *   Copied images can be directly pasted into the editing module.

**Full Changelog**: BETA1.5...BETA1.5.1

---

## Beta 1.5

*Date: 14 Mar*
*Commit: 3266f5b*

### 🌟 Summary of Version B.1.5

#### ✨ New Features

*   **Preview of latent images** during inference.
*   **New photo editing filters:**
    *   Vibrance: Selectively enhances color intensity.
    *   Curves: Provides precise control over brightness and contrast.
    *   Adaptive sharpness: Dynamically adjusts sharpness for better clarity.
    *   Noise: Adds artistic grain effects for texture.
    *   Color gradient: Applies smooth color transitions for stylized effects.
    *   Color shift: Alters the color palette for creative changes.
*   **Additional image formats:**
    *   Portrait: Optimized for vertical-oriented visuals.
    *   Landscape: Suitable for wide horizontal images.

#### 🔧 Changes

*   Interface reorganization to include the new features.
*   Added a module folder for future expansions.
*   Ongoing improvement to code documentation.

#### 🛠️ Fixes

*   Resolved multiple bugs across new and existing features.

---

## BETA 1.0

*Date: 11 Mar*
*Commit: 463bcac*

*   **Support for English and French:** the application can now be used in either language. Modify the config file and set `language` to `en` for English or `fr` for French.
*   **Language selection during installation:** it's now possible to change the language during the installation process.
*   **Predefined style management:** users can now select a predefined style.
*   **Addition of the `style.json` file** in the `config` directory for style management.
*   **Console messages now feature colorization** for important information.

---

## BETA 0.1 Pre-release

*Date: 21 Feb*
*Commit: 3df93d0*

🚀 **cyberbill_SDXL Bêta 0.1 : Disponible maintenant !**

Je suis heureux de vous présenter la version Bêta 0.1 de mon logiciel, cyberbill_SDXL. Ce projet est le fruit de ma passion pour la technologie et la création d'images.

🌟 **Fonctionnalités principales :**

*   Compatibilité avec CUDA 12.6 : Optimisé pour les cartes Nvidia RTX.
*   Installation simple : Quelques étapes suffisent pour commencer.
*   Chargement de modèles personnalisés : Utilisez vos propres modèles ou le modèle MegaChonkXL intégré pour des résultats satisfaisants.
*   Options de personnalisation avancées : Paramètres de guidage, choix des samplers, ajustement des VAE et des Loras pour des créations uniques.

📥 **Comment utiliser :**

*   Téléchargez le code source ou `cyberbill_SDXL.zip` et décompressez-le.
*   Installez CUDA 12.6, si ce n'est pas déjà fait.
*   Exécutez `install.bat` puis `start.bat`.
*   Commencez à créer et partagez vos œuvres avec la communauté.

🎨 **Partagez vos créations :**

Je serais ravi de voir ce que vous créez avec cyberbill_SDXL! N'hésitez pas à partager vos œuvres sur les forums ou réseaux sociaux, et à me faire part de vos retours pour améliorer le logiciel.
