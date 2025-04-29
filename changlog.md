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
