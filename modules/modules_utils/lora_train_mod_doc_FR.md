# Documentation du Module LoRATraining_mod.py

## Table des Matières
1.  [Introduction](#introduction)
2.  [Fonctionnalités Principales](#fonctionnalités-principales)
3.  [Composants Clés](#composants-clés)
    *   [Constantes et Variables Globales](#constantes-et-variables-globales)
    *   [Classe `DreamBoothDataset`](#classe-dreamboothdataset)
    *   Fonction `collate_fn`
    *   Classe `LoRATrainingModule`
        *   Initialisation (`__init__`)
        *   Création de l'Interface Utilisateur (`create_tab`)
        *   Gestion du Thread d'Entraînement (`run_preparation_and_training_thread`)
        *   Logique Principale d'Entraînement (`_actual_preparation_and_training`)
        *   Encodage des Prompts (`_encode_prompt`)
        *   Calcul des `time_ids` (`_compute_time_ids`)
4.  Flux de Travail
5.  Dépendances
6.  Configuration
7.  Notes d'Utilisation et Bonnes Pratiques

---

## 1. Introduction

Le module `LoRATraining_mod.py` est conçu pour permettre l'entraînement de modèles LoRA (Low-Rank Adaptation) pour Stable Diffusion XL (SDXL) directement au sein de l'application CyberBill Image Generator. Il offre une interface utilisateur pour configurer les paramètres d'entraînement, préparer les données (y compris le légendage automatique des images) et lancer le processus d'entraînement. L'objectif est de simplifier la création de LoRAs personnalisés sans nécessiter une configuration manuelle complexe d'environnements d'entraînement.

---

## 2. Fonctionnalités Principales

*   **Préparation Automatisée des Données** :
    *   Recadrage des images sources à la taille cible.
    *   Légendage automatique des images en utilisant un modèle Florence-2 (configurable).
    *   Ajout d'un mot-déclencheur (trigger word) aux légendes.
*   **Entraînement LoRA pour SDXL** :
    *   Support pour l'entraînement de l'UNet et optionnellement des encodeurs de texte (CLIPTextModel et CLIPTextModelWithProjection).
    *   Utilisation de la bibliothèque PEFT (Parameter-Efficient Fine-Tuning) de Hugging Face pour l'adaptation LoRA.
    *   Configuration flexible des hyperparamètres d'entraînement (taux d'apprentissage, taille de batch, nombre d'époques, etc.).
    *   Choix de l'optimiseur (AdamW, AdamW8bit, Lion).
    *   Support de la précision mixte (fp16, bf16, fp32) pour optimiser l'utilisation de la mémoire et la vitesse.
*   **Interface Utilisateur Intégrée** :
    *   Un onglet dédié dans l'application Gradio pour gérer toutes les étapes.
    *   Affichage en temps réel des logs d'entraînement.
    *   Boutons pour démarrer et arrêter le processus d'entraînement.
*   **Sauvegarde des Modèles** :
    *   Sauvegarde des checkpoints LoRA à des intervalles définis.
    *   Sauvegarde du modèle LoRA final au format compatible Diffusers (`.safetensors`) dans le répertoire LORAS_DIR configuré.
*   **Gestion des Ressources** :
    *   Déchargement du modèle de base SDXL principal avant de charger les composants pour l'entraînement afin de libérer de la VRAM.
    *   Nettoyage de la mémoire GPU après l'entraînement.

---

## 3. Composants Clés

### Constantes et Variables Globales

*   `MODULE_NAME`: "LoRATraining" - Nom du module.
*   `SUPPORTED_IMAGE_EXTENSIONS`: Liste des extensions d'images supportées pour les données d'instance (ex: `['.png', '.jpg', '.jpeg', '.webp']`).

### Classe `DreamBoothDataset`

Cette classe hérite de `torch.utils.data.Dataset` et est responsable de charger et de prétraiter les images et leurs légendes pour l'entraînement.

*   **`__init__(self, instance_data_root, instance_prompt_fallback, tokenizer_one, tokenizer_two, target_size=1024, center_crop=False)`**
    *   `instance_data_root`: Chemin vers le dossier contenant les images d'instance et leurs fichiers `.txt` de légendes.
    *   `instance_prompt_fallback`: Légende à utiliser si un fichier `.txt` est manquant pour une image.
    *   `tokenizer_one`, `tokenizer_two`: Les tokenizers SDXL (CLIP ViT-L et OpenCLIP ViT-bigG).
    *   `target_size`: La résolution à laquelle les images seront redimensionnées et recadrées.
    *   `center_crop`: Booléen indiquant s'il faut utiliser un recadrage central ou aléatoire.
    *   Lors de l'initialisation, elle parcourt `instance_data_root`, collecte les chemins des images, lit les légendes associées (ou utilise le fallback), et stocke les tailles originales des images.
    *   Elle définit également les transformations d'images (redimensionnement, recadrage, conversion en tenseur, normalisation).

*   **`__len__(self)`**: Retourne le nombre total d'images d'instance.

*   **`__getitem__(self, index)`**: Retourne un dictionnaire contenant un exemple d'entraînement :
    *   `instance_images`: L'image prétraitée (tenseur normalisé).
    *   `input_ids_one`, `input_ids_two`: Les IDs des tokens de la légende, tokenisés par les deux tokenizers respectifs.
    *   `original_size_hw`: Taille originale de l'image (hauteur, largeur).
    *   `crop_coords_top_left_yx`: Coordonnées (y, x) du coin supérieur gauche du recadrage appliqué à l'image redimensionnée.
    *   Gère les erreurs de chargement d'image en retournant un autre exemple aléatoire.

### Fonction `collate_fn`

Cette fonction est utilisée par `torch.utils.data.DataLoader` pour assembler les exemples individuels retournés par `DreamBoothDataset` en un batch.

*   **`collate_fn(examples)`**:
    *   Prend une liste de dictionnaires (chaque dictionnaire est un exemple du dataset).
    *   Empile les tenseurs d'images (`pixel_values`) et les IDs de tokens (`input_ids_one`, `input_ids_two`).
    *   Retourne un dictionnaire contenant les tenseurs batchés et les listes de métadonnées (`original_sizes_hw`, `crop_coords_top_left_yx`).

### Classe `LoRATrainingModule`

C'est la classe principale qui gère l'interface utilisateur, la logique d'entraînement et l'interaction avec les autres parties de l'application.

#### Initialisation (`__init__`)

*   **`__init__(self, global_translations, model_manager_instance, gestionnaire_instance, global_config)`**:
    *   Stocke les instances de `global_translations`, `ModelManager`, `GestionModule`, et `global_config`.
    *   Initialise des états comme `is_training` (booléen) et `stop_event` (pour arrêter l'entraînement).
    *   Définit les chemins par défaut pour les projets LoRA (`LORA_PROJECTS_DIR`) et la sauvegarde finale des LoRAs (`LORAS_DIR`) en se basant sur `global_config`.
    *   Récupère le nom du modèle SDXL par défaut pour l'entraînement depuis `global_config`.
    *   Détermine le `device` (CPU/GPU) et `torch_dtype` (précision des tenseurs).

#### Création de l'Interface Utilisateur (`create_tab`)

*   **`create_tab(self, module_translations_from_gestionnaire)`**:
    *   Construit l'onglet Gradio pour l'entraînement LoRA.
    *   **Préparation des Données (Colonne de Gauche)**:
        *   `input_images_dir`: Champ pour le chemin du dossier des images sources.
        *   `trigger_word`: Mot-déclencheur à ajouter aux légendes.
        *   `concept`: Nom du concept pour l'organisation des données.
        *   `caption_task_dropdown`: Choix de la tâche Florence-2 pour le légendage.
    *   **Paramètres d'Entraînement (Colonne de Gauche)**:
        *   `lora_output_name`: Nom du fichier LoRA final.
        *   `base_model_dropdown`: Sélection du modèle SDXL de base pour l'entraînement.
        *   `training_project_dir`: Dossier racine pour les projets d'entraînement.
        *   `epochs`, `learning_rate_dropdown`, `batch_size`, `resolution`: Hyperparamètres de base.
        *   Options Avancées (dans des accordéons) :
            *   `network_dim` (rang LoRA), `network_alpha`.
            *   `train_unet_only`, `train_text_encoder`: Cases à cocher pour spécifier les parties du modèle à entraîner.
            *   `optimizer_dropdown`, `lr_scheduler_dropdown`, `mixed_precision_dropdown`.
            *   `save_every_n_epochs`: Fréquence de sauvegarde des checkpoints.
    *   **Contrôles et Logs (Colonne de Droite)**:
        *   `prepare_train_button`: Bouton pour lancer la préparation et l'entraînement.
        *   `stop_button`: Bouton pour arrêter l'entraînement en cours.
        *   `status_output_html`: Zone HTML pour afficher les logs d'état.
    *   Connecte les événements des boutons aux fonctions correspondantes (`run_preparation_and_training_thread`, `stop_training_wrapper`).

#### Gestion du Thread d'Entraînement (`run_preparation_and_training_thread`)

*   **`run_preparation_and_training_thread(self, *args)`**:
    *   Cette fonction est un générateur qui est appelé lorsque l'utilisateur clique sur "Préparer et Entraîner".
    *   Elle vérifie si un entraînement est déjà en cours.
    *   Réinitialise l'état (`is_training`, `stop_event`, `status_log_list`).
    *   Met à jour l'interface pour indiquer le début du processus.
    *   Crée une `queue.Queue` pour permettre au thread d'entraînement de communiquer les messages de log au thread principal de l'UI.
    *   Lance la fonction `_actual_preparation_and_training` dans un nouveau `threading.Thread`.
    *   Boucle pour récupérer les messages de la `log_queue` et mettre à jour l'HTML des logs (`status_output_html`) via `yield`.
    *   Attend la fin du thread d'entraînement.
    *   Met à jour l'état final de l'interface.

#### Logique Principale d'Entraînement (`_actual_preparation_and_training`)

C'est la fonction exécutée dans le thread séparé. Elle contient la logique métier de l'entraînement.

*   **`_actual_preparation_and_training(self, log_queue, input_images_dir, ..., save_every_n_epochs)`**:
    1.  **Validation des Entrées**: Vérifie que les chemins et paramètres essentiels sont fournis.
    2.  **Préparation des Données d'Instance**:
        *   Crée les dossiers de projet (`training_project_dir / lora_output_name / data / concept_images`).
        *   Parcourt les images du `input_images_dir`.
        *   Pour chaque image :
            *   La recadre à une taille fixe (ex: 1024x1024) si elle est plus grande. Les images plus petites sont ignorées.
            *   Sauvegarde l'image recadrée dans le dossier de données du projet.
            *   Génère une légende en utilisant `generate_prompt_from_image` (Florence-2).
            *   Ajoute le `trigger_word` à la légende.
            *   Sauvegarde la légende dans un fichier `.txt` correspondant.
        *   Décharge le modèle de légendage (Florence-2) après utilisation.
    3.  **Chargement du Modèle de Base et Configuration PEFT**:
        *   Détermine le `torch_dtype` de chargement en fonction du choix de `mixed_precision_choice`.
        *   Charge le pipeline `StableDiffusionXLPipeline` à partir du fichier `.safetensors` du modèle de base sélectionné.
        *   Extrait les composants nécessaires : `tokenizer_one`, `tokenizer_two`, `text_encoder_one`, `text_encoder_two`, `vae`, `unet`, `noise_scheduler`.
        *   Déplace les composants sur le bon `device` et avec le bon `dtype`.
        *   Désactive les gradients (`requires_grad_(False)`) et met en mode `eval()` les modèles qui ne seront pas entraînés (VAE, et les encodeurs de texte/UNet si non LoRAfiés).
        *   Supprime l'objet `pipeline` complet pour libérer la mémoire.
        *   Configure `LoraConfig` pour l'UNet (et optionnellement pour les encodeurs de texte).
        *   Applique `unet.add_adapter(lora_config)` (et de même pour les encodeurs de texte si `train_text_encoder` est coché).
        *   Si la précision mixte `fp16` est utilisée, caste les paramètres LoRA en `fp32` pour la stabilité de l'entraînement via `cast_training_params`.
        *   Logue le nombre de paramètres entraînables.
    4.  **Création du `DreamBoothDataset` et `DataLoader`**:
        *   Instancie `DreamBoothDataset` avec les données préparées.
        *   Crée un `DataLoader` pour itérer sur le dataset par batches.
    5.  **Configuration de l'Optimiseur et du Scheduler**:
        *   Collecte les paramètres nécessitant des gradients (ceux des adaptateurs LoRA).
        *   Instancie l'optimiseur choisi (AdamW, AdamW8bit avec `bitsandbytes`, ou Lion).
        *   Instancie un scheduler de taux d'apprentissage (`get_scheduler` de Diffusers).
    6.  **Boucle d'Entraînement**:
        *   Itère sur le nombre d'époques.
        *   Pour chaque époque, itère sur les batches du `DataLoader`.
        *   **Forward Pass**:
            *   Encode les images du batch en latents avec le VAE (en `torch.float32` et `torch.no_grad`).
            *   Ajoute du bruit aux latents selon un `timestep` aléatoire.
            *   Encode les légendes du batch en embeddings de texte avec les deux encodeurs de texte (avec `torch.no_grad` si les encodeurs ne sont pas entraînés).
            *   Calcule les `add_time_ids` (contiennent la taille originale, les coordonnées de crop, et la taille cible) pour SDXL.
            *   Prédit le bruit avec l'UNet (LoRAfié) en lui passant les latents bruités, les timesteps, les embeddings de prompt, et les `added_cond_kwargs` (qui incluent les `pooled_prompt_embeds` et `add_time_ids`).
        *   **Calcul de la Perte**: Calcule la perte MSE (Mean Squared Error) entre le bruit prédit par l'UNet et le bruit original ajouté.
        *   **Backward Pass et Optimisation**:
            *   Effectue la rétropropagation (`loss.backward()`).
            *   Applique le gradient clipping (`torch.nn.utils.clip_grad_norm_`) pour éviter l'explosion des gradients.
            *   Met à jour les poids avec l'optimiseur (`optimizer.step()`).
            *   Met à jour le scheduler de taux d'apprentissage (`lr_scheduler.step()`).
            *   Utilise `torch.amp.GradScaler` si la précision mixte `fp16` est activée.
        *   Logue la progression (époque, étape, perte).
        *   **Sauvegarde des Checkpoints**: Si `save_every_n_epochs` est configuré, sauvegarde les poids LoRA de l'UNet (et des encodeurs de texte si entraînés) dans un sous-dossier `checkpoints`. Utilise `unet.save_pretrained(..., adapter_name="default")`.
    7.  **Fin de l'Entraînement**:
        *   Si l'entraînement est arrêté par l'utilisateur, logue un message et nettoie.
        *   **Sauvegarde Finale du LoRA**:
            *   Crée le dossier de destination final dans `LORAS_DIR / lora_output_name`.
            *   Récupère les `state_dict` des adaptateurs LoRA de l'UNet (et des encodeurs de texte) via `get_peft_model_state_dict`.
            *   Utilise `StableDiffusionXLPipeline.save_lora_weights` pour sauvegarder les poids LoRA au format `.safetensors` compatible avec Diffusers.
        *   Logue la fin de l'entraînement et le temps total.
    8.  **Nettoyage**: Supprime les objets volumineux (modèles, dataloader, etc.) et vide le cache CUDA.

#### Encodage des Prompts (`_encode_prompt`)

*   **`_encode_prompt(self, text_encoders, tokenizers, prompt_text_list, text_input_ids_list=None)`**:
    *   Prend en entrée les deux encodeurs de texte SDXL, leurs tokenizers respectifs, et soit une liste de chaînes de prompts, soit une liste d'IDs de tokens pré-tokenisés.
    *   Pour chaque encodeur :
        *   Tokenize les prompts (si `text_input_ids_list` n'est pas fourni).
        *   Passe les IDs de tokens à travers l'encodeur de texte.
        *   Extrait les embeddings de l'avant-dernière couche cachée (`prompt_embeds_out[2][-2]`).
        *   Le deuxième encodeur (CLIPTextModelWithProjection) fournit également les `pooled_prompt_embeds` (`prompt_embeds_out[0]`).
    *   Concatène les embeddings des deux encodeurs pour former `final_prompt_embeds`.
    *   Retourne `final_prompt_embeds` et `pooled_prompt_embeds`.

#### Calcul des `time_ids` (`_compute_time_ids`)

*   **`_compute_time_ids(self, original_size_hw, crop_coords_top_left_yx, target_size_hw, device, dtype)`**:
    *   Fonction utilitaire pour SDXL qui prépare les `add_time_ids`.
    *   Ces IDs encodent la taille originale de l'image, les coordonnées du recadrage, et la taille cible de l'image.
    *   Retourne un tenseur contenant ces informations, formaté pour être utilisé par l'UNet.

---

## 4. Flux de Travail

1.  **Utilisateur**: Fournit le chemin des images sources, un mot-déclencheur, un nom de concept, et configure les paramètres d'entraînement via l'interface Gradio.
2.  **Clic sur "Préparer et Entraîner"**:
    *   `run_preparation_and_training_thread` est appelé.
    *   Un nouveau thread est lancé pour exécuter `_actual_preparation_and_training`.
3.  **Préparation des Données (Thread)**:
    *   Les images sont copiées, recadrées, et sauvegardées dans un dossier de projet.
    *   Les légendes sont générées (Florence-2) et sauvegardées en fichiers `.txt`, préfixées par le mot-déclencheur.
4.  **Chargement et Configuration des Modèles (Thread)**:
    *   Le modèle SDXL de base est chargé.
    *   Les adaptateurs LoRA sont ajoutés à l'UNet (et aux encodeurs de texte si spécifié).
5.  **Entraînement (Thread)**:
    *   Le `DreamBoothDataset` et le `DataLoader` sont créés.
    *   L'optimiseur et le scheduler sont configurés.
    *   La boucle d'entraînement principale s'exécute, mettant à jour les poids LoRA.
    *   Des checkpoints sont sauvegardés périodiquement.
6.  **Sauvegarde Finale (Thread)**:
    *   Le LoRA final est sauvegardé au format `.safetensors` dans le répertoire `LORAS_DIR`.
7.  **Mise à Jour de l'UI (Thread Principal)**:
    *   Pendant tout le processus, les messages de log envoyés via la `log_queue` par le thread d'entraînement sont affichés dans l'interface.
    *   Les boutons sont activés/désactivés en conséquence.

---

## 5. Dépendances

*   **PyTorch**: Pour les opérations tensorielles et l'entraînement de réseaux neuronaux.
*   **Diffusers (Hugging Face)**: Pour les pipelines SDXL, les modèles (UNet, VAE, Schedulers), et les utilitaires d'entraînement.
*   **Transformers (Hugging Face)**: Pour les tokenizers et les modèles d'encodeurs de texte (CLIP).
*   **PEFT (Hugging Face)**: Pour l'implémentation de LoRA (`LoraConfig`, `get_peft_model_state_dict`, etc.).
*   **Gradio**: Pour la création de l'interface utilisateur.
*   **PIL (Pillow)**: Pour la manipulation d'images.
*   **bitsandbytes** (Optionnel): Pour l'optimiseur AdamW8bit (économie de mémoire).
*   **lion-pytorch** (Optionnel): Pour l'optimiseur Lion.
*   **tqdm**: Pour les barres de progression en console.

---

## 6. Configuration

Le module utilise `global_config` (un dictionnaire passé lors de l'initialisation) pour certains chemins et paramètres :

*   `LORA_PROJECTS_DIR`: Dossier racine où les données temporaires et les checkpoints de chaque projet d'entraînement LoRA sont stockés. (Défaut: "LoRA_Projects_Internal")
*   `LORAS_DIR`: Dossier où les fichiers LoRA finaux (`.safetensors`) sont sauvegardés pour être utilisés par l'application. (Défaut: "models/loras")
*   `DEFAULT_SDXL_MODEL_FOR_LORA`: Chemin vers le fichier `.safetensors` du modèle SDXL de base à utiliser par défaut pour l'entraînement.
*   `MODELS_DIR`: Utilisé pour localiser le modèle de base si `DEFAULT_SDXL_MODEL_FOR_LORA` est un nom de fichier relatif.

---

## 7. Notes d'Utilisation et Bonnes Pratiques

*   **Mémoire GPU (VRAM)**: L'entraînement LoRA, même s'il est "efficient en paramètres", peut être gourmand en VRAM, surtout pour SDXL.
    *   Utilisez la précision mixte (`fp16` ou `bf16` si votre GPU le supporte) pour réduire l'utilisation de la VRAM.
    *   L'optimiseur `AdamW8bit` (via `bitsandbytes`) peut également aider à réduire la consommation de mémoire.
    *   Réduisez la `batch_size` si vous rencontrez des erreurs "Out of Memory" (OOM). Une taille de batch de 1 est souvent nécessaire pour les GPU avec moins de VRAM.
    *   La `resolution` a un impact significatif sur la VRAM. 1024x1024 est standard pour SDXL.
*   **Qualité des Données**: La qualité et la cohérence de vos images d'entraînement sont cruciales.
    *   Utilisez des images claires et bien cadrées du concept que vous souhaitez entraîner.
    *   Assurez-vous que les légendes générées (ou manuelles) décrivent précisément le contenu de l'image et incluent le mot-déclencheur.
*   **Hyperparamètres**:
    *   **Taux d'apprentissage (Learning Rate)**: Un des paramètres les plus importants. Commencez avec une valeur comme `1e-4` ou `5e-5` et ajustez. Un taux trop élevé peut "brûler" le modèle (résultats de mauvaise qualité), un taux trop bas peut rendre l'entraînement très lent.
    *   **Nombre d'Époques (Epochs)**: Dépend de la taille de votre dataset. Pour de petits datasets (10-20 images), quelques dizaines d'époques peuvent suffire. Pour des datasets plus grands, moins d'époques peuvent être nécessaires. Surveillez la perte.
    *   **Network Dimension (Rank)**: Un rang plus élevé (ex: 64, 128) permet au LoRA d'apprendre plus de détails mais augmente la taille du fichier et le temps d'entraînement. Un rang plus bas (ex: 8, 16, 32) est plus léger.
    *   **Network Alpha**: Souvent réglé à la moitié du `network_dim` ou égal au `network_dim`. Il agit comme un facteur d'échelle pour les poids LoRA.
*   **Mot-Déclencheur (Trigger Word)**: Choisissez un mot unique qui n'est pas courant pour éviter les conflits avec des concepts existants dans le modèle de base.
*   **Sauvegarde des Checkpoints**: Utilisez `save_every_n_epochs` pour sauvegarder des versions intermédiaires de votre LoRA. Cela vous permet de revenir en arrière si l'entraînement diverge ou si vous souhaitez tester différentes étapes.
*   **Arrêt de l'Entraînement**: Le bouton "Arrêter" permet d'interrompre proprement l'entraînement. Le LoRA ne sera pas sauvegardé dans sa version finale si l'arrêt se produit avant la fin, mais les checkpoints intermédiaires (si activés) seront conservés.
*   **Nettoyage**: Le dossier de projet (`LORA_PROJECTS_DIR / votre_lora_nom`) peut devenir volumineux avec les données et les checkpoints. Pensez à le nettoyer manuellement après avoir obtenu un LoRA final satisfaisant.

