# d:\image_to_text\cyberbill_SDXL\cyberbill_image_generator\modules\ImageEnhancement_mod.py
import os
import json
import time
from datetime import datetime
import traceback
from PIL import Image, ImageEnhance
import numpy as np
import gradio as gr
import torch # Garder torch pour le device et le nettoyage mémoire
import gc
import sys
import pkg_resources
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2
# Imports for Diffusers Upscaler
from diffusers import LDMSuperResolutionPipeline
# Imports for OneRestore (Preprocessing/Postprocessing)
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Keep for postprocessing

# Import helpers and models from separate files
from modules.ImageEnhancement_helper import load_embedder_ckpt, load_restore_ckpt
from Utils.model_manager import ModelManager
from Utils.utils import txt_color, translate, GestionModule,  enregistrer_image

# --- Configuration du Module ---
MODULE_NAME = "ImageEnhancement"
module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

# Charger les traductions spécifiques au module
try:
    with open(module_json_path, 'r', encoding="utf-8") as f:
        module_data = json.load(f)
    # Utiliser les traductions FR par défaut pour ce module
    module_translations_fr = module_data.get("language", {}).get("fr", {})
except FileNotFoundError:
    print(txt_color("[ATTENTION]", "warning"), f"Fichier JSON '{module_json_path}' non trouvé pour {MODULE_NAME}. Utilisation de traductions vides.")
    module_translations_fr = {}
except json.JSONDecodeError:
    print(txt_color("[ERREUR]", "erreur"), f"Erreur de décodage du fichier JSON '{module_json_path}' pour {MODULE_NAME}.")
    module_translations_fr = {}

# --- Constantes pour les choix ---
TASK_COLORIZE = "Colorisation"
TASK_UPSCALE = "Upscale" # La tâche reste, mais le facteur est fixe (4x)
TASK_RESTORE = "Restauration"
TASK_RETOUCH = "Retouche Auto"

# --- Pas de build_generator nécessaire ---

def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
    """Initialise le module ImageEnhancement."""
    print(txt_color("[OK] ", "ok"), f"Initialisation du module {MODULE_NAME}")
    # Passer les traductions FR spécifiques au module
    return ImageEnhancementModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class ImageEnhancementModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
        self.gestionnaire = gestionnaire_instance
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance

        # Stocker les traductions spécifiques au module
        self.module_translations = global_translations if global_translations is not None else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.global_config is None:
            print(txt_color("[ERREUR]", "erreur"), f"La configuration globale (global_config) n'a pas été reçue par {MODULE_NAME}Module.")

    def create_tab(self, module_translations):
        """Crée l'onglet Gradio pour ce module."""
        # Mettre à jour les traductions du module si elles sont passées ici
        self.module_translations = module_translations

        with gr.Tab(translate("enhancement_tab", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('enhancement_title', self.module_translations)}")

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label=translate("image_input", self.module_translations), type="pil")
                    task_radio = gr.Radio(
                        label=translate("task_choice", self.module_translations),
                        choices=[
                            translate("task_colorize", self.module_translations),
                            translate("task_upscale", self.module_translations), # L'option reste
                            translate("task_restore", self.module_translations),
                            translate("task_retouch", self.module_translations)
                        ],
                        value=None # Pas de valeur par défaut
                    )
                    # Dropdown supprimé car le modèle LDM est fixe 4x
                    process_button = gr.Button(translate("process_button", self.module_translations))

                with gr.Column(scale=1):
                    image_output = gr.Image(label=translate("image_output", self.module_translations), type="pil")

            # Wrapper function no longer needs upscale_factor_str
            def process_image_wrapper(image, task_choice, progress=gr.Progress(track_tqdm=True)):
                if image is None:
                    gr.Warning(translate("error_no_image", self.module_translations))
                    return None
                if task_choice is None:
                    gr.Warning(translate("error_no_task", self.module_translations))
                    return None

                start_time = time.time()
                progress(0, desc=translate("processing_image", self.module_translations))

                # Mapper le choix traduit vers la clé constante interne
                task_key = None
                task_map = {
                    translate("task_colorize", self.module_translations): TASK_COLORIZE,
                    translate("task_upscale", self.module_translations): TASK_UPSCALE,
                    translate("task_restore", self.module_translations): TASK_RESTORE,
                    translate("task_retouch", self.module_translations): TASK_RETOUCH,
                }
                task_key = task_map.get(task_choice)

                processed_image = None
                try:
                    # Déchargement du modèle principal si nécessaire
                    main_model_unloaded = False
                    if self.model_manager.get_current_pipe() is not None:
                        progress(0.1, desc=translate("dechargement_modele_principal", self.global_translations))
                        print(txt_color("[INFO]", "info"), translate("dechargement_modele_principal", self.global_translations))
                        self.model_manager.unload_model(gradio_mode=False) # Appeler unload_model du manager

                    # Exécution de la tâche sélectionnée
                    if task_key == TASK_COLORIZE:
                        processed_image = self.run_colorization(image, progress)
                    elif task_key == TASK_UPSCALE:
                        # No scale factor needed for this model
                        processed_image = self.run_upscale(image, progress)
                    elif task_key == TASK_RESTORE:
                        processed_image = self.run_restoration(image, progress)
                    elif task_key == TASK_RETOUCH:
                        processed_image = self.run_retouch(image, progress)
                    else:
                        # Ce cas ne devrait pas arriver si task_choice est validé
                        raise ValueError("Tâche inconnue sélectionnée.")

                    # Si une fonction run_* retourne None (erreur interne gérée), on arrête
                    if processed_image is None:
                        # L'erreur a déjà été loggée ou affichée via gr.Error/gr.Warning
                        return None

                    # Sauvegarde de l'image traitée
                    progress(0.9, desc=translate("sauvegarde_image", self.module_translations))
                    date_str = datetime.now().strftime("%Y_%m_%d")
                    heure_str = datetime.now().strftime("%H_%M_%S")
                    save_dir_base = self.global_config.get("SAVE_DIR")
                    if not save_dir_base:
                         error_msg = "Erreur: Le chemin de sauvegarde (SAVE_DIR) n'est pas défini dans la configuration."
                         print(txt_color("[ERREUR]", "erreur"), error_msg)
                         raise gr.Error(error_msg) # Remonter l'erreur à Gradio
                    save_dir = os.path.join(save_dir_base, date_str)
                    os.makedirs(save_dir, exist_ok=True)
                    # Utiliser la clé constante pour le nom de fichier
                    task_name_for_file = task_key.lower() if task_key else "unknown_task"
                    # Ajouter '4x' au nom de fichier pour l'upscale
                    if task_key == TASK_UPSCALE:
                        filename_part = f"enhanced_{task_name_for_file}_4x_{date_str}_{heure_str}.png"
                    else:
                        filename_part = f"enhanced_{task_name_for_file}_{date_str}_{heure_str}.png"
                    chemin_image = os.path.join(save_dir, filename_part)
                    # Utiliser la fonction d'enregistrement globale
                    enregistrer_image(processed_image, chemin_image, self.global_translations, "PNG")

                    # Afficher le temps total et un message de succès
                    elapsed_time = f"{(time.time() - start_time):.2f} sec"
                    print(txt_color("[INFO] ","info"),f"{translate('temps_total_generation', self.module_translations)} : {elapsed_time}")
                    gr.Info(f"{translate('success_processing', self.module_translations)} ({elapsed_time})", 3.0)

                    return processed_image


                # Gestion globale des exceptions
                except gr.Error as gr_e: # Remonter les erreurs Gradio directement
                    raise gr_e
                except Exception as e:
                    error_msg = translate("error_processing", self.module_translations)
                    print(txt_color("[ERREUR]", "erreur"), f"{error_msg} ({task_choice}): {e}")
                    traceback.print_exc()
                    # Afficher une erreur générique dans Gradio
                    raise gr.Error(f"{error_msg}: {str(e)[:100]}...")
                finally:
                    # --- Rechargement Modèle Principal (si déchargé et si souhaité) ---
                    # Pour l'instant, on suppose que l'utilisateur rechargera si besoin.
                    pass

            # Dropdown visibility logic removed

            process_button.click(
                fn=process_image_wrapper,
                inputs=[image_input, task_radio], # Removed dropdown from inputs
                outputs=[image_output],
            )
        return tab

    # --- Helper _load_and_run_model ---
    def _load_and_run_model(self, model_key, load_function, run_function, image, progress, progress_start=0.2, progress_end=0.8, run_args=None):
        """Helper générique pour charger, exécuter et décharger un modèle(s)."""
        models = None # Utiliser 'models' pour potentiellement contenir plusieurs modèles (ex: dict)
        result_image = None
        model_display_name = model_key # Utiliser la clé du modèle pour l'affichage

        try:
            # Chargement
            loading_msg_template = translate('loading_model', self.module_translations)
            loading_msg = loading_msg_template.format(model_name=model_display_name)
            progress(progress_start, desc=loading_msg)
            print(txt_color("[INFO]", "info"), loading_msg)

            # Appeler la fonction de chargement fournie
            models = load_function() # Peut retourner un objet unique ou une collection (dict)
            print(txt_color("[OK]", "ok"), f"Modèle {model_display_name} chargé.")

            # Exécution
            executing_msg = translate("executing_model", self.module_translations).format(model_name=model_display_name) # Nouvelle clé
            progress((progress_start + progress_end) / 2, desc=executing_msg)
            print(txt_color("[INFO]", "info"), executing_msg)

            # Préparer les arguments pour run_function
            # L'image est toujours le premier argument après le modèle/pipeline
            args_to_pass = [models, image] # Passer 'models' (peut être un dict ou un objet)
            if run_args:
                args_to_pass.extend(run_args)

            # Appeler la fonction d'exécution fournie avec les arguments préparés
            result_image = run_function(*args_to_pass)

            # Vérifier si l'exécution a retourné None (indiquant une erreur gérée)
            if result_image is None:
                 # L'erreur a déjà été loggée dans run_function, on propage None
                 print(txt_color("[AVERTISSEMENT]", "warning"), f"L'exécution de {model_display_name} a retourné None.")
                 return None

            print(txt_color("[OK]", "ok"), f"Exécution {model_display_name} terminée.")
            return result_image

        except Exception as e:
            # Gérer les erreurs non capturées dans load_function ou run_function
            error_msg = f"Erreur avec le modèle {model_display_name}"
            print(txt_color("[ERREUR]", "erreur"), f"{error_msg}: {e}")
            traceback.print_exc()
            # Afficher l'erreur dans Gradio
            gr.Error(f"{error_msg}: {str(e)[:100]}...")
            return None # Retourner None en cas d'erreur non gérée
        finally:
            # Déchargement
            unloading_msg_template = translate('unloading_model', self.module_translations)
            unloading_msg = unloading_msg_template.format(model_name=model_display_name)
            progress(progress_end, desc=unloading_msg)
            print(txt_color("[INFO]", "info"), unloading_msg)
            # Gérer la suppression si 'models' est un dictionnaire ou un objet unique
            if models is not None:
                if isinstance(models, dict):
                    for key in list(models.keys()): # Itérer sur une copie des clés
                        print(f"[DEBUG] Suppression du modèle: {key}")
                        del models[key]
                else:
                    print(f"[DEBUG] Suppression du modèle: {model_display_name}")
                    del models
                models = None # Assurer que la variable est None après la suppression
            # Nettoyage mémoire GPU/CPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            # This message might be slightly inaccurate if multiple models were loaded/unloaded
            unloaded_msg_template = translate('model_unloaded', self.module_translations)
            unloaded_msg = unloaded_msg_template.format(model_name=model_display_name)
            print(txt_color("[OK]", "ok"), unloaded_msg)

    # --- run_colorization avec ModelScope ---
    def run_colorization(self, image: Image.Image, progress):
        """Exécute la tâche de colorisation en utilisant modelscope."""
        print(txt_color("[INFO]", "info"), translate("colorization_start", self.module_translations))

        model_id = 'damo/cv_ddcolor_image-colorization' # Modèle ModelScope

        # Fonction pour charger le pipeline modelscope
        def load_modelscope_colorizer():
            device_param = 'gpu' if self.device.type == 'cuda' else 'cpu'
            print(f"[INFO] Tentative de chargement du pipeline ModelScope Colorization sur '{device_param}'")
            # Créer et retourner le pipeline
            colorizer_pipe = pipeline(Tasks.image_colorization, model=model_id, device=device_param)
            return colorizer_pipe

        # Fonction pour exécuter le pipeline modelscope
        def run_modelscope_colorizer(pipe, img_pil):
            # Mettre le try/except ici pour gérer les erreurs spécifiques à l'exécution
            try:
                # 1. Convertir PIL (RGB) en NumPy (BGR) pour ModelScope/OpenCV
                img_rgb_np = np.array(img_pil)
                # Vérifier si l'image a bien 3 canaux
                if len(img_rgb_np.shape) != 3 or img_rgb_np.shape[2] != 3:
                     print(txt_color("[AVERTISSEMENT]", "warning"), translate("image_not_rgb", self.module_translations).format(shape=img_rgb_np.shape))
                     if len(img_rgb_np.shape) == 3 and img_rgb_np.shape[2] == 4:
                         img_pil = img_pil.convert('RGB')
                         img_rgb_np = np.array(img_pil)
                     elif len(img_rgb_np.shape) == 2: # Grayscale
                         img_pil = img_pil.convert('RGB')
                         img_rgb_np = np.array(img_pil)
                     else:
                         raise ValueError(translate("unsupported_input_format", self.module_translations))

                img_bgr_np = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)

                # 2. Exécuter le pipeline ModelScope
                result = pipe(img_bgr_np)

                # 3. Extraire l'image de sortie (NumPy BGR)
                output_img_bgr = result[OutputKeys.OUTPUT_IMG]

                # 4. Convertir NumPy (BGR) en PIL (RGB) pour Gradio
                output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
                final_image_pil = Image.fromarray(output_img_rgb)

                return final_image_pil

            except Exception as e_pipe:
                 print(txt_color("[ERREUR]", "erreur"), translate("modelscope_pipeline_error", self.module_translations).format(error=e_pipe))
                 traceback.print_exc()
                 return None # Retourner None en cas d'erreur

        # Appel de l'helper avec les fonctions de chargement et d'exécution
        return self._load_and_run_model(model_id, load_modelscope_colorizer, run_modelscope_colorizer, image, progress)

    # --- run_upscale avec Diffusers LDM ---
    def run_upscale(self, image: Image.Image, progress):
        """Exécute la tâche d'upscale 4x en utilisant Diffusers LDM."""
        print(txt_color("[INFO]", "info"), translate("upscale_start_diffusers", self.module_translations)) # Nouvelle clé

        model_id = "CompVis/ldm-super-resolution-4x-openimages"
        # Configurer les paramètres d'inférence (ajuster si nécessaire)
        num_inference_steps = 100
        eta = 1.0

        # Fonction pour charger le pipeline Diffusers
        def load_diffusers_upscaler():
            print(f"[INFO] Tentative de chargement du pipeline Diffusers Upscaler sur '{self.device}'")
            # Utiliser float16 pour économiser de la VRAM sur GPU si compatible
            dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            upscaler_pipe = LDMSuperResolutionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            upscaler_pipe = upscaler_pipe.to(self.device)
            return upscaler_pipe

        # Fonction pour exécuter le pipeline Diffusers
        def run_diffusers_upscaler(pipe, img_pil):
            try:
                # Diffusers prend une image PIL RGB
                if img_pil.mode != "RGB":
                    print(txt_color("[AVERTISSEMENT]", "warning"), translate("converting_to_rgb_diffusers", self.module_translations)) # Nouvelle clé
                    img_pil = img_pil.convert("RGB")

                # Exécuter le pipeline
                upscaled_result = pipe(img_pil, num_inference_steps=num_inference_steps, eta=eta)
                output_image = upscaled_result.images[0]

                if not isinstance(output_image, Image.Image):
                     raise TypeError(translate("diffusers_output_not_pil", self.module_translations).format(type=type(output_image))) # Nouvelle clé

                return output_image

            except Exception as e_pipe:
                 print(txt_color("[ERREUR]", "erreur"), translate("diffusers_pipeline_error", self.module_translations).format(error=e_pipe)) # Nouvelle clé
                 traceback.print_exc()
                 return None # Retourner None en cas d'erreur

        # Appel de l'helper sans run_args car le facteur est fixe (4x)
        return self._load_and_run_model(model_id, load_diffusers_upscaler, run_diffusers_upscaler, image, progress)
    # --- FIN run_upscale ---

    # --- run_restoration avec OneRestore (refactorisé) ---
    def run_restoration(self, image: Image.Image, progress):
        """Exécute la tâche de restauration en utilisant OneRestore."""
        print(txt_color("[INFO]", "info"), translate("restoration_start_onerestore", self.module_translations)) # Nouvelle clé

        # --- Configuration OneRestore ---
        # Chemins relatifs au dossier principal du projet ou configurables
        # Utilisation de os.path.dirname(__file__) pour obtenir le chemin du module actuel
        module_dir = os.path.dirname(__file__) # Chemin vers le dossier 'modules'
        # Les modèles sont maintenant directement dans 'modules/ImageEnhancement_models/'
        models_dir = os.path.join(module_dir, "ImageEnhancement_models")

        embedder_ckpt_path = os.path.join(models_dir, "embedder_model.tar")
        restorer_ckpt_path = os.path.join(models_dir, "onerestore_real.tar") # Utiliser le nom de fichier correct
        model_display_name = "OneRestore"

        # Vérifier l'existence des fichiers checkpoints
        if not os.path.exists(embedder_ckpt_path):
            error_msg = translate("onerestore_embedder_not_found", self.module_translations).format(path=embedder_ckpt_path)
            print(txt_color("[ERREUR]", "erreur"), error_msg)
            gr.Error(error_msg)
            return None
        if not os.path.exists(restorer_ckpt_path):
             error_msg = translate("onerestore_restorer_not_found", self.module_translations).format(path=restorer_ckpt_path)
             print(txt_color("[ERREUR]", "erreur"), error_msg)
             gr.Error(error_msg)
             return None

        # --- Fonctions Helper spécifiques à OneRestore ---

        # Fonction pour charger les modèles OneRestore (Embedder + Restorer)
        def load_onerestore_models():
            # Utilise les fonctions importées depuis ImageEnhancement_helper
            embedder = load_embedder_ckpt(self.device, freeze_model=True, ckpt_name=embedder_ckpt_path)
            restorer = load_restore_ckpt(self.device, freeze_model=True, ckpt_name=restorer_ckpt_path)
            # Retourne les modèles dans un dictionnaire
            return {"embedder": embedder, "restorer": restorer}

        # Fonction pour exécuter les modèles OneRestore
        def run_onerestore_models(models, img_pil):
            embedder = models["embedder"]
            restorer = models["restorer"]
            try:
                # --- Prétraitement ---
                # +++ AJOUT: Vérification et redimensionnement de l'image d'entrée +++
                MAX_DIMENSION = 1024 # Définir une dimension maximale (ex: 1024 pixels). Ajustez si nécessaire.
                width, height = img_pil.size
                if width > MAX_DIMENSION or height > MAX_DIMENSION:
                    resize_warning = self.module_translations.get("onerestore_resizing_input", "Image too large ({orig_w}x{orig_h}). Resizing to {max_dim}px max for OneRestore.").format(
                        orig_w=width, orig_h=height, max_dim=MAX_DIMENSION
                    )
                    print(txt_color("[AVERTISSEMENT]", "warning"), resize_warning)
                    gr.Warning(resize_warning) # Informer l'utilisateur via Gradio

                    # Utiliser LANCZOS pour un redimensionnement de meilleure qualité
                    try:
                        resampling_filter = Image.Resampling.LANCZOS
                    except AttributeError:
                        # Fallback pour les anciennes versions de Pillow
                        resampling_filter = Image.LANCZOS

                    # Crée une copie pour ne pas modifier l'original en place avec thumbnail
                    img_pil_copy = img_pil.copy()
                    img_pil_copy.thumbnail((MAX_DIMENSION, MAX_DIMENSION), resampling_filter)
                    img_pil = img_pil_copy # Remplacer l'image par la version redimensionnée

                print("[INFO] Prétraitement de l'image pour OneRestore...")
                img_np = np.array(img_pil)
                if img_np.ndim == 2:
                    print(txt_color("[AVERTISSEMENT]", "warning"), translate("onerestore_grayscale_input", self.module_translations))
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                elif img_np.shape[2] == 4:
                     print(txt_color("[AVERTISSEMENT]", "warning"), translate("onerestore_rgba_input", self.module_translations))
                     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

                # Tensor pour Restorer (normalisé [0, 1])
                lq_re = torch.from_numpy((img_np / 255.0).transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

                # Tensor pour Embedder (redimensionné et normalisé [-1, 1] via self.transform dans Embedder)
                img_pil_rgb = Image.fromarray(img_np)
                transform_resize_for_embedder = transforms.Compose([
                    transforms.Resize([224, 224]),
                    transforms.ToTensor() # Convertit en [0, 1]
                ])
                lq_em = transform_resize_for_embedder(img_pil_rgb).unsqueeze(0).to(self.device)
                # La normalisation spécifique à l'Embedder est appliquée dans son forward

                # --- Inférence ---
                print("[INFO] Exécution de l'inférence OneRestore (mode automatique)...")
                with torch.no_grad():
                    # Embedder
                    text_embedding, _, [text] = embedder(lq_em, 'image_encoder')
                    print(txt_color("[INFO]", "info"), translate("onerestore_detected_degradation", self.module_translations).format(degradation=text))

                    # Restorer
                    output_tensor = restorer(lq_re, text_embedding)

                # --- Post-traitement ---
                print("[INFO] Post-traitement de l'image restaurée...")
                output_tensor = output_tensor.squeeze(0).cpu()
                # Le modèle OneRestore dénormalise déjà en interne (self.denorm)
                output_tensor = torch.clamp(output_tensor, 0.0, 1.0) # Assurer la plage [0, 1]
                restored_image_pil = TF.to_pil_image(output_tensor)

                return restored_image_pil

            except Exception as e_pipe:
                print(txt_color("[ERREUR]", "erreur"), translate("onerestore_pipeline_error", self.module_translations).format(error=e_pipe))
                traceback.print_exc()
                return None # Indiquer l'échec

        # Appel de l'helper générique
        return self._load_and_run_model(model_display_name, load_onerestore_models, run_onerestore_models, image, progress)

    # --- run_retouch (Exemple Simple) ---
    def run_retouch(self, image: Image.Image, progress):
        """Exécute la tâche de retouche automatique (exemple simple)."""
        print(txt_color("[INFO]", "info"), translate("retouch_start", self.module_translations))
        try:
            # Exemple simple: Augmenter légèrement le contraste et la netteté
            progress(0.3, desc="Application retouche contraste...")
            enhancer_contrast = ImageEnhance.Contrast(image)
            retouched_image = enhancer_contrast.enhance(1.2) # Augmentation légère du contraste

            progress(0.6, desc="Application retouche netteté...")
            enhancer_sharpness = ImageEnhance.Sharpness(retouched_image)
            retouched_image = enhancer_sharpness.enhance(1.1) # Augmentation légère de la netteté

            progress(0.5, desc=translate("applying_saturation", self.module_translations)) # Nouvelle clé possible
            enhancer_color = ImageEnhance.Color(retouched_image)
            retouched_image = enhancer_color.enhance(1.15) 

            print(txt_color("[OK]", "ok"), translate("retouch_simple_applied", self.module_translations))
            return retouched_image
        except Exception as e:
            # Logger l'erreur et retourner l'image originale
            error_msg = translate("retouch_simple_error", self.module_translations).format(error=e)
            print(txt_color("[ERREUR]", "erreur"), error_msg)
            traceback.print_exc()
            # Afficher une erreur dans Gradio
            gr.Error(error_msg)
            return image # Retourner l'image originale en cas d'erreur
