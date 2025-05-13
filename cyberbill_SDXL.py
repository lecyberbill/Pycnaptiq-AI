import random
import importlib
import os
import math
import shutil
import time
import threading
import traceback
from datetime import datetime
import json
import gradio as gr
import numpy as np
from gradio import update
from diffusers import  StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, \
    LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, KDPM2DiscreteScheduler, StableDiffusionXLInpaintPipeline, \
    KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, DPMSolverSDEScheduler, DPMSolverSinglestepScheduler
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from core.version import version
from Utils.callback_diffuser import latents_to_rgb, create_callback_on_step_end, create_inpainting_callback
from Utils.model_manager import ModelManager
from core.translator import translate_prompt
from core.Inpaint import apply_mask_effects
from core.image_prompter import init_image_prompter, generate_prompt_from_image
from Utils.utils import GestionModule, enregistrer_etiquettes_image_html, finalize_html_report_if_needed, charger_configuration, gradio_change_theme, lister_fichiers, styles_fusion, create_progress_bar_html, load_modules_js,\
    telechargement_modele, txt_color, str_to_bool, load_locales, translate, get_language_options, enregistrer_image, preparer_metadonnees_image, check_gpu_availability, ImageSDXLchecker
from Utils.sampler_utils import SAMPLER_DEFINITIONS, get_sampler_choices, get_sampler_key_from_display_name, apply_sampler_to_pipe
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process
torch.backends.cudnn.deterministic = True
import queue
from compel import Compel, ReturnedEmbeddingsType
from io import BytesIO
from presets.presets_Manager import PresetManager
import functools
from functools import partial
from core.batch_runner import run_batch_from_json
from core.pipeline_executor import execute_pipeline_task_async


print (f"cyberbill_SDXL version {txt_color(version(),'info')}")
# Load the configuration first
config = charger_configuration()
# Initialisation de la langue

DEFAULT_LANGUAGE = config.get("LANGUAGE", "fr")  # Utilisez 'fr' comme langue par défaut si 'LANGUAGE' n'est pas défini.
translations = load_locales(DEFAULT_LANGUAGE)

# --- Importer et initialiser PresetManager ---
preset_manager = PresetManager(translations)
# --- États pour la sauvegarde des presets ---

selected_sampler_key_state = gr.State(value=None)


# Dossiers contenant les modèles
MODELS_DIR = config["MODELS_DIR"]
VAE_DIR = config["VAE_DIR"]
LORAS_DIR = config["LORAS_DIR"]
INPAINT_MODELS_DIR = config["INPAINT_MODELS_DIR"]
SAVE_DIR = config["SAVE_DIR"]
IMAGE_FORMAT = config["IMAGE_FORMAT"].upper() 
FORMATS = config["FORMATS"]
FORMATS = [f'{item["dimensions"]} {translate(item["orientation"], translations)}' for item in FORMATS]
NEGATIVE_PROMPT = config["NEGATIVE_PROMPT"]
GRADIO_THEME = config["GRADIO_THEME"]
AUTHOR= config["AUTHOR"]
SHARE= config["SHARE"]
OPEN_BROWSER= config["OPEN_BROWSER"]
DEFAULT_MODEL= config["DEFAULT_MODEL"]
PRESETS_PER_PAGE = config.get("PRESETS_PER_PAGE", 12) # Nombre de presets à afficher par page
PRESET_COLS_PER_ROW = config.get("PRESET_COLS_PER_ROW", 4)
SAVE_BATCH_JSON_PATH = config.get("SAVE_BATCH_JSON_PATH", "Output\\json_batch_files")

for style in config["STYLES"]:
        style["name"] = translate(style["key"], translations)
STYLES=config["STYLES"]
PREVIEW_QUEUE = []

# --- Initialisation du ModelManager ---
device, torch_dtype, vram_total_gb = check_gpu_availability(translations)
model_manager = ModelManager(config, translations, device, torch_dtype, vram_total_gb)

# --- Utiliser ModelManager pour lister les fichiers ---
modeles_disponibles = model_manager.list_models(model_type="standard")
vaes = model_manager.list_vaes() # Inclut "Auto"
init_image_prompter(device, translations) 
modeles_impaint = model_manager.list_models(model_type="inpainting")

if not modeles_impaint or modeles_impaint[0] == translate("aucun_modele_trouve", translations):
    modeles_impaint = [translate("aucun_modele_trouve", translations)]
    
gestionnaire = GestionModule(
    translations=translations, config=config,
    model_manager_instance=model_manager,
    preset_manager_instance=preset_manager
)


if modeles_disponibles and modeles_disponibles[0] == translate("aucun_modele_trouve", translations):
    reponse = input(f"{txt_color(translate('attention', translations),'erreur')} {translate('aucun_modele_trouve', translations)} {translate('telecharger_modele_question', translations)} ")
    if reponse.lower() == "o" or reponse.lower() == "oui" or reponse.lower() == "y" or reponse.lower() == "yes":
        lien_modele = "https://huggingface.co/QuadPipe/MegaChonkXL/resolve/main/MegaChonk-XL-v2.3.1.safetensors?download=true"
        nom_fichier = "MegaChonk-XL-v2.3.1.safetensors"
        if telechargement_modele(lien_modele, nom_fichier, MODELS_DIR, translations):
            # Recharge la liste des modèles après le téléchargement
            modeles_disponibles = lister_fichiers(MODELS_DIR, translations)
            if not modeles_disponibles:
                modeles_disponibles = [translate("aucun_modele_trouve", translations)]

print( f"{txt_color(translate('Safety', translations), 'erreur')}")

# Vérifier que le format est valide
if IMAGE_FORMAT not in ["PNG", "JPG", "WEBP"]:
    print(f"⚠️ Format {IMAGE_FORMAT}", f"{txt_color(translate('non_valide', translations), 'erreur')}", translate("utilisation_webp", translations))
    IMAGE_FORMAT = "WEBP"

 # Créer un pool de threads pour l'écriture asynchrone

html_executor = ThreadPoolExecutor(max_workers=5)
image_executor = ThreadPoolExecutor(max_workers=10) 

# Initialisation des variables globales

model_selectionne = None
vae_selctionne = "Défaut VAE"
loras_charges = {}
# Charger tous les modules
gestionnaire.charger_tous_les_modules()

# Charger le code JavaScript depuis le fichier centralisé
js_code = load_modules_js()


# Flag pour arrêter la génération
stop_event = threading.Event()

stop_gen = threading.Event()


# Flag pour signaler qu'une tâche est en cours
processing_event = threading.Event()
# Flag to indicate if an image generation is in progress
is_generating = False

global_selected_sampler_key = "sampler_euler"

# =========================
# Définition des fonctions
# =========================


def handle_module_toggle(module_name, new_state, gestionnaire_instance, preset_manager_instance):
    gestionnaire_instance.set_module_active(module_name, new_state)
    status_message = f"Module '{module_name}' {'activé' if new_state else 'désactivé'}."
    gr.Info(status_message, 1.5)





def style_choice(selected_style_user, STYLES):
    """Choisi un style dans la liste des styles.
        Args:
            selected_style_user (str): nom du style choisi par l'utilisateur
            STYLES (list): liste de dictionnaire de style
        Return:
            retourne l'enrée du dictionnaire de style selectionné
    """
    selected_style = next((item for item in STYLES if item["name"] == selected_style_user), None)
    return selected_style

#==========================
# Fonction GENERATION PROMPT A PARTIR IMAGE
#==========================

def generate_prompt_wrapper(image, current_translations):
    return generate_prompt_from_image(image, current_translations)

#==========================
# Fonction GENERATION IMAGE
#==========================

def generate_image(text, style_selection, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images, pag_enabled, pag_scale, pag_applied_layers_str, *lora_inputs):
    """Génère des images avec Stable Diffusion en utilisant execute_pipeline_task_async."""
    global lora_charges, model_selectionne, vae_selctionne, is_generating, global_selected_sampler_key, PREVIEW_QUEUE # Assurer que PREVIEW_QUEUE est global

    lora_checks = lora_inputs[:4]
    lora_dropdowns = lora_inputs[4:8]
    lora_scales = lora_inputs[8:]

    bouton_charger_update_off = gr.update(interactive=False)
    btn_generate_update_off = gr.update(interactive=False)
    bouton_charger_update_on = gr.update(interactive=True)
    btn_generate_update_on = gr.update(interactive=True)
    final_save_button_state = gr.update(interactive=False)
    btn_save_preset_off = gr.update(interactive=False)

    initial_images = []
    initial_seeds = ""
    initial_time = translate("preparation", translations)
    initial_html = ""
    initial_preview = None
    initial_progress = ""
    output_gen_data_json = None
    output_preview_image = None

    yield (
        initial_images, initial_seeds, initial_time, initial_html, initial_preview, initial_progress,
        bouton_charger_update_off, btn_generate_update_off,
        btn_save_preset_off,
        output_gen_data_json,
        output_preview_image
    )

    is_generating = True

    final_images = []
    final_seeds = ""
    final_message = ""
    final_html_msg = ""
    final_preview_img = None
    final_progress_html = ""


    try:

        if model_manager.get_current_pipe() is None:
            print(txt_color("[ERREUR] ","erreur"), translate("erreur_pas_modele", translations))
            gr.Warning(translate("erreur_pas_modele", translations), 4.0)
            final_message = translate("erreur_pas_modele", translations)
            final_html_msg = f"<p style='color: red;'>{final_message}</p>"
            yield (
                [], "", final_message, final_html_msg, 
                None, "",                             
                bouton_charger_update_on, btn_generate_update_on, 
                gr.update(interactive=False),         
                None, None                            
            )
            return # Stop the generator


        if model_manager.current_model_type != 'standard': # <-- Vérifier le type via ModelManager
            error_message = translate("erreur_mauvais_type_modele", translations)
            print(txt_color("[ERREUR] ","erreur"), error_message)
            gr.Warning(error_message, 4.0)
            # Mettre à jour les variables finales pour le bloc finally
            final_message = error_message
            final_html_msg = f"<p style='color: red;'>{error_message}</p>"
            yield (
                [], "", final_message, final_html_msg,
                None, "",
                bouton_charger_update_on, btn_generate_update_on,
                gr.update(interactive=False),
                None, None
            )
            return # Stop the generator


        #initialisation du chrono
        start_time = time.time()
        # Réinitialiser l'état d'arrêt
        stop_event.clear()
        stop_gen.clear()

        seeds = [random.randint(1, 10**19 - 1) for _ in range(num_images)] if seed_input == -1 else [seed_input] * num_images
        prompt_text = translate_prompt(text, translations) if traduire else text

        prompt_en, negative_prompt_str, selected_style_display_names = styles_fusion(
            style_selection,
            prompt_text,
            NEGATIVE_PROMPT,
            STYLES,
            translations
        )

        print(txt_color("[INFO] ","info"), f"Prompt Positif Final (string): {prompt_en}")
        print(txt_color("[INFO] ","info"), f"Prompt Négatif Final (string): {negative_prompt_str}")

        # --- Utiliser compel pour les prompts positif ET négatif ---
        compel = model_manager.get_current_compel()      
        conditioning, pooled = compel(prompt_en)
        neg_conditioning, neg_pooled = compel(negative_prompt_str)

        selected_format_parts = selected_format.split(":")[0].strip() # Utiliser selected_format ici
        width, height = map(int, selected_format_parts.split("*"))
        images = [] # Initialize a list to store all generated images
        seed_strings = []
        formatted_seeds = ""
        current_data_for_preset = None
        preview_image_for_preset = None

        lora_ui_config = {
            'lora_checks': lora_checks,
            'lora_dropdowns': lora_dropdowns,
            'lora_scales': lora_scales
        }
        message_lora = model_manager.apply_loras(lora_ui_config, gradio_mode=True)
        erreur_keyword = translate("erreur", translations).lower()
        if message_lora and erreur_keyword in message_lora.lower():
            gr.Warning(message_lora, duration=4.0)
            final_message = message_lora
            final_html_msg = f"<p style='color: red;'>{final_message}</p>"
            yield (
                [], "", final_message, final_html_msg,
                None, "",
                bouton_charger_update_on, btn_generate_update_on,
                gr.update(interactive=False),
                None, None
            )
            return # Stop the generator
        elif message_lora: # Si message mais pas erreur (ex: succès)
            print(txt_color("[INFO]", "info"), f"Message LoRA: {message_lora}")

        # Préparer pag_applied_layers
        pag_applied_layers = []
        if pag_enabled and pag_applied_layers_str:
            pag_applied_layers = [s.strip() for s in pag_applied_layers_str.split(',') if s.strip()]
            print(txt_color("[INFO]", "info"), f"Message LoRA: {message_lora}")

        # principal loop for genrated images
        for idx, seed in enumerate(seeds):
            depart_time = time.time()
            PREVIEW_QUEUE.clear() # Vider la queue d'aperçu pour cette image
            # final_image_container = {} # Remplacé par result_container de l'executor
            html_message_result = translate("generation_en_cours", translations)

            if stop_event.is_set(): # Vérifie l'arrêt global avant de commencer
                print(txt_color("[INFO] ","info"), translate("arrete_demande_apres", translations), f"{idx} {translate('images', translations)}.")
                gr.Info(translate("arrete_demande_apres", translations) + f"{idx} {translate('images', translations)}.", 3.0)
                final_message = translate("generation_arretee", translations)
                break # Sortir de la boucle for

            print(txt_color("[INFO] ","info"), f"{translate('generation_image', translations)} {idx+1} {translate('seed_utilise', translations)} {seed}")
            gr.Info(translate('generation_image', translations) + f"{idx+1} {translate('seed_utilise', translations)} {seed}", 3.0)

            # --- NOUVEAU : Appel à execute_pipeline_task_async ---
            progress_update_queue = queue.Queue() # Queue pour la progression de CETTE image
            pipe = model_manager.get_current_pipe() 
            pipeline_thread, result_container = execute_pipeline_task_async(
                pipe=pipe,
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                width=width,
                height=height,
                device=device,
                stop_event=stop_gen, # Utiliser stop_gen pour arrêter CETTE image
                translations=translations,
                progress_queue=progress_update_queue,
                preview_queue=PREVIEW_QUEUE, # Passer la queue globale d'aperçu
                pag_enabled=pag_enabled,
                pag_scale=pag_scale,
                pag_applied_layers=pag_applied_layers # Passer la liste
            )
            # --- FIN NOUVEAU ---

            # --- Boucle de mise à jour UI (adaptée) ---
            last_preview_index = 0
            last_progress_html = ""
            last_yielded_preview = None
            # On boucle tant que le thread tourne OU qu'il reste des éléments dans les queues
            while pipeline_thread.is_alive() or last_preview_index < len(PREVIEW_QUEUE) or not progress_update_queue.empty():

                # Lire la progression
                current_step, total_steps = None, num_steps
                while not progress_update_queue.empty():
                    try:
                        current_step, total_steps = progress_update_queue.get_nowait()
                    except queue.Empty: break
                new_progress_html = last_progress_html
                if current_step is not None:
                    progress_percent = int((current_step / total_steps) * 100)
                    new_progress_html = create_progress_bar_html(current_step, total_steps, progress_percent)

                # Lire les aperçus
                preview_img_to_yield = None
                preview_yielded_in_loop = False
                while last_preview_index < len(PREVIEW_QUEUE):
                    preview_img_to_yield = PREVIEW_QUEUE[last_preview_index]
                    last_preview_index += 1
                    last_yielded_preview = preview_img_to_yield
                    # Yield avec l'aperçu et la dernière progression connue
                    yield (
                        images, formatted_seeds, f"{idx+1}/{num_images}...", html_message_result,
                        preview_img_to_yield, new_progress_html, # Utiliser new_progress_html
                        bouton_charger_update_off, btn_generate_update_off,
                        gr.update(interactive=False), # Save preset button
                        output_gen_data_json,
                        output_preview_image
                    )
                    preview_yielded_in_loop = True
                
                if not preview_yielded_in_loop and new_progress_html != last_progress_html:
                     yield (
                         images, formatted_seeds, f"{idx+1}/{num_images}...", html_message_result,
                         last_yielded_preview, # <--- Utiliser le dernier aperçu au lieu de None
                         new_progress_html, # Yield la nouvelle progression
                         bouton_charger_update_off, btn_generate_update_off,
                         gr.update(interactive=False), # Save preset button
                         output_gen_data_json,
                         output_preview_image
                     )


                last_progress_html = new_progress_html # Mettre à jour la dernière progression connue

                time.sleep(0.05)
            # --- Fin Boucle de mise à jour UI ---

            pipeline_thread.join() # Attendre la fin effective du thread
            PREVIEW_QUEUE.clear() # Vider la queue d'aperçu après la fin

            # --- Gérer le résultat après la fin du thread ---
            final_status = result_container.get("status")
            final_image = result_container.get("final")
            error_details = result_container.get("error")

            # Déterminer la barre de progression finale
            final_progress_html = ""
            if final_status == "success":
                final_progress_html = create_progress_bar_html(num_steps, num_steps, 100)
            elif final_status == "error":
                final_progress_html = f'<p style="color: red;">{translate("erreur_lors_generation", translations)}</p>'
            # Si stopped, on laisse vide ou on met un message spécifique

            # Vérifier si l'arrêt a été demandé (global ou spécifique à l'image)
            if stop_event.is_set() or stop_gen.is_set() or final_status == "stopped":
                print(txt_color("[INFO]", "info"), translate("generation_arretee_pas_sauvegarde", translations))
                gr.Info(translate("generation_arretee_pas_sauvegarde", translations), 3.0)
                final_message = translate("generation_arretee", translations)
                # Yield un état intermédiaire avant de sortir de la boucle for
                yield (
                    images, " ".join(seed_strings), translate("generation_arretee", translations),
                    translate("generation_arretee_pas_sauvegarde", translations), None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, # buttons
                    gr.update(interactive=False), # Save preset button
                    None, # last_successful_generation_data
                    None  # last_successful_preview_image
                )
                break # Sortir de la boucle for des images

            # Gérer l'erreur du pipeline
            elif final_status == "error":
                error_msg = str(error_details) if error_details else "Unknown pipeline error"
                print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_pipeline', translations)}: {error_msg}")
                gr.Warning(f"{translate('erreur_pipeline', translations)}: {error_msg}", 4.0)
                yield (
                    images, " ".join(seed_strings), translate("erreur_pipeline", translations),
                    error_msg, None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, # buttons
                    gr.update(interactive=False), # Save preset button
                    None, # last_successful_generation_data
                    None  # last_successful_preview_image
                )
                continue # Passer à l'image suivante

            # Gérer le cas où aucune image n'est retournée malgré le succès
            elif final_image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image_genere", translations))
                gr.Warning(translate("erreur_pas_image_genere", translations), 4.0)
                yield (
                    images, " ".join(seed_strings), translate("erreur_pas_image_genere", translations),
                    translate("erreur_pas_image_genere", translations), None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, # buttons
                    gr.update(interactive=False), # Save preset button
                    None, # last_successful_generation_data
                    None  # last_successful_preview_image
                )
                continue # Passer à l'image suivante

            # --- Si la génération a réussi et une image est présente ---
            # Préparation des données pour le preset
            current_data_for_preset = {
                 "model": model_manager.current_model_name,  
                 "vae": model_manager.current_vae_name,  
                 "original_prompt": prompt_text,
                 "prompt": prompt_en,
                 "negative_prompt": negative_prompt_str,
                 "styles": json.dumps(selected_style_display_names if selected_style_display_names else []),
                 "guidance_scale": guidance_scale,
                 "num_steps": num_steps,
                 "sampler_key": global_selected_sampler_key,
                 "seed": seed,
                 "width": width,
                 "height": height,
                 "loras": json.dumps([{"name": name, "weight": weight} for name, weight in model_manager.loaded_loras.items()]), # Utiliser model_manager
                 "pag_enabled": pag_enabled,
                 "pag_scale": pag_scale,
                 "custom_pipeline_id": "hyoungwoncho/sd_perturbed_attention_guidance" if pag_enabled else None, # <-- AJOUT CUSTOM PIPELINE ID
                 "pag_applied_layers": pag_applied_layers_str, # Sauvegarder la chaîne brute
                 "rating": 0,
                 "notes": ""
            }
            preview_image_for_preset = final_image.copy()
            output_gen_data_json = json.dumps(current_data_for_preset)
            output_preview_image = preview_image_for_preset

            # Calcul du temps, sauvegarde, métadonnées
            temps_generation_image = f"{(time.time() - depart_time):.2f} sec"
            date_str = datetime.now().strftime("%Y_%m_%d")
            heure_str = datetime.now().strftime("%H_%M_%S")
            save_dir = os.path.join(SAVE_DIR, date_str)
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{date_str}_{heure_str}_{seed}_{width}x{height}_{idx+1}.{IMAGE_FORMAT.lower()}"
            chemin_image = os.path.join(save_dir, filename)

            lora_info = [f"{lora_dropdowns[i]} ({lora_scales[i]:.2f})" for i, check in enumerate(lora_checks) if check and lora_dropdowns[i] != translate("aucun_lora_disponible", translations)]
            lora_info_str = ", ".join([f"{name}({weight:.2f})" for name, weight in model_manager.loaded_loras.items()]) if model_manager.loaded_loras else translate("aucun_lora", translations)
            style_info_str = ", ".join(selected_style_display_names) if selected_style_display_names else translate("Aucun_style", translations)

            donnees_xmp = {
                 "Module": "SDXL Image Generation", "Creator": AUTHOR,
                 "Model": os.path.splitext(model_manager.current_model_name)[0] if model_manager.current_model_name else "N/A", # <-- Utiliser ModelManager
                 "VAE": model_manager.current_vae_name, # <-- Utiliser ModelManager
                 "Steps": num_steps, "Guidance": guidance_scale,
                 "Sampler": pipe.scheduler.__class__.__name__,
                 "IMAGE": f"{idx+1} {translate('image_sur',translations)} {num_images}",
                 "Inference": num_steps, "Style": style_info_str,
                 "original_prompt (User)": prompt_text, "Prompt": prompt_en,
                 "Negatif Prompt": negative_prompt_str, "Seed": seed,
                 "Size": selected_format_parts, # Utiliser selected_format_parts ici
                 "Loras": lora_info_str,
                 "Generation Time": temps_generation_image,
                 "PAG Enabled": pag_enabled,
                 "PAG Scale": f"{pag_scale:.2f}" if pag_enabled else "N/A", # Correction f-string
                 "PAG Custom Pipeline": "hyoungwoncho/sd_perturbed_attention_guidance" if pag_enabled else "N/A", # <-- AJOUT CUSTOM PIPELINE ID aux métadonnées XMP
                 "PAG Applied Layers": pag_applied_layers_str if pag_enabled else "N/A"
             }

            metadata_structure, prep_message = preparer_metadonnees_image(
                final_image, donnees_xmp, translations, chemin_image
            )
            print(txt_color("[INFO]", "info"), prep_message)

            gr.Info(translate("image_sauvegarder", translations) + " " + chemin_image, 3.0)
            image_future = image_executor.submit(enregistrer_image, final_image, chemin_image, translations, IMAGE_FORMAT, metadata_to_save=metadata_structure)
            html_future = html_executor.submit(enregistrer_etiquettes_image_html, chemin_image, donnees_xmp, translations, is_last_image=(idx == num_images - 1))

            print(txt_color("[OK] ","ok"),translate("image_sauvegarder", translations), txt_color(f"{filename}","ok"))
            images.append(final_image)
            torch.cuda.empty_cache()

            seed_strings.append(f"[{seed}]")
            formatted_seeds = " ".join(seed_strings)

            # Attendre le résultat HTML
            try:
                html_message_result = html_future.result(timeout=10)
            except Exception as html_err:
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_lors_generation_html', translations)}: {html_err}")
                 html_message_result = translate("erreur_lors_generation_html", translations)

            # Yield final pour cette image réussie
            yield (
                images, formatted_seeds, f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}",
                html_message_result, final_image, final_progress_html, # Utiliser final_progress_html
                bouton_charger_update_off, btn_generate_update_off,
                gr.update(interactive=False), # Save preset button
                output_gen_data_json,
                output_preview_image
            )
        # --- Fin de la boucle for des images ---

        # --- Après la boucle (gestion message final, état bouton preset) ---
        final_images = images
        final_seeds = formatted_seeds
        final_html_msg = html_message_result
        final_preview_img = None # Plus d'aperçu à la fin
        final_progress_html = "" # Effacer la barre de progression

        if not final_message: # Si aucun message d'erreur/arrêt n'a été défini
            elapsed_time = f"{(time.time() - start_time):.2f} sec"
            final_message = translate('temps_total_generation', translations) + " : " + elapsed_time
            print(txt_color("[INFO] ","info"), final_message)
            gr.Info(final_message, 3.0)

        # Activer le bouton de sauvegarde seulement si la génération n'a pas été arrêtée et qu'au moins une image a été générée
        final_save_button_state = gr.update(interactive=False)
        if not stop_event.is_set() and not stop_gen.is_set() and final_images:
             final_save_button_state = gr.update(interactive=True)

    except Exception as e:
        # --- Gestion des erreurs inattendues ---
        is_generating = False
        print(txt_color("[ERREUR] ","erreur"), f"{translate('erreur_lors_generation', translations)} : {e}")
        traceback.print_exc()
        final_message = f"{translate('erreur_lors_generation', translations)} : {str(e)}"
        final_images = []
        final_seeds = ""
        final_html_msg = f"<p style='color: red;'>{final_message}</p>"
        final_preview_img = None
        final_progress_html = ""
        final_save_button_state = gr.update(interactive=False)
        output_gen_data_json = None # Réinitialiser en cas d'erreur
        output_preview_image = None

    finally:
        # --- Bloc finally ---
        is_generating = False
        # --- AJOUT POUR FINALISER LE HTML SI ARRÊT PRÉMATURÉ ---
        if (stop_event.is_set() or stop_gen.is_set()) and final_images:
            # La génération a été arrêtée, mais certaines images ont été créées et mises en tampon.
            # Nous devons déterminer le chemin du rapport HTML pour cette exécution.
            if 'date_str' in locals() and date_str: # Vérifier si date_str est défini
                chemin_rapport_html_actuel = os.path.join(SAVE_DIR, date_str, "rapport.html")
                print(txt_color("[INFO]", "info"), f"Tentative de finalisation du rapport HTML pour {chemin_rapport_html_actuel} suite à un arrêt.")
                
                finalisation_msg = finalize_html_report_if_needed(chemin_rapport_html_actuel, translations)
                print(txt_color("[INFO]", "info"), f"Résultat finalisation HTML: {finalisation_msg}")
                
                # Mettre à jour final_html_msg si nécessaire.
                if "erreur" not in finalisation_msg.lower() and final_html_msg and isinstance(final_html_msg, str):
                    final_html_msg += f"<br/>{finalisation_msg}"
                elif "erreur" not in finalisation_msg.lower():
                    final_html_msg = finalisation_msg
            else:
                print(txt_color("[AVERTISSEMENT]", "warning"), "Impossible de déterminer le chemin du rapport HTML pour la finalisation (date_str non défini).")
        # --- FIN AJOUT ---
        yield (
            final_images, final_seeds, final_message, final_html_msg,
            final_preview_img, final_progress_html,
            bouton_charger_update_on, btn_generate_update_on,
            final_save_button_state, # Utiliser l'état calculé
            output_gen_data_json, # Passer les données finales (ou None si erreur)
            output_preview_image # Passer l'image finale (ou None si erreur)
        )


#==========================
# Fonction INPAINTED IMAGE
#==========================

def generate_inpainted_image(text, original_image_pil, image_mask_editor_dict, num_steps, strength, guidance_scale, traduire):
    """Génère une image inpainted avec Stable Diffusion XL."""
    global  stop_gen

    # --- Définir les états des boutons ---
    btn_gen_inp_off = gr.update(interactive=False)
    btn_load_inp_off = gr.update(interactive=False)
    btn_gen_inp_on = gr.update(interactive=True)
    btn_load_inp_on = gr.update(interactive=True)

    # --- Placeholders pour le premier yield ---
    initial_slider_output = [None, None]
    initial_msg_load = "" # Message chargement/HTML
    initial_msg_status = translate("preparation", translations) # Message statut
    initial_progress = "" # Barre de progression

    # --- Yield initial pour désactiver les boutons ---
    yield initial_slider_output, initial_msg_load, initial_msg_status, initial_progress, btn_load_inp_off, btn_gen_inp_off

    # --- Initialiser les variables pour le résultat final ---
    final_slider_output_result = [None, None]
    final_msg_load_result = ""
    final_msg_status_result = ""
    final_progress_result = ""

    try:
        start_time = time.time()
        stop_gen.clear()
        if model_manager.get_current_pipe() is None:
            msg = translate("erreur_pas_modele_inpainting", translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, 4.0)
            final_msg_status_result = msg
            yield (
                [None, None],  # inpainting_image_slider
                "",            # message_chargement_inpainting
                final_msg_status_result, # message_inpainting
                "",            # progress_inp_html_output
                btn_load_inp_on, # bouton_charger_inpainting
                btn_gen_inp_on   # bouton_generate_inpainting
            )
            return # Stop the generator
        pipe = model_manager.get_current_pipe() 
        if not isinstance(pipe, StableDiffusionXLInpaintPipeline):
            error_message = translate("erreur_mauvais_type_modele_inpainting", translations)
            print(txt_color("[ERREUR] ", "erreur"), error_message)
            gr.Warning(error_message, 4.0)
            # Mettre à jour les variables finales pour le bloc finally
            final_msg_status_result = error_message
            final_msg_load_result = f"<p style='color: red;'>{error_message}</p>"
            yield (
                [None, None],
                final_msg_load_result,
                final_msg_status_result,
                "",
                btn_load_inp_on,
                btn_gen_inp_on
            )
            return

        if original_image_pil is None or image_mask_editor_dict is None:
            msg = translate("erreur_image_mask_manquant", translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, 4.0)
            final_msg_status_result = msg
            yield (
                [None, None], "", final_msg_status_result, "",
                btn_load_inp_on, btn_gen_inp_on
            )
            return


        # Vérifications de type (gardées par sécurité)
        if not isinstance(original_image_pil, Image.Image):
             msg = f"Type de données de l'éditeur de masque invalide: {type(image_mask_editor_dict)}"
             print(txt_color("[ERREUR]", "erreur"), msg)
             final_msg_status_result = translate("erreur_type_image_originale_invalide", translations) # Add translation key
             yield (
                 [None, None], "", final_msg_status_result, "",
                 btn_load_inp_on, btn_gen_inp_on
             )
             return
        if not isinstance(image_mask_editor_dict, dict):
             msg = f"Type de données de l'éditeur de masque invalide: {type(image_mask_editor_dict)}"
             print(txt_color("[ERREUR]", "erreur"), msg)
             final_msg_status_result = translate("erreur_type_masque_invalide", translations) # Add translation key
             yield (
                 [None, None], "", final_msg_status_result, "",
                 btn_load_inp_on, btn_gen_inp_on
             )
             return
        
        
        pipeline_mask_pil = create_opaque_mask_from_editor(image_mask_editor_dict, original_image_pil.size, translations)


        actual_total_steps = math.ceil(num_steps * strength)
        if actual_total_steps <= 0: # Sécurité si strength est 0 ou très proche
            actual_total_steps = 1
            final_msg_status_result = translate("erreur_strength_trop_faible", translations) # Add translation key
            gr.Warning(final_msg_status_result, 3.0)
            yield (
                [original_image_pil, None], "", final_msg_status_result, "",
                btn_load_inp_on, btn_gen_inp_on
            )
            return


        # Translate the prompt if requested
        prompt_text = translate_prompt(text, translations) if traduire else text
        compel = model_manager.get_current_compel() # <-- Obtenir Compel
        conditioning, pooled = compel(prompt_text)
        
        active_adapters = pipe.get_active_adapters()
        for adapter_name in active_adapters:
            active_adapters.global_pipe.set_adapters(adapter_name, 0)
        
        image_rgb = original_image_pil.convert("RGB")
        final_image_container = {}

        # Créer la queue pour la progression
        progress_update_queue = queue.Queue()

        inpainting_callback = create_inpainting_callback(
            stop_gen,
            actual_total_steps, 
            translations,
            progress_update_queue 
        )

        # Run the inpainting pipeline
        def run_pipeline():
            print(txt_color("[INFO] ", "info"), translate("debut_inpainting", translations))
            gr.Info(translate("debut_inpainting", translations), 3.0)
            try: # Ajouter try/except
                inpainted_image_result = pipe(
                    pooled_prompt_embeds=pooled,
                    prompt_embeds=conditioning,
                    image=image_rgb, # L'image originale
                    mask_image=pipeline_mask_pil, # Le masque binaire généré
                    width=original_image_pil.width,
                    height=original_image_pil.height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    callback_on_step_end=inpainting_callback # Utilise stop_gen
                ).images[0]
                if not stop_gen.is_set():
                    final_image_container["final"] = inpainted_image_result
            except InterruptedError: # Gérer l'interruption explicitement si le callback la lève
                 print(txt_color("[INFO]", "info"), translate("inpainting_interrompu_interne", translations))
            except Exception as e:
                 # Ne pas imprimer l'erreur si c'est juste une interruption signalée par _interrupt
                 if not (hasattr(pipe, '_interrupt') and pipe._interrupt):
                     print(txt_color("[ERREUR]", "erreur"), f"Erreur dans run_pipeline (inpainting): {e}")
                     final_image_container["error"] = e


        thread = threading.Thread(target=run_pipeline)
        thread.start()

        last_progress_html = "" 

        while thread.is_alive() or not progress_update_queue.empty():
            # Lire la dernière progression de la queue (sans bloquer)
            current_step, total_steps = None, actual_total_steps
            while not progress_update_queue.empty():
                try:
                    current_step, total_steps = progress_update_queue.get_nowait()
                except queue.Empty:
                    break # Sortir si la queue est vide
            new_progress_html = last_progress_html 
            if current_step is not None:
                progress_percent = int((current_step / total_steps) * 100)
                new_progress_html = create_progress_bar_html(current_step, total_steps, progress_percent)
                # Yield la mise à jour (image=None, messages inchangés, html mis à jour)
                yield [original_image_pil, None], gr.update(), gr.update(), new_progress_html, btn_load_inp_off, btn_gen_inp_off
            time.sleep(0.05)  
        thread.join()

        current_final_progress_html = ""

        if not stop_gen.is_set() and "error" not in final_image_container:
             final_progress_html = create_progress_bar_html(actual_total_steps, actual_total_steps, 100)
        elif "error" in final_image_container:
             final_progress_html = f'<p style="color: red;">{translate("erreur_lors_inpainting", translations)}</p>'
        
        if hasattr(pipe, '_interrupt'):
            pipe._interrupt = False
        
        if "error" in final_image_container:
             error_message = f"{translate('erreur_lors_inpainting', translations)}: {final_image_container['error']}"
             print(txt_color("[ERREUR]", "erreur"), error_message)
             final_msg_status_result = translate('erreur_lors_inpainting', translations)
             final_msg_load_result = error_message # Mettre l'erreur détaillée ici
             current_final_progress_html = f'<p style="color: red;">{translate("erreur_lors_inpainting", translations)}</p>'


        elif stop_gen.is_set():
            print(txt_color("[INFO]", "info"), translate("inpainting_arrete_pas_sauvegarde", translations))
            gr.Info(translate("inpainting_arrete_pas_sauvegarde", translations), 3.0)
            final_msg_status_result = translate("inpainting_arrete", translations)
            final_msg_load_result = translate("inpainting_arrete_pas_sauvegarde", translations)
        
        else:
            inpainted_image = final_image_container.get("final", None)
        
        
            if inpainted_image is None:
                 err_msg = translate("erreur_pas_image_inpainting", translations)
                 print(txt_color("[ERREUR]", "erreur"), err_msg)
                 gr.Warning(err_msg, 4.0)
                 final_msg_status_result = translate("erreur_lors_inpainting", translations)
                 final_msg_load_result = err_msg
            
            else:
                # --- SUCCÈS ---
                final_slider_output_result = [original_image_pil, inpainted_image] # Pour ImageSlider
                current_final_progress_html = create_progress_bar_html(actual_total_steps, actual_total_steps, 100)
       
                temps_generation_image = f"{(time.time() - start_time):.2f} sec"
                date_str = datetime.now().strftime("%Y_%m_%d")
                heure_str = datetime.now().strftime("%H_%M_%S")
                save_dir = os.path.join(SAVE_DIR, date_str)
                os.makedirs(save_dir, exist_ok=True)
                filename = f"inpainting_{date_str}_{heure_str}_{original_image_pil.width}x{original_image_pil.height}.{IMAGE_FORMAT.lower()}"
                chemin_image = os.path.join(save_dir, filename)
                donnees_xmp = {
                    "Module": "SDXL Inpainting",
                    "Creator": AUTHOR,
                    "Model": os.path.splitext(model_manager.current_model_name)[0] if model_manager.current_model_name else "N/A", # <-- Utiliser ModelManager
                    "Steps": num_steps,
                    "Guidance": guidance_scale,
                    "Strength": strength,
                    "Prompt": prompt_text,
                    "Size": f"{original_image_pil.width}x{original_image_pil.height}",
                    "Generation Time": temps_generation_image
                }

                metadata_structure, prep_message = preparer_metadonnees_image(
                    inpainted_image, # L'image PIL générée
                    donnees_xmp,
                    translations, # Utiliser les traductions globales
                    chemin_image # Passer le chemin pour déterminer le format
                )

                print(txt_color("[INFO]", "info"), prep_message)

                image_future = image_executor.submit(
                    enregistrer_image,
                    inpainted_image,
                    chemin_image,
                    translations,
                    IMAGE_FORMAT, # Le format global
                    metadata_to_save=metadata_structure # Passer la structure préparée
                )
                html_future = html_executor.submit(
                    enregistrer_etiquettes_image_html,
                    chemin_image,
                    donnees_xmp, # Passer les mêmes données
                    translations,
                    is_last_image=True # Inpainting génère une seule image
                )
        
                try:
                    html_message_result = html_future.result(timeout=10)
                    final_msg_load_result = html_message_result # Message HTML pour le yield final
                except Exception as html_err:
                    print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_lors_generation_html', translations)}: {html_err}")
                    final_msg_load_result = translate("erreur_lors_generation_html", translations)

                final_msg_status_result = translate("inpainting_reussi", translations) # Message statut pour le yield final
                print(txt_color("[OK] ", "ok"), translate("fin_inpainting", translations))
                gr.Info(translate("fin_inpainting", translations), 3.0)

        # Mettre à jour la variable de progression finale pour le finally
        final_progress_result = current_final_progress_html

    except (ValueError, RuntimeError) as e: # Erreurs spécifiques (ex: traduction)
        print(txt_color("[ERREUR]", "erreur"), f"Erreur spécifique interceptée dans inpainting: {e}")
        traceback.print_exc()
        final_msg_status_result = f"Erreur: {e}"


    except Exception as e:
        err_msg = f"{translate('erreur_lors_inpainting', translations)}: {e}"
        print(txt_color("[ERREUR] ", "erreur"), err_msg)
        traceback.print_exc()
        final_msg_status_result = translate('erreur_lors_inpainting', translations)
        final_msg_load_result = str(e)

    finally:
        if 'pipe' in locals() and hasattr(pipe, '_interrupt'):
            pipe._interrupt = False
        yield final_slider_output_result, final_msg_load_result, final_msg_status_result, final_progress_result, btn_load_inp_on, btn_gen_inp_on


#==========================
# Fonctions pour arrêter la génération image
#==========================


def stop_generation():
    """Déclenche l'arrêt de la génération"""
    stop_event.set()
    gr.Info(translate("arreter", translations), 3.0)
    return translate("arreter", translations)

def stop_generation_process():
    stop_gen.set()
    gr.Info(translate("arreter", translations), 3.0)
    return translate("arreter", translations)


# =========================
# Chargement d'un modèle avant chargement interface
# =========================


initial_model_value = modeles_disponibles[0] if modeles_disponibles and modeles_disponibles[0] != translate("aucun_modele_trouve", translations) else None
initial_vae_value = None
initial_button_text = translate("charger_modele_pour_commencer", translations)
initial_button_interactive = False
initial_message_chargement = translate("aucun_modele_charge", translations)
message_retour = None 
model_selectionne = None  
vae_selctionne = None 

if DEFAULT_MODEL and os.path.basename(DEFAULT_MODEL) in modeles_disponibles:
    print(f"{txt_color('[INFO]','info')}", f"{DEFAULT_MODEL} {translate('va_se_charger', translations)} {MODELS_DIR}")

    erreur_chargement = None

    try:
        # 1. Charger dans des variables temporaires
        success, message_retour = model_manager.load_model(
            model_name=os.path.basename(DEFAULT_MODEL),
            vae_name="Défaut VAE", # Ou lire depuis config si besoin
            model_type="standard",
            gradio_mode=False # Pas d'UI Gradio ici
        )


    except Exception as e_load:
        traceback.print_exc()
        erreur_chargement = e_load



    # 2. Vérifier le succès basé sur temp_pipe ET l'absence d'erreur
    if success and not erreur_chargement:
        try:
            # 3. Affecter les globales SEULEMENT si succès
            model_selectionne = model_manager.current_model_name
            vae_selctionne = model_manager.current_vae_name

            # --- Mettre à jour les variables initiales EN CAS DE SUCCÈS ---
            initial_model_value = os.path.basename(DEFAULT_MODEL)
            initial_vae_value = "Défaut VAE"
            initial_button_text = translate("generer", translations)
            initial_button_interactive = True
            # Utiliser le message retourné par charger_modele s'il existe
            initial_message_chargement = message_retour if message_retour else translate("modele_charge_pret", translations)

            # DEBUG POST-UPDATE
            # ... (prints comme avant) ...

        except Exception as e_inner:
            # ... (gestion d'erreur interne comme avant) ...
            traceback.print_exc()
            # Réinitialiser explicitement
            initial_model_value = None
            initial_vae_value = None
            initial_button_text = translate("charger_modele_pour_commencer", translations)
            initial_button_interactive = False
            initial_message_chargement = f"Erreur interne après chargement: {e_inner}"
            model_selectionne = None
            vae_selctionne = None

    else:
        # --- Le chargement a ÉCHOUÉ (soit temp_pipe est None, soit erreur_chargement existe) ---
        # Utiliser le message d'erreur ou un message par défaut
        initial_message_chargement = message_retour if message_retour else translate("erreur_chargement_modele_defaut", translations)
        # Les autres initial_* gardent leur valeur par défaut

elif DEFAULT_MODEL:
    pass

# =========================
# Model Loading Functions (update_globals_model, update_globals_model_inpainting) for gradio
# =========================

def update_globals_model(nom_fichier, nom_vae, pag_is_enabled): # <-- AJOUT pag_is_enabled
    global  model_selectionne, vae_selctionne, loras_charges
    try:
        custom_pipeline_to_use = None
        if pag_is_enabled:
            # Vous pouvez rendre cette chaîne configurable si nécessaire.
            # Pour l'instant, nous utilisons l'exemple que vous avez fourni.
            custom_pipeline_to_use = "hyoungwoncho/sd_perturbed_attention_guidance"
            print(txt_color("[INFO]", "info"), f"PAG activé, tentative d'utilisation du custom_pipeline: {custom_pipeline_to_use}")
            gr.Info(f"PAG activé, tentative de chargement avec custom_pipeline: {custom_pipeline_to_use}", 3.0)


        # --- Utiliser ModelManager pour charger ---
        success, message = model_manager.load_model(
            model_name=nom_fichier,
            vae_name=nom_vae,
            model_type="standard",
            gradio_mode=True,
            custom_pipeline_id=custom_pipeline_to_use # <-- PASSER L'ARGUMENT
        )
    except Exception as e:
        # Gérer les erreurs inattendues pendant l'appel à load_model
        print(txt_color("[ERREUR]", "erreur"), f"Erreur inattendue lors de l'appel à model_manager.load_model: {e}")
        traceback.print_exc()
        success = False
        message = f"Erreur interne: {e}"
        # Assurer que le manager est propre en cas d'erreur grave
        model_manager.unload_model()


    if success:
        # Chargement réussi
        model_selectionne = nom_fichier
        vae_selctionne = nom_vae
        loras_charges.clear()
        etat_interactif = True
        texte_bouton = translate("generer", translations) # Texte normal

    else:
        # Chargement échoué
        etat_interactif = False
        texte_bouton = translate("charger_modele_pour_commencer", translations) # Texte désactivé
        selected_sampler_key_state.value = None

    update_interactif = gr.update(interactive=etat_interactif)
    update_texte = gr.update(value=texte_bouton)
    # Retourner le message et les deux mises à jour pour le bouton
    return message, update_interactif, update_texte

def update_globals_model_inpainting(nom_fichier):
    global model_selectionne
        # Chargement réussi
    model_selectionne = nom_fichier
    try:
        # --- Utiliser ModelManager pour charger ---
        success, message = model_manager.load_model(
            model_name=nom_fichier,
            vae_name="Auto", # Inpainting utilise souvent le VAE intégré
            model_type="inpainting",
            gradio_mode=True
        )
    except Exception as e:
        print(txt_color("[ERREUR]", "erreur"), f"Erreur inattendue lors de l'appel à model_manager.load_model (inpainting): {e}")
        traceback.print_exc()
        success = False
        message = f"Erreur interne: {e}"
        model_manager.unload_model()

    if success:
        # Chargement réussi
        model_selectionne = nom_fichier
        etat_interactif = True
        texte_bouton = translate("generer_inpainting", translations) 
    else:
        # Chargement échoué
        model_selectionne = None
        etat_interactif = False # Texte désactivé

    update_interactif = gr.update(interactive=etat_interactif)
    update_texte = gr.update(value=texte_bouton)

    # Retourner le message et les deux mises à jour pour le bouton
    return message, update_interactif, update_texte

#==========================
# Outils pour gérer les iamges du module Inpainting
#==========================


def handle_image_mask_interaction(image_mask_value_dict, current_original_bg_props):
    """
    Gère les interactions avec ImageMask (chargement d'image, dessin).
    Met à jour validated_image_state si l'image de fond change.
    Met à jour original_editor_background_props_state avec les props de l'image de fond de l'éditeur.
    """
    if image_mask_value_dict is None:
        print(txt_color("[DEBUG]", "info"), "handle_image_mask_interaction: image_mask_value_dict est None.")
        return None, None # Pour validated_image_state et original_editor_background_props_state

    current_background_pil = image_mask_value_dict.get("background") # Attendu comme PIL.Image ou None
    if current_background_pil is None:
        # Cas où l'utilisateur a effacé l'image dans ImageMask
        print(txt_color("[DEBUG]", "info"), "handle_image_mask_interaction: Le fond de ImageMask est None (effacé?).")
        if current_original_bg_props is not None: 
            return None, None # Réinitialiser validated_image et props
        else: # Pas d'image avant, toujours pas d'image
            return gr.update(), gr.update() 

    # Vérifier si c'est une image PIL si non None
    if current_background_pil is not None and not isinstance(current_background_pil, Image.Image):
        print(txt_color("[AVERTISSEMENT]", "warning"), f"handle_image_mask_interaction: Le fond de ImageMask n'est pas une PIL.Image. Type: {type(current_background_pil)}. Retour sans mise à jour.")
        return gr.update(), gr.update()  


    # Si current_background_pil est None à ce stade (après la première vérification), on retourne None
    if current_background_pil is None:
        return None, None

    new_bg_props = (current_background_pil.width, current_background_pil.height, current_background_pil.mode)

    if current_original_bg_props is None or new_bg_props != current_original_bg_props:
        # L'image de fond a changé (nouvelle image ou première image)
        print(txt_color("[INFO]", "info"), "Nouvelle image de fond détectée dans ImageMask, validation en cours.")
        image_checker = ImageSDXLchecker(current_background_pil, translations)
        processed_background_for_pipeline = image_checker.redimensionner_image()

        if isinstance(processed_background_for_pipeline, Image.Image):
            # Mettre à jour validated_image_state ET original_editor_background_props_state
            return processed_background_for_pipeline, new_bg_props
        else:
            print(txt_color("[AVERTISSEMENT]", "warning"), "L'image de fond de ImageMask n'est pas valide après traitement par ImageSDXLchecker.")
            # Réinitialiser les deux états si l'image n'est pas valide
            return None, None
    else:
        return gr.update(), gr.update()
  

def create_opaque_mask_from_editor(editor_dict, target_size, translations):
    """
    Crée un masque binaire opaque (PIL, mode 'L') à partir des layers dessinés
    dans ImageEditor. Les zones dessinées (non transparentes dans les layers)
    deviennent blanches (255), le reste noir (0).
    """
    if editor_dict is None or not isinstance(editor_dict, dict):
        print(txt_color("[AVERTISSEMENT]", "warning"), translate("erreur_donnees_editeur_mask_invalides", translations))
        return Image.new('L', target_size, 0) # Masque noir par défaut

    layers_pil_from_editor = editor_dict.get("layers", []) # Attendu comme liste de PIL.Image
    
    if not layers_pil_from_editor: # Pas de layers dessinés
        return Image.new('L', target_size, 0) # Masque noir

    # Créer un masque composite RGBA transparent de la taille cible
    composite_rgba_mask = Image.new('RGBA', target_size, (0, 0, 0, 0))

    for layer_pil in layers_pil_from_editor:
        if layer_pil is None:
            continue
        try:
            # S'assurer que le layer est en mode RGBA pour l'alpha_composite
            if not isinstance(layer_pil, Image.Image):
                print(txt_color("[AVERTISSEMENT]", "warning"), f"Un élément dans 'layers' n'est pas une image PIL: {type(layer_pil)}")
                continue
            layer_to_composite = layer_pil.convert('RGBA')
            # Redimensionner le layer si nécessaire pour correspondre à target_size
            # Ceci suppose que le layer original a les dimensions du "background" de l'éditeur
            # Si l'image de fond de l'éditeur est différente de target_size, il faut ajuster.
            # Pour l'instant, on assume que le dessin est fait sur une image déjà à target_size
            # ou que le redimensionnement est géré en amont si le fond de l'éditeur change.
            # Idéalement, target_size est la taille de l'image de fond de l'éditeur.
            if layer_to_composite.size != target_size:
                # Ceci peut arriver si l'image de fond de l'éditeur a été redimensionnée
                # avant que le masque ne soit généré pour le pipeline.
                # On redimensionne le layer pour qu'il corresponde à l'image du pipeline.
                print(txt_color("[INFO]", "info"), f"Redimensionnement du layer de masque de {layer_to_composite.size} à {target_size}")
                layer_to_composite = layer_to_composite.resize(target_size, Image.Resampling.NEAREST)

            # Alpha composite le layer sur notre masque composite
            composite_rgba_mask.alpha_composite(layer_to_composite)
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_traitement_layer_mask', translations)}: {e}")
            continue

    # Convertir le masque RGBA composite en un masque binaire 'L'
    # Les pixels non totalement transparents dans le composite deviennent blancs (255)
    alpha_channel = composite_rgba_mask.split()[-1] # Obtenir le canal Alpha
    binary_mask_pil = alpha_channel.point(lambda p: 255 if p > 0 else 0, mode='L')
    
    return binary_mask_pil
#################################################
#FONCTIONS pour les presets :
#################################################
def handle_preset_rename_click(preset_id, current_trigger): # Ajouter current_trigger
    """Met à jour l'état d'édition ET déclenche un refresh via le trigger."""
    print(f"[Action Preset {preset_id}] Clic Renommer. Trigger actuel: {current_trigger}")
    # Retourner l'update pour l'état d'édition ET l'update pour le trigger
    return gr.update(value=preset_id), gr.update(value=current_trigger + 1)

def handle_preset_cancel_click(current_trigger): # Ajouter current_trigger
    """Annule l'édition en cours ET déclenche un refresh."""
    print("[Action Preset] Clic Annuler Édition.")
    # Mettre l'ID à None ET incrémenter le trigger
    return gr.update(value=None), gr.update(value=current_trigger + 1)

def handle_preset_rename_submit(preset_id, new_name, current_trigger):
    """Soumet le nouveau nom pour le preset."""
    print(f"[Action Preset {preset_id}] Submit Renommer vers '{new_name}'.")
    if not preset_id or not new_name:
        gr.Warning(translate("erreur_nouveau_nom_vide", translations))
        return gr.update(value=preset_id), gr.update() # Garder l'édition, ne pas rafraîchir

    success, message = preset_manager.rename_preset(preset_id, new_name)
    if success:
        gr.Info(message)
        # Succès: Annuler l'édition et déclencher refresh
        return gr.update(value=None), gr.update(value=current_trigger + 1)
    else:
        gr.Warning(message)
        # Échec: Garder l'édition, ne pas rafraîchir
        return gr.update(value=preset_id), gr.update()

def handle_preset_delete_click(preset_id, current_trigger, page, search, sort, filter_models, filter_samplers, filter_loras, current_search_value):
    """Supprime le preset, déclenche un refresh ET met à jour la pagination."""
     # Initialiser les updates en cas d'erreur (3 éléments maintenant)
    trigger_update_on_error = gr.update()
    pagination_update_on_error = gr.update()
    search_update_on_error = gr.update()

    success, message = preset_manager.delete_preset(preset_id)
    if success:
        gr.Info(message)
        pagination_update = update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras)
        # --- LE HACK ---
        current_search_str = current_search_value if current_search_value else ""
        if current_search_str.endswith(" "):
            temp_search_value = current_search_str[:-1] # Enlever l'espace
        else:
            temp_search_value = current_search_str + " " # Ajouter un espace
        
        search_update_hack = gr.update(value=temp_search_value)
        # --- FIN HACK ---
        # Retourner l'update du trigger ET l'update de la pagination
        return gr.update(value=current_trigger + 1), pagination_update,search_update_hack
    else:
        gr.Warning(message)
        # Retourner des updates vides si erreur
        return trigger_update_on_error, pagination_update_on_error, search_update_on_error

def handle_preset_rating_change(preset_id, new_rating_value):
    """Met à jour la note du preset."""

    if preset_id is not None and new_rating_value is not None:
        success, message = preset_manager.update_preset_rating(preset_id, int(new_rating_value))
        if not success:
            gr.Warning(message)
    # Pas de retour nécessaire, la mise à jour DB suffit (sauf si tri par note actif)

def update_pagination_and_trigger_refresh(page, search, sort, filter_models, filter_samplers, filter_loras, current_trigger):
    """Met à jour l'UI de pagination ET incrémente le trigger de refresh."""
    # Calculer les updates pour la pagination
    pagination_updates = update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras)
    # Préparer l'update pour le trigger
    trigger_update = gr.update(value=current_trigger + 1)
    # Retourner les updates de pagination + l'update du trigger
    return list(pagination_updates) + [trigger_update]

# --- Handlers pour la pagination ---
def handle_page_change(direction, current_page):
    """Calcule la nouvelle page."""
    new_page = current_page + direction
    # Retourne SEULEMENT l'update pour l'état de la page
    return gr.update(value=new_page)

# --- Fonction pour mettre à jour les filtres après sauvegarde ---
def update_filter_choices_after_save():
    models, samplers, loras = get_filter_options()
    return gr.update(choices=models), gr.update(choices=samplers), gr.update(choices=loras)


# --- Remettre la fonction de sauvegarde (adaptée pour retourner les updates de filtre) ---
def handle_save_preset(preset_name, preset_notes, current_gen_data_json, preview_image_pil, current_trigger, current_search_value):
    """
    Gère l'appel à preset_manager pour sauvegarder le preset.
    Appelée par l'événement .click() de bouton_save_current_preset.
    Retourne les updates pour vider les champs et mettre à jour les filtres.
    """
    # --- AJOUT PRINT POUR DEBUG ---
    # --- FIN AJOUT ---
    # --- Initialiser les updates pour les filtres (au cas où on échoue avant) ---
    filter_updates_on_error = [gr.update(), gr.update(), gr.update()]
    trigger_update_on_error = gr.update()
    search_update_on_error = gr.update()
    # --- FIN INITIALISATION ---

    if not preset_name:
        gr.Warning(translate("erreur_nom_preset_vide", translations), 3.0)
        # Retourner 5 updates vides
        return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error

    current_gen_data = None
    if isinstance(current_gen_data_json, str):
        try:
            current_gen_data = json.loads(current_gen_data_json)
        except json.JSONDecodeError as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_decodage_json_preset', translations)}: {e}")
            gr.Warning(translate("erreur_interne_decodage_json", translations), 3.0)
            # Retourner 5 updates vides
            return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error
    else:
         # Si ce n'est pas une chaîne, c'est probablement None ou déjà un dict
         if isinstance(current_gen_data_json, dict):
             current_gen_data = current_gen_data_json
         else:
             print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_type_inattendu_json_preset', translations)}: {type(current_gen_data_json)}")
             gr.Warning(translate("erreur_interne_donnees_generation_invalides", translations), 3.0)
             # Retourner 5 updates vides
             return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error

    # Vérifier si l'image est une instance PIL valide
    if not isinstance(preview_image_pil, Image.Image):
         # Essayer de charger depuis BytesIO si c'est des bytes (peut arriver depuis gr.State)
         if isinstance(preview_image_pil, bytes):
             try:
                 preview_image_pil = Image.open(BytesIO(preview_image_pil))
                 print(txt_color("[INFO]", "info"), translate("info_image_preview_chargee_bytes", translations))
             except Exception as img_err:
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_chargement_image_preview_bytes', translations)}: {img_err}")
                 preview_image_pil = None # Marquer comme invalide
         else:
             preview_image_pil = None # Marquer comme invalide si ce n'est ni Image ni bytes

    if not isinstance(current_gen_data, dict) or preview_image_pil is None:
         print(txt_color("[ERREUR]", "erreur"), translate("erreur_donnees_generation_ou_image_manquantes", translations))
         print(f"  Type data après JSON: {type(current_gen_data)}")
         print(f"  Type image après vérif: {type(preview_image_pil)}")
         gr.Warning(translate("erreur_pas_donnees_generation", translations), 3.0)
         # Retourner 5 updates vides
         return gr.update(), gr.update(), *filter_updates_on_error,  trigger_update_on_error, search_update_on_error

    data_to_save = current_gen_data.copy()
    data_to_save['notes'] = preset_notes

    # --- APPEL À LA SAUVEGARDE ---
    # C'est ici que 'success' et 'message' sont définis
    try:
        success, message = preset_manager.save_gen_image_preset(preset_name, data_to_save, preview_image_pil)
    except Exception as save_err:
        print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_appel_sauvegarde_preset', translations)}: {save_err}")
        traceback.print_exc()
        success = False
        message = translate("erreur_interne_sauvegarde_preset", translations)# Ajouter clé de traduction
    # --- FIN APPEL ---

    if success: # Maintenant 'success' est défini
        gr.Info(message, 3.0)
        # --- Succès: Récupérer les updates réelles pour les filtres ---
        try:
            update_model, update_sampler, update_lora = update_filter_choices_after_save()
            current_search_str = current_search_value if current_search_value else ""
            if current_search_str.endswith(" "):
                temp_search_value = current_search_str[:-1] # Enlever l'espace
            else:
                temp_search_value = current_search_str + " " # Ajouter un espace
            search_update_hack = gr.update(value=temp_search_value)
            return gr.update(value=""), gr.update(value=""), update_model, update_sampler, update_lora, gr.update(value=current_trigger + 1), search_update_hack
        except Exception as filter_err:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_update_filtres_preset', translations)}: {filter_err}")
            return gr.update(value=""), gr.update(value=""), *filter_updates_on_error, gr.update(value=current_trigger + 1), search_update_hack
    else:
        gr.Warning(message, 4.0)
        return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error

def handle_preset_load_click(preset_id):
    """
    Charge les données d'un preset et met à jour les contrôles de l'UI de génération.
    Appelée par le bouton 'Charger' d'un preset.
    """
    print(f"[Action Preset {preset_id}] Clic Charger.")
    preset_data = preset_manager.load_preset_data(preset_id)

    if preset_data is None:
        msg = translate("erreur_chargement_preset_introuvable", translations).format(preset_id)
        gr.Warning(msg)
        # Retourner des updates vides pour tous les outputs attendus
        num_lora_slots = 4 # Assurez-vous que cela correspond au nombre de slots LoRA
        # 7 contrôles de base + 3 par LoRA + 1 message = 7 + 3*4 + 1 = 20 outputs
        return [gr.update()] * (7 + 3 * num_lora_slots + 1)

    try:
        original_prompt_value = preset_data.get('original_prompt', preset_data.get('prompt', ''))
        if original_prompt_value is None:
            original_prompt_value = preset_data.get('prompt', '')
        # Le prompt négatif n'est pas directement dans l'UI, on le charge pas ici
        styles_data = preset_data.get('styles', [])
        loaded_style_names = []
        if isinstance(styles_data, str):
            try:
                loaded_style_names = json.loads(styles_data)
                if not isinstance(loaded_style_names, list):
                    loaded_style_names = []
            except json.JSONDecodeError:
                loaded_style_names = [] 
        elif isinstance(styles_data, list):
            loaded_style_names = styles_data
        else:
            # Gérer les autres types inattendus
            loaded_style_names = []
        model_name = preset_data.get('model', None)
        vae_name_from_preset = preset_data.get('vae', "Défaut VAE")
        guidance = preset_data.get('guidance_scale', 7.0)
        steps = preset_data.get('num_steps', 30)
        sampler_key = preset_data.get('sampler_key', 'sampler_euler')
        sampler_display = translate(sampler_key, translations) # Traduire la clé pour l'affichage
        seed = preset_data.get('seed', -1)
        width = preset_data.get('width', 1024)
        height = preset_data.get('height', 1024)
        loras_data = preset_data.get('loras', []) # Récupérer la donnée (peut être str ou list)
        loaded_loras = []

        # --- Chargement des paramètres PAG ---
        custom_pipeline_id_preset = preset_data.get('custom_pipeline_id')
        pag_enabled_preset = bool(custom_pipeline_id_preset) # PAG est activé si un custom_pipeline_id est présent
        pag_scale_preset = preset_data.get('pag_scale', 1.5)
        pag_applied_layers_preset = preset_data.get('pag_applied_layers', "m0") # Valeur par défaut si non trouvée
        # --- Fin chargement PAG ---

         # --- Validation Modèle ---
        available_models = model_manager.list_models(model_type="standard", gradio_mode=True) # <-- Utiliser ModelManager
        model_update = gr.update()
        if model_name and model_name in available_models:
            model_update = gr.update(value=model_name)
        elif model_name:
            print(f"[WARN Preset Load] Modèle '{model_name}' du preset non trouvé dans les options actuelles.")

         # --- Validation VAE ---
        available_vae_files  = lister_fichiers(VAE_DIR, translations, gradio_mode=True)
        vae_value_for_ui = "Auto"

        if vae_name_from_preset == "Défaut VAE":
            vae_value_for_ui = "Défaut VAE"
        elif vae_name_from_preset in available_vae_files:
            vae_value_for_ui = vae_name_from_preset
        else:
            gr.Warning(translate("erreur_vae_preset_introuvable", translations).format(vae_name_from_preset))
            vae_value_for_ui = "Défaut VAE"

        vae_update = gr.update()

         # --- Validation LORAS---
        if isinstance(loras_data, str):
            # Si c'est une chaîne, essayer de la décoder
            try:
                loaded_loras = json.loads(loras_data)
                # Assurer que le résultat est bien une liste après décodage
                if not isinstance(loaded_loras, list):
                    loaded_loras = [] # Réinitialiser si le type est incorrect
            except json.JSONDecodeError:
                loaded_loras = [] # Réinitialiser en cas d'erreur de décodage
        elif isinstance(loras_data, list):
            # Si c'est déjà une liste, l'utiliser directement
            loaded_loras = loras_data
        else:
            # Gérer les autres types inattendus
            loaded_loras = []
        # --- Préparer la chaîne de format exacte pour le dropdown ---
        format_string = f"{width}*{height}"
        orientation_key = None
        # Retrouver l'orientation à partir de la config originale
        for fmt in config["FORMATS"]:
            dims = fmt.get("dimensions", "")
            if dims == f"{width}*{height}":
                orientation_key = fmt.get("orientation")
                break
        if orientation_key:
            # Utiliser la clé pour obtenir la traduction actuelle
            format_string += f" {translate(orientation_key, translations)}"
        else:
            # Fallback si non trouvé (ne devrait pas arriver)
            format_string = f"{width}*{height} {translate('orientation_inconnue', translations)}" # Ajoutez cette clé si besoin

        # --- Vérifier si le format existe dans les choix actuels ---
        if format_string not in FORMATS:
             format_string = FORMATS[0] # Ou une autre valeur par défaut

        # --- Préparer les updates pour les LoRAs ---
        num_lora_slots = 4
        lora_check_updates = [gr.update(value=False) for _ in range(num_lora_slots)]
        lora_dd_updates = [gr.update(value=None, interactive=False, choices=[translate("aucun_lora_disponible", translations)]) for _ in range(num_lora_slots)]
        lora_scale_updates = [gr.update(value=0) for _ in range(num_lora_slots)]

        # Obtenir la liste actuelle des LoRAs disponibles
        available_loras = model_manager.list_loras(gradio_mode=True)
        has_available_loras = bool(available_loras) and translate("aucun_modele_trouve", translations) not in available_loras and translate("repertoire_not_found", translations) not in available_loras
        lora_choices = available_loras if has_available_loras else [translate("aucun_lora_disponible", translations)]

        for i, lora_info in enumerate(loaded_loras):
            if i >= num_lora_slots: break # Ne charger que le nombre de slots disponibles
            lora_name = lora_info.get('name')
            lora_weight = lora_info.get('weight')
            if lora_name and lora_weight is not None:
                # Vérifier si le LoRA chargé est dans la liste actuelle
                if lora_name in lora_choices:
                    lora_check_updates[i] = gr.update(value=True)
                    # Mettre à jour les choix, la valeur ET rendre interactif
                    lora_dd_updates[i] = gr.update(choices=lora_choices, value=lora_name, interactive=True)
                    lora_scale_updates[i] = gr.update(value=lora_weight)
                else:
                    warn_msg = f"LoRA '{lora_name}' du preset non trouvé. Ignoré pour le slot {i+1}."
                    print(txt_color("[WARN]", "warning"), warn_msg)
                    # Laisser le slot désactivé

        # --- Préparer l'update du Sampler et appliquer au backend ---
        sampler_update_msg, success = apply_sampler_to_pipe(model_manager.get_current_pipe(), sampler_key, translations)
        # Vérifier si le sampler chargé est valide
        if success:
            global global_selected_sampler_key # Mettre à jour la globale
            global_selected_sampler_key = sampler_key
            sampler_update = gr.update(value=sampler_display) # Mettre à jour le dropdown UI
            gr.Info(sampler_update_msg, 2.0) # Afficher le succès
        else:
            # Gérer l'échec (ex: sampler du preset non valide/compatible)
            gr.Warning(sampler_update_msg)
            # Appliquer un sampler par défaut et mettre à jour l'UI
            default_sampler_key = "sampler_euler"
            default_sampler_display = translate(default_sampler_key, translations)
            apply_sampler_to_pipe(model_manager.get_current_pipe(), default_sampler_key, translations)  # Appliquer le défaut
            global_selected_sampler_key = default_sampler_key # Mettre à jour la globale avec le défaut
            sampler_update = gr.update(value=default_sampler_display) 
             # Le sampler a déjà été appliqué par apply_sampler() ci-dessus

        # --- Préparer l'update des Styles ---
        # Filtrer les styles chargés pour ne garder que ceux présents dans les options actuelles
        valid_style_choices = [style["name"] for style in STYLES if style["name"] != translate("Aucun_style", translations)]
        final_style_selection = [s_name for s_name in loaded_style_names if s_name in valid_style_choices]
        if len(final_style_selection) != len(loaded_style_names):
            print("[WARN] Certains styles du preset ne sont plus disponibles et ont été ignorés.")


        # --- Construire la liste des outputs dans l'ordre exact attendu par .click() ---
        outputs_list = [
            model_update, 
            vae_update, 
            gr.update(value=original_prompt_value),                 # text_input
            gr.update(value=final_style_selection),  # style_dropdown
            gr.update(value=guidance),               # guidance_slider
            gr.update(value=steps),                  # num_steps_slider
            gr.update(value=format_string),          # format_dropdown
            sampler_update,                          # sampler_dropdown
            gr.update(value=seed),                   # seed_input
            *lora_check_updates,                     # lora_checks (4)
            *lora_dd_updates,                        # lora_dropdowns (4)
            *lora_scale_updates,                     # lora_scales (4)
            gr.update(value=pag_enabled_preset),     # pag_enabled_checkbox
            gr.update(value=pag_scale_preset),       # pag_scale_slider
            gr.update(value=pag_applied_layers_preset), # pag_applied_layers_input
            gr.update(value=translate("preset_charge_succes", translations).format(preset_data.get('name', f'ID: {preset_id}')))# message_chargement
        ]
        print(f"[INFO] Preset '{preset_data.get('name', f'ID: {preset_id}')}' chargé dans l'UI.")
        gr.Info(translate("preset_charge_succes", translations).format(preset_data.get('name', f'ID: {preset_id}')), 2.0)
        return outputs_list
    except Exception as e:
        print(txt_color("[ERREUR]", "erreur"), f"Erreur lors du chargement du preset {preset_id}: {e}")
        traceback.print_exc()
        gr.Warning(translate("erreur_generale_chargement_preset", translations))
        # Retourner des updates vides pour tous les outputs attendus (MAJ du nombre)
        num_lora_slots = 4 # Nombre de slots LoRA
        return [gr.update()] * (2 + 7 + 3 * num_lora_slots + 1 + 3) # +3 pour PAG (checkbox, slider, textbox)
        
def reset_page_state_only():
    """Retourne simplement une mise à jour pour mettre l'état de la page à 1."""
    print("[Reset Page State] Mise à jour de current_preset_page à 1.")
    return gr.update(value=1)

def handle_page_dropdown_change(page_selection):
    """Gère le changement du dropdown de page."""
    print(f"!!! Dropdown Page Change Détecté: Nouvelle page = {page_selection} !!!")
    # Retourner SEULEMENT la valeur sélectionnée pour mettre à jour l'état
    return page_selection


############################################################
############################################################
#####################USER INTERFACE#########################
############################################################
############################################################

block_kwargs = {"theme": gradio_change_theme(GRADIO_THEME)}
if js_code:
    block_kwargs["js"] = js_code


with gr.Blocks(**block_kwargs) as interface:
    # --- États ---
    # --- États pour les Presets ---
    # État pour le trigger de rafraîchissement manuel/après action
    preset_refresh_trigger = gr.State(0)
    # État pour la page actuelle
    current_preset_page_state = gr.State(1)
    # --- Fin États Presets ---
    last_successful_generation_data = gr.State(value=None) # Pour stocker les données JSON
    last_successful_preview_image = gr.State(value=None) 

    # --- Fin États -

    gr.Markdown(f"# Cyberbill SDXL images generator version {version()}")


############################################################
########***************************************************
########TAB GENRATION IMAGE
########***************************************************
############################################################
  
    with gr.Tab(translate("generation_image", translations)):
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                text_input = gr.Textbox(label=translate("prompt", translations), info=translate("entrez_votre_texte_ici", translations), elem_id="promt_input")
                traduire_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", translations), value=False, info=translate("traduire_en_anglais", translations))
                style_dropdown = gr.Dropdown(
                    choices=[style["name"] for style in STYLES if style["name"] != translate("Aucun_style", translations)], # Exclure "Aucun style" des choix multiples
                    value=[], # Valeur par défaut : liste vide
                    label=translate("styles", translations),
                    info=translate("Selectionnez_un_ou_plusieurs_styles", translations), 
                    multiselect=True,
                    max_choices=4
                )
                use_image_checkbox = gr.Checkbox(label=translate("generer_prompt_image", translations), value=False)
                time_output = gr.Textbox(label=translate("temps_rendu", translations), interactive=False)
                html_output = gr.Textbox(label=translate("mise_a_jour_html", translations), interactive=False)
                message_chargement = gr.Textbox(label=translate("statut", translations), value=initial_message_chargement)     
                image_input = gr.Image(label=translate("telechargez_image", translations), type="pil", visible=False)


            with gr.Column(scale=1, min_width=200):
                image_output = gr.Gallery(label=translate("images_generees", translations))
                progress_html_output = gr.HTML(value="")
                guidance_slider = gr.Slider(1, 20, value=7, label=translate("guidage", translations))
                num_steps_slider = gr.Slider(1, 50, value=30, label=translate("etapes", translations), step=1)
                format_dropdown = gr.Dropdown(choices=FORMATS, value=FORMATS[3], label=translate("format", translations))
                with gr.Accordion(translate("pag_options_label", translations), open=False): # Nouvelle clé
                    pag_enabled_checkbox = gr.Checkbox(
                        label=translate("enable_pag_label", translations), # Nouvelle clé
                        value=False
                    )
                    pag_scale_slider = gr.Slider(
                        minimum=0.0, maximum=10.0, value=1.5, step=0.1,
                        label=translate("pag_scale_label", translations), # Nouvelle clé
                        interactive=True,
                        visible=False # Initialement caché
                    )
                    pag_applied_layers_input = gr.Textbox(
                        label=translate("pag_applied_layers_label", translations), # Nouvelle clé
                        info=translate("pag_applied_layers_info", translations), # Nouvelle clé
                        value="m0", # Valeur par défaut comme dans l'exemple
                        visible=False # Initialement caché
                    )

                seed_input = gr.Number(label=translate("seed", translations), value=-1, elem_id="seed_input_main_gen") # Ajout elem_id
                num_images_slider = gr.Slider(1, 200, value=1, label=translate("nombre_images_generer", translations), step=1)                
                
                                    
            with gr.Column(scale=1, min_width=200):
                with gr.Row():
                    with gr.Column():
                        preview_image_output = gr.Image(height=170, label=translate("apercu_etapes", translations),interactive=False)
                        seed_output = gr.Textbox(label=translate("seed_utilise", translations))
                        value = DEFAULT_MODEL if DEFAULT_MODEL else None
                        modele_dropdown = gr.Dropdown(label=translate("selectionner_modele", translations), choices=modeles_disponibles, value=initial_model_value, allow_custom_value=True)
                        vae_dropdown = gr.Dropdown(label=translate("selectionner_vae", translations), choices=vaes, value=initial_vae_value, allow_custom_value=True)
                        sampler_display_choices = get_sampler_choices(translations) # Obtenir les choix depuis l'utilitaire
                        default_sampler_display = translate(global_selected_sampler_key, translations) # Traduire la clé par défaut

                        sampler_dropdown = gr.Dropdown(
                            label=translate("selectionner_sampler", translations),
                            choices=sampler_display_choices,
                            value=default_sampler_display,
                            allow_custom_value=True,
                        )                        
                        bouton_charger = gr.Button(translate("charger_modele", translations))
                                                

                        
            with gr.Column():
                texte_bouton_gen_initial = translate("charger_modele_pour_commencer", translations) # Utiliser la même clé ou une clé spécifique
                btn_generate = gr.Button(value=initial_button_text, interactive=initial_button_interactive, variant="primary")
                btn_stop = gr.Button(translate("arreter", translations), variant="stop")
                btn_stop_after_gen = gr.Button(translate("stop_apres_gen", translations), variant="stop")
                bouton_lister = gr.Button(translate("lister_modeles", translations))
                with gr.Accordion(translate("batch_runner_accordion_title", translations), open=False): # Nouvelle clé de traduction
                    gr.Markdown(f"#### {translate('batch_runner_title', translations)}") # Titre interne
                    batch_json_file_input = gr.File(
                        label=translate("upload_batch_json", translations),
                        file_types=[".json"],
                        file_count="single"
                    )
                    batch_run_button = gr.Button(translate("run_batch_button", translations), variant="primary")
                    batch_status_output = gr.Textbox(label=translate("batch_status", translations), interactive=False)
                    batch_progress_output = gr.HTML()
                    batch_gallery_output = gr.Gallery(label=translate("batch_generated_images", translations), height="auto", interactive=False) # Ajustez height si besoin
                    batch_stop_button = gr.Button(translate("stop_batch_button", translations), variant="stop", interactive=False)                
                with gr.Accordion(translate("sauvegarder_preset_section", translations), open=False): # Nouvelle clé de traduction pour le titre
                    preset_name_input = gr.Textbox(
                        label=translate("nom_preset", translations),
                        placeholder=translate("entrez_nom_preset", translations)
                    )
                    preset_notes_input = gr.Textbox(
                        label=translate("notes_preset", translations),
                        placeholder=translate("entrez_notes_preset", translations),
                        lines=3
                    )
                    # Le bouton est maintenant DANS l'accordéon et déclenche la sauvegarde
                    bouton_save_current_preset = gr.Button(
                        translate("confirmer_sauvegarde_preset", translations), # Nouveau texte pour le bouton
                        variant="primary", # Le rendre plus visible
                        interactive=False # Toujours inactif par défaut
                    )

                with gr.Accordion(label="Lora", open=False) as lora_section:
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(translate("lora_explications", translations))
                            lora_dropdowns = []
                            lora_checks = []
                            lora_scales = []
                            for i in range(1,5):
                                with gr.Group():
                                    lora_check = gr.Checkbox(label=f"Lora {i}", value=False)
                                    lora_dropdown = gr.Dropdown(choices=["Aucun LORA disponible"], label=translate("selectionner_lora", translations), interactive=False)
                                    lora_scale_slider = gr.Slider(0, 1, value=0, label=translate("poids_lora", translations))
                                    lora_checks.append(lora_check)
                                    lora_dropdowns.append(lora_dropdown)
                                    lora_scales.append(lora_scale_slider)

                                    # Add the change event here
                                    lora_checks[i-1].change( # Utiliser l'index correct
                                        fn=lambda check, current_choices: gr.update(interactive=check and bool(current_choices) and current_choices[0] != translate("aucun_lora_disponible", translations)),
                                        inputs=[lora_checks[i-1], lora_dropdowns[i-1]], # Ajouter les choix actuels comme input
                                        outputs=[lora_dropdown]
                                    )
                    lora_message = gr.Textbox(label=translate("message_lora", translations), value="")
                    
                with gr.Accordion(translate("gestion_modules", translations), open=False): # Ou gr.Tab
                    gr.Markdown(translate("activer_desactiver_modules", translations)) # Ajouter clé de traduction
                    with gr.Column():
                        # Récupérer les détails des modules chargés
                        loaded_modules_details = gestionnaire.get_module_details()

                        if not loaded_modules_details:
                            gr.Markdown(f"*{translate('aucun_module_charge_pour_gestion', translations)}*") # Ajouter clé
                        else:
                            for module_detail in loaded_modules_details:
                                module_name = module_detail["name"]
                                display_name = module_detail["display_name"]
                                is_active = module_detail["is_active"]

                                # Créer une checkbox pour chaque module
                                module_checkbox = gr.Checkbox(
                                    label=display_name,
                                    value=is_active,
                                    elem_id=f"module_toggle_{module_name}" # ID unique si besoin
                                )

                                # Lier l'événement .change()
                                module_checkbox.change(
                                    fn=functools.partial(handle_module_toggle, module_name, gestionnaire_instance=gestionnaire, preset_manager_instance=preset_manager),
                                    inputs=[module_checkbox], # L'input est la nouvelle valeur de la checkbox
                                    outputs=[] # Pas d'output direct, la fonction met à jour l'état interne
                                )
                def mettre_a_jour_listes():
                    modeles = lister_fichiers(MODELS_DIR, translations, gradio_mode=True)
                    vaes = ["Défaut VAE"] + lister_fichiers(VAE_DIR, translations, gradio_mode=True)
                    loras = model_manager.list_loras(gradio_mode=True)

                    has_loras = bool(loras) and loras[0] != translate("aucun_modele_trouve", translations) and loras[0] != translate("repertoire_not_found", translations)
                    lora_choices = loras if has_loras else ["Aucun LORA disponible"]
                    lora_updates = [gr.update(choices=lora_choices, interactive=has_loras, value=None) for _ in range(4)] # we set the value to None
                    return (
                        gr.update(choices=modeles),
                        gr.update(choices=vaes),
                        *lora_updates,
                        gr.update(value=translate("lora_trouve", translations) + ", ".join(loras) if has_loras else translate("aucun_lora_disponible", translations))
                    )


############################################################
########TAB PRESETS
############################################################


    with gr.Tab(translate("Preset", translations)) as preset_tab:
        def get_filter_options():
            filter_data = preset_manager.get_distinct_preset_filters()
            models = filter_data.get('models', [])
            sampler_keys_in_presets = filter_data.get('samplers', [])
            loras = filter_data.get('loras', []) # Peut nécessiter un traitement spécial si stocké en JSON
            sampler_display_names = [translate(s_key, translations) for s_key in sampler_keys_in_presets]
            sampler_display_names = list(set(sampler_display_names))
            sampler_display_names.sort()
            # --- FIN CORRECTION ---

            return models, sampler_display_names, loras

        initial_models, initial_samplers, initial_loras = get_filter_options()
        # --- Contrôles (Statiques) ---
        with gr.Row():
            preset_search_input = gr.Textbox(
                label=translate("rechercher_preset", translations),
                placeholder="..."
                # scale=2 # Ajuster l'échelle si besoin
            )
            preset_sort_dropdown = gr.Dropdown(
                # Vous pouvez réorganiser cette liste comme vous le souhaitez
                choices=["Date Création", "Nom A-Z", "Nom Z-A", "Date Utilisation", "Note"],
                value="Date Création", # Ceci détermine la sélection par défaut
                label=translate("trier_par", translations)
            )

        with gr.Row():
            preset_filter_model = gr.Dropdown(
                label=translate("filtrer_par_modele", translations),
                choices=initial_models,
                value=[], # Permettre sélection multiple
                multiselect=True,
                elem_id="preset_filter_model"
            )
            preset_filter_sampler = gr.Dropdown(
                label=translate("filtrer_par_sampler", translations),
                choices=initial_samplers,
                value=[],
                multiselect=True,
                elem_id="preset_filter_sampler"
            )
            preset_filter_lora = gr.Dropdown(
                label=translate("filtrer_par_lora", translations),
                choices=initial_loras,
                value=[],
                multiselect=True,
                elem_id="preset_filter_lora"
            )
            preset_page_dropdown = gr.Dropdown(
                label=translate("page", translations), # Ajouter clé de traduction
                choices=[1], # Initialiser avec la page 1
                value=1,
                interactive=False, # Sera activé si plusieurs pages
                elem_id="preset_page_dropdown"
            )
        # --- Fin Contrôles ---

        @gr.render(inputs=[
            current_preset_page_state,
            preset_search_input,
            preset_sort_dropdown,
            preset_filter_model,
            preset_filter_sampler,
            preset_filter_lora,
            preset_refresh_trigger
        ])
        def render_presets_with_decorator(page, search, sort, filter_models, filter_samplers, filter_loras, trigger_val):
            """
            Décorée par @gr.render. Récupère les presets et crée l'UI dynamiquement.
            Les liaisons d'événements sont faites ici.
            Retourne les composants UI et les updates pour la pagination.
            """

            sampler_keys_to_filter = []
            if filter_samplers: # Si des samplers sont sélectionnés
                # Créer un mapping inverse: {nom_affiche_traduit: cle_interne}
                reverse_sampler_map = {v: k for k, v in translations.items() if k.startswith("sampler_")}
                for sampler_display_name in filter_samplers:
                    internal_key = reverse_sampler_map.get(sampler_display_name)
                    if internal_key:
                        sampler_keys_to_filter.append(internal_key)
                    else:
                        print(txt_color("[ERREUR ]", "erreur"), translate("gestion_samplers_erreur", translations).format(sampler_display_name))
                        # Gérer le cas où la traduction inverse échoue (ne devrait pas arriver)
            # 1. Récupérer les données
            all_presets_data = preset_manager.load_presets_for_display(
                preset_type='gen_image', search_term=search, sort_by=sort,
                selected_models=filter_models or None,
                selected_samplers=sampler_keys_to_filter or None, # <-- Utiliser les clés internes ici
                selected_loras=filter_loras or None
            )

            # 2. Calculer la pagination
            total_presets = len(all_presets_data)
            total_pages = math.ceil(total_presets / PRESETS_PER_PAGE) if total_presets > 0 else 1
            current_page = max(1, min(page, total_pages))
            start_index = (current_page - 1) * PRESETS_PER_PAGE
            end_index = start_index + PRESETS_PER_PAGE
            presets_for_page = all_presets_data[start_index:end_index]

            # --- Fonction utilitaire safe_get (identique à avant) ---
            def safe_get_from_row(row, key, default=None):
                try:
                    return row[key] if key in row.keys() else default
                except (IndexError, TypeError): return default

            gen_ui_outputs = [
                modele_dropdown,
                vae_dropdown,
                text_input,
                style_dropdown,
                guidance_slider,
                num_steps_slider,
                format_dropdown,
                sampler_dropdown,
                seed_input,
                # Ajouter tous les composants LoRA individuellement
                *lora_checks,    # Dépaqueter la liste des 4 checkboxes
                *lora_dropdowns, # Dépaqueter la liste des 4 dropdowns
                *lora_scales,    # Dépaqueter la liste des 4 sliders
                message_chargement # Le message de statut de l'onglet génération
            ]

            delete_inputs = [
                preset_refresh_trigger,
                current_preset_page_state, # Page actuelle
                preset_search_input,       # Recherche actuelle
                preset_sort_dropdown,      # Tri actuel
                preset_filter_model,       # Filtres actuels...
                preset_filter_sampler,
                preset_filter_lora,
                preset_search_input        # <--- AJOUT : Recherche actuelle pour le hack
            ]
        # --- Définir les outputs pour la suppression (incluant la recherche) ---
            delete_outputs = [
                preset_refresh_trigger,    # Output pour le trigger
                pagination_dd_output,      # Output pour le dropdown de pagination
                preset_search_input        # <--- AJOUT : Output pour la recherche (hack)
            ]
        # --- Fin définition ---

            if not presets_for_page:
                gr.Markdown(f"*{translate('aucun_preset_trouve', translations)}*", key="no_presets_found_md") # Clé pour le cas vide
            else:
                num_rows_for_page = math.ceil(len(presets_for_page) / PRESET_COLS_PER_ROW)
                preset_idx_on_page = 0
                for r in range(num_rows_for_page):
                    # --- PAS DE key= sur gr.Row ---
                    with gr.Row(equal_height=False):
                        for c in range(PRESET_COLS_PER_ROW):
                            if preset_idx_on_page < len(presets_for_page):
                                preset_data = presets_for_page[preset_idx_on_page]
                                preset_id = safe_get_from_row(preset_data, "id", f"ERREUR_ID_{preset_idx_on_page}")
                                preset_name = safe_get_from_row(preset_data, "name", "ERREUR_NOM")

                                # --- PAS DE key= sur gr.Column ---
                                with gr.Column(scale=0, min_width=200):
                                    # --- Composants avec key= ---
                                    image_bytes = safe_get_from_row(preset_data, "preview_image")
                                    preview_img = None
                                    if image_bytes:
                                        try: preview_img = Image.open(BytesIO(image_bytes))
                                        except Exception: pass
                                    gr.Image(value=preview_img, height=128, width=128, show_label=True, interactive=False, show_download_button=False, key=f"preset_img_{preset_id}")

                                    gr.Textbox(value=preset_name, show_label=False, interactive=False, key=f"preset_name_display_{preset_id}")

                                    preset_notes = safe_get_from_row(preset_data, 'notes')
                                    if preset_notes:
                                        # --- PAS DE key= sur gr.Accordion ---
                                        with gr.Accordion(translate("voir_notes", translations), open=False):
                                            # --- MAIS key= sur gr.Markdown ---
                                            gr.Markdown(preset_notes, key=f"preset_notes_md_{preset_id}")

                                    rating_value = safe_get_from_row(preset_data, "rating", 0)
                                    rating_comp = gr.Radio(
                                        choices=[str(r) for r in range(1, 6)], value=str(rating_value) if rating_value > 0 else None,
                                        label=translate("evaluation", translations), interactive=True, key=f"preset_rating_{preset_id}"
                                    )

                                    try:
                                        model_name = safe_get_from_row(preset_data, 'model', '?')
                                        sampler_key_name = safe_get_from_row(preset_data, 'sampler_key', '?')
                                        sampler_display_name = translate(sampler_key_name, translations) if sampler_key_name != '?' else '?'

                                        details_md = f"- **Modèle:** {model_name}\n- **Sampler:** {sampler_key_name}"
                                        # --- PAS DE key= sur gr.Accordion ---
                                        with gr.Accordion(translate("details_techniques", translations), open=False):
                                            # --- MAIS key= sur gr.Markdown ---
                                            gr.Markdown(details_md, key=f"preset_details_md_{preset_id}")
                                    except Exception: pass

                                    # --- Boutons avec key= ---
                                    load_btn = gr.Button(translate("charger", translations) + " 💾", size="sm", key=f"preset_load_{preset_id}")
                                    delete_btn = gr.Button(translate("supprimer", translations) + " 🗑️", variant="stop", size="sm", key=f"preset_delete_{preset_id}")

                                    # --- Liaison des événements (directement ici) ---
                                    if isinstance(preset_id, int): # Vérifier que l'ID est valide avant de lier
                                        load_btn.click(
                                            fn=partial(handle_preset_load_click, preset_id), # Utilise partial pour passer l'ID
                                            inputs=[], # Pas d'inputs directs depuis l'UI ici
                                            outputs=gen_ui_outputs # La liste des composants à mettre à jour
                                        )
                                        delete_btn.click(
                                            fn=partial(handle_preset_delete_click, preset_id), # Utilise partial pour l'ID
                                            inputs=delete_inputs, # Utiliser la liste d'inputs définie plus haut
                                            outputs=delete_outputs # Utiliser la liste d'outputs définie plus haut
                                        )                                   
                                        rating_comp.change(fn=partial(handle_preset_rating_change, preset_id), inputs=[rating_comp], outputs=[])
                                        # load_btn.click(...) # À ajouter plus tard

                                preset_idx_on_page += 1


############################################################
########TAB INPAINTING
############################################################
        def mettre_a_jour_listes_inpainting():
            modeles = lister_fichiers(INPAINT_MODELS_DIR, translations)
            return gr.update(choices=modeles)

    with gr.Tab(translate("Inpainting", translations)):
        validated_image_state = gr.State(value=None)
        original_editor_background_props_state = gr.State(value=None)
        with gr.Row():
            with gr.Column():
                image_mask_input = gr.ImageMask(
                    label=translate("image_avec_mask", translations),
                    brush=gr.Brush(colors=["#FF0000"], color_mode="fixed"),
                    type="pil", # <--- CHANGEMENT ICI
                    sources=["upload", "clipboard"], # Permettre l'upload direct
                    interactive=True
                )
                inpainting_prompt = gr.Textbox(label=translate("prompt_inpainting", translations))
                traduire_inpainting_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", translations), value=False, info=translate("traduire_en_anglais", translations))
                guidance_inpainting_slider = gr.Slider(1, 20, value=7, label=translate("guidage", translations))
                num_steps_inpainting_slider = gr.Slider(1, 50, value=30, label=translate("etapes", translations), step=1)
                strength_inpainting_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.89, step=0.01, label=translate("force_inpainting", translations))
            with gr.Column():               
                modele_inpainting_dropdown = gr.Dropdown(label=translate("selectionner_modele_inpainting", translations), choices=modeles_impaint, value=value, allow_custom_value=True)
                bouton_lister_inpainting = gr.Button(translate("lister_modeles_inpainting", translations))
                bouton_charger_inpainting = gr.Button(translate("charger_modele_inpainting", translations))
                message_chargement_inpainting = gr.Textbox(label=translate("statut_inpainting", translations), value=translate("aucun_modele_charge_inpainting", translations))
                message_inpainting = gr.Textbox(label=translate("message_inpainting", translations), interactive=False)                
                
                texte_bouton_inpaint_initial = translate("charger_modele_pour_commencer", translations)
                bouton_generate_inpainting = gr.Button(value=texte_bouton_inpaint_initial, interactive=False, variant="primary")
                bouton_stop_inpainting = gr.Button(translate("arreter_inpainting", translations), variant="stop")


            with gr.Column():
                inpainting_image_slider = gr.ImageSlider(label=translate("comparaison_inpainting", translations), interactive=False) # Nouvelle clé
                progress_inp_html_output = gr.HTML(value="")


############################################################
########TAB MODULES
############################################################
    gestionnaire.creer_tous_les_onglets(translations)






    use_image_checkbox.change(fn=lambda use_image: gr.update(visible=use_image), inputs=use_image_checkbox, outputs=image_input)
    
    bouton_lister.click(
        fn=mettre_a_jour_listes,
        outputs=[modele_dropdown, vae_dropdown, *lora_dropdowns, lora_message]
    )
                
    bouton_charger.click(
        fn=update_globals_model,
        inputs=[modele_dropdown, vae_dropdown, pag_enabled_checkbox], # <-- AJOUT pag_enabled_checkbox
        outputs=[message_chargement, btn_generate, btn_generate]
    )

    image_input.change(
        fn=generate_prompt_wrapper,
        inputs=[image_input, gr.State(translations)], 
        outputs=text_input)

    btn_generate.click(
        generate_image,
        inputs=[
            text_input,
            style_dropdown,
            guidance_slider,
            num_steps_slider,
            format_dropdown,
            traduire_checkbox,
            seed_input,
            num_images_slider,
            pag_enabled_checkbox, # <-- AJOUT PAG
            pag_scale_slider,     # <-- AJOUT PAG
            pag_applied_layers_input, # <-- AJOUT PAG
            *lora_checks,
            *lora_dropdowns,
            *lora_scales
        ],
        outputs=[
            image_output,
            seed_output,
            time_output,
            html_output,
            preview_image_output,
            progress_html_output,
            bouton_charger,
            btn_generate,
            bouton_save_current_preset,
            last_successful_generation_data,
            last_successful_preview_image
        ]
    )
    btn_stop.click(stop_generation_process, outputs=time_output)
    btn_stop_after_gen.click(stop_generation, outputs=time_output)
    

    bouton_save_current_preset.click(
        fn=handle_save_preset,
        inputs=[
            preset_name_input,
            preset_notes_input,
            last_successful_generation_data, # État contenant le JSON
            last_successful_preview_image,
            preset_refresh_trigger,
            preset_search_input    # État contenant l'image PIL
        ],
        outputs=[
            preset_name_input, # Pour vider si succès
            preset_notes_input, # Pour vider si succès
            preset_filter_model, # Pour mettre à jour les choix
            preset_filter_sampler, # Pour mettre à jour les choix
            preset_filter_lora,
            preset_refresh_trigger,
            preset_search_input     # Pour mettre à jour les choix
        ]
    )
    # --- Connexions pour le Batch Runner ---
    batch_runner_inputs = [
        batch_json_file_input,      # Correspond à json_file_obj
        gr.State(config),           # Correspond à config <-- CORRECTED
        gr.State(translations),     # Correspond à translations
        gr.State(device),           # Correspond à device
        # Composants UI (dans l'ordre attendu par run_batch_from_json)
        batch_status_output,        # Correspond à ui_status_output
        batch_progress_output,      # Correspond à ui_progress_output
        batch_gallery_output,       # Correspond à ui_gallery_output
        batch_run_button,           # Correspond à ui_run_button
        batch_stop_button           # Correspond à ui_stop_button
    ]

    # Liste des outputs (correspond aux composants UI à mettre à jour)
    batch_runner_outputs = [
        batch_status_output,
        batch_progress_output,
        batch_gallery_output,
        batch_run_button,
        batch_stop_button
    ]

    def batch_runner_wrapper(*args, progress=gr.Progress(track_tqdm=True)):
        """Fonction wrapper qui appelle run_batch_from_json avec yield from."""
        # gestionnaire et stop_event sont accessibles depuis la portée englobante
        yield from run_batch_from_json(
            model_manager,
            stop_event,
            *args,
            progress=progress
        )

    batch_run_button.click(
        # Utiliser la fonction wrapper nommée au lieu de la lambda
        fn=batch_runner_wrapper, # <--- MODIFIÉ
        inputs=batch_runner_inputs,
        outputs=batch_runner_outputs
    )

    # La connexion du bouton Stop du batch reste inchangée
    batch_stop_button.click(
        fn=stop_generation, # La fonction globale d'arrêt existante
        inputs=None,
        outputs=None # L'arrêt est géré par l'événement stop_event
    )
    # 

    module_checkbox.change(
            fn=functools.partial(
            handle_module_toggle, module_name,
            gestionnaire_instance=gestionnaire,
            preset_manager_instance=preset_manager 
        ),
        inputs=[module_checkbox],
         outputs=[]
    )

    def handle_sampler_change(selected_display_name):
        """Gère le changement de sampler dans l'UI principale."""
        global global_selected_sampler_key # Accéder à la variable globale

        sampler_key = get_sampler_key_from_display_name(selected_display_name, translations)
        if sampler_key and model_manager.get_current_pipe() is not None:
            message, success = apply_sampler_to_pipe(model_manager.get_current_pipe(), sampler_key, translations)
            if success:
                global_selected_sampler_key = sampler_key # Mettre à jour la clé globale si succès
                gr.Info(message, 3.0) # Afficher l'info Gradio ici
            else:
                gr.Warning(message, 4.0) # Afficher l'avertissement Gradio ici
            return message # Retourner le message pour le Textbox de statut
        else:
            if model_manager.get_current_pipe() is None:
                error_msg = translate("erreur_pas_modele_pour_sampler", translations)
                gr.Warning(error_msg, 4.0)
        error_msg = f"{translate('erreur_sampler_inconnu', translations)}: {selected_display_name}"
        gr.Warning(error_msg, 4.0)
        return error_msg

    sampler_dropdown.change(
        fn=handle_sampler_change,
        inputs=sampler_dropdown,
        outputs=[message_chargement] # Mettre à jour le Textbox de statut
    )

    # Logique pour afficher/masquer le slider pag_scale
    def toggle_pag_scale_visibility_main(pag_enabled):
        # Met à jour la visibilité des deux composants PAG
        return gr.update(visible=pag_enabled), gr.update(visible=pag_enabled)

    pag_enabled_checkbox.change(
        fn=toggle_pag_scale_visibility_main,
        inputs=[pag_enabled_checkbox],
        outputs=[pag_scale_slider, pag_applied_layers_input] # Mettre à jour les deux
    )
# Liaisons pour l'onglet Inpainting
    # L'événement .change de image_mask_input gère maintenant le chargement d'image et le dessin.
    # Il met à jour validated_image_state et mask_image_output.


    image_mask_input.change(
        fn=handle_image_mask_interaction, # Appel direct
        inputs=[image_mask_input, original_editor_background_props_state], # Ajouter le nouvel état en input
        outputs=[validated_image_state, original_editor_background_props_state] # Mettre à jour les deux états
    )
 
    bouton_stop_inpainting.click(
        fn=stop_generation_process,
        outputs=message_inpainting
    )

    bouton_lister_inpainting.click(
        fn=mettre_a_jour_listes_inpainting,
        outputs=modele_inpainting_dropdown
    )

    bouton_charger_inpainting.click(
        fn=update_globals_model_inpainting,
        inputs=[modele_inpainting_dropdown],
        outputs=[message_chargement_inpainting, bouton_generate_inpainting, bouton_generate_inpainting]
    )
    
    bouton_generate_inpainting.click(
        fn=generate_inpainted_image,
        inputs=[
            inpainting_prompt,
            validated_image_state,  
            image_mask_input, 
            num_steps_inpainting_slider,
            strength_inpainting_slider,
            guidance_inpainting_slider,
            traduire_inpainting_checkbox
        ],
        outputs=[
            inpainting_image_slider, 
            message_chargement_inpainting,
            message_inpainting,
            progress_inp_html_output,
            bouton_charger_inpainting,
            bouton_generate_inpainting
        ]
    )

with interface: # Re-ouvrir le contexte pour ajouter les liaisons

    # --- Inputs communs pour le rendu de page ---
    render_page_inputs = [
        current_preset_page_state,
        preset_search_input,
        preset_sort_dropdown,
        preset_filter_model,
        preset_filter_sampler,
        preset_filter_lora,
        preset_refresh_trigger
    ]
    pagination_dd_inputs = [
        current_preset_page_state, # La page actuelle est nécessaire
        preset_search_input, preset_sort_dropdown, preset_filter_model,
        preset_filter_sampler, preset_filter_lora
    ]

    pagination_dd_output = preset_page_dropdown 
 
    def update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras):
        """Met à jour le Dropdown de pagination."""
        # Recalculer le nombre total de pages
        all_presets_data = preset_manager.load_presets_for_display(
            preset_type='gen_image', search_term=search, sort_by=sort,
            selected_models=filter_models or None, selected_samplers=filter_samplers or None, selected_loras=filter_loras or None
        )
        total_presets = len(all_presets_data)
        total_pages = math.ceil(total_presets / PRESETS_PER_PAGE) if total_presets > 0 else 1
        current_page = max(1, min(page, total_pages)) # Assurer que la page est valide


        # Créer la liste des choix pour le dropdown
        page_choices = list(range(1, total_pages + 1))

        # Retourner l'update pour le dropdown
        return gr.update(
            choices=page_choices,
            value=current_page,
            interactive=(total_pages > 1) # Activer seulement si plus d'une page
        )


    def reset_page_and_update_pagination(*args):
        # Le premier arg est la page actuelle, on l'ignore pour le calcul de pagination
        pagination_updates = update_pagination_display(1, *args[1:]) # Calculer pagination pour page 1
        # Retourner l'update pour la page (déclenche @gr.render) + updates pagination
        return [gr.update(value=1)] + list(pagination_updates)
    def reset_page_and_update_pagination_dd(*args):
        # Le premier arg est la page actuelle (ignoré), les suivants sont les filtres
        pagination_dd_update = update_pagination_display(1, *args[1:]) # Calculer dropdown pour page 1
        # Retourner update pour l'état page + update pour le dropdown
        return gr.update(value=1), pagination_dd_update

    # 1. Mettre l'état de la page à 1

    # 2. Appeler explicitement le rendu APRÈS la mise à jour de l'état


    preset_search_input.input(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs, # Prend page actuelle + filtres
        outputs=[current_preset_page_state, pagination_dd_output] # Met à jour état page ET dropdown
    )


    preset_sort_dropdown.change(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, pagination_dd_output]
    )
    preset_filter_model.change(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, pagination_dd_output]
    )
    preset_filter_sampler.change(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, pagination_dd_output]
    )
    preset_filter_lora.change(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, pagination_dd_output]
    )

    # Initialisation de mask_image_output au démarrage avec un placeholder
    # (si handle_image_mask_interaction n'est pas appelée au démarrage par les sliders)
    def init_inpainting_outputs():
        # Laisser image_mask_input s'initialiser avec son état par défaut Gradio
        initial_editor_val_for_load = gr.update() 
        # Pour validated_image_state (PIL Image ou None)
        initial_validated_img = None
        initial_original_bg_props = None
        # Pour inpainting_image_slider (liste de 2 images ou [None, None])
        initial_slider_val = [None, None]
        return initial_editor_val_for_load, initial_validated_img, initial_original_bg_props, initial_slider_val


    def refresh_trigger_and_update_pagination_dd(current_page, trigger, *filter_args):
        pagination_dd_update = update_pagination_display(current_page, *filter_args)
        initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val = init_inpainting_outputs()

        return gr.update(value=trigger + 1), pagination_dd_update, initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val

 
    preset_page_dropdown.change(
        fn=handle_page_dropdown_change, # Utiliser la fonction nommée
        inputs=[preset_page_dropdown],
        outputs=[current_preset_page_state] # Cible l'état
    )

    def initial_load_update_pagination_dd(*filter_args):
        # Pour les presets
        pagination_dd_update = update_pagination_display(1, *filter_args) # Page 1 initiale
        # Pour l'inpainting (initialiser validated_image_state et mask_image_output)
        initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val = init_inpainting_outputs()
        return (
            gr.update(value=1), pagination_dd_update, 
            initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val
        )

    interface.load(
        fn=initial_load_update_pagination_dd,
        inputs=pagination_dd_inputs[1:], # Juste les filtres initiaux
        outputs=[current_preset_page_state, pagination_dd_output, 
                 image_mask_input, validated_image_state, original_editor_background_props_state, 
                 inpainting_image_slider] # Ajouter le nouvel état aux outputs
    )


interface.launch(inbrowser=str_to_bool(OPEN_BROWSER), pwa=True, share=str_to_bool(SHARE))