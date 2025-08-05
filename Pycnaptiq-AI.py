import sys
from pathlib import Path

# --- AJOUT: Ajouter le répertoire des modules au chemin système ---
# Cela permet d'importer des bibliothèques locales comme hi_diffusers
project_root_dir = Path(__file__).resolve().parent
modules_dir_abs = project_root_dir / "modules"
if str(modules_dir_abs) not in sys.path:
    sys.path.insert(0, str(modules_dir_abs))
# --- FIN AJOUT ---

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
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from core.version import version, version_date 
from Utils.callback_diffuser import latents_to_rgb, create_callback_on_step_end, create_inpainting_callback 
from Utils.model_manager import ModelManager
from core.translator import translate_prompt
from core.Inpaint import apply_mask_effects
from core.image_prompter import init_image_prompter, generate_prompt_from_image, MODEL_ID_FLORENCE2 as DEFAULT_FLORENCE2_MODEL_ID_FROM_PROMPTER, DEFAULT_FLORENCE2_TASK
from Utils import llm_prompter_util
from Utils.utils import GestionModule, enregistrer_etiquettes_image_html, finalize_html_report_if_needed, charger_configuration, gradio_change_theme, lister_fichiers, styles_fusion, create_progress_bar_html, load_modules_js, \
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
from functools import partial # Keep functools import
from Utils.gest_mem import create_memory_accordion_ui, update_memory_stats, empty_working_set
from core.batch_runner import run_batch_from_json
from core.pipeline_executor import execute_pipeline_task_async
import gc




# Load the configuration first
config = charger_configuration()
# Initialisation de la langue
DEFAULT_LANGUAGE = config.get("LANGUAGE", "fr")  # Utilisez 'fr' comme langue par défaut si 'LANGUAGE' n'est pas défini.
translations = load_locales(DEFAULT_LANGUAGE)
print(txt_color("[INFO]", "info"), f"{translate('app_version_label', translations)} {txt_color(version(),'info')}")

# --- Affichage de la date de version localisée ---
date_objet = version_date()

# Noms des mois pour le formatage manuel
mois_fr = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
mois_en = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

jour = date_objet.day
index_mois = date_objet.month - 1 # Les listes sont indexées à partir de 0
annee = date_objet.year
date_formatee_str = ""

if DEFAULT_LANGUAGE == "fr":
    if 0 <= index_mois < len(mois_fr):
        nom_mois = mois_fr[index_mois]
        date_formatee_str = f"{jour} {nom_mois} {annee}"
elif DEFAULT_LANGUAGE == "en":
    if 0 <= index_mois < len(mois_en):
        nom_mois = mois_en[index_mois]
        date_formatee_str = f"{nom_mois} {jour}, {annee}"

if not date_formatee_str: # Fallback si la langue n'est pas gérée ou si l'index du mois est incorrect
    date_formatee_str = date_objet.isoformat()

print(txt_color("[INFO]", "info"), f"{translate('version_date_label', translations)}: {txt_color(date_formatee_str, 'info')}")
# --- Fin affichage date de version ---

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
# ---  Initialisation du LLM Prompter ---
LLM_PROMPTER_MODEL_PATH = config.get("LLM_PROMPTER_MODEL_PATH", "Qwen/Qwen3-0.6B")
# --- Initialisation du Image Prompter (Florence-2) ---
FLORENCE2_MODEL_ID_CONFIG = config.get("FLORENCE2_MODEL_ID", DEFAULT_FLORENCE2_MODEL_ID_FROM_PROMPTER)
init_image_prompter(device, translations, model_id=FLORENCE2_MODEL_ID_CONFIG, load_now=False)# --- Utiliser ModelManager pour lister les fichiers ---
modeles_disponibles = model_manager.list_models(model_type="standard")
vaes = model_manager.list_vaes() # Inclut "Auto"


# --- Lister les LORAS au démarrage ---
initial_lora_choices = model_manager.list_loras(gradio_mode=True)
has_initial_loras = bool(initial_lora_choices) and translate("aucun_modele_trouve", translations) not in initial_lora_choices and translate("repertoire_not_found", translations) not in initial_lora_choices
lora_initial_dropdown_choices = initial_lora_choices if has_initial_loras else [translate("aucun_lora_disponible", translations)]
initial_lora_message = translate("lora_trouve", translations) + ", ".join(initial_lora_choices) if has_initial_loras else translate("aucun_lora_disponible", translations)
modeles_impaint = model_manager.list_models(model_type="inpainting")

if not modeles_impaint or modeles_impaint[0] == translate("aucun_modele_trouve", translations):
    modeles_impaint = [translate("aucun_modele_trouve", translations)]
    # --- AJOUT : Proposer le téléchargement du modèle d'inpainting par défaut ---
    reponse_inpaint = input(
        f"{txt_color(translate('attention', translations), 'erreur')} "
        f"{translate('aucun_modele_inpainting_trouve', translations)} " # Nouvelle clé de traduction
        f"{translate('telecharger_modele_question', translations)} "
        f"(wangqyqq/sd_xl_base_1.0_inpainting_0.1.safetensors) ? (o/n): "
    )
    if reponse_inpaint.lower() in ["o", "oui", "y", "yes"]:
        lien_modele_inpaint = "https://civitai.com/api/download/models/916706?type=Model&format=SafeTensor&size=full&fp=fp16"
        nom_fichier_inpaint = "sd_xl_base_1.0_inpainting_0.1.safetensors"
        if telechargement_modele(lien_modele_inpaint, nom_fichier_inpaint, INPAINT_MODELS_DIR, translations):
            modeles_impaint = model_manager.list_models(model_type="inpainting") # Recharger la liste
            if not modeles_impaint or modeles_impaint[0] == translate("aucun_modele_trouve", translations):
                modeles_impaint = [translate("aucun_modele_trouve", translations)] # S'assurer que la liste est correcte même après échec de re-listage
    
gestionnaire = GestionModule(
    translations=translations, config=config,
    language=DEFAULT_LANGUAGE,  # <-- AJOUT EXPLICITE DE LA LANGUE
    model_manager_instance=model_manager,
    preset_manager_instance=preset_manager
)
gestionnaire.language = DEFAULT_LANGUAGE # S'assurer que la langue est bien passée


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
    # DEFAULT_FLORENCE2_TASK est importé depuis core.image_prompter
    # FLORENCE2_MODEL_ID_CONFIG est défini globalement
    return generate_prompt_from_image(
        image,
        current_translations,
        task=DEFAULT_FLORENCE2_TASK, # Utiliser la tâche par défaut définie dans image_prompter
        unload_after=True, # Important pour cyberbill_SDXL.py
        model_id=FLORENCE2_MODEL_ID_CONFIG # Utiliser l'ID configuré ou le défaut du prompter
    )

#==========================
# Fonction GENERATION IMAGE
#==========================

def generate_image(text, style_selection, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images, pag_enabled, pag_scale, pag_applied_layers_str, 
                   # Les états pour l'amélioration du prompt sont passés ici
                   original_user_prompt_for_cycle, # Le prompt original avant toute amélioration du cycle actuel
                   prompt_is_currently_enhanced,   # Booléen: le 'text' actuel est-il le résultat d'une amélioration IA ?
                   enhancement_cycle_is_active,    # Booléen: un cycle Améliorer/Refaire/Valider est-il en cours ?
                   *lora_inputs):
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

    # 'text' est le prompt actuellement dans la textbox (peut être original ou amélioré)
    # 'original_user_prompt_for_cycle' est le prompt utilisateur avant le début du cycle d'amélioration actuel
    # 'prompt_is_currently_enhanced' indique si 'text' est le résultat d'une amélioration IA
    
    prompt_to_use_for_sdxl = text # Par défaut, on utilise le texte de la textbox
    prompt_to_log_as_original = text # Par défaut, le texte de la textbox est l'original

    if enhancement_cycle_is_active or prompt_is_currently_enhanced:
        # Si un cycle est actif OU si le prompt est marqué comme amélioré (même si cycle terminé par validation),
        # alors le prompt original pour les logs/métadonnées est celui stocké au début du cycle.
        prompt_to_log_as_original = original_user_prompt_for_cycle
    # 'prompt_to_use_for_sdxl' (qui est 'text') est déjà le prompt potentiellement amélioré.

    if traduire:
        translated_version = translate_prompt(prompt_to_use_for_sdxl, translations)
        if translated_version != prompt_to_use_for_sdxl: # Afficher message seulement si traduction a eu lieu
            gr.Info(translate("prompt_traduit_pour_generation", translations), 2.0)
        prompt_to_use_for_sdxl = translated_version

    yield (
        # text_input n'est plus un output direct ici car il est géré par la logique d'amélioration
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
            is_generating = False
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
            is_generating = False
            return # Stop the generator


        #initialisation du chrono
        start_time = time.time()
        # Réinitialiser l'état d'arrêt
        stop_event.clear()
        stop_gen.clear()

        seeds = [random.randint(1, 10**19 - 1) for _ in range(num_images)] if seed_input == -1 else [seed_input] * num_images

        prompt_en, negative_prompt_str, selected_style_display_names = styles_fusion(
            style_selection,
            prompt_to_use_for_sdxl, # Utiliser le prompt traité (traduit si besoin)
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
        image_paths = [] # Changed from `images` to store paths
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
            # print(txt_color("[INFO]", "info"), f"Message LoRA: {message_lora}") # This line seems to be a duplicate log for LoRA, should be PAG

        # principal loop for genrated images
        for idx, seed in enumerate(seeds):
            depart_time = time.time()
            PREVIEW_QUEUE.clear() # Vider la queue d'aperçu pour cette image
            html_message_result = translate("generation_en_cours", translations)

            if stop_event.is_set(): # Vérifie l'arrêt global avant de commencer
                print(txt_color("[INFO] ","info"), translate("arrete_demande_apres", translations), f"{idx} {translate('images', translations)}.")
                gr.Info(translate("arrete_demande_apres", translations) + f"{idx} {translate('images', translations)}.", 3.0)
                final_message = translate("generation_arretee", translations)
                break # Sortir de la boucle for

            print(txt_color("[INFO] ","info"), f"{translate('generation_image', translations)} {idx+1} {translate('seed_utilise', translations)} {seed}")
            gr.Info(translate('generation_image', translations) + f"{idx+1} {translate('seed_utilise', translations)} {seed}", 3.0)

            progress_update_queue = queue.Queue() 
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
                stop_event=stop_gen, 
                translations=translations,
                progress_queue=progress_update_queue,
                preview_queue=PREVIEW_QUEUE, 
                pag_enabled=pag_enabled,
                pag_scale=pag_scale,
                pag_applied_layers=pag_applied_layers 
            )

            last_preview_index = 0
            last_progress_html = ""
            last_yielded_preview = None
            while pipeline_thread.is_alive() or last_preview_index < len(PREVIEW_QUEUE) or not progress_update_queue.empty():
                current_step, total_steps = None, num_steps
                while not progress_update_queue.empty():
                    try:
                        current_step, total_steps = progress_update_queue.get_nowait()
                    except queue.Empty: break
                new_progress_html = last_progress_html
                if current_step is not None:
                    progress_percent = int((current_step / total_steps) * 100)
                    new_progress_html = create_progress_bar_html(current_step, total_steps, progress_percent)

                preview_img_to_yield = None
                preview_yielded_in_loop = False
                while last_preview_index < len(PREVIEW_QUEUE):
                    preview_img_to_yield = PREVIEW_QUEUE[last_preview_index]
                    last_preview_index += 1
                    last_yielded_preview = preview_img_to_yield
                    yield (
                        image_paths, formatted_seeds, f"{idx+1}/{num_images}...", html_message_result,
                        preview_img_to_yield, new_progress_html, 
                        bouton_charger_update_off, btn_generate_update_off,
                        gr.update(interactive=False), 
                        output_gen_data_json,
                        output_preview_image
                    )
                    preview_yielded_in_loop = True
                
                if not preview_yielded_in_loop and new_progress_html != last_progress_html:
                     yield (
                         image_paths, formatted_seeds, f"{idx+1}/{num_images}...", html_message_result,
                         last_yielded_preview, 
                         new_progress_html, 
                         bouton_charger_update_off, btn_generate_update_off,
                         gr.update(interactive=False), 
                         output_gen_data_json,
                         output_preview_image
                     )
                last_progress_html = new_progress_html 
                time.sleep(0.05)
            pipeline_thread.join() 
            PREVIEW_QUEUE.clear() 

            final_status = result_container.get("status")
            final_image = result_container.get("final")
            error_details = result_container.get("error")

            final_progress_html = ""
            if final_status == "success":
                final_progress_html = create_progress_bar_html(num_steps, num_steps, 100)
            elif final_status == "error":
                final_progress_html = f'<p style="color: red;">{translate("erreur_lors_generation", translations)}</p>'

            if stop_event.is_set() or stop_gen.is_set() or final_status == "stopped":
                print(txt_color("[INFO]", "info"), translate("generation_arretee_pas_sauvegarde", translations))
                gr.Info(translate("generation_arretee_pas_sauvegarde", translations), 3.0)
                final_message = translate("generation_arretee", translations)
                yield (
                    image_paths, " ".join(seed_strings), translate("generation_arretee", translations),
                    translate("generation_arretee_pas_sauvegarde", translations), None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, 
                    gr.update(interactive=False), 
                    None, 
                    None  
                )
                break 
            elif final_status == "error":
                error_msg = str(error_details) if error_details else "Unknown pipeline error"
                print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_pipeline', translations)}: {error_msg}")
                gr.Warning(f"{translate('erreur_pipeline', translations)}: {error_msg}", 4.0)
                yield (
                    image_paths, " ".join(seed_strings), translate("erreur_pipeline", translations),
                    error_msg, None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, 
                    gr.update(interactive=False), 
                    None, 
                    None  
                )
                continue 
            elif final_image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image_genere", translations))
                gr.Warning(translate("erreur_pas_image_genere", translations), 4.0)
                yield (
                    image_paths, " ".join(seed_strings), translate("erreur_pas_image_genere", translations),
                    translate("erreur_pas_image_genere", translations), None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, 
                    gr.update(interactive=False), 
                    None, 
                    None  
                )
                continue 
            current_data_for_preset = {
                 "model": model_manager.current_model_name,  
                 "vae": model_manager.current_vae_name,  
                 "original_user_prompt": prompt_to_log_as_original, # Le prompt avant toute amélioration du cycle
                 "prompt": prompt_en, # Le prompt final utilisé pour SDXL (peut être amélioré et/ou traduit)
                 "current_prompt_is_enhanced": prompt_is_currently_enhanced, # Etat d'amélioration
                 "enhancement_cycle_active": enhancement_cycle_is_active, # Etat du cycle
                 "negative_prompt": negative_prompt_str,
                 "styles": json.dumps(selected_style_display_names if selected_style_display_names else []),
                 "guidance_scale": guidance_scale,
                 "num_steps": num_steps,
                 "sampler_key": global_selected_sampler_key,
                 "seed": seed,
                 "width": width,
                 "height": height,
                 "loras": json.dumps([{"name": name, "weight": weight} for name, weight in model_manager.loaded_loras.items()]),
                 "pag_enabled": pag_enabled,
                 "pag_scale": pag_scale,
                 "custom_pipeline_id": "hyoungwoncho/sd_perturbed_attention_guidance" if pag_enabled else None,
                 "pag_applied_layers": pag_applied_layers_str,
                 "rating": 0,
                 "notes": ""
            }
            preview_image_for_preset = final_image.copy()
            output_gen_data_json = json.dumps(current_data_for_preset)
            output_preview_image = preview_image_for_preset

            temps_generation_image = f"{(time.time() - depart_time):.2f} sec"
            date_str = datetime.now().strftime("%Y_%m_%d")
            heure_str = datetime.now().strftime("%H_%M_%S")
            save_dir = os.path.join(SAVE_DIR, date_str)
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{date_str}_{heure_str}_{seed}_{width}x{height}_{idx+1}.{IMAGE_FORMAT.lower()}"
            chemin_image = os.path.join(save_dir, filename)

            lora_info_str = ", ".join([f"{name}({weight:.2f})" for name, weight in model_manager.loaded_loras.items()]) if model_manager.loaded_loras else translate("aucun_lora", translations)
            style_info_str = ", ".join(selected_style_display_names) if selected_style_display_names else translate("Aucun_style", translations)

            donnees_xmp = {
                 "Module": "SDXL Image Generation", "Creator": AUTHOR,
                 "Model": os.path.splitext(model_manager.current_model_name)[0] if model_manager.current_model_name else "N/A", 
                 "VAE": model_manager.current_vae_name, 
                 "Steps": num_steps, "Guidance": guidance_scale,
                 "Sampler": pipe.scheduler.__class__.__name__,
                 "IMAGE": f"{idx+1} {translate('image_sur',translations)} {num_images}",
                 "Inference": num_steps, "Style": style_info_str,
                 "original_prompt (User)": prompt_to_log_as_original, # Utiliser le prompt original loggué
                 "Prompt": prompt_en, # Prompt final utilisé
                 "LLM_Enhanced": prompt_is_currently_enhanced, # Utiliser l'état d'amélioration
                 "Negatif Prompt": negative_prompt_str, "Seed": seed,
                 "Size": selected_format_parts, 
                 "Loras": lora_info_str,
                 "Generation Time": temps_generation_image,
                 "PAG Enabled": pag_enabled,
                 "PAG Scale": f"{pag_scale:.2f}" if pag_enabled else "N/A", 
                 "PAG Custom Pipeline": "hyoungwoncho/sd_perturbed_attention_guidance" if pag_enabled else "N/A", 
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
            image_paths.append(chemin_image)
            
            # Aggressive memory cleanup
            del final_image
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            empty_working_set(translations)

            seed_strings.append(f"[{seed}]")
            formatted_seeds = " ".join(seed_strings)
 
            try:
                html_message_result = html_future.result(timeout=10)
            except Exception as html_err:
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_lors_generation_html', translations)}: {html_err}")
                 html_message_result = translate("erreur_lors_generation_html", translations)

            yield (
                image_paths, formatted_seeds, f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}",
                html_message_result, None, final_progress_html, # Do not yield the image object again
                bouton_charger_update_off, btn_generate_update_off,
                gr.update(interactive=False), 
                output_gen_data_json,
                output_preview_image
            )

        final_images = image_paths
        final_seeds = formatted_seeds
        final_html_msg = html_message_result
        final_preview_img = None 
        final_progress_html = "" 

        if not final_message: 
            elapsed_time = f"{(time.time() - start_time):.2f} sec"
            final_message = translate('temps_total_generation', translations) + " : " + elapsed_time
            print(txt_color("[INFO] ","info"), final_message)
            gr.Info(final_message, 3.0)

        final_save_button_state = gr.update(interactive=False)
        if not stop_event.is_set() and not stop_gen.is_set() and final_images:
             final_save_button_state = gr.update(interactive=True)

    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f"{translate('erreur_lors_generation', translations)} : {e}")
        traceback.print_exc()
        final_message = f"{translate('erreur_lors_generation', translations)} : {str(e)}"
        final_images = []
        final_seeds = ""
        final_html_msg = f"<p style='color: red;'>{final_message}</p>"
        final_preview_img = None
        final_progress_html = ""
        final_save_button_state = gr.update(interactive=False)
        output_gen_data_json = None 
        output_preview_image = None

    finally: 
        is_generating = False
        if (stop_event.is_set() or stop_gen.is_set()) and final_images:
            if 'date_str' in locals() and date_str: 
                chemin_rapport_html_actuel = os.path.join(SAVE_DIR, date_str, "rapport.html")
                print(txt_color("[INFO]", "info"), f"Tentative de finalisation du rapport HTML pour {chemin_rapport_html_actuel} suite à un arrêt.")
                
                finalisation_msg = finalize_html_report_if_needed(chemin_rapport_html_actuel, translations)
                print(txt_color("[INFO]", "info"), f"Résultat finalisation HTML: {finalisation_msg}")
                
                if "erreur" not in finalisation_msg.lower() and final_html_msg and isinstance(final_html_msg, str):
                    final_html_msg += f"<br/>{finalisation_msg}"
                elif "erreur" not in finalisation_msg.lower():
                    final_html_msg = finalisation_msg
            else:
                print(txt_color("[AVERTISSEMENT]", "warning"), "Impossible de déterminer le chemin du rapport HTML pour la finalisation (date_str non défini).")
        yield (
            final_images, final_seeds, final_message, final_html_msg,
            final_preview_img, final_progress_html,
            bouton_charger_update_on, btn_generate_update_on,
            final_save_button_state, 
            output_gen_data_json, 
            output_preview_image 
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
        empty_working_set(translations) # Add this line
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
initial_vae_value = "Auto" # Default to "Auto"
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
            model_name=os.path.basename(DEFAULT_MODEL), # type: ignore
            vae_name="Auto", # Utiliser "Auto" pour le VAE intégré
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
            initial_vae_value = "Auto" # S'assurer que c'est "Auto"
            initial_button_text = translate("generer", translations)
            initial_button_interactive = True
            # Utiliser le message retourné par charger_modele s'il existe
            initial_message_chargement = message_retour if message_retour else translate("modele_charge_pret", translations)

        except Exception as e_inner:
            traceback.print_exc()
            initial_model_value = None
            initial_vae_value = None
            initial_button_text = translate("charger_modele_pour_commencer", translations)
            initial_button_interactive = False
            initial_message_chargement = f"Erreur interne après chargement: {e_inner}"
            model_selectionne = None
            vae_selctionne = None

    else:
        initial_message_chargement = message_retour if message_retour else translate("erreur_chargement_modele_defaut", translations)

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
            custom_pipeline_to_use = "hyoungwoncho/sd_perturbed_attention_guidance"
            print(txt_color("[INFO]", "info"), f"PAG activé, tentative d'utilisation du custom_pipeline: {custom_pipeline_to_use}")
            gr.Info(f"PAG activé, tentative de chargement avec custom_pipeline: {custom_pipeline_to_use}", 3.0)

        success, message = model_manager.load_model(
            model_name=nom_fichier,
            vae_name=nom_vae,
            model_type="standard",
            gradio_mode=True,
            custom_pipeline_id=custom_pipeline_to_use 
        )
    except Exception as e:
        print(txt_color("[ERREUR]", "erreur"), f"Erreur inattendue lors de l'appel à model_manager.load_model: {e}")
        traceback.print_exc()
        success = False
        message = f"Erreur interne: {e}"
        model_manager.unload_model()


    if success:
        model_selectionne = nom_fichier
        vae_selctionne = nom_vae
        loras_charges.clear()
        etat_interactif = True
        texte_bouton = translate("generer", translations) 

    else:
        etat_interactif = False
        texte_bouton = translate("charger_modele_pour_commencer", translations) 
        selected_sampler_key_state.value = None

    update_interactif = gr.update(interactive=etat_interactif)
    update_texte = gr.update(value=texte_bouton)
    return message, update_interactif, update_texte

def update_globals_model_inpainting(nom_fichier):
    global model_selectionne
    # Initialisation des valeurs par défaut en cas d'échec
    etat_interactif = False
    # Texte pour le bouton de génération (bouton_generate_inpainting) en cas d'échec
    texte_bouton = translate("charger_modele_pour_commencer", translations) 
    message_final = "" # Message de statut pour message_chargement_inpainting

    try:
        success, message_chargement = model_manager.load_model(
            model_name=nom_fichier,
            vae_name="Auto", 
            model_type="inpainting",
            gradio_mode=True
        )
        message_final = message_chargement
    except Exception as e:
        print(txt_color("[ERREUR]", "erreur"), f"Erreur inattendue lors de l'appel à model_manager.load_model (inpainting): {e}")
        traceback.print_exc()
        success = False
        message = f"Erreur interne: {e}"
        model_manager.unload_model()

    if success:
        model_selectionne = nom_fichier # Assigner le modèle sélectionné uniquement en cas de succès
        etat_interactif = True
        texte_bouton = translate("generer_inpainting", translations) 
    else:
        model_selectionne = None # Réinitialiser si le chargement échoue
        # etat_interactif et texte_bouton sont déjà initialisés pour le cas d'échec

    update_interactif_gen_btn = gr.update(interactive=etat_interactif)
    update_texte_gen_btn = gr.update(value=texte_bouton)

    # Les outputs de cette fonction sont:
    # 1. message_chargement_inpainting (message de statut)
    # 2. bouton_generate_inpainting (interactivité)
    # 3. bouton_generate_inpainting (valeur/texte)
    return message_final, update_interactif_gen_btn, update_texte_gen_btn

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
        return None, None 

    current_background_pil = image_mask_value_dict.get("background") 
    if current_background_pil is None:
        if current_original_bg_props is not None: 
            return None, None 
        else: 
            return gr.update(), gr.update() 

    if current_background_pil is not None and not isinstance(current_background_pil, Image.Image):
        print(txt_color("[AVERTISSEMENT]", "warning"), f"handle_image_mask_interaction: Le fond de ImageMask n'est pas une PIL.Image. Type: {type(current_background_pil)}. Retour sans mise à jour.")
        return gr.update(), gr.update()  

    if current_background_pil is None:
        return None, None

    new_bg_props = (current_background_pil.width, current_background_pil.height, current_background_pil.mode)

    if current_original_bg_props is None or new_bg_props != current_original_bg_props:
        print(txt_color("[INFO]", "info"), "Nouvelle image de fond détectée dans ImageMask, validation en cours.")
        image_checker = ImageSDXLchecker(current_background_pil, translations)
        processed_background_for_pipeline = image_checker.redimensionner_image()

        if isinstance(processed_background_for_pipeline, Image.Image):
            return processed_background_for_pipeline, new_bg_props
        else:
            print(txt_color("[AVERTISSEMENT]", "warning"), "L'image de fond de ImageMask n'est pas valide après traitement par ImageSDXLchecker.")
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
        return Image.new('L', target_size, 0) 

    layers_pil_from_editor = editor_dict.get("layers", []) 
    
    if not layers_pil_from_editor: 
        return Image.new('L', target_size, 0) 

    composite_rgba_mask = Image.new('RGBA', target_size, (0, 0, 0, 0))

    for layer_pil in layers_pil_from_editor:
        if layer_pil is None:
            continue
        try:
            if not isinstance(layer_pil, Image.Image):
                print(txt_color("[AVERTISSEMENT]", "warning"), f"Un élément dans 'layers' n'est pas une image PIL: {type(layer_pil)}")
                continue
            layer_to_composite = layer_pil.convert('RGBA')
            if layer_to_composite.size != target_size:
                print(txt_color("[INFO]", "info"), f"Redimensionnement du layer de masque de {layer_to_composite.size} à {target_size}")
                layer_to_composite = layer_to_composite.resize(target_size, Image.Resampling.NEAREST)

            composite_rgba_mask.alpha_composite(layer_to_composite)
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_traitement_layer_mask', translations)}: {e}")
            continue

    alpha_channel = composite_rgba_mask.split()[-1] 
    binary_mask_pil = alpha_channel.point(lambda p: 255 if p > 0 else 0, mode='L')
    
    return binary_mask_pil
#################################################
#FONCTIONS pour les presets :
#################################################
def handle_preset_rename_click(preset_id, current_trigger): 
    print(txt_color("[INFO]", "info"), f"{translate('preset_action_rename_click', translations)} {translate('current_trigger_log', translations)}: {current_trigger}")

    return gr.update(value=preset_id), gr.update(value=current_trigger + 1)

def handle_preset_cancel_click(current_trigger): 
    print("[Action Preset] Clic Annuler Édition.")
    return gr.update(value=None), gr.update(value=current_trigger + 1)

def handle_preset_rename_submit(preset_id, new_name, current_trigger):
    """Soumet le nouveau nom pour le preset."""
    print(f"[Action Preset {preset_id}] Submit Renommer vers '{new_name}'.")
    if not preset_id or not new_name:
        gr.Warning(translate("erreur_nouveau_nom_vide", translations))
        return gr.update(value=preset_id), gr.update() 

    success, message = preset_manager.rename_preset(preset_id, new_name)
    if success:
        gr.Info(message)
        return gr.update(value=None), gr.update(value=current_trigger + 1)
    else:
        gr.Warning(message)
        return gr.update(value=preset_id), gr.update()

def handle_preset_delete_click(preset_id, current_trigger, page, search, sort, filter_models, filter_samplers, filter_loras, current_search_value):
    """Supprime le preset, déclenche un refresh ET met à jour la pagination."""
    trigger_update_on_error = gr.update()
    pagination_update_on_error = gr.update()
    search_update_on_error = gr.update()

    success, message = preset_manager.delete_preset(preset_id)
    if success:
        gr.Info(message)
        pagination_update = update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras)
        current_search_str = current_search_value if current_search_value else ""
        if current_search_str.endswith(" "):
            temp_search_value = current_search_str[:-1] 
        else:
            temp_search_value = current_search_str + " " 
        
        search_update_hack = gr.update(value=temp_search_value)
        return gr.update(value=current_trigger + 1), pagination_update,search_update_hack
    else:
        gr.Warning(message)
        return trigger_update_on_error, pagination_update_on_error, search_update_on_error

def handle_preset_rating_change(preset_id, new_rating_value):
    """Met à jour la note du preset."""

    if preset_id is not None and new_rating_value is not None:
        success, message = preset_manager.update_preset_rating(preset_id, int(new_rating_value))
        if not success:
            gr.Warning(message)

def update_pagination_and_trigger_refresh(page, search, sort, filter_models, filter_samplers, filter_loras, current_trigger):
    """Met à jour l'UI de pagination ET incrémente le trigger de refresh."""
    pagination_updates = update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras)
    trigger_update = gr.update(value=current_trigger + 1)
    return list(pagination_updates) + [trigger_update]

def handle_page_change(direction, current_page):
    """Calcule la nouvelle page."""
    new_page = current_page + direction
    return gr.update(value=new_page)

def update_filter_choices_after_save():
    models, samplers, loras = get_filter_options()
    return gr.update(choices=models), gr.update(choices=samplers), gr.update(choices=loras)

def handle_save_preset(preset_name, preset_notes, current_gen_data_json, preview_image_pil, current_trigger, current_search_value):
    """
    Gère l'appel à preset_manager pour sauvegarder le preset.
    Appelée par l'événement .click() de bouton_save_current_preset.
    Retourne les updates pour vider les champs et mettre à jour les filtres.
    """
    filter_updates_on_error = [gr.update(), gr.update(), gr.update()]
    trigger_update_on_error = gr.update()
    search_update_on_error = gr.update()

    if not preset_name:
        gr.Warning(translate("erreur_nom_preset_vide", translations), 3.0)
        return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error

    current_gen_data = None
    if isinstance(current_gen_data_json, str):
        try:
            current_gen_data = json.loads(current_gen_data_json)
        except json.JSONDecodeError as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_decodage_json_preset', translations)}: {e}")
            gr.Warning(translate("erreur_interne_decodage_json", translations), 3.0)
            return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error
    else:
         if isinstance(current_gen_data_json, dict):
             current_gen_data = current_gen_data_json
         else:
             print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_type_inattendu_json_preset', translations)}: {type(current_gen_data_json)}")
             gr.Warning(translate("erreur_interne_donnees_generation_invalides", translations), 3.0)
             return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error

    if not isinstance(preview_image_pil, Image.Image):
         if isinstance(preview_image_pil, bytes):
             try:
                 preview_image_pil = Image.open(BytesIO(preview_image_pil))
                 print(txt_color("[INFO]", "info"), translate("info_image_preview_chargee_bytes", translations))
             except Exception as img_err:
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_chargement_image_preview_bytes', translations)}: {img_err}")
                 preview_image_pil = None 
         else:
             preview_image_pil = None 

    if not isinstance(current_gen_data, dict) or preview_image_pil is None:
         print(txt_color("[ERREUR]", "erreur"), translate("erreur_donnees_generation_ou_image_manquantes", translations))
         print(f"  Type data après JSON: {type(current_gen_data)}")
         print(f"  Type image après vérif: {type(preview_image_pil)}")
         gr.Warning(translate("erreur_pas_donnees_generation", translations), 3.0)
         return gr.update(), gr.update(), *filter_updates_on_error,  trigger_update_on_error, search_update_on_error

    data_to_save = current_gen_data.copy()
    data_to_save['notes'] = preset_notes
    data_to_save['original_user_prompt'] = current_gen_data.get('original_user_prompt', data_to_save.get('prompt', ''))
    data_to_save['current_prompt_is_enhanced'] = current_gen_data.get('current_prompt_is_enhanced', False)
    data_to_save['enhancement_cycle_active'] = current_gen_data.get('enhancement_cycle_active', False)


    try:
        success, message = preset_manager.save_gen_image_preset(preset_name, data_to_save, preview_image_pil)
    except Exception as save_err:
        print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_appel_sauvegarde_preset', translations)}: {save_err}")
        traceback.print_exc()
        success = False
        message = translate("erreur_interne_sauvegarde_preset", translations)

    if success: 
        gr.Info(message, 3.0)
        try:
            update_model, update_sampler, update_lora = update_filter_choices_after_save()
            current_search_str = current_search_value if current_search_value else ""
            if current_search_str.endswith(" "):
                temp_search_value = current_search_str[:-1] 
            else:
                temp_search_value = current_search_str + " " 
            search_update_hack = gr.update(value=temp_search_value)
            return gr.update(value=""), gr.update(value=""), update_model, update_sampler, update_lora, gr.update(value=current_trigger + 1), search_update_hack
        except Exception as filter_err:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_update_filtres_preset', translations)}: {filter_err}")
            # search_update_hack n'est pas défini ici, donc on ne peut pas le retourner.
            # On retourne les filter_updates_on_error à la place.
            return gr.update(value=""), gr.update(value=""), *filter_updates_on_error, gr.update(value=current_trigger + 1), search_update_on_error # Utiliser search_update_on_error
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
        num_lora_slots = 4
        # Ajuster le nombre d'outputs pour correspondre à la nouvelle structure
        # text_input, original_user_prompt_state, current_prompt_is_enhanced_state, enhancement_cycle_active_state,
        # enhance_or_redo_button, validate_prompt_button
        # + les anciens (model, vae, style, guidance, steps, format, sampler, seed, 4*lora_checks, 4*lora_dd, 4*lora_scales, pag_enabled, pag_scale, pag_applied_layers, message_chargement)
        # Total: 6 (nouveaux) + 7 (base) + 3*4 (loras) + 3 (PAG) + 1 (message) = 6 + 7 + 12 + 3 + 1 = 29
        return [gr.update()] * (6 + 7 + 3 * num_lora_slots + 3 + 1)


    try:
        # Le prompt à afficher dans text_input est celui sous la clé 'prompt'
        prompt_to_display = preset_data.get('prompt', '')
        # Le prompt original du cycle d'amélioration
        original_user_prompt_from_preset = preset_data.get('original_user_prompt', prompt_to_display)
        # Les états d'amélioration
        current_prompt_is_enhanced_from_preset = preset_data.get('current_prompt_is_enhanced', False)
        enhancement_cycle_active_from_preset = preset_data.get('enhancement_cycle_active', False)
        last_ai_output_from_preset = preset_data.get('last_ai_enhanced_output', None) # Charger le dernier output IA

        traduire_checkbox_preset = preset_data.get('traduire', False)
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
            loaded_style_names = []

        model_name = preset_data.get('model', None)
        # Original VAE name from preset, could be None, "Défaut VAE", or a real VAE name
        raw_vae_name_from_preset = preset_data.get('vae')

        # Determine the VAE to attempt to use, mapping "Défaut VAE" or None to "Auto"
        if raw_vae_name_from_preset is None or raw_vae_name_from_preset == "Défaut VAE":
            vae_to_attempt_loading = "Auto"
        else:
            vae_to_attempt_loading = raw_vae_name_from_preset

        guidance = preset_data.get('guidance_scale', 7.0)
        steps = preset_data.get('num_steps', 30)
        sampler_key = preset_data.get('sampler_key', 'sampler_euler')
        sampler_display = translate(sampler_key, translations) 
        seed_val = preset_data.get('seed', -1) # Renommé pour éviter conflit avec module seed
        width = preset_data.get('width', 1024)
        height = preset_data.get('height', 1024)
        loras_data = preset_data.get('loras', [])
        loaded_loras = []

        custom_pipeline_id_preset = preset_data.get('custom_pipeline_id')
        pag_enabled_preset = bool(custom_pipeline_id_preset) 
        pag_scale_preset = preset_data.get('pag_scale', 1.5)
        pag_applied_layers_preset = preset_data.get('pag_applied_layers', "m0") 

        available_models = model_manager.list_models(model_type="standard", gradio_mode=True)
        model_update = gr.update()
        if model_name and model_name in available_models:
            model_update = gr.update(value=model_name)
        elif model_name:
            print(txt_color("[AVERTISSEMENT]", "warning"), f"{translate('preset_load_model_not_found_warn', translations)}")


        available_vae_choices_for_ui = model_manager.list_vaes() # These are the valid choices for the dropdown
        
        # Determine the final VAE value for the UI dropdown
        final_vae_for_ui_dropdown = "Auto" # Default to "Auto" if preferred VAE not found
        if vae_to_attempt_loading in available_vae_choices_for_ui:
            final_vae_for_ui_dropdown = vae_to_attempt_loading
        else:
            # Warn only if the intended VAE was something specific and not found,
            # and it wasn't already "Auto" (which we've now defaulted to).
            if vae_to_attempt_loading != "Auto":
                gr.Warning(translate("erreur_vae_preset_introuvable", translations).format(vae_to_attempt_loading))
        
        vae_update = gr.update(value=final_vae_for_ui_dropdown)

        if isinstance(loras_data, str):
            try:
                loaded_loras = json.loads(loras_data)
                if not isinstance(loaded_loras, list):
                    loaded_loras = [] 
            except json.JSONDecodeError:
                loaded_loras = [] 
        elif isinstance(loras_data, list):
            loaded_loras = loras_data
        else:
            loaded_loras = []
        format_string = f"{width}*{height}"
        orientation_key = None
        for fmt in config["FORMATS"]:
            dims = fmt.get("dimensions", "")
            if dims == f"{width}*{height}":
                orientation_key = fmt.get("orientation")
                break
        if orientation_key:
            format_string += f" {translate(orientation_key, translations)}"
        else:
            format_string = f"{width}*{height} {translate('orientation_inconnue', translations)}" 

        if format_string not in FORMATS:
             format_string = FORMATS[0] 

        num_lora_slots = 4
        lora_check_updates = [gr.update(value=False) for _ in range(num_lora_slots)]
        lora_dd_updates = [gr.update(value=None, interactive=False, choices=[translate("aucun_lora_disponible", translations)]) for _ in range(num_lora_slots)]
        lora_scale_updates = [gr.update(value=0) for _ in range(num_lora_slots)]

        available_loras = model_manager.list_loras(gradio_mode=True)
        has_available_loras = bool(available_loras) and translate("aucun_modele_trouve", translations) not in available_loras and translate("repertoire_not_found", translations) not in available_loras
        lora_choices = available_loras if has_available_loras else [translate("aucun_lora_disponible", translations)]

        for i, lora_info in enumerate(loaded_loras):
            if i >= num_lora_slots: break 
            lora_name = lora_info.get('name')
            lora_weight = lora_info.get('weight')
            if lora_name and lora_weight is not None:
                if lora_name in lora_choices:
                    lora_check_updates[i] = gr.update(value=True)
                    lora_dd_updates[i] = gr.update(choices=lora_choices, value=lora_name, interactive=True)
                    lora_scale_updates[i] = gr.update(value=lora_weight)
                else:
                    warn_msg = f"LoRA '{lora_name}' du preset non trouvé. Ignoré pour le slot {i+1}."
                    print(txt_color("[WARN]", "warning"), warn_msg)

        sampler_update_msg, success = apply_sampler_to_pipe(model_manager.get_current_pipe(), sampler_key, translations)
        if success:
            global global_selected_sampler_key 
            global_selected_sampler_key = sampler_key
            sampler_update = gr.update(value=sampler_display) 
            gr.Info(sampler_update_msg, 2.0) 
        else:
            gr.Warning(sampler_update_msg)
            default_sampler_key = "sampler_euler"
            default_sampler_display = translate(default_sampler_key, translations)
            apply_sampler_to_pipe(model_manager.get_current_pipe(), default_sampler_key, translations)  
            global_selected_sampler_key = default_sampler_key 
            sampler_update = gr.update(value=default_sampler_display) 

        valid_style_choices = [style["name"] for style in STYLES if style["name"] != translate("Aucun_style", translations)]
        final_style_selection = [s_name for s_name in loaded_style_names if s_name in valid_style_choices]
        if len(final_style_selection) != len(loaded_style_names):
            print(txt_color("[WARNING]", "warning"), translate('preset_styles_unavailable_warn', translations))


        # Mises à jour pour les boutons d'amélioration
        enhance_button_text_update = translate("refaire_amelioration_btn", translations) if enhancement_cycle_active_from_preset else translate("ameliorer_prompt_ia_btn", translations)
        enhance_button_interactive_update = bool(prompt_to_display.strip())
        validate_button_visible_update = enhancement_cycle_active_from_preset
        validate_button_interactive_update = enhancement_cycle_active_from_preset

        outputs_list = [
            model_update, 
            vae_update, 
            gr.update(value=prompt_to_display), # text_input
            gr.update(value=final_style_selection),  
            gr.update(value=guidance),               
            gr.update(value=steps),                  
            gr.update(value=format_string),          
            sampler_update,                          
            gr.update(value=seed_val), # seed_input               
            *lora_check_updates,                     
            *lora_dd_updates,                        
            *lora_scale_updates,                     
            gr.update(value=pag_enabled_preset),     
            gr.update(value=pag_scale_preset),       
            gr.update(value=pag_applied_layers_preset), 
            # Nouveaux outputs pour les états et boutons d'amélioration
            gr.update(value=original_user_prompt_from_preset), # original_user_prompt_state
            gr.update(value=current_prompt_is_enhanced_from_preset), # current_prompt_is_enhanced_state
            gr.update(value=enhancement_cycle_active_from_preset), # enhancement_cycle_active_state
            gr.update(value=last_ai_output_from_preset), # last_ai_enhanced_output_state
            gr.update(value=enhance_button_text_update, interactive=enhance_button_interactive_update), # enhance_or_redo_button
            gr.update(visible=validate_button_visible_update, interactive=validate_button_interactive_update), # validate_prompt_button
            gr.update(value=translate("preset_charge_succes", translations).format(preset_data.get('name', f'ID: {preset_id}')))
        ]
        print(txt_color("[INFO]", "info"), f"{translate('preset_loaded_ui_log', translations).format(name=preset_data.get('name', f'ID: {preset_id}'))}")

        gr.Info(translate("preset_charge_succes", translations).format(preset_data.get('name', f'ID: {preset_id}')), 2.0) 
        
        return outputs_list
    except Exception as e:
        print(txt_color("[ERREUR]", "erreur"), f"Erreur lors du chargement du preset {preset_id}: {e}")
        traceback.print_exc()
        gr.Warning(translate("erreur_generale_chargement_preset", translations))
        num_lora_slots = 4 
        return [gr.update()] * (2 + 7 + 3 * num_lora_slots + 1 + 3 + 1 + 3 + 2) # +1 pour last_ai_enhanced_output_state
        
def reset_page_state_only():
    """Retourne simplement une mise à jour pour mettre l'état de la page à 1."""
    print(txt_color("[INFO]", "info"), translate('reset_page_state_log', translations))

    return gr.update(value=1)

def handle_page_dropdown_change(page_selection):
    """Gère le changement du dropdown de page."""
    print(txt_color("[INFO]", "info"), f"{translate('page_dropdown_change_log', translations).format(page=page_selection)}")

    return page_selection


############################################################
############################################################
#####################USER INTERFACE#########################
############################################################
############################################################
# États pour la nouvelle logique d'amélioration du prompt
original_user_prompt_state = gr.State(value="") # Stocke le prompt utilisateur avant la 1ère amélioration du cycle
current_prompt_is_enhanced_state = gr.State(value=False) # True si text_input contient un prompt fraîchement amélioré par l'IA
enhancement_cycle_active_state = gr.State(value=False) # True si un cycle Améliorer/Refaire/Valider est en cours
last_ai_enhanced_output_state = gr.State(value=None) # Stocke le dernier prompt sorti du LLM


def on_enhance_or_redo_button_click(current_text_in_box, original_prompt_for_cycle, cycle_is_active, llm_model_path, current_translations):
    """
    Gère le clic sur le bouton qui peut être "Améliorer le prompt" ou "Refaire l'amélioration".
    Charge le LLM si besoin, améliore le prompt et met à jour l'UI en conséquence.
    Met à jour le text_input avec le prompt amélioré.
    """
    prompt_to_enhance_this_time = ""
    new_original_prompt_for_cycle = original_prompt_for_cycle

    if not current_text_in_box.strip() and not cycle_is_active : # Si le prompt est vide ET que ce n'est pas un "Refaire"
        gr.Warning(translate("llm_enhancement_no_prompt", current_translations), 3.0)
        return (current_text_in_box,
                gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=True), # Garder interactif
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=False),
                gr.update(value=False),
                gr.update(value=None))

    if llm_prompter_util.llm_model_prompter is None or llm_prompter_util.llm_tokenizer_prompter is None:
        print(f"{txt_color('[INFO]', 'info')} Tentative de chargement du LLM Prompter pour l'amélioration du prompt...")
        gr.Info(translate("llm_prompter_loading_on_demand", current_translations), 3.0)
        success_load = llm_prompter_util.init_llm_prompter(llm_model_path, current_translations)
        if not success_load:
            gr.Warning(translate("llm_prompter_load_failed_on_demand", current_translations).format(model_path=llm_model_path), 5.0)
            return (current_text_in_box,
                    gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=True), # Garder interactif
                    gr.update(visible=False),
                    gr.update(value=original_prompt_for_cycle),
                    gr.update(value=False),
                    gr.update(value=False),
                    gr.update(value=None))
        else:
            gr.Info(translate("llm_prompter_loaded_successfully", current_translations), 3.0)

    if not cycle_is_active:
        # C'est un clic "Améliorer le prompt"
        # La vérification du prompt vide est faite au début
        prompt_to_enhance_this_time = current_text_in_box
        new_original_prompt_for_cycle = current_text_in_box
    else:
        # C'est un clic "Refaire l'amélioration"
        prompt_to_enhance_this_time = original_prompt_for_cycle # Utiliser le prompt original du cycle

    gr.Info(translate("llm_prompt_enhancing_in_progress", current_translations), 2.0)
    enhanced_prompt_candidate = llm_prompter_util.generate_enhanced_prompt(prompt_to_enhance_this_time, llm_model_path, translations=current_translations)

    if enhanced_prompt_candidate and enhanced_prompt_candidate.strip() and enhanced_prompt_candidate.strip().lower() != prompt_to_enhance_this_time.strip().lower():
        gr.Info(translate("prompt_enrichi_applique", current_translations), 2.0)
        return (enhanced_prompt_candidate,
                gr.update(value=translate("refaire_amelioration_btn", current_translations), interactive=True),
                gr.update(visible=True, interactive=True),
                gr.update(value=new_original_prompt_for_cycle),
                gr.update(value=True),
                gr.update(value=True),
                gr.update(value=enhanced_prompt_candidate))
    else:
        gr.Warning(translate("llm_prompt_enhancement_failed_or_same", current_translations), 3.0)
        if not cycle_is_active: # Echec sur un "Améliorer"
            return (current_text_in_box,
                    gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=True),
                    gr.update(visible=False),
                    gr.update(value=new_original_prompt_for_cycle),
                    gr.update(value=False),
                    gr.update(value=False),
                    gr.update(value=None))
        else: # Echec sur un "Refaire"
            return (current_text_in_box, # Garder le prompt amélioré précédent dans text_input
                    gr.update(value=translate("refaire_amelioration_btn", current_translations), interactive=True),
                    gr.update(visible=True, interactive=True), # Garder Valider visible
                    gr.update(value=new_original_prompt_for_cycle),
                    gr.update(value=True),
                    gr.update(value=True),
                    gr.update(value=current_text_in_box))

def on_validate_prompt_button_click(validated_prompt_text, current_translations):
    """Gère le clic sur le bouton 'Valider le prompt'."""
    gr.Info(translate("prompt_amelioration_validee", current_translations), 2.0)
    return (gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=True),
            gr.update(visible=False),
            gr.update(value=validated_prompt_text),
            gr.update(value=False),
            gr.update(value=False),
            gr.update(value=None))

def handle_text_input_change(text_value,
                             last_ai_output_val, # from last_ai_enhanced_output_state
                             is_cycle_active_val,    # from enhancement_cycle_active_state
                             llm_model_path, current_translations):
    """
    Gère la soumission (submit) ou la modification (input) du champ de saisie du prompt.
    Réinitialise le cycle d'amélioration UNIQUEMENT si le texte est réellement modifié par l'utilisateur pendant un cycle actif.
    Active/désactive le bouton "Améliorer..." et gère le déchargement du LLM si le prompt est vidé.
    """
    enhance_button_interactive = bool(text_value.strip()) # Le bouton est interactif si le prompt n'est pas vide

    if not text_value:
        if llm_prompter_util.llm_model_prompter is not None or llm_prompter_util.llm_tokenizer_prompter is not None:
            llm_prompter_util.unload_llm_prompter(current_translations)
            gr.Info(translate("llm_prompter_unloaded_due_to_empty_prompt", current_translations), 3.0)
        return (gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=enhance_button_interactive),
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=False),
                gr.update(value=False),
                gr.update(value=None))
    else:
        if is_cycle_active_val and text_value == last_ai_output_val:
            # Un cycle est actif, et l'utilisateur a soumis (ex: Entrée) le texte exact
            # qui était le dernier output de l'IA. Ne pas réinitialiser le cycle.
            # Les boutons "Refaire" et "Valider" doivent rester.
            # Le bouton "Améliorer/Refaire" reste "Refaire" et interactif.
            return (gr.update(value=translate("refaire_amelioration_btn", current_translations), interactive=True),
                    gr.update(visible=True, interactive=True), # Valider reste visible et interactif
                    gr.update(), # original_user_prompt_state (pas de changement)
                    gr.update(value=True), # current_prompt_is_enhanced_state (doit rester True)
                    gr.update(value=True), # enhancement_cycle_active_state (doit rester True)
                    gr.update()) # last_ai_enhanced_output_state (pas de changement)
        else:
            # Soit aucun cycle n'était actif, soit l'utilisateur a modifié le texte
            # par rapport au dernier output de l'IA. Réinitialiser le cycle.
            if is_cycle_active_val: # Uniquement si un cycle était actif et que le texte a changé
                gr.Info(translate("prompt_modifie_reinitialisation_amelioration", current_translations), 2.0)

            return (gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=enhance_button_interactive),
                    gr.update(visible=False), # Cacher Valider
                    gr.update(value=text_value), # Le texte actuel devient la nouvelle base pour original_user_prompt_state
                    gr.update(value=False),   # current_prompt_is_enhanced_state (ce n'est plus un output IA direct)
                    gr.update(value=False),   # enhancement_cycle_active_state (le cycle est terminé/réinitialisé)
                    gr.update(value=None))    # last_ai_enhanced_output_state (plus d'output IA pertinent pour ce cycle)

block_kwargs = {"theme": gradio_change_theme(GRADIO_THEME)}
if js_code:
    block_kwargs["js"] = js_code


with gr.Blocks(**block_kwargs) as interface:
    # --- États ---
    # --- États pour les Presets ---
    preset_refresh_trigger = gr.State(0)
    # Remplacer prompt_was_enhanced_state par les nouveaux états
    original_user_prompt_state = gr.State(value="")
    current_prompt_is_enhanced_state = gr.State(value=False)
    enhancement_cycle_active_state = gr.State(value=False)
    last_ai_enhanced_output_state = gr.State(value=None)
    current_preset_page_state = gr.State(1)
    # --- Fin États Presets ---
    last_successful_generation_data = gr.State(value=None)
    last_successful_preview_image = gr.State(value=None)

    # --- Memory Management Accordion ---
    # Create the UI components for memory management
    memory_ui_components = create_memory_accordion_ui(translations, model_manager) # Returns a dict with "all_stats_html"
    all_stats_html_component = memory_ui_components["all_stats_html"]
    enable_live_memory_stats_checkbox = memory_ui_components["enable_live_stats_checkbox"]
    memory_stats_update_interval_slider = memory_ui_components["memory_interval_slider"]

    gr.Markdown(f"# Pycnaptiq-AI {version()}", elem_id="main_title_markdown") # Added elem_id for potential future use

############################################################
########***************************************************
########TAB GENRATION IMAGE
########***************************************************
############################################################

    with gr.Tab(translate("generation_image", translations)):
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                text_input = gr.Textbox(label=translate("prompt", translations), info=translate("entrez_votre_texte_ici", translations), elem_id="promt_input")
                # Nouveaux boutons pour l'amélioration du prompt
                with gr.Row():
                    enhance_or_redo_button = gr.Button(
                        translate("ameliorer_prompt_ia_btn", translations),
                        interactive=True # Toujours interactif au démarrage
                    )
                    validate_prompt_button = gr.Button(
                        translate("valider_prompt_btn", translations),
                        interactive=False, visible=False
                    )
                traduire_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", translations), value=False, info=translate("traduire_en_anglais", translations))
                # enhance_prompt_checkbox est supprimée
                style_dropdown = gr.Dropdown(
                    choices=[style["name"] for style in STYLES if style["name"] != translate("Aucun_style", translations)],
                    value=[],
                    label=translate("styles", translations),
                    info=translate("Selectionnez_un_ou_plusieurs_styles", translations),
                    multiselect=True,
                    max_choices=4
                )
                use_image_checkbox = gr.Checkbox(label=translate("generer_prompt_image", translations), value=False)
                time_output = gr.Textbox(label=translate("temps_rendu", translations), interactive=False)
                html_output = gr.Textbox(label=translate("mise_a_jour_html", translations), interactive=False) # Cet output n'est plus utilisé par generate_image
                message_chargement = gr.Textbox(label=translate("statut", translations), value=initial_message_chargement)
                image_input = gr.Image(label=translate("telechargez_image", translations), type="pil", visible=False)


            with gr.Column(scale=1, min_width=200):
                image_output = gr.Gallery(label=translate("images_generees", translations))
                progress_html_output = gr.HTML(value="")
                guidance_slider = gr.Slider(1, 20, value=7, label=translate("guidage", translations))
                num_steps_slider = gr.Slider(1, 50, value=30, label=translate("etapes", translations), step=1)
                format_dropdown = gr.Dropdown(choices=FORMATS, value=FORMATS[3], label=translate("format", translations))
                with gr.Accordion(translate("pag_options_label", translations), open=False):
                    pag_enabled_checkbox = gr.Checkbox(
                        label=translate("enable_pag_label", translations),
                        value=False
                    )
                    pag_scale_slider = gr.Slider(
                        minimum=0.0, maximum=10.0, value=1.5, step=0.1,
                        label=translate("pag_scale_label", translations),
                        interactive=True,
                        visible=False
                    )
                    pag_applied_layers_input = gr.Textbox(
                        label=translate("pag_applied_layers_label", translations),
                        info=translate("pag_applied_layers_info", translations),
                        value="m0",
                        visible=False
                    )

                seed_input = gr.Number(label=translate("seed", translations), value=-1, elem_id="seed_input_main_gen")
                num_images_slider = gr.Slider(1, 30, value=1, label=translate("nombre_images_generer", translations), step=1)


            with gr.Column(scale=1, min_width=200):
                with gr.Row():
                    with gr.Column():
                        preview_image_output = gr.Image(height=170, label=translate("apercu_etapes", translations),interactive=False)
                        seed_output = gr.Textbox(label=translate("seed_utilise", translations))
                        value = DEFAULT_MODEL if DEFAULT_MODEL else None
                        modele_dropdown = gr.Dropdown(label=translate("selectionner_modele", translations), choices=modeles_disponibles, value=initial_model_value, allow_custom_value=True)
                        vae_dropdown = gr.Dropdown(label=translate("selectionner_vae", translations), choices=vaes, value=initial_vae_value, allow_custom_value=True)
                        sampler_display_choices = get_sampler_choices(translations)
                        default_sampler_display = translate(global_selected_sampler_key, translations)

                        sampler_dropdown = gr.Dropdown(
                            label=translate("selectionner_sampler", translations),
                            choices=sampler_display_choices,
                            value=default_sampler_display,
                            allow_custom_value=True,
                        )
                        bouton_charger = gr.Button(translate("charger_modele", translations))



            with gr.Column():
                texte_bouton_gen_initial = translate("charger_modele_pour_commencer", translations)
                btn_generate = gr.Button(value=initial_button_text, interactive=initial_button_interactive, variant="primary")
                btn_stop = gr.Button(translate("arreter", translations), variant="stop")
                btn_stop_after_gen = gr.Button(translate("stop_apres_gen", translations), variant="stop")
                bouton_lister = gr.Button(translate("lister_modeles", translations))
                with gr.Accordion(translate("batch_runner_accordion_title", translations), open=False):
                    gr.Markdown(f"#### {translate('batch_runner_title', translations)}")
                    batch_json_file_input = gr.File(
                        label=translate("upload_batch_json", translations),
                        file_types=[".json"],
                        file_count="single"
                    )
                    batch_run_button = gr.Button(translate("run_batch_button", translations), variant="primary")
                    batch_status_output = gr.Textbox(label=translate("batch_status", translations), interactive=False)
                    batch_progress_output = gr.HTML()
                    batch_gallery_output = gr.Gallery(label=translate("batch_generated_images", translations), height="auto", interactive=False)
                    batch_stop_button = gr.Button(translate("stop_batch_button", translations), variant="stop", interactive=False)
                with gr.Accordion(translate("sauvegarder_preset_section", translations), open=False):
                    preset_name_input = gr.Textbox(
                        label=translate("nom_preset", translations),
                        placeholder=translate("entrez_nom_preset", translations)
                    )
                    preset_notes_input = gr.Textbox(
                        label=translate("notes_preset", translations),
                        placeholder=translate("entrez_notes_preset", translations),
                        lines=3
                    )
                    bouton_save_current_preset = gr.Button(
                        translate("confirmer_sauvegarde_preset", translations),
                        variant="primary",
                        interactive=False
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
                                    lora_dropdown = gr.Dropdown(choices=lora_initial_dropdown_choices, label=translate("selectionner_lora", translations), interactive=has_initial_loras)
                                    lora_scale_slider = gr.Slider(0, 1, value=0, label=translate("poids_lora", translations))
                                    lora_checks.append(lora_check)
                                    lora_dropdowns.append(lora_dropdown)
                                    lora_scales.append(lora_scale_slider)

                                    # Récupérer les composants courants pour la clarté dans le .change
                                    current_lora_check = lora_checks[i-1]
                                    current_lora_dropdown = lora_dropdowns[i-1]

                                    current_lora_check.change(
                                        fn=lambda chk, has_loras_flag: gr.update(interactive=chk and has_loras_flag),
                                        inputs=[current_lora_check, gr.State(has_initial_loras)], # Utiliser has_initial_loras
                                        outputs=[current_lora_dropdown] # Cibler explicitement le dropdown courant
                                    )
                    lora_message = gr.Textbox(label=translate("message_lora", translations), value=initial_lora_message)

                with gr.Accordion(translate("gestion_modules", translations), open=False):
                    gr.Markdown(translate("activer_desactiver_modules", translations))
                    with gr.Column():
                        loaded_modules_details = gestionnaire.get_module_details()

                        if not loaded_modules_details:
                            gr.Markdown(f"*{translate('aucun_module_charge_pour_gestion', translations)}*")
                        else:
                            for module_detail in loaded_modules_details:
                                module_name = module_detail["name"]
                                display_name = module_detail["display_name"]
                                is_active = module_detail["is_active"]

                                module_checkbox = gr.Checkbox(
                                    label=display_name,
                                    value=is_active,
                                    elem_id=f"module_toggle_{module_name}"
                                )
                                module_checkbox.change(
                                    fn=functools.partial(handle_module_toggle, module_name, gestionnaire_instance=gestionnaire, preset_manager_instance=preset_manager),
                                    inputs=[module_checkbox],
                                    outputs=[]
                                )
                def mettre_a_jour_listes():
                    modeles = lister_fichiers(MODELS_DIR, translations, gradio_mode=True)
                    vaes = model_manager.list_vaes() # Utilise model_manager qui inclut "Auto"
                    loras = model_manager.list_loras(gradio_mode=True)

                    has_loras = bool(loras) and loras[0] != translate("aucun_modele_trouve", translations) and loras[0] != translate("repertoire_not_found", translations)
                    lora_choices = loras if has_loras else ["Aucun LORA disponible"]
                    lora_updates = [gr.update(choices=lora_choices, interactive=has_loras, value=None) for _ in range(4)]
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
            loras = filter_data.get('loras', [])
            sampler_display_names = [translate(s_key, translations) for s_key in sampler_keys_in_presets]
            sampler_display_names = list(set(sampler_display_names))
            sampler_display_names.sort()
            return models, sampler_display_names, loras

        initial_models, initial_samplers, initial_loras = get_filter_options()
        with gr.Row():
            preset_search_input = gr.Textbox(
                label=translate("rechercher_preset", translations),
                placeholder="..."
            )
            preset_sort_dropdown = gr.Dropdown(
                choices=["Date Création", "Nom A-Z", "Nom Z-A", "Date Utilisation", "Note"],
                value="Date Création",
                label=translate("trier_par", translations)
            )

        with gr.Row():
            preset_filter_model = gr.Dropdown(
                label=translate("filtrer_par_modele", translations),
                choices=initial_models,
                value=[],
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
                label=translate("page", translations),
                choices=[1],
                value=1,
                interactive=False,
                elem_id="preset_page_dropdown"
            )

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
            if filter_samplers:
                reverse_sampler_map = {v: k for k, v in translations.items() if k.startswith("sampler_")}
                for sampler_display_name in filter_samplers:
                    internal_key = reverse_sampler_map.get(sampler_display_name)
                    if internal_key:
                        sampler_keys_to_filter.append(internal_key)
                    else:
                        print(txt_color("[ERREUR ]", "erreur"), translate("gestion_samplers_erreur", translations).format(sampler_display_name))
            all_presets_data = preset_manager.load_presets_for_display(
                preset_type='gen_image', search_term=search, sort_by=sort,
                selected_models=filter_models or None,
                selected_samplers=sampler_keys_to_filter or None,
                selected_loras=filter_loras or None
            )

            total_presets = len(all_presets_data)
            total_pages = math.ceil(total_presets / PRESETS_PER_PAGE) if total_presets > 0 else 1
            current_page = max(1, min(page, total_pages))
            start_index = (current_page - 1) * PRESETS_PER_PAGE
            end_index = start_index + PRESETS_PER_PAGE
            presets_for_page = all_presets_data[start_index:end_index]

            def safe_get_from_row(row, key, default=None):
                try:
                    return row[key] if key in row.keys() else default
                except (IndexError, TypeError): return default

            # Liste des outputs pour handle_preset_load_click
            # Doit correspondre à la nouvelle structure avec les états d'amélioration
            gen_ui_outputs_for_preset_load = [
                modele_dropdown,
                vae_dropdown,
                text_input, # Le prompt à afficher
                style_dropdown,
                guidance_slider,
                num_steps_slider,
                format_dropdown,
                sampler_dropdown,
                seed_input,
                *lora_checks,
                *lora_dropdowns,
                *lora_scales,
                pag_enabled_checkbox,
                pag_scale_slider,
                pag_applied_layers_input,
                # Nouveaux outputs pour les états et boutons d'amélioration
                original_user_prompt_state,
                current_prompt_is_enhanced_state,
                enhancement_cycle_active_state,
                last_ai_enhanced_output_state, # Ajouté ici
                enhance_or_redo_button,
                validate_prompt_button,
                message_chargement
            ]


            delete_inputs = [
                preset_refresh_trigger,
                current_preset_page_state,
                preset_search_input,
                preset_sort_dropdown,
                preset_filter_model,
                preset_filter_sampler,
                preset_filter_lora,
                preset_search_input
            ]
            delete_outputs = [
                preset_refresh_trigger,
                pagination_dd_output,
                preset_search_input
            ]

            if not presets_for_page:
                gr.Markdown(f"*{translate('aucun_preset_trouve', translations)}*", key="no_presets_found_md")
            else:
                num_rows_for_page = math.ceil(len(presets_for_page) / PRESET_COLS_PER_ROW)
                preset_idx_on_page = 0
                for r in range(num_rows_for_page):
                    with gr.Row(equal_height=False):
                        for c in range(PRESET_COLS_PER_ROW):
                            if preset_idx_on_page < len(presets_for_page):
                                preset_data = presets_for_page[preset_idx_on_page]
                                preset_id = safe_get_from_row(preset_data, "id", f"ERREUR_ID_{preset_idx_on_page}")
                                preset_name = safe_get_from_row(preset_data, "name", "ERREUR_NOM")

                                with gr.Column(scale=0, min_width=200):
                                    image_bytes = safe_get_from_row(preset_data, "preview_image")
                                    preview_img = None
                                    if image_bytes:
                                        try: preview_img = Image.open(BytesIO(image_bytes))
                                        except Exception: pass
                                    gr.Image(value=preview_img, height=128, width=128, show_label=True, interactive=False, show_download_button=False, key=f"preset_img_{preset_id}")

                                    gr.Textbox(value=preset_name, show_label=False, interactive=False, key=f"preset_name_display_{preset_id}")

                                    preset_notes = safe_get_from_row(preset_data, 'notes')
                                    if preset_notes:
                                        with gr.Accordion(translate("voir_notes", translations), open=False):
                                            gr.Markdown(preset_notes, key=f"preset_notes_md_{preset_id}")

                                    rating_value = safe_get_from_row(preset_data, "rating", 0)
                                    rating_comp = gr.Radio(
                                        choices=[str(r) for r in range(1, 6)], value=str(rating_value) if rating_value > 0 else None,
                                        label=translate("evaluation", translations), interactive=True, key=f"preset_rating_{preset_id}"
                                    )

                                    try:
                                        model_name_disp = safe_get_from_row(preset_data, 'model', '?') # Renommé pour éviter conflit
                                        sampler_key_name = safe_get_from_row(preset_data, 'sampler_key', '?')
                                        # sampler_display_name = translate(sampler_key_name, translations) if sampler_key_name != '?' else '?' # Déjà fait plus haut

                                        details_md = f"- **Modèle:** {model_name_disp}\n- **Sampler:** {sampler_key_name}" # Utiliser sampler_key_name
                                        with gr.Accordion(translate("details_techniques", translations), open=False):
                                            gr.Markdown(details_md, key=f"preset_details_md_{preset_id}")
                                    except Exception: pass

                                    load_btn = gr.Button(translate("charger", translations) + " 💾", size="sm", key=f"preset_load_{preset_id}")
                                    delete_btn = gr.Button(translate("supprimer", translations) + " 🗑️", variant="stop", size="sm", key=f"preset_delete_{preset_id}")

                                    if isinstance(preset_id, int):
                                        load_btn.click(
                                            fn=partial(handle_preset_load_click, preset_id),
                                            inputs=[],
                                            outputs=gen_ui_outputs_for_preset_load
                                        )
                                        delete_btn.click(
                                            fn=partial(handle_preset_delete_click, preset_id),
                                            inputs=delete_inputs,
                                            outputs=delete_outputs
                                        )
                                        rating_comp.change(fn=partial(handle_preset_rating_change, preset_id), inputs=[rating_comp], outputs=[])
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
                    type="pil",
                    sources=["upload", "clipboard"],
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
                inpainting_image_slider = gr.ImageSlider(label=translate("comparaison_inpainting", translations), interactive=False)
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
        inputs=[modele_dropdown, vae_dropdown, pag_enabled_checkbox],
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
            pag_enabled_checkbox,
            pag_scale_slider,
            pag_applied_layers_input,
            # Nouveaux états pour l'amélioration du prompt
            original_user_prompt_state,
            current_prompt_is_enhanced_state,
            enhancement_cycle_active_state,
            *lora_checks,
            *lora_dropdowns,
            *lora_scales
        ],
        outputs=[
            # text_input n'est plus un output direct de generate_image
            image_output,
            seed_output,
            time_output,
            html_output, # Cet output n'est plus utilisé par generate_image pour le prompt
            preview_image_output,
            progress_html_output,
            bouton_charger,
            btn_generate,
            bouton_save_current_preset,
            last_successful_generation_data,
            last_successful_preview_image
        ]
    )
    btn_stop.click(stop_generation, outputs=time_output)
    btn_stop_after_gen.click(stop_generation_process, outputs=time_output)


    bouton_save_current_preset.click(
        fn=handle_save_preset,
        inputs=[
            preset_name_input,
            preset_notes_input,
            last_successful_generation_data,
            last_successful_preview_image,
            preset_refresh_trigger,
            preset_search_input
        ],
        outputs=[
            preset_name_input,
            preset_notes_input,
            preset_filter_model,
            preset_filter_sampler,
            preset_filter_lora,
            preset_refresh_trigger,
            preset_search_input
        ]
    )
    batch_runner_inputs = [
        batch_json_file_input,
        gr.State(config),
        gr.State(translations),
        gr.State(device),
        batch_status_output,
        batch_progress_output,
        batch_gallery_output,
        batch_run_button,
        batch_stop_button
    ]

    batch_runner_outputs = [
        batch_status_output,
        batch_progress_output,
        batch_gallery_output,
        batch_run_button,
        batch_stop_button
    ]

    def batch_runner_wrapper(*args, progress=gr.Progress(track_tqdm=True)):
        """Fonction wrapper qui appelle run_batch_from_json avec yield from."""
        yield from run_batch_from_json(
            model_manager,
            stop_event,
            *args,
            progress=progress
        )

    batch_run_button.click(
        fn=batch_runner_wrapper,
        inputs=batch_runner_inputs,
        outputs=batch_runner_outputs
    )

    batch_stop_button.click(
        fn=stop_generation,
        inputs=None,
        outputs=None
    )

    # La liaison pour module_checkbox est déjà dans la boucle de création des modules.

    def handle_sampler_change(selected_display_name):
        """Gère le changement de sampler dans l'UI principale."""
        global global_selected_sampler_key

        sampler_key = get_sampler_key_from_display_name(selected_display_name, translations)
        if sampler_key and model_manager.get_current_pipe() is not None:
            message, success = apply_sampler_to_pipe(model_manager.get_current_pipe(), sampler_key, translations)
            if success:
                global_selected_sampler_key = sampler_key
                gr.Info(message, 3.0)
            else:
                gr.Warning(message, 4.0)
            return message
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
        outputs=[message_chargement]
    )

    def toggle_pag_scale_visibility_main(pag_enabled):
        return gr.update(visible=pag_enabled), gr.update(visible=pag_enabled)

    pag_enabled_checkbox.change(
        fn=toggle_pag_scale_visibility_main,
        inputs=[pag_enabled_checkbox],
        outputs=[pag_scale_slider, pag_applied_layers_input]
    )

    # Liaisons pour la nouvelle logique d'amélioration du prompt
    enhance_or_redo_button.click(
        fn=on_enhance_or_redo_button_click,
        inputs=[text_input, original_user_prompt_state, enhancement_cycle_active_state, gr.State(LLM_PROMPTER_MODEL_PATH), gr.State(translations)],
        outputs=[text_input, enhance_or_redo_button, validate_prompt_button, original_user_prompt_state, current_prompt_is_enhanced_state, enhancement_cycle_active_state, last_ai_enhanced_output_state]
    )

    validate_prompt_button.click(
        fn=on_validate_prompt_button_click,
        inputs=[text_input, gr.State(translations)],
        outputs=[enhance_or_redo_button, validate_prompt_button, original_user_prompt_state, current_prompt_is_enhanced_state, enhancement_cycle_active_state, last_ai_enhanced_output_state]
    )

    # Utiliser .input() pour réagir à chaque frappe et .submit() pour la soumission finale
    # La fonction handle_text_input_change gère maintenant l'interactivité du bouton et la réinitialisation du cycle.
    text_input.input( # Réagit à chaque modification du texte
        fn=handle_text_input_change,
        inputs=[text_input, last_ai_enhanced_output_state, enhancement_cycle_active_state, gr.State(LLM_PROMPTER_MODEL_PATH), gr.State(translations)],
        outputs=[
            enhance_or_redo_button,
            validate_prompt_button,
            original_user_prompt_state,
            current_prompt_is_enhanced_state,
            enhancement_cycle_active_state,
            last_ai_enhanced_output_state
        ]
    )
    text_input.submit( # Réagit à la soumission (Entrée)
        fn=handle_text_input_change,
        inputs=[text_input, last_ai_enhanced_output_state, enhancement_cycle_active_state, gr.State(LLM_PROMPTER_MODEL_PATH), gr.State(translations)],
        outputs=[
            enhance_or_redo_button,
            validate_prompt_button,
            original_user_prompt_state,
            current_prompt_is_enhanced_state,
            enhancement_cycle_active_state,
            last_ai_enhanced_output_state
        ]
    )

    image_mask_input.change(
        fn=handle_image_mask_interaction,
        inputs=[image_mask_input, original_editor_background_props_state],
        outputs=[validated_image_state, original_editor_background_props_state]
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

with interface:

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
        current_preset_page_state,
        preset_search_input, preset_sort_dropdown, preset_filter_model,
        preset_filter_sampler, preset_filter_lora
    ]

    pagination_dd_output = preset_page_dropdown

    def update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras):
        """Met à jour le Dropdown de pagination."""
        all_presets_data = preset_manager.load_presets_for_display(
            preset_type='gen_image', search_term=search, sort_by=sort,
            selected_models=filter_models or None, selected_samplers=filter_samplers or None, selected_loras=filter_loras or None
        )
        total_presets = len(all_presets_data)
        total_pages = math.ceil(total_presets / PRESETS_PER_PAGE) if total_presets > 0 else 1
        current_page = max(1, min(page, total_pages))


        page_choices = list(range(1, total_pages + 1))

        return gr.update(
            choices=page_choices,
            value=current_page,
            interactive=(total_pages > 1)
        )


    def reset_page_and_update_pagination(*args):
        pagination_updates = update_pagination_display(1, *args[1:])
        return [gr.update(value=1)] + list(pagination_updates)
    def reset_page_and_update_pagination_dd(*args):
        pagination_dd_update = update_pagination_display(1, *args[1:])
        return gr.update(value=1), pagination_dd_update


    preset_search_input.input(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, pagination_dd_output]
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

    def init_inpainting_outputs():
        initial_editor_val_for_load = gr.update()
        initial_validated_img = None
        initial_original_bg_props = None
        initial_slider_val = [None, None]
        return initial_editor_val_for_load, initial_validated_img, initial_original_bg_props, initial_slider_val


    def refresh_trigger_and_update_pagination_dd(current_page, trigger, *filter_args):
        pagination_dd_update = update_pagination_display(current_page, *filter_args)
        initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val = init_inpainting_outputs()

        return gr.update(value=trigger + 1), pagination_dd_update, initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val


    preset_page_dropdown.change(
        fn=handle_page_dropdown_change,
        inputs=[preset_page_dropdown],
        outputs=[current_preset_page_state]
    )

    def initial_load_update_pagination_dd(*filter_args):
        pagination_dd_update = update_pagination_display(1, *filter_args)
        initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val = init_inpainting_outputs()
        return (
            gr.update(value=1), pagination_dd_update,
            initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val
        )

    interface.load(
        fn=initial_load_update_pagination_dd,
        inputs=pagination_dd_inputs[1:],
        outputs=[current_preset_page_state, pagination_dd_output,
                 image_mask_input, validated_image_state, original_editor_background_props_state,
                 inpainting_image_slider]
    )

    # --- Gestion des statistiques mémoire ---
    # Affichage initial statique des statistiques mémoire au chargement
    interface.load(
        fn=lambda: update_memory_stats(translations, model_manager.device), # Utilise la fonction de gest_mem
        outputs=[all_stats_html_component]
    )

    # Générateur pour streamer les mises à jour des statistiques mémoire
    def stream_live_memory_stats(enable_live, update_interval, translations_state, device_state):
        if not enable_live:
            # Si la mise à jour en direct est désactivée, on peut s'assurer que les stats sont à jour une dernière fois
            # ou simplement ne rien faire si elles sont déjà statiques.
            # Gradio arrêtera ce générateur si 'enable_live' devient False.
            # La valeur statique est déjà affichée par interface.load ou la dernière MàJ.
            yield update_memory_stats(translations_state, device_state) # Assure un dernier affichage statique
            return

        last_update_time = 0
        while True: # Gradio gère l'arrêt de cette boucle lorsque les entrées changent ou que l'interface se ferme
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                yield update_memory_stats(translations_state, device_state) # Utiliser la fonction importée
                last_update_time = current_time
            time.sleep(0.2) # Petit temps de pause pour rendre la boucle non bloquante et permettre à Gradio de l'interrompre

    # Connecter le générateur aux contrôles de l'interface utilisateur
    # Lorsque la case à cocher ou le curseur change, le générateur `stream_live_memory_stats` est (re)lancé.
    inputs_for_memory_stream = [
        enable_live_memory_stats_checkbox,
        memory_stats_update_interval_slider,
        gr.State(translations),
        gr.State(model_manager.device)
    ]
    enable_live_memory_stats_checkbox.change(fn=stream_live_memory_stats, inputs=inputs_for_memory_stream, outputs=[all_stats_html_component])
    memory_stats_update_interval_slider.change(fn=stream_live_memory_stats, inputs=inputs_for_memory_stream, outputs=[all_stats_html_component])

def afficher_logo_ascii():
    logo = """
                                                                                 
                                                                      ███                    ███ 
                                                                ███                           ██ 
████████   ██     ██   ██████  ████████    ██████   ██ █████   ██████ ██    ██████               
████   ███  ██   ███ ████   █  ███   ███   █   ███  ████   ███  ███   ██  ███   ███  ██████   ██ 
███     ██  ███ ███  ██        ██     ██   ████████ ███     ██  ███   ██  ██     ███      ██  ██ 
███     ██   █████   ███       ██     ██  ███   ███ ███     ██  ███   ██  ██     ███  ██████  ██ 
██████████    ████    ████████ ██     ██  █████████ █████████    ████ ███  █████████ ██   ██  ██ 
███ ███       ███       ████   ██     ██    ███ ██  ███ ███       ███ ██     ███ ███ ███████  ██ 
███          ███                                    ███                          ███             
███         ███                                     ███                          ██              


    """

    print(txt_color(logo, "warning"))

afficher_logo_ascii()

print(txt_color("[INFO]", "info"), f"{translate('gradio_version_log', translations)}: {gr.__version__}")
interface.launch(inbrowser=str_to_bool(OPEN_BROWSER), pwa=True, share=str_to_bool(SHARE), allowed_paths=[SAVE_DIR])

