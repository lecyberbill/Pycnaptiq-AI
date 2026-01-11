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
from Utils.callback_diffuser import latents_to_rgb, create_callback_on_step_end, create_inpainting_callback 
from Utils.model_manager import ModelManager
from core.translator import translate_prompt
from core.Inpaint import apply_mask_effects
from core.image_prompter import init_image_prompter, generate_prompt_from_image, MODEL_ID_FLORENCE2 as DEFAULT_FLORENCE2_MODEL_ID_FROM_PROMPTER, DEFAULT_FLORENCE2_TASK
from Utils import llm_prompter_util
from Utils.utils import GestionModule, enregistrer_etiquettes_image_html, finalize_html_report_if_needed, gradio_change_theme, lister_fichiers, styles_fusion, create_progress_bar_html, load_modules_js, \
    telechargement_modele, txt_color, str_to_bool, translate, get_language_options, enregistrer_image, preparer_metadonnees_image, check_gpu_availability, ImageSDXLchecker
from Utils.sampler_utils import get_sampler_choices, get_sampler_key_from_display_name, apply_sampler_to_pipe 
from core.config import (
    config, translations, DEFAULT_LANGUAGE, MODELS_DIR, VAE_DIR, LORAS_DIR, 
    INPAINT_MODELS_DIR, SAVE_DIR, SAVE_BATCH_JSON_PATH, IMAGE_FORMAT, RAW_FORMATS as FORMATS, 
    NEGATIVE_PROMPT, GRADIO_THEME, AUTHOR, SHARE, OPEN_BROWSER, DEFAULT_MODEL, 
    PRESETS_PER_PAGE, PRESET_COLS_PER_ROW, STYLES, device, torch_dtype, vram_total_gb, 
    LLM_PROMPTER_MODEL_PATH, FLORENCE2_MODEL_ID_CONFIG, APP_VERSION, APP_VERSION_DATE_STR, print_config_summary, PREVIEW_QUEUE
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process
torch.backends.cudnn.deterministic = True
import queue
from core.sdxl_logic import generate_image, generate_inpainted_image
from core.inpainting_utils import handle_image_mask_interaction, create_opaque_mask_from_editor
from core.model_loaders import handle_model_selection, handle_inpainting_model_selection, get_model_list_updates, get_inpainting_model_list_updates, check_and_download_default_models
from core.ui_handlers import (
    handle_sampler_change_logic, batch_runner_wrapper_logic, toggle_pag_scale_visibility_logic, 
    generate_prompt_ui_logic, stream_live_memory_stats_logic, 
    generate_image_ui_wrapper, generate_inpainted_image_ui_wrapper, handle_save_preset_ui_wrapper
)
from Utils.preset_handlers import (
    handle_preset_rename_click, handle_preset_cancel_click, handle_preset_rename_submit,
    handle_preset_delete_click, handle_preset_rating_change, update_pagination_and_trigger_refresh,
    handle_page_change, update_filter_choices_after_save,
    handle_preset_load_click, reset_page_state_only, handle_page_dropdown_change,
    get_filter_options, update_pagination_display, render_presets_with_decorator
)
from Utils.gest_mem import create_memory_accordion_ui, update_memory_stats
from compel import Compel, ReturnedEmbeddingsType
from io import BytesIO
from presets.presets_Manager import PresetManager
import functools
from functools import partial # Keep functools import
# --- Initialisation des Managers ---
preset_manager = PresetManager(translations)
model_manager = ModelManager(config, translations, device, torch_dtype, vram_total_gb)
gestionnaire = GestionModule(
    translations=translations, config=config,
    language=DEFAULT_LANGUAGE,
    model_manager_instance=model_manager,
    preset_manager_instance=preset_manager
)

# --- Initialisation des modèles (Vérification/Téléchargement) ---
modeles_disponibles, modeles_impaint = check_and_download_default_models(model_manager, translations, MODELS_DIR, INPAINT_MODELS_DIR)
vaes = model_manager.list_vaes() # Inclut "Auto"

# --- Lister les LORAS au démarrage ---
initial_lora_choices = model_manager.list_loras(gradio_mode=True)
has_initial_loras = bool(initial_lora_choices) and translate("aucun_modele_trouve", translations) not in initial_lora_choices and translate("repertoire_not_found", translations) not in initial_lora_choices
lora_initial_dropdown_choices = initial_lora_choices if has_initial_loras else [translate("aucun_lora_disponible", translations)]
initial_lora_message = translate("lora_trouve", translations) + ", ".join(initial_lora_choices) if has_initial_loras else translate("aucun_lora_disponible", translations)


# Logic moved to core/model_loaders.py and core/config.py
print( txt_color(translate('Safety', translations), 'erreur'))

 # Créer un pool de threads pour l'écriture asynchrone

html_executor = ThreadPoolExecutor(max_workers=5)
image_executor = ThreadPoolExecutor(max_workers=10) 

# --- AJOUT: Logique de l'historique de session ---
APP_START_TIME = time.time()

def get_session_images():
    """Récupère les images générées pendant la session actuelle."""
    images = []
    formats = ('.jpg', '.jpeg', '.png', '.webp')
    
    if not os.path.exists(SAVE_DIR):
        return []

    # Parcours récursif pour trouver toutes les images
    for root, _, files in os.walk(SAVE_DIR):
        for file in files:
            if file.lower().endswith(formats):
                file_path = os.path.join(root, file)
                try:
                    mtime = os.path.getmtime(file_path)
                    if mtime > APP_START_TIME:
                        images.append((file_path, f"Image - {datetime.fromtimestamp(mtime).strftime('%H:%M:%S')}"))
                except OSError:
                    continue
    
    # Trier par date de modification (la plus récente en premier)
    images.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    return images

def clear_session_history_logic():
    """Réinitialise l'heure de début de session pour 'vider' l'historique visuellement."""
    global APP_START_TIME
    APP_START_TIME = time.time()
    return []
# --- FIN AJOUT ---

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
    return generate_prompt_ui_logic(image, current_translations, FLORENCE2_MODEL_ID_CONFIG)
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

def update_globals_model_ui_wrapper(nom_fichier, nom_vae, pag_is_enabled):
    global model_selectionne, vae_selctionne, loras_charges
    msg, upd_int, upd_txt, new_model, new_vae = handle_model_selection(
        nom_fichier, nom_vae, pag_is_enabled, model_manager, translations
    )
    if new_model:
        model_selectionne = new_model
        vae_selctionne = new_vae
        loras_charges.clear()
    else:
        # Fallback for UI dropdown state if loading failed? 
        # Usually ModelManager already unloads so we just follow its lead.
        model_selectionne = None
        model_selectionne = None
    
    return msg, upd_int, upd_txt

def update_globals_model_inpainting_ui_wrapper(nom_fichier):
    global model_selectionne
    msg, upd_int, upd_txt, new_model = handle_inpainting_model_selection(
        nom_fichier, model_manager, translations
    )
    if new_model:
        model_selectionne = new_model
    else:
        model_selectionne = None
        
    return msg, upd_int, upd_txt

#==========================
# Outils pour gérer les iamges du module Inpainting
#==========================


# Logic moved to core/inpainting_utils.py and Utils/preset_handlers.py


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


# UI wrappers moved to core/ui_handlers.py


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

    gr.Markdown(f"# Pycnaptiq-AI {APP_VERSION}", elem_id="main_title_markdown") # Added elem_id for potential future use

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
                format_choices = [f"{fmt['dimensions']} : {translate(fmt['orientation'], translations)}" for fmt in FORMATS]
                format_dropdown = gr.Dropdown(choices=format_choices, value=format_choices[3] if len(format_choices) > 3 else format_choices[0], label=translate("format", translations))
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
                
                # --- AJOUT: Accordéon Upscaling par Tiles ---
                with gr.Accordion(translate("upscale_tiles_label", translations), open=False):
                    upscale_tiles_enable = gr.Checkbox(label=translate("upscale_tiles_enable", translations), value=False)
                    with gr.Group(visible=False) as upscale_tiles_group:
                        upscale_tiles_factor = gr.Slider(1.0, 4.0, value=2.0, step=0.1, label=translate("upscale_tiles_factor", translations))
                        upscale_tiles_size = gr.Slider(512, 1024, value=768, step=128, label=translate("upscale_tiles_size", translations))
                        upscale_tiles_overlap = gr.Slider(0, 256, value=64, step=8, label=translate("upscale_tiles_overlap", translations))
                        upscale_tiles_denoising = gr.Slider(0.05, 0.8, value=0.35, step=0.05, label=translate("upscale_tiles_denoising", translations))
                    
                    upscale_tiles_enable.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[upscale_tiles_enable],
                        outputs=[upscale_tiles_group]
                    )

                # --- AJOUT: Accordéon ControlNet & Adapters ---
                with gr.Accordion(translate("controlnet_adapters_label", translations), open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(f"### {translate('controlnet_label', translations)}")
                            controlnet_image_input = gr.Image(label=translate("image_reference_pose", translations), type="pil")
                            controlnet_model_dropdown = gr.Dropdown(
                                label=translate("modele_controlnet", translations),
                                choices=model_manager.list_controlnets() + [translate("aucun", translations)],
                                value=translate("aucun", translations)
                            )
                            controlnet_scale_slider = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label=translate("strength_controlnet", translations))
                            
                            # --- AJOUT: Éditeur OpenPose Interactif (via Module) ---
                            openpose_mod = gestionnaire.modules.get("openpose_editor")
                            if openpose_mod and openpose_mod.is_active:
                                openpose_mod.instance.create_ui(controlnet_image_input)
                        with gr.Column():
                            gr.Markdown(f"### {translate('ip_adapter_label', translations)}")
                            ip_adapter_image_input = gr.Image(label=translate("image_reference_style", translations), type="pil")
                            ip_adapter_model_dropdown = gr.Dropdown(
                                label=translate("modele_ip_adapter", translations),
                                choices=model_manager.list_ip_adapters() + [translate("aucun", translations)],
                                value=translate("aucun", translations)
                            )
                            ip_adapter_scale_slider = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label=translate("strength_ip_adapter", translations))

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
# mettre_a_jour_listes moved to core/model_loaders.py


############################################################
########TAB PRESETS
############################################################


    with gr.Tab(translate("Preset", translations)) as preset_tab:
        initial_models, initial_samplers, initial_loras = get_filter_options(preset_manager, translations)
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

        # --- Definitions for re-rendering and pagination ---
        render_page_inputs = [
            current_preset_page_state,
            preset_search_input,
            preset_sort_dropdown,
            preset_filter_model,
            preset_filter_sampler,
            preset_filter_lora,
            preset_refresh_trigger
        ]
        pagination_dd_inputs = render_page_inputs[:-1] # Inputs for updating the dropdown, without the trigger

        gen_ui_outputs_for_preset_load = [
            modele_dropdown, vae_dropdown, text_input, style_dropdown, guidance_slider,
            num_steps_slider, format_dropdown, sampler_dropdown, seed_input,
            *lora_checks, *lora_dropdowns, *lora_scales,
            pag_enabled_checkbox, pag_scale_slider, pag_applied_layers_input,
            original_user_prompt_state, current_prompt_is_enhanced_state,
            enhancement_cycle_active_state, last_ai_enhanced_output_state,
            enhance_or_redo_button, validate_prompt_button, message_chargement
        ]

        delete_inputs = [
            preset_refresh_trigger, current_preset_page_state, preset_search_input,
            preset_sort_dropdown, preset_filter_model, preset_filter_sampler,
            preset_filter_lora, preset_search_input
        ]
        delete_outputs = [
            preset_refresh_trigger, preset_page_dropdown, preset_search_input
        ]

        @gr.render(inputs=render_page_inputs)
        def render_presets_wrapper(*args):
             render_presets_with_decorator(
                 *args,
                 preset_manager=preset_manager,
                 model_manager=model_manager,
                 translations=translations,
                 config=config,
                 STYLES=STYLES,
                  FORMATS=format_choices,
                PRESETS_PER_PAGE=PRESETS_PER_PAGE,
                 PRESET_COLS_PER_ROW=PRESET_COLS_PER_ROW,
                 gen_ui_outputs_for_preset_load=gen_ui_outputs_for_preset_load,
                 delete_inputs=delete_inputs,
                 delete_outputs=delete_outputs
             )

############################################################
########TAB INPAINTING
############################################################
# mettre_a_jour_listes_inpainting moved to core/model_loaders.py
# Logic moved to core/model_loaders.py

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
                initial_inpaint_value = modeles_impaint[0] if modeles_impaint and modeles_impaint[0] != translate("aucun_modele_trouve", translations) else None
                modele_inpainting_dropdown = gr.Dropdown(label=translate("selectionner_modele_inpainting", translations), choices=modeles_impaint, value=initial_inpaint_value, allow_custom_value=True)
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
        fn=lambda: get_model_list_updates(MODELS_DIR, model_manager, translations),
        outputs=[modele_dropdown, vae_dropdown, *lora_dropdowns, lora_message]
    )

    bouton_charger.click(
        fn=update_globals_model_ui_wrapper,
        inputs=[modele_dropdown, vae_dropdown, pag_enabled_checkbox],
        outputs=[message_chargement, btn_generate, btn_generate]
    )

    image_input.change(
        fn=generate_prompt_wrapper,
        inputs=[image_input, gr.State(translations)],
        outputs=text_input)

    btn_generate.click(
        fn=partial(generate_image_ui_wrapper, model_manager, translations, config, device, stop_event, stop_gen, SAVE_DIR, AUTHOR, IMAGE_FORMAT, image_executor, html_executor, STYLES, NEGATIVE_PROMPT, PREVIEW_QUEUE),
        inputs=[
            text_input, style_dropdown, guidance_slider, num_steps_slider, format_dropdown, traduire_checkbox, seed_input, num_images_slider,
            pag_enabled_checkbox, pag_scale_slider, pag_applied_layers_input,
            original_user_prompt_state, current_prompt_is_enhanced_state, enhancement_cycle_active_state,
            controlnet_image_input, controlnet_model_dropdown, controlnet_scale_slider, ip_adapter_image_input, ip_adapter_scale_slider,
            ip_adapter_model_dropdown,
            upscale_tiles_enable, upscale_tiles_factor, upscale_tiles_size, upscale_tiles_overlap, upscale_tiles_denoising,
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
        fn=partial(handle_save_preset_ui_wrapper, preset_manager, translations),
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

    def batch_runner_wrapper_ui_bridge(*args, progress=gr.Progress(track_tqdm=True)):
        yield from batch_runner_wrapper_logic(model_manager, stop_event, *args, progress=progress)

    batch_run_button.click(
        fn=batch_runner_wrapper_ui_bridge,
        inputs=batch_runner_inputs,
        outputs=batch_runner_outputs
    )

    batch_stop_button.click(
        fn=stop_generation,
        inputs=None,
        outputs=None
    )

    # La liaison pour module_checkbox est déjà dans la boucle de création des modules.

    def handle_sampler_change_ui_bridge(selected_display_name):
        global global_selected_sampler_key
        message, success, sampler_key = handle_sampler_change_logic(selected_display_name, model_manager, translations)
        if success:
            global_selected_sampler_key = sampler_key
        return message

    sampler_dropdown.change(
        fn=handle_sampler_change_ui_bridge,
        inputs=sampler_dropdown,
        outputs=[message_chargement]
    )

    pag_enabled_checkbox.change(
        fn=toggle_pag_scale_visibility_logic,
        inputs=[pag_enabled_checkbox],
        outputs=[pag_scale_slider, pag_applied_layers_input]
    )

    # Liaisons pour la nouvelle logique d'amélioration du prompt
    enhance_or_redo_button.click(
        fn=llm_prompter_util.on_enhance_or_redo_button_click,
        inputs=[text_input, original_user_prompt_state, enhancement_cycle_active_state, gr.State(LLM_PROMPTER_MODEL_PATH), gr.State(translations)],
        outputs=[text_input, enhance_or_redo_button, validate_prompt_button, original_user_prompt_state, current_prompt_is_enhanced_state, enhancement_cycle_active_state, last_ai_enhanced_output_state]
    )


    validate_prompt_button.click(
        fn=llm_prompter_util.on_validate_prompt_button_click,
        inputs=[text_input, gr.State(translations)],
        outputs=[enhance_or_redo_button, validate_prompt_button, original_user_prompt_state, current_prompt_is_enhanced_state, enhancement_cycle_active_state, last_ai_enhanced_output_state]
    )


    # Note: L'ouverture/fermeture du container se fait côté JS pour plus de réactivité
    # via un listener sur openpose_editor_btn (géré dans le module)
    
    # Utiliser .input() pour réagir à chaque frappe et .submit() pour la soumission finale
    # La fonction handle_text_input_change gère maintenant l'interactivité du bouton et la réinitialisation du cycle.
    text_input.input( # Réagit à chaque modification du texte
        fn=llm_prompter_util.handle_text_input_change,
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
        fn=llm_prompter_util.handle_text_input_change,
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
        fn=partial(handle_image_mask_interaction, translations=translations),
        inputs=[image_mask_input, original_editor_background_props_state],
        outputs=[validated_image_state, original_editor_background_props_state]
    )


    bouton_stop_inpainting.click(
        fn=stop_generation_process,
        outputs=message_inpainting
    )

    bouton_lister_inpainting.click(
        fn=lambda: get_inpainting_model_list_updates(INPAINT_MODELS_DIR, translations),
        outputs=modele_inpainting_dropdown
    )

    bouton_charger_inpainting.click(
        fn=update_globals_model_inpainting_ui_wrapper,
        inputs=[modele_inpainting_dropdown],
        outputs=[message_chargement_inpainting, bouton_generate_inpainting, bouton_generate_inpainting]
    )

    bouton_generate_inpainting.click(
        fn=partial(generate_inpainted_image_ui_wrapper, model_manager, translations, stop_gen, SAVE_DIR, AUTHOR, IMAGE_FORMAT, image_executor, html_executor),
        inputs=[
            inpainting_prompt,
            validated_image_state,
            image_mask_input, # 1: tuple (image_base, mask)
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

    # Redundant definitions moved to line 544

    # update_pagination_display moved to Utils/preset_handlers.py


    def reset_page_and_update_pagination(*args):
        pagination_updates = update_pagination_display(1, args[1], args[2], args[3], args[4], args[5], preset_manager, PRESETS_PER_PAGE)
        return [gr.update(value=1)] + list(pagination_updates)
    def reset_page_and_update_pagination_dd(*args):
        pagination_dd_update = update_pagination_display(1, args[1], args[2], args[3], args[4], args[5], preset_manager, PRESETS_PER_PAGE)
        return gr.update(value=1), pagination_dd_update


    preset_search_input.input(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, preset_page_dropdown]
    )


    preset_sort_dropdown.change(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, preset_page_dropdown]
    )
    preset_filter_model.change(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, preset_page_dropdown]
    )
    preset_filter_sampler.change(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, preset_page_dropdown]
    )
    preset_filter_lora.change(
        fn=reset_page_and_update_pagination_dd,
        inputs=pagination_dd_inputs,
        outputs=[current_preset_page_state, preset_page_dropdown]
    )

    def init_inpainting_outputs():
        initial_editor_val_for_load = gr.update()
        initial_validated_img = None
        initial_original_bg_props = None
        initial_slider_val = [None, None]
        return initial_editor_val_for_load, initial_validated_img, initial_original_bg_props, initial_slider_val


    def refresh_trigger_and_update_pagination_dd(current_page, trigger, *filter_args):
        pagination_dd_update = update_pagination_display(current_page, filter_args[0], filter_args[1], filter_args[2], filter_args[3], filter_args[4], preset_manager, PRESETS_PER_PAGE)
        initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val = init_inpainting_outputs()

        return gr.update(value=trigger + 1), pagination_dd_update, initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val


    preset_page_dropdown.change(
        fn=handle_page_dropdown_change,
        inputs=[preset_page_dropdown, gr.State(translations)],
        outputs=[current_preset_page_state]
    )

    def initial_load_update_pagination_dd(*filter_args):
        pagination_dd_update = update_pagination_display(1, *filter_args, preset_manager, PRESETS_PER_PAGE)
        initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val = init_inpainting_outputs()
        return (
            gr.update(value=1), pagination_dd_update,
            initial_im_mask_val, initial_validated_img, initial_orig_props, initial_slider_val
        )

    interface.load(
        fn=initial_load_update_pagination_dd,
        inputs=pagination_dd_inputs[1:],
        outputs=[current_preset_page_state, preset_page_dropdown,
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
    # stream_live_memory_stats logic moved to core/ui_handlers.py

    # Connecter le générateur aux contrôles de l'interface utilisateur
    # Lorsque la case à cocher ou le curseur change, le générateur `stream_live_memory_stats_logic` est (re)lancé.
    inputs_for_memory_stream = [
        enable_live_memory_stats_checkbox,
        memory_stats_update_interval_slider,
        gr.State(translations),
        gr.State(model_manager.device)
    ]
    enable_live_memory_stats_checkbox.change(fn=stream_live_memory_stats_logic, inputs=inputs_for_memory_stream, outputs=[all_stats_html_component])
    memory_stats_update_interval_slider.change(fn=stream_live_memory_stats_logic, inputs=inputs_for_memory_stream, outputs=[all_stats_html_component])

    # --- AJOUT: UI de l'historique de session ---
    with gr.Accordion(label=translate("session_history_label", translations), open=False):
        with gr.Row():
            btn_refresh_history = gr.Button(translate("refresh_history_btn", translations), variant="secondary")
            btn_clear_history = gr.Button(translate("clear_session_btn", translations), variant="stop")
        
        session_gallery = gr.Gallery(
            label=None, 
            columns=4, 
            rows=2, 
            height="auto",
            object_fit="contain",
            interactive=False,
            elem_id="session_history_gallery"
        )
        
        # Timer pour la mise à jour automatique (toutes les 10 secondes)
        history_timer = gr.Timer(10)
        history_timer.tick(fn=get_session_images, outputs=session_gallery)
        
        btn_refresh_history.click(fn=get_session_images, outputs=session_gallery)
        btn_clear_history.click(fn=clear_session_history_logic, outputs=session_gallery)

    # Chargement initial de l'historique
    interface.load(fn=get_session_images, outputs=session_gallery)

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

