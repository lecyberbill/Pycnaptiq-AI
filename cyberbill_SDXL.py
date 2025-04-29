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
from gradio import update
from diffusers import  StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, \
    LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, KDPM2DiscreteScheduler, StableDiffusionXLInpaintPipeline, \
    KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, DPMSolverSDEScheduler, DPMSolverSinglestepScheduler
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from core.version import version
from Utils.callback_diffuser import latents_to_rgb, create_callback_on_step_end, create_inpainting_callback
from core.trannslator import translate_prompt
from core.Inpaint import apply_mask_effects
from Utils.model_loader import charger_modele, charger_modele_inpainting, charger_lora, decharge_lora, gerer_lora
from Utils.utils import enregistrer_etiquettes_image_html,charger_configuration, gradio_change_theme, lister_fichiers, GestionModule, styles_fusion, create_progress_bar_html,\
    telechargement_modele, txt_color, str_to_bool, load_locales, translate, get_language_options, enregistrer_image, preparer_metadonnees_image, check_gpu_availability, decharger_modele, ImageSDXLchecker
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


# Check if photo_editing_mod is loaded
#photo_editing_loaded = "photo_editing" in gestionnaire.get_loaded_modules()
#initialisation des modèles

modeles_disponibles = lister_fichiers(MODELS_DIR, translations)
vaes = ["Défaut VAE"] + lister_fichiers(VAE_DIR, translations)
modeles_impaint = lister_fichiers(INPAINT_MODELS_DIR, translations)

if not modeles_impaint:
    modeles_impaint = [translate("aucun_modele_trouve", translations)]
    


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

# Call the function to check GPU availability
device, torch_dtype, vram_total_gb = check_gpu_availability(translations)
gestionnaire = GestionModule(
    translations=translations,
    language=DEFAULT_LANGUAGE,
    config=config,
    device=device, # <-- Passer
    torch_dtype=torch_dtype, # <-- Passer
    vram_total_gb=vram_total_gb # <-- Passer
    # global_pipe et global_compel sont initialisés à None par défaut
)

# Initialisation des variables globales

model_selectionne = None
vae_selctionne = "Défaut VAE"
loras_charges = {}
# Charger tous les modules
gestionnaire.charger_tous_les_modules()

# Get the javascript code
js_code = gestionnaire.get_js_code()


# Flag pour arrêter la génération
stop_event = threading.Event()

stop_gen = threading.Event()


# Flag pour signaler qu'une tâche est en cours
processing_event = threading.Event()
# Flag to indicate if an image generation is in progress
is_generating = False

global_selected_sampler_key = "sampler_euler"

# Charger le modèle et le processeur
caption_model = AutoModelForCausalLM.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v2.0", trust_remote_code=True).to(device)
caption_processor = AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v2.0", trust_remote_code=True)
# =========================
# Définition des fonctions
# =========================





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

def generate_caption(image):
    """generate a prompt from an image."""

    if image:
        # Préparer les entrées
        inputs = caption_processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
        print(txt_color("[INFO] ", "info"), translate("prompt_calcul", translations))
        # Générer le texte
        generated_ids = caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = caption_processor.post_process_generation(
            generated_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height)
        )

        # Libérer la mémoire GPU
        torch.cuda.empty_cache()
        prompt = parsed_answer.get('<DETAILED_CAPTION>', '').strip('{}').strip('"')
        print(txt_color("[INFO] ", "info"), translate("prompt_calculé", translations), f"{prompt}")
        return prompt
    return ""

def update_prompt(image):
    """update the prompt"""
    if image:  
        return generate_caption(image)
    return ""

#==========================
# Fonction GENERATION IMAGE
#==========================

def generate_image(text, style_selection, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images, *lora_inputs):
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

        if gestionnaire.global_pipe is None:
            print(txt_color("[ERREUR] ","erreur"), translate("erreur_pas_modele", translations))
            gr.Warning(translate("erreur_pas_modele", translations), 4.0)
            final_message = translate("erreur_pas_modele", translations)
            return

        if not isinstance(gestionnaire.global_pipe, StableDiffusionXLPipeline):
            error_message = translate("erreur_mauvais_type_modele", translations)
            print(txt_color("[ERREUR] ","erreur"), error_message)
            gr.Warning(error_message, 4.0)
            # Mettre à jour les variables finales pour le bloc finally
            final_message = error_message
            final_html_msg = f"<p style='color: red;'>{error_message}</p>"
            return


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

        # --- Utiliser gestionnaire.global_compel pour les prompts positif ET négatif ---
        conditioning, pooled = gestionnaire.global_compel(prompt_en)
        neg_conditioning, neg_pooled = gestionnaire.global_compel(negative_prompt_str)

        selected_format_parts = selected_format.split(":")[0].strip() # Utiliser selected_format ici
        width, height = map(int, selected_format_parts.split("*"))
        images = [] # Initialize a list to store all generated images
        seed_strings = []
        formatted_seeds = ""
        current_data_for_preset = None
        preview_image_for_preset = None

        message_lora = gerer_lora(gestionnaire.global_pipe, loras_charges, lora_checks, lora_dropdowns, lora_scales, LORAS_DIR, translations)
        if message_lora:
            gr.Warning(message_lora, 4.0)
            final_message = message_lora
            return


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

            pipeline_thread, result_container = execute_pipeline_task_async(
                pipe=gestionnaire.global_pipe,
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
                preview_queue=PREVIEW_QUEUE # Passer la queue globale d'aperçu
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

                # Vérifier si l'arrêt global a été demandé pendant la boucle
                if stop_event.is_set():
                    stop_gen.set() # Signaler au thread de s'arrêter via son propre event
                    print(txt_color("[INFO]", "info"), translate("arret_global_detecte", translations))
                    break # Sortir de la boucle while

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
                    bouton_charger_update_off, btn_generate_update_off,
                    gr.update(interactive=False) # Save preset button
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
                    bouton_charger_update_off, btn_generate_update_off,
                    gr.update(interactive=False) # Save preset button
                )
                continue # Passer à l'image suivante

            # Gérer le cas où aucune image n'est retournée malgré le succès
            elif final_image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image_genere", translations))
                gr.Warning(translate("erreur_pas_image_genere", translations), 4.0)
                yield (
                    images, " ".join(seed_strings), translate("erreur_pas_image_genere", translations),
                    translate("erreur_pas_image_genere", translations), None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off,
                    gr.update(interactive=False) # Save preset button
                )
                continue # Passer à l'image suivante

            # --- Si la génération a réussi et une image est présente ---
            # Préparation des données pour le preset
            current_data_for_preset = {
                 "model": model_selectionne,
                 "vae": vae_selctionne,
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
                 "loras": json.dumps([
                     {"name": lora_dropdowns[i], "weight": lora_scales[i]}
                     for i, check in enumerate(lora_checks) if check and lora_dropdowns[i] != translate("aucun_lora_disponible", translations)
                 ]),
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
            lora_info_str = ", ".join(lora_info) if lora_info else translate("aucun_lora", translations)
            style_info_str = ", ".join(selected_style_display_names) if selected_style_display_names else translate("Aucun_style", translations)

            donnees_xmp = {
                 "Module": "SDXL Image Generation", "Creator": AUTHOR,
                 "Model": os.path.splitext(model_selectionne)[0],
                 "VAE": os.path.splitext(vae_selctionne)[0] if vae_selctionne else "Défaut VAE",
                 "Steps": num_steps, "Guidance": guidance_scale,
                 "Sampler": gestionnaire.global_pipe.scheduler.__class__.__name__,
                 "IMAGE": f"{idx+1} {translate('image_sur',translations)} {num_images}",
                 "Inference": num_steps, "Style": style_info_str,
                 "original_prompt (User)": prompt_text, "Prompt": prompt_en,
                 "Negatif Prompt": negative_prompt_str, "Seed": seed,
                 "Size": selected_format_parts, # Utiliser selected_format_parts ici
                 "Loras": lora_info_str,
                 "Generation Time": temps_generation_image
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

def generate_inpainted_image(text, image, mask, num_steps, strength, guidance_scale, traduire):
    """Génère une image inpainted avec Stable Diffusion XL."""
    global  stop_gen

    # --- Définir les états des boutons ---
    btn_gen_inp_off = gr.update(interactive=False)
    btn_load_inp_off = gr.update(interactive=False)
    btn_gen_inp_on = gr.update(interactive=True)
    btn_load_inp_on = gr.update(interactive=True)

    # --- Placeholders pour le premier yield ---
    initial_image = None
    initial_msg_load = "" # Message chargement/HTML
    initial_msg_status = translate("preparation", translations) # Message statut
    initial_progress = "" # Barre de progression

    # --- Yield initial pour désactiver les boutons ---
    yield initial_image, initial_msg_load, initial_msg_status, initial_progress, btn_load_inp_off, btn_gen_inp_off

    # --- Initialiser les variables pour le résultat final ---
    final_image_result = None
    final_msg_load_result = ""
    final_msg_status_result = ""
    final_progress_result = ""

    try:
        start_time = time.time()
        stop_gen.clear()
        if gestionnaire.global_pipe is None:
            msg = translate("erreur_pas_modele_inpainting", translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, 4.0)
            final_msg_status_result = msg
            return

        if not isinstance(gestionnaire.global_pipe, StableDiffusionXLInpaintPipeline):
            error_message = translate("erreur_mauvais_type_modele_inpainting", translations)
            print(txt_color("[ERREUR] ", "erreur"), error_message)
            gr.Warning(error_message, 4.0)
            # Mettre à jour les variables finales pour le bloc finally
            final_msg_status_result = error_message
            final_msg_load_result = f"<p style='color: red;'>{error_message}</p>"
            # Important : retourner immédiatement
            return # Sortir de la fonction ici

        if image is None or mask is None:
            msg = translate("erreur_image_mask_manquant", translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, 4.0)
            final_msg_status_result = msg

            return


        # Vérifications de type (gardées par sécurité)
        if not isinstance(image, Image.Image):
             msg = f"Type d'image invalide reçu pour l'inpainting: {type(image)}"
             print(txt_color("[ERREUR]", "erreur"), msg)
             final_msg_status_result = "Erreur type image"
             return
        if not isinstance(mask, Image.Image):
             msg = f"Type de masque invalide reçu pour l'inpainting: {type(mask)}"
             print(txt_color("[ERREUR]", "erreur"), msg)
             final_msg_status_result = "Erreur type masque"
             return


        actual_total_steps = math.ceil(num_steps * strength)
        if actual_total_steps <= 0: # Sécurité si strength est 0 ou très proche
            actual_total_steps = 1


        # Translate the prompt if requested
        prompt_text = translate_prompt(text, translations) if traduire else text
        conditioning, pooled = gestionnaire.global_compel(prompt_text)
        
        active_adapters = gestionnaire.global_pipe.get_active_adapters()
        for adapter_name in active_adapters:
            gestionnaire.global_pipe.set_adapters(adapter_name, 0)
        
        image_rgb = image.convert("RGB")
        mask_rgb = mask
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
                inpainted_image_result = gestionnaire.global_pipe(
                    pooled_prompt_embeds=pooled,
                    prompt_embeds=conditioning,
                    image=image_rgb,
                    mask_image=mask_rgb,
                    width=image.width,
                    height=image.height,
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
                 if not (hasattr(gestionnaire.global_pipe, '_interrupt') and gestionnaire.global_pipe._interrupt):
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
                yield None, gr.update(), gr.update(), last_progress_html, btn_load_inp_off, btn_gen_inp_off
            time.sleep(0.05)  
        thread.join()

        current_final_progress_html = ""

        if not stop_gen.is_set() and "error" not in final_image_container:
             final_progress_html = create_progress_bar_html(actual_total_steps, actual_total_steps, 100)
        elif "error" in final_image_container:
             final_progress_html = f'<p style="color: red;">{translate("erreur_lors_inpainting", translations)}</p>'
        
        if hasattr(gestionnaire.global_pipe, '_interrupt'):
            gestionnaire.global_pipe._interrupt = False
        
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
                final_image_result = inpainted_image # Image pour le yield final
                current_final_progress_html = create_progress_bar_html(actual_total_steps, actual_total_steps, 100)
       
                temps_generation_image = f"{(time.time() - start_time):.2f} sec"
                date_str = datetime.now().strftime("%Y_%m_%d")
                heure_str = datetime.now().strftime("%H_%M_%S")
                save_dir = os.path.join(SAVE_DIR, date_str)
                os.makedirs(save_dir, exist_ok=True)
                filename = f"inpainting_{date_str}_{heure_str}_{image.width}x{image.height}.{IMAGE_FORMAT.lower()}"
                chemin_image = os.path.join(save_dir, filename)
                donnees_xmp = {
                    "Module": "SDXL Inpainting",
                    "Creator": AUTHOR,
                    "Model": os.path.splitext(model_selectionne)[0],
                    "Steps": num_steps,
                    "Guidance": guidance_scale,
                    "Strength": strength,
                    "Prompt": prompt_text,
                    "Size": f"{image.width}x{image.height}",
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
        if hasattr(gestionnaire.global_pipe, '_interrupt'):
            gestionnaire.global_pipe._interrupt = False
        yield final_image_result, final_msg_load_result, final_msg_status_result, final_progress_result, btn_load_inp_on, btn_gen_inp_on


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


initial_model_value = None
initial_vae_value = None
initial_button_text = translate("charger_modele_pour_commencer", translations)
initial_button_interactive = False
initial_message_chargement = translate("aucun_modele_charge", translations)
message_retour = None 
model_selectionne = None  
vae_selctionne = None 

if DEFAULT_MODEL and os.path.basename(DEFAULT_MODEL) in modeles_disponibles:
    print(f"{txt_color('[INFO]','info')}", f"{DEFAULT_MODEL} {translate('va_se_charger', translations)} {MODELS_DIR}")

    temp_pipe = None # Initialiser à None
    temp_compel = None
    message_retour_list = []
    erreur_chargement = None

    try:
        # 1. Charger dans des variables temporaires
        temp_pipe, temp_compel, *message_retour_list = charger_modele(
            DEFAULT_MODEL, "Défaut VAE", translations, MODELS_DIR, VAE_DIR,
            device, torch_dtype, vram_total_gb, gestionnaire.global_pipe, gestionnaire.global_compel
        )
        message_retour = message_retour_list[0] if message_retour_list else None

    except Exception as e_load:
        traceback.print_exc()
        erreur_chargement = e_load
        message_retour = str(e_load) # Utiliser le message d'erreur



    # 2. Vérifier le succès basé sur temp_pipe ET l'absence d'erreur
    if temp_pipe and not erreur_chargement:
        try:
            # 3. Affecter les globales SEULEMENT si succès
            gestionnaire.global_pipe = temp_pipe
            gestionnaire.global_compel = temp_compel
            model_selectionne = os.path.basename(DEFAULT_MODEL)
            vae_selctionne = "Défaut VAE"

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
            gestionnaire.global_pipe = None
            gestionnaire.global_compel = None
            model_selectionne = None
            vae_selctionne = None

    else:
        # --- Le chargement a ÉCHOUÉ (soit temp_pipe est None, soit erreur_chargement existe) ---
        # Utiliser le message d'erreur ou un message par défaut
        initial_message_chargement = message_retour if message_retour else translate("erreur_chargement_modele_defaut", translations)
        gestionnaire.global_pipe = None
        gestionnaire.global_compel = None
        # Les autres initial_* gardent leur valeur par défaut

elif DEFAULT_MODEL:
    # --- Le modèle par défaut n'a pas été trouvé dans le dossier ---
    gestionnaire.global_pipe = None
    gestionnaire.global_compel = None

if gestionnaire.global_pipe: # Vérifier si le chargement initial (via DEFAULT_MODEL) a réussi
    print(txt_color("[INFO]", "info"), "Mise à jour de l'état initial du gestionnaire après chargement réussi.")
    # Utiliser les variables globales qui ont été mises à jour lors du chargement réussi
    gestionnaire.set_current_model_info(model_selectionne, vae_selctionne)
    # Assurez-vous que global_selected_sampler_key est défini avec une valeur par défaut avant ce point
    gestionnaire.set_current_sampler(global_selected_sampler_key)
else:
    print(txt_color("[WARN]", "warning"), "Aucun modèle initial chargé, l'état du gestionnaire reste vide.")
    # Assurer que l'état est None si aucun modèle n'est chargé
    gestionnaire.set_current_model_info(None, None)
    gestionnaire.set_current_sampler(None)
# =========================
# Model Loading Functions (update_globals_model, update_globals_model_inpainting) for gradio
# =========================

def update_globals_model(nom_fichier, nom_vae):
    global  model_selectionne, vae_selctionne, loras_charges
    gestionnaire.set_current_model_info(nom_fichier, nom_vae)
    returned_values = charger_modele(nom_fichier, nom_vae, translations, MODELS_DIR, VAE_DIR, device, torch_dtype, vram_total_gb, gestionnaire.global_pipe, gestionnaire.global_compel, gradio_mode=True)
    if len(returned_values) >= 3:
        gestionnaire.global_pipe = returned_values[0]
        gestionnaire.global_compel = returned_values[1]
        message = returned_values[2] # Ou *message si plus de 3 valeurs
    else:
         # Gérer le cas où charger_modele ne retourne pas assez de valeurs
         print(txt_color("[ERREUR]", "erreur"), "Retour inattendu de charger_modele")
         gestionnaire.set_current_model_info(None, None)
         gestionnaire.set_current_sampler(None)
         gestionnaire.global_pipe = None
         gestionnaire.global_compel = None
         message = "Erreur interne lors du chargement."


    if gestionnaire.global_pipe is not None:
        # Chargement réussi
        model_selectionne = nom_fichier
        vae_selctionne = nom_vae
        loras_charges.clear()
        etat_interactif = True
        texte_bouton = translate("generer", translations) # Texte normal

    else:
        # Chargement échoué
        model_selectionne = None
        vae_selctionne = None
        gestionnaire.update_global_pipe(None)
        gestionnaire.update_global_compel(None)
        etat_interactif = False
        texte_bouton = translate("charger_modele_pour_commencer", translations) # Texte désactivé
        selected_sampler_key_state.value = None

    update_interactif = gr.update(interactive=etat_interactif)
    update_texte = gr.update(value=texte_bouton)
    # Retourner le message et les deux mises à jour pour le bouton
    return message, update_interactif, update_texte

def update_globals_model_inpainting(nom_fichier):
    global model_selectionne
    gestionnaire.global_pipe, gestionnaire.global_compel, message = charger_modele_inpainting(nom_fichier, translations, INPAINT_MODELS_DIR, device, torch_dtype, vram_total_gb, gestionnaire.global_pipe, gestionnaire.global_compel)

    if gestionnaire.global_pipe is not None:
        # Chargement réussi
        model_selectionne = nom_fichier
        etat_interactif = True
        texte_bouton = translate("generer_inpainting", translations) # Texte normal
    else:
        # Chargement échoué
        model_selectionne = None
        gestionnaire.update_global_pipe(None)
        gestionnaire.update_global_compel(None)
        etat_interactif = False
        texte_bouton = translate("charger_modele_inpaint_pour_commencer", translations) # Texte désactivé

    update_interactif = gr.update(interactive=etat_interactif)
    update_texte = gr.update(value=texte_bouton)

    # Retourner le message et les deux mises à jour pour le bouton
    return message, update_interactif, update_texte

#==========================
# Outils pour gérer les iamges du module Inpainting
#==========================

def process_uploaded_image(uploaded_image):
    """Vérifie et redimensionne l'image uploadée avant de l'envoyer à ImageMask."""
    if uploaded_image is None:
        return None

    print(txt_color("[INFO]", "info"), translate("verification_image_entree", translations))
    image_checker = ImageSDXLchecker(uploaded_image, translations)
    resized_image = image_checker.redimensionner_image()

    if not isinstance(resized_image, Image.Image):
        print(txt_color("[ERREUR]", "erreur"), translate("erreur_redimensionnement_image_fond", translations).format(type(resized_image)))
        gr.Warning(translate("erreur_redimensionnement_image_fond", translations).format(type(resized_image)), 4.0)
        return None 

    print(txt_color("[INFO]", "info"), translate("image_prete_pour_mask", translations))
    return resized_image

# Fonction simplifiée pour les effets du masque ---
def update_inpainting_mask_effects(image_and_mask, opacity, blur_radius):
    """Applique seulement les effets d'opacité/flou sur le masque actuel."""
    processed_mask = apply_mask_effects(image_and_mask, translations, opacity, blur_radius)
    return processed_mask


# --- Fonction pour mettre à jour ImageMask ET calculer le masque initial ---
def update_image_mask_and_initial_effects(validated_image, opacity, blur_radius):
    """Met à jour ImageMask et calcule le masque initial."""
    if validated_image is None:
        return None, None # Pour ImageMask et mask_output
    initial_mask_for_effects = Image.new('L', validated_image.size, 0) # Masque noir
    temp_dict = {"background": validated_image, "layers": [initial_mask_for_effects]}
    initial_processed_mask = apply_mask_effects(temp_dict, translations, opacity, blur_radius)
    # Retourner l'image pour ImageMask et le masque initial pour mask_output
    return validated_image, initial_processed_mask

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

         # --- Validation Modèle ---
        available_models = lister_fichiers(MODELS_DIR, translations, gradio_mode=True)
        model_update = gr.update() # Par défaut, ne rien changer
        if model_name and model_name in available_models:
            model_update = gr.update(value=model_name)
        elif model_name:
            print(f"[WARN Preset Load] Modèle '{model_name}' du preset non trouvé dans les options actuelles.")

         # --- Validation VAE ---
        available_vae_files  = lister_fichiers(VAE_DIR, translations, gradio_mode=True)
        vae_value_for_ui = "Défaut VAE"

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
        available_loras = lister_fichiers(LORAS_DIR, translations, gradio_mode=True)
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
                    print(f"[WARN] LoRA '{lora_name}' du preset non trouvé. Ignoré pour le slot {i+1}.")
                    # Laisser le slot désactivé

        # --- Préparer l'update du Sampler et appliquer au backend ---
        sampler_update_msg, success = apply_sampler_to_pipe(gestionnaire.global_pipe, sampler_key, translations)
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
            apply_sampler_to_pipe(gestionnaire.global_pipe, default_sampler_key, translations) # Appliquer le défaut
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
        num_lora_slots = 4
        return [gr.update()] * (2 + 7 + 3 * num_lora_slots + 1) 

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
                seed_input = gr.Number(label=translate("seed", translations), value=-1)
                num_images_slider = gr.Slider(1, 200, value=1, label=translate("nombre_images_generer", translations), step=1)                
                
                                    
            with gr.Column(scale=1, min_width=200):
                with gr.Row():
                    with gr.Column():
                        preview_image_output = gr.Image(height=170, label=translate("apercu_etapes", translations),interactive=False)
                        seed_output = gr.Textbox(label=translate("seed_utilise", translations))
                        value = DEFAULT_MODEL if DEFAULT_MODEL else None
                        modele_dropdown = gr.Dropdown(label=translate("selectionner_modele", translations), choices=modeles_disponibles, value=initial_model_value, allow_custom_value=True)
                        vae_dropdown = gr.Dropdown(label=translate("selectionner_vae", translations), choices=vaes, value=initial_vae_value)
                        sampler_display_choices = get_sampler_choices(translations) # Obtenir les choix depuis l'utilitaire
                        default_sampler_display = translate(global_selected_sampler_key, translations) # Traduire la clé par défaut

                        sampler_dropdown = gr.Dropdown(
                            label=translate("selectionner_sampler", translations),
                            choices=sampler_display_choices,
                            value=default_sampler_display # Utiliser le nom traduit par défaut
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
                                    lora_check.change(
                                        fn=lambda check: gr.update(interactive=check),
                                        inputs=[lora_check],
                                        outputs=[lora_dropdown]
                                    )
                    lora_message = gr.Textbox(label=translate("message_lora", translations), value="")

                def mettre_a_jour_listes():
                    modeles = lister_fichiers(MODELS_DIR, translations, gradio_mode=True)
                    vaes = ["Défaut VAE"] + lister_fichiers(VAE_DIR, translations, gradio_mode=True)
                    loras = lister_fichiers(LORAS_DIR, translations, gradio_mode=True)

                    has_loras = bool(loras) and translate("aucun_modele_trouve",translations) not in loras and translate("repertoire_not_found",translations) not in loras
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
        with gr.Row():
            with gr.Column():
                image_mask_input = gr.ImageMask(
                    label=translate("image_avec_mask", translations), 
                    type="pil", 
                    sources=[], 
                    interactive=True
                )
                image_upload_input = gr.Image(
                    label=translate("telecharger_image_inpainting", translations), # Nouvelle clé
                    type="pil",
                    sources=["upload", "clipboard"] # Permettre upload et coller
                )
            with gr.Column():
                mask_image_output = gr.Image(type="pil", label=translate("sortie_mask_inpainting", translations), interactive=False)
            with gr.Column():
                opacity_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    label=translate("Mask_opacity", translations),
                )
                blur_slider = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=1,
                    label=translate("Mask_blur", translations),
                )
                inpainting_prompt = gr.Textbox(label=translate("prompt_inpainting", translations))
                traduire_inpainting_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", translations), value=False, info=translate("traduire_en_anglais", translations))
                guidance_inpainting_slider = gr.Slider(1, 20, value=7, label=translate("guidage", translations))
                num_steps_inpainting_slider = gr.Slider(1, 50, value=30, label=translate("etapes", translations), step=1)
                strength_inpainting_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.89, step=0.01, label=translate("force_inpainting", translations))
                modele_inpainting_dropdown = gr.Dropdown(label=translate("selectionner_modele_inpainting", translations), choices=modeles_impaint, value=value)
                bouton_lister_inpainting = gr.Button(translate("lister_modeles_inpainting", translations))
                bouton_charger_inpainting = gr.Button(translate("charger_modele_inpainting", translations))
                message_chargement_inpainting = gr.Textbox(label=translate("statut_inpainting", translations), value=translate("aucun_modele_charge_inpainting", translations))
                message_inpainting = gr.Textbox(label=translate("message_inpainting", translations), interactive=False)                
                
                texte_bouton_inpaint_initial = translate("charger_modele_pour_commencer", translations)
                bouton_generate_inpainting = gr.Button(value=texte_bouton_inpaint_initial, interactive=False, variant="primary")
                bouton_stop_inpainting = gr.Button(translate("arreter_inpainting", translations), variant="stop")


            with gr.Column():
            
                inpainting_ressultion_output = gr.Image(type="pil", label=translate("sortie_inpainting", translations),interactive=False)
                progress_inp_html_output = gr.HTML(value="")

            # --- Mettre à jour les événements .change() ---
            def update_mask_and_preview(image_and_mask, opacity, blur_radius):
                """Met à jour l'aperçu du masque ET l'aperçu redimensionné de l'original."""
                if image_and_mask and isinstance(image_and_mask, dict) and "background" in image_and_mask:
                    original_img = image_and_mask["background"]
                    processed_mask = apply_mask_effects(image_and_mask, translations, opacity, blur_radius)

                    # --- AJOUT : Redimensionnement avec PIL ---
                    target_height = 150
                    width_orig, height_orig = original_img.size
                    if height_orig > target_height: # Redimensionner seulement si nécessaire
                        ratio = target_height / height_orig
                        new_width = int(width_orig * ratio)
                        # Utiliser Image.Resampling.LANCZOS pour Pillow >= 9.0.0
                        # Utiliser Image.LANCZOS pour les versions plus anciennes
                        try:
                            resampling_filter = Image.Resampling.LANCZOS
                        except AttributeError:
                            resampling_filter = Image.LANCZOS
                        resized_original = original_img.resize((new_width, target_height), resampling_filter)
                    else:
                        resized_original = original_img # Pas besoin de redimensionner

                    # Retourner le masque traité ET l'original redimensionné
                    return processed_mask, resized_original
                    # --- FIN AJOUT ---
                else:
                    return None, None
            




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
        inputs=[modele_dropdown, vae_dropdown],
        outputs=[message_chargement, btn_generate, btn_generate]
    )

    image_input.change(fn=generate_caption, inputs=image_input, outputs=text_input)

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
        gr.State(config),           # Correspond à config
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
            gestionnaire,
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



    def handle_sampler_change(selected_display_name):
        """Gère le changement de sampler dans l'UI principale."""
        global global_selected_sampler_key # Accéder à la variable globale

        sampler_key = get_sampler_key_from_display_name(selected_display_name, translations)
        if sampler_key:
            message, success = apply_sampler_to_pipe(gestionnaire.global_pipe, sampler_key, translations)
            if success:
                global_selected_sampler_key = sampler_key # Mettre à jour la clé globale si succès
                gestionnaire.set_current_sampler(sampler_key)
                gr.Info(message, 3.0) # Afficher l'info Gradio ici
            else:
                gr.Warning(message, 4.0) # Afficher l'avertissement Gradio ici
            return message # Retourner le message pour le Textbox de statut
        else:
            error_msg = f"{translate('erreur_sampler_inconnu', translations)}: {selected_display_name}"
            gr.Warning(error_msg, 4.0)
            return error_msg

    sampler_dropdown.change(
        fn=handle_sampler_change,
        inputs=sampler_dropdown,
        outputs=[message_chargement] # Mettre à jour le Textbox de statut
    )

# Liaisons pour l'onglet Inpainting
    image_upload_input.upload(
        fn=process_uploaded_image,
        inputs=[image_upload_input],
        outputs=[validated_image_state] # Met à jour SEULEMENT l'état
    )
    image_upload_input.change( # Gérer aussi l'effacement
        fn=process_uploaded_image,
        inputs=[image_upload_input],
        outputs=[validated_image_state]
    )
    validated_image_state.change(
        fn=update_image_mask_and_initial_effects,
        inputs=[validated_image_state, opacity_slider, blur_slider],
        outputs=[image_mask_input, mask_image_output]
    )            
    
    opacity_slider.change(
        fn=update_inpainting_mask_effects,
        inputs=[image_mask_input, opacity_slider, blur_slider],
        outputs=[mask_image_output],
    )
    blur_slider.change(
        fn=update_inpainting_mask_effects,
        inputs=[image_mask_input, opacity_slider, blur_slider],
        outputs=[mask_image_output],
    )
    image_mask_input.change( # Déclenché quand on dessine
        fn=update_inpainting_mask_effects,
        inputs=[image_mask_input, opacity_slider, blur_slider],
        outputs=[mask_image_output],
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
            mask_image_output,
            num_steps_inpainting_slider,
            strength_inpainting_slider,
            guidance_inpainting_slider,
            traduire_inpainting_checkbox
        ],
        outputs=[
            inpainting_ressultion_output,
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

    # ... (idem pour preset_filter_model.change, preset_filter_sampler.change, preset_filter_lora.change) ...

    # --- Liaison du bouton Refresh manuel ---
    # Incrémente le trigger (déclenche @gr.render) ET met à jour la pagination
    def refresh_trigger_and_update_pagination_dd(current_page, trigger, *filter_args):
        pagination_dd_update = update_pagination_display(current_page, *filter_args) # Utilise page actuelle
        return gr.update(value=trigger + 1), pagination_dd_update

    # 1. Mettre à jour l'état de la page
    preset_page_dropdown.change(
        fn=handle_page_dropdown_change, # Utiliser la fonction nommée
        inputs=[preset_page_dropdown],
        outputs=[current_preset_page_state] # Cible l'état
    )

    def initial_load_update_pagination_dd(*filter_args):
        pagination_dd_update = update_pagination_display(1, *filter_args) # Page 1 initiale
        return gr.update(value=1), pagination_dd_update

    interface.load(
        fn=initial_load_update_pagination_dd,
        inputs=pagination_dd_inputs[1:], # Juste les filtres initiaux
        outputs=[current_preset_page_state, pagination_dd_output] # Met à jour état page ET dropdown
    )


interface.launch(inbrowser=str_to_bool(OPEN_BROWSER), pwa=True, share=str_to_bool(SHARE))