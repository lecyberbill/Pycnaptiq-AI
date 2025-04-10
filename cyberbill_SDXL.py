import random
import importlib
import os
import shutil
import time
import threading
from datetime import datetime
import json
import gradio as gr
from diffusers import AutoPipelineForText2Image,StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, \
    LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, KDPM2DiscreteScheduler, StableDiffusionXLInpaintPipeline, \
    KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, DPMSolverSDEScheduler, DPMSolverSinglestepScheduler
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from core.version import version
from Utils.callback_diffuser import latents_to_rgb, create_callback_on_step_end, interrupt_diffusers_callback
from core.trannslator import translate_prompt
from core.Inpaint import apply_mask_effects
from Utils.model_loader import charger_modele, charger_modele_inpainting, charger_lora, decharge_lora, gerer_lora
from Utils.utils import enregistrer_etiquettes_image_html,charger_configuration, gradio_change_theme, lister_fichiers, GestionModule,\
    telechargement_modele, txt_color, str_to_bool, load_locales, translate, get_language_options, enregistrer_image, check_gpu_availability, decharger_modele, ImageSDXLchecker
#from modules.retouche import ImageEditor
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process
torch.backends.cudnn.deterministic = True

from compel import Compel, ReturnedEmbeddingsType
from io import BytesIO 


print (f"cyberbill_SDXL version {txt_color(version(),'info')}")
# Load the configuration first
config = charger_configuration()
# Initialisation de la langue

DEFAULT_LANGUAGE = config.get("LANGUAGE", "fr")  # Utilisez 'fr' comme langue par défaut si 'LANGUAGE' n'est pas défini.
translations = load_locales(DEFAULT_LANGUAGE)


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

# Initialisation des variables globales
pipe = None
model_selectionne = None
vae_selctionne = "Défaut VAE"
compel = None
loras_charges = {}


# Créer une instance du gestionnaire de modules
gestionnaire = GestionModule(translations=translations, language=DEFAULT_LANGUAGE, global_pipe=pipe, global_compel=compel, config=config)
# Charger tous les modules
gestionnaire.charger_tous_les_modules()

# Get the javascript code
js_code = gestionnaire.get_js_code()


# Flag pour arrêter la génération
stop_event = threading.Event()

stop_gen = threading.Event()

callback_on_step_end = create_callback_on_step_end(PREVIEW_QUEUE, stop_gen)
callback_on_step_end_diffuser = interrupt_diffusers_callback(stop_gen)


# Flag pour signaler qu'une tâche est en cours
processing_event = threading.Event()
# Flag to indicate if an image generation is in progress
is_generating = False



# Charger le modèle et le processeur
caption_model = AutoModelForCausalLM.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v2.0", trust_remote_code=True).to(device)
caption_processor = AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v2.0", trust_remote_code=True)

# Sampler disponibles
sampler_options = [
    translate("sampler_euler", translations),
    translate("sampler_dpmpp_2m", translations),
    translate("sampler_dpmpp_2s_a", translations),
    translate("sampler_lms", translations),
    translate("sampler_ddim", translations),
    translate("sampler_pndm", translations),
    translate("sampler_dpm2", translations),
    translate("sampler_dpm2_a", translations),
    translate("sampler_dpm_fast", translations),
    translate("sampler_dpm_adaptive", translations),
    translate("sampler_heun", translations),
    translate("sampler_dpmpp_sde", translations),
    translate("sampler_dpmpp_3m_sde", translations),
    translate("sampler_dpmpp_2m", translations),
    translate("sampler_euler_a", translations),
    translate("sampler_lms", translations),
    translate("sampler_unipc", translations),
]


# =========================
# Définition des fonctions
# =========================



# selection des sampler
def apply_sampler(sampler_selection):
    info = f"{translate('sampler_change', translations)}{sampler_selection}"
    if pipe is not None:
        if sampler_selection == translate("sampler_euler", translations):
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3)
            return info
        elif sampler_selection == translate("sampler_dpmpp_2m", translations):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_dpmpp_2s_a", translations):
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_lms", translations):
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_ddim", translations):
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_pndm", translations):
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_dpm2", translations):
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_dpm2_a", translations):
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_dpm_fast", translations):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_dpm_adaptive", translations):
            pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_heun", translations):
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_dpmpp_sde", translations):
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)  # Note: Utilisation de DPMSolverSDEScheduler
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_dpmpp_3m_sde", translations):
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_euler_a", translations):
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        elif sampler_selection == translate("sampler_unipc", translations):
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            gr.Info(info,3.0)
            return info
        else:
            print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_sampler_inconnu', translations)} {sampler_selection}")  # Utilisation de la clé de traduction
            gr.Warning(translate('erreur_sampler_inconnu', translations) + " " + sampler_selection, 4.0)
            return f"{translate('erreur_sampler_inconnu', translations)} {sampler_selection}"

    else:
        return translate("charger_modele_avant_sampler", translations)


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

pipe, compel, *message = charger_modele(DEFAULT_MODEL, "Défaut VAE", translations, MODELS_DIR, VAE_DIR, device, torch_dtype, vram_total_gb, pipe, compel)

compel = compel

#générer prompt à partir d'une image

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



def generate_image(text, style_selection, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images, *lora_inputs):#, enhance_checkbox):
    """Génère des images avec Stable Diffusion."""
    global pipe, compel, lora_charges
    
    lora_checks = lora_inputs[:4]
    lora_dropdowns = lora_inputs[4:8]
    lora_scales = lora_inputs[8:]

    selected_style = style_choice(style_selection, STYLES)
    try:
        
        if pipe is None:
            print(txt_color("[ERREUR] ","erreur"), translate("erreur_pas_modele", translations))
            gr.Warning(translate("erreur_pas_modele", translations), 4.0)
            return None, None, translate("erreur_pas_modele", translations)
        
        #initialisation du chrono
        start_time = time.time()
        # Réinitialiser l'état d'arrêt
        stop_event.clear()
        stop_gen.clear()  

        seeds = [random.randint(1, 10**19 - 1) for _ in range(num_images)] if seed_input == -1 else [seed_input] * num_images        
        prompt_text = translate_prompt(text, translations) if traduire else text
        prompt_en = selected_style["prompt"].replace("{prompt}", prompt_text) if selected_style else text
        if selected_style:
            negative_prompt = selected_style["negative_prompt"] if selected_style["name"] != translate("Aucun_style", translations) else NEGATIVE_PROMPT
        else:
            negative_prompt = NEGATIVE_PROMPT           
        
        conditioning, pooled = compel(prompt_en)  
        
        selected_format = selected_format.split(":")[0].strip()
        width, height = map(int, selected_format.split("*"))
        images = [] # Initialize a list to store all generated images
        seed_strings = []
        formatted_seeds = ""

        message_lora = gerer_lora(pipe, loras_charges, lora_checks, lora_dropdowns, lora_scales, LORAS_DIR, translations)
        if message_lora:
            gr.Warning(message_lora, 4.0)
            return None, None, message_lora    
       

        # principal loop for genrated images
        for idx, seed in enumerate(seeds):
            depart_time = time.time()
            PREVIEW_QUEUE.clear()
            final_image_container = {}
            html_message_result = translate("generation_en_cours", translations)
            if stop_event.is_set():  # Vérifie si un arrêt est demandé
                print(txt_color("[INFO] ","info"), translate("arrete_demande_apres", translations), f"{idx} {translate('images', translations)}.")
                gr.Info(translate("arrete_demande_apres", translations) + f"{idx} {translate('images', translations)}.", 3.0)
                break
                
            print(txt_color("[INFO] ","info"), f"{translate('generation_image', translations)} {idx+1} {translate('seed_utilise', translations)} {seed}")
            gr.Info(translate('generation_image', translations) + f"{idx+1} {translate('seed_utilise', translations)} {seed}", 3.0)
            is_generating = True
            
            def run_pipeline():
                generator = torch.Generator(device=device).manual_seed(seed)
                final_image = pipe(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt, # Fournir le prompt négatif textuel
                    generator=generator,
                    width=width,
                    height=height,
                    callback_on_step_end=callback_on_step_end, 
                    callback_on_step_end_tensor_inputs=["latents"]
                )
                if not stop_gen.is_set():
                    final_image_container["final"] = final_image.images[0]

            thread = threading.Thread(target=run_pipeline)
            
            thread.start()



            # Calcul du temps de génération pour cette imag
            
            # Tant que la génération est en cours ou que de nouveaux aperçus sont disponibles, yield chaque image latente
            last_index = 0
            while thread.is_alive() or last_index < len(PREVIEW_QUEUE):
                while last_index < len(PREVIEW_QUEUE):
                    preview_img = PREVIEW_QUEUE[last_index]
                    last_index += 1
                    # On yield uniquement la mise à jour de l'aperçu (les autres outputs restent inchangés)
                    yield images, None, None, None, preview_img
                time.sleep(0.1)

            thread.join() # Attendre la fin du thread (ou son interruption)
            is_generating = False # Mettre à jour l'état de génération
            PREVIEW_QUEUE.clear()

            # --- NOUVELLE VÉRIFICATION : Après la fin du thread ---
            if stop_gen.is_set():
                print(txt_color("[INFO]", "info"), translate("generation_arretee_pas_sauvegarde", translations))
                gr.Info(translate("generation_arretee_pas_sauvegarde", translations), 3.0)
                # On sort de la boucle for, car l'arrêt concerne toute la tâche
                yield images, " ".join(seed_strings), translate("generation_arretee", translations), translate("generation_arretee_pas_sauvegarde", translations), None
                break # Important: sortir de la boucle for

            # --- Si la génération n'a PAS été arrêtée ---
            final_image = final_image_container.get("final", None)

            if final_image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image_genere", translations))
                gr.Warning(translate("erreur_pas_image_genere", translations), 4.0)
                # Continuer à la prochaine image du lot si possible
                yield images, " ".join(seed_strings), f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}", translate("erreur_pas_image_genere", translations), None
                continue # Passer à l'itération suivante
            
            temps_generation_image = f"{(time.time() - depart_time):.2f} sec"            
            

            date_str = datetime.now().strftime("%Y_%m_%d")
            heure_str = datetime.now().strftime("%H_%M_%S")
            save_dir = os.path.join(SAVE_DIR, date_str)
            os.makedirs(save_dir, exist_ok=True)

            lora_info = []
            for i, (check, dropdown, scale) in enumerate(zip(lora_checks, lora_dropdowns, lora_scales)):
                if check:
                    lora_name = dropdown
                    if lora_name != "Aucun LORA disponible":
                        lora_info.append(f"{lora_name} ({scale:.2f})")

            lora_info_str = ", ".join(lora_info) if lora_info else translate("aucun_lora", translations)
          
            donnees_xmp =  {
                    "IMAGE": f"{idx+1} {translate('image_sur',translations)} {num_images}",
                    "Creator": AUTHOR,
                    "Seed": seed,
                    "Inference": num_steps,
                    "Guidance": guidance_scale,
                    "Prompt": prompt_en,
                    "Negatif Prompt": negative_prompt,
                    "Style": selected_style["name"] ,
                    "Dimension": selected_format,
                    "Modèle": os.path.splitext(model_selectionne)[0],
                    "VAE": os.path.splitext(vae_selctionne)[0] if vae_selctionne else "Défaut VAE",
                    "Sampler": pipe.scheduler.__class__.__name__,
                    "Loras": lora_info_str,
                    "Temps de génération": temps_generation_image 
                }

            filename = f"{date_str}_{heure_str}_{seed}_{width}x{height}_{idx+1}.{IMAGE_FORMAT.lower()}"
            chemin_image = os.path.join(save_dir, filename)

            if chemin_image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_chemin_image_none", translations))
                gr.Warning(translate("erreur_chemin_image_none", translations), 4.0)
                yield images, formatted_seeds, f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}", translate("erreur_chemin_image_none", translations), None
                continue

            gr.Info(translate("image_sauvegarder", translations) + " " + chemin_image, 3.0)    
            is_generating = False
            
            # sauvegarde de l'image
            # enregistrer_image(final_image, chemin_image, donnees_xmp, translations, IMAGE_FORMAT)
            # Déléguer la tâche d'enregistrement de l'image au ThreadPoolExecutor
            image_future = image_executor.submit(enregistrer_image, final_image, chemin_image, translations, IMAGE_FORMAT)
                                   
            is_last_image = (idx == num_images - 1)
            # Déléguer la tâche d'écriture dans le ThreadPoolExecutor
            html_message = html_executor.submit(enregistrer_etiquettes_image_html, chemin_image, donnees_xmp, translations, is_last_image)
            
            print(txt_color("[OK] ","ok"),translate("image_sauvegarder", translations), txt_color(f"{filename}","ok"))
            images.append(final_image) # Append each generated image to the list
            torch.cuda.empty_cache()

            # Append the seed value as a string to the list, formatted as [seed]
            seed_strings.append(f"[{seed}]")
            # join the string with a space
            formatted_seeds = " ".join(seed_strings)
            # Attendre et récupérer le résultat HTML (ou gérer l'erreur)
            try:
                html_message_result = html_message.result(timeout=10) # Ajout d'un timeout
            except Exception as html_err:
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_lors_generation_html', translations)}: {html_err}")
                 html_message_result = translate("erreur_lors_generation_html", translations)
                       
            yield images, formatted_seeds, f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}", html_message_result, final_image# return the all list image
                   
        # --- Fin de la boucle ---
        elapsed_time = f"{(time.time() - start_time):.2f} sec"
        final_message = translate('temps_total_generation', translations) + " : " + elapsed_time
        if stop_gen.is_set() or stop_event.is_set(): # Vérifier si arrêté
             final_message = translate("generation_arretee", translations)

        print(txt_color("[INFO] ","info"), final_message)
        gr.Info(final_message, 3.0)

        return images, formatted_seeds, final_message

    except Exception as e:
        # ... (gestion des exceptions inchangée) ...
        is_generating = False # Assurer que le flag est réinitialisé en cas d'erreur
        print(txt_color("[ERREUR] ","erreur"), f"{translate('erreur_lors_generation', translations)} : {e}")
        raise gr.Error (f"{translate('erreur_lors_generation', translations)} : {e}", 4.0)
        # Retourner des valeurs cohérentes en cas d'erreur
        return [], "", f"{translate('erreur_lors_generation', translations)} : {e}"



def generate_inpainted_image(text, image, mask, num_steps, strength, guidance_scale, traduire):
    """Génère une image inpainted avec Stable Diffusion XL."""
    global pipe, compel
    try:
        start_time = time.time()
        stop_gen.clear()
        if pipe is None:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_modele_inpainting", translations))
            gr.Warning(translate("erreur_pas_modele_inpainting", translations), 4.0)
            return None, translate("erreur_pas_modele_inpainting", translations)

        if image is None or mask is None:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_image_mask_manquant", translations))
            gr.Warning(translate("erreur_image_mask_manquant", translations), 4.0)
            return None, translate("erreur_image_mask_manquant", translations)

         # --- CORRECTION : 'image' et 'mask' sont déjà des objets PIL ---
        # Vérifier les types par sécurité
        if not isinstance(image, Image.Image):
             print(txt_color("[ERREUR]", "erreur"), f"generate_inpainted_image a reçu un type invalide pour 'image': {type(image)}")
             raise gr.Error(f"Type d'image invalide reçu pour l'inpainting: {type(image)}", 4.0)
             # return None, "Erreur type image", "Erreur type image"

        if not isinstance(mask, Image.Image):
             print(txt_color("[ERREUR]", "erreur"), f"generate_inpainted_image a reçu un type invalide pour 'mask': {type(mask)}")
             raise gr.Error(f"Type de masque invalide reçu pour l'inpainting: {type(mask)}", 4.0)
             # return None, "Erreur type masque", "Erreur type masque"

        # Translate the prompt if requested
        prompt_text = translate_prompt(text, translations) if traduire else text
        conditioning, pooled = compel(prompt_text)
        active_adapters = pipe.get_active_adapters()
        for adapter_name in active_adapters:
            pipe.set_adapters(adapter_name, 0)
        
        image_rgb = image.convert("RGB")
        mask_rgb = mask
        final_image_container = {}

        # Run the inpainting pipeline
        def run_pipeline():
            print(txt_color("[INFO] ", "info"), translate("debut_inpainting", translations))
            gr.Info(translate("debut_inpainting", translations), 3.0)
            try: # Ajouter try/except
                inpainted_image = pipe(
                    pooled_prompt_embeds=pooled,
                    prompt_embeds=conditioning,
                    image=image_rgb,
                    mask_image=mask_rgb,
                    width=image.width,
                    height=image.height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    callback_on_step_end=callback_on_step_end_diffuser # Utilise stop_gen
                ).images[0]
                if not stop_gen.is_set():
                    final_image_container["final"] = inpainted_image
            except Exception as e:
                 if "Interrupt" in str(e): # Exemple
                     print(txt_color("[INFO]", "info"), translate("inpainting_interrompu_interne", translations))
                 else:
                     print(txt_color("[ERREUR]", "erreur"), f"Erreur dans run_pipeline (inpainting): {e}")


        thread = threading.Thread(target=run_pipeline)
        thread.start()
        thread.join()
        inpainted_image = final_image_container.get("final", None)
        
        if stop_gen.is_set():
            print(txt_color("[INFO]", "info"), translate("inpainting_arrete_pas_sauvegarde", translations))
            gr.Info(translate("inpainting_arrete_pas_sauvegarde", translations), 3.0)
            # Retourner des valeurs indiquant l'arrêt
            return None, translate("inpainting_arrete", translations), translate("inpainting_arrete_pas_sauvegarde", translations)

        # --- Si l'inpainting n'a PAS été arrêté ---
        inpainted_image = final_image_container.get("final", None)

        if inpainted_image is None:
             print(txt_color("[ERREUR]", "erreur"), translate("erreur_pas_image_inpainting", translations)) # Nouvelle clé
             gr.Warning(translate("erreur_pas_image_inpainting", translations), 4.0)
             return None, translate("erreur_lors_inpainting", translations), translate("erreur_pas_image_inpainting", translations)

       
        temps_generation_image = f"{(time.time() - start_time):.2f} sec"
        date_str = datetime.now().strftime("%Y_%m_%d")
        heure_str = datetime.now().strftime("%H_%M_%S")
        save_dir = os.path.join(SAVE_DIR, date_str)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{date_str}_{heure_str}_inpainting_{image.width}x{image.height}.{IMAGE_FORMAT.lower()}"
        chemin_image = os.path.join(save_dir, filename)
        donnees_xmp = {
            "IMAGE": translate("inpainting", translations),
            "Creator": AUTHOR,
            "Inference": num_steps,
            "Guidance": guidance_scale,
            "Prompt": prompt_text,
            "Modèle": os.path.splitext(model_selectionne)[0],
            "Dimension": f"{image.width}x{image.height}",
            "Temps de génération": temps_generation_image
        }

        image_future = image_executor.submit(enregistrer_image, inpainted_image, chemin_image, translations, IMAGE_FORMAT)
        html_message = html_executor.submit(enregistrer_etiquettes_image_html, chemin_image, donnees_xmp, translations, is_last_image=True)

        # Attendre et récupérer le résultat HTML
        try:
            html_message_result = html_future.result(timeout=10)
        except Exception as html_err:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_lors_generation_html', translations)}: {html_err}")
            html_message_result = translate("erreur_lors_generation_html", translations)

        print(txt_color("[OK] ", "ok"), translate("fin_inpainting", translations))
        gr.Info(translate("fin_inpainting", translations), 3.0)

        return inpainted_image, html_message_result, translate("inpainting_reussi", translations)

    except (ValueError, RuntimeError) as e: # Erreurs spécifiques (ex: traduction)
        print(txt_color("[ERREUR]", "erreur"), f"Erreur spécifique interceptée dans inpainting: {e}")
        raise gr.Error(str(e), 4.0)
        # return None, str(e), str(e) # Ajuster le nombre de retours si nécessaire
    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_lors_inpainting', translations)}: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{translate('erreur_lors_inpainting', translations)}: {e}", 4.0)




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


if DEFAULT_MODEL in modeles_disponibles:
    print(f"{txt_color('[INFO]','info')}", f"{DEFAULT_MODEL} {translate('va_se_charger', translations)} {MODELS_DIR}")
    # Use DEFAULT_MODEL and "Défaut VAE" directly
    pipe, compel, *message = charger_modele(DEFAULT_MODEL, "Défaut VAE", translations, MODELS_DIR, VAE_DIR, device, torch_dtype, vram_total_gb, pipe, compel)
    model_selectionne = DEFAULT_MODEL
    vae_selctionne = "Défaut VAE"

else:
    print(f"{txt_color('[INFO]','info')}", f"{DEFAULT_MODEL} {translate('pas_de_modele_dans', translations)} : {MODELS_DIR}")
    DEFAULT_MODEL = ""
    model_selectionne = ""
    vae_selctionne = ""

# =========================
# Model Loading Functions (update_globals_model, update_globals_model_inpainting) for gradio
# =========================

def update_globals_model(nom_fichier, nom_vae):
    global pipe, compel, model_selectionne, vae_selctionne
    pipe, compel, *message = charger_modele(nom_fichier, nom_vae, translations, MODELS_DIR, VAE_DIR, device, torch_dtype, vram_total_gb, pipe, compel, gradio_mode=True)
    if pipe is not None:
        # Chargement réussi
        model_selectionne = nom_fichier
        vae_selctionne = nom_vae
        loras_charges.clear()
        gestionnaire.update_global_pipe(pipe)
        gestionnaire.update_global_compel(compel)
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

    update_interactif = gr.update(interactive=etat_interactif)
    update_texte = gr.update(value=texte_bouton)
        # Retourner le message et les deux mises à jour pour le bouton
    return message, update_interactif, update_texte

def update_globals_model_inpainting(nom_fichier):
    global pipe, compel, model_selectionne 
    pipe, compel, message = charger_modele_inpainting(nom_fichier, translations, INPAINT_MODELS_DIR, device, torch_dtype, vram_total_gb, pipe, compel) 
    if pipe is not None:
        # Chargement réussi
        model_selectionne = nom_fichier
        gestionnaire.update_global_pipe(pipe)
        gestionnaire.update_global_compel(compel)
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

# Assure-toi que numpy est importé
import numpy as np
# Assure-toi que Image est importé depuis PIL
from PIL import Image, ImageOps
import os

# ... (autres imports et code)

def process_uploaded_image(uploaded_image):
    """Vérifie et redimensionne l'image uploadée avant de l'envoyer à ImageMask."""
    if uploaded_image is None:
        return None # Ne rien faire si aucune image

    print(txt_color("[INFO]", "info"), translate("verification_image_entree", translations))
    image_checker = ImageSDXLchecker(uploaded_image, translations)
    resized_image = image_checker.redimensionner_image()

    if not isinstance(resized_image, Image.Image):
        print(txt_color("[ERREUR]", "erreur"), translate("erreur_redimensionnement_image_fond", translations).format(type(resized_image)))
        gr.Warning(translate("erreur_redimensionnement_image_fond", translations).format(type(resized_image)), 4.0)
        return None # Retourner None en cas d'erreur

    print(txt_color("[INFO]", "info"), translate("image_prete_pour_mask", translations)) # Nouvelle clé
    # Retourner l'image (potentiellement redimensionnée) pour l'afficher dans ImageMask
    return resized_image

# --- MODIFICATION : Fonction simplifiée pour les effets du masque ---
def update_inpainting_mask_effects(image_and_mask, opacity, blur_radius):
    """Applique seulement les effets d'opacité/flou sur le masque actuel."""
    # Utilise directement apply_mask_effects (qui gère déjà les erreurs internes)
    processed_mask = apply_mask_effects(image_and_mask, translations, opacity, blur_radius)
    return processed_mask


# --- NOUVELLE fonction pour mettre à jour seulement les effets ---
def update_inpainting_mask_effects(resized_original, resized_mask, opacity, blur_radius):
    """Applique seulement les effets d'opacité/flou sur le masque déjà redimensionné."""
    if resized_original and resized_mask:
        # Recréer le dictionnaire attendu par apply_mask_effects
        temp_dict_for_effects = {
            "background": resized_original, # Nécessaire pour la structure, même si non utilisé
            "layers": [resized_mask],       # Le masque redimensionné stocké
            "composite": None
        }
        processed_mask = apply_mask_effects(temp_dict_for_effects, translations, opacity, blur_radius)
        return processed_mask
    else:
        # Si les images stockées ne sont pas valides
        return None

# --- Fonction pour mettre à jour ImageMask ET calculer le masque initial ---
def update_image_mask_and_initial_effects(validated_image, opacity, blur_radius):
    """Met à jour ImageMask et calcule le masque initial."""
    if validated_image is None:
        return None, None # Pour ImageMask et mask_output

    # Créer un masque initial vide (ou basé sur validated_image si nécessaire)
    # Ici, on suppose qu'apply_mask_effects peut gérer un dict sans masque réel
    # ou on crée un masque noir initial.
    # Pour être sûr, créons un masque noir de la bonne taille.
    initial_mask_for_effects = Image.new('L', validated_image.size, 0) # Masque noir
    temp_dict = {"background": validated_image, "layers": [initial_mask_for_effects]}

    initial_processed_mask = apply_mask_effects(temp_dict, translations, opacity, blur_radius)

    # Retourner l'image pour ImageMask et le masque initial pour mask_output
    return validated_image, initial_processed_mask

# --- Fonction pour les effets (reste la même) ---
def update_inpainting_mask_effects(image_and_mask, opacity, blur_radius):
    """Applique seulement les effets d'opacité/flou sur le masque actuel."""
    processed_mask = apply_mask_effects(image_and_mask, translations, opacity, blur_radius)
    return processed_mask



# =========================
# Interface utilisateur (Gradio)
# =========================

block_kwargs = {"theme": gradio_change_theme(GRADIO_THEME)}
if js_code:
    block_kwargs["js"] = js_code


with gr.Blocks(**block_kwargs) as interface:
     gr.Markdown(f"# Cyberbill SDXL images generator version {version()}")


############################################################
########TAB GENRATION IMAGE
############################################################   
     with gr.Tab(translate("generation_image", translations)):
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                text_input = gr.Textbox(label=translate("prompt", translations), info=translate("entrez_votre_texte_ici", translations), elem_id="promt_input")
                traduire_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", translations), value=False, info=translate("traduire_en_anglais", translations))
                style_dropdown = gr.Dropdown(choices=[style["name"] for style in STYLES], value=translate("Aucun_style", translations), label=translate("styles", translations), info=translate("Selectionnez_un_style_predefini", translations))
                use_image_checkbox = gr.Checkbox(label=translate("generer_prompt_image", translations), value=False)
                time_output = gr.Textbox(label=translate("temps_rendu", translations), interactive=False)
                html_output = gr.Textbox(label=translate("mise_a_jour_html", translations), interactive=False)
                message_chargement = gr.Textbox(label=translate("statut", translations), value=translate("aucun_modele_charge", translations))       
                
                image_input = gr.Image(label=translate("telechargez_image", translations), type="pil", visible=False)
                use_image_checkbox.change(fn=lambda use_image: gr.update(visible=use_image), inputs=use_image_checkbox, outputs=image_input)
            with gr.Column(scale=1, min_width=200):
                image_output = gr.Gallery(label=translate("images_generees", translations))
                guidance_slider = gr.Slider(1, 20, value=7, label=translate("guidage", translations))
                num_steps_slider = gr.Slider(1, 50, value=30, label=translate("etapes", translations), step=1)
                format_dropdown = gr.Dropdown(choices=FORMATS, value=FORMATS[3], label=translate("format", translations))
                seed_input = gr.Number(label=translate("seed", translations), value=-1)
                num_images_slider = gr.Slider(1, 200, value=1, label=translate("nombre_images_generer", translations), step=1)                
                
                   #enhance_checkbox = gr.Checkbox(label=translate("enhance_image", translations), value=False, visible=photo_editing_loaded)

                                    
            with gr.Column(scale=1, min_width=200):
                with gr.Row():
                    with gr.Column():
                        preview_image_output = gr.Image(height=170, label=translate("apercu_etapes", translations),interactive=False)
                        seed_output = gr.Textbox(label=translate("seed_utilise", translations))
                        value = DEFAULT_MODEL if DEFAULT_MODEL else None
                        modele_dropdown = gr.Dropdown(label=translate("selectionner_modele", translations), choices=modeles_disponibles, value=value)
                        vae_dropdown = gr.Dropdown(label=translate("selectionner_vae", translations), choices=vaes, value=value)
                        sampler_dropdown = gr.Dropdown(label=translate("selectionner_sampler", translations), choices=sampler_options)
                        bouton_charger = gr.Button(translate("charger_modele", translations))

                        
            with gr.Column():
                texte_bouton_gen_initial = translate("charger_modele_pour_commencer", translations) # Utiliser la même clé ou une clé spécifique
                btn_generate = gr.Button(value=texte_bouton_gen_initial, interactive=False)
                btn_stop = gr.Button(translate("arreter", translations))
                btn_stop_after_gen = gr.Button(translate("stop_apres_gen", translations))
                bouton_lister = gr.Button(translate("lister_modeles", translations))
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

                bouton_lister.click(
                    fn=mettre_a_jour_listes,
                    outputs=[modele_dropdown, vae_dropdown, *lora_dropdowns, lora_message]
                )
                
                bouton_charger.click(
                    fn=update_globals_model,
                    inputs=[modele_dropdown, vae_dropdown],
                    outputs=[message_chargement, btn_generate, btn_generate]
                )
                sampler_dropdown.change(fn=apply_sampler, inputs=sampler_dropdown, outputs=message_chargement)

        image_input.change(fn=generate_caption, inputs=image_input, outputs=text_input)

        btn_generate.click(
            generate_image,
            inputs=[text_input, style_dropdown, guidance_slider, num_steps_slider, format_dropdown, traduire_checkbox, seed_input, num_images_slider, *lora_checks, *lora_dropdowns, *lora_scales],  # enhance_checkbox],
            outputs=[image_output, seed_output, time_output, html_output, preview_image_output]
        )
        btn_stop.click(stop_generation_process, outputs=time_output)
        btn_stop_after_gen.click(stop_generation, outputs=time_output)

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
                bouton_generate_inpainting = gr.Button(value=texte_bouton_inpaint_initial, interactive=False)
                bouton_stop_inpainting = gr.Button(translate("arreter_inpainting", translations))

            with gr.Column():
            
                inpainting_ressultion_output = gr.Image(type="pil", label=translate("sortie_inpainting", translations),interactive=False)

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
                fn=lambda text, validated_img_from_state, processed_mask_from_output, num_steps, guidance_scale, strength, traduire: generate_inpainted_image(
                    text,
                    validated_img_from_state,
                    processed_mask_from_output,
                    num_steps,
                    strength,
                    guidance_scale,
                    traduire
                ),
                inputs=[
                    inpainting_prompt,
                    validated_image_state,
                    mask_image_output,
                    num_steps_inpainting_slider,
                    guidance_inpainting_slider,
                    strength_inpainting_slider,
                    traduire_inpainting_checkbox
                ],
                outputs=[inpainting_ressultion_output, message_chargement_inpainting, message_inpainting]
            )  
############################################################
########TAB MODULES
############################################################
     gestionnaire.creer_tous_les_onglets(translations)


interface.launch(inbrowser=str_to_bool(OPEN_BROWSER), pwa=True, share=str_to_bool(SHARE))