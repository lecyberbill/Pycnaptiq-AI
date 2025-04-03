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
from Utils.previewer import latents_to_rgb, create_callback_on_step_end
from core.Inpaint import apply_mask_effects, extract_image
from Utils.model_loader import charger_modele, charger_modele_inpainting, charger_lora, decharge_lora
from Utils.utils import enregistrer_etiquettes_image_html,charger_configuration, gradio_change_theme, lister_fichiers, GestionModule,\
    telechargement_modele, txt_color, str_to_bool, load_locales, translate, get_language_options, enregistrer_image, check_gpu_availability
#from modules.retouche import ImageEditor
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process
from core.version import version
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
pipe = None
model_selectionne = None
vae_selctionne = None
compel = None



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

# Flag pour signaler qu'une tâche est en cours
processing_event = threading.Event()
# Flag to indicate if an image generation is in progress
is_generating = False

# Chargement du modèle de traduction
translation_model = "Helsinki-NLP/opus-mt-fr-en"
translator = pipeline("translation", model=translation_model)
print(txt_color("[OK]","ok"), f"{translate('modele_traduction_charge',translations)} '{translation_model}' {translate('charge_model', translations)}")

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


# fonction de traduction
def translate_prompt(prompt_fr):
    """Traduit un prompt français en anglais."""
    try:
        traduction = translator(prompt_fr)[0]["translation_text"]
        print(txt_color("[INFO] ", "info"), f"{translate('traduction_effectuee', translations)} {prompt_fr} -> {traduction}")
        gr.Info(translate('traduction_effectuee', translations) + " " + prompt_fr + " -> " + traduction, 3.0)
        return traduction
    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_traduction', translations)}: {e}")
        raise gr.Error(f"{translate('erreur_traduction', translations)}: {e}", 4.0)
        return f"{translate('erreur_traduction', translations)}: {e}"


def generate_image(text, style_selection, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images, lora_scale):#, enhance_checkbox):
    """Génère des images avec Stable Diffusion."""
    global pipe, compel
    
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
        prompt_text = translate_prompt(text) if traduire else text
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
        active_adapters = pipe.get_active_adapters()        
        for adapter_name in active_adapters:
            pipe.set_adapters(adapter_name, lora_scale)
        

        # principal loop for genrated images
        for idx, seed in enumerate(seeds):
            depart_time = time.time()
            PREVIEW_QUEUE.clear()
            final_image_container = {}
            if stop_event.is_set():  # Vérifie si un arrêt est demandé
                html_message = html_executor.submit(enregistrer_etiquettes_image_html, chemin_image, donnees_xmp, translations, is_last_image=True)
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

            thread.join()
            
            # Récupération de l'image finale (qui sera ajoutée à la galerie)
            final_image = final_image_container.get("final", None)
            PREVIEW_QUEUE.clear()
            if final_image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image_genere", translations))
                gr.Warning(translate("erreur_pas_image_genere", translations), 4.0)
                yield images, formatted_seeds, f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}", translate("erreur_pas_image_genere", translations), None
                continue
            
            temps_generation_image = f"{(time.time() - depart_time):.2f} sec"            
            

            date_str = datetime.now().strftime("%Y_%m_%d")
            heure_str = datetime.now().strftime("%H_%M_%S")
            save_dir = os.path.join(SAVE_DIR, date_str)
            os.makedirs(save_dir, exist_ok=True)
          
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
                    "VAE": os.path.splitext(vae_selctionne)[0],
                    "Sampler": pipe.scheduler.__class__.__name__,
                    "Loras": active_adapters,
                    "Poids Lora":lora_scale,
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
            if html_message.done():
                html_message_result = html_message.result()  # Get the result
            elif not html_message.done():
                # If the task is not done, display the error message
                 html_message_result = translate("erreur_lors_generation_html", translations)
                       
            yield images, formatted_seeds, f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}", html_message_result, final_image# return the all list image
                   
        
        elapsed_time = f"{(time.time() - start_time):.2f} sec"
             
        print(txt_color("[INFO] ","info"),f"{translate('temps_total_generation', translations)} : {elapsed_time}")
        gr.Info(translate('temps_total_generation', translations) + " : " + elapsed_time, 3.0)
        #we return all the seed list
        return images, formatted_seeds, elapsed_time

    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f"{translate('erreur_lors_generation', translations)} : {e}")
        raise gr.Error (f"{translate('erreur_lors_generation', translations)} : {e}", 4.0)
        return None, None, f"{translate('erreur_lors_generation', translations)} : {e}"



def generate_inpainted_image(text, image, mask, num_steps, guidance_scale, traduire):
    """Génère une image inpainted avec Stable Diffusion XL."""
    global pipe, compel
    try:
        start_time = time.time()
        if pipe is None:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_modele_inpainting", translations))
            gr.Warning(translate("erreur_pas_modele_inpainting", translations), 4.0)
            return None, translate("erreur_pas_modele_inpainting", translations)

        if image is None or mask is None:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_image_mask_manquant", translations))
            gr.Warning(translate("erreur_image_mask_manquant", translations), 4.0)
            return None, translate("erreur_image_mask_manquant", translations)

        # Translate the prompt if requested
        prompt_text = translate_prompt(text) if traduire else text
        conditioning, pooled = compel(prompt_text)
        active_adapters = pipe.get_active_adapters()
        for adapter_name in active_adapters:
            pipe.set_adapters(adapter_name, 0)
        
        image = Image.fromarray(image).convert("RGB") # Convert to PIL Image
        mask = mask.convert("RGB")
        mask = ImageOps.invert(mask)
        final_image_container = {}

        # Run the inpainting pipeline
        def run_pipeline():
            print(txt_color("[INFO] ", "info"), translate("debut_inpainting", translations))
            gr.Info(translate("debut_inpainting", translations), 3.0)
            
            inpainted_image = pipe(
                pooled_prompt_embeds=pooled,
                prompt_embeds=conditioning,
                image=image.convert("RGB"),         # PIL Image
                mask_image=mask.convert("RGB"),     # PIL Image
                width=image.width,                  # Largeur de l'image
                height=image.height,                # Hauteur de l'image
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=0.89,
            ).images[0]
            final_image_container["final"] = inpainted_image

        thread = threading.Thread(target=run_pipeline)
        thread.start()
        thread.join()
        inpainted_image = final_image_container.get("final", None)
        
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

        print(txt_color("[OK] ", "ok"), translate("fin_inpainting", translations))
        gr.Info(translate("fin_inpainting", translations), 3.0)

        return inpainted_image, html_message.result(), translate("inpainting_reussi", translations)        

    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_lors_inpainting', translations)}: {e}")
        raise gr.Error(f"{translate('erreur_lors_inpainting', translations)}: {e}", 4.0)
        return None, f"{translate('erreur_lors_inpainting', translations)}: {e}"




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
    model_selectionne = nom_fichier
    vae_selctionne = nom_vae
    return message

def update_globals_model_inpainting(nom_fichier):
    global pipe, compel, model_selectionne 
    pipe, compel, message = charger_modele_inpainting(nom_fichier, translations, INPAINT_MODELS_DIR, device, torch_dtype, vram_total_gb, pipe, compel) 
    model_selectionne = nom_fichier
    return message

def update_globals_lora(nom_lora):
    global pipe
    message = charger_lora(nom_lora, pipe, LORAS_DIR, translations)
    return message

def update_globals_decharge_lora():
    global pipe
    message = decharge_lora(pipe, translations)
    return message


# =========================
# Interface utilisateur (Gradio)
# =========================

block_kwargs = {"theme": gradio_change_theme(GRADIO_THEME)}
if js_code:
    block_kwargs["js"] = js_code


with gr.Blocks(**block_kwargs) as interface:
     gr.Markdown(f"# Cyberbill SDXL images generator version {version()}")


############################################################
########TAB GENRATION IAMGE
############################################################   
     with gr.Tab(translate("generation_image", translations)):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                text_input = gr.Textbox(label=translate("prompt", translations), info=translate("entrez_votre_texte_ici", translations))
                traduire_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", translations), value=False, info=translate("traduire_en_anglais", translations))
                style_dropdown = gr.Dropdown(choices=[style["name"] for style in STYLES], value=translate("Aucun_style", translations), label=translate("styles", translations), info=translate("Selectionnez_un_style_predefini", translations))
                use_image_checkbox = gr.Checkbox(label=translate("generer_prompt_image", translations), value=False)
                image_input = gr.Image(label=translate("telechargez_image", translations), type="pil", visible=False)
                use_image_checkbox.change(fn=lambda use_image: gr.update(visible=use_image), inputs=use_image_checkbox, outputs=image_input)
                
                guidance_slider = gr.Slider(1, 20, value=7, label=translate("guidage", translations))
                num_steps_slider = gr.Slider(1, 50, value=30, label=translate("etapes", translations), step=1)
                format_dropdown = gr.Dropdown(choices=FORMATS, value=FORMATS[3], label=translate("format", translations))
                seed_input = gr.Number(label=translate("seed", translations), value=-1)
                num_images_slider = gr.Slider(1, 200, value=1, label=translate("nombre_images_generer", translations), step=1)
                
                with gr.Row():
                   #enhance_checkbox = gr.Checkbox(label=translate("enhance_image", translations), value=False, visible=photo_editing_loaded)
                    btn_stop = gr.Button(translate("arreter", translations))
                    btn_stop_after_gen = gr.Button(translate("stop_apres_gen", translations))
                    btn_generate = gr.Button(translate("generer", translations))
                
            with gr.Column(scale=2, min_width=300):
                image_output = gr.Gallery(label=translate("images_generees", translations))
                
                with gr.Row():
                    with gr.Column():
                        preview_image_output = gr.Image(height=170, label=translate("apercu_etapes", translations),interactive=False)
                        value = DEFAULT_MODEL if DEFAULT_MODEL else None
                        modele_dropdown = gr.Dropdown(label=translate("selectionner_modele", translations), choices=modeles_disponibles, value=value)
                        vae_dropdown = gr.Dropdown(label=translate("selectionner_vae", translations), choices=vaes, value=value)
                        sampler_dropdown = gr.Dropdown(label=translate("selectionner_sampler", translations), choices=sampler_options)
                        bouton_lister = gr.Button(translate("lister_modeles", translations))
                        bouton_charger = gr.Button(translate("charger_modele", translations))
                        
                    with gr.Column():
                        seed_output = gr.Textbox(label=translate("seed_utilise", translations))
                        time_output = gr.Textbox(label=translate("temps_rendu", translations), interactive=False)
                        html_output = gr.Textbox(label=translate("mise_a_jour_html", translations), interactive=False)
                        message_chargement = gr.Textbox(label=translate("statut", translations), value=translate("aucun_modele_charge", translations))
                        use_lora_checkbox = gr.Checkbox(label=translate("utiliser_lora", translations), value=False)

                        with gr.Column(visible=False) as lora_section:
                            lora_dropdown = gr.Dropdown(choices=["Aucun LORA disponible"], label=translate("selectionner_lora", translations))
                            lora_scale_slider = gr.Slider(0, 1, value=0, label=translate("poids_lora", translations))
                            load_lora_button = gr.Button(translate("charger_lora", translations))
                            unload_lora_button = gr.Button(translate("decharger_lora", translations))
                            lora_message = gr.Textbox(label=translate("message_lora", translations), value="")

                def mettre_a_jour_listes():
                    modeles = lister_fichiers(MODELS_DIR, translations, gradio_mode=True)
                    vaes = ["Défaut VAE"] + lister_fichiers(VAE_DIR, translations, gradio_mode=True)
                    loras = lister_fichiers(LORAS_DIR, translations, gradio_mode=True)

                    has_loras = bool(loras) and translate("aucun_modele_trouve",translations) not in loras and translate("repertoire_not_found",translations) not in loras
                    lora_choices = loras if has_loras else ["Aucun LORA disponible"]

                    return (
                        gr.update(choices=modeles),
                        gr.update(choices=vaes),
                        gr.update(choices=lora_choices, interactive=has_loras),
                        gr.update(interactive=has_loras, value=False),
                        gr.update(value=translate("lora_trouve", translations) + ", ".join(loras) if has_loras else translate("aucun_lora_disponible", translations))
                    )

                bouton_lister.click(
                    fn=mettre_a_jour_listes,
                    outputs=[modele_dropdown, vae_dropdown, lora_dropdown, use_lora_checkbox, lora_message]
                )

                use_lora_checkbox.change(
                    lambda use_lora: gr.update(visible=use_lora),
                    inputs=use_lora_checkbox,
                    outputs=lora_section,
                )
                
                load_lora_button.click(
                    fn=update_globals_lora,
                    inputs=[lora_dropdown],
                    outputs=lora_message
                )

                unload_lora_button.click(
                    fn=update_globals_decharge_lora,
                    outputs=lora_message
                )

                bouton_charger.click(
                    fn=update_globals_model,
                    inputs=[modele_dropdown, vae_dropdown],
                    outputs=message_chargement
                )
                sampler_dropdown.change(fn=apply_sampler, inputs=sampler_dropdown, outputs=message_chargement)

        image_input.change(fn=generate_caption, inputs=image_input, outputs=text_input)

        btn_generate.click(
            generate_image, 
            inputs=[text_input, style_dropdown, guidance_slider, num_steps_slider, format_dropdown, traduire_checkbox, seed_input, num_images_slider, lora_scale_slider], #enhance_checkbox],  
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
        with gr.Row():
            with gr.Column():
                image_mask_input = gr.ImageMask(label=translate("image_avec_mask", translations))
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
                bouton_generate_inpainting = gr.Button(translate("generer_inpainting", translations))
            with gr.Column():
                mask_image_output = gr.Image(type="pil", label=translate("sortie_mask_inpainting", translations), interactive=False)
                inpainting_ressultion_output = gr.Image(type="pil", label=translate("sortie_inpainting", translations),interactive=False)
                modele_inpainting_dropdown = gr.Dropdown(label=translate("selectionner_modele_inpainting", translations), choices=modeles_impaint, value=value)
                bouton_lister_inpainting = gr.Button(translate("lister_modeles_inpainting", translations))
                bouton_charger_inpainting = gr.Button(translate("charger_modele_inpainting", translations))
                message_chargement_inpainting = gr.Textbox(label=translate("statut_inpainting", translations), value=translate("aucun_modele_charge_inpainting", translations))
                message_inpainting = gr.Textbox(label=translate("message_inpainting", translations), interactive=False)
            
            opacity_slider.change(
                fn=lambda image_and_mask, opacity, blur_radius: apply_mask_effects(image_and_mask, translations, opacity, blur_radius) if image_and_mask else None,
                inputs=[image_mask_input, opacity_slider, blur_slider],
                outputs=[mask_image_output],
            )

            blur_slider.change(
                fn=lambda image_and_mask, opacity, blur_radius: apply_mask_effects(image_and_mask, translations, opacity, blur_radius) if image_and_mask else None,
                inputs=[image_mask_input, opacity_slider, blur_slider],
                outputs=[mask_image_output],
            )
            image_mask_input.change(
                fn=lambda image_and_mask, opacity, blur_radius: apply_mask_effects(image_and_mask, translations, opacity, blur_radius) if image_and_mask else None,
                inputs=[image_mask_input, opacity_slider, blur_slider],
                outputs=[mask_image_output],
            )

            bouton_lister_inpainting.click(
                fn=mettre_a_jour_listes_inpainting,
                outputs=modele_inpainting_dropdown
            )

            bouton_charger_inpainting.click(
                fn=update_globals_model_inpainting,
                inputs=[modele_inpainting_dropdown],
                outputs=message_chargement_inpainting
            )
            bouton_generate_inpainting.click(
                fn=lambda text, image_and_mask, mask, num_steps, guidance_scale, traduire: generate_inpainted_image(text, image_and_mask["background"], mask, num_steps, guidance_scale, traduire),
                inputs=[inpainting_prompt, image_mask_input, mask_image_output, num_steps_inpainting_slider, guidance_inpainting_slider, traduire_inpainting_checkbox],
                outputs=[inpainting_ressultion_output, message_chargement_inpainting, message_inpainting]
            )
                    
############################################################
########TAB MODULES
############################################################
     gestionnaire.creer_tous_les_onglets(translations)


interface.launch(inbrowser=str_to_bool(OPEN_BROWSER), pwa=True, share=str_to_bool(SHARE))
