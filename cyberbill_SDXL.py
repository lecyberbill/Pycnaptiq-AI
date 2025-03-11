import random
import os
import time
import threading
from datetime import datetime
import json
import gradio as gr
from diffusers import AutoPipelineForText2Image,StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, \
    LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, KDPM2DiscreteScheduler, \
    KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, DPMSolverSDEScheduler
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from utils import enregistrer_etiquettes_image_html,charger_configuration, gradio_change_theme, lister_fichiers, \
    telechargement_modele, txt_color, str_to_bool, load_locales, translate, get_language_options, enregistrer_image
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from version import version
from compel import Compel, ReturnedEmbeddingsType


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
SAVE_DIR = config["SAVE_DIR"]
IMAGE_FORMAT = config["IMAGE_FORMAT"].upper() 
FORMATS = config["FORMATS"]
NEGATIVE_PROMPT = config["NEGATIVE_PROMPT"]
GRADIO_THEME = config["GRADIO_THEME"]
AUTHOR= config["AUTHOR"]
SHARE= config["SHARE"]
OPEN_BROWSER= config["OPEN_BROWSER"]
DEFAULT_MODEL= config["DEFAULT_MODEL"]
for style in config["STYLES"]:
        style["name"] = translate(style["key"], translations)
STYLES=config["STYLES"]


#initialisation des modèles

modeles_disponibles = lister_fichiers(MODELS_DIR, translations)


# initialisation de ccompel



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


# Vérifier que le format est valide
if IMAGE_FORMAT not in ["PNG", "JPG", "WEBP"]:
    print(f"⚠️ Format {IMAGE_FORMAT}", f"{txt_color(translate('non_valide', translations), 'erreur')}", translate("utilisation_webp", translations))
    IMAGE_FORMAT = "WEBP"

 # Créer un pool de threads pour l'écriture asynchrone

html_executor = ThreadPoolExecutor(max_workers=5)
image_executor = ThreadPoolExecutor(max_workers=10)


# Vérifier si CUDA est disponible
if torch.cuda.is_available():
    gpu_id = 0  # ID du GPU (ajuste si nécessaire)
    vram_total = torch.cuda.get_device_properties(gpu_id).total_memory  # en octets
    vram_total_gb = vram_total / (1024 ** 3)  # conversion en Go

    print(translate("vram_detecte", translations), f"{txt_color(f'{vram_total_gb:.2f} Go', 'info')}")

    # Activer expandable_segments si VRAM < 10 Go
    if vram_total_gb < 10:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True max_split_size_mb:512"
        medvram = True
        print(translate("pytroch_active", translations))
    # Détermination du device et du type de données
    device = "cuda"
    torch_dtype = torch.float16
else:
    print(txt_color(translate("cuda_dispo", translations),"erreur"))
    device = "cpu"
    torch_dtype = torch.float32

print(txt_color(f'{translate("utilistation_device", translations)} : {device} + dtype {torch_dtype}','info'))

# Initialisation des variables globales
pipe = None
model_selectionne = None
vae_selctionne = None
compel = None
# Flag pour arrêter la génération
stop_event = threading.Event()

# Flag pour signaler qu'une tâche est en cours
processing_event = threading.Event()


# Chargement du modèle de traduction
translation_model = "Helsinki-NLP/opus-mt-fr-en"
translator = pipeline("translation", model=translation_model)
print(txt_color("[OK]","ok"), f"{translate('modele_traduction_charge',translations)} '{translation_model}' {translate('charge_model', translations)}")


# Charger le modèle et le processeur
caption_model = AutoModelForCausalLM.from_pretrained(
    "MiaoshouAI/Florence-2-base-PromptGen-v2.0", trust_remote_code=True
).to(device)
caption_processor = AutoProcessor.from_pretrained(
    "MiaoshouAI/Florence-2-base-PromptGen-v2.0", trust_remote_code=True
)

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
            return info
        elif sampler_selection == translate("sampler_dpmpp_2m", translations):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_dpmpp_2s_a", translations):
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_lms", translations):
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_ddim", translations):
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_pndm", translations):
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_dpm2", translations):
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_dpm2_a", translations):
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_dpm_fast", translations):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_dpm_adaptive", translations):
            pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_heun", translations):
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_dpmpp_sde", translations):
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)  # Note: Utilisation de DPMSolverSDEScheduler
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_euler_a", translations):
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        elif sampler_selection == translate("sampler_unipc", translations):
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ", "ok"), info)
            return info
        else:
            print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_sampler_inconnu', translations)} {sampler_selection}")  # Utilisation de la clé de traduction
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

def charger_modele(nom_fichier, nom_vae):
    """Charge un modèle spécifique."""
    
    # Importation global non recommandée, mais ici utilisée pour simplifier l'exemple
    global pipe, model_selectionne, vae_selctionne, compel
    
    # Si aucun modèle n'est sélectionné, affiche un message et retourne
    if nom_fichier is None or nom_fichier == translate("aucun_modele_trouve", translations) or not nom_fichier:
        print(txt_color("[ERREUR] ","erreur"),translate("aucun_modele_selectionne", translations))
        return translate("aucun_modele_selectionne", translations)+ translate("verifier_config", translations)

    if nom_vae is None or nom_vae == "Défaut VAE" or not nom_vae:
        nom_vae = "Défaut VAE"
    
    # Construit le chemin vers le fichier de modèle en se basant sur la constante MODELS_DIR et le nom du fichier
    chemin_modele = os.path.join(MODELS_DIR, nom_fichier)
    chemin_vae = os.path.join(VAE_DIR, nom_vae)
    
    # Si une pipe est déjà chargée, on la supprime pour libérer de la mémoire GPU et éviter les conflits
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
        print(txt_color("[OK] ","ok"),translate("modele_precedent_decharger", translations))
    
    # Essaye de charger le modèle à partir du fichier spécifié
    try:
        print(txt_color("[INFO] ","info"),f"{translate('chargement_modele',translations)} : {nom_fichier} {translate('chargement_vae',translations)} : {nom_vae}")
        
        # Charge le modèle et met à jour la variable globale 'pipe'
        pipe = StableDiffusionXLPipeline.from_single_file(
            chemin_modele,
            use_safetensors=True,            
            safety_checker=None, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True if (device == "cuda" and vram_total_gb < 10) else False,
            load_device=device
         )
         
        pipe = AutoPipelineForText2Image.from_pipe(pipe)
                         
        if not nom_vae == "Défaut VAE":
            pipe.vae = vae=AutoencoderKL.from_single_file(chemin_vae, torch_dtype=torch_dtype)
            print (txt_color("[OK] ","ok"),translate("vae_charge", translations), f"{nom_vae}")
        else:
            print (txt_color("[INFO] ","info"), translate("aucun_vae_selectionne", translations))
            
        # 
        # Si le dispositif est GPU, met le modèle dans l'espace de stockage GPU
        
        pipe = pipe.to(device) if device == "cuda" else pipe
        
        if device == "cuda" and vram_total_gb < 10:
        # Attention slicing : permet de découper le calcul de l'attention
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            pipe.enable_xformers_memory_efficient_attention()
            print(txt_color("[INFO] ","info"), translate("optimisation_attention", translations))
        
        # Met à jour le nom du modèle sélectionné et retourne un message de succès
        model_selectionne = nom_fichier
        vae_selctionne = nom_vae
        
        # initialisation de compel
        compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
        )
        print(txt_color("[OK] ","ok"),translate("modele_charge", translations), f": {nom_fichier}")
        return translate("modele_charge", translations), f": {nom_fichier} ", f"{translate('vae_charge', translations)} {nom_vae}"

    # Si une erreur se produit lors du chargement, retourne un message d'erreur
    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"),translate("erreur_chargement_modele", translations), f": {e}")
        return translate("erreur_chargement_modele", translations), f": {e}"


def charger_lora(nom_lora):
    # décharge le LORA avant de charger un nouveau LORA
    pipe.disable_lora()
    pipe.enable_lora()
    lora_path = os.path.join(LORAS_DIR, nom_lora)

    if not os.path.exists(lora_path):
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_fichier_lora", translations))
        return translate("erreur_fichier_lora", translations)

    if pipe is None:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_modele", translations))
        return translate("erreur_pas_modele", translations)

    adapter_nom = os.path.splitext(nom_lora)[0]
    adapter_nom = adapter_nom.replace(".", "_")
    try:
        print(txt_color("[INFO] ", "info"), translate("lora_charge_depuis", translations), f"{lora_path}")
        # pipe.load_lora_weights(lora_path, weight_name=nom_lora)  # Charger le LORA
        pipe.load_lora_weights(lora_path, weight_name=nom_lora, adapter_name=adapter_nom)  # Charger le LORA
        print(txt_color("[OK] ", "ok"), translate("lora_charge", translations), f"{adapter_nom}")
        return translate("lora_charge", translations), f" {adapter_nom}"
    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_lora_chargement", translations), f": {e}")
        return translate("lora_non_compatible", translations)



#générer prompt à partir d'une image

def generate_caption(image):
    """generate a prompt from an image."""

    if image:
        # Préparer les entrées
        inputs = caption_processor(text="<GENERATE_TAGS>", images=image, return_tensors="pt").to(device)
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
            generated_text, task="<GENERATE_TAGS>", image_size=(image.width, image.height)
        )

        # Libérer la mémoire GPU
        torch.cuda.empty_cache()
        prompt = parsed_answer.get('<GENERATE_TAGS>', '').strip('{}').strip('"')
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
        return traduction
    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_traduction', translations)}: {e}")
        return f"{translate('erreur_traduction', translations)}: {e}"


def generate_image(text, style_selection, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images, lora_scale):
    """Génère des images avec Stable Diffusion."""
    global compel
    
    selected_style = style_choice(style_selection, STYLES)

    try:
        
        if pipe is None:
            print(txt_color("[ERREUR] ","erreur"), translate("erreur_pas_modele", translations))
            return None, None, translate("erreur_pas_modele", translations)
        
        #initialisation du chrono
        start_time = time.time()
        # Réinitialiser l'état d'arrêt
        stop_event.clear()  
        
        seeds = [random.randint(1, 10**19 - 1) for _ in range(num_images)] if seed_input == -1 else [seed_input] * num_images
        
        prompt_text = translate_prompt(text) if traduire else text
        prompt_en = selected_style["prompt"].replace("{prompt}", prompt_text) if selected_style else text

        if selected_style:
            negative_prompt = selected_style["negative_prompt"] if selected_style["name"] != translate("Aucun_style", translations) else NEGATIVE_PROMPT
        else:
            negative_prompt = NEGATIVE_PROMPT
    
        
        conditioning, pooled = compel(prompt_en)  
        
        width, height = map(int, selected_format.split("*"))
        images = [] # Initialize a list to store all generated images
        
        active_adapters = pipe.get_active_adapters()
        
        for adapter_name in active_adapters:
            pipe.set_adapters(adapter_name, lora_scale)

            

        # NEW: Initialize an empty list to store seed values as strings
        seed_strings = []
        for idx, seed in enumerate(seeds):
            depart_time = time.time()
            if stop_event.is_set():  # Vérifie si un arrêt est demandé
                html_message = html_executor.submit(enregistrer_etiquettes_image_html, chemin_image, donnees_xmp, translations, is_last_image=True)
                print(txt_color("[INFO] ","info"), translate("arrete_demande_apres", translations), f"{idx} {translate('images', translations)}.")
                break
                
            print(txt_color("[INFO] ","info"), f"{translate('generation_image', translations)} {idx+1} {translate('seed_utilise', translations)} {seed}")
            generator = torch.Generator(device=device).manual_seed(seed)
            generated_image = pipe(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt, # Fournir le prompt négatif textuel
                generator=generator,
                width=width,
                height=height
            ).images[0]
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
            
            # sauvegarde de l'image
            # enregistrer_image(generated_image, chemin_image, donnees_xmp, translations, IMAGE_FORMAT)
            # Déléguer la tâche d'enregistrement de l'image au ThreadPoolExecutor
            image_future = image_executor.submit(enregistrer_image, generated_image, chemin_image, donnees_xmp, translations, IMAGE_FORMAT)
                        
            
            is_last_image = (idx == num_images - 1)
            # Déléguer la tâche d'écriture dans le ThreadPoolExecutor
            html_message = html_executor.submit(enregistrer_etiquettes_image_html, chemin_image, donnees_xmp, translations, is_last_image)
            
            print(txt_color("[OK] ","ok"),translate("image_sauvegarder", translations), txt_color(f"{filename}","ok"))
            images.append(generated_image) # Append each generated image to the list
            torch.cuda.empty_cache()

            # Append the seed value as a string to the list, formatted as [seed]
            seed_strings.append(f"[{seed}]")
            # join the string with a space
            formatted_seeds = " ".join(seed_strings)
            if html_message.done() :
                # Récupérer le résultat
                html_message_result = html_message.result()
            else:
                html_message_result = translate("erreur_lors_generation_html", translations)
                       
            yield images, formatted_seeds, f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}", html_message_result # return the all list image
            
        
   
        elapsed_time = f"{(time.time() - start_time):.2f} sec"
        
        
        print(txt_color("[INFO] ","info"),f"{translate('temps_total_generation', translations)} : {elapsed_time}")
        #we return all the seed list
        return images, formatted_seeds, elapsed_time

    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f"{translate('erreur_lors_generation', translations)} : {e}")
        return None, None, f"{translate('erreur_lors_generation', translations)} : {e}"


def stop_generation():
    """Déclenche l'arrêt de la génération"""
    stop_event.set()
    return translate("arreter", translations)


def decharge_lora():
    # Récupère la liste des adaptateurs actifs
    active_adapters = pipe.get_active_adapters()
    
    # Itère sur chaque adaptateur et le désactive
    for adapter_name in active_adapters:
        pipe.delete_adapters(adapter_name)
        print(txt_color("[INFO] ","info"), f"{translate('lora_decharge_nom', translations)} '{adapter_name}'")
    
    return translate("lora_decharge", translations)

#fonction de retouche d'image 

def apply_filters(image, contrast, saturation, color_boost, grayscale, blur_radius, sharpness_factor, rotation_angle, mirror_type):
    if image is None:
        return None
    
    img = Image.fromarray(image)
    
    if grayscale:
        img = img.convert("L")
    
    # Appliquer les transformations géométriques
    if rotation_angle != 0:
        img = img.rotate(rotation_angle, expand=True)  # expand=True pour éviter le recadrage

    if mirror_type == "horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif mirror_type == "vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
    # Appliquer le flou (si un rayon de flou est spécifié)
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Appliquer la netteté (si un facteur de netteté est spécifié)
    if sharpness_factor > 0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness_factor)
    
    # Appliquer le contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    
    # Appliquer la saturation (uniquement si l'image est en couleur)
    if not grayscale:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)
    
    # Intensifier les couleurs (boost général)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(color_boost)
    
    return np.array(img)

# =========================
# Chargement d'un modèle avant chargement interface
# =========================


if DEFAULT_MODEL in modeles_disponibles:
    print(f"{txt_color('[INFO]','info')}", f"{DEFAULT_MODEL} {translate('va_se_charger', translations)} {MODELS_DIR}")
    charger_modele(DEFAULT_MODEL, "Défaut VAE")

else:
    print(f"{txt_color('[INFO]','info')}", f"{DEFAULT_MODEL} {translate('pas_de_modele_dans', translations)} : {MODELS_DIR}")
    DEFAULT_MODEL=""


# =========================
# Interface utilisateur (Gradio)
# =========================
with gr.Blocks(theme=gradio_change_theme(GRADIO_THEME)) as interface:
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label=translate("prompt", translations), info=translate("entrez_votre_texte_ici", translations))
            traduire_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", translations), value=False, info=translate("traduire_en_anglais", translations))
            style_dropdown = gr.Dropdown(choices=[style["name"] for style in STYLES], value=translate("Aucun_style", translations), label=translate("styles", translations), info=translate("Selectionnez_un_style_predefini", translations))
            use_image_checkbox = gr.Checkbox(label=translate("generer_prompt_image", translations), value=False)
            image_input = gr.Image(label=translate("telechargez_image", translations), type="pil", visible=False)
            use_image_checkbox.change(fn=lambda use_image: gr.update(visible=use_image), inputs=use_image_checkbox, outputs=image_input)
            
            guidance_slider = gr.Slider(1, 20, value=7, label=translate("guidage", translations))
            num_steps_slider = gr.Slider(1, 50, value=30, label=translate("etapes", translations), step=1)
            format_dropdown = gr.Dropdown(choices=FORMATS, value="768*1280", label=translate("format", translations))
            seed_input = gr.Number(label=translate("seed", translations), value=-1)
            num_images_slider = gr.Slider(1, 200, value=1, label=translate("nombre_images_generer", translations), step=1)
            
            with gr.Row():
                btn_stop = gr.Button(translate("arreter", translations))
                btn_generate = gr.Button(translate("generer", translations))

        with gr.Column():
            image_output = gr.Gallery(label=translate("images_generees", translations))
            seed_output = gr.Textbox(label=translate("seed_utilise", translations))
            time_output = gr.Textbox(label=translate("temps_rendu", translations), interactive=False)
            html_output = gr.Textbox(label=translate("mise_a_jour_html", translations), interactive=False)

            bouton_lister = gr.Button(translate("lister_modeles", translations))
            value = DEFAULT_MODEL if DEFAULT_MODEL else None
            modele_dropdown = gr.Dropdown(label=translate("selectionner_modele", translations), choices=modeles_disponibles, value=value)
            vae_dropdown = gr.Dropdown(label=translate("selectionner_vae", translations), choices=["Défaut VAE"], value="Défaut VAE")
            sampler_dropdown = gr.Dropdown(label=translate("selectionner_sampler", translations), choices=sampler_options)
            
            bouton_charger = gr.Button(translate("charger_modele", translations))
            message_chargement = gr.Textbox(label=translate("statut", translations), value=translate("aucun_modele_charge", translations))

            use_lora_checkbox = gr.Checkbox(label=translate("utiliser_lora", translations), value=False)

            with gr.Column(visible=False) as lora_section:
                lora_dropdown = gr.Dropdown(choices=["Aucun LORA disponible"], label=translate("selectionner_lora", translations))
                lora_scale_slider = gr.Slider(0, 1, value=0, label=translate("poids_lora", translations))
                load_lora_button = gr.Button(translate("charger_lora", translations))
                unload_lora_button = gr.Button(translate("decharger_lora", translations))
                lora_message = gr.Textbox(label=translate("message_lora", translations), value="")

            def mettre_a_jour_listes():
                modeles = lister_fichiers(MODELS_DIR, translations)
                vaes = ["Défaut VAE"] + lister_fichiers(VAE_DIR, translations)
                loras = lister_fichiers(LORAS_DIR, translations)

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
                fn=charger_lora,  
                inputs=[lora_dropdown],  
                outputs=lora_message  
            )
            
            unload_lora_button.click(
                fn=decharge_lora,   
                outputs=lora_message  
            )

            bouton_charger.click(fn=charger_modele, inputs=[modele_dropdown, vae_dropdown], outputs=message_chargement)
            sampler_dropdown.change(fn=apply_sampler, inputs=sampler_dropdown, outputs=message_chargement)

    image_input.change(fn=generate_caption, inputs=image_input, outputs=text_input)

    btn_generate.click(
        generate_image, 
        inputs=[text_input, style_dropdown, guidance_slider, num_steps_slider, format_dropdown, traduire_checkbox, seed_input, num_images_slider, lora_scale_slider],  
        outputs=[image_output, seed_output, time_output, html_output]
    )

    btn_stop.click(stop_generation, outputs=time_output)

    edit_section_checkbox = gr.Checkbox(label=translate("activer_la_retouche_dimage", translations), value=False)

    with gr.Row(visible=False) as edit_section:
        image_input = gr.Image(label=translate("selectionner_une_image", translations), type="numpy")
        image_output = gr.Image(label=translate("apercu_des_modifications", translations), type="numpy")

    with gr.Row(visible=False) as edit_controls:
        with gr.Column():
            contrast = gr.Slider(0.5, 2.0, 1.0, step=0.1, label=translate("contraste", translations))
            saturation = gr.Slider(0.5, 2.0, 1.0, step=0.1, label=translate("saturation", translations))
            color_boost = gr.Slider(0.5, 2.0, 1.0, step=0.1, label=translate("intensite_des_couleurs", translations))
            blur_radius = gr.Slider(0, 10, 0, step=1, label=translate("rayon_de_flou", translations))
            sharpness_factor = gr.Slider(0, 5, 1, step=0.1, label=translate("facteur_de_nettete", translations))

        with gr.Column():
            grayscale = gr.Checkbox(label=translate("noir_et_blanc", translations))
            rotation_angle = gr.Slider(0, 360, 0, step=90, label=translate("angle_de_rotation_90", translations))
            mirror_type = gr.Dropdown(choices=["aucun", "horizontal", "vertical"], value="aucun", label=translate("type_de_miroir", translations))

    edit_section_checkbox.change(
        lambda visible: [gr.update(visible=visible)] * 2, 
        inputs=edit_section_checkbox, 
        outputs=[edit_section, edit_controls]
    )

    inputs = [image_input, contrast, saturation, color_boost, grayscale, blur_radius, sharpness_factor, rotation_angle, mirror_type]
    for inp in inputs:
        inp.change(apply_filters, inputs=inputs, outputs=image_output)

interface.launch(inbrowser=str_to_bool(OPEN_BROWSER), pwa=True, share=str_to_bool(SHARE))