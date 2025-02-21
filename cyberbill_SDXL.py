import gradio as gr
from diffusers import AutoPipelineForText2Image,StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, \
    LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, KDPM2DiscreteScheduler, \
    KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, DPMSolverSDEScheduler
import torch
import random
import os
import time
import threading
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from utils import fichier_recap, enregistrer_etiquettes_image_html,charger_configuration, gradio_change_theme, lister_fichiers, \
    telechargement_modele, txt_color, str_to_bool
from version import version
from concurrent.futures import ThreadPoolExecutor
import json



# =========================
# Initialisation des variables
# =========================


print (f"cyberbill_SDXL version {txt_color(version(),'info')}")
config = charger_configuration()
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

#initialisation des modèles 

modeles_disponibles = lister_fichiers(MODELS_DIR)



if "Aucun modèle trouvé." in modeles_disponibles:
    reponse = input(f"{txt_color('Attention','erreur')} Aucun modèle trouvé. Voulez-vous télécharger un modèle maintenant (O/N) oui ou non ? ")
    if reponse.lower() == "O":
        lien_modele = "https://huggingface.co/QuadPipe/MegaChonkXL/resolve/main/MegaChonk-XL-v2.3.1.safetensors?download=true"
        nom_fichier = "MegaChonk-XL-v2.3.1.safetensors"
        if telechargement_modele(lien_modele, nom_fichier,MODELS_DIR):
            # Recharge la liste des modèles après le téléchargement
            modeles_disponibles = lister_fichiers(MODELS_DIR)
            if not modeles_disponibles:
                modeles_disponibles = ["Aucun modèle trouvé."]


# Vérifier que le format est valide
if IMAGE_FORMAT not in ["PNG", "JPG", "WEBP"]:
    print(f"⚠️ Format {IMAGE_FORMAT}", f"{txt_color('invalide', 'erreur')}", "utilisation de WEBP par défaut.")
    IMAGE_FORMAT = "WEBP"

# Créer un pool de threads pour l'écriture asynchrone
executor = ThreadPoolExecutor(max_workers=4)

# Vérifier si CUDA est disponible
if torch.cuda.is_available():
    gpu_id = 0  # ID du GPU (ajuste si nécessaire)
    vram_total = torch.cuda.get_device_properties(gpu_id).total_memory  # en octets
    vram_total_gb = vram_total / (1024 ** 3)  # conversion en Go

    print(f"VRAM détectée : {txt_color(f'{vram_total_gb:.2f} Go', 'info')}")

    # Activer expandable_segments si VRAM < 10 Go
    if vram_total_gb < 10:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True max_split_size_mb:512"
        medvram = True
        print("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True activé")
    # Détermination du device et du type de données
    device = "cuda"
    torch_dtype = torch.float16
else:
    print(txt_color("CUDA non disponible, exécution sur CPU","erreur"))
    device = "cpu"
    torch_dtype = torch.float32

print(txt_color(f'Utilisation de : {device} avec dtype {torch_dtype}','info'))

# Initialisation des variables globales
pipe = None
model_selectionne = None
vae_selctionne = None
# Flag pour arrêter la génération
stop_event = threading.Event()

# Chargement du tokenizer pour compter les tokens
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Chargement du modèle de traduction
translation_model = "Helsinki-NLP/opus-mt-fr-en"
translator = pipeline("translation", model=translation_model)
print(txt_color("[OK]","ok"), f"Modèle de traduction '{translation_model}' chargé")


# Charger le modèle et le processeur
caption_model = AutoModelForCausalLM.from_pretrained(
    "MiaoshouAI/Florence-2-base-PromptGen-v2.0", trust_remote_code=True
).to(device)
caption_processor = AutoProcessor.from_pretrained(
    "MiaoshouAI/Florence-2-base-PromptGen-v2.0", trust_remote_code=True
)

# Sampler disponibles
sampler_options = [
    "EulerDiscreteScheduler (Rapide et détaillé)",
    "DPM++ 2M Karras (Photoréaliste et détaillé)",
    "Euler Ancestral (Artistique et fluide)",
    "LMSDiscreteScheduler (Équilibré et polyvalent)",
    "DDIMScheduler (Rapide et créatif)",
    "PNDMScheduler (Stable et photoréaliste)",
    "KDPM2DiscreteScheduler (Détaillé et net)",
    "KDPM2AncestralDiscreteScheduler (Artistique et net)",
    "DPMSolverMultistepScheduler (Rapide et de haute qualité)",
    "DEISMultistepScheduler (Excellent pour les détails fins)",
    "HeunDiscreteScheduler (Bon compromis vitesse/qualité)",
    "DPM++ SDE Karras (Photoréaliste et avec réduction du bruit)",
    "DPM++ 2M SDE Karras (Combine photoréalisme et réduction du bruit)",
    "Euler A (Euler Ancestral, version abrégée)",
    "LMS (Linear Multistep Method, version abrégée)",
    "PLMS (P-sampler - Pseudo Linear Multistep Method)"
]

# =========================
# Définition des fonctions
# =========================

def enregistrer_image(image, chemin_image, donnees_xmp):
    """Enregistre l'image et écrit les métadonnées."""
    try:
        image.save(chemin_image, format=IMAGE_FORMAT)
        enregistrer_etiquettes_image_html(chemin_image, donnees_xmp)
        print(txt_color("[OK] ","ok"), f"Image sauvegardée : {chemin_image}")
    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f" Erreur lors de la sauvegarde de l'image : {e}")


# selection des sampler
def apply_sampler(sampler_selection):
    info = f"Le sampler a été changé en : {sampler_selection}"
    if pipe is not None:
        if sampler_selection == "EulerDiscreteScheduler (Rapide et détaillé)":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"), info)
            return info
        elif sampler_selection == "DPM++ 2M Karras (Photoréaliste et détaillé)":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"), info)
            return info
        elif sampler_selection == "Euler Ancestral (Artistique et fluide)":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "LMSDiscreteScheduler (Équilibré et polyvalent)":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "DDIMScheduler (Rapide et créatif)":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "PNDMScheduler (Stable et photoréaliste)":
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "KDPM2DiscreteScheduler (Détaillé et net)":
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "KDPM2AncestralDiscreteScheduler (Artistique et net)":
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "DPMSolverMultistepScheduler (Rapide et de haute qualité)":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "DEISMultistepScheduler (Excellent pour les détails fins)":
            pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "HeunDiscreteScheduler (Bon compromis vitesse/qualité)":
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "DPM++ SDE Karras (Photoréaliste et avec réduction du bruit)":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config) # Note: Utilisation de DPMSolverSDEScheduler
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "DPM++ 2M SDE Karras (Combine photoréalisme et réduction du bruit)":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) # 
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "Euler A (Euler Ancestral, version abrégée)":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "LMS (Linear Multistep Method, version abrégée)":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config) # Similaire à LMSDiscreteScheduler
            print(txt_color("[OK] ","ok"),info)
            return info
        elif sampler_selection == "PLMS (P-sampler - Pseudo Linear Multistep Method)":
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config) #  PLMS souvent implémenté via PNDMScheduler ou une variante
            print(txt_color("[OK] ","ok"),info)
            return info
        else:
            print(txt_color("[ERREUR] ","erreur"),f"Sampler non reconnu : {sampler_selection}") # Gestion pour les options non reconnues

    else:
        return "Merci de charger un modèle avant de changer le sampler."

#Compteur de token
def count_tokens(text):
    """Compte le nombre de tokens dans un texte."""
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    
    if token_count > 77:
        excess = token_count - 77
        return f"⚠️ Trop long ({token_count} tokens, max 77).  ❌ {excess} token en trop."
    else:
        return f"✅ Nombre de tokens valide : {token_count}"




def charger_modele(nom_fichier, nom_vae):
    """Charge un modèle spécifique."""
    
    # Importation global non recommandée, mais ici utilisée pour simplifier l'exemple
    global pipe, model_selectionne, vae_selctionne 
    
    # Si aucun modèle n'est sélectionné, affiche un message et retourne
    if nom_fichier == "Aucun modèle trouvé.":
        print(txt_color("[ERREUR] ","erreur"),"Aucun modèle sélectionné.")
        return "Aucun modèle sélectionné. Veuillez choisir un modèle dans la liste."
    
    # Construit le chemin vers le fichier de modèle en se basant sur la constante MODELS_DIR et le nom du fichier
    chemin_modele = os.path.join(MODELS_DIR, nom_fichier)
    chemin_vae = os.path.join(VAE_DIR, nom_vae)
    
    # Si une pipe est déjà chargée, on la supprime pour libérer de la mémoire GPU et éviter les conflits
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
        print(txt_color("[OK] ","ok"),"Modèle précédent déchargé.")
    
    # Essaye de charger le modèle à partir du fichier spécifié
    try:
        print(txt_color("[OK] ","ok"),f"Chargement du modèle  : {nom_fichier} et du vae : {nom_vae}")
        
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
            print (txt_color("[OK] ","ok"),f"vae {nom_vae} chargé")
        else:
            print (txt_color("[INFO] ","info"), "aucun vae sélectionné")
            
        # 
        # Si le dispositif est GPU, met le modèle dans l'espace de stockage GPU
        
        pipe = pipe.to(device) if device == "cuda" else pipe
        
        if device == "cuda" and vram_total_gb < 10:
        # Attention slicing : permet de découper le calcul de l'attention
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            pipe.enable_xformers_memory_efficient_attention()
            print(txt_color("[INFO] ","info"), "Optimisation : Attention slicing activé")
        
        # Met à jour le nom du modèle sélectionné et retourne un message de succès
        model_selectionne = nom_fichier
        vae_selctionne = nom_vae
        print(txt_color("[OK] ","ok"),f"Modèle chargé avec succès : {nom_fichier}")
        return f"Modèle '{nom_fichier}' chargé. Vae chargé : {nom_vae}"
        
    # Si une erreur se produit lors du chargement, retourne un message d'erreur
    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f"Erreur lors du chargement du modèle : {e}")
        return f"Erreur lors du chargement du modèle : {e}"


def charger_lora(nom_lora):
    #décharge le LORA avant de charger un nouveau LORA
    pipe.disable_lora()
    pipe.enable_lora() 
    lora_path = os.path.join(LORAS_DIR, nom_lora)
    
    if not os.path.exists(lora_path):
        return f"Erreur : Le fichier LORA '{nom_lora}' n'existe pas."

    if pipe is None:
        return "Erreur : Aucun modèle principal chargé."
    
    adapter_nom = os.path.splitext(nom_lora)[0]
    adapter_nom = adapter_nom.replace(".", "_")
    try:
        print(txt_color("[INFO] ","info"),f"Chargement du LORA depuis {lora_path}")
        #pipe.load_lora_weights(lora_path, weight_name=nom_lora)  # Charger le LORA
        pipe.load_lora_weights(lora_path, weight_name=nom_lora, adapter_name=adapter_nom)  # Charger le LORA
        print(txt_color("[OK] ","ok"),f"Lora {adapter_nom} chargé")
        return f"Lora {adapter_nom} chargé"
    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"),f"Erreur lors du chargement du LORA : {e}")
        return f"Erreur lors du chargement du LORA : non compatible"


#générer prompt à partir d'une image

def generate_caption(image):

    if image:
        # Préparer les entrées
        inputs = caption_processor(text="<GENERATE_TAGS>", images=image, return_tensors="pt").to(device)
    
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
        return prompt

def update_prompt(image):
    # Vérifie si l'image est présente et valide
    if image:  # Assurer que l'image est non vide
        return generate_caption(image)
      # Retourner une chaîne vide si l'image est absente ou vide


# fonction de traduction
def traduire_prompt(prompt_fr):
    """Traduit un prompt français en anglais."""
    try:
        traduction = translator(prompt_fr)[0]["translation_text"]
        print(txt_color("[INFO] ","info"), f"Traduction : {prompt_fr} -> {traduction}")
        return traduction
    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f"Erreur lors de la traduction : {e}")
        return f"Erreur lors de la traduction : {e}"

def generate_image(text, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images, lora_scale):
    """Génère des images avec Stable Diffusion."""
    
    try:
        #initialisation du chrono
        start_time = time.time()
        # Réinitialiser l'état d'arrêt
        stop_event.clear()  
        
        seeds = [random.randint(1, 10**19 - 1) for _ in range(num_images)] if seed_input == -1 else [seed_input] * num_images
        prompt_en = traduire_prompt(text) if traduire else text 
            
           
        
        width, height = map(int, selected_format.split("*"))
        images = []
        
        active_adapters = pipe.get_active_adapters()
        
        for adapter_name in active_adapters:
            pipe.set_adapters(adapter_name, lora_scale)

            

        for idx, seed in enumerate(seeds):
            depart_time = time.time()
            if stop_event.is_set():  # Vérifie si un arrêt est demandé
                print(txt_color("[INFO] ","info"), f"Arrêt demandé après {idx} images.")
                return images, seeds[:idx], f"Arrêt demandé après {idx} images."
                
            print(txt_color("[INFO] ","info"), f"Génération d'image {idx+1} avec seed {seed}")
            generator = torch.Generator(device=device).manual_seed(seed)
            generated_image = pipe(
                prompt=prompt_en,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                negative_prompt=NEGATIVE_PROMPT, # Fournir le prompt négatif textuel
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
                    "IMAGE": f"{idx+1} sur {num_images}",
                    "Creator": AUTHOR,
                    "Seed": seed,
                    "Inference": num_steps,
                    "Guidance": guidance_scale,
                    "Prompt": prompt_en,
                    "Negatif Prompt": NEGATIVE_PROMPT,
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

            # Déléguer la tâche d'écriture dans le ThreadPoolExecutor
            executor.submit(enregistrer_image, generated_image, chemin_image, donnees_xmp)
            
            print(txt_color("[OK] ","ok"),f"Image sauvegardée :", txt_color(f"{filename}","ok"))
            images.append(generated_image)
            torch.cuda.empty_cache()
            # Utiliser yield pour envoyer l'image à la galerie Gradio au fur et à mesure
            yield images, seeds[:idx+1], f"{idx+1}/{num_images} images générées..."  # Retourner l'état de la galerie avec la nouvelle image
        
        elapsed_time = f"{(time.time() - start_time):.2f} sec"
        print(txt_color("[INFO] ","info"),f"Temps total de génération : {elapsed_time}")
        return images, seeds, elapsed_time

    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f"Erreur lors de la génération : {e}")
        return [], [], f"Erreur : {e}"

def stop_generation():
    """Déclenche l'arrêt de la génération"""
    stop_event.set()
    return "Arrêt en cours..."

def decharge_lora():
    # Récupère la liste des adaptateurs actifs
    active_adapters = pipe.get_active_adapters()
    
    # Itère sur chaque adaptateur et le désactive
    for adapter_name in active_adapters:
        pipe.delete_adapters(adapter_name)
        print(txt_color("[INFO] ","info"), f"Lora '{adapter_name}' déchargé")
    
    return "Lora(s) déchargé(s)"

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
    print(f"{txt_color('[INFO]','info')}", f"{DEFAULT_MODEL} va se charger à partir de : {MODELS_DIR}")
    charger_modele(DEFAULT_MODEL, "Défaut VAE")

else:
    print(f"{txt_color('[INFO]','info')}", f"{DEFAULT_MODEL} n'a pas été trouvé dans votre repertoire de modèles : {MODELS_DIR}")
    DEFAULT_MODEL=""


# =========================
# Interface utilisateur (Gradio)
# =========================
theme_gradio = gradio_change_theme(GRADIO_THEME)

with gr.Blocks(theme=theme_gradio) as interface:
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Prompt", info="Entrez votre texte ici")
            token_count_output = gr.Textbox(label="Compteur de tokens", interactive=False)
            traduire_checkbox = gr.Checkbox(label="Traduire en anglais", value=False, info="Si coché, le texte sera traduit en anglais avant la génération de l'image")
            use_image_checkbox = gr.Checkbox(label="Générer un prompt à partir d'une image", value=False)
            image_input = gr.Image(label="Téléchargez une image", type="pil", visible=False)
            use_image_checkbox.change(fn=lambda use_image: gr.update(visible=use_image), inputs=use_image_checkbox, outputs=image_input)
            
            guidance_slider = gr.Slider(1, 20, value=7, label="Guidage")
            num_steps_slider = gr.Slider(1, 50, value=30, label="Étapes", step=1)
            format_dropdown = gr.Dropdown(choices=FORMATS, value="768*1280", label="Format")
            seed_input = gr.Number(label="Seed", value=-1)
            num_images_slider = gr.Slider(1, 200, value=1, label="Nombre d'images à générer", step=1)
            
            with gr.Row():
                btn_stop = gr.Button("Arrêter")
                btn_generate = gr.Button("Générer")

        with gr.Column():
            image_output = gr.Gallery(label="Images générées")
            seed_output = gr.Textbox(label="Seed utilisé")
            time_output = gr.Textbox(label="Temps de rendu", interactive=False)
            
            bouton_lister = gr.Button("Lister les modèles")
            value = DEFAULT_MODEL if DEFAULT_MODEL else None
            modele_dropdown = gr.Dropdown(label="Sélectionner un modèle", choices=modeles_disponibles, value=value)
            vae_dropdown = gr.Dropdown(label="Sélectionner un VAE", choices=["Défaut VAE"], value="Défaut VAE")
            sampler_dropdown = gr.Dropdown(label="Sélectionner un sampler", choices=sampler_options)
            
            bouton_charger = gr.Button("Charger le modèle")
            message_chargement = gr.Textbox(label="Statut", value="Aucun modèle chargé.")

            use_lora_checkbox = gr.Checkbox(label="Utiliser un LORA", value=False)

            with gr.Column(visible=False) as lora_section:
                lora_dropdown = gr.Dropdown(choices=["Aucun LORA disponible"], label="Sélectionner un LORA")
                lora_scale_slider = gr.Slider(0, 1, value=0, label="Poids du LORA")
                load_lora_button = gr.Button("Charger le LORA")
                unload_lora_button = gr.Button("Décharger le Lora")
                lora_message = gr.Textbox(label="Message LORA", value="")

            def mettre_a_jour_listes():
                modeles = lister_fichiers(MODELS_DIR)
                vaes = ["Défaut VAE"] + lister_fichiers(VAE_DIR)
                loras = lister_fichiers(LORAS_DIR)

                has_loras = bool(loras) and "Aucun modèle trouvé." not in loras
                lora_choices = loras if has_loras else ["Aucun LORA disponible"]

                return (
                    gr.update(choices=modeles),
                    gr.update(choices=vaes),
                    gr.update(choices=lora_choices, interactive=has_loras),
                    gr.update(interactive=has_loras, value=False),
                    gr.update(value="LORAs trouvés : " + ", ".join(loras) if has_loras else "Aucun LORA disponible")
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

    text_input.input(count_tokens, text_input, token_count_output)
    image_input.change(fn=generate_caption, inputs=image_input, outputs=text_input)

    btn_generate.click(
        generate_image, 
        inputs=[text_input, guidance_slider, num_steps_slider, format_dropdown, traduire_checkbox, seed_input, num_images_slider, lora_scale_slider], 
        outputs=[image_output, seed_output, time_output]
    )

    btn_stop.click(stop_generation, outputs=time_output)

    edit_section_checkbox = gr.Checkbox(label="Activer la retouche d'image", value=False)

    with gr.Row(visible=False) as edit_section:
        image_input = gr.Image(label="Sélectionner une image", type="numpy")
        image_output = gr.Image(label="Aperçu des modifications", type="numpy")

    with gr.Row(visible=False) as edit_controls:
        with gr.Column():
            contrast = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Contraste")
            saturation = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Saturation")
            color_boost = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Intensité des couleurs")
            blur_radius = gr.Slider(0, 10, 0, step=1, label="Rayon de flou")
            sharpness_factor = gr.Slider(0, 5, 1, step=0.1, label="Facteur de netteté")

        with gr.Column():
            grayscale = gr.Checkbox(label="Noir et blanc")
            rotation_angle = gr.Slider(0, 360, 0, step=90, label="Angle de rotation (90°)")
            mirror_type = gr.Dropdown(choices=["aucun", "horizontal", "vertical"], value="aucun", label="Type de miroir")

    edit_section_checkbox.change(
        lambda visible: [gr.update(visible=visible)] * 2, 
        inputs=edit_section_checkbox, 
        outputs=[edit_section, edit_controls]
    )

    inputs = [image_input, contrast, saturation, color_boost, grayscale, blur_radius, sharpness_factor, rotation_angle, mirror_type]
    for inp in inputs:
        inp.change(apply_filters, inputs=inputs, outputs=image_output)

interface.launch(inbrowser=str_to_bool(OPEN_BROWSER), pwa=True, share=str_to_bool(SHARE))

