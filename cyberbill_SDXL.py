import gradio as gr
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
import torch
import random
import os
import time
import threading
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from utils import fichier_recap, enregistrer_etiquettes_image_html,charger_configuration, gradio_change_theme
from version import version
from concurrent.futures import ThreadPoolExecutor
import json


# =========================
# Initialisation des variables
# =========================
print (f"cyberbill_SDXL version {version()}")
config = charger_configuration()
# Dossiers contenant les modèles
MODELS_DIR = config["MODELS_DIR"]
VAE_DIR = config["VAE_DIR"]
SAVE_DIR = config["SAVE_DIR"]
IMAGE_FORMAT = config["IMAGE_FORMAT"].upper() 
FORMATS = config["FORMATS"]
NEGATIVE_PROMPT = config["NEGATIVE_PROMPT"]
GRADIO_THEME = config["GRADIO_THEME"]

# Vérifier que le format est valide
if IMAGE_FORMAT not in ["PNG", "JPG", "WEBP"]:
    print(f"⚠️ Format {IMAGE_FORMAT} invalide, utilisation de WEBP par défaut.")
    IMAGE_FORMAT = "WEBP"

# Créer un pool de threads pour l'écriture asynchrone
executor = ThreadPoolExecutor(max_workers=4)

# Vérifier si CUDA est disponible
if torch.cuda.is_available():
    gpu_id = 0  # ID du GPU (ajuste si nécessaire)
    vram_total = torch.cuda.get_device_properties(gpu_id).total_memory  # en octets
    vram_total_gb = vram_total / (1024 ** 3)  # conversion en Go

    print(f"VRAM détectée : {vram_total_gb:.2f} Go")

    # Activer expandable_segments si VRAM < 10 Go
    if vram_total_gb < 10:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True activé")

    # Détermination du device et du type de données
    device = "cuda"
    torch_dtype = torch.float16
else:
    print("CUDA non disponible, exécution sur CPU")
    device = "cpu"
    torch_dtype = torch.float32

print(f"Utilisation de : {device} avec dtype {torch_dtype}")

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
print(f"Modèle de traduction '{translation_model}' chargé")


# Charger le modèle et le processeur
caption_model = AutoModelForCausalLM.from_pretrained(
    "MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True
).to(device)
caption_processor = AutoProcessor.from_pretrained(
    "MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True
)

# Sampler disponibles
sampler_options = [
    "EulerDiscreteScheduler (Rapide et détaillé)",
    "DPM++ 2M Karras (Photoréaliste et détaillé)",
    "Euler Ancestral (Artistique et fluide)"
    ]

# =========================
# Définition des fonctions
# =========================

def enregistrer_image(image, chemin_image, donnees_xmp):
    """Enregistre l'image et écrit les métadonnées."""
    try:
        image.save(chemin_image, format=IMAGE_FORMAT)
        enregistrer_etiquettes_image_html(chemin_image, donnees_xmp)
        print(f"Image sauvegardée : {chemin_image}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image : {e}")


# selection des sampler
def apply_sampler(sampler_selection):
    if pipe is not None:
        if sampler_selection == "EulerDiscreteScheduler (Rapide et détaillé)":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            print(f"Le sampler a été changé en {sampler_selection}")
        elif sampler_selection == "DPM++ 2M Karras (Photoréaliste et détaillé)":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            print(f"Le sampler a été changé en {sampler_selection}")
        elif sampler_selection == "Euler Ancestral (Artistique et fluide)":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            print(f"Le sampler a été changé en {sampler_selection}")
    else:
        return "Merci de charger un modèle avant de changer le sampler."
    
    # Retourner un message confirmant le changement
    return f"Le sampler a été changé en : {sampler_selection}"

#Compteur de token
def count_tokens(text):
    """Compte le nombre de tokens dans un texte."""
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    
    if token_count > 77:
        excess = token_count - 77
        return f"⚠️ Trop long ({token_count} tokens, max 77).  ❌ {excess} token en trop."
    else:
        return f"✅ Nombre de tokens valide : {token_count}"

# liste fichiers .safetensors
def lister_fichiers(dir, ext=".safetensors"):
    """List files in a directory with a specific extension."""
    
    # Try to get the list of files from the specified directory. 
    try:
        fichiers = [f for f in os.listdir(dir) if f.endswith(ext)]
        
        # If no files are found, print a specific message and return an empty list.
        if not fichiers:
            print("No models found.")
            return ["Aucun modèle trouvé."]
            
    except FileNotFoundError:
        # If the directory doesn't exist, print a specific error message and return an empty list. 
        print(f"Directory not found : {dir}")
        return ["Répertoire non trouvé."]
        
    else:
        # If files are found, print them out and return the file_list. 
        print(f"Files found in {dir}: {fichiers}")
        return fichiers


def charger_modele(nom_fichier, nom_vae):
    """Charge un modèle spécifique."""
    
    # Importation global non recommandée, mais ici utilisée pour simplifier l'exemple
    global pipe, model_selectionne, vae_selctionne
    
    # Construit le chemin vers le fichier de modèle en se basant sur la constante MODELS_DIR et le nom du fichier
    chemin_modele = os.path.join(MODELS_DIR, nom_fichier)
    chemin_vae = os.path.join(VAE_DIR, nom_vae)
    
    # Si une pipe est déjà chargée, on la supprime pour libérer de la mémoire GPU et éviter les conflits
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
        print("Modèle précédent déchargé.")
    
    # Essaye de charger le modèle à partir du fichier spécifié
    try:
        print(f"Chargement du modèle  : {nom_fichier} et du vae : {nom_vae}")
        
        # Charge le modèle et met à jour la variable globale 'pipe'
        pipe = StableDiffusionXLPipeline.from_single_file(
            chemin_modele, 
            safety_checker=None, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True if (device == "cuda" and vram_total_gb < 10) else False,
            load_device=device
         )
         
        if not nom_vae == "Défaut VAE":
            pipe.vae = vae=AutoencoderKL.from_single_file(chemin_vae, torch_dtype=torch_dtype)
            print (f"vae {nom_vae} chargé")
        else:
            print ("aucun vae sélectionné")
            
        # 
        # Si le dispositif est GPU, met le modèle dans l'espace de stockage GPU
        
        pipe = pipe.to(device) if device == "cuda" else pipe
        
        if device == "cuda" and vram_total_gb < 10:
        # Attention slicing : permet de découper le calcul de l'attention
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            print("Optimisation : Attention slicing activé")
        
        # Met à jour le nom du modèle sélectionné et retourne un message de succès
        model_selectionne = nom_fichier
        vae_selctionne = nom_vae
        print(f"Modèle chargé avec succès : {nom_fichier}")
        return f"Modèle '{nom_fichier}' chargé. Vae chargé : {nom_vae}"
        
    # Si une erreur se produit lors du chargement, retourne un message d'erreur
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return f"Erreur lors du chargement du modèle : {e}"

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
        print(f"Traduction : {prompt_fr} -> {traduction}")
        return traduction
    except Exception as e:
        print(f"Erreur lors de la traduction : {e}")
        return f"Erreur lors de la traduction : {e}"

def generate_image(text, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images):
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
        

        for idx, seed in enumerate(seeds):
            if stop_event.is_set():  # Vérifie si un arrêt est demandé
                print(f"Arrêt demandé après {idx} images.")
                return images, seeds[:idx], f"Arrêt demandé après {idx} images."
                
            print(f"Génération d'image {idx+1} avec seed {seed}")
            generator = torch.Generator(device=device).manual_seed(seed)
            generated_image = pipe(
                prompt_en,
                num_inference_steps=num_steps,
                # guidance_scale=guidance_scale,
                negative_prompt= NEGATIVE_PROMPT,
                generator=generator,
                width=width,
                height=height
            ).images[0]
            
            date_str = datetime.now().strftime("%Y_%m_%d")
            heure_str = datetime.now().strftime("%H_%M_%S")
            save_dir = os.path.join(SAVE_DIR, date_str)
            os.makedirs(save_dir, exist_ok=True)
            
            donnees_xmp =  {
                    "IMAGE": f"{idx+1} sur {num_images}",
                    "Creator": "William GODEFROID",
                    "Seed": seed,
                    "Inference": num_steps,
                    "Guidance": guidance_scale,
                    "Prompt": prompt_en,
                    "Dimension": selected_format,
                    "Model": model_selectionne,
                    "VAE": vae_selctionne,
                    "Sampler": pipe.scheduler.__class__.__name__
                }

            filename = f"{date_str}_{heure_str}_{seed}_{width}x{height}_{idx+1}.{IMAGE_FORMAT.lower()}"
            chemin_image = os.path.join(save_dir, filename)

            # Déléguer la tâche d'écriture dans le ThreadPoolExecutor
            executor.submit(enregistrer_image, generated_image, chemin_image, donnees_xmp)
            
            print(f"Image sauvegardée : {filename}")
            images.append(generated_image)
            torch.cuda.empty_cache()
            # Utiliser yield pour envoyer l'image à la galerie Gradio au fur et à mesure
            yield images, seeds[:idx+1], f"{idx+1}/{num_images} images générées..."  # Retourner l'état de la galerie avec la nouvelle image
        
        elapsed_time = f"{(time.time() - start_time):.2f} sec"
        print(f"Temps total de génération : {elapsed_time}")
        return images, seeds, elapsed_time

    except Exception as e:
        print(f"Erreur lors de la génération : {e}")
        return [], [], f"Erreur : {e}"

def stop_generation():
    """Déclenche l'arrêt de la génération"""
    stop_event.set()
    return "Arrêt en cours..."


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
# Interface utilisateur (Gradio)
# =========================
theme_gradio = gradio_change_theme(GRADIO_THEME)

with gr.Blocks(theme=theme_gradio) as interface:
     
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Prompt", info="Entrez votre texte ici")
            token_count_output = gr.Textbox(label="Compteur de tokens", interactive=False)
            traduire_checkbox = gr.Checkbox(label="Traduire en anglais", value=False, info="Si coché, le texte sera traduit en anglais avant la génération de l'image")
            use_image_checkbox = gr.Checkbox(label="Générer un prompt à partir d'une image", value=False, info="Si coché, vous pourrez télécharger une image et générer un prompt à partir de celle-ci.")
            image_input = gr.Image(label="Téléchargez une image", type="pil", visible=False)
            use_image_checkbox.change(fn=lambda use_image: gr.update(visible=use_image), inputs=use_image_checkbox, outputs=image_input)
            
            guidance_slider = gr.Slider(1, 20, value=7, label="Guidage", info="Amplifier ou réduire la qualité de l'image")
            num_steps_slider = gr.Slider(1, 50, value=30, label="Étapes", step=1, info="Nombre d'étapes de génération de l'image")
            format_dropdown = gr.Dropdown(choices=FORMATS, value="768*1280", label="Format", info="Choisissez le format de sortie souhaité pour votre image")
            seed_input = gr.Number(label="Seed", value=-1, info="Saisir une valeur pour générer des images identiques à chaque fois ou laisser -1 pour des résultats aléatoires")
            num_images_input = gr.Number(label="Nombre d'images", value=1)
            with gr.Row():
                btn_stop = gr.Button("Arrêter")
                btn_generate = gr.Button("Générer")
  

        with gr.Column():
            image_output = gr.Gallery(label="Images générées")
            seed_output = gr.Textbox(label="Seed utilisé")
            time_output = gr.Textbox(label="Temps de rendu", interactive=False)
            
            bouton_lister = gr.Button("Lister les modèles")
            # Dropdown pour sélectionner le modèle
            modele_dropdown = gr.Dropdown(label="Sélectionner un modèle", choices=[])
            # Nouveau dropdown pour sélectionner le VAE, valeur par défaut "Défaut VAE"
            vae_dropdown = gr.Dropdown(
                label="Sélectionner un VAE",
                choices=["Défaut VAE"],
                value="Défaut VAE",
                info="Choisissez un VAE externe ou 'Défaut VAE' pour utiliser le VAE par défaut"
            )
    
            sampler_dropdown = gr.Dropdown(
                label="Sélectionner un sampler",
                choices=sampler_options,
                info="Choisissez un sampler pour la génération d'images"
            )
            
            
            bouton_charger = gr.Button("Charger le modèle")
            message_chargement = gr.Textbox(label="Statut", value="Aucun modèle chargé.")
    

    
            # Mise à jour des dropdowns pour modèles et VAE lors du clic sur le bouton lister
            bouton_lister.click(
                fn=lambda: (
                    gr.update(choices=lister_fichiers(MODELS_DIR)),
                    gr.update(choices=["Défaut VAE"] + lister_fichiers(VAE_DIR))
                ),
                outputs=[modele_dropdown, vae_dropdown]
            )
    
            # Le bouton charger envoie désormais le modèle ET le VAE sélectionné
            bouton_charger.click(
                fn=charger_modele,
                inputs=[modele_dropdown, vae_dropdown],
                outputs=message_chargement
            )
    
            sampler_dropdown.change(
                fn=apply_sampler,
                inputs=sampler_dropdown,
                outputs=message_chargement
            )
            


    text_input.input(count_tokens, text_input, token_count_output)
    image_input.change(fn=generate_caption, inputs=image_input, outputs=text_input)

    btn_generate.click(
        generate_image, 
        inputs=[text_input, guidance_slider, num_steps_slider, format_dropdown, traduire_checkbox, seed_input, num_images_input], 
        outputs=[image_output, seed_output, time_output]
    )

    btn_stop.click(stop_generation, outputs=time_output)
    
    # Ajout d'une case à cocher pour afficher/masquer la section de retouche
    edit_section_checkbox = gr.Checkbox(label="Activer la retouche d'image", value=False)
    
    with gr.Row(visible=False) as edit_section:
        image_input = gr.Image(label="Sélectionner une image", type="numpy")
        image_output = gr.Image(label="Aperçu des modifications", type="numpy")
    
    with gr.Row(visible=False) as edit_controls:
        with gr.Column():  # Colonne pour les retouches de couleur
            contrast = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Contraste")
            saturation = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Saturation")
            color_boost = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Intensité des couleurs")
            blur_radius = gr.Slider(0, 10, 0, step=1, label="Rayon de flou")
            sharpness_factor = gr.Slider(0, 5, 1, step=0.1, label="Facteur de netteté")
            
    
        with gr.Column():  # Colonne pour les transformations géométriques
            grayscale = gr.Checkbox(label="Noir et blanc")
            rotation_angle = gr.Slider(0, 360, 0, step=90, label="Angle de rotation (90°)")
            mirror_type = gr.Dropdown(choices=["aucun", "horizontal", "vertical"], value="aucun", label="Type de miroir")

    
    edit_section_checkbox.change(lambda visible: [gr.update(visible=visible)] * 2, inputs=edit_section_checkbox, outputs=[edit_section, edit_controls])
    
    
    inputs = [image_input,  contrast, saturation, color_boost, grayscale, blur_radius, sharpness_factor, rotation_angle, mirror_type]
    for inp in inputs:
            inp.change(apply_filters, inputs=inputs, outputs=image_output)


    
interface.launch(inbrowser=True, pwa=True, share=False)
