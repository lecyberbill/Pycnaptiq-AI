import torch
import time
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline, AutoencoderKL, AutoPipelineForText2Image
import os
from pathlib import Path
import gradio as gr
import requests
from Utils.utils import txt_color, translate, decharger_modele
from compel import Compel, ReturnedEmbeddingsType
import gc
import warnings
#from RealESRGAN import RealESRGAN




def charger_modele(nom_fichier, nom_vae, translations, MODELS_DIR, VAE_DIR, device, torch_dtype, vram_total_gb, pipe=None, compel=None, gradio_mode=False):
    """Charge un modèle spécifique."""
    
    # Get the project's root directory
    root_dir = Path(__file__).parent.parent

    if not nom_fichier:
        print(txt_color("[ERREUR] ", "erreur"), translate("aucun_modele_selectionne", translations))
        if gradio_mode:
            gr.Warning(translate("aucun_modele_selectionne", translations), 4.0)
        return None, None, translate("aucun_modele_selectionne", translations)
    
    # Determine if MODELS_DIR is absolute or relative
    if os.path.isabs(MODELS_DIR):
        # If it's absolute, use it directly
        chemin_modele = os.path.join(MODELS_DIR, nom_fichier)
    else:
        # If it's relative, join it with the root directory
        chemin_modele = os.path.join(root_dir, MODELS_DIR, nom_fichier)

    # Determine if VAE_DIR is absolute or relative
    if os.path.isabs(VAE_DIR):
        # If it's absolute, use it directly
        chemin_vae = os.path.join(VAE_DIR, nom_vae) if nom_vae and nom_vae != "Défaut VAE" else None
    else:
        # If it's relative, join it with the root directory
        chemin_vae = os.path.join(root_dir, VAE_DIR, nom_vae) if nom_vae and nom_vae != "Défaut VAE" else None


    pipe, compel = decharger_modele(pipe, compel, translations)
    
    try:
        print(txt_color("[INFO] ", "info"), f"{translate('chargement_modele', translations)} : {nom_fichier}")
        if gradio_mode:
            gr.Info(translate('chargement_modele', translations) + f" : {nom_fichier}", 3.0)
            


        if not os.path.exists(chemin_modele):
            print(txt_color("[ERREUR] ", "erreur"), f"{translate('modele_non_trouve', translations)}: {chemin_modele}")
            if gradio_mode:
                gr.Warning(translate("modele_non_trouve", translations) + f": {chemin_modele}", 4.0)
            return None, None, translate("modele_non_trouve", translations), f": {chemin_modele}"
        
        pipe = StableDiffusionXLPipeline.from_single_file(
            chemin_modele,
            use_safetensors=True,
            safety_checker=None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=(device == "cuda" and vram_total_gb < 10),
            load_device=device
        )
        
        if chemin_vae:
            pipe.vae = AutoencoderKL.from_single_file(chemin_vae, torch_dtype=torch_dtype)
            print(txt_color("[OK] ", "ok"), translate("vae_charge", translations), f": {nom_vae}")
            if gradio_mode:
                gr.Info(translate("vae_charge", translations) + f": {nom_vae}", 3.0)
        
        pipe.to(device)
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        
        if device == "cuda" and vram_total_gb < 10:
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
            print(txt_color("[INFO] ", "info"), translate("optimisation_attention", translations))
            if gradio_mode:
                gr.Info(translate("optimisation_attention", translations), 3.0) 
        
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        
        print(txt_color("[OK] ", "ok"), translate("modele_charge", translations), f": {nom_fichier}")
        if gradio_mode:
            gr.Info(translate("modele_charge", translations) + f": {nom_fichier}", 3.0) 
        return pipe, compel, translate("modele_charge", translations), f": {nom_fichier}"
    
    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_chargement_modele", translations), f": {e}")
        if gradio_mode:
            raise  gr.Error(translate("erreur_chargement_modele", translations) + " " + f": {e}", 4.0 )
        return None, None, translate("erreur_chargement_modele", translations), f": {e}"


def charger_modele_inpainting(nom_fichier, translations, INPAINT_MODELS_DIR, device, torch_dtype, vram_total_gb, pipe=None,compel=None): 
    """Charge un modèle spécifique pour l'inpainting."""
    
    root_dir = Path(__file__).parent.parent
    
    # Si aucun modèle n'est sélectionné, affiche un message et retourne
    if nom_fichier is None or nom_fichier == translate("aucun_modele_trouve", translations) or not nom_fichier:
        print(txt_color("[ERREUR] ", "erreur"), translate("aucun_modele_selectionne", translations))
        gr.Warning(translate("aucun_modele_selectionne", translations), 4.0)
        return None, translate("aucun_modele_selectionne", translations) + translate("verifier_config", translations)


        
    # Determine if INPAINT_MODELS_DIR is absolute or relative
    if os.path.isabs(INPAINT_MODELS_DIR):
        # If it's absolute, use it directly
        chemin_modele = os.path.join(INPAINT_MODELS_DIR, nom_fichier)
    else:
        # If it's relative, join it with the root directory
        chemin_modele = os.path.join(root_dir, INPAINT_MODELS_DIR, nom_fichier)

    # Si une pipe est déjà chargée, on la supprime pour libérer de la mémoire GPU et éviter les conflits
    pipe, compel = decharger_modele(pipe, compel, translations)

    # Essaye de charger le modèle à partir du fichier spécifié
    try:
        print(txt_color("[INFO] ", "info"), f"{translate('chargement_modele', translations)} : {nom_fichier}")
        gr.Info(f"{translate('chargement_modele', translations)} : {nom_fichier}", 3.0)
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            chemin_modele,
            use_safetensors=True,
            safety_checker=None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True if (device == "cuda" and vram_total_gb < 10) else False,
            load_device=device,
        )
        pipe = pipe.to(device) if device == "cuda" else pipe

        if device == "cuda" and vram_total_gb < 10:
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()
            print(txt_color("[INFO] ", "info"), translate("optimisation_attention", translations))
            gr.Info(translate("optimisation_attention", translations), 3.0)
        
        # initialisation de compel
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        model_selectionne = nom_fichier
        print(txt_color("[OK] ", "ok"), translate("modele_charge", translations), f": {nom_fichier}")
        gr.Info(translate("modele_charge", translations) + " " + f": {nom_fichier}", 3.0)
        return pipe, compel, translate("modele_charge", translations)

    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_chargement_modele", translations), f": {e}")
        raise gr.Error(translate("erreur_chargement_modele", translations) + " " + f": {e}", 4.0)
        return None, None, translate("erreur_chargement_modele", translations), f": {e}"

def charger_lora(nom_lora, pipe, LORAS_DIR, translations):
    """Charge un LORA spécifique."""
    if pipe is None:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_modele", translations))
        gr.Warning(translate("erreur_pas_modele", translations), 4.0)
        return translate("erreur_pas_modele", translations)

    lora_path = os.path.join(LORAS_DIR, nom_lora)

    if not os.path.exists(lora_path):
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_fichier_lora", translations))
        gr.Warning(translate("erreur_fichier_lora", translations), 4.0)
        return translate("erreur_fichier_lora", translations)

    adapter_nom = os.path.splitext(nom_lora)[0]
    adapter_nom = adapter_nom.replace(".", "_")
    try:
        print(txt_color("[INFO] ", "info"), translate("lora_charge_depuis", translations), f"{lora_path}")
        gr.Info(translate("lora_charge_depuis", translations) + " " + f"{lora_path}", 3.0)
        pipe.load_lora_weights(lora_path, weight_name=nom_lora, adapter_name=adapter_nom)  # Charger le LORA
        print(txt_color("[OK] ", "ok"), translate("lora_charge", translations), f"{adapter_nom}")
        gr.Info(translate("lora_charge", translations) + " " + f"{adapter_nom}", 3.0)
        return translate("lora_charge", translations), f" {adapter_nom}"
    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_lora_chargement", translations), f": {e}")
        raise gr.Error(translate("erreur_lora_chargement", translations) + " " + f": {e}", 4.0)
        return translate("lora_non_compatible", translations)

def decharge_lora(pipe, translations):
    """Décharge tous les LORAs."""
    if pipe is None:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_modele", translations))
        gr.Warning(translate("erreur_pas_modele", translations), 4.0)
        return translate("erreur_pas_modele", translations)
    # Récupère la liste des adaptateurs actifs
    active_adapters = pipe.get_active_adapters()

    # Itère sur chaque adaptateur et le désactive
    for adapter_name in active_adapters:
        pipe.delete_adapters(adapter_name)
        print(txt_color("[INFO] ", "info"), f"{translate('lora_decharge_nom', translations)} '{adapter_name}'")
        gr.Info(translate("lora_decharge_nom", translations) + f" '{adapter_name}'", 3.0)

    return translate("lora_decharge", translations)
