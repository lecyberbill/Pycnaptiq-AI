import os
import traceback
import gradio as gr
from Utils.utils import txt_color, translate, lister_fichiers, telechargement_modele

def check_and_download_default_models(model_manager, translations, MODELS_DIR, INPAINT_MODELS_DIR):
    """
    Vérifie si des modèles sont présents et propose le téléchargement par défaut si nécessaire.
    """
    modeles_disponibles = model_manager.list_models(model_type="standard")
    if modeles_disponibles and modeles_disponibles[0] == translate("aucun_modele_trouve", translations):
        reponse = input(f"{txt_color(translate('attention', translations),'erreur')} {translate('aucun_modele_trouve', translations)} {translate('telecharger_modele_question', translations)} ")
        if reponse.lower() in ["o", "oui", "y", "yes"]:
            lien_modele = "https://huggingface.co/QuadPipe/MegaChonkXL/resolve/main/MegaChonk-XL-v2.3.1.safetensors?download=true"
            nom_fichier = "MegaChonk-XL-v2.3.1.safetensors"
            if telechargement_modele(lien_modele, nom_fichier, MODELS_DIR, translations):
                modeles_disponibles = lister_fichiers(MODELS_DIR, translations)
                if not modeles_disponibles:
                    modeles_disponibles = [translate("aucun_modele_trouve", translations)]
    
    modeles_impaint = model_manager.list_models(model_type="inpainting")
    if not modeles_impaint or modeles_impaint[0] == translate("aucun_modele_trouve", translations):
        modeles_impaint = [translate("aucun_modele_trouve", translations)]
        reponse_inpaint = input(
            f"{txt_color(translate('attention', translations), 'erreur')} "
            f"{translate('aucun_modele_inpainting_trouve', translations)} "
            f"{translate('telecharger_modele_question', translations)} "
            f"(wangqyqq/sd_xl_base_1.0_inpainting_0.1.safetensors) ? (o/n): "
        )
        if reponse_inpaint.lower() in ["o", "oui", "y", "yes"]:
            lien_modele_inpaint = "https://civitai.com/api/download/models/916706?type=Model&format=SafeTensor&size=full&fp=fp16"
            nom_fichier_inpaint = "sd_xl_base_1.0_inpainting_0.1.safetensors"
            if telechargement_modele(lien_modele_inpaint, nom_fichier_inpaint, INPAINT_MODELS_DIR, translations):
                modeles_impaint = model_manager.list_models(model_type="inpainting")
                if not modeles_impaint or modeles_impaint[0] == translate("aucun_modele_trouve", translations):
                    modeles_impaint = [translate("aucun_modele_trouve", translations)]
    
    return modeles_disponibles, modeles_impaint

def handle_model_selection(
    nom_fichier, 
    nom_vae, 
    pag_is_enabled, 
    model_manager, 
    translations
):
    """
    Handles loading a standard model and updates UI state.
    Returns: (status_message, interactif_update, button_text_update, new_model_name, new_vae_name)
    """
    try:
        custom_pipeline_to_use = None
        if pag_is_enabled:
            custom_pipeline_to_use = "hyoungwoncho/sd_perturbed_attention_guidage"
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
        gr.Error(f"Erreur interne lors du chargement : {e}")
        model_manager.unload_model()

    if success:
        new_model_name = nom_fichier
        new_vae_name = nom_vae
        etat_interactif = True
        texte_bouton = translate("generer", translations) 
    else:
        new_model_name = None
        new_vae_name = "Défaut VAE"
        etat_interactif = False
        texte_bouton = translate("charger_modele_pour_commencer", translations) 

    update_interactif = gr.update(interactive=etat_interactif)
    update_texte = gr.update(value=texte_bouton)
    
    return message, update_interactif, update_texte, new_model_name, new_vae_name

def handle_inpainting_model_selection(
    nom_fichier, 
    model_manager, 
    translations
):
    """
    Handles loading an inpainting model and updates UI state.
    Returns: (status_message, interactif_update, button_text_update, new_model_name)
    """
    # Initialisation des valeurs par défaut en cas d'échec
    etat_interactif = False
    texte_bouton = translate("charger_modele_pour_commencer", translations) 
    message_final = ""
    new_model_name = None

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
        gr.Error(f"Erreur interne lors du chargement (inpainting) : {e}")
        model_manager.unload_model()

    if success:
        new_model_name = nom_fichier
        etat_interactif = True
        texte_bouton = translate("generer_inpainting", translations) 
    else:
        new_model_name = None

    update_interactif_gen_btn = gr.update(interactive=etat_interactif)
    update_texte_gen_btn = gr.update(value=texte_bouton)

    return message_final, update_interactif_gen_btn, update_texte_gen_btn, new_model_name

def get_model_list_updates(models_dir, model_manager, translations, num_loras=4):
    """
    Returns updates for model, VAE, and LoRA dropdowns.
    """
    modeles = lister_fichiers(models_dir, translations, gradio_mode=True)
    vaes = model_manager.list_vaes()
    loras = model_manager.list_loras(gradio_mode=True)

    has_loras = bool(loras) and loras[0] != translate("aucun_modele_trouve", translations) and loras[0] != translate("repertoire_not_found", translations)
    lora_choices = loras if has_loras else ["Aucun LORA disponible"]
    lora_updates = [gr.update(choices=lora_choices, interactive=has_loras, value=None) for _ in range(num_loras)]
    
    lora_msg = translate("lora_trouve", translations) + ", ".join(loras) if has_loras else translate("aucun_lora_disponible", translations)
    
    return (
        gr.update(choices=modeles),
        gr.update(choices=vaes),
        *lora_updates,
        gr.update(value=lora_msg)
    )

def get_inpainting_model_list_updates(inpaint_models_dir, translations):
    """
    Returns updates for inpainting model dropdown.
    """
    modeles = lister_fichiers(inpaint_models_dir, translations)
    return gr.update(choices=modeles)
