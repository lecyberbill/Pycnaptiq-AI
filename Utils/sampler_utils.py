# Utils/sampler_utils.py
from diffusers import (
    EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, HeunDiscreteScheduler,
    DPMSolverSDEScheduler, DPMSolverSinglestepScheduler # UniPC n'a pas sa propre classe, souvent PNDM ou autre est utilisé
)
# Importer PNDMScheduler pour UniPC si c'est le cas, ou ajuster selon la config réelle
# from diffusers import UniPCMultistepScheduler # Si UniPC a sa propre classe maintenant

# --- AJOUT: Importer translate et txt_color depuis Utils.utils ---
# Assurez-vous que le chemin d'importation est correct par rapport à votre structure
try:
    from .utils import translate, txt_color
except ImportError:
    # Fallback si l'import relatif échoue (par exemple, si exécuté comme script principal)
    from utils import translate, txt_color


# --- Définition Centralisée des Samplers ---
# Structure: clé_interne: { "class": ClasseScheduler, "translation_key": "clé_pour_translate" }
SAMPLER_DEFINITIONS = {
    "sampler_euler": {"class": EulerDiscreteScheduler, "translation_key": "sampler_euler"},
    "sampler_dpmpp_2m": {"class": DPMSolverMultistepScheduler, "translation_key": "sampler_dpmpp_2m"},
    "sampler_dpmpp_2s_a": {"class": EulerAncestralDiscreteScheduler, "translation_key": "sampler_dpmpp_2s_a"},
    "sampler_lms": {"class": LMSDiscreteScheduler, "translation_key": "sampler_lms"},
    "sampler_ddim": {"class": DDIMScheduler, "translation_key": "sampler_ddim"},
    "sampler_pndm": {"class": PNDMScheduler, "translation_key": "sampler_pndm"},
    "sampler_dpm2": {"class": KDPM2DiscreteScheduler, "translation_key": "sampler_dpm2"},
    "sampler_dpm2_a": {"class": KDPM2AncestralDiscreteScheduler, "translation_key": "sampler_dpm2_a"},
    "sampler_dpm_fast": {"class": DPMSolverMultistepScheduler, "translation_key": "sampler_dpm_fast"}, # Souvent DPM++ 2M
    "sampler_dpm_adaptive": {"class": DEISMultistepScheduler, "translation_key": "sampler_dpm_adaptive"}, # Souvent DEIS
    "sampler_heun": {"class": HeunDiscreteScheduler, "translation_key": "sampler_heun"},
    "sampler_dpmpp_sde": {"class": DPMSolverSDEScheduler, "translation_key": "sampler_dpmpp_sde"},
    "sampler_dpmpp_3m_sde": {"class": DPMSolverSinglestepScheduler, "translation_key": "sampler_dpmpp_3m_sde"}, # Souvent DPM Solver Single
    "sampler_euler_a": {"class": EulerAncestralDiscreteScheduler, "translation_key": "sampler_euler_a"},
    "sampler_unipc": {"class": PNDMScheduler, "translation_key": "sampler_unipc"}, # Souvent PNDM, à vérifier
    # Ajoutez d'autres samplers ici si nécessaire
}

# --- Fonctions Utilitaires ---

def get_sampler_choices(translations):
    """Retourne la liste des noms de samplers traduits pour l'UI."""
    choices = []
    for definition in SAMPLER_DEFINITIONS.values():
        # --- CORRECTION: Utilisation correcte de translate ---
        choices.append(translate(definition["translation_key"], translations))
    # Optionnel: Trier les choix par ordre alphabétique si désiré
    # choices.sort()
    return choices

def get_sampler_key_from_display_name(display_name, translations):
    """Trouve la clé interne du sampler à partir de son nom affiché."""
    for key, definition in SAMPLER_DEFINITIONS.items():
        if translate(definition["translation_key"], translations) == display_name:
            return key
    return None # Retourne None si non trouvé

def apply_sampler_to_pipe(pipe, sampler_key, translations):
    """Applique le scheduler correspondant à la clé interne sur le pipeline."""
    if pipe is None:
        # --- CORRECTION: Utilisation correcte de translate ---
        return translate("erreur_pas_modele", translations), False # Message d'erreur, Échec

    if sampler_key in SAMPLER_DEFINITIONS:
        scheduler_class = SAMPLER_DEFINITIONS[sampler_key]["class"]
        try:
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
            # --- CORRECTION: Utilisation correcte de translate ---
            display_name = translate(SAMPLER_DEFINITIONS[sampler_key]["translation_key"], translations)
            # --- CORRECTION: Utilisation correcte de translate ---
            success_msg = f"{translate('sampler_change', translations)}{display_name}"
            # --- AJOUT: Utilisation de txt_color (maintenant importé) ---
            print(txt_color("[OK] ", "ok"), success_msg)
            # gr.Info(success_msg, 3.0) # On ne peut pas faire de gr.Info depuis un utilitaire
            return success_msg, True # Message de succès, Succès
        except Exception as e:
            error_msg = f"Erreur application sampler '{sampler_key}': {e}"
            # --- AJOUT: Utilisation de txt_color (maintenant importé) ---
            print(txt_color("[ERREUR] ", "erreur"), error_msg)
            return error_msg, False # Message d'erreur, Échec
    else:
        # --- CORRECTION: Utilisation correcte de translate ---
        error_msg = f"{translate('erreur_sampler_inconnu', translations)}: {sampler_key}"
        # --- AJOUT: Utilisation de txt_color (maintenant importé) ---
        print(txt_color("[ERREUR] ", "erreur"), error_msg)
        return error_msg, False # Message d'erreur, Échec

