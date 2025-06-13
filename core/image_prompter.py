# core/image_prompter.py
import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from Utils.utils import txt_color, translate
import traceback

# --- Configuration ---
# Le modèle par défaut est celui que vous utilisiez. Peut être surchargé par la configuration globale.
MODEL_ID_FLORENCE2 = os.environ.get("FLORENCE2_MODEL_ID", "MiaoshouAI/Florence-2-base-PromptGen-v2.0")

# --- Variables globales pour le modèle (chargées une seule fois) ---
_caption_model = None
_caption_processor = None
_caption_device = None
_caption_translations_global = None # Stocke les traductions passées lors de init_image_prompter
_current_caption_model_id = None

# Liste des tâches Florence-2 disponibles
# (Conservée de votre version originale, car elle correspond à vos traductions dans ImageToText_mod.json)
FLORENCE2_TASKS = [
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<CAPTION>",
    "<GENERATE_TAGS>",
    "<ANALYZE>",
    "<MIXED_CAPTION>",
    "<MIXED_CAPTION_PLUS>"
]
DEFAULT_FLORENCE2_TASK = FLORENCE2_TASKS[0] # <DETAILED_CAPTION>

def _get_translate_func(translations_param=None):
    """Retourne une fonction translate, préférant locale si disponible, else globale."""
    active_translations = translations_param if translations_param else _caption_translations_global
    if active_translations:
        # La lambda retourne maintenant la chaîne brute de translate
        return lambda key: translate(key, active_translations)
    # Le fallback retourne aussi la clé brute
    return lambda key: key

def _load_caption_model_if_needed(model_id_to_load, device_to_use, translations_for_log):
    global _caption_model, _caption_processor, _current_caption_model_id, _caption_device
    t = _get_translate_func(translations_for_log)

    if _caption_model is not None and _caption_processor is not None and \
       _current_caption_model_id == model_id_to_load and _caption_model.device.type == device_to_use:
        return True, t("caption_model_already_loaded")

    if (_caption_model is not None or _caption_processor is not None) and _current_caption_model_id != model_id_to_load:
        print(txt_color("[INFO]", "info"), t("caption_model_unloading_previous").format(old_model=str(_current_caption_model_id), new_model=model_id_to_load))
        _caption_model = None
        _caption_processor = None
        _current_caption_model_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(txt_color("[INFO]", "info"), t("loading_caption_model").format(model_id=model_id_to_load))
    try:
        dtype = torch.float16 if device_to_use != "cpu" else torch.float32
        _caption_processor = AutoProcessor.from_pretrained(model_id_to_load, trust_remote_code=True)
        _caption_model = AutoModelForCausalLM.from_pretrained(
            model_id_to_load,
            torch_dtype=dtype,
            trust_remote_code=True
        ).eval().to(device_to_use)
        _current_caption_model_id = model_id_to_load
        _caption_device = device_to_use # Mettre à jour le device global
        msg = t("caption_model_loaded").format(model_id=model_id_to_load)
        print(txt_color("[OK]", "ok"), msg)
        return True, msg
    except Exception as e:
        msg = t("error_loading_caption_model").format(model_id=model_id_to_load, error=str(e))
        print(txt_color("[ERREUR]", "erreur"), msg)
        traceback.print_exc()
        _caption_model = None
        _caption_processor = None
        _current_caption_model_id = None
        return False, msg

def init_image_prompter(device, translations_param, model_id=MODEL_ID_FLORENCE2, load_now=True):
    """Initialise le module Image Prompter. Peut charger le modèle immédiatement ou non."""
    global _caption_device, _caption_translations_global
    _caption_device = device
    _caption_translations_global = translations_param
    t = _get_translate_func(translations_param)
    print(txt_color("[INFO]", "info"), t("initializing_image_prompter"))

    if load_now:
        return _load_caption_model_if_needed(model_id, _caption_device, _caption_translations_global)
    else:
        # Si on ne charge pas maintenant, retourner un message indiquant que c'est initialisé mais pas chargé.
        return True, t("image_prompter_initialized_not_loaded") # Nouvelle clé de traduction

def unload_caption_model(translations_param=None):
    """Décharge explicitement le modèle de génération de prompt."""
    global _caption_model, _caption_processor, _current_caption_model_id
    t = _get_translate_func(translations_param)

    if _caption_model is None and _caption_processor is None:
        msg = t("caption_model_not_loaded_to_unload")
        print(txt_color("[INFO]", "info"), msg)
        return msg

    unloading_model_id = _current_caption_model_id or "Unknown"
    print(txt_color("[INFO]", "info"), t("caption_model_unloading").format(model_id=unloading_model_id))
    _caption_model = None
    _caption_processor = None
    _current_caption_model_id = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    msg = t("caption_model_unloaded_success").format(model_id=unloading_model_id)
    print(txt_color("[OK]", "ok"), msg)
    return msg

def generate_prompt_from_image(image: Image.Image, translations_param: dict, task: str = DEFAULT_FLORENCE2_TASK, unload_after=False, model_id=MODEL_ID_FLORENCE2):
    """Génère un prompt détaillé à partir d'une image PIL."""
    global _caption_model, _caption_processor, _caption_device # _caption_device est mis à jour par _load_caption_model_if_needed
    t = _get_translate_func(translations_param)

    # Charger le modèle si besoin (ou s'il a changé)
    # _caption_device est utilisé comme device par défaut s'il est déjà défini, sinon le device de init sera utilisé.
    # S'il n'a jamais été défini (ne devrait pas arriver si init_image_prompter est appelé), il faudrait un fallback.
    device_to_use = _caption_device if _caption_device else "cpu" # Fallback simple

    loaded_successfully, load_message = _load_caption_model_if_needed(model_id, device_to_use, translations_param)
    if not loaded_successfully:
        return f"[{t('erreur').upper()}] {load_message}"

    if task not in FLORENCE2_TASKS:
        print(txt_color("[WARN]", "warning"), t("invalid_florence2_task").format(task=task, default_task=DEFAULT_FLORENCE2_TASK, available_tasks=", ".join(FLORENCE2_TASKS)))
        task = DEFAULT_FLORENCE2_TASK

    print(txt_color("[INFO]", "info"), t("using_florence2_task").format(task=task))

    if not isinstance(image, Image.Image):
        msg = t("invalid_image_type_for_caption").format(type=type(image).__name__)
        print(txt_color("[WARN]", "warning"), msg)
        return f"[{t('erreur').upper()}] {msg}"

    try:
        # S'assurer que les inputs sont sur le même device que le modèle
        inputs = _caption_processor(text=task, images=image, return_tensors="pt").to(_caption_model.device)
        print(txt_color("[INFO]", "info"), t("prompt_calcul"))
        
        # Utiliser le dtype du modèle pour la génération
        model_dtype = _caption_model.dtype
        inputs = inputs.to(model_dtype)
        _caption_model.to(model_dtype)

        generated_ids = _caption_model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, do_sample=False, num_beams=3)
        generated_text = _caption_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = _caption_processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
        
        # Conserver la logique de parsing originale de l'utilisateur
        prompt = parsed_answer.get(task, "")
        if isinstance(prompt, str): # Appliquer strip seulement si c'est une chaîne
            prompt = prompt.strip('{}').strip('"')
            
        print(txt_color("[INFO]", "info"), t("prompt_calculé"), f"{prompt}")
        return prompt
    except Exception as e:
        error_msg = t("error_generating_caption").format(error=str(e))
        print(txt_color("[ERREUR]", "erreur"), error_msg)
        traceback.print_exc()
        return f"[{t('erreur').upper()}] {error_msg}"
    finally:
        if unload_after:
            unload_caption_model(translations_param) # Passer les traductions pour les logs
            print(txt_color("[INFO]", "info"), t("caption_model_unloaded_after_generation"))
        # Le torch.cuda.empty_cache() est déjà dans unload_caption_model