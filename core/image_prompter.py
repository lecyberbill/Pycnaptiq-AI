# core/image_prompter.py
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from Utils.utils import txt_color, translate
import traceback

# --- Configuration (peut être externalisée si nécessaire) ---
CAPTION_MODEL_ID = "MiaoshouAI/Florence-2-base-PromptGen-v2.0"

# --- Variables globales pour le modèle (chargées une seule fois) ---
_caption_model = None
_caption_processor = None
_caption_device = None

def init_image_prompter(device, translations):
    """Charge le modèle et le processeur pour la génération de prompt."""
    global _caption_model, _caption_processor, _caption_device
    print(txt_color("[INFO]", "info"), translate("initializing_image_prompter", translations))
    if _caption_model is not None and _caption_processor is not None:
        print(txt_color("[INFO]", "info"), translate("caption_model_already_loaded", translations)) # Ajouter cette clé
        return True, translate("caption_model_already_loaded", translations)

    _caption_device = device
    try:
        print(txt_color("[INFO]", "info"), translate("loading_caption_model", translations).format(CAPTION_MODEL_ID)) # Ajouter cette clé
        _caption_model = AutoModelForCausalLM.from_pretrained(CAPTION_MODEL_ID, trust_remote_code=True).to(_caption_device)
        _caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID, trust_remote_code=True)
        print(txt_color("[OK]", "ok"), translate("caption_model_loaded", translations).format(CAPTION_MODEL_ID)) # Ajouter cette clé
        return True, translate("caption_model_loaded", translations).format(CAPTION_MODEL_ID)
    except Exception as e:
        error_msg = translate("error_loading_caption_model", translations).format(CAPTION_MODEL_ID, e) # Ajouter cette clé
        print(txt_color("[ERREUR]", "erreur"), error_msg)
        traceback.print_exc()
        _caption_model = None
        _caption_processor = None
        return False, error_msg

def generate_prompt_from_image(image: Image.Image, translations: dict):
    """Génère un prompt détaillé à partir d'une image PIL."""
    if _caption_model is None or _caption_processor is None or _caption_device is None:
        error_msg = translate("caption_model_not_initialized", translations) # Ajouter cette clé
        print(txt_color("[ERREUR]", "erreur"), error_msg)
        return f"[{translate('erreur', translations).upper()}] {error_msg}" # Retourner un message d'erreur clair

    if not isinstance(image, Image.Image):
        print(txt_color("[WARN]", "warning"), translate("invalid_image_type_for_caption", translations).format(type(image))) # Ajouter cette clé
        return "" # Retourner vide si l'image n'est pas valide

    try:
        inputs = _caption_processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt").to(_caption_device)
        print(txt_color("[INFO]", "info"), translate("prompt_calcul", translations))
        generated_ids = _caption_model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, do_sample=False, num_beams=3)
        generated_text = _caption_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = _caption_processor.post_process_generation(generated_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height))
        prompt = parsed_answer.get('<DETAILED_CAPTION>', '').strip('{}').strip('"')
        print(txt_color("[INFO]", "info"), translate("prompt_calculé", translations), f"{prompt}")
        return prompt
    except Exception as e:
        error_msg = translate("error_generating_caption", translations).format(e) # Ajouter cette clé
        print(txt_color("[ERREUR]", "erreur"), error_msg)
        traceback.print_exc()
        return f"[{translate('erreur', translations).upper()}] {error_msg}"
    finally:
        # Libérer la mémoire GPU si possible (peut être optionnel selon l'usage)
        if _caption_device and _caption_device.type == 'cuda':
            torch.cuda.empty_cache()