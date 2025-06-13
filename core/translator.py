import os
from transformers import pipeline
from Utils.utils import txt_color, translate
import gradio as gr

# --- Chargement du modèle de traduction ---



try:
    # Vous pouvez rendre le nom du modèle configurable si nécessaire
    translation_model_name = "Helsinki-NLP/opus-mt-fr-en"
    translator = pipeline("translation", model=translation_model_name)
    print(txt_color("[OK]", "ok"), f"Translation model '{translation_model_name}' loaded.")
except Exception as e:
    print(txt_color("[ERREUR]", "erreur"), f"Could not load translation model '{translation_model_name}': {e}")
    translator = None # Définir translator à None en cas d'échec


# --- Fonction de traduction ---
def translate_prompt(prompt_fr, translations):
    """
    Translates a French prompt into English using the loaded translator instance.

    Args:
        prompt_fr (str): The French prompt to translate.
        translations (dict): The dictionary containing translations for messages.

    Returns:
        str: The translated English prompt.

    Raises:
        RuntimeError: If the translator model is not loaded.
        ValueError: If there is an error during translation.
    """

    if translator is None:
        error_message = translate('erreur_traducteur_non_charge', translations)
        print(txt_color("[ERREUR]", "erreur"), error_message)
        # Lever une exception pour signaler l'erreur clairement
        raise RuntimeError(error_message)

    try:
        # Vérifier si le prompt est vide ou None
        if not prompt_fr:
            return "" # Retourner une chaîne vide si le prompt est vide

        # Utiliser l'instance 'translator' chargée dans ce module
        traduction = translator(prompt_fr)[0]["translation_text"]
        # Utiliser les traductions passées pour les messages de log
        print(txt_color("[INFO]", "info"), f"{translate('traduction_effectuee', translations)}: {prompt_fr} -> {traduction}")
        return traduction
    except Exception as e:
        error_message = translate('erreur_traduction', translations)
        print(txt_color("[ERREUR]", "erreur"), f"{error_message}: {e}")
        # Lever une exception standard pour indiquer l'échec
        raise ValueError(f"{error_message}: {e}")