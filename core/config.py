# core/config.py
import os
import gradio as gr
from datetime import datetime
from Utils.utils import (
    charger_configuration, load_locales, translate, 
    check_gpu_availability, txt_color
)
from core.version import version, version_date
from presets.presets_Manager import PresetManager
from core.image_prompter import MODEL_ID_FLORENCE2 as DEFAULT_FLORENCE2_MODEL_ID_FROM_PROMPTER
from core.logger import logger

# 1. Charger la configuration brute
config = charger_configuration()

# 2. Initialisation de la langue et des traductions
DEFAULT_LANGUAGE = config.get("LANGUAGE", "fr")
translations = load_locales(DEFAULT_LANGUAGE)

# 3. Métadonnées de version
APP_VERSION = version()
APP_VERSION_DATE_OBJ = version_date()

def get_formatted_version_date(lang):
    """Retourne la date de version formatée selon la langue."""
    mois_fr = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
    mois_en = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    
    day = APP_VERSION_DATE_OBJ.day
    month_idx = APP_VERSION_DATE_OBJ.month - 1
    year = APP_VERSION_DATE_OBJ.year
    
    if lang == "fr":
        return f"{day} {mois_fr[month_idx]} {year}"
    elif lang == "en":
        return f"{mois_en[month_idx]} {day}, {year}"
    return APP_VERSION_DATE_OBJ.isoformat()

APP_VERSION_DATE_STR = get_formatted_version_date(DEFAULT_LANGUAGE)

# 4. Chemins et dossiers
MODELS_DIR = config["MODELS_DIR"]
VAE_DIR = config["VAE_DIR"]
LORAS_DIR = config["LORAS_DIR"]
INPAINT_MODELS_DIR = config["INPAINT_MODELS_DIR"]
SAVE_DIR = config["SAVE_DIR"]
SAVE_BATCH_JSON_PATH = config.get("SAVE_BATCH_JSON_PATH", "Output\\json_batch_files")
CONTROLNET_DIR = config.get("CONTROLNET_DIR", "models/controlnet")
IP_ADAPTER_DIR = config.get("IP_ADAPTER_DIR", "models/ip_adapter")

# 5. Paramètres d'interface et génération
IMAGE_FORMAT = config.get("IMAGE_FORMAT", "WEBP").upper()
if IMAGE_FORMAT not in ["PNG", "JPG", "WEBP"]:
    print(f"⚠️ Format {IMAGE_FORMAT}", f"{txt_color(translate('non_valide', translations), 'erreur')}", translate("utilisation_webp", translations))
    IMAGE_FORMAT = "WEBP"

RAW_FORMATS = config["FORMATS"]
NEGATIVE_PROMPT = config["NEGATIVE_PROMPT"]
GRADIO_THEME = config["GRADIO_THEME"]
AUTHOR = config["AUTHOR"]
SHARE = config["SHARE"]
OPEN_BROWSER = config["OPEN_BROWSER"]
DEFAULT_MODEL = config["DEFAULT_MODEL"]
PRESETS_PER_PAGE = config.get("PRESETS_PER_PAGE", 12)
PRESET_COLS_PER_ROW = config.get("PRESET_COLS_PER_ROW", 4)
PREVIEW_QUEUE = []

# 6. Styles (traduits)
for style in config["STYLES"]:
    style["name"] = translate(style["key"], translations)
STYLES = config["STYLES"]

# 7. Matériel (Device / Dtype)
device, torch_dtype, vram_total_gb = check_gpu_availability(translations)

# 8. Modèles spécifiques (LLM, Florence)
LLM_PROMPTER_MODEL_PATH = config.get("LLM_PROMPTER_MODEL_PATH", "Qwen/Qwen3-0.6B")
FLORENCE2_MODEL_ID_CONFIG = config.get("FLORENCE2_MODEL_ID", DEFAULT_FLORENCE2_MODEL_ID_FROM_PROMPTER)

def print_config_summary():
    """Affiche un résumé de la configuration au démarrage."""
    logger.info(f"{translate('app_version_label', translations)} {APP_VERSION}")
    logger.info(f"{translate('version_date_label', translations)}: {APP_VERSION_DATE_STR}")
