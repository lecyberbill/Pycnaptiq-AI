# d:\image_to_text\cyberbill_SDXL\cyberbill_image_generator\modules\batch_generator_mod.py
import os
import json
import time
import gradio as gr
import pandas as pd
from pathlib import Path
# --- AJOUT: Importer 're' pour l'expression régulière ---
import re
from Utils.utils import txt_color, translate, lister_fichiers
from Utils.sampler_utils import get_sampler_choices, get_sampler_key_from_display_name
from core.translator import translate_prompt
from Utils.model_manager import ModelManager
from Utils.utils import GestionModule
# --- Chargement des métadonnées ---
module_json_path = os.path.join(os.path.dirname(__file__), "batch_generator_mod.json")
try:
    with open(module_json_path, 'r', encoding="utf-8") as f:
        module_data = json.load(f)
except FileNotFoundError:
    print(f"[ERREUR] Fichier JSON non trouvé pour Batch Generator: {module_json_path}")
    module_data = {"name": "Batch Generator (Erreur JSON)"}

# --- Fonction d'initialisation ---
def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
    """Initialise le module Générateur de Batch."""
    print(txt_color("[OK] ", "ok"), f"Initialisation du module: {module_data.get('name', 'Batch Generator')}")
    return BatchGeneratorModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

# --- Classe principale du module ---
class BatchGeneratorModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
        """Initialise la classe BatchGeneratorModule."""
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.gestionnaire = gestionnaire_instance
        self.global_config = global_config
        self.module_translations = {}

        self.models_dir = self.global_config.get("MODELS_DIR", "models")
        self.vae_dir = self.global_config.get("VAE_DIR", "vae")
        self.loras_dir = self.global_config.get("LORAS_DIR", "lora")
        self.styles_config = self.global_config.get("STYLES", [])
        self.formats_config = self.global_config.get("FORMATS", [{"dimensions": "1024*1024", "orientation": "orientation_carre"}])
        self.negative_prompt_default = self.global_config.get("NEGATIVE_PROMPT", "")
        # --- Lire le chemin de sauvegarde depuis la config ---
        self.save_batch_json_path = self.global_config.get("SAVE_BATCH_JSON_PATH", "Output\\json_batch_files") # Default

        print(txt_color("[INFO]", "info"), f"{module_data.get('name', 'Batch Generator')}: Configuration reçue: {'Oui' if self.global_config else 'Non'}")
        print(txt_color("[INFO]", "info"), f"{module_data.get('name', 'Batch Generator')}: Gestionnaire reçu: {'Oui' if self.gestionnaire else 'Non'}")
        # --- Afficher le chemin de sauvegarde ---
        print(txt_color("[INFO]", "info"), f"{module_data.get('name', 'Batch Generator')}: Chemin sauvegarde JSON: {self.save_batch_json_path}")

    # --- Méthode pour générer et sauvegarder le JSON ---
    def generate_batch_json(self, batch_list):
        """Génère la chaîne JSON et la sauvegarde automatiquement."""
        if not batch_list:
             gr.Warning(translate("batch_empty_warn", self.module_translations))
             return "[]"

        json_string = json.dumps(batch_list, indent=2, ensure_ascii=False)

        # --- Logique de sauvegarde automatique ---
        try:
            save_dir = Path(self.save_batch_json_path)
            save_dir.mkdir(parents=True, exist_ok=True) # Créer le dossier si besoin

            # Trouver le prochain numéro de fichier
            existing_files = save_dir.glob("batch_*.json")
            max_num = 0
            for f in existing_files:
                match = re.match(r"batch_(\d+)\.json", f.name)
                if match:
                    max_num = max(max_num, int(match.group(1)))

            next_num = max_num + 1
            new_filename = f"batch_{next_num:03d}.json" # Nomme batch_001.json, batch_002.json etc.
            filepath = save_dir / new_filename

            # Écrire le fichier
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_string)

            # --- Message Gradio ---
            success_message_ui = translate("auto_save_json_success_msg", self.module_translations).format(filepath=str(filepath))
            gr.Info(success_message_ui)
            # --- Message Console ---
            success_message_log = translate("log_save_json_success", self.module_translations).format(filepath=str(filepath))
            print(f"{txt_color('[OK]', 'ok')} {success_message_log}")

        except Exception as e:
            # --- Message Gradio ---
            error_message_ui = translate("auto_save_json_error_msg", self.module_translations).format(path=self.save_batch_json_path, error=str(e))
            gr.Error(error_message_ui)
            # --- Message Console ---
            error_message_log = translate("log_save_json_error", self.module_translations).format(filepath=self.save_batch_json_path, error=str(e))
            print(f"{txt_color('[ERREUR]', 'erreur')} {error_message_log}")

        return json_string # Retourne toujours le JSON pour l'affichage

    def create_tab(self, module_translations):
        """Crée l'onglet Gradio pour le générateur de batch."""
        self.module_translations = module_translations
        tab_name = translate("batch_generator_tab_name", self.module_translations)
        tab_title = translate("batch_generator_tab_title", self.module_translations)

        available_models = lister_fichiers(self.models_dir, self.global_translations, gradio_mode=True)
        available_vaes = ["Auto"] + lister_fichiers(self.vae_dir, self.global_translations, ext=".safetensors", gradio_mode=True)
        available_loras = lister_fichiers(self.loras_dir, self.global_translations, gradio_mode=True)
        has_loras = bool(available_loras) and translate("aucun_modele_trouve", self.global_translations) not in available_loras and translate("repertoire_not_found", self.global_translations) not in available_loras
        lora_choices = available_loras if has_loras else [translate("aucun_lora_disponible", self.module_translations)]
        style_choices = [translate(style.get("key", ""), self.global_translations) for style in self.styles_config if style.get("key") != "style_none"]
        format_choices = [f'{item["dimensions"]} {translate(item["orientation"], self.global_translations)}' for item in self.formats_config]
        default_format = format_choices[0] if format_choices else "1024*1024"
        sampler_display_choices = get_sampler_choices(self.global_translations)
        default_sampler_display = translate("sampler_euler", self.global_translations)

        dataframe_columns = [
            "model", "vae", "prompt", "negative_prompt", "styles",
            "sampler_key", "steps", "guidance_scale", "seed", "num_images",
            "width", "height", "loras", "output_filename", "rating", "notes"
        ]

        # --- Fonctions Logiques (internes à create_tab) ---
        def add_task_to_batch(current_batch_list,
                              model, vae, sampler_display,
                              prompt_input, neg_prompt_input, styles_list,
                              translate_prompt_value,
                              steps, cfg, seed,
                              format_str,
                              filename, num_images_input,
                              *loras_all_inputs):
            """Ajoute une nouvelle tâche à la liste du batch."""
            sampler_key = get_sampler_key_from_display_name(sampler_display, self.global_translations)
            if not sampler_key:
                warning_message = translate("sampler_not_recognized_warn", self.module_translations).format(sampler=sampler_display)
                gr.Warning(warning_message)
                sampler_key = "sampler_euler"

            try:
                dims_part = format_str.split(" ")[0]
                width, height = map(int, dims_part.split('*'))
            except Exception:
                width, height = 1024, 1024

            active_loras_list = []
            num_lora_slots = 4
            for i in range(num_lora_slots):
                check_idx, dd_idx, scale_idx = i * 3, i * 3 + 1, i * 3 + 2
                is_active = loras_all_inputs[check_idx]
                lora_name = loras_all_inputs[dd_idx]
                lora_weight = loras_all_inputs[scale_idx]
                if is_active and lora_name != translate("aucun_lora_disponible", self.module_translations):
                    active_loras_list.append({"name": lora_name, "weight": float(lora_weight)})

            prompt_final = prompt_input
            if translate_prompt_value:
                try:
                    prompt_traduit = translate_prompt(prompt_input, self.module_translations)
                    print(f"Prompt original: {prompt_input} -> Traduit: {prompt_traduit}")
                    prompt_final = prompt_traduit
                except Exception as e:
                    print(f"{txt_color('[ERREUR]', 'erreur')} Échec de la traduction du prompt: {e}")
                    warning_message = translate("prompt_translation_failed_warn", self.module_translations).format(error=str(e))
                    gr.Warning(warning_message)
                    prompt_final = prompt_input

            new_task = {
                "model": model if model else None, "vae": vae if vae else "Auto",
                "original_prompt": prompt_input, "prompt": prompt_final,
                "negative_prompt": neg_prompt_input if neg_prompt_input else self.negative_prompt_default,
                "styles": styles_list if styles_list else [],
                "sampler_key": sampler_key,
                "steps": int(steps) if steps is not None else 25,
                "guidance_scale": float(cfg) if cfg is not None else 7.0,
                "seed": int(seed) if seed is not None else -1,
                "num_images": int(num_images_input) if num_images_input is not None else 1,
                "width": width, "height": height, "loras": active_loras_list,
                "output_filename": filename if filename else None,
                "rating": 0, "notes": ""
            }
            current_batch_list.append(new_task)
            df = pd.DataFrame(current_batch_list, columns=dataframe_columns)
            gr.Info(translate("task_added_msg", self.module_translations))
            return current_batch_list, df

        def clear_batch_tasks():
            """Vide la liste du batch et le DataFrame."""
            gr.Info(translate("batch_cleared_msg", self.module_translations))
            return [], pd.DataFrame(columns=dataframe_columns)

        # --- Interface Gradio ---
        with gr.Tab(tab_name) as tab:
            gr.Markdown(f"## {tab_title}")
            gr.Markdown(translate("batch_generator_description", self.module_translations))

            batch_state = gr.State([])

            with gr.Blocks():
                gr.Markdown(f"### {translate('param_section_title', self.module_translations)}")
                with gr.Row():
                    prompt_input = gr.Textbox(label=translate("prompt_label", self.module_translations), lines=3, scale=3)
                    neg_prompt_input = gr.Textbox(label=translate("neg_prompt_label", self.module_translations), lines=3, placeholder=self.negative_prompt_default, scale=2)
                    translate_prompt_checkbox = gr.Checkbox(label=translate("batch_translate_prompt_checkbox", self.module_translations), value=False, info=translate("batch_translate_prompt_info", self.module_translations), scale=1)
                    steps_input = gr.Number(label=translate("steps_label", self.module_translations), value=25, precision=0, minimum=1, scale=1)
                    cfg_input = gr.Number(label=translate("cfg_scale_label", self.module_translations), value=7.0, minimum=0.0, scale=1)
                    seed_input = gr.Number(label=translate("seed_label", self.module_translations), value=-1, precision=0, scale=1)
                    num_images_input = gr.Number(label=translate("num_images_label", self.module_translations), value=1, precision=0, minimum=1, maximum=100, step=1, scale=1) # AJOUT DU CHAMP
                with gr.Row():
                    format_input = gr.Dropdown(label=translate("format", self.global_translations), choices=format_choices, value=default_format, scale=1)
                    model_input = gr.Dropdown(label=translate("selectionner_modele", self.global_translations), choices=available_models, scale=2, allow_custom_value=True)
                    vae_input = gr.Dropdown(label=translate("selectionner_vae", self.global_translations), choices=available_vaes, value="Auto", scale=2)
                    sampler_input = gr.Dropdown(label=translate("selectionner_sampler", self.global_translations), choices=sampler_display_choices, value=default_sampler_display, scale=2)
                    style_input = gr.Dropdown(label=translate("styles", self.global_translations), choices=style_choices, multiselect=True, max_choices=4, scale=2)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        lora_inputs_components = []
                        with gr.Accordion(translate("lora_accordion_title", self.module_translations), open=False):
                            for r in range(2):
                                with gr.Row():
                                    for i_rel in range(2):
                                        i_abs = r * 2 + i_rel
                                        with gr.Column():
                                            with gr.Group():
                                                lora_check = gr.Checkbox(label=translate("lora_checkbox_label", self.module_translations).format(i_abs + 1), value=False)
                                                lora_dropdown = gr.Dropdown(choices=lora_choices, label=translate("selectionner_lora", self.global_translations), interactive=False, value=(lora_choices[0] if lora_choices else None))
                                                lora_scale = gr.Slider(0, 1, value=0.8, label=translate("poids_lora", self.global_translations))
                                                lora_check.change(lambda x: gr.update(interactive=x), inputs=lora_check, outputs=lora_dropdown)
                                                lora_inputs_components.extend([lora_check, lora_dropdown, lora_scale])
                    with gr.Column(scale=1):
                         filename_input = gr.Textbox(label=translate("output_filename_label", self.module_translations), placeholder=translate("output_filename_placeholder", self.module_translations))
                         add_button = gr.Button(translate("add_task_button", self.module_translations), variant="primary")

            with gr.Blocks():
                gr.Markdown(f"### {translate('batch_list_title', self.module_translations)}")
                with gr.Row():
                    batch_display = gr.DataFrame(headers=dataframe_columns, interactive=False, wrap=True, scale=3)
                    with gr.Column(scale=1):
                         generate_button = gr.Button(translate("generate_json_button", self.module_translations))
                         clear_button = gr.Button(translate("clear_batch_button", self.module_translations), variant="stop")
                with gr.Row():
                    json_output_display = gr.JSON(label=translate("json_output_label", self.module_translations))


            # --- Connexions des événements ---
            add_button_inputs = [
                batch_state, model_input, vae_input, sampler_input,
                prompt_input, neg_prompt_input, style_input,
                translate_prompt_checkbox, steps_input, cfg_input, seed_input, # num_images_input sera ici
                format_input, filename_input, num_images_input, *lora_inputs_components # AJOUT DE num_images_input
            ]
            add_button.click(fn=add_task_to_batch, inputs=add_button_inputs, outputs=[batch_state, batch_display])

            # --- generate_button appelle maintenant la méthode de classe ---
            generate_button.click(
                fn=self.generate_batch_json, # Utilise la méthode de l'instance
                inputs=[batch_state],
                outputs=[json_output_display]
            )

            clear_button.click(fn=clear_batch_tasks, inputs=[], outputs=[batch_state, batch_display])

            # --- SUPPRESSION: Connexion pour le bouton de sauvegarde manuelle ---

        return tab

    # --- SUPPRESSION: La fonction save_batch_json_to_file n'est plus nécessaire ici ---
