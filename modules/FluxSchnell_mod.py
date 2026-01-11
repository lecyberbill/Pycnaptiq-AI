# FluxSchnell_mod.py
import os
import json
import queue
import threading
import time
import gc
import traceback
from datetime import datetime
import random

import gradio as gr
import torch
from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline # <-- MODIFIÉ pour inclure FluxImg2ImgPipeline

from Utils.utils import (
    txt_color,
    translate,
    enregistrer_image,
    preparer_metadonnees_image,
    enregistrer_etiquettes_image_html,
    create_progress_bar_html,
    ImageSDXLchecker, # <-- AJOUT pour vérifier l'image d'entrée
    styles_fusion, # <-- AJOUT POUR LES STYLES
)
from Utils.callback_diffuser import create_inpainting_callback
from Utils.model_manager import ModelManager, FLUX_SCHNELL_MODEL_ID, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY, HuggingFaceAuthError # <-- AJOUT HuggingFaceAuthError
from core.translator import translate_prompt
from diffusers.utils import load_image # Bien que non utilisé directement ici, c'est dans l'exemple
from core.image_prompter import generate_prompt_from_image, FLORENCE2_TASKS # AJOUT pour image_prompter
from Utils import llm_prompter_util # <-- AJOUT pour l'amélioration du prompt
from Utils.gradio_components import create_prompt_interface, create_generation_settings, create_lora_interface, create_output_interface


MODULE_NAME = "FluxSchnell"

module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

# FLUX_PRIOR_REDUX_MODEL_ID n'est plus nécessaire car nous utilisons FluxPipeline pour img2img
# FLUX_PRIOR_REDUX_MODEL_ID = "black-forest-labs/FLUX.1-Redux-dev" 
try:
    with open(module_json_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)
except FileNotFoundError:
    print(f"[ERREUR] Fichier JSON du module {MODULE_NAME} introuvable.")
    module_data = {"name": MODULE_NAME} 
except json.JSONDecodeError:
    print(f"[ERREUR] Erreur de décodage JSON pour le module {MODULE_NAME}.")
    module_data = {"name": MODULE_NAME} 

def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME)}")
    return FluxSchnellModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class FluxSchnellModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.gestionnaire = gestionnaire_instance
        self.stop_event = threading.Event()
        self.styles = self.load_styles() # <-- AJOUT POUR LES STYLES
        self.default_negative_prompt = self.global_config.get("NEGATIVE_PROMPT", "") # <-- AJOUT POUR LES STYLES
        self.module_translations = {}
        # --- AJOUT: Initialisation pour l'amélioration du prompt ---
        self.llm_prompter_model_path = self.global_config.get("LLM_PROMPTER_MODEL_PATH", "Qwen/Qwen3-0.6B")
        # Les états pour l'amélioration du prompt seront initialisés dans create_tab
        # --- FIN AJOUT ---

        # --- AJOUT: UI components for Hugging Face login ---
        self.hf_token_textbox = None
        self.hf_login_button = None
        self.hf_login_group = None # To control visibility of all login elements
        
        # Dimensions spécifiques à FLUX.1-schnell
        self.flux_dimensions = [
            "704x1408", "704x1344", "768x1344", "768x1280", "832x1216", "832x1152",
            "896x1152", "896x1088", "960x1088", "960x1024", "1024x1024", "1024x960",
            "1088x960", "1088x896", "1152x896", "1152x832", "1216x832", "1280x768",
            "1344x768", "1344x704", "1408x704", "1472x704", "1536x640", "1600x640",
            "1664x576", "1728x576"
        ]
        # Obtenir les LoRAs disponibles
        self.available_loras = self.model_manager.list_loras(gradio_mode=True)
        self.has_loras = bool(self.available_loras) and \
                         translate("aucun_modele_trouve", self.global_translations) not in self.available_loras and \
                         translate("repertoire_not_found", self.global_translations) not in self.available_loras
        self.lora_choices_for_ui = self.available_loras if self.has_loras else [translate("aucun_lora_disponible", self.global_translations)]
        # --- AJOUT: Logique pour les modèles FLUX locaux ---
        self.flux_models_dir = self.model_manager.flux_models_dir
        self.available_flux_models = self.list_flux_models()



    def refresh_lora_list(self):
        """Rafraîchit la liste des LoRAs disponibles et met à jour les dropdowns."""
        print(txt_color("[INFO]", "info"), translate("refreshing_lora_list", self.module_translations)) # Nouvelle clé
        self.available_loras = self.model_manager.list_loras(gradio_mode=True)
        self.has_loras = bool(self.available_loras) and translate("aucun_modele_trouve", self.global_translations) not in self.available_loras and translate("repertoire_not_found", self.global_translations) not in self.available_loras
        new_choices = self.available_loras if self.has_loras else [translate("aucun_lora_disponible", self.global_translations)]
        gr.Info(translate("lora_list_refreshed", self.module_translations).format(count=len(self.available_loras) if self.has_loras else 0)) # Nouvelle clé
        # Retourner une liste d'updates pour chaque dropdown LoRA
        return [gr.update(choices=new_choices, interactive=self.has_loras) for _ in self.flux_lora_dropdowns]
    def load_styles(self):
        """Charge les styles depuis styles.json."""
        # Chemin vers le fichier styles.json dans le dossier config
        styles_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "styles.json" # Remonte de deux niveaux pour atteindre la racine du projet puis config
        )
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                styles_data = json.load(f)
            for style in styles_data: # Traduire les noms des styles
                style["name"] = translate(style["key"], self.global_translations)
            return styles_data
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"Erreur chargement styles.json pour FLUX: {e}")
            return []

    def list_flux_models(self):
        """Scanne le répertoire des modèles FLUX et retourne une liste de modèles disponibles."""
        models = [] # On ne met plus le modèle HF par défaut ici
        if not os.path.isdir(self.flux_models_dir):
            print(f"[AVERTISSEMENT] Le répertoire des modèles FLUX n'a pas été trouvé : {self.flux_models_dir}")
            os.makedirs(self.flux_models_dir, exist_ok=True)
        try:
            for f in os.listdir(self.flux_models_dir):
                if f.endswith(".safetensors"):
                    models.append(f)
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"Erreur lors du scan du répertoire des modèles FLUX : {e}")
        return models

    def refresh_flux_models_ui(self):
        """Scanne à nouveau le répertoire des modèles et met à jour le dropdown."""
        print(txt_color("[INFO]", "info"), translate("refreshing_flux_model_list_log", self.module_translations)) # Nouvelle clé
        self.available_flux_models = self.list_flux_models()
        gr.Info(translate("flux_model_list_refreshed", self.module_translations).format(count=len(self.available_flux_models))) # Nouvelle clé
        return gr.update(choices=self.available_flux_models)

    def stop_generation(self):
        self.stop_event.set()
        print(txt_color("[INFO]", "info"), translate("stop_requested", self.module_translations))

    def create_tab(self, module_translations):
        self.module_translations = module_translations

        with gr.Tab(translate("flux_schnell_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('flux_schnell_tab_title', self.module_translations)}")

            # --- AJOUT: États pour l'amélioration du prompt ---
            flux_original_user_prompt_state = gr.State(value="")
            flux_current_prompt_is_enhanced_state = gr.State(value=False)
            flux_enhancement_cycle_active_state = gr.State(value=False)
            flux_last_ai_enhanced_output_state = gr.State(value=None)
            # --- FIN AJOUT ---

            with gr.Row():
                # --- COLONNE 1: SETTINGS (Prompt, Gen Settings, Image inputs) ---
                with gr.Column(scale=1, variant="panel"):
                    (self.flux_prompt, 
                     self.flux_enhance_or_redo_button, 
                     self.flux_validate_prompt_button, 
                     self.flux_traduire_checkbox, 
                     self.flux_style_dropdown) = create_prompt_interface(self, self.module_translations, prefix="flux_schnell_")

                    (self.flux_resolution_dropdown, 
                     self.flux_steps_slider, 
                     self.flux_guidance_scale_slider, 
                     self.flux_seed_input, 
                     self.flux_num_images_slider) = create_generation_settings(
                         self, self.module_translations, 
                         prefix="flux_schnell_",
                         allowed_resolutions=self.flux_dimensions, 
                         default_steps=4, max_steps=20
                     )
                     
                    self.flux_use_image_prompt_checkbox = gr.Checkbox(
                        label=translate("generer_prompt_image", self.global_translations),
                        value=False
                    )
                    self.flux_image_input_for_prompt = gr.Image(
                        label=translate("telechargez_image", self.global_translations),
                        type="pil", 
                        visible=False
                    )
                    self.flux_use_img2img_checkbox = gr.Checkbox(
                        label=translate("flux_schnell_use_img2img_label", self.module_translations),
                        value=False,
                        info=translate("flux_schnell_use_img2img_info", self.module_translations)
                    )
                    self.flux_img2img_image_input = gr.Image(
                        label=translate("flux_schnell_img2img_input_label", self.module_translations),
                        type="pil",
                        elem_id="flux_schnell_img2img_input",
                        visible=False
                    )
                    self.flux_img2img_strength_slider = gr.Slider( 
                        minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                        label=translate("flux_schnell_img2img_strength_label", self.module_translations),
                        info=translate("flux_schnell_img2img_strength_info", self.module_translations),
                        visible=False
                    )

                # --- COLONNE 2: RENDU & GESTION ---
                with gr.Column(scale=1, variant="panel"):
                    # --- Partie 1: Sortie (Status, Galerie, Boutons) ---
                    (self.flux_message_chargement, 
                     self.flux_result_output, 
                     self.flux_bouton_gen, 
                     self.flux_bouton_stop, 
                     self.flux_progress_html) = create_output_interface(self.module_translations, prefix="flux_schnell_")

                    gr.Markdown("---")

                    # --- Partie 2: GESTION (Modèles, LoRAs) ---
                    with gr.Group():
                        gr.Markdown(f"### {translate('flux_schnell_model_select_label', self.module_translations)}")
                        with gr.Row():
                            self.flux_model_dropdown = gr.Dropdown(
                                label=None,
                                choices=self.available_flux_models,
                                value=self.available_flux_models[0] if self.available_flux_models else None,
                                info=translate("flux_schnell_model_select_info", self.module_translations),
                                scale=8,
                            )
                            self.flux_refresh_models_button = gr.Button(
                                value="🔄", min_width=60, scale=1, elem_id="flux_refresh_models_button"
                            )

                        with gr.Row():
                            self.flux_bouton_charger_default = gr.Button(
                                translate("flux_schnell_load_default_button", self.module_translations),                 
                            )
                            self.flux_bouton_charger_local = gr.Button(
                                translate("flux_schnell_load_local_button", self.module_translations),               
                            )

                        self.flux_bouton_charger_img2img = gr.Button(
                            translate("flux_schnell_load_img2img_button", self.module_translations), 
                            visible=False
                        )

                    # --- Hugging Face Login UI ---
                    with gr.Group(visible=False) as self.hf_login_group:
                        gr.Markdown(
                            f"**{translate('hf_login_instructions_title', self.module_translations)}**\n"
                            f"{translate('hf_login_instructions_step1', self.module_translations)}\n"
                            f"{translate('hf_login_instructions_step2', self.module_translations)}\n"
                            f"{translate('hf_login_instructions_step3', self.module_translations)}\n"
                            f"{translate('hf_login_instructions_step4', self.module_translations)}\n"
                            f"{translate('hf_login_instructions_step5', self.module_translations)}\n"
                            f"{translate('hf_login_instructions_step6', self.module_translations)}\n"
                            f"{translate('hf_login_instructions_step7', self.module_translations)}\n"
                            f"{translate('hf_login_instructions_step8', self.module_translations)}\n"
                            f"{translate('hf_login_instructions_step9', self.module_translations)}"
                        )
                        self.hf_token_textbox = gr.Textbox(
                            label=translate("hf_token_label", self.module_translations),
                            type="password",
                            placeholder=translate("hf_token_placeholder", self.module_translations)
                        )
                    self.hf_login_button = gr.Button(
                        translate("hf_login_button", self.module_translations),
                    )

                    gr.Markdown("---")
                    
                    (self.flux_lora_checks, 
                     self.flux_lora_dropdowns, 
                     self.flux_lora_scales, 
                     self.flux_lora_message, 
                     self.flux_refresh_lora_button) = create_lora_interface(self, self.module_translations)

            # --- CLICK LISTENERS FOR MODEL LOADING BUTTONS ---
            self.flux_bouton_charger_default.click(
                fn=self.load_default_text_to_image_model_ui,
                inputs=[],
                outputs=[
                    self.flux_message_chargement,
                    self.flux_bouton_gen,
                    self.hf_login_group,
                    self.hf_token_textbox,
                ],
            )

            self.flux_bouton_charger_local.click(
                fn=self.load_local_text_to_image_model_ui,
                inputs=[
                    self.flux_model_dropdown,
                ],
                outputs=[
                    self.flux_message_chargement,
                    self.flux_bouton_gen,
                    self.hf_login_group,
                    self.hf_token_textbox,
                ],
            )

            self.flux_bouton_charger_img2img.click(
                fn=self.load_image_to_image_model_ui,
                inputs=[],
                outputs=[
                    self.flux_message_chargement,
                    self.flux_bouton_gen,
                    self.hf_login_group,
                    self.hf_token_textbox,
                ],
            )

            self.hf_login_button.click(
                fn=self.login_and_retry_load_flux_model_ui,
                inputs=[self.hf_token_textbox],
                outputs=[
                    self.flux_message_chargement,
                    self.flux_bouton_gen,
                    self.hf_login_group,
                    self.hf_token_textbox,
                ]
            )

            # --- GEN BUTTON ---
            flux_gen_inputs = [
                self.flux_prompt,
                self.flux_traduire_checkbox,
                self.flux_style_dropdown,
                self.flux_num_images_slider,
                self.flux_use_img2img_checkbox,
                self.flux_steps_slider,
                self.flux_resolution_dropdown,
                self.flux_guidance_scale_slider,
                self.flux_img2img_image_input,
                self.flux_img2img_strength_slider,
                self.flux_seed_input,
                flux_original_user_prompt_state,
                flux_current_prompt_is_enhanced_state,
                flux_enhancement_cycle_active_state,
            ]
            for chk in self.flux_lora_checks: flux_gen_inputs.append(chk)
            for dd in self.flux_lora_dropdowns: flux_gen_inputs.append(dd)
            for sc in self.flux_lora_scales: flux_gen_inputs.append(sc)

            self.flux_bouton_gen.click(
                fn=self.flux_schnell_gen,
                inputs=flux_gen_inputs,
                outputs=[
                    self.flux_result_output,
                    self.flux_progress_html,
                    self.flux_bouton_gen,
                    self.flux_bouton_stop,
                    self.flux_lora_message
                ],
            )

            # --- EVENTS ---
            self.flux_use_image_prompt_checkbox.change(
                fn=lambda use_image: gr.update(visible=use_image),
                inputs=self.flux_use_image_prompt_checkbox,
                outputs=self.flux_image_input_for_prompt
            )
            self.flux_image_input_for_prompt.change(
                fn=self.update_prompt_from_image_flux,
                inputs=[self.flux_image_input_for_prompt, self.flux_use_image_prompt_checkbox, gr.State(self.module_translations)],
                outputs=self.flux_prompt
            )
            self.flux_img2img_image_input.change(
                fn=lambda img: gr.update(interactive=img is None),
                inputs=self.flux_img2img_image_input,
                outputs=self.flux_resolution_dropdown
            )
            self.flux_use_img2img_checkbox.change(
                fn=lambda use_img2img: (
                    gr.update(visible=use_img2img),
                    gr.update(visible=use_img2img),
                    gr.update(visible=use_img2img),
                    gr.update(visible=not use_img2img),
                    gr.update(visible=not use_img2img),
                    gr.update(visible=not use_img2img)
                ),
                inputs=self.flux_use_img2img_checkbox,
                outputs=[
                    self.flux_img2img_image_input,
                    self.flux_img2img_strength_slider,
                    self.flux_bouton_charger_img2img,
                    self.flux_model_dropdown,
                    self.flux_bouton_charger_default,
                    self.flux_bouton_charger_local
                ]
            )

            self.flux_refresh_lora_button.click(
                fn=self.refresh_lora_list,
                inputs=None,
                outputs=self.flux_lora_dropdowns
            )
            self.flux_refresh_models_button.click(
                fn=self.refresh_flux_models_ui,
                inputs=None,
                outputs=[self.flux_model_dropdown]
            )

            # --- LLM ENHANCEMENT ---
            self.flux_enhance_or_redo_button.click(
                fn=self.on_flux_enhance_or_redo_button_click,
                inputs=[self.flux_prompt, flux_original_user_prompt_state, flux_enhancement_cycle_active_state, gr.State(self.llm_prompter_model_path), gr.State(self.module_translations)],
                outputs=[self.flux_prompt, self.flux_enhance_or_redo_button, self.flux_validate_prompt_button, flux_original_user_prompt_state, flux_current_prompt_is_enhanced_state, flux_enhancement_cycle_active_state, flux_last_ai_enhanced_output_state]
            )
            self.flux_validate_prompt_button.click(
                fn=self.on_flux_validate_prompt_button_click,
                inputs=[self.flux_prompt, gr.State(self.module_translations)],
                outputs=[self.flux_enhance_or_redo_button, self.flux_validate_prompt_button, flux_original_user_prompt_state, flux_current_prompt_is_enhanced_state, flux_enhancement_cycle_active_state, flux_last_ai_enhanced_output_state]
            )
            self.flux_prompt.input(
                fn=self.handle_flux_text_input_change,
                inputs=[self.flux_prompt, flux_last_ai_enhanced_output_state, flux_enhancement_cycle_active_state, gr.State(self.llm_prompter_model_path), gr.State(self.module_translations)],
                outputs=[
                    self.flux_enhance_or_redo_button,
                    self.flux_validate_prompt_button,
                    flux_original_user_prompt_state,
                    flux_current_prompt_is_enhanced_state,
                    flux_enhancement_cycle_active_state,
                    flux_last_ai_enhanced_output_state
                ]
            )
            self.flux_prompt.submit(
                fn=self.handle_flux_text_input_change,
                inputs=[self.flux_prompt, flux_last_ai_enhanced_output_state, flux_enhancement_cycle_active_state, gr.State(self.llm_prompter_model_path), gr.State(self.module_translations)],
                outputs=[
                    self.flux_enhance_or_redo_button,
                    self.flux_validate_prompt_button,
                    flux_original_user_prompt_state,
                    flux_current_prompt_is_enhanced_state,
                    flux_enhancement_cycle_active_state,
                    flux_last_ai_enhanced_output_state
                ]
            )

            self.flux_bouton_stop.click(fn=self.stop_generation, inputs=None, outputs=None)
        return tab

    def _common_load_logic(self, selected_model_name, model_type_key, use_fp8=False, from_single_file=False):
        # Initial state: show loading message, disable generate button, hide login UI
        hf_login_group_update = gr.update(visible=False)
        hf_token_textbox_update = gr.update(value="") # Clear any previous token

        yield (
            gr.update(value=translate("flux_schnell_loading_model", self.module_translations)),
            gr.update(interactive=False),
            hf_login_group_update,
            hf_token_textbox_update,
        )

        try:
            success, message = self.model_manager.load_model(
                model_name=selected_model_name,
                vae_name="Auto",
                model_type=model_type_key,
                gradio_mode=True,
                from_single_file=from_single_file,
                use_fp8=use_fp8
            )
        except HuggingFaceAuthError as e:
            gr.Warning(translate("error_hf_auth_required", self.module_translations), 5.0)
            hf_login_group_update = gr.update(visible=True) # Show login UI
            yield (
                gr.update(value=translate("flux_schnell_model_not_loaded_auth_needed", self.module_translations)),
                gr.update(interactive=False),
                hf_login_group_update,
                hf_token_textbox_update,
            )
            return # Exit generator after yielding auth error

        except Exception as e:
            gr.Error(f"{translate('flux_schnell_error_loading_model', self.module_translations)}: {e}")
            yield (
                gr.update(value=f"{translate('flux_schnell_model_not_loaded', self.module_translations)}: {e}"),
                gr.update(interactive=False),
                hf_login_group_update,
                hf_token_textbox_update,
            )
            return # Exit generator after yielding general error

        # If successful
        if success:
            pipe = self.model_manager.get_current_pipe()
            if pipe and isinstance(pipe, (FluxPipeline, FluxImg2ImgPipeline)): # Check both types
                message += f" {translate('flux_schnell_model_config_applied', self.module_translations)}"
                gr.Info(translate('flux_schnell_model_config_applied', self.module_translations))
            yield (
                gr.update(value=message),
                gr.update(interactive=True),
                hf_login_group_update,
                hf_token_textbox_update,
            )
        else:
            # If load_model returned False for other reasons (not auth error)
            yield (
                gr.update(value=message),
                gr.update(interactive=False),
                hf_login_group_update,
                hf_token_textbox_update,
            )

    def load_default_text_to_image_model_ui(self):
        yield from self._common_load_logic(
            selected_model_name=FLUX_SCHNELL_MODEL_ID,
            model_type_key=FLUX_SCHNELL_MODEL_TYPE_KEY,
            use_fp8=False, # Default model does not use FP8
            from_single_file=False # Default model is from Hugging Face
        )

    def load_local_text_to_image_model_ui(self, selected_model_from_dropdown):
        is_single_file = selected_model_from_dropdown.endswith(".safetensors")
        if not is_single_file: # Prevent loading HF model with this button
            gr.Warning(translate("flux_schnell_select_local_model_warning", self.module_translations), 3.0) # New translation key
            # Yield current state to prevent UI from getting stuck
            yield (
                gr.update(value=translate("flux_schnell_model_not_loaded", self.module_translations)),
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(value=""),
            )
            return

        yield from self._common_load_logic(
            selected_model_name=selected_model_from_dropdown,
            model_type_key=FLUX_SCHNELL_MODEL_TYPE_KEY, # Local models are text-to-image
            use_fp8=False, # FP8 is disabled
            from_single_file=True # Always true for local models
        )

    def load_image_to_image_model_ui(self):
        yield from self._common_load_logic(
            selected_model_name=FLUX_SCHNELL_MODEL_ID, # Or a specific img2img model ID if different
            model_type_key=FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY,
            use_fp8=False, # Assuming img2img model doesn't use FP8 by default
            from_single_file=False # Assuming img2img model is from Hugging Face
        )

    # AJOUT: Nouvelle fonction pour le login et le rechargement
    def login_and_retry_load_flux_model_ui(self, hf_token): # MODIFIED: removed model/fp8/img2img specific args
        if not hf_token:
            gr.Warning(translate("hf_token_empty_warn", self.module_translations), 3.0)
            return (
                gr.update(value=translate("hf_token_empty_warn", self.module_translations)),
                gr.update(interactive=False),
                gr.update(visible=True), # Keep login UI visible
                gr.update(value=hf_token), # Keep token in textbox
            )

        gr.Info(translate("hf_logging_in", self.module_translations), 3.0)
        try:
            from huggingface_hub import login # Ensure login is imported here
            login(token=hf_token)
            gr.Info(translate("hf_login_success", self.module_translations), 3.0)
            # After successful login, we just return a general "ready to load" state.
            # The user is expected to click the desired load button again.
            return (
                gr.update(value=translate("flux_schnell_model_not_loaded", self.module_translations)), # Ready to load
                gr.update(interactive=False), # Keep generate button disabled until model is loaded
                gr.update(visible=False), # Hide login UI
                gr.update(value=""), # Clear token
            )
        except Exception as e:
            gr.Error(f"{translate('hf_login_failed', self.module_translations)}: {e}")
            return (
                gr.update(value=f"{translate('hf_login_failed', self.module_translations)}: {e}"),
                gr.update(interactive=False),
                gr.update(visible=True), # Keep login UI visible
                gr.update(value=hf_token), # Keep token in textbox
            )

    def update_prompt_from_image_flux(self, image_pil, use_image_flag, current_module_translations): # MODIFIÉ
        """Génère un prompt si l'image est fournie et le checkbox est coché pour FLUX."""
        if use_image_flag and image_pil is not None:
            task_for_florence = "<DETAILED_CAPTION>" 
            print(txt_color("[INFO]", "info"), translate("flux_schnell_generating_prompt_from_image", current_module_translations)) 
            generated_prompt = generate_prompt_from_image(image_pil, current_module_translations, task=task_for_florence)
            if generated_prompt.startswith(f"[{translate('erreur', current_module_translations).upper()}]"):
                gr.Warning(generated_prompt, duration=5.0)
                return gr.update() 
            else:
                return gr.update(value=generated_prompt) 
        elif not use_image_flag: 
            return gr.update()
        return gr.update()

    def flux_schnell_gen(
        self,
        prompt_libre,
        traduire_flag,
        selected_styles,
        num_images,
        use_img2img_checkbox_value,
        steps,
        resolution_str,
        guidance_scale,
        img2img_input_pil,
        img2img_strength,
        seed_input,
        original_user_prompt_for_cycle,
        prompt_is_currently_enhanced,
        enhancement_cycle_is_active,
        *loras_all_inputs
    ):
        module_translations = self.module_translations
        start_time_total = time.time()
        self.stop_event.clear()
        
        pipe = self.model_manager.get_current_pipe()
        is_img2img_mode = use_img2img_checkbox_value and img2img_input_pil is not None

        # 1. Vérifier si un modèle est chargé
        if pipe is None:
            gr.Warning(translate("flux_schnell_error_no_model", module_translations))
            yield [], "", gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return

        # 2. Vérifier si le type de modèle chargé correspond à l'opération demandée
        expected_pipe_class = FluxImg2ImgPipeline if is_img2img_mode else FluxPipeline
        if not isinstance(pipe, expected_pipe_class):
            mode_attendu = "Image-to-Image" if is_img2img_mode else "Text-to-Image"
            mode_charge = "Image-to-Image" if isinstance(pipe, FluxImg2ImgPipeline) else "Text-to-Image"
            error_msg = translate("flux_schnell_error_wrong_model_type", self.module_translations).format(mode_attendu=mode_attendu, mode_charge=mode_charge)
            gr.Error(error_msg)
            yield [], error_msg, gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return
        # --- FIN DE LA NOUVELLE LOGIQUE DE VÉRIFICATION ---

        # Préparer le message de progression initial (avant le chargement potentiel du modèle)
        width, height = 0, 0
        if is_img2img_mode:
            checker = ImageSDXLchecker(img2img_input_pil, self.global_translations, max_pixels=1024*1408) 
            processed_input_image = checker.redimensionner_image() 
            input_image_width, input_image_height = processed_input_image.size
            print(txt_color("[INFO]", "info"), f"Img2Img mode. Input image dimensions: {input_image_width}x{input_image_height}")
            width, height = input_image_width, input_image_height
        else: 
            try:
                width, height = map(int, resolution_str.split('x'))
            except ValueError:
                msg = f"Format de résolution invalide: {resolution_str}. Utilisation de 768x1280 par défaut."
                print(txt_color("[ERREUR]", "erreur"), msg)
                gr.Warning(msg, duration=4.0)
                width, height = 768, 1280

        initial_gallery = []
        initial_progress = create_progress_bar_html(0, int(steps), 0, f"{translate('preparation', module_translations)} ({width}x{height})")

        # Afficher la progression initiale (après chargement si nécessaire)
        btn_gen_off = gr.update(interactive=False)
        btn_stop_on = gr.update(interactive=True)
        lora_status_message_update = gr.update() 

        yield initial_gallery, initial_progress, btn_gen_off, btn_stop_on, lora_status_message_update
        
        # --- Logique d'utilisation du prompt (original ou amélioré) ---
        prompt_to_use_for_flux = prompt_libre # Par défaut, le texte de la textbox
        prompt_to_log_as_original = prompt_libre

        if enhancement_cycle_is_active or prompt_is_currently_enhanced:
            prompt_to_log_as_original = original_user_prompt_for_cycle
        # 'prompt_to_use_for_flux' (qui est 'prompt_libre' à ce stade) est déjà le prompt potentiellement amélioré.

        if not (prompt_to_use_for_flux and prompt_to_use_for_flux.strip()) and not selected_styles: # Vérifier avec le prompt à utiliser
            msg = translate("flux_schnell_error_no_prompt", module_translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return
        
        if prompt_to_use_for_flux and prompt_to_use_for_flux.strip():
            base_user_prompt = translate_prompt(prompt_to_use_for_flux, module_translations) if traduire_flag else prompt_to_use_for_flux
            if traduire_flag and base_user_prompt != prompt_to_use_for_flux:
                gr.Info(translate("prompt_traduit_pour_generation", self.global_translations), 2.0)
        else:
            base_user_prompt = "" # Si le prompt est vide mais qu'il y a des styles
        final_prompt_text_for_flux, _, style_names_used = styles_fusion(
            selected_styles,
            base_user_prompt,
            "", 
            self.styles, 
            module_translations,
        )
        num_lora_slots = len(self.flux_lora_checks)
        lora_ui_config = {
            'lora_checks': loras_all_inputs[:num_lora_slots],
            'lora_dropdowns': loras_all_inputs[num_lora_slots : 2*num_lora_slots],
            'lora_scales': loras_all_inputs[2*num_lora_slots : 3*num_lora_slots]
        }
        lora_message_from_manager = self.model_manager.apply_loras(lora_ui_config, gradio_mode=True)
        lora_status_message_update = gr.update(value=lora_message_from_manager)
        if lora_message_from_manager and translate("erreur", self.global_translations).lower() in lora_message_from_manager.lower():
            gr.Warning(lora_message_from_manager, duration=4.0)
        elif lora_message_from_manager:
            gr.Info(lora_message_from_manager, duration=3.0)

        progress_queue = queue.Queue()
        generated_images_gallery = []
        final_message_text = ""

        def pipeline_thread_target(pipe, kwargs, result_queue):
            try:
                result = pipe(**kwargs).images[0]
                result_queue.put(result)
            except Exception as e:
                result_queue.put(e)

        try:
            for i in range(int(num_images)):
                if self.stop_event.is_set():
                    final_message_text = translate("generation_arretee", module_translations)
                    print(txt_color("[INFO]", "info"), final_message_text)
                    gr.Info(final_message_text, 3.0)
                    break

                current_seed_val = random.randint(0, 2**32 - 1) if seed_input == -1 else int(seed_input) + i
                generator = torch.Generator(device=self.model_manager.device).manual_seed(current_seed_val)
                
                image_info_text = f"{translate('image', module_translations)} {i+1}/{num_images}"
                print(txt_color("[INFO]", "info"), f"{translate('flux_schnell_generation_start', module_translations)} ({image_info_text}), Seed: {current_seed_val}")

                result_queue = queue.Queue()
                
                # Vider la queue de progression avant de commencer
                while not progress_queue.empty():
                    try: progress_queue.get_nowait()
                    except queue.Empty: break

                callback_for_progress = create_inpainting_callback(
                    self.stop_event,
                    total_steps=int(steps),
                    translations=module_translations,
                    progress_queue=progress_queue
                )

                if use_img2img_checkbox_value and img2img_input_pil is None:
                    msg = translate("flux_schnell_error_no_image_for_img2img", module_translations)
                    print(txt_color("[ERREUR] ", "erreur"), msg)
                    gr.Warning(msg, duration=4.0)
                    yield generated_images_gallery, msg, gr.update(interactive=True), gr.update(interactive=False), lora_status_message_update
                    return

                pipeline_kwargs = {
                    "prompt": "",  
                    "prompt_2": final_prompt_text_for_flux,
                    "num_inference_steps": int(steps),
                    "guidance_scale": float(guidance_scale),
                    "generator": generator,
                    "max_sequence_length": 512,
                    "callback_on_step_end": callback_for_progress
                }

                if is_img2img_mode:
                    pipeline_kwargs.update({
                        "image": processed_input_image,
                        "strength": float(img2img_strength),
                        "width": width,
                        "height": height,
                    })
                else:
                    pipeline_kwargs.update({"width": width, "height": height})

                start_time_image = time.time()
                thread = threading.Thread(target=pipeline_thread_target, args=(pipe, pipeline_kwargs, result_queue))
                thread.start()

                while thread.is_alive():
                    try:
                        progress_step, total_steps_from_cb = progress_queue.get_nowait()
                        progress_percent = (progress_step / total_steps_from_cb) * 100
                        progress_text = f"{image_info_text} - {progress_step}/{total_steps_from_cb} {translate('etapes', module_translations)}"
                        current_progress_html = create_progress_bar_html(progress_step, total_steps_from_cb, progress_percent, progress_text)
                        yield generated_images_gallery, current_progress_html, btn_gen_off, btn_stop_on, lora_status_message_update
                    except queue.Empty:
                        time.sleep(0.1) # Attendre un peu si la queue est vide

                thread.join()
                
                try:
                    result = result_queue.get_nowait()
                    if isinstance(result, Exception):
                        raise result
                    result_image = result
                    temps_image_gen_sec = time.time() - start_time_image
                except (queue.Empty, Exception) as e_gen:
                    error_msg = f"{translate('flux_schnell_error_generation', module_translations)} ({image_info_text}): {e_gen}"
                    print(txt_color("[ERREUR]", "erreur"), error_msg)
                    traceback.print_exc()
                    final_message_text = f'<p style="color:red;">{error_msg}</p>'
                    gr.Error(error_msg)
                    current_progress_html = create_progress_bar_html(0, int(steps), 0, f"{image_info_text} - {translate('erreur', module_translations)}")
                    yield generated_images_gallery, current_progress_html, btn_gen_off, btn_stop_on, lora_status_message_update
                    continue

                if self.stop_event.is_set(): 
                    final_message_text = translate("generation_arretee_apres_image_courante", module_translations)
                    break

                generated_images_gallery.append(result_image)
                
                output_width, output_height = result_image.size 
                current_time_str = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"flux_schnell_{current_time_str}_img{i+1}_{output_width}x{output_height}_seed{current_seed_val}.{self.global_config['IMAGE_FORMAT'].lower()}"
                date_str_save = datetime.now().strftime("%Y_%m_%d")
                save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str_save)
                os.makedirs(save_dir, exist_ok=True)
                chemin_image = os.path.join(save_dir, output_filename)

                xmp_data = {
                    "Module": "FLUX.1-Schnell",
                    "Creator": self.global_config.get("AUTHOR", "CyberBill"),
                    "Model": self.model_manager.current_model_name,
                    "Steps": steps,
                    "GuidanceScale": guidance_scale,
                    "Size": f"{output_width}x{output_height}", 
                    "Seed": current_seed_val,
                    "GenerationTimeSeconds": f"{temps_image_gen_sec:.2f}",
                    "LLM_Enhanced": prompt_is_currently_enhanced,
                    "OriginalUserPrompt": prompt_to_log_as_original,
                    "FinalPromptForFlux": final_prompt_text_for_flux,
                    "TranslatedToEnglish": "Oui" if traduire_flag and base_user_prompt != prompt_to_use_for_flux else "Non",
                    "StylesUsed": ", ".join(style_names_used) if style_names_used else "None"
                }
                if is_img2img_mode:
                    xmp_data["InputImageType"] = "Image"
                    xmp_data["Strength"] = f"{img2img_strength:.2f}"
                    xmp_data["InputImageSize"] = f"{input_image_width}x{input_image_height}"
                else: 
                    xmp_data["MaxSequenceLength"] = 512

                xmp_data["Prompt"] = final_prompt_text_for_flux
                xmp_data["LoRAs"] = json.dumps(self.model_manager.loaded_loras if self.model_manager.loaded_loras else "Aucun")
                
                metadata_structure, prep_message = preparer_metadonnees_image(result_image, xmp_data, self.global_translations, chemin_image)
                print(txt_color("[INFO]", "info"), prep_message)
                enregistrer_image(result_image, chemin_image, self.global_translations, self.global_config["IMAGE_FORMAT"].upper(), metadata_to_save=metadata_structure)
                enregistrer_etiquettes_image_html(chemin_image, xmp_data, module_translations, is_last_image=(i == int(num_images) - 1))
                
                print(txt_color("[OK]", "ok"), f"{translate('image', module_translations)} {i+1}/{num_images} {translate('generer_en', module_translations)} {temps_image_gen_sec:.2f} sec")
                
                progress_text_info_done = f"{image_info_text} ({output_width}x{output_height})"
                current_progress_html = create_progress_bar_html(
                    int(steps), int(steps), 100, f"{progress_text_info_done} - {translate('termine', module_translations)}"
                )
                yield generated_images_gallery, current_progress_html, btn_gen_off, btn_stop_on, lora_status_message_update
            
            if not self.stop_event.is_set() and not final_message_text.startswith('<p style="color:red;">'):
                temps_total_final = f"{(time.time() - start_time_total):.2f}"
                if int(num_images) > 1:
                    final_message_text = translate("batch_complete", module_translations).format(num_images=num_images, time=temps_total_final)
                else:
                    final_message_text = translate("flux_schnell_generation_complete", module_translations).format(time=temps_total_final)
                print(txt_color("[OK]", "ok"), final_message_text)
                gr.Info(final_message_text, duration=3.0)
            elif not final_message_text: 
                 final_message_text = translate("generation_arretee", module_translations)
        finally:
            # --- NETTOYAGE EXPLICITE POUR LIBÉRER LA MÉMOIRE ---
            # Forcer le garbage collector à nettoyer les objets non référencés.
            gc.collect()
            # Vider le cache CUDA si un GPU est utilisé.
            if self.model_manager.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            yield generated_images_gallery, final_message_text, gr.update(interactive=True), gr.update(interactive=False), lora_status_message_update

    # --- AJOUT: Fonctions pour l'amélioration du prompt (adaptées de Pycnaptiq-AI.py) ---
    # --- MÉTHODES POUR L'AMÉLIORATION DU PROMPT (LLM) ---
    def on_flux_enhance_or_redo_button_click(self, current_prompt, original_user_prompt, cycle_active, model_path, translations):
        return llm_prompter_util.on_enhance_or_redo_button_click(current_prompt, original_user_prompt, cycle_active, model_path, translations)

    def on_flux_validate_prompt_button_click(self, current_prompt, translations):
        return llm_prompter_util.on_validate_prompt_button_click(current_prompt, translations)

    def handle_flux_text_input_change(self, current_prompt, last_ai_output, cycle_active, model_path, translations):
        return llm_prompter_util.handle_text_input_change(current_prompt, last_ai_output, cycle_active, model_path, translations)
    # --- FIN AJOUT ---
    # --- FIN AJOUT ---
