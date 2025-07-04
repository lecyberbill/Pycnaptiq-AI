# SD3-5-TURBO_mod.py
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

from Utils.utils import (
    txt_color,
    translate,
    enregistrer_image,
    preparer_metadonnees_image,
    enregistrer_etiquettes_image_html,
    create_progress_bar_html,
    styles_fusion,
)
from Utils.callback_diffuser import create_inpainting_callback
from Utils.model_manager import ModelManager, HuggingFaceAuthError
from core.translator import translate_prompt
from core.pipeline_executor import execute_pipeline_task_async
from Utils import llm_prompter_util # <-- AJOUT pour l'amélioration du prompt

MODULE_NAME = "SD3-5-TURBO"
SD3_5_TURBO_MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"
SD3_5_TURBO_MODEL_TYPE_KEY = "sd3_5_turbo"

module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

try:
    with open(module_json_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"[ERREUR] Chargement JSON pour {MODULE_NAME} a échoué: {e}")
    module_data = {"name": MODULE_NAME}

def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME)}")
    return SD3_5_TURBO_Module(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class SD3_5_TURBO_Module:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.gestionnaire = gestionnaire_instance
        self.stop_event = threading.Event()
        self.styles = self.load_styles()
        self.default_negative_prompt = self.global_config.get("NEGATIVE_PROMPT", "")
        self.module_translations = {}
        # --- AJOUT: Initialisation pour l'amélioration du prompt ---
        self.llm_prompter_model_path = self.global_config.get("LLM_PROMPTER_MODEL_PATH", "Qwen/Qwen3-0.6B")
        # --- AJOUT: LoRA ---
        self.available_loras = self.model_manager.list_loras(gradio_mode=True)
        self.has_loras = bool(self.available_loras) and \
                         translate("aucun_modele_trouve", self.global_translations) not in self.available_loras and \
                         translate("repertoire_not_found", self.global_translations) not in self.available_loras
        self.lora_choices_for_ui = self.available_loras if self.has_loras else [translate("aucun_lora_disponible", self.module_translations)]
        # --- FIN AJOUT ---
        
        # Utiliser les résolutions standard de la configuration globale
        self.allowed_resolutions = [f"{format['dimensions'].replace('*', 'x')}" for format in self.global_config.get("FORMATS", [])]
        if not self.allowed_resolutions:
            self.allowed_resolutions = ["1024x1024", "1152x896", "896x1152", "1216x832", "832x1216", "1344x768", "768x1344", "1536x640", "640x1536"]

    def load_styles(self):
        styles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "styles.json")
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                styles_data = json.load(f)
            for style in styles_data:
                style["name"] = translate(style["key"], self.global_translations)
            return styles_data
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"Erreur chargement styles.json pour {MODULE_NAME}: {e}")
            return []

    def stop_generation(self):
        self.stop_event.set()
        print(txt_color("[INFO]", "info"), translate("stop_requested", self.module_translations))

    def create_tab(self, module_translations):
        self.module_translations = module_translations

        with gr.Tab(translate("sd3_5_turbo_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('sd3_5_turbo_tab_title', self.module_translations)}")

            # --- AJOUT: États pour l'amélioration du prompt ---
            sd3_original_user_prompt_state = gr.State(value="")
            sd3_current_prompt_is_enhanced_state = gr.State(value=False)
            sd3_enhancement_cycle_active_state = gr.State(value=False)
            sd3_last_ai_enhanced_output_state = gr.State(value=None)
            # --- FIN AJOUT ---

            with gr.Row():
                with gr.Column(scale=2):
                    self.sd3_prompt = gr.Textbox(
                        label=translate("sd3_5_turbo_prompt_label", self.module_translations),
                        info=translate("sd3_5_turbo_prompt_info", self.module_translations),
                        placeholder=translate("sd3_5_turbo_prompt_placeholder", self.module_translations),
                        lines=3,
                    )
                    # --- AJOUT: Boutons d'amélioration du prompt ---
                    with gr.Row():
                        self.sd3_enhance_or_redo_button = gr.Button(
                            translate("ameliorer_prompt_ia_btn", self.module_translations),
                            interactive=True
                        )
                        self.sd3_validate_prompt_button = gr.Button(
                            translate("valider_prompt_btn", self.module_translations),
                            interactive=False,
                            visible=False
                        )
                    # --- FIN AJOUT ---
                    self.sd3_traduire_checkbox = gr.Checkbox(
                        label=translate("traduire_en_anglais", self.module_translations),
                        value=False,
                        info=translate("traduire_prompt_libre", self.module_translations),
                    )
                    self.sd3_style_dropdown = gr.Dropdown(
                        label=translate("selectionner_styles", self.module_translations),
                        choices=[style["name"] for style in self.styles if style["name"] != translate("Aucun_style", self.global_translations)],
                        value=[], multiselect=True,
                    )
                    with gr.Row():
                        self.sd3_resolution_dropdown = gr.Dropdown(
                            label=translate("resolution_label", self.global_translations),
                            choices=self.allowed_resolutions,
                            value="1024x1024",
                            info=translate("resolution_info_sd3_5_turbo", self.module_translations),
                        )
                        self.sd3_steps_slider = gr.Slider(
                            minimum=1, maximum=20, value=4, step=1,
                            label=translate("sd3_5_turbo_steps_label", self.module_translations)
                        )
                        self.sd3_guidance_scale_slider = gr.Slider(
                            minimum=0.0, maximum=10.0, value=0.0, step=0.1,
                            label=translate("sd3_5_turbo_guidance_scale_label", self.module_translations)
                        )
                    with gr.Row():
                        self.sd3_seed_input = gr.Number(
                            label=translate("seed_label", self.global_translations), value=-1,
                            info=translate("seed_info_neg_one_random", self.global_translations)
                        )
                        self.sd3_num_images_slider = gr.Slider(
                            minimum=1, maximum=20, value=1, step=1,
                            label=translate("nombre_images", self.module_translations),
                        )
                    # --- AJOUT: LoRA Accordion ---
                    with gr.Accordion(translate("lora_section_title", self.module_translations), open=False) as lora_accordion_sd3:
                        self.sd3_lora_checks = []
                        self.sd3_lora_dropdowns = []
                        self.sd3_lora_scales = []
                        for i in range(1, 3): # Start with 2 LoRA slots
                            with gr.Group():
                                lora_check = gr.Checkbox(label=f"LoRA {i}", value=False)
                                lora_dropdown = gr.Dropdown(
                                    choices=self.lora_choices_for_ui,
                                    label=translate("selectionner_lora", self.global_translations),
                                    interactive=self.has_loras
                                )
                                lora_scale_slider = gr.Slider(0, 1, value=0.8, label=translate("poids_lora", self.global_translations))
                                self.sd3_lora_checks.append(lora_check)
                                self.sd3_lora_dropdowns.append(lora_dropdown)
                                self.sd3_lora_scales.append(lora_scale_slider)

                                lora_check.change(
                                    fn=lambda chk, has_loras_flag: gr.update(interactive=chk and has_loras_flag),
                                    inputs=[lora_check, gr.State(self.has_loras)],
                                    outputs=[lora_dropdown]
                                )
                        self.sd3_lora_message = gr.Textbox(label=translate("message_lora", self.global_translations), interactive=False)
                        self.sd3_refresh_lora_button = gr.Button(
                            translate("refresh_lora_list", self.module_translations),
                            variant="secondary")
                    # --- FIN AJOUT ---

                with gr.Column(scale=1):
                    self.sd3_message_chargement = gr.Textbox(
                        label=translate("sd3_5_turbo_model_status", self.module_translations),
                        value=translate("sd3_5_turbo_model_not_loaded", self.module_translations),
                        interactive=False,
                    )
                    self.sd3_bouton_charger = gr.Button(translate("sd3_5_turbo_load_button", self.module_translations))
                    self.sd3_result_output = gr.Gallery(label=translate("output_image", self.module_translations))
                    with gr.Row():
                        self.sd3_bouton_gen = gr.Button(
                            value=translate("sd3_5_turbo_generate_button", self.module_translations),
                            interactive=False, variant="primary"
                        )
                        self.sd3_bouton_stop = gr.Button(
                            translate("arreter", self.module_translations),
                            interactive=False, variant="stop",
                        )
                    self.sd3_progress_html = gr.HTML()

            self.sd3_bouton_charger.click(
                fn=self.load_sd3_model_ui,
                inputs=None,
                outputs=[self.sd3_message_chargement, self.sd3_bouton_gen],
            )

            sd3_gen_inputs = [
                self.sd3_prompt, self.sd3_traduire_checkbox, self.sd3_style_dropdown,
                self.sd3_num_images_slider, self.sd3_steps_slider, self.sd3_resolution_dropdown,
                self.sd3_guidance_scale_slider, self.sd3_seed_input,
                # --- AJOUT: États d'amélioration du prompt ---
                sd3_original_user_prompt_state,
                sd3_current_prompt_is_enhanced_state,
                sd3_enhancement_cycle_active_state,
            ]
            # Add LoRA inputs
            for chk in self.sd3_lora_checks: sd3_gen_inputs.append(chk)
            for dd in self.sd3_lora_dropdowns: sd3_gen_inputs.append(dd)
            for sc in self.sd3_lora_scales: sd3_gen_inputs.append(sc)

            self.sd3_bouton_gen.click(
                fn=self.sd3_5_turbo_gen,
                inputs=sd3_gen_inputs,
                outputs=[self.sd3_result_output, self.sd3_progress_html, self.sd3_bouton_gen, self.sd3_bouton_stop, self.sd3_lora_message],
            )

            # --- AJOUT: Liaisons pour l'amélioration du prompt ---
            self.sd3_enhance_or_redo_button.click(fn=self.on_sd3_enhance_or_redo_button_click, inputs=[self.sd3_prompt, sd3_original_user_prompt_state, sd3_enhancement_cycle_active_state, gr.State(self.llm_prompter_model_path), gr.State(self.module_translations)], outputs=[self.sd3_prompt, self.sd3_enhance_or_redo_button, self.sd3_validate_prompt_button, sd3_original_user_prompt_state, sd3_current_prompt_is_enhanced_state, sd3_enhancement_cycle_active_state, sd3_last_ai_enhanced_output_state])
            self.sd3_validate_prompt_button.click(fn=self.on_sd3_validate_prompt_button_click, inputs=[self.sd3_prompt, gr.State(self.module_translations)], outputs=[self.sd3_enhance_or_redo_button, self.sd3_validate_prompt_button, sd3_original_user_prompt_state, sd3_current_prompt_is_enhanced_state, sd3_enhancement_cycle_active_state, sd3_last_ai_enhanced_output_state])
            self.sd3_prompt.input(fn=self.handle_sd3_text_input_change, inputs=[self.sd3_prompt, sd3_last_ai_enhanced_output_state, sd3_enhancement_cycle_active_state, gr.State(self.llm_prompter_model_path), gr.State(self.module_translations)], outputs=[self.sd3_enhance_or_redo_button, self.sd3_validate_prompt_button, sd3_original_user_prompt_state, sd3_current_prompt_is_enhanced_state, sd3_enhancement_cycle_active_state, sd3_last_ai_enhanced_output_state])
            self.sd3_prompt.submit(fn=self.handle_sd3_text_input_change, inputs=[self.sd3_prompt, sd3_last_ai_enhanced_output_state, sd3_enhancement_cycle_active_state, gr.State(self.llm_prompter_model_path), gr.State(self.module_translations)], outputs=[self.sd3_enhance_or_redo_button, self.sd3_validate_prompt_button, sd3_original_user_prompt_state, sd3_current_prompt_is_enhanced_state, sd3_enhancement_cycle_active_state, sd3_last_ai_enhanced_output_state])
            # --- FIN AJOUT ---
            
            # --- AJOUT: LoRA Refresh Button Click ---
            self.sd3_refresh_lora_button.click(
                fn=self.refresh_lora_list,
                inputs=None,
                outputs=self.sd3_lora_dropdowns
            )

            self.sd3_bouton_stop.click(fn=self.stop_generation, inputs=None, outputs=None)
        return tab

    def load_sd3_model_ui(self):
        yield gr.update(value=translate("sd3_5_turbo_loading_model", self.module_translations)), gr.update(interactive=False)
        
        success, message = self.model_manager.load_model(
            model_name=SD3_5_TURBO_MODEL_ID,
            model_type=SD3_5_TURBO_MODEL_TYPE_KEY,
            gradio_mode=True,
        )

        if success:
            message += f" {translate('sd3_5_turbo_model_config_applied', self.module_translations)}"
            gr.Info(translate('sd3_5_turbo_model_config_applied', self.module_translations))
            yield gr.update(value=message), gr.update(interactive=True)
        else:
            yield gr.update(value=message), gr.update(interactive=False)

    def sd3_5_turbo_gen(
        self, prompt_libre, traduire_flag, selected_styles, num_images,
        steps, resolution_str, guidance_scale, seed_input,
        # --- AJOUT: Paramètres d'amélioration du prompt ---
        original_user_prompt_for_cycle,
        prompt_is_currently_enhanced,
        enhancement_cycle_is_active,
        *loras_all_inputs,
        # --- FIN AJOUT ---
    ):
        start_time_total = time.time()
        self.stop_event.clear()

        try:
            width, height = map(int, resolution_str.split('x'))
        except ValueError:
            gr.Warning(f"Format de résolution invalide: {resolution_str}. Utilisation de 1024x1024.", 4.0)
            width, height = 1024, 1024

        initial_progress = create_progress_bar_html(0, int(steps), 0, translate("preparation", self.module_translations))
        yield [], initial_progress, gr.update(interactive=False), gr.update(interactive=True), gr.update()

        pipe = self.model_manager.get_current_pipe()
        if pipe is None or self.model_manager.current_model_type != SD3_5_TURBO_MODEL_TYPE_KEY:
            gr.Warning(translate("sd3_5_turbo_error_no_model", self.module_translations), 4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return

        # --- Logique d'utilisation du prompt (original ou amélioré) ---
        prompt_to_use_for_sd3 = prompt_libre
        prompt_to_log_as_original = prompt_libre
        if enhancement_cycle_is_active or prompt_is_currently_enhanced:
            prompt_to_log_as_original = original_user_prompt_for_cycle
        # --- FIN Logique ---

        if not (prompt_to_use_for_sd3 and prompt_to_use_for_sd3.strip()) and not selected_styles:
            gr.Warning(translate("sd3_5_turbo_error_no_prompt", self.module_translations), 4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return
        
        base_user_prompt = translate_prompt(prompt_to_use_for_sd3, self.module_translations) if traduire_flag else prompt_to_use_for_sd3
        final_prompt_text, _, style_names_used = styles_fusion(
            selected_styles, base_user_prompt, "", self.styles, self.module_translations
        )

        # --- AJOUT: LoRA Application ---
        num_lora_slots = len(self.sd3_lora_checks)
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
        

        generated_images_gallery = []
        for i in range(int(num_images)):
            if self.stop_event.is_set():
                break

            current_seed = random.randint(0, 2**32 - 1) if int(seed_input) == -1 else int(seed_input) + i
            image_info_text = f"{translate('image', self.module_translations)} {i+1}/{num_images}"
            print(txt_color("[INFO]", "info"), f"{translate('sd3_5_turbo_generation_start', self.module_translations)} ({image_info_text}), Seed: {current_seed}")

            # --- MODIFICATION: Séparation des arguments pour l'exécuteur et pour le pipeline ---
            progress_queue = queue.Queue()

            # Arguments pour la fonction execute_pipeline_task_async
            executor_args = {
                "pipe": pipe,
                "num_inference_steps": int(steps),
                "guidance_scale": float(guidance_scale),
                "seed": current_seed,
                "width": width,
                "height": height,
                "device": self.model_manager.device,
                "stop_event": self.stop_event,
                "translations": self.module_translations,
                "progress_queue": progress_queue,
                "preview_queue": None, # SD3.5 Turbo ne supporte pas les aperçus de latents
            }

            # Arguments spécifiques au pipeline SD3.5 (passés via **kwargs)
            sd3_specific_kwargs = {
                "prompt": "",  # Laisser vide pour le premier encodeur CLIP
                "prompt_2": "", # Laisser vide pour le deuxième encodeur CLIP
                "prompt_3": final_prompt_text, # Passer le prompt complet à l'encodeur T5-XXL
                "max_sequence_length": 512,
            }

            thread, result_container = execute_pipeline_task_async(**executor_args, **sd3_specific_kwargs)

            # Boucle de mise à jour de la progression
            last_progress_html = ""
            while thread.is_alive() or not progress_queue.empty():
                if self.stop_event.is_set():
                    break

                current_step_prog, total_steps_prog = None, int(steps)
                while not progress_queue.empty():
                    try:
                        current_step_prog, total_steps_prog = progress_queue.get_nowait()
                    except queue.Empty:
                        break

                if current_step_prog is not None:
                    progress_percent = int((current_step_prog / total_steps_prog) * 100)
                    step_info_text = f"Step {current_step_prog}/{total_steps_prog}"
                    new_progress_html = create_progress_bar_html(
                        current_step=current_step_prog, total_steps=total_steps_prog,
                        progress_percent=progress_percent, text_info=f"{image_info_text} - {step_info_text}"
                    )
                    yield generated_images_gallery, new_progress_html, gr.update(interactive=False), gr.update(interactive=True), lora_status_message_update
                    last_progress_html = new_progress_html

                time.sleep(0.05)

            thread.join()

            if result_container["status"] == "success" and result_container["final"]:
                result_image = result_container["final"]
                generated_images_gallery.append(result_image)
                
                current_time_str = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"sd3_5_turbo_{current_time_str}_img{i+1}_{width}x{height}_seed{current_seed}.{self.global_config['IMAGE_FORMAT'].lower()}"
                date_str_save = datetime.now().strftime("%Y_%m_%d")
                save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str_save)
                os.makedirs(save_dir, exist_ok=True)
                chemin_image = os.path.join(save_dir, output_filename)

                xmp_data = {
                    "Module": "SD3.5 Turbo", "Creator": self.global_config.get("AUTHOR", "CyberBill"),
                    "Model": SD3_5_TURBO_MODEL_ID, "Steps": steps, "GuidanceScale": guidance_scale,
                    "Styles": ", ".join(style_names_used) if style_names_used else "None", "Size": f"{width}x{height}", "Seed": current_seed,
                    # --- AJOUT: Métadonnées d'amélioration ---
                    "LLM_Enhanced": prompt_is_currently_enhanced,
                    "OriginalUserPrompt": prompt_to_log_as_original,
                    "FinalPrompt": final_prompt_text,
                    "LoRAs": json.dumps(self.model_manager.loaded_loras if self.model_manager.loaded_loras else "Aucun"),
                    # --- FIN AJOUT ---
                }
                metadata_structure, prep_message = preparer_metadonnees_image(result_image, xmp_data, self.global_translations, chemin_image)
                print(txt_color("[INFO]", "info"), prep_message)
                enregistrer_image(result_image, chemin_image, self.global_translations, self.global_config["IMAGE_FORMAT"].upper(), metadata_to_save=metadata_structure)
                enregistrer_etiquettes_image_html(chemin_image, xmp_data, self.module_translations, is_last_image=(i == int(num_images) - 1))

                final_progress_html = create_progress_bar_html(int(steps), int(steps), 100, f"{image_info_text} - {translate('termine', self.module_translations)}")
                yield generated_images_gallery, final_progress_html, gr.update(interactive=False), gr.update(interactive=True), lora_status_message_update
            else:
                error_msg = result_container.get("error", "Erreur inconnue")
                print(txt_color("[ERREUR]", "erreur"), f"{translate('sd3_5_turbo_error_generation', self.module_translations)}: {error_msg}")
                gr.Error(f"{translate('sd3_5_turbo_error_generation', self.module_translations)}: {error_msg}")
                break

        final_message = ""
        if self.stop_event.is_set():
            final_message = translate("generation_arretee", self.module_translations)
        else:
            final_message = translate("sd3_5_turbo_generation_complete", self.module_translations).format(time=f"{(time.time() - start_time_total):.2f}")
        
        yield generated_images_gallery, final_message, gr.update(interactive=True), gr.update(interactive=False), lora_status_message_update

    def refresh_lora_list(self):
        """Rafraîchit la liste des LoRAs disponibles et met à jour les dropdowns."""
        print(txt_color("[INFO]", "info"), translate("refreshing_lora_list", self.module_translations))
        self.available_loras = self.model_manager.list_loras(gradio_mode=True)
        self.has_loras = bool(self.available_loras) and translate("aucun_modele_trouve", self.global_translations) not in self.available_loras and translate("repertoire_not_found", self.global_translations) not in self.available_loras
        new_choices = self.available_loras if self.has_loras else [translate("aucun_lora_disponible", self.module_translations)]
        gr.Info(translate("lora_list_refreshed", self.module_translations).format(count=len(self.available_loras) if self.has_loras else 0))
        return [gr.update(choices=new_choices, interactive=self.has_loras) for _ in self.sd3_lora_dropdowns]

    # --- AJOUT: Fonctions pour l'amélioration du prompt (adaptées de FluxSchnell_mod.py) ---
    def on_sd3_enhance_or_redo_button_click(self, current_text_in_box, original_prompt_for_cycle, cycle_is_active, llm_model_path, current_translations):
        prompt_to_enhance_this_time = ""
        new_original_prompt_for_cycle = original_prompt_for_cycle

        if not current_text_in_box.strip() and not cycle_is_active:
            gr.Warning(translate("llm_enhancement_no_prompt", current_translations), 3.0)
            return (current_text_in_box, gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=True), gr.update(visible=False), gr.update(value=""), gr.update(value=False), gr.update(value=False), gr.update(value=None))

        if llm_prompter_util.llm_model_prompter is None or llm_prompter_util.llm_tokenizer_prompter is None:
            gr.Info(translate("llm_prompter_loading_on_demand", current_translations), 3.0)
            if not llm_prompter_util.init_llm_prompter(llm_model_path, current_translations):
                gr.Warning(translate("llm_prompter_load_failed_on_demand", current_translations).format(model_path=llm_model_path), 5.0)
                return (current_text_in_box, gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=True), gr.update(visible=False), gr.update(value=original_prompt_for_cycle), gr.update(value=False), gr.update(value=False), gr.update(value=None))
            else:
                gr.Info(translate("llm_prompter_loaded_successfully", current_translations), 3.0)

        if not cycle_is_active:
            prompt_to_enhance_this_time = current_text_in_box
            new_original_prompt_for_cycle = current_text_in_box
        else:
            prompt_to_enhance_this_time = original_prompt_for_cycle

        gr.Info(translate("llm_prompt_enhancing_in_progress", current_translations), 2.0)
        enhanced_prompt_candidate = llm_prompter_util.generate_enhanced_prompt(prompt_to_enhance_this_time, llm_model_path, translations=current_translations)

        if enhanced_prompt_candidate and enhanced_prompt_candidate.strip() and enhanced_prompt_candidate.strip().lower() != prompt_to_enhance_this_time.strip().lower():
            gr.Info(translate("prompt_enrichi_applique", current_translations), 2.0)
            return (enhanced_prompt_candidate, gr.update(value=translate("refaire_amelioration_btn", current_translations), interactive=True), gr.update(visible=True, interactive=True), gr.update(value=new_original_prompt_for_cycle), gr.update(value=True), gr.update(value=True), gr.update(value=enhanced_prompt_candidate))
        else:
            gr.Warning(translate("llm_prompt_enhancement_failed_or_same", current_translations), 3.0)
            btn_text = translate("refaire_amelioration_btn", current_translations) if cycle_is_active else translate("ameliorer_prompt_ia_btn", current_translations)
            val_visible = cycle_is_active
            return (current_text_in_box, gr.update(value=btn_text, interactive=True), gr.update(visible=val_visible, interactive=val_visible), gr.update(value=original_prompt_for_cycle), gr.update(value=cycle_is_active), gr.update(value=cycle_is_active), gr.update(value=current_text_in_box if cycle_is_active else None))

    def on_sd3_validate_prompt_button_click(self, validated_prompt_text, current_translations):
        gr.Info(translate("prompt_amelioration_validee", current_translations), 2.0)
        return (gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=True), gr.update(visible=False), gr.update(value=validated_prompt_text), gr.update(value=True), gr.update(value=False), gr.update(value=None))

    def handle_sd3_text_input_change(self, text_value, last_ai_output_val, is_cycle_active_val, llm_model_path, current_translations):
        enhance_button_interactive = bool(text_value.strip())
        if not text_value:
            if llm_prompter_util.llm_model_prompter is not None or llm_prompter_util.llm_tokenizer_prompter is not None:
                llm_prompter_util.unload_llm_prompter(current_translations)
                gr.Info(translate("llm_prompter_unloaded_due_to_empty_prompt", current_translations), 3.0)
            return (gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=enhance_button_interactive), gr.update(visible=False), gr.update(value=""), gr.update(value=False), gr.update(value=False), gr.update(value=None))
        else:
            if is_cycle_active_val and text_value == last_ai_output_val:
                return (gr.update(value=translate("refaire_amelioration_btn", current_translations), interactive=True), gr.update(visible=True, interactive=True), gr.update(), gr.update(value=True), gr.update(value=True), gr.update())
            else:
                if is_cycle_active_val: gr.Info(translate("prompt_modifie_reinitialisation_amelioration", current_translations), 2.0)
                return (gr.update(value=translate("ameliorer_prompt_ia_btn", current_translations), interactive=enhance_button_interactive), gr.update(visible=False), gr.update(value=text_value), gr.update(value=False), gr.update(value=False), gr.update(value=None))
    # --- FIN AJOUT ---
