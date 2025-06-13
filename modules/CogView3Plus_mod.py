# CogView3Plus_mod.py
import os
import json
import queue
import threading
import time
import gc
import traceback
from datetime import datetime

import gradio as gr
import torch
from PIL import Image
from diffusers import CogView3PlusPipeline # Import spécifique

from Utils.utils import (
    txt_color,
    translate,
    enregistrer_image,
    preparer_metadonnees_image,
    enregistrer_etiquettes_image_html,
    styles_fusion,
    create_progress_bar_html,
)
from Utils.callback_diffuser import create_inpainting_callback
from Utils.model_manager import ModelManager, COGVIEW3PLUS_MODEL_ID, COGVIEW3PLUS_MODEL_TYPE_KEY # Import constants
from core.translator import translate_prompt
from core.pipeline_executor import execute_pipeline_task_async
from core.image_prompter import generate_prompt_from_image

# --- Configuration et Constantes ---
MODULE_NAME = "cogview3plus"
# COGVIEW3PLUS_MODEL_ID et COGVIEW3PLUS_MODEL_TYPE_KEY sont importés depuis ModelManager

# JSON associé à ce module
module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

try:
    with open(module_json_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)
except FileNotFoundError:
    print(f"[ERREUR] Fichier JSON du module {MODULE_NAME} introuvable.")
    module_data = {"name": MODULE_NAME} # Fallback
except json.JSONDecodeError:
    print(f"[ERREUR] Erreur de décodage JSON pour le module {MODULE_NAME}.")
    module_data = {"name": MODULE_NAME} # Fallback

def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
    """Initialise le module CogView3Plus."""
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME)}")
    return CogView3PlusModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class CogView3PlusModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
        """Initialise la classe CogView3PlusModule."""
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.styles = self.load_styles()
        self.gestionnaire = gestionnaire_instance
        self.default_negative_prompt = self.global_config.get("NEGATIVE_PROMPT", "")
        self.stop_event = threading.Event()
        self.module_translations = {}
        self.allowed_resolutions = self.global_config.get("COGVIEW3PLUS_ALLOWED_RESOLUTIONS")

        if not self.allowed_resolutions:
            formats_config = self.global_config.get("FORMATS", [])
            if formats_config and isinstance(formats_config, list):
                parsed_resolutions = []
                for item in formats_config:
                    if isinstance(item, dict):
                        dimensions_str = item.get("dimensions")
                        if dimensions_str and isinstance(dimensions_str, str):
                            processed_res = dimensions_str.replace('*', 'x')
                            if 'x' in processed_res.lower():
                                try:
                                    w_str, h_str = processed_res.split('x')
                                    int(w_str)
                                    int(h_str)
                                    parsed_resolutions.append(processed_res)
                                except ValueError:
                                    # Utiliser self.module_translations si disponible, sinon global_translations
                                    active_translations = self.module_translations if self.module_translations else self.global_translations
                                    print(txt_color("[WARNING]", "warning"), f"{translate('format_dimension_ignore_conversion', active_translations).format(dimensions_str=dimensions_str, processed_res=processed_res)}")
                if parsed_resolutions:
                    self.allowed_resolutions = parsed_resolutions

        if not self.allowed_resolutions:
             self.allowed_resolutions = ["1024x1024"]
             
    def load_styles(self):
        """Charge les styles depuis styles.json."""
        styles_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "styles.json")
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                styles_data = json.load(f)
            for style in styles_data:
                style["name"] = translate(style["key"], self.global_translations)
            return styles_data
        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), f"Erreur chargement styles.json: {e}")
            return []

    def stop_generation(self):
        """Active l'événement pour arrêter la génération en cours."""
        self.stop_event.set()
        print(txt_color("[INFO]", "info"), translate("stop_requested", self.module_translations))

    def create_tab(self, module_translations):
        """Crée l'onglet Gradio pour ce module."""
        self.module_translations = module_translations

        with gr.Tab(translate("cogview3plus_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('cogview3plus_tab_title', self.module_translations)}")
            with gr.Row():
                with gr.Column(scale=2):
                    self.cogview3plus_prompt = gr.Textbox(
                        label=translate("cogview3plus_prompt_label", self.module_translations),
                        info=translate("cogview3plus_prompt_info", self.module_translations),
                        placeholder=translate("cogview3plus_prompt_placeholder", self.module_translations),
                        lines=3,
                    )
                    self.cogview3plus_traduire_checkbox = gr.Checkbox(
                        label=translate("traduire_en_anglais", self.module_translations),
                        value=False,
                        info=translate("traduire_prompt_libre", self.module_translations),
                    )
                    self.cogview3plus_style_dropdown = gr.Dropdown(
                        label=translate("selectionner_styles", self.module_translations),
                        choices=[style["name"] for style in self.styles],
                        value=[],
                        multiselect=True,
                        info=translate("selectionner_styles_info", self.module_translations),
                    )
                    self.cogview3plus_use_image_prompt_checkbox = gr.Checkbox(
                        label=translate("generer_prompt_image", self.global_translations),
                        value=False
                    )
                    self.cogview3plus_image_input_for_prompt = gr.Image(label=translate("telechargez_image", self.global_translations), type="pil", visible=False)
                    
                    with gr.Row():
                        self.cogview3plus_resolution_dropdown = gr.Dropdown(
                            label=translate("resolution_label", self.global_translations), 
                            choices=self.allowed_resolutions,
                            value=self.allowed_resolutions[0] if self.allowed_resolutions else "1024x1024",
                            info=translate("resolution_info_cogview3plus", self.module_translations) 
                        )
                        self.cogview3plus_steps_slider = gr.Slider(
                            minimum=1, maximum=100, value=50, step=1, # Default 50
                            label=translate("cogview3plus_steps_label", self.module_translations)
                        )
                        self.cogview3plus_guidance_scale_slider = gr.Slider(
                            minimum=1.0, maximum=15.0, value=7.0, step=0.1, # Default 7.0
                            label=translate("cogview3plus_guidance_scale_label", self.module_translations)
                        )
                    with gr.Row():
                         self.cogview3plus_num_images_slider = gr.Slider(
                            minimum=1, maximum=4, value=1, step=1, # Limité pour CogView3Plus
                            label=translate("nombre_images", self.module_translations),
                            interactive=True
                        )
                    with gr.Row():
                        self.cogview3plus_bouton_gen = gr.Button(
                            value=translate("cogview3plus_generate_button", self.module_translations), interactive=False
                        )
                        self.cogview3plus_bouton_stop = gr.Button(
                            translate("arreter", self.module_translations),
                            interactive=False, variant="stop",
                        )
                    self.cogview3plus_progress_html = gr.HTML()

                with gr.Column(scale=1):
                    self.cogview3plus_message_chargement = gr.Textbox(
                        label=translate("cogview3plus_model_status", self.module_translations),
                        value=translate("cogview3plus_model_not_loaded", self.module_translations),
                        interactive=False,
                    )
                    self.cogview3plus_bouton_charger = gr.Button(
                        translate("cogview3plus_load_button", self.module_translations)
                    )
                    self.cogview3plus_result_output = gr.Gallery(
                        label=translate("output_image", self.module_translations),
                    )

            self.cogview3plus_use_image_prompt_checkbox.change(
                fn=lambda use_image: gr.update(visible=use_image),
                inputs=self.cogview3plus_use_image_prompt_checkbox,
                outputs=self.cogview3plus_image_input_for_prompt
            )
            self.cogview3plus_image_input_for_prompt.change(
                fn=self.update_prompt_from_image,
                inputs=[self.cogview3plus_image_input_for_prompt, self.cogview3plus_use_image_prompt_checkbox, gr.State(self.global_translations)],
                outputs=self.cogview3plus_prompt
            )
            self.cogview3plus_bouton_charger.click(
                fn=self.load_cogview3plus_model_ui,
                inputs=None,
                outputs=[self.cogview3plus_message_chargement, self.cogview3plus_bouton_gen],
            )
            self.cogview3plus_bouton_gen.click(
                fn=self.cogview3plus_gen,
                inputs=[
                    self.cogview3plus_prompt,
                    self.cogview3plus_traduire_checkbox,
                    self.cogview3plus_style_dropdown,
                    self.cogview3plus_num_images_slider,
                    self.cogview3plus_steps_slider,
                    self.cogview3plus_resolution_dropdown,
                    self.cogview3plus_guidance_scale_slider,
                ],
                outputs=[
                    self.cogview3plus_result_output,
                    self.cogview3plus_progress_html,
                    self.cogview3plus_bouton_gen,
                    self.cogview3plus_bouton_stop,
                ],
            )
            self.cogview3plus_bouton_stop.click(fn=self.stop_generation, inputs=None, outputs=None)
        return tab

    def load_cogview3plus_model_ui(self):
        """Wrapper UI pour charger le modèle CogView3Plus."""
        yield gr.update(value=translate("cogview3plus_loading_model", self.module_translations)), gr.update(interactive=False)

        success, message = self.model_manager.load_model(
            model_name=COGVIEW3PLUS_MODEL_ID,
            vae_name="Auto", 
            model_type=COGVIEW3PLUS_MODEL_TYPE_KEY,
            gradio_mode=True,
        )

        if success:
            pipe = self.model_manager.get_current_pipe()
            if pipe and isinstance(pipe, CogView3PlusPipeline):
                # Les configurations (offload, slicing, tiling) sont gérées par ModelManager
                # lors du chargement pour COGVIEW3PLUS_MODEL_TYPE_KEY
                message += f" {translate('cogview3plus_model_config_applied', self.module_translations)}"
                gr.Info(translate('cogview3plus_model_config_applied', self.module_translations))
            yield gr.update(value=message), gr.update(interactive=True)
        else:
            yield gr.update(value=message), gr.update(interactive=False)

    def update_prompt_from_image(self, image_pil, use_image_flag, global_translations):
        """Génère un prompt si l'image est fournie et le checkbox est coché."""
        if use_image_flag and image_pil is not None:
            print(txt_color("[INFO]", "info"), translate("cogview3plus_generating_prompt_from_image", self.module_translations))
            generated_prompt = generate_prompt_from_image(image_pil, self.module_translations)
            if generated_prompt.startswith(f"[{translate('erreur', self.module_translations).upper()}]"):
                gr.Warning(generated_prompt, duration=5.0)
                return gr.update()
            else:
                return gr.update(value=generated_prompt)
        elif not use_image_flag:
            return gr.update()
        return gr.update()

    def cogview3plus_gen(
        self,
        prompt_libre,
        traduire,
        selected_styles,
        num_images,
        steps,
        resolution_str,
        guidance_scale
    ):
        """Génère une ou plusieurs images en utilisant le modèle CogView3Plus chargé."""
        module_translations = self.module_translations
        start_time_total = time.time()
        self.stop_event.clear()

        try:
            width, height = map(int, resolution_str.split('x'))
        except ValueError:
            msg = f"Format de résolution invalide: {resolution_str}. Utilisation de 1024x1024 par défaut."
            print(txt_color("[ERREUR]", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            width, height = 1024, 1024

        initial_gallery = []
        initial_progress = create_progress_bar_html(0, int(steps), 0, f"{translate('preparation', module_translations)} ({width}x{height})")
        btn_gen_off = gr.update(interactive=False)
        btn_stop_on = gr.update(interactive=True)

        yield initial_gallery, initial_progress, btn_gen_off, btn_stop_on

        pipe = self.model_manager.get_current_pipe()
        if pipe is None or self.model_manager.current_model_type != COGVIEW3PLUS_MODEL_TYPE_KEY:
            msg = translate("cogview3plus_error_no_model", module_translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False)
            return

        if not isinstance(pipe, CogView3PlusPipeline):
            msg = f"Erreur: Le pipe chargé n'est pas une instance de CogView3PlusPipeline. Type: {type(pipe)}"
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Error(msg)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False)
            return

        if prompt_libre and prompt_libre.strip():
            base_user_prompt = translate_prompt(prompt_libre, module_translations) if traduire else prompt_libre
        else:
            msg = translate("cogview3plus_error_no_prompt", module_translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False)
            return

        final_prompt_text, _, style_names_used = styles_fusion(
            selected_styles,
            base_user_prompt,
            "", 
            self.styles,
            module_translations,
        )

        progress_queue = queue.Queue()
        base_seed = int(time.time())
        generated_images_gallery = []
        final_message = ""

        for i in range(int(num_images)):
            if self.stop_event.is_set():
                final_message = translate("generation_arretee", module_translations)
                print(txt_color("[INFO]", "info"), final_message)
                gr.Info(final_message, 3.0)
                break

            current_seed = base_seed + i
            image_info_text = f"{translate('image', module_translations)} {i+1}/{num_images}"
            print(txt_color("[INFO]", "info"), f"{translate('cogview3plus_generation_start', module_translations)} ({image_info_text})")

            while not progress_queue.empty():
                try: progress_queue.get_nowait()
                except queue.Empty: break
            
            simple_callback = create_inpainting_callback(self.stop_event, int(steps), module_translations, progress_queue)

            thread, result_container = execute_pipeline_task_async(
                pipe=pipe,
                prompt=final_prompt_text,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                width=width,
                height=height,
                seed=current_seed,
                device=self.model_manager.device,
                stop_event=self.stop_event,
                translations=module_translations,
                progress_queue=progress_queue,
                preview_queue=None,
            )

            last_progress_html = ""
            while thread.is_alive() or not progress_queue.empty():
                while not progress_queue.empty():
                    try:
                        current_step_prog, total_steps_prog = progress_queue.get_nowait()
                        progress_percent = int((current_step_prog / total_steps_prog) * 100)
                        step_info_text = f"Step {current_step_prog}/{total_steps_prog}"
                        last_progress_html = create_progress_bar_html(
                            current_step=current_step_prog,
                            total_steps=total_steps_prog,
                            progress_percent=progress_percent,
                            text_info=f"{image_info_text} - {step_info_text}"
                        )
                        yield generated_images_gallery, last_progress_html, btn_gen_off, btn_stop_on
                    except queue.Empty:
                        break
                time.sleep(0.05)
            thread.join()

            if result_container["status"] == "success" and result_container["final"]:
                result_image = result_container["final"]
                generated_images_gallery.append(result_image)
                temps_image_gen = time.time()
                
                current_time_str = time.strftime("%Y%m%d_%H%M%S")
                style_filename_part = "_".join(style_names_used) if style_names_used else "NoStyle"
                style_filename_part = style_filename_part.replace(" ", "_")[:30]
                output_filename = f"cogview3plus_{style_filename_part}_{current_time_str}_img{i+1}_{width}x{height}.{self.global_config['IMAGE_FORMAT'].lower()}"
                date_str = datetime.now().strftime("%Y_%m_%d")
                save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str)
                os.makedirs(save_dir, exist_ok=True)
                chemin_image = os.path.join(save_dir, output_filename)

                temps_gen_formatted = f"{(temps_image_gen - start_time_total):.2f}"

                xmp_data = {
                    "Module": "CogView3-Plus",
                    "Creator": self.global_config.get("AUTHOR", "CyberBill"),
                    "Model": COGVIEW3PLUS_MODEL_ID,
                    "Steps": steps,
                    "GuidanceScale": guidance_scale,
                    "Styles": ", ".join(style_names_used) if style_names_used else "None",
                    "Prompt": final_prompt_text,
                    "Size": f"{width}x{height}",
                    "Generation Time": f"{temps_gen_formatted} sec",
                    "Seed": current_seed
                }
                metadata_structure, prep_message = preparer_metadonnees_image(result_image, xmp_data, self.global_translations, chemin_image)
                print(txt_color("[INFO]", "info"), prep_message)
                enregistrer_image(result_image, chemin_image, self.global_translations, self.global_config["IMAGE_FORMAT"].upper(), metadata_to_save=metadata_structure)
                enregistrer_etiquettes_image_html(chemin_image, xmp_data, module_translations, is_last_image=(i == int(num_images) - 1))
                
                print(txt_color("[OK]", "ok"), f"{translate('image', module_translations)} {i+1}/{num_images} {translate('generer_en', module_translations)} {temps_gen_formatted} sec")
                yield generated_images_gallery, last_progress_html, btn_gen_off, btn_stop_on
            elif result_container["status"] == "stopped":
                break
            else:
                error_msg = f"{translate('cogview3plus_error_generation', module_translations)} ({image_info_text}): {result_container.get('error', 'Unknown error')}"
                print(txt_color("[ERREUR]", "erreur"), error_msg)
                final_message = f'<p style="color:red;">{error_msg}</p>'
                gr.Error(error_msg)
            
            # Nettoyage mémoire après chaque image (sauf si arrêté globalement par self.stop_event)
            if result_container["status"] != "stopped" and not self.stop_event.is_set():
                print(txt_color("[INFO]", "info"), f"Nettoyage de la mémoire (pré-GC) pour l'image {i+1}/{num_images}...")
                
                # Supprimer explicitement les grosses variables de cette itération
                # result_image est défini si status == "success"
                if 'result_image' in locals() and result_image is not None:
                    try:
                        del result_image
                        print(txt_color("[INFO]", "info"), "Variable 'result_image' supprimée.")
                    except NameError:
                        pass # Ne devrait pas arriver avec 'in locals()'
                
                # result_container et thread sont recréés à chaque itération
                if 'result_container' in locals() and result_container is not None:
                    try:
                        del result_container
                        print(txt_color("[INFO]", "info"), "Variable 'result_container' supprimée.")
                    except NameError:
                        pass
                
                if 'thread' in locals() and thread is not None:
                    try:
                        del thread
                        print(txt_color("[INFO]", "info"), "Variable 'thread' supprimée.")
                    except NameError:
                        pass

                print(txt_color("[INFO]", "info"), f"Exécution de gc.collect() et torch.cuda.empty_cache() pour l'image {i+1}/{num_images}...")
                gc.collect()
                if self.model_manager.device.type == 'cuda':
                    torch.cuda.empty_cache()

        if not self.stop_event.is_set():
            temps_total_final = f"{(time.time() - start_time_total):.2f}"
            if int(num_images) > 1:
                final_message = translate("batch_complete", module_translations).format(num_images=num_images, time=temps_total_final)
            else:
                final_message = translate("cogview3plus_generation_complete", module_translations).format(time=temps_total_final)
            print(txt_color("[OK]", "ok"), final_message)
            gr.Info(final_message, duration=3.0)
        else:
            final_message = translate("generation_arretee", module_translations)

        gc.collect()
        if self.model_manager.device.type == 'cuda':
            torch.cuda.empty_cache()
        yield generated_images_gallery, final_message, gr.update(interactive=True), gr.update(interactive=False)
