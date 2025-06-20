# RealEdit_mod.py
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
from PIL import Image, ImageOps
import requests # For the example image download, not strictly needed for user upload

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler # Keep EulerAncestralDiscreteScheduler for setting it

from core.pipeline_executor import execute_pipeline_task_async # <-- AJOUT
from Utils.utils import (
    txt_color,
    translate,
    enregistrer_image,
    preparer_metadonnees_image,
    enregistrer_etiquettes_image_html,
    create_progress_bar_html,
    ImageSDXLchecker,
) # Keep ImageSDXLchecker
from Utils.callback_diffuser import create_inpainting_callback # Can be used for progress
from Utils.model_manager import ModelManager
from core.translator import translate_prompt

# --- Configuration et Constantes ---
MODULE_NAME = "realedit" # Lowercase name for consistency
REALEDIT_MODEL_ID = "peter-sushko/RealEdit"
REALEDIT_MODEL_TYPE_KEY = "realedit_instructpix2pix" # Unique key for ModelManager

module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

try:
    with open(module_json_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)
except FileNotFoundError:
    print(f"[ERREUR] Fichier JSON du module {MODULE_NAME} introuvable: {module_json_path}")
    module_data = {"name": MODULE_NAME.capitalize()} 
except json.JSONDecodeError:
    print(f"[ERREUR] Erreur de décodage JSON pour le module {MODULE_NAME}: {module_json_path}")
    module_data = {"name": MODULE_NAME.capitalize()} 

def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME.capitalize())}")
    return RealEditModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class RealEditModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.gestionnaire = gestionnaire_instance
        self.stop_event = threading.Event()
        self.module_translations = {} # Will be populated by create_tab

    def stop_generation(self):
        self.stop_event.set()
        # Use self.module_translations if available, otherwise global
        active_translations = self.module_translations if self.module_translations else self.global_translations
        print(txt_color("[INFO]", "info"), translate("stop_requested", active_translations))

    def create_tab(self, module_translations_arg):
        self.module_translations = module_translations_arg

        with gr.Tab(translate("realedit_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('realedit_tab_title', self.module_translations)}")

            with gr.Row():
                with gr.Column(scale=1):
                    self.re_image_input = gr.Image(
                        label=translate("realedit_image_input_label", self.module_translations),
                        type="pil"
                    )
                    self.re_prompt_input = gr.Textbox(
                        label=translate("realedit_prompt_label", self.module_translations),
                        info=translate("realedit_prompt_info", self.module_translations),
                        placeholder=translate("realedit_prompt_placeholder", self.module_translations),
                        lines=2
                    )
                    self.re_translate_checkbox = gr.Checkbox(
                        label=translate("traduire_en_anglais", self.global_translations), # Use global key
                        value=False
                    )
                    self.re_steps_slider = gr.Slider(
                        minimum=10, maximum=150, value=50, step=1,
                        label=translate("realedit_steps_label", self.module_translations)
                    )
                    self.re_image_guidance_scale_slider = gr.Slider(
                        minimum=1.0, maximum=10.0, value=2.0, step=0.1,
                        label=translate("realedit_image_guidance_scale_label", self.module_translations)
                    )

                with gr.Column(scale=1):
                    self.re_model_status_textbox = gr.Textbox(
                        label=translate("realedit_model_status", self.module_translations),
                        value=translate("realedit_model_not_loaded", self.module_translations),
                        interactive=False
                    )
                    self.re_load_model_button = gr.Button(
                        translate("realedit_load_button", self.module_translations)
                    )
                    self.re_output_image = gr.Image(
                        label=translate("output_image", self.global_translations), # Use global key
                        type="pil"
                    )
                    self.re_generate_button = gr.Button(
                        translate("realedit_generate_button", self.module_translations),
                        interactive=False, # Disabled until model is loaded
                        variant="primary"
                    )
                    self.re_stop_button = gr.Button(
                        translate("arreter", self.global_translations), # Use global key
                        interactive=False,
                        variant="stop"
                    )
                    self.re_progress_html = gr.HTML()

            self.re_load_model_button.click(
                fn=self.load_realedit_model_ui,
                inputs=None,
                outputs=[self.re_model_status_textbox, self.re_generate_button]
            )

            self.re_generate_button.click(
                fn=self.realedit_gen,
                inputs=[
                    self.re_image_input,
                    self.re_prompt_input,
                    self.re_translate_checkbox,
                    self.re_steps_slider,
                    self.re_image_guidance_scale_slider
                ],
                outputs=[
                    self.re_output_image,
                    self.re_progress_html,
                    self.re_generate_button,
                    self.re_stop_button
                ]
            )
            self.re_stop_button.click(fn=self.stop_generation, inputs=None, outputs=None)
        return tab

    def load_realedit_model_ui(self):
        yield gr.update(value=translate("realedit_loading_model", self.module_translations)), gr.update(interactive=False)

        success, message = self.model_manager.load_model(
            model_name=REALEDIT_MODEL_ID,
            model_type=REALEDIT_MODEL_TYPE_KEY, # Pass the new model type key
            gradio_mode=True
        )

        if success:
            # Set the scheduler after loading, if ModelManager doesn't handle it for this type
            pipe = self.model_manager.get_current_pipe()
            if pipe and isinstance(pipe, StableDiffusionInstructPix2PixPipeline):
                try:
                    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                    message += f" (+ {EulerAncestralDiscreteScheduler.__name__})"
                    print(txt_color("[INFO]", "info"), f"Scheduler set to {EulerAncestralDiscreteScheduler.__name__} for RealEdit.")
                except Exception as e_sched:
                    message += f" (Erreur config scheduler: {e_sched})"
                    print(txt_color("[ERREUR]", "erreur"), f"Erreur configuration scheduler RealEdit: {e_sched}")
            
            yield gr.update(value=message), gr.update(interactive=True)
        else:
            yield gr.update(value=message), gr.update(interactive=False)

    def realedit_gen(
        self,
        input_image_pil,
        edit_prompt,
        translate_flag,
        num_steps,
        image_guidance_scale
    ):
        active_translations = self.module_translations if self.module_translations else self.global_translations
        start_time_total = time.time()
        self.stop_event.clear()

        initial_output_image = None
        initial_progress = create_progress_bar_html(0, int(num_steps), 0, translate("preparation", active_translations)) # Initial progress bar
        btn_gen_off = gr.update(interactive=False)
        btn_stop_on = gr.update(interactive=True)

        yield initial_output_image, initial_progress, btn_gen_off, btn_stop_on

        pipe = self.model_manager.get_current_pipe()
        if pipe is None or self.model_manager.current_model_type != REALEDIT_MODEL_TYPE_KEY:
            msg = translate("realedit_error_no_model", active_translations)
            print(txt_color("[ERREUR]", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            yield None, msg, gr.update(interactive=True), gr.update(interactive=False)
            return

        if input_image_pil is None:
            msg = translate("realedit_error_no_image", active_translations)
            gr.Warning(msg, duration=4.0)
            yield None, msg, gr.update(interactive=True), gr.update(interactive=False)
            return

        if not (edit_prompt and edit_prompt.strip()):
            msg = translate("realedit_error_no_prompt", active_translations)
            gr.Warning(msg, duration=4.0)
            yield None, msg, gr.update(interactive=True), gr.update(interactive=False)
            return

        # Image conformity check
        checker = ImageSDXLchecker(input_image_pil, self.global_translations, max_pixels=1024*1024*2) # Adjust max_pixels if needed
        image_to_edit_pil = checker.redimensionner_image()

        final_edit_prompt = translate_prompt(edit_prompt, self.global_translations) if translate_flag else edit_prompt
        if translate_flag and final_edit_prompt != edit_prompt:
             gr.Info(translate("prompt_traduit_pour_generation", self.global_translations), 2.0)

        print(txt_color("[INFO]", "info"), f"{translate('realedit_generation_start', active_translations)} Prompt: {final_edit_prompt}")

        progress_queue = queue.Queue() # Queue for progress updates
        
        # Create the callback function
        # InstructPix2Pix doesn't typically provide latent previews via callback,
        # so create_inpainting_callback is suitable for just progress updates.
        callback_progress = create_inpainting_callback(
            self.stop_event,
            int(num_steps),
            active_translations,
            progress_queue
        )

        try:
            # Execute the pipeline in a separate thread
            pipeline_thread, result_container = execute_pipeline_task_async(
                pipe=pipe,
                prompt=final_edit_prompt, # Pass prompt as text
                image=image_to_edit_pil, # Pass the input image (corrected argument name)
                num_inference_steps=int(num_steps),
                image_guidance_scale=float(image_guidance_scale),
                # Other required args for execute_pipeline_task_async (even if not used by RealEdit pipe directly)
                guidance_scale=0.0, # Dummy value, RealEdit uses image_guidance_scale
                seed=0, # Dummy seed
                width=image_to_edit_pil.width, # Pass image dimensions
                height=image_to_edit_pil.height,
                device=self.model_manager.device,
                stop_event=self.stop_event,
                translations=active_translations,
                progress_queue=progress_queue,
                preview_queue=None, # No latent previews for RealEdit
                # callback_on_step_end is handled internally by execute_pipeline_task_async
            )

            # Loop to update UI from the progress queue
            last_progress_html = ""
            while pipeline_thread.is_alive() or not progress_queue.empty():
                if self.stop_event.is_set():
                    raise InterruptedError("Generation stopped by user.")
                
                # Read all available progress updates from the queue
                current_step_prog, total_steps_prog = None, int(num_steps)
                while not progress_queue.empty():
                    try:
                        current_step_prog, total_steps_prog = progress_queue.get_nowait()
                    except queue.Empty:
                        break # Exit inner loop if queue is empty
                
                # Update progress HTML if there was a new step
                new_progress_html = last_progress_html
                if current_step_prog is not None:
                    progress_percent = int((current_step_prog / total_steps_prog) * 100)
                    step_info_text = f"Step {current_step_prog}/{total_steps_prog}"
                    new_progress_html = create_progress_bar_html(
                        current_step=current_step_prog,
                        total_steps=total_steps_prog,
                        progress_percent=progress_percent,
                        text_info=step_info_text # Use step info as text
                    )
                    # Yield update to the UI
                    yield None, new_progress_html, btn_gen_off, btn_stop_on
                    last_progress_html = new_progress_html
                
                time.sleep(0.05) # Small delay to prevent blocking UI thread

            # Ensure the thread has finished
            pipeline_thread.join()

            # Retrieve the result from the container
            edited_image = result_container.get("final")
            generation_status = result_container.get("status")
            error_details = result_container.get("error")

            # Handle stop event or errors after the thread finishes
            if generation_status == "stopped":
                final_message_text = translate("generation_arretee", active_translations)
                print(txt_color("[INFO]", "info"), final_message_text)
                gr.Info(final_message_text, 3.0)
                # Yield final state: no image, stop message, re-enable buttons
                yield None, final_message_text, gr.update(interactive=True), gr.update(interactive=False)
                return # Exit the generator

            if generation_status == "error":
                error_msg = translate("realedit_error_generation", active_translations)
                full_error = f"{error_msg}: {error_details}"
                print(txt_color("[ERREUR]", "erreur"), full_error)
                traceback.print_exc()
                gr.Error(full_error)
                # Yield final state: no image, error message, re-enable buttons
                yield None, full_error, gr.update(interactive=True), gr.update(interactive=False)
                return # Exit the generator

            if edited_image is None:
                 # This case should ideally be covered by status == "error" or "stopped",
                 # but as a fallback check:
                 error_msg = translate("realedit_error_generation", active_translations) # Reuse error key
                 full_error = f"{error_msg}: No image returned by pipeline."
                 print(txt_color("[ERREUR]", "erreur"), full_error)
                 gr.Error(full_error)
                 yield None, full_error, gr.update(interactive=True), gr.update(interactive=False)
                 return # Exit the generator

            if self.stop_event.is_set():
                final_message_text = translate("generation_arretee", active_translations)
                yield None, final_message_text, gr.update(interactive=True), gr.update(interactive=False)
                return

            temps_gen_sec = time.time() - start_time_total
            
            # Save the image
            current_time_str = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"realedit_{current_time_str}_{edited_image.width}x{edited_image.height}.{self.global_config['IMAGE_FORMAT'].lower()}"
            date_str_save = datetime.now().strftime("%Y_%m_%d")
            save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str_save)
            os.makedirs(save_dir, exist_ok=True)
            chemin_image = os.path.join(save_dir, output_filename)

            xmp_data = {
                "Module": "RealEdit",
                "Creator": self.global_config.get("AUTHOR", "CyberBill"),
                "Model": REALEDIT_MODEL_ID,
                "InstructionPrompt": final_edit_prompt,
                "Steps": num_steps,
                "ImageGuidanceScale": image_guidance_scale,
                "OriginalImageSize": f"{input_image_pil.width}x{input_image_pil.height}",
                "EditedImageSize": f"{edited_image.width}x{edited_image.height}",
                "GenerationTimeSeconds": f"{temps_gen_sec:.2f}"
            }
            metadata_structure, prep_message = preparer_metadonnees_image(edited_image, xmp_data, self.global_translations, chemin_image)
            print(txt_color("[INFO]", "info"), prep_message)
            enregistrer_image(edited_image, chemin_image, self.global_translations, self.global_config["IMAGE_FORMAT"].upper(), metadata_to_save=metadata_structure)
            enregistrer_etiquettes_image_html(chemin_image, xmp_data, active_translations, is_last_image=True)
            
            final_message_text = translate("realedit_generation_complete", active_translations).format(time=temps_gen_sec)
            print(txt_color("[OK]", "ok"), final_message_text)
            gr.Info(final_message_text, duration=3.0)
            final_progress_html = create_progress_bar_html(int(num_steps), int(num_steps), 100, translate("termine", active_translations))
            yield edited_image, final_progress_html, gr.update(interactive=True), gr.update(interactive=False)

        except InterruptedError:
            final_message_text = translate("generation_arretee", active_translations)
            print(txt_color("[INFO]", "info"), final_message_text)
            gr.Info(final_message_text, 3.0)
            yield None, final_message_text, gr.update(interactive=True), gr.update(interactive=False)
        except Exception as e_gen:
            error_msg = translate("realedit_error_generation", active_translations)
            full_error = f"{error_msg}: {e_gen}"
            print(txt_color("[ERREUR]", "erreur"), full_error)
            traceback.print_exc()
            gr.Error(full_error)
            yield None, full_error, gr.update(interactive=True), gr.update(interactive=False)
        finally:
            gc.collect()
            if self.model_manager.device.type == 'cuda':
                torch.cuda.empty_cache()