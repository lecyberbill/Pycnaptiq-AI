# test_module_mod.py
import os
import json
import time
import threading
import random
import traceback
from datetime import datetime

import gradio as gr
from PIL import Image

# Application standard utilities
from Utils.utils import txt_color, translate, create_progress_bar_html
from Utils.model_manager import ModelManager
from Utils import llm_prompter_util
from Utils.gradio_components import create_prompt_interface, create_output_interface

# --- Module Metadata ---
MODULE_NAME = "test_module"
module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

try:
    with open(module_json_path, 'r', encoding="utf-8") as f:
        module_data = json.load(f)
except Exception:
    module_data = {"name": "Test Module Template"}

# --- Entry point for dynamic loading ---
def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire, global_config=None):
    """
    Function called by GestionModule to initialize the module.
    """
    print(txt_color("[OK] ", "ok"), f"Initializing skeleton module: {module_data.get('name')}")
    return TestModule(global_translations, model_manager_instance, gestionnaire, global_config)

class TestModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire, global_config=None):
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.gestionnaire = gestionnaire
        self.global_config = global_config or {}
        
        # Compat for standard components
        self.styles = [] 
        self.lora_choices_for_ui = [translate("aucun_lora_disponible", self.global_translations)]
        self.has_loras = False
        
        # Thread-safe cancellation
        self.stop_event = threading.Event()
        self.module_translations = {}

    def stop_generation(self):
        """Signals the current process to stop."""
        self.stop_event.set()
        return translate("stop_requested", self.module_translations)

    def create_tab(self, module_translations):
        """
        Creates the Gradio UI for the module.
        'module_translations' is the merged dictionary (global + module).
        """
        self.module_translations = module_translations
        
        with gr.Tab(translate("test_module_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('test_module_tab_title', self.module_translations)}")
            
            # --- STATES FOR LLM PROMPT ENHANCEMENT CYCLE ---
            test_original_prompt_state = gr.State(value="")
            test_prompt_is_enhanced_state = gr.State(value=False)
            test_enhancement_cycle_active_state = gr.State(value=False)
            test_last_ai_output_state = gr.State(value=None)
            
            with gr.Row():
                # COLUMN 1: INPUTS
                with gr.Column(scale=1, variant="panel"):
                    (self.test_prompt, 
                     self.test_enhance_btn, 
                     self.test_validate_btn, 
                     self.test_traduire_chk, 
                     self.test_style_dd) = create_prompt_interface(self, self.module_translations, prefix="test_")
                    
                    with gr.Accordion(translate("parametres_generation", self.module_translations), open=True):
                        self.test_steps = gr.Slider(1, 50, value=20, label=translate("etapes", self.module_translations))
                        self.test_seed = gr.Number(label="Seed", value=-1, precision=0)

                # COLUMN 2: OUTPUTS
                with gr.Column(scale=1, variant="panel"):
                    (self.test_status, 
                     self.test_gallery, 
                     self.test_gen_btn, 
                     self.test_stop_btn, 
                     self.test_progress_html) = create_output_interface(self.module_translations, prefix="test_")

            # --- EVENT BINDINGS ---
            
            # 1. LLM Enhancement Cycle
            self.test_enhance_btn.click(
                fn=llm_prompter_util.on_enhance_or_redo_button_click,
                inputs=[self.test_prompt, test_original_prompt_state, test_enhancement_cycle_active_state, 
                        gr.State(self.global_config.get("LLM_PROMPTER_MODEL_PATH")), gr.State(self.module_translations)],
                outputs=[self.test_prompt, self.test_enhance_btn, self.test_validate_btn, test_original_prompt_state, 
                         test_prompt_is_enhanced_state, test_enhancement_cycle_active_state, test_last_ai_output_state]
            )
            
            self.test_prompt.input(
                fn=llm_prompter_util.handle_text_input_change,
                inputs=[self.test_prompt, test_last_ai_output_state, test_enhancement_cycle_active_state, 
                        gr.State(self.global_config.get("LLM_PROMPTER_MODEL_PATH")), gr.State(self.module_translations)],
                outputs=[self.test_enhance_btn, self.test_validate_btn, test_original_prompt_state, 
                         test_prompt_is_enhanced_state, test_enhancement_cycle_active_state, test_last_ai_output_state]
            )

            # 2. Generation Process
            self.test_gen_btn.click(
                fn=self.ui_bridge_process,
                inputs=[self.test_prompt, self.test_steps, self.test_seed],
                outputs=[self.test_gallery, self.test_progress_html, self.test_gen_btn, self.test_stop_btn, self.test_status]
            )
            
            self.test_stop_btn.click(
                fn=self.stop_generation, 
                outputs=[self.test_status]
            )

        return tab

    def ui_bridge_process(self, prompt, steps, seed):
        """
        UI Bridge for the generation process.
        Uses a generator (yield) to update UI state and progress.
        """
        self.stop_event.clear()
        
        # Initial UI State: Disable Gen, Enable Stop
        yield [], gr.update(), gr.update(interactive=False), gr.update(interactive=True), translate("chargement", self.module_translations)
        
        try:
            # Business Logic Execution
            for i in range(int(steps)):
                if self.stop_event.is_set():
                    yield [], "", gr.update(interactive=True), gr.update(interactive=False), translate("generation_arretee", self.module_translations)
                    return
                
                # Simulate work
                time.sleep(0.1)
                
                # Progress Update
                progress_pct = ((i + 1) / steps) * 100
                progress_msg = f"Step {i+1}/{int(steps)}"
                progress_html = create_progress_bar_html(i + 1, int(steps), progress_pct, progress_msg)
                
                yield [], progress_html, gr.update(interactive=False), gr.update(interactive=True), translate("en_cours", self.module_translations)

            # Final Result
            dummy_img = Image.new('RGB', (512, 512), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            yield [dummy_img], "", gr.update(interactive=True), gr.update(interactive=False), translate("termine", self.module_translations)
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error: {str(e)}"
            yield [], "", gr.update(interactive=True), gr.update(interactive=False), error_msg

    def get_pipe(self):
        """Helper to access the global pipeline."""
        return self.model_manager.get_current_pipe()
