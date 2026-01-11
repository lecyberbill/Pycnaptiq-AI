# openpose_editor_mod.py
import os
import json
import base64
import io
from PIL import Image
from Utils.utils import txt_color, translate, GestionModule

# Obtenir le chemin du fichier JSON du module
module_json_path = os.path.join(os.path.dirname(__file__), "openpose_editor_mod.json")

# Charger les données du module
with open(module_json_path, 'r', encoding="utf-8") as f:
    module_data = json.load(f)

# On utilise Gradio pour l'UI
import gradio as gr

def initialize(translations, model_manager, gestionnaire, config):
    """Initialise le module OpenPose Editor."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    return OpenPoseEditor(translations, model_manager, gestionnaire, config)

class OpenPoseEditor:
    def __init__(self, translations, model_manager, gestionnaire, config):
        self.translations = translations
        self.config = config
        self.module_translations = module_data["language"].get(
            config.get("LANGUAGE", "fr") if config else "fr", 
            module_data["language"]["en"]
        )
        self.instance_targets = {}

    def create_tab(self, translations):
        """Crée un onglet dédié pour l'éditeur OpenPose."""
        with gr.TabItem(translate("openpose_editor_label", self.module_translations)):
            self.create_ui(controlnet_image_input=None, suffix="_tab")

    def create_ui(self, controlnet_image_input=None, suffix=""):
        """
        Crée l'interface de l'éditeur OpenPose.
        suffix: permet d'avoir plusieurs instances avec des IDs uniques.
        """
        with gr.Accordion(translate("openpose_editor_label", self.module_translations), open=(suffix != "")):
            with gr.Row():
                openpose_editor_btn = gr.Button(
                    translate("openpose_editor_btn", self.module_translations), 
                    elem_id=f"openpose_editor_btn_main{suffix}"
                )
            
            self.instance_targets[suffix] = (controlnet_image_input is not None)
            openpose_suffix_state = gr.State(value=suffix)
            
            # Le conteneur HTML pour le canvas
            openpose_canvas = gr.HTML(
                value=f'<style>'
                      f'.op-hidden {{ position: absolute !important; left: -9999px !important; top: -9999px !important; width: 1px !important; height: 1px !important; overflow: hidden !important; }}'
                      f'</style>'
                      f'<div id="openpose_container{suffix}" style="display:none; margin-top:10px;">'
                      f'<canvas id="openpose_canvas{suffix}" width="512" height="512" style="border:1px solid #444; background:#000; cursor:crosshair;"></canvas>'
                      f'<div style="margin-top:10px; display:flex; gap:10px;">'
                      f'<button id="op_apply{suffix}" class="lg primary svelte-cmf5ev" style="padding: 10px;">{translate("openpose_editor_apply", self.module_translations)}</button>'
                      f'<button id="op_add{suffix}" class="lg svelte-cmf5ev" style="padding: 10px;">{translate("openpose_editor_add_person", self.module_translations)}</button>'
                      f'<button id="op_reset{suffix}" class="lg svelte-cmf5ev" style="padding: 10px;">{translate("openpose_editor_reset", self.module_translations)}</button>'
                      f'<button id="op_clear{suffix}" class="lg svelte-cmf5ev" style="padding: 10px;">{translate("openpose_editor_clear", self.module_translations)}</button>'
                      f'</div></div>',
                elem_id=f"openpose_html_container{suffix}"
            )

            # Input pour l'image de fond
            openpose_bg_input = gr.Image(
                label=translate("openpose_editor_bg_label", self.module_translations),
                type="pil",
                elem_id=f"openpose_bg_input{suffix}"
            )

            # Output pour l'image de pose générée
            openpose_result_output = gr.Image(
                label=translate("openpose_editor_apply", self.module_translations),
                type="pil",
                elem_id=f"openpose_result_output{suffix}",
                interactive=False
            )
            
            # Composants cachés pour le pont JS -> Python
            openpose_buffer = gr.Textbox(elem_id=f"openpose_data_buffer{suffix}", elem_classes=["op-hidden"])
            openpose_trigger = gr.Button(elem_id=f"openpose_apply_trigger{suffix}", elem_classes=["op-hidden"])
            
            # Composants cachés pour le pont Python -> JS (Background)
            openpose_bg_buffer = gr.Textbox(elem_id=f"openpose_bg_buffer{suffix}", elem_classes=["op-hidden"])

            # Envoi de l'image de fond au JS via le buffer
            def set_background(image):
                if image is None:
                    return ""
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"

            openpose_bg_input.change(
                fn=set_background,
                inputs=[openpose_bg_input],
                outputs=[openpose_bg_buffer]
            )

            # Liaison du bouton invisible pour traiter l'image
            outputs = [openpose_result_output]
            if controlnet_image_input:
                outputs.append(controlnet_image_input)

            openpose_trigger.click(
                fn=self.process_openpose_canvas,
                inputs=[openpose_buffer, openpose_suffix_state],
                outputs=outputs
            )

        return openpose_editor_btn

    def process_openpose_canvas(self, base64_data, suffix):
        """Décode l'image Base64 du canvas et met à jour l'entrée ControlNet."""
        if not base64_data or not base64_data.startswith("data:image/"):
            if self.instance_targets.get(suffix):
                return [gr.update(), gr.update()]
            return gr.update()
        
        try:
            # Extraire les données base64
            header, encoded = base64_data.split(",", 1)
            data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(data))
            
            if self.instance_targets.get(suffix):
                return [image, image]
            return image
        except Exception as e:
            print(f"Erreur décodage OpenPose dans le module: {e}")
            if self.instance_targets.get(suffix):
                return [None, None]
            return None
