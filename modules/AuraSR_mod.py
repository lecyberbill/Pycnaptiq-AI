# test_module_mod.py
import os
import json
import time
from datetime import datetime
from Utils.utils import txt_color, translate, GestionModule, decharger_modele, enregistrer_image  
# Obtenir le chemin du fichier JSON du module
module_json_path = os.path.join(os.path.dirname(__file__), "AuraSR_mod.json")

# Créer une instance de GestionModule pour gérer les dépendances
with open(module_json_path, 'r', encoding="utf-8") as f:
    module_data = json.load(f)
module_manager = GestionModule(translations=module_data["language"]["fr"])


# Maintenant, on peut faire les imports en toute sécurité
import gradio as gr
from aura_sr import AuraSR

def initialize(global_translations, global_pipe = None, global_compel=None, global_config=None):
    """Initialise le module AuraSR."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    return AuraSRModule(global_translations, global_pipe, global_compel, global_config)

class AuraSRModule:
    def __init__(self, global_translations, global_pipe=None, global_compel=None, global_config=None, *args, **kwargs):
        """Initialise la classe TestModule."""
        self.global_translations = global_translations
        self.global_config = global_config
        if global_pipe is not None:
            self.global_pipe = global_pipe
        else:
            self.global_pipe = None
        
        if global_compel is not None:
            self.global_combel = global_compel 
        else:
            self.global_compel = None
            
        
    
    
    def create_tab(self, module_translations):
        """Crée l'onglet Gradio pour ce module."""
        with gr.Tab(translate("aura_sr_tab", module_translations)) as tab:
            gr.Markdown(f"## {translate('aura_sr_title', module_translations)}")

            with gr.Row():  # Row to arrange input and output side-by-side
                image_input = gr.Image(label=translate("image_input", module_translations), type="pil")
                image_output = gr.Image(label=translate("image_output", module_translations), type="pil")


            upscale_button = gr.Button(translate("upscale_button", module_translations))  # Button below

            def upscale_image(image):
                start_time = time.time()
                upscaled_image = self.upscaleImage_BY_AuraSR(image, module_translations)
                date_str = datetime.now().strftime("%Y_%m_%d")
                heure_str = datetime.now().strftime("%H_%M_%S")
                save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str)
                os.makedirs(save_dir, exist_ok=True)
                chemin_image = os.path.join(save_dir, f"aura_sr_{module_translations.get('upscaled_image', 'upscaled_image')}_{date_str}_{heure_str}.jpg")
                enregistrer_image(upscaled_image, chemin_image, self.global_translations, "JPEG")
                elapsed_time = f"{(time.time() - start_time):.2f} sec"
                print(txt_color("[INFO] ","info"),f"{translate('temps_total_generation', self.global_translations)} : {elapsed_time}")
                gr.Info(f"{translate('temps_total_generation', self.global_translations)} : {elapsed_time}", 3.0)
                return upscaled_image

            upscale_button.click(fn=upscale_image, inputs=image_input, outputs=image_output, api_name="upscale_image")

        return tab
    
    def upscaleImage_BY_AuraSR(self, image, module_translations):
        """
        Upscale an image using AuraSR.

        Args:
            image (PIL.Image): The image to upscale.
            scale (int): The scale factor for upscaling.
            self.global_compel: The global compel object.
            self.global_pipe: The global pipeline object.

        Returns:
            PIL.Image: The upscaled image.
        """
        if self.global_pipe is not None:
            self.global_pipe, self.global_compel = decharger_modele(self.global_pipe, None, self.global_translations)  # decharger_modele doesn't use compel in this case
            self.global_pipe = None
            self.global_compel = None
        
        
        aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")                
        print(translate('Upscaling_image_with_AuraSR', module_translations))
        gr.Info(translate('Upscaling_image_with_AuraSR', module_translations), 3.0)
        upscaled_image = aura_sr.upscale_4x_overlapped(image)
        return upscaled_image 
        
        
