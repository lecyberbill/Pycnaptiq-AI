#upscaler_mod
import os
import json
from Utils.utils import txt_color, translate, GestionModule, lister_fichiers, decharger_modele, check_gpu_availability

#JSON associated at this module
module_json_path = os.path.join(os.path.dirname(__file__), "upscaler_SDXL_mod.json")


# Create a ManagementModule instance to manage dependencies
with open(module_json_path, 'r', encoding="utf-8") as f:
    module_data = json.load(f)


# Now we can do imports safely

import gradio as gr
from diffusers import StableDiffusionUpscalePipeline
import torch
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType


def initialize(global_translations, global_pipe=None, global_compel=None,  global_config=None):
    """Initialise le module test."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    return UpscalerSDXLModule(global_translations, global_pipe, global_compel, global_config=None)

class UpscalerSDXLModule:
    def __init__(self, global_translations, global_pipe=None, global_compel=None, global_config=None):
        """Initialise la classe UpscalerModule."""
        self.global_config = None
        self.upscalerModel_path = module_data["upscaler_model_dir"]
        self.global_translations = global_translations
        self.liste_modeles = lister_fichiers(self.upscalerModel_path, self.global_translations)
        self.device, self.torch_dtype, self.vram_total_gb = check_gpu_availability(self.global_translations)
        if global_compel is not None:
            self.global_compel = global_compel
        else:
            self.global_compel = None

        if global_pipe is not None:
            self.global_pipe = global_pipe
        else:
            self.global_pipe = None        

    def create_tab(self, module_translations):
        """
        Crée l'onglet Gradio pour ce module.

        Args:
        module_translations (dict): Le dictionnaire de traductions du module.

        Returns:
        gr.Tab: L'onglet Gradio créé.
        """
        with gr.Tab(translate("upscaler_tab_name", module_translations)) as tab:  # Crée un onglet avec un nom traduit
            gr.Markdown(f"## {translate('upscaler_title', module_translations)}")
            with gr.Row():
                self.modele_upscaler_dropdown = gr.Dropdown(label=translate("selectionner_modele", module_translations), choices=self.liste_modeles, value=self.liste_modeles[0] if self.liste_modeles else None)
                self.bouton_lister = gr.Button(translate("lister_modeles", module_translations))
                self.bouton_charger_upscaler = gr.Button(translate("charger_modele", module_translations))
            self.message_chargement_upscaler = gr.Textbox(label=translate("statut", module_translations), value=translate("aucun_modele_charge", module_translations))

            with gr.Row():
                self.image_input = gr.Image(label=translate("input_image", module_translations), type="pil")
                self.prompt_input = gr.Textbox(label=translate("prompt", module_translations))
                self.upscaled_image_output = gr.Image(label=translate("output_image", module_translations), type="pil")
            with gr.Row():
                self.step_slider = gr.Slider(minimum=1, maximum=100, value=75, step=1, label=translate("etapes", module_translations))
                self.guidance_slider = gr.Slider(minimum=1, maximum=20, value=9, step=0.5, label=translate("guidage", module_translations))
                self.noise_slider = gr.Slider(minimum=0, maximum=100, value=20, step=1, label=translate("noise", module_translations))
            self.bouton_upscale = gr.Button(translate("upscale_image", module_translations))

            self.bouton_lister.click(
                fn=self.mettre_a_jour_listes,
                outputs=self.modele_upscaler_dropdown
            )

            self.bouton_charger_upscaler.click(
                fn=self.charger_modele_upscaler_gradio,
                inputs=self.modele_upscaler_dropdown,
                outputs=self.message_chargement_upscaler
            )
            self.bouton_upscale.click(
                fn=self.upscale_image,
                inputs=[self.prompt_input, self.image_input, self.step_slider, self.guidance_slider, self.noise_slider],
                outputs=self.upscaled_image_output
            )

        return tab
    
    def mettre_a_jour_listes(self):
        """Met à jour la liste des modèles disponibles."""
        self.liste_modeles = lister_fichiers(self.upscalerModel_path, self.global_translations)
        return gr.update(choices=self.liste_modeles)

    def charger_modele_upscaler(self, nom_fichier, upscalerModel_path, module_translations):
        """
        Charge un modèle d'upscaling à partir d'un fichier et le stocke dans self.global_pipe.

        Args:
            nom_fichier (str): Le nom du fichier contenant le modèle à charger.
            upscalerModel_path (str): Le chemin du dossier contenant le modèle à charger.
            module_translations (dict): Le dictionnaire de traductions du module.

        Returns:
            tuple: A tuple containing:
                - self.global_pipe (StableDiffusionUpscalePipeline or None): The loaded pipeline or None if an error occurred.
                - message (str): A message to display in Gradio.
        """
        try:
            if not nom_fichier:
                print(txt_color("[ERREUR] ", "erreur"), translate("aucun_modele_selectionne", module_translations))
                gr.Warning(translate("aucun_modele_selectionne", module_translations), 4.0)
                return None, translate("aucun_modele_selectionne", module_translations)
            
            chemin_modele = os.path.join(upscalerModel_path, nom_fichier)
            
            if not os.path.exists(chemin_modele):
                print(txt_color("[ERREUR] ", "erreur"), f"{translate('modele_non_trouve', module_translations)}: {chemin_modele}")
                gr.Warning(translate("modele_non_trouve", module_translations) + f": {chemin_modele}", 4.0)
                return None, translate("modele_non_trouve", module_translations)

            if self.global_pipe is not None:
                self.global_pipe, self.global_compel = decharger_modele(self.global_pipe, None, self.global_translations)  # decharger_modele doesn't use compel in this case
                self.global_pipe = None

            print(txt_color("[INFO] ", "info"), f"{translate('chargement_modele', module_translations)} : {nom_fichier}")
            gr.Info(translate("chargement_modele", module_translations) + f" : {nom_fichier}", 3.0)
            self.global_pipe = StableDiffusionUpscalePipeline.from_single_file(
                chemin_modele,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
            ).to(self.device)

            self.global_pipe.enable_xformers_memory_efficient_attention()
            self.global_pipe.enable_model_cpu_offload()
            self.global_pipe.enable_vae_slicing()
            self.global_pipe.enable_vae_tiling()

            # Initialisation de compel
            self.global_compel = Compel(
            tokenizer=self.global_pipe.tokenizer,
            text_encoder=self.global_pipe.text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=False
        )

            print(txt_color("[OK] ", "ok"), translate("modele_charge", module_translations), f": {nom_fichier}")
            gr.Info(translate("modele_charge", module_translations) + f": {nom_fichier}", 3.0)
            return self.global_pipe, translate("modele_charge", module_translations)

        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_chargement_modele", module_translations), f": {e}")
            raise gr.Error(translate("erreur_chargement_modele", module_translations), f": {e}", 4.0)
            return None, translate("erreur_chargement_modele", module_translations) + f": {e}"
    
    def charger_modele_upscaler_gradio(self, nom_fichier):
        """
        Charge un modèle d'upscaling et retourne un message pour Gradio.
        """
        _, message = self.charger_modele_upscaler(nom_fichier, self.upscalerModel_path, self.global_translations)
        return message

    def upscale_image(self, prompt, image, step, guidance, noise_level):
        """
        Upscales an image using the loaded upscaler model.

        Args:
            prompt (str): The prompt to guide the upscaling process.
            image (PIL.Image.Image): The image to upscale.

        Returns:
            PIL.Image.Image: The upscaled image, or None if an error occurs.
        """
        try:
            if self.global_pipe is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_modele_upscaler", self.global_translations))
                gr.Warning(translate("erreur_pas_modele_upscaler", self.global_translations), 4.0)
                return None

            if image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image", self.global_translations))
                gr.Warning(translate("erreur_pas_image", self.global_translations), 4.0)
                return None

            print(txt_color("[INFO] ", "info"), translate("debut_upscaling", self.global_translations))
            gr.Info(translate("debut_upscaling", self.global_translations), 3.0)
        

            # Convert the prompt to embeddings using Compel
            # Use only the first element of the returned tuple
            embeddings = self.global_compel(prompt)
            conditioning = embeddings
            pooled = None
            if isinstance(embeddings, tuple):
                conditioning = embeddings[0]
                pooled = embeddings[1]

            # Upscale the image
            upscaled_image = self.global_pipe(
                prompt_embeds=conditioning,
                image=image,
                num_inference_steps=step,
                guidance_scale=guidance,
                noise_level=noise_level
            ).images[0]

            print(txt_color("[OK] ", "ok"), translate("fin_upscaling", self.global_translations))
            gr.Info(translate("fin_upscaling", self.global_translations), 3.0)
            return upscaled_image

        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_upscaling", self.global_translations), f": {e}")
            raise gr.Error(translate("erreur_upscaling", self.global_translations), f": {e}", 4.0)
            return None
