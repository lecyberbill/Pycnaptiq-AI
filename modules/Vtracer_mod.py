# Vtracer_mod.py
import os
import json
import io
import tempfile
import vtracer # For vectorizing images
from datetime import datetime # Pour nommer les fichiers

import gradio as gr
from PIL import Image

from Utils.utils import (
    txt_color,
    translate,
    # enregistrer_image, # N'est plus nécessaire pour sauvegarder l'image d'entrée
    # preparer_metadonnees_image, # N'est plus nécessaire si on ne prépare pas de méta pour une image raster
    enregistrer_etiquettes_image_html, # Ajout pour mettre à jour le rapport HTML
)
from Utils.model_manager import ModelManager # Standard import for modules

MODULE_NAME = "vtracer"

module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

try:
    with open(module_json_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)
except FileNotFoundError:
    print(f"[ERREUR] Fichier JSON du module {MODULE_NAME} introuvable: {module_json_path}")
    module_data = {"name": MODULE_NAME, "language": {"fr": {}}} # Fallback
except json.JSONDecodeError:
    print(f"[ERREUR] Erreur de décodage JSON pour le module {MODULE_NAME}: {module_json_path}")
    module_data = {"name": MODULE_NAME, "language": {"fr": {}}} # Fallback

def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
    """Initialise le module Vtracer."""
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME)}")
    return VtracerModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class VtracerModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.gestionnaire = gestionnaire_instance
        self.module_translations = {} # Will be populated by create_tab

    def _translate(self, key, **kwargs):
        """Helper to use module-specific translations, falling back to global."""
        # Use self.module_translations if populated, otherwise fallback to global_translations
        active_translations = self.module_translations if self.module_translations else self.global_translations
        return translate(key, active_translations).format(**kwargs)

    def convert_image_to_svg_method(self, image_pil, color_mode, hierarchical, mode, filter_speckle,
                                    color_precision, layer_difference, corner_threshold,
                                    length_threshold, max_iterations, splice_threshold, path_precision):
        """Converts a PIL image to SVG using vtracer with customizable parameters."""

        if image_pil is None:
            gr.Warning(self._translate("vtracer_error_no_image"))
            return None, None

        # Convert PIL image to bytes for vtracer compatibility
        img_byte_array = io.BytesIO()
        image_pil.save(img_byte_array, format='PNG') # vtracer expects PNG bytes in this example
        img_bytes = img_byte_array.getvalue()

        # Perform the conversion
        svg_str = vtracer.convert_raw_image_to_svg(
            img_bytes,
            img_format='png', # Explicitly state format
            colormode=color_mode.lower(),
            hierarchical=hierarchical.lower(),
            mode=mode.lower(),
            filter_speckle=int(filter_speckle),
            color_precision=int(color_precision),
            layer_difference=int(layer_difference),
            corner_threshold=int(corner_threshold),
            length_threshold=float(length_threshold),
            max_iterations=int(max_iterations),
            splice_threshold=int(splice_threshold),
            path_precision=int(path_precision)
        )

        # Préparer les chemins et noms de fichiers
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str_save = datetime.now().strftime("%Y_%m_%d")
        save_dir = os.path.join(self.global_config.get("SAVE_DIR", "."), date_str_save)
        os.makedirs(save_dir, exist_ok=True)

        # Nom de fichier pour le SVG de sortie persistant
        svg_output_filename = f"vtracer_output_{current_time_str}_{image_pil.width}x{image_pil.height}.svg"
        chemin_svg_saved_persistently = os.path.join(save_dir, svg_output_filename)

        # Sauvegarder le SVG de manière persistante
        try:
            with open(chemin_svg_saved_persistently, 'w', encoding='utf-8') as f_svg_persistent:
                f_svg_persistent.write(svg_str)
            print(txt_color("[OK]", "ok"), f"SVG sauvegardé de manière persistante : {chemin_svg_saved_persistently}")
        except Exception as e_svg_save:
            print(txt_color("[ERREUR]", "erreur"), f"Erreur lors de la sauvegarde persistante du SVG : {e_svg_save}")
            # Continuer même si la sauvegarde persistante échoue, pour permettre le téléchargement

        # Sauvegarder la chaîne SVG dans un fichier temporaire pour le téléchargement (pour le composant gr.File)
        temp_svg_file = tempfile.NamedTemporaryFile(delete=False, suffix='.svg', mode='w', encoding='utf-8')
        temp_svg_file.write(svg_str)
        temp_svg_file.close()

        # --- Intégration de la sauvegarde d'image et du rapport HTML ---
        try:
            if self.global_config:
                # Les métadonnées décrivent la génération du fichier SVG.
                xmp_data = {
                    "Module": "VTracer SVG Converter",
                    "SourceImageType": "Raster", # Indiquer que la source était une image raster
                    "Creator": self.global_config.get("AUTHOR", "CyberBill"),
                    "OriginalImageSize": f"{image_pil.width}x{image_pil.height}",
                    "SVG_ColorMode": color_mode,
                    "SVG_Hierarchical": hierarchical,
                    "SVG_Mode": mode,
                    "SVG_FilterSpeckle": int(filter_speckle),
                    "SVG_ColorPrecision": int(color_precision),
                    "SVG_LayerDifference": int(layer_difference),
                    "SVG_CornerThreshold": int(corner_threshold),
                    "SVG_LengthThreshold": float(length_threshold),
                    "SVG_MaxIterations": int(max_iterations),
                    "SVG_SpliceThreshold": int(splice_threshold),
                    "SVG_PathPrecision": int(path_precision),
                    "GeneratedSVGFile": svg_output_filename # Nom du fichier SVG généré
                }

                # preparer_metadonnees_image et enregistrer_image ne sont plus nécessaires ici
                # car nous ne sauvegardons plus l'image d'entrée raster.
                
                # Enregistrer les informations dans le rapport HTML, en utilisant le chemin du SVG sauvegardé.
                # La miniature <img> sera probablement cassée, mais le lien fonctionnera.
                enregistrer_etiquettes_image_html(chemin_svg_saved_persistently, xmp_data, self.module_translations, is_last_image=True)
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"Erreur lors de la sauvegarde de l'image/métadonnées pour VTracer: {e}")
            # Ne pas bloquer le retour du SVG même si la sauvegarde échoue

        # Préparer le HTML pour afficher le SVG. Utiliser un div pour un meilleur contrôle de la mise en page.
        # The viewBox should match the original image dimensions for correct scaling.
        svg_display_html = f'<div style="width:100%; max-height:500px; overflow:auto; border:1px solid #ddd; background-color: #f9f9f9;">' \
                           f'<svg preserveAspectRatio="xMidYMid meet" viewBox="0 0 {image_pil.width} {image_pil.height}" style="width:100%; height:auto;">{svg_str}</svg>' \
                           f'</div>'

        return svg_display_html, temp_svg_file.name

    def create_tab(self, module_translations_from_gestionnaire):
        """Crée l'onglet Gradio pour le module Vtracer."""
        self.module_translations = module_translations_from_gestionnaire

        with gr.Tab(self._translate("vtracer_tab_name")) as tab:
            gr.Markdown(f"## {self._translate('vtracer_tab_title')}") # Main title for the tab

            with gr.Blocks() as vtracer_interface:
                with gr.Row():
                    # --- Colonne de Gauche (Réglages) ---
                    with gr.Column(scale=1): # Les réglages
                        gr.Markdown(f"### {self._translate('vtracer_settings_label')}") # New key
                        color_mode_input = gr.Radio(choices=["Color", "Binary"], value="Color", label=self._translate("vtracer_color_mode_label"))
                        hierarchical_input = gr.Radio(choices=["Stacked", "Cutout"], value="Stacked", label=self._translate("vtracer_hierarchical_label"))
                        mode_input = gr.Radio(choices=["Spline", "Polygon", "None"], value="Spline", label=self._translate("vtracer_mode_label"))
                        filter_speckle_input = gr.Slider(minimum=1, maximum=10, value=4, step=1, label=self._translate("vtracer_filter_speckle_label"))
                        color_precision_input = gr.Slider(minimum=1, maximum=8, value=6, step=1, label=self._translate("vtracer_color_precision_label"))
                        layer_difference_input = gr.Slider(minimum=1, maximum=32, value=16, step=1, label=self._translate("vtracer_layer_difference_label"))
                        corner_threshold_input = gr.Slider(minimum=10, maximum=90, value=60, step=1, label=self._translate("vtracer_corner_threshold_label"))
                        length_threshold_input = gr.Slider(minimum=3.5, maximum=10, value=4.0, step=0.5, label=self._translate("vtracer_length_threshold_label"))
                        max_iterations_input = gr.Slider(minimum=1, maximum=20, value=10, step=1, label=self._translate("vtracer_max_iterations_label"))
                        splice_threshold_input = gr.Slider(minimum=10, maximum=90, value=45, step=1, label=self._translate("vtracer_splice_threshold_label"))
                        path_precision_input = gr.Slider(minimum=1, maximum=10, value=8, step=1, label=self._translate("vtracer_path_precision_label"))

                    # --- Colonne de Droite (Image Entrée, Sortie SVG, Boutons) ---
                    with gr.Column(scale=2): # L'image et la sortie
                        input_image_component = gr.Image(type="pil", label=self._translate("vtracer_upload_image_label"))
                        svg_output_html = gr.HTML(label=self._translate("vtracer_svg_output_label"))
                        svg_file_download = gr.File(label=self._translate("vtracer_download_svg_label"))
                        
                        with gr.Row():
                            clear_button = gr.Button(value=self._translate("vtracer_clear_button_label")) # New key
                            submit_button = gr.Button(value=self._translate("vtracer_submit_button_label"), variant="primary") # New key

                # --- Logique des boutons ---
                all_inputs = [
                    input_image_component, color_mode_input, hierarchical_input, mode_input,
                    filter_speckle_input, color_precision_input, layer_difference_input,
                    corner_threshold_input, length_threshold_input, max_iterations_input,
                    splice_threshold_input, path_precision_input
                ]
                all_outputs_for_clear = all_inputs + [svg_output_html, svg_file_download]


                submit_button.click(
                    fn=self.convert_image_to_svg_method,
                    inputs=all_inputs,
                    outputs=[svg_output_html, svg_file_download]
                )

                def clear_all_fields():
                    return (
                        gr.update(value=None),  # input_image_component
                        gr.update(value="Color"),  # color_mode_input
                        gr.update(value="Stacked"),  # hierarchical_input
                        gr.update(value="Spline"),  # mode_input
                        gr.update(value=4),  # filter_speckle_input
                        gr.update(value=6),  # color_precision_input
                        gr.update(value=16),  # layer_difference_input
                        gr.update(value=60),  # corner_threshold_input
                        gr.update(value=4.0),  # length_threshold_input
                        gr.update(value=10),  # max_iterations_input
                        gr.update(value=45),  # splice_threshold_input
                        gr.update(value=8),  # path_precision_input
                        gr.update(value=""),  # svg_output_html
                        gr.update(value=None)  # svg_file_download
                    )

                clear_button.click(
                    fn=clear_all_fields,
                    inputs=None,
                    outputs=all_outputs_for_clear
                )

        return tab