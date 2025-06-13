# cogview4_mod.py
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
from diffusers import CogView4Pipeline # Import spécifique

from Utils.utils import (
    txt_color,
    translate,
    enregistrer_image,
    preparer_metadonnees_image,
    enregistrer_etiquettes_image_html,
    styles_fusion,
    create_progress_bar_html,
)
from Utils.callback_diffuser import create_inpainting_callback # Peut être simplifié si pas d'aperçu
from Utils.model_manager import ModelManager
from core.translator import translate_prompt
from core.pipeline_executor import execute_pipeline_task_async
from core.image_prompter import generate_prompt_from_image

# --- Configuration et Constantes ---
MODULE_NAME = "cogview4"
COGVIEW4_MODEL_ID = "THUDM/CogView4-6B"
COGVIEW4_MODEL_TYPE_KEY = "cogview4" # Clé pour ModelManager

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
    """Initialise le module CogView4."""
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME)}")
    return CogView4Module(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class CogView4Module:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
        """Initialise la classe CogView4Module."""
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.styles = self.load_styles()
        self.gestionnaire = gestionnaire_instance
        self.default_negative_prompt = self.global_config.get("NEGATIVE_PROMPT", "") # CogView4 ne l'utilise pas directement
        self.stop_event = threading.Event()
        self.module_translations = {}
        self.allowed_resolutions = self.global_config.get("COGVIEW4_ALLOWED_RESOLUTIONS")

        # Priorité 2: Dériver de FORMATS si COGVIEW4_ALLOWED_RESOLUTIONS n'est pas défini ou est vide
        if not self.allowed_resolutions:
            formats_config = self.global_config.get("FORMATS", [])
            if formats_config and isinstance(formats_config, list):
                parsed_resolutions = []
                for item in formats_config:
                    if isinstance(item, dict):
                        dimensions_str = item.get("dimensions")
                        if dimensions_str and isinstance(dimensions_str, str):
                            # Remplacer '*' par 'x'
                            processed_res = dimensions_str.replace('*', 'x')
                            # Valider basiquement le format "nombrexnombre"
                            if 'x' in processed_res.lower():
                                try:
                                    w_str, h_str = processed_res.split('x')
                                    int(w_str)
                                    int(h_str)
                                    parsed_resolutions.append(processed_res)
                                except ValueError:
                                    print(txt_color("[WWARNING]", "warning"), f"{translate('format_dimension_ignore_conversion', self.module_translations).format(dimensions_str=dimensions_str, processed_res=processed_res)}")
                if parsed_resolutions: # Utiliser seulement si on a pu parser quelque chose
                    self.allowed_resolutions = parsed_resolutions

        # Priorité 3: Fallback si tout échoue ou si la liste est vide après parsing
        if not self.allowed_resolutions: # S'assurer qu'il y a au moins une résolution valide
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

        with gr.Tab(translate("cogview4_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('cogview4_tab_title', self.module_translations)}")
            with gr.Row():
                with gr.Column(scale=2):
                    self.cogview4_prompt = gr.Textbox(
                        label=translate("cogview4_prompt_label", self.module_translations),
                        info=translate("cogview4_prompt_info", self.module_translations),
                        placeholder=translate("cogview4_prompt_placeholder", self.module_translations),
                        lines=3,
                    )
                    self.cogview4_traduire_checkbox = gr.Checkbox(
                        label=translate("traduire_en_anglais", self.module_translations),
                        value=False,
                        info=translate("traduire_prompt_libre", self.module_translations),
                    )
                    self.cogview4_style_dropdown = gr.Dropdown(
                        label=translate("selectionner_styles", self.module_translations),
                        choices=[style["name"] for style in self.styles],
                        value=[],
                        multiselect=True,
                        info=translate("selectionner_styles_info", self.module_translations),
                    )
                    self.cogview4_use_image_prompt_checkbox = gr.Checkbox(
                        label=translate("generer_prompt_image", self.global_translations),
                        value=False
                    )
                    self.cogview4_image_input_for_prompt = gr.Image(label=translate("telechargez_image", self.global_translations), type="pil", visible=False)
                    
                    with gr.Row():
                        self.cogview4_resolution_dropdown = gr.Dropdown(
                            label=translate("resolution_label", self.global_translations), 
                            choices=self.allowed_resolutions,
                            value=self.allowed_resolutions[0] if self.allowed_resolutions else "1024x1024",
                            info=translate("resolution_info_cogview4", self.module_translations) 
                        )
                        self.cogview4_steps_slider = gr.Slider(
                            minimum=1, maximum=100, value=50, step=1,
                            label=translate("cogview4_steps_label", self.module_translations)
                        )
                        self.cogview4_guidance_scale_slider = gr.Slider(
                            minimum=1.0, maximum=10.0, value=3.5, step=0.1,
                            label=translate("cogview4_guidance_scale_label", self.module_translations)
                        )
                    with gr.Row():
                         self.cogview4_num_images_slider = gr.Slider(
                            minimum=1, maximum=10, value=1, step=1, # Limité pour CogView4 pour l'instant
                            label=translate("nombre_images", self.module_translations),
                            interactive=True
                        )
                    with gr.Row():
                        self.cogview4_bouton_gen = gr.Button(
                            value=translate("cogview4_generate_button", self.module_translations), interactive=False
                        )
                        self.cogview4_bouton_stop = gr.Button(
                            translate("arreter", self.module_translations),
                            interactive=False, variant="stop",
                        )
                    self.cogview4_progress_html = gr.HTML()

                with gr.Column(scale=1):
                    self.cogview4_message_chargement = gr.Textbox(
                        label=translate("cogview4_model_status", self.module_translations),
                        value=translate("cogview4_model_not_loaded", self.module_translations),
                        interactive=False,
                    )
                    self.cogview4_bouton_charger = gr.Button(
                        translate("cogview4_load_button", self.module_translations)
                    )
                    self.cogview4_result_output = gr.Gallery(
                        label=translate("output_image", self.module_translations),
                    )

            self.cogview4_use_image_prompt_checkbox.change(
                fn=lambda use_image: gr.update(visible=use_image),
                inputs=self.cogview4_use_image_prompt_checkbox,
                outputs=self.cogview4_image_input_for_prompt
            )
            self.cogview4_image_input_for_prompt.change(
                fn=self.update_prompt_from_image,
                inputs=[self.cogview4_image_input_for_prompt, self.cogview4_use_image_prompt_checkbox, gr.State(self.global_translations)],
                outputs=self.cogview4_prompt
            )
            self.cogview4_bouton_charger.click(
                fn=self.load_cogview4_model_ui,
                inputs=None,
                outputs=[self.cogview4_message_chargement, self.cogview4_bouton_gen],
            )
            self.cogview4_bouton_gen.click(
                fn=self.cogview4_gen,
                inputs=[
                    self.cogview4_prompt,
                    self.cogview4_traduire_checkbox,
                    self.cogview4_style_dropdown,
                    self.cogview4_num_images_slider,
                    self.cogview4_steps_slider,
                    self.cogview4_resolution_dropdown, # Ajout de la résolution aux inputs
                    self.cogview4_guidance_scale_slider,
                ],
                outputs=[
                    self.cogview4_result_output,
                    self.cogview4_progress_html,
                    self.cogview4_bouton_gen,
                    self.cogview4_bouton_stop,
                ],
            )
            self.cogview4_bouton_stop.click(fn=self.stop_generation, inputs=None, outputs=None)
        return tab

    def load_cogview4_model_ui(self):
        """Wrapper UI pour charger le modèle CogView4."""
        yield gr.update(value=translate("cogview4_loading_model", self.module_translations)), gr.update(interactive=False)

        success, message = self.model_manager.load_model(
            model_name=COGVIEW4_MODEL_ID,
            vae_name="Auto", # CogView4Pipeline gère son VAE
            model_type=COGVIEW4_MODEL_TYPE_KEY,
            gradio_mode=True,
            # Spécifier le dtype ici si ModelManager le supporte, sinon il sera appliqué après
            # torch_dtype=torch.bfloat16 # ModelManager devrait idéalement gérer cela
        )

        if success:
            pipe = self.model_manager.get_current_pipe()
            if pipe and isinstance(pipe, CogView4Pipeline):
                try:
                    # Appliquer les configurations spécifiques après le chargement
                    pipe.enable_model_cpu_offload()
                    if hasattr(pipe, 'vae') and pipe.vae is not None:
                        pipe.vae.enable_slicing()
                        pipe.vae.enable_tiling()
                    else:
                        print(txt_color("[AVERTISSEMENT]", "warning"), "Le pipe CogView4 n'a pas d'attribut 'vae' ou VAE est None. Slicing/Tiling VAE non appliqué.")

                    message += f" {translate('cogview4_model_config_applied', self.module_translations)}"
                    gr.Info(translate('cogview4_model_config_applied', self.module_translations))
                except Exception as e_config:
                    config_err_msg = f"Erreur lors de la configuration spécifique de CogView4: {e_config}"
                    print(txt_color("[ERREUR]", "erreur"), config_err_msg)
                    message += f" ({config_err_msg})"
                    gr.Warning(config_err_msg)
            yield gr.update(value=message), gr.update(interactive=True)
        else:
            yield gr.update(value=message), gr.update(interactive=False)

    def update_prompt_from_image(self, image_pil, use_image_flag, global_translations):
        """Génère un prompt si l'image est fournie et le checkbox est coché."""
        if use_image_flag and image_pil is not None:
            print(txt_color("[INFO]", "info"), translate("cogview4_generating_prompt_from_image", self.module_translations))
            # Utiliser les traductions du module pour le message interne de generate_prompt_from_image
            generated_prompt = generate_prompt_from_image(image_pil, self.module_translations)
            if generated_prompt.startswith(f"[{translate('erreur', self.module_translations).upper()}]"):
                gr.Warning(generated_prompt, duration=5.0)
                return gr.update()
            else:
                return gr.update(value=generated_prompt)
        elif not use_image_flag:
            return gr.update()
        return gr.update()

    def cogview4_gen(
        self,
        prompt_libre,
        traduire,
        selected_styles,
        num_images,
        steps,
        resolution_str, # Nouvelle entrée pour la résolution
        guidance_scale  # guidance_scale est maintenant après resolution_str
    ):
        """Génère une ou plusieurs images en utilisant le modèle CogView4 chargé."""
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
        # Afficher la résolution dans la barre de progression
        initial_progress = create_progress_bar_html(0, int(steps), 0, f"{translate('preparation', module_translations)} ({width}x{height})")
        btn_gen_off = gr.update(interactive=False)
        btn_stop_on = gr.update(interactive=True)

        yield initial_gallery, initial_progress, btn_gen_off, btn_stop_on

        pipe = self.model_manager.get_current_pipe()
        if pipe is None or self.model_manager.current_model_type != COGVIEW4_MODEL_TYPE_KEY:
            msg = translate("cogview4_error_no_model", module_translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False)
            return

        if not isinstance(pipe, CogView4Pipeline):
            msg = f"Erreur: Le pipe chargé n'est pas une instance de CogView4Pipeline. Type: {type(pipe)}"
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Error(msg)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False)
            return

        if prompt_libre and prompt_libre.strip():
            base_user_prompt = translate_prompt(prompt_libre, module_translations) if traduire else prompt_libre
        else:
            msg = translate("cogview4_error_no_prompt", module_translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False)
            return

        final_prompt_text, _, style_names_used = styles_fusion(
            selected_styles,
            base_user_prompt,
            "", # CogView4 n'utilise pas de prompt négatif de cette manière
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
            print(txt_color("[INFO]", "info"), f"{translate('cogview4_generation_start', module_translations)} ({image_info_text})")

            while not progress_queue.empty():
                try: progress_queue.get_nowait()
                except queue.Empty: break

            # Pour CogView4, le callback standard de diffusers pour les étapes intermédiaires
            # n'est pas directement applicable de la même manière que pour SDXL.
            # La progression sera donc plus globale ou basée sur le temps.
            # Nous utilisons execute_pipeline_task_async qui gère un callback simple.
            # Le callback pour CogView4 pourrait ne pas fournir d'aperçus latents.
            
            # Création d'un callback simple pour la progression
            # Note: CogView4Pipeline ne supporte pas `callback_on_step_end` de la même manière que SDXL.
            # La progression sera donc plus indicative.
            simple_callback = create_inpainting_callback(self.stop_event, int(steps), module_translations, progress_queue)


            thread, result_container = execute_pipeline_task_async(
                pipe=pipe,
                prompt=final_prompt_text,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                width=width, # Utiliser la largeur sélectionnée
                height=height, # Utiliser la hauteur sélectionnée
                seed=current_seed, # Utilisation du seed pour la reproductibilité si supporté
                device=self.model_manager.device, # Assurer que le device est passé
                stop_event=self.stop_event,
                translations=module_translations,
                progress_queue=progress_queue,
                preview_queue=None, # Pas d'aperçu latent pour CogView4
                # callback_on_step_end=simple_callback, # CogView4Pipeline ne le prend pas en charge
                # Les arguments spécifiques à SDXL comme prompt_embeds sont omis
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
                temps_image_gen = time.time() # Temps pour cette image spécifique
                
                current_time_str = time.strftime("%Y%m%d_%H%M%S")
                style_filename_part = "_".join(style_names_used) if style_names_used else "NoStyle"
                style_filename_part = style_filename_part.replace(" ", "_")[:30]
                output_filename = f"cogview4_{style_filename_part}_{current_time_str}_img{i+1}_{width}x{height}.{self.global_config['IMAGE_FORMAT'].lower()}"
                date_str = datetime.now().strftime("%Y_%m_%d")
                save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str)
                os.makedirs(save_dir, exist_ok=True)
                chemin_image = os.path.join(save_dir, output_filename)

                temps_gen_formatted = f"{(temps_image_gen - start_time_total):.2f}" # Temps depuis le début total

                xmp_data = {
                    "Module": "CogView4",
                    "Creator": self.global_config.get("AUTHOR", "CyberBill"),
                    "Model": COGVIEW4_MODEL_ID,
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
                error_msg = f"{translate('cogview4_error_generation', module_translations)} ({image_info_text}): {result_container.get('error', 'Unknown error')}"
                print(txt_color("[ERREUR]", "erreur"), error_msg)
                final_message = f'<p style="color:red;">{error_msg}</p>'
                gr.Error(error_msg) # Afficher l'erreur dans l'UI Gradio

        if not self.stop_event.is_set():
            temps_total_final = f"{(time.time() - start_time_total):.2f}"
            if int(num_images) > 1:
                final_message = translate("batch_complete", module_translations).format(num_images=num_images, time=temps_total_final)
            else:
                final_message = translate("cogview4_generation_complete", module_translations).format(time=temps_total_final)
            print(txt_color("[OK]", "ok"), final_message)
            gr.Info(final_message, duration=3.0)
        else:
            final_message = translate("generation_arretee", module_translations)

        gc.collect()
        if self.model_manager.device.type == 'cuda':
            torch.cuda.empty_cache()
        yield generated_images_gallery, final_message, gr.update(interactive=True), gr.update(interactive=False)
