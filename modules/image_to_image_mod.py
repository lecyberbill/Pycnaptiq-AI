# i2i_mod
import os
import json
import queue
import threading
# AJOUT: glob pour lister les fichiers plus facilement
import glob
from Utils.utils import (
    txt_color, # Garder txt_color
    translate,
    enregistrer_image,
    preparer_metadonnees_image,
    enregistrer_etiquettes_image_html,
    ImageSDXLchecker,
    styles_fusion,
    create_progress_bar_html,
)
from Utils.callback_diffuser import create_inpainting_callback
from Utils.model_manager import ModelManager # <-- AJOUT
from core.translator import translate_prompt
from datetime import datetime

# JSON associated at this module
module_json_path = os.path.join(os.path.dirname(__file__), "image_to_image_mod.json")


# Create a ManagementModule instance to manage dependencies
with open(module_json_path, "r", encoding="utf-8") as f:
    module_data = json.load(f)


# Now we can do imports safely

import gradio as gr
import time
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
import torch
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
import gc
import traceback # Import traceback for detailed error logging


def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None): # <-- Accepter ModelManager
    """Initialise le module test."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    return Image2imageSDXLModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)  # <-- Passer ModelManager


class Image2imageSDXLModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None): # <-- Accepter ModelManager
        """Initialise la classe Image2imageSDXLModule."""
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance # <-- Stocker ModelManager
        self.Image2ImageModel_path = global_config["MODELS_DIR"]
        self.vae_dir = global_config.get("VAE_DIR", "models/vae")
        self.liste_vaes = ["Auto"] + self.model_manager.list_vaes()
        self.styles = self.load_styles()
        self.gestionnaire = gestionnaire_instance
        self.default_negative_prompt = self.global_config.get("NEGATIVE_PROMPT", "")
        self.current_model_name = None
        self.current_vae_name = "Auto"
        self.stop_event = threading.Event()
        # AJOUT: Extensions d'image supportées pour le batch
        self.supported_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        self.module_translations = {} # Initialize module_translations


    def stop_generation(self, module_translations):
        """Active l'événement pour arrêter la génération en cours."""
        self.stop_event.set()
        print(
            txt_color("[INFO]", "info"),
            translate("stop_requested", module_translations),
        )

    def load_styles(self):
        """Loads styles from styles.json."""
        self.styles_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "styles.json")
        try:
            with open(self.styles_path, "r", encoding="utf-8") as f:
                styles_data = json.load(f)
            # --- AJOUT: Traduire les noms de style ici ---
            for style in styles_data:
                style["name"] = translate(style["key"], self.global_translations)
            # --- FIN AJOUT ---
            return styles_data
        except FileNotFoundError:
            print(
                txt_color("[ERREUR] ", "erreur"),
                f"styles.json not found at {self.styles_path}",
            )
            return []
        except json.JSONDecodeError:
            print(
                txt_color("[ERREUR] ", "erreur"),
                f"Invalid JSON format in styles.json at {self.styles_path}",
            )
            return []

    def create_tab(self, module_translations):
        """
        Crée l'onglet Gradio pour ce module.

        Args:
        module_translations (dict): Le dictionnaire de traductions du module.

        Returns:
        gr.Tab: L'onglet Gradio créé.
        """
        self.module_translations = module_translations # <-- Stocker les traductions spécifiques
        with gr.Tab(translate("i2i_tab_name", self.module_translations)) as tab: # Utiliser self.module_translations
            gr.Markdown(f"## {translate('i2i_tab_title', self.module_translations)}") # Utiliser self.module_translations
            with gr.Row():
                with gr.Column():
                    self.mode_selector = gr.Radio(
                        choices=[
                            translate("mode_single_image", self.module_translations),
                            translate("mode_batch_folder", self.module_translations),
                        ],
                        value=translate("mode_single_image", self.module_translations),
                        label=translate("mode_selection_label", self.module_translations),
                    )

                    self.image_input = gr.Image(
                        label=translate("input_image", self.module_translations),
                        type="pil",
                        visible=True, # Visible par défaut
                        interactive=True,
                    )

                    self.batch_folder_input = gr.Textbox(
                        label=translate("batch_folder_label", self.module_translations),
                        placeholder=translate("batch_folder_placeholder", self.module_translations),
                        visible=False, # Caché par défaut
                        interactive=True,
                    )
                    # +++ AJOUT: Aperçu de l'image en cours (batch) +++
                    self.batch_current_image_preview = gr.Image(
                        label=translate("batch_current_preview_label", self.module_translations), # Nouvelle clé
                        type="pil",
                        visible=False, # Caché par défaut
                        interactive=False, # Non interactif
                        height=200 # Ajustez la hauteur si besoin
                    )
                    # +++ FIN AJOUT +++

                with gr.Column():
                    available_models = self.model_manager.list_models(model_type="standard")
                    no_model_message = translate("aucun_modele_trouve", self.global_translations) # Assurez-vous que c'est bien la clé utilisée par list_models
                    
                    if available_models and available_models[0] != no_model_message:
                        default_model_value = available_models[0]
                    elif available_models and available_models[0] == no_model_message:
                        default_model_value = no_model_message
                    else:
                        default_model_value = None
                    self.modele_i2i_dropdown = gr.Dropdown(
                        label=translate("selectionner_modele", self.module_translations),
                        choices=available_models, 
                        value=default_model_value,
                        allow_custom_value=True,
                    )
                    self.vae_dropdown = gr.Dropdown(
                        label=translate("selectionner_vae", self.module_translations),
                        choices=self.liste_vaes,
                        value="Auto",
                        info=translate("selectionner_vae_info", self.module_translations),
                        allow_custom_value=True,
                    )
                    self.bouton_lister = gr.Button(
                        translate("lister_modeles", self.module_translations)
                    )
                    self.bouton_charger_i2i = gr.Button(
                        translate("charger_modele", self.module_translations)
                    )
                    self.message_chargement_i2i = gr.Textbox(
                        label=translate("statut", self.module_translations),
                        value=translate("aucun_modele_charge", self.module_translations),
                    )

                    self.i2i_prompt_libre = gr.Textbox(
                        label=translate("prompt_libre", self.module_translations),
                        info=translate("prompt_libre_info", self.module_translations),
                        placeholder=translate("prompt_libre_placeholder", self.module_translations),
                    )
                    self.i2i_traduire_checkbox = gr.Checkbox(
                        label=translate("traduire_en_anglais", self.module_translations),
                        value=False,
                        info=translate("traduire_prompt_libre", self.module_translations),
                    )

                    self.style_dropdown = gr.Dropdown(
                        label=translate("selectionner_styles", self.module_translations),
                        # Utiliser les noms traduits chargés dans __init__
                        choices=[style["name"] for style in self.styles],
                        value=[],
                        multiselect=True,
                        info=translate("selectionner_styles_info", self.module_translations),
                    )
                    with gr.Row():
                        self.i2i_progress_html = gr.HTML()
                    with gr.Row():
                        self.step_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=45,
                            step=1,
                            label=translate("etapes", self.module_translations),
                        )
                        self.guidance_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=9,
                            step=0.5,
                            label=translate("guidage", self.module_translations),
                        )
                        self.strength_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.6,
                            step=0.01,
                            label=translate("strength", self.module_translations),
                        )
                        texte_bouton_initial = translate(
                            "charger_modele_pour_commencer", self.module_translations
                        )
                        self.bouton_i2i_gen = gr.Button(
                            value=texte_bouton_initial, interactive=False
                        )
                        self.i2i_bouton_stop = gr.Button(
                            translate("arreter", self.module_translations),
                            interactive=False,
                            variant="stop",
                            scale=1,
                        )

                with gr.Column():
                    # --- MODIFICATION: Sortie en Gallery ---
                    self.result_output = gr.Gallery(
                        label=translate("output_image", self.module_translations),
                        # type="pil" # Gallery gère les images PIL
                    )
                    # --- FIN MODIFICATION ---

            # --- AJOUT: Logique pour afficher/cacher les inputs ET l'aperçu selon le mode ---
            def switch_mode_ui(mode_choice):
                 # Utiliser self.module_translations ici aussi pour la comparaison
                is_single_mode = mode_choice == translate("mode_single_image", self.module_translations)
                is_batch_mode = mode_choice == translate("mode_batch_folder", self.module_translations) # Explicite pour clarté
                return {
                    self.image_input: gr.update(visible=is_single_mode, interactive=is_single_mode),
                    self.batch_folder_input: gr.update(visible=is_batch_mode, interactive=is_batch_mode),
                    # +++ AJOUT: Afficher/cacher l'aperçu batch +++
                    self.batch_current_image_preview: gr.update(visible=is_batch_mode)
                }

            self.mode_selector.change(
                fn=switch_mode_ui,
                inputs=[self.mode_selector],
                # +++ AJOUT: Ajouter l'aperçu aux outputs +++
                outputs=[self.image_input, self.batch_folder_input, self.batch_current_image_preview],
            )
            # --- FIN AJOUT ---

            self.bouton_lister.click(
                fn=self.mettre_a_jour_listes,
                outputs=[self.modele_i2i_dropdown, self.vae_dropdown],
            )

            self.bouton_charger_i2i.click(
                # --- MODIFICATION: Appeler model_manager.load_model ---
                fn=self.charger_modele_i2i_via_manager, # Nouvelle fonction wrapper
                inputs=[self.modele_i2i_dropdown, self.vae_dropdown],
                # --- FIN MODIFICATION ---
                outputs=[
                    self.message_chargement_i2i,
                    self.bouton_i2i_gen,
                    self.bouton_i2i_gen,
                ],
            )
            self.bouton_i2i_gen.click(
                fn=self.image_to_image_gen,
                inputs=[
                    # --- AJOUT: Mode et dossier ---
                    self.mode_selector,
                    self.batch_folder_input,
                    # --- FIN AJOUT ---
                    self.i2i_prompt_libre,
                    self.i2i_traduire_checkbox,
                    self.style_dropdown,
                    self.image_input, # Toujours passé, mais utilisé conditionnellement
                    self.step_slider,
                    self.guidance_slider,
                    self.strength_slider,
                ],
                outputs=[
                    self.result_output,
                    self.i2i_progress_html,
                    self.bouton_i2i_gen,
                    self.i2i_bouton_stop,
                    # +++ AJOUT: Output pour l'aperçu batch +++
                    self.batch_current_image_preview
                ],
            )
            self.i2i_bouton_stop.click(
                fn=self.stop_generation,
                inputs=[gr.State(self.module_translations)], # Passer l'état des traductions
                outputs=None)

        return tab

    def mettre_a_jour_listes(self):
        """Met à jour la liste des modèles et des VAEs disponibles."""
        self.liste_modeles = self.model_manager.list_models( # <-- Utiliser ModelManager
            model_type="standard", # Specify the model type
            gradio_mode=True
        )
        # list_vaes likely takes no arguments based on cyberbill_SDXL.py usage
        self.liste_vaes = ["Auto"] + self.model_manager.list_vaes() # <-- Remove arguments
        return gr.update(choices=self.liste_modeles), gr.update(choices=self.liste_vaes)

    # --- SUPPRESSION des méthodes charger_modele_i2i et charger_modele_i2i_gradio ---

    # --- NOUVELLE méthode wrapper pour appeler ModelManager ---
    def charger_modele_i2i_via_manager(self, nom_fichier, nom_vae):
        """
        Charge un modèle d'image_to_image et VAE, retourne un message pour Gradio
        et met à jour l'état interactif ET le texte du bouton de génération.
        """
        # Utiliser self.module_translations qui est déjà défini
        try:
            # Appeler la méthode load_model du ModelManager
            success, message = self.model_manager.load_model(
                model_name=nom_fichier,
                vae_name=nom_vae,
                model_type="img2img", # Spécifier le type
                gradio_mode=True
            )
        except gr.Error as gr_e: # Intercepter l'erreur Gradio levée par charger_modele_i2i
            message = str(gr_e)
            success = False # Marquer comme échec
            # Le ModelManager devrait gérer son état interne en cas d'erreur
        except Exception as e: # Autres exceptions inattendues
            message = f"{translate('erreur_inattendue_chargement', self.module_translations)}: {e}" # Nouvelle clé
            print(txt_color("[ERREUR]", "erreur"), message)
            traceback.print_exc()
            success = False # Marquer comme échec
            # Le ModelManager devrait gérer son état interne en cas d'erreur

        if success: # Vérifier le succès retourné par le ModelManager
            etat_interactif = True
            texte_bouton = translate("image_to_image_gen", self.module_translations)
        else:
            etat_interactif = False
            texte_bouton = translate(
                "charger_modele_pour_commencer", self.module_translations
            )

        update_interactif = gr.update(interactive=etat_interactif)
        update_texte = gr.update(value=texte_bouton)
        # Mettre à jour les noms courants après chargement réussi
        self.current_model_name = self.model_manager.current_model_name
        self.current_vae_name = self.model_manager.current_vae_name

        return message, update_interactif, update_texte

    # --- MODIFICATION: Fonction image_to_image_gen pour gérer les modes ---
    def image_to_image_gen(
        self,
        mode,
        batch_folder,
        prompt_libre,
        traduire,
        selected_styles,
        single_image,
        step,
        guidance,
        strength,
    ):
        """
        Génère une ou plusieurs images en utilisant le modèle i2i chargé.
        Gère les modes "Image Unique" et "Dossier Batch".
        """
        module_translations = self.module_translations # <-- UTILISER LES TRADUCTIONS STOCKÉES
        valeur_attendue_batch = translate("mode_batch_folder", module_translations)
        is_batch_mode = mode == valeur_attendue_batch

        # --- Placeholders pour les yields ---
        initial_gallery = []
        initial_progress = create_progress_bar_html(0, 100, 0, translate("preparation", module_translations))
        btn_gen_off = gr.update(interactive=False)
        btn_stop_on = gr.update(interactive=True)
        # +++ AJOUT: Placeholder pour l'aperçu +++
        initial_preview = None

        # Désactiver les boutons et vider l'aperçu au début
        yield initial_gallery, initial_progress, btn_gen_off, btn_stop_on, initial_preview


        # --- Vérifications initiales ---
        pipe = self.model_manager.get_current_pipe() # <-- Obtenir pipe
        compel = self.model_manager.get_current_compel() # <-- Obtenir compel
        if pipe is None or compel is None:
            msg = translate("erreur_pas_modele_i2i", module_translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, 4.0)
            # Retourner des updates pour réactiver les boutons
            yield [], "", gr.update(interactive=True), gr.update(interactive=False), None
            return

        # Vérifier le type de modèle chargé
        if self.model_manager.current_model_type != "img2img":
            msg = translate("erreur_mauvais_type_modele_i2i", module_translations) # Nouvelle clé
            print(txt_color("[ERREUR]", "erreur"), msg)
            gr.Warning(msg, 4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False), None
            return

        input_images_paths = [] # Liste pour stocker les chemins (batch) ou l'image PIL (single)
        if is_batch_mode:
            if not batch_folder or not os.path.isdir(batch_folder):
                msg = translate("erreur_dossier_batch_invalide", module_translations)
                print(txt_color("[ERREUR] ", "erreur"), msg)
                gr.Warning(msg, 4.0)
                yield [], "", gr.update(interactive=True), gr.update(interactive=False), None
                return
            # Lister les images dans le dossier
            for ext in self.supported_extensions:
                input_images_paths.extend(glob.glob(os.path.join(batch_folder, f"*{ext}")))
            if not input_images_paths:
                msg = translate("erreur_aucune_image_dossier", module_translations)
                print(txt_color("[ERREUR] ", "erreur"), msg)
                gr.Warning(msg, 4.0)
                yield [], "", gr.update(interactive=True), gr.update(interactive=False), None
                return
            print(txt_color("[INFO]", "info"), f"{translate('batch_images_trouvees', module_translations)}: {len(input_images_paths)}")
        else:
            if single_image is None:
                msg = translate("erreur_pas_image", module_translations)
                print(txt_color("[ERREUR] ", "erreur"), msg)
                gr.Warning(msg, 4.0)
                yield [], "", gr.update(interactive=True), gr.update(interactive=False), None
                return
            # En mode single, on simule une liste d'une seule image
            input_images_paths = [single_image] # La liste contient l'objet PIL directement

        # --- Initialisation avant la boucle ---
        start_time_total = time.time()
        self.stop_event.clear()
        generated_images_gallery = [] # Pour accumuler les images pour la galerie
        total_images_to_process = len(input_images_paths)
        final_message = ""

        # Désactiver les boutons pendant la génération (déjà fait avec le premier yield)
        # yield generated_images_gallery, create_progress_bar_html(0, 100, 0, translate("preparation", module_translations)), gr.update(interactive=False), gr.update(interactive=True), None

        try:
            # --- Traitement du Prompt (fait une seule fois) ---
            base_user_prompt = ""
            if prompt_libre and prompt_libre.strip():
                print(
                    txt_color("[INFO]", "info"),
                    translate("utilisation_prompt_libre", module_translations),
                )
                base_user_prompt = (
                    translate_prompt(prompt_libre, module_translations)
                    if traduire
                    else prompt_libre
                )

            final_prompt_text, final_negative_prompt, style_names_used = styles_fusion(
                selected_styles,
                base_user_prompt,
                self.default_negative_prompt,
                self.styles,
                module_translations,
            )

            if not final_prompt_text.strip():
                msg = translate("erreur_aucun_prompt_final", module_translations)
                print(txt_color("[AVERTISSEMENT]", "erreur"), msg)
                gr.Warning(msg, 4.0)
                raise ValueError(msg)

            # Utiliser Compel avec le prompt final (fait une seule fois)
            conditioning, pooled = compel(final_prompt_text) # <-- Utiliser compel
            neg_conditioning, neg_pooled = compel( # <-- Utiliser compel
                final_negative_prompt
            )
            # --- Fin Traitement Prompt ---

            # --- Boucle Principale (Batch ou Single) ---
            for idx, image_source in enumerate(input_images_paths):
                if self.stop_event.is_set():
                    final_message = translate("generation_arretee", module_translations)
                    print(txt_color("[INFO]", "info"), final_message)
                    gr.Info(final_message, 3.0)
                    break

                start_time_image = time.time()
                current_progress_html = ""
                final_image_container = {}
                progress_queue = queue.Queue()
                # +++ AJOUT: Placeholder pour l'aperçu de cette itération +++
                current_preview_update = None

                try:
                    # Charger l'image si c'est un chemin (mode batch)
                    if is_batch_mode:
                        current_filename = os.path.basename(image_source)
                        print(txt_color("[INFO]", "info"), f"{translate('batch_processing_image', module_translations)} {idx+1}/{total_images_to_process}: {current_filename}")
                        current_input_image = Image.open(image_source).convert("RGB")
                        
                        target_preview_height = 200 # Hauteur cible (doit correspondre à gr.Image)
                        w_orig, h_orig = current_input_image.size
                        resized_preview_image = current_input_image # Par défaut, garder l'original

                        if h_orig > target_preview_height: # Redimensionner seulement si l'image est plus haute
                            ratio = target_preview_height / h_orig
                            w_new = int(w_orig * ratio)
                            try:
                                # Utiliser Image.Resampling.LANCZOS pour Pillow >= 9.0.0
                                resampling_filter = Image.Resampling.LANCZOS
                            except AttributeError:
                                # Utiliser Image.LANCZOS pour les versions plus anciennes
                                resampling_filter = Image.LANCZOS
                            print (txt_color("[INFO]", "info"), f"{translate('Resizing_preview_from', module_translations)} {w_orig}x{h_orig} to {w_new}x{target_preview_height}" )
                            resized_preview_image = current_input_image.resize((w_new, target_preview_height), resampling_filter)
                        # Mettre à jour l'aperçu avec l'image redimensionnée
                        current_preview_update = gr.update(value=resized_preview_image)
                    else:
                        current_input_image = image_source
                        current_filename = f"single_image_{idx}"
                        # Pas d'aperçu séparé en mode single

                    # +++ AJOUT: Premier yield avec l'aperçu mis à jour (si batch) +++
                    if is_batch_mode:
                         yield generated_images_gallery, "", btn_gen_off, btn_stop_on, current_preview_update


                    # Redimensionnement de l'image si besoin
                    image_checker = ImageSDXLchecker(current_input_image, module_translations)
                    image_resized = image_checker.redimensionner_image()

                    actual_steps = max(1, int(step * strength))
                    generator = torch.Generator(device=self.model_manager.device).manual_seed(
                        int(time.time()) + idx
                    )

                    callback = create_inpainting_callback(
                        self.stop_event, actual_steps, module_translations, progress_queue
                    )

                    # --- Thread pour l'inférence ---
                    def run_pipeline_inner(img_in):
                        try:
                            result = pipe( # <-- Utiliser pipe
                                pooled_prompt_embeds=pooled,
                                prompt_embeds=conditioning,
                                negative_prompt_embeds=neg_conditioning,
                                negative_pooled_prompt_embeds=neg_pooled,
                                image=img_in,
                                num_inference_steps=step,
                                guidance_scale=guidance,
                                strength=strength,
                                generator=generator,
                                callback_on_step_end=callback,
                            )
                            if not self.stop_event.is_set():
                                final_image_container["final"] = result.images[0]
                        except Exception as e_inner:
                            if not self.stop_event.is_set():
                                print(
                                    txt_color("[ERREUR]", "erreur"),
                                    f"Erreur dans run_pipeline_inner (i2i): {e_inner}",
                                )
                                final_image_container["error"] = e_inner

                    thread = threading.Thread(target=run_pipeline_inner, args=(image_resized,))
                    thread.start()

                    # --- Boucle de progression pour l'image actuelle ---
                    last_progress_html_inner = ""
                    while thread.is_alive() or not progress_queue.empty():
                        current_step_prog, total_steps_prog = None, actual_steps
                        while not progress_queue.empty():
                            try:
                                current_step_prog, total_steps_prog = progress_queue.get_nowait()
                            except queue.Empty:
                                break

                        if current_step_prog is not None:
                            progress_percent = int((current_step_prog / total_steps_prog) * 100)
                            batch_info_text = f"{translate('batch_progress_image', module_translations)} {idx+1}/{total_images_to_process}"
                            last_progress_html_inner = create_progress_bar_html(
                                current_step=current_step_prog,
                                total_steps=total_steps_prog,
                                progress_percent=progress_percent,
                                text_info=batch_info_text
                            )
                            # Yield la progression, garder l'aperçu actuel
                            yield generated_images_gallery, last_progress_html_inner, btn_gen_off, btn_stop_on, gr.update()

                        time.sleep(0.05)
                    # --- Fin boucle de progression ---

                    thread.join()

                    # Vérifier les erreurs ou l'arrêt après le thread
                    if "error" in final_image_container:
                        error_msg = f"{translate('erreur_image_to_image', module_translations)} ({current_filename}): {final_image_container['error']}"
                        print(txt_color("[ERREUR]", "erreur"), error_msg)
                        current_progress_html = f'<p style="color:red;">{error_msg}</p>'
                        yield generated_images_gallery, current_progress_html, btn_gen_off, btn_stop_on, gr.update()
                        continue

                    if self.stop_event.is_set():
                        final_message = translate("generation_arretee", module_translations)
                        print(txt_color("[INFO]", "info"), final_message)
                        gr.Info(final_message, 3.0)
                        break

                    result_image = final_image_container.get("final")
                    if result_image is None:
                        msg = f"{translate('erreur_pas_image_genere', module_translations)} ({current_filename})"
                        print(txt_color("[ERREUR]", "erreur"), msg)
                        current_progress_html = f'<p style="color:red;">{msg}</p>'
                        yield generated_images_gallery, current_progress_html, btn_gen_off, btn_stop_on, gr.update()
                        continue

                    # --- Succès pour cette image ---
                    generated_images_gallery.append(result_image)
                    temps_generation_image = f"{(time.time() - start_time_image):.2f} sec"
                    current_time_str = time.strftime("%Y%m%d_%H%M%S")
                    width, height = result_image.size
                    style_filename_part = "_".join(style_names_used) if style_names_used else "NoStyle"
                    style_filename_part = style_filename_part.replace(" ", "_")[:30]

                    # --- MODIFICATION POUR LE NOM DE FICHIER EN BATCH ---
                    if is_batch_mode:
                        # Utiliser l'index du batch au lieu du nom de fichier original
                        output_filename = f"i2i_batch_{idx+1}_{style_filename_part}_{current_time_str}_{height}x{width}.{self.global_config['IMAGE_FORMAT'].lower()}"
                    else:
                        # Garder la logique existante pour le mode single
                        base_filename_part = os.path.splitext(current_filename)[0] # current_filename est "single_image_0" ici
                        output_filename = f"i2i_{base_filename_part}_{style_filename_part}_{current_time_str}_{height}x{width}.{self.global_config['IMAGE_FORMAT'].lower()}"
                    # --- FIN MODIFICATION ---

                    date_str = datetime.now().strftime("%Y_%m_%d")
                    save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str)
                    os.makedirs(save_dir, exist_ok=True)
                    chemin_image = os.path.join(save_dir, output_filename)

                    enregistrer_image(
                        result_image,
                        chemin_image,
                        self.global_translations,
                        self.global_config["IMAGE_FORMAT"],
                    )
                    xmp_data = {
                        "Module": "Image to Image SDXL",
                        "Creator": self.global_config.get("AUTHOR", "CyberBill"), # Utiliser .get avec fallback
                        "Model": self.model_manager.current_model_name, # <-- Utiliser ModelManager
                        "VAE": self.current_vae_name,
                        "Steps": step,
                        "Guidance": guidance,
                        "Styles": ", ".join(style_names_used) if style_names_used else "None",
                        "Prompt": final_prompt_text,
                        "Negative prompt:": final_negative_prompt,
                        "Strength": strength,
                        "Size": f"{width} x {height}",
                        "Generation Time": temps_generation_image,
                        "Original File": current_filename if is_batch_mode else "N/A"
                    }

                    metadata_structure, prep_message = preparer_metadonnees_image(
                        result_image,
                        xmp_data,
                        self.global_translations, # Utiliser les traductions globales
                        chemin_image # Passer le chemin pour déterminer le format
                    )
                    # Afficher le message de la préparation (succès ou échec)
                    print(txt_color("[INFO]", "info"), prep_message)

                    # --- Sauvegarde de l'image avec les métadonnées intégrées ---
                    enregistrer_image(
                        result_image,
                        chemin_image,
                        self.global_translations, # Traductions pour les messages d'enregistrement
                        self.global_config["IMAGE_FORMAT"].upper(), # Passer le format déterminé plus haut
                        metadata_to_save=metadata_structure # Passer la structure préparée
                    )

                    enregistrer_etiquettes_image_html(
                        chemin_image, xmp_data, module_translations, True
                    )

                    print(
                        txt_color("[OK]", "ok"),
                        f"Image {idx+1}/{total_images_to_process} générée et sauvegardée: {output_filename}",
                    )
                    # Yield la galerie mise à jour, vider la progression, garder l'aperçu
                    yield generated_images_gallery, "", btn_gen_off, btn_stop_on, gr.update()

                except FileNotFoundError as e_fnf:
                    error_msg = f"{translate('erreur_fichier_introuvable', module_translations)} ({image_source}): {e_fnf}"
                    print(txt_color("[ERREUR]", "erreur"), error_msg)
                    current_progress_html = f'<p style="color:red;">{error_msg}</p>'
                    yield generated_images_gallery, current_progress_html, btn_gen_off, btn_stop_on, gr.update()
                    continue
                except Exception as e_img:
                    error_msg = f"{translate('erreur_traitement_image', module_translations)} ({current_filename}): {e_img}"
                    print(txt_color("[ERREUR]", "erreur"), error_msg)
                    traceback.print_exc() # Ajouter trace pour debug
                    current_progress_html = f'<p style="color:red;">{error_msg}</p>'
                    yield generated_images_gallery, current_progress_html, btn_gen_off, btn_stop_on, gr.update()
                    continue

            # --- Fin de la boucle principale ---
            if not final_message:
                temps_total = f"{(time.time() - start_time_total):.2f} sec"
                final_message = f"{translate('generation_terminee_en', module_translations)} {temps_total}"
                print(txt_color("[OK]", "ok"), final_message)
                gr.Info(final_message, 3.0)

        except Exception as e_global:
            error_message = f"{translate('erreur_image_to_image', module_translations)}: {e_global}"
            print(txt_color("[ERREUR] ", "erreur"), error_message)
            traceback.print_exc()
            final_message = f'<p style="color:red;">{error_message}</p>'
            if not generated_images_gallery: generated_images_gallery = []

        finally:
            # Réactiver les boutons et vider l'aperçu à la fin
            final_preview_clear = gr.update(value=None)
            yield generated_images_gallery, final_message, gr.update(interactive=True), gr.update(interactive=False), final_preview_clear # <-- Utiliser pipe
            if hasattr(pipe, '_interrupt'):
                pipe._interrupt = False
            gc.collect()
            torch.cuda.empty_cache()
    # --- FIN MODIFICATION ---
