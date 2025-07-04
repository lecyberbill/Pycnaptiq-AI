# sana_sprint_mod.py
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
# from compel import Compel, ReturnedEmbeddingsType # Compel n'est plus utilisé ici
from diffusers import SanaSprintPipeline # Import spécifique

from Utils.utils import (
    txt_color,
    translate,
    enregistrer_image,
    preparer_metadonnees_image,
    enregistrer_etiquettes_image_html,
    styles_fusion,
    create_progress_bar_html,
)
# Utiliser le callback standard pour la progression et l'aperçu (même si Sana n'a pas d'aperçu latent)
from Utils.callback_diffuser import create_inpainting_callback
from Utils.model_manager import ModelManager
from core.translator import translate_prompt
from core.pipeline_executor import execute_pipeline_task_async # Utiliser l'exécuteur asynchrone
from core.image_prompter import generate_prompt_from_image # <-- Importer la fonction

# --- Configuration et Constantes ---
MODULE_NAME = "sana_sprint"
# --- SUPPRESSION: Le dictionnaire SANA_MODELS sera initialisé dans la classe pour permettre la traduction ---
SANA_MODEL_TYPE_KEY = "sana_sprint" # Clé générique pour ce type de modèle
FIXED_OUTPUT_SIZE = 1024 # Taille de sortie fixe pour Sana Sprint

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
    """Initialise le module Sana Sprint."""
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME)}")
    return SanaSprintModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class SanaSprintModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance, global_config=None):
        """Initialise la classe SanaSprintModule."""
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.styles = self.load_styles()
        self.gestionnaire = gestionnaire_instance
        self.default_negative_prompt = self.global_config.get("NEGATIVE_PROMPT", "")
        self.stop_event = threading.Event()
        self.module_translations = {} # Initialiser

        # --- AJOUT: Initialisation du dictionnaire de modèles traduisibles ---
        self.sana_models = {
            translate("sana_model_0_6b_fast", self.global_translations): "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            translate("sana_model_1_6b_quality", self.global_translations): "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
        }
        # --- FIN AJOUT ---
        # --- AJOUT: Gérer le pipeline Sana localement pour éviter les conflits ---
        self.pipe = None
        self.models_loaded = False
        # --- AJOUT: Suivre le modèle actuellement chargé ---
        self.current_sana_model_id = None
        self.device = model_manager_instance.device if model_manager_instance else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def load_styles(self):
        """Charge les styles depuis styles.json."""
        styles_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "styles.json")
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                styles_data = json.load(f)
            # Traduire les noms de style
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
        self.module_translations = module_translations # Stocker les traductions spécifiques

        with gr.Tab(translate("sana_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('sana_tab_title', self.module_translations)}")
            with gr.Row():
                with gr.Column(scale=2): # Colonne de gauche plus large
                    self.sana_prompt = gr.Textbox(
                        label=translate("sana_prompt_label", self.module_translations),
                        info=translate("sana_prompt_info", self.module_translations),
                        placeholder=translate("sana_prompt_placeholder", self.module_translations),
                        lines=3,
                    )
                    self.sana_traduire_checkbox = gr.Checkbox(
                        label=translate("traduire_en_anglais", self.module_translations), 
                        value=False,
                        info=translate("traduire_prompt_libre", self.module_translations), 
                    )
                    self.sana_style_dropdown = gr.Dropdown(
                        label=translate("selectionner_styles", self.module_translations), 
                        choices=[style["name"] for style in self.styles],
                        value=[],
                        multiselect=True,
                        info=translate("selectionner_styles_info", self.module_translations), 
                    )
                    # --- AJOUT: Checkbox et Image pour prompt depuis image ---
                    self.sana_use_image_prompt_checkbox = gr.Checkbox(
                        label=translate("generer_prompt_image", self.global_translations), # Réutiliser clé globale
                        value=False
                    )
                    self.sana_image_input_for_prompt = gr.Image(label=translate("telechargez_image", self.global_translations), type="pil", visible=False) # Réutiliser clé globale
                    # --- FIN AJOUT ---
                    # --- MODIFICATION: Remplacer slider étapes par slider nombre d'images ---
                    with gr.Row():
                        self.sana_num_images_slider = gr.Slider(
                            minimum=1,
                            maximum=200, # Nombre max d'images
                            value=1,     # Défaut à 1 image
                            step=1,
                            label=translate("nombre_images", self.module_translations),  
                            interactive=True
                        )
                    with gr.Row():
                        self.sana_bouton_gen = gr.Button(
                            value=translate("sana_generate_button", self.module_translations), interactive=False # Désactivé au début
                        )
                        self.sana_bouton_stop = gr.Button(
                            translate("arreter", self.module_translations),  
                            interactive=False,
                            variant="stop",
                        )
                    self.sana_progress_html = gr.HTML()

                with gr.Column(scale=1): # Colonne de droite
                    self.sana_message_chargement = gr.Textbox(
                        label=translate("sana_model_status", self.module_translations),
                        value=translate("sana_model_not_loaded", self.module_translations),
                        interactive=False,
                    )
                    # --- AJOUT: Sélecteur de modèle ---
                    self.sana_model_selector = gr.Radio(
                        choices=list(self.sana_models.keys()), # --- MODIFICATION: Utiliser self.sana_models ---
                        value=list(self.sana_models.keys())[0], # Défaut sur le premier (0.6B)
                        label=translate("sana_model_version_label", self.module_translations) # Nouvelle clé de traduction
                    )
                    self.sana_bouton_charger = gr.Button(
                        translate("sana_load_button", self.module_translations)
                    )
                    self.sana_result_output = gr.Gallery(
                        label=translate("output_image", self.module_translations), # Réutiliser clé globale
                    )

            # --- Logique pour afficher/cacher l'input image ---
            self.sana_use_image_prompt_checkbox.change(
                fn=lambda use_image: gr.update(visible=use_image),
                inputs=self.sana_use_image_prompt_checkbox,
                outputs=self.sana_image_input_for_prompt
            )
            # --- AJOUT: Déclencher la génération de prompt sur changement d'image ---
            self.sana_image_input_for_prompt.change(
                fn=self.update_prompt_from_image,
                # Inputs: l'image, l'état du checkbox, les traductions globales
                inputs=[self.sana_image_input_for_prompt, self.sana_use_image_prompt_checkbox, gr.State(self.global_translations)],
                outputs=self.sana_prompt # Met à jour le textbox du prompt
            )
            # --- FIN AJOUT ---
            # --- Logique des boutons ---
            self.sana_bouton_charger.click(
                fn=self.load_sana_model_ui,
                inputs=[self.sana_model_selector], # --- MODIFICATION: Passer le sélecteur en input ---
                # --- MODIFICATION: Mettre aussi à jour le bouton de chargement lui-même ---
                outputs=[self.sana_message_chargement, self.sana_bouton_gen, self.sana_bouton_charger],
            )

            # --- AJOUT: Réactiver le bouton de chargement si on change de modèle ---
            self.sana_model_selector.change(
                fn=self.on_model_selection_change,
                inputs=[self.sana_model_selector],
                outputs=[self.sana_bouton_charger]
            )
            # --- FIN AJOUT ---

            self.sana_bouton_gen.click(
                fn=self.sana_sprint_gen,
                inputs=[
                    self.sana_prompt,
                    self.sana_traduire_checkbox,
                    self.sana_style_dropdown,
                    self.sana_num_images_slider, # Utiliser le nouveau slider
                    # --- SUPPRIMÉ: Ces entrées ne sont plus nécessaires pour la génération ---
                    # self.sana_use_image_prompt_checkbox,
                    # self.sana_image_input_for_prompt,
                ],
                outputs=[
                    self.sana_result_output,
                    self.sana_progress_html,
                    self.sana_bouton_gen,
                    self.sana_bouton_stop,
                ],
            )
            self.sana_bouton_stop.click(fn=self.stop_generation, inputs=None, outputs=None)

        return tab

    def load_sana_model_ui(self, selected_model_name):
        """Charge le pipeline Sana Sprint de manière isolée pour éviter les conflits de composants."""
        # --- MODIFICATION: Logique de chargement améliorée ---
        model_id_to_load = self.sana_models.get(selected_model_name) # --- MODIFICATION: Utiliser self.sana_models ---
        if not model_id_to_load:
            error_msg = f"Nom de modèle inconnu: {selected_model_name}"
            yield gr.update(value=error_msg), gr.update(interactive=False), gr.update(interactive=True)
            return

        if self.models_loaded and self.current_sana_model_id == model_id_to_load:
            msg = translate("sana_model_already_loaded", self.module_translations)
            yield gr.update(value=msg), gr.update(interactive=True), gr.update(interactive=False)
            return

        # Décharger le modèle principal s'il existe pour libérer de la VRAM
        if self.model_manager.get_current_pipe() is not None or (self.models_loaded and self.current_sana_model_id != model_id_to_load):
            unload_msg = translate("unloading_main_model_before_sana", self.module_translations)
            print(txt_color("[INFO]", "info"), unload_msg)
            yield gr.update(value=unload_msg), gr.update(interactive=False), gr.update(interactive=False)
            self.model_manager.unload_model(gradio_mode=False)
            print(txt_color("[OK]", "ok"), translate("main_model_unloaded_sana", self.module_translations))

        loading_msg = translate("sana_loading_model", self.module_translations)
        print(txt_color("[INFO]", "info"), loading_msg)
        yield gr.update(value=loading_msg), gr.update(interactive=False), gr.update(interactive=False)

        try:
            # --- AJOUT: Déterminer le dtype optimal pour la stabilité numérique ---
            # Le modèle 1.6B peut produire des NaN (résultant en une image noire) en float16.
            # bfloat16 est recommandé pour la stabilité s'il est supporté.
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
                print(txt_color("[INFO]", "info"), "Le GPU supporte bfloat16. Utilisation pour une meilleure stabilité.")
            else:
                compute_dtype = torch.float16
                print(txt_color("[AVERTISSEMENT]", "info"), "Le GPU ne supporte pas bfloat16. Utilisation de float16 (peut être instable avec le modèle 1.6B).")
            # Charger le pipeline complet directement garantit la compatibilité de tous ses composants (VAE, Transformer, etc.)
            self.pipe = SanaSprintPipeline.from_pretrained(
                model_id_to_load, # Utiliser l'ID du modèle sélectionné
                torch_dtype=compute_dtype
            )
            self.pipe.to(self.device)
            self.models_loaded = True
            self.current_sana_model_id = model_id_to_load # Stocker l'ID du modèle chargé
            
            success_msg = translate("sana_model_loaded_success", self.module_translations)
            print(txt_color("[OK]", "ok"), success_msg)
            # UI: message de succès, activer le bouton de génération, désactiver le bouton de chargement
            yield gr.update(value=success_msg), gr.update(interactive=True), gr.update(interactive=False)

        except Exception as e:
            self.pipe = None
            self.models_loaded = False
            self.current_sana_model_id = None
            error_msg = f"{translate('sana_error_loading_model', self.module_translations)}: {e}"
            print(txt_color("[ERREUR]", "erreur"), error_msg)
            traceback.print_exc()
            # UI: message d'erreur, désactiver le bouton de génération, réactiver le bouton de chargement
            yield gr.update(value=error_msg), gr.update(interactive=False), gr.update(interactive=True)

    # --- AJOUT: Fonction pour mettre à jour le prompt depuis l'image ---
    def update_prompt_from_image(self, image_pil, use_image_flag, global_translations):
        """Génère un prompt si l'image est fournie et le checkbox est coché."""
        if use_image_flag and image_pil is not None:
            print(txt_color("[INFO]", "info"), translate("sana_generating_prompt_from_image", self.module_translations))
            generated_prompt = generate_prompt_from_image(image_pil, global_translations)
            # Vérifier si la génération a échoué
            if generated_prompt.startswith(f"[{translate('erreur', global_translations).upper()}]"):
                gr.Warning(generated_prompt, duration=5.0)
                return gr.update() # Ne pas modifier le prompt en cas d'erreur
            else:
                # Mettre à jour le textbox avec le prompt généré
                return gr.update(value=generated_prompt)
        elif not use_image_flag:
            # Si la case n'est pas cochée, ne rien faire (ne pas écraser le prompt existant)
            return gr.update()
        # Si la case est cochée mais l'image est effacée (image_pil is None), ne rien faire non plus pour l'instant.
        return gr.update()
    # --- FIN AJOUT ---

    # --- AJOUT: Gérer l'interactivité du bouton de chargement ---
    def on_model_selection_change(self, selected_model_name):
        """Réactive le bouton de chargement si un modèle différent est sélectionné."""
        selected_model_id = self.sana_models.get(selected_model_name) # --- MODIFICATION: Utiliser self.sana_models ---

        # Si un modèle est chargé et qu'il est différent de celui sélectionné, on active le bouton
        if self.models_loaded and self.current_sana_model_id != selected_model_id:
            return gr.update(interactive=True)
        # Si aucun modèle n'est chargé, le bouton doit être actif
        elif not self.models_loaded:
            return gr.update(interactive=True)
        # Sinon (modèle chargé est le même que celui sélectionné), le bouton reste inactif
        else:
            return gr.update(interactive=False)
    # --- FIN AJOUT ---
    def sana_sprint_gen(
        self,
        prompt_libre,
        traduire,
        selected_styles,
        num_images, # Renommer l'input
        # --- SUPPRIMÉ: Ces paramètres ne sont plus utilisés ici ---
        # use_image_for_prompt,
        # image_for_prompt,
    ):
        """Génère une ou plusieurs images en utilisant le modèle Sana Sprint chargé."""
        module_translations = self.module_translations
        start_time_total = time.time()
        self.stop_event.clear()

        steps = 2 # Étapes fixes pour Sana Sprint
        initial_gallery = []
        initial_progress = create_progress_bar_html(0, steps, 0, translate("preparation", module_translations))
        btn_gen_off = gr.update(interactive=False)
        btn_stop_on = gr.update(interactive=True)

        yield initial_gallery, initial_progress, btn_gen_off, btn_stop_on

        # --- Vérifications ---
        # --- MODIFICATION: Utiliser le pipeline géré localement ---
        if not self.models_loaded or self.pipe is None:
            msg = translate("sana_error_no_model", module_translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False)
            return
        
        pipe = self.pipe
        # --- FIN MODIFICATION ---
        
        # --- Préparation du Prompt (simplifié) ---
        # Utiliser directement le prompt du champ texte
        if prompt_libre and prompt_libre.strip():
             base_user_prompt = translate_prompt(prompt_libre, module_translations) if traduire else prompt_libre
        else:
            # Si le champ texte est vide
            msg = translate("sana_error_no_prompt", module_translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, duration=4.0)
            yield [], "", gr.update(interactive=True), gr.update(interactive=False)
            return

        # --- MODIFICATION: Ignorer le prompt négatif retourné par styles_fusion ---
        final_prompt_text, _, style_names_used = styles_fusion(
            selected_styles,
            base_user_prompt,
            "", # Passer une chaîne vide comme base négative
            self.styles,
            module_translations,
        )

        # --- Préparation pour l'exécution asynchrone ---
        progress_queue = queue.Queue()
        # preview_queue = queue.Queue() # L'aperçu est désactivé
        base_seed = int(time.time()) # Seed de base basé sur le temps

        # --- MODIFICATION: Ne PAS utiliser Compel pour Sana Sprint ---
        # Le code Compel a été supprimé

        # --- MODIFICATION: Boucle pour générer plusieurs images ---
        generated_images_gallery = []
        final_message = ""

        for i in range(num_images):
            if self.stop_event.is_set():
                final_message = translate("generation_arretee", module_translations)
                print(txt_color("[INFO]", "info"), final_message)
                gr.Info(final_message, 3.0)
                break

            current_seed = base_seed + i # Seed unique pour chaque image
            image_info_text = f"{translate('image', module_translations)} {i+1}/{num_images}" # Info pour la barre de progression
            print(txt_color("[INFO]", "info"), f"{translate('sana_generation_start', module_translations)} ({image_info_text})")

            # Vider la queue de progression avant chaque image
            while not progress_queue.empty():
                try: progress_queue.get_nowait()
                except queue.Empty: break

            thread, result_container = execute_pipeline_task_async(
                pipe=pipe,
                num_inference_steps=steps, # Toujours 2
                prompt=final_prompt_text, # Passer le texte brut
                negative_prompt=None, # Ne pas passer de négatif
                prompt_embeds=None, # Explicitement None
                pooled_prompt_embeds=None,
                negative_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                guidance_scale=4.5, # Sana n'utilise probablement pas guidance
                seed=current_seed, # Utiliser le seed courant
                width=FIXED_OUTPUT_SIZE,
                height=FIXED_OUTPUT_SIZE,
                device=self.model_manager.device,
                stop_event=self.stop_event,
                translations=module_translations,
                progress_queue=progress_queue,
                preview_queue=None # --- MODIFICATION: Désactiver l'aperçu pour Sana ---
            )

            # --- Boucle de Progression pour l'image actuelle ---
            last_progress_html = ""
            while thread.is_alive() or not progress_queue.empty():
                # Traiter la progression
                while not progress_queue.empty():
                    try:
                        current_step_prog, total_steps_prog = progress_queue.get_nowait()
                        progress_percent = int((current_step_prog / total_steps_prog) * 100)
                        # Ajouter l'info de l'image actuelle
                        step_info_text = f"Step {current_step_prog}/{total_steps_prog}"
                        last_progress_html = create_progress_bar_html(
                            current_step=current_step_prog,
                            total_steps=total_steps_prog,
                            progress_percent=progress_percent,
                            text_info=f"{image_info_text} - {step_info_text}" # Texte combiné
                        )
                        yield generated_images_gallery, last_progress_html, btn_gen_off, btn_stop_on
                    except queue.Empty:
                        break
                time.sleep(0.05)

            thread.join() # Attendre la fin du thread pour cette image

            # --- Traitement du Résultat pour l'image actuelle ---
            if result_container["status"] == "success" and result_container["final"]:
                result_image = result_container["final"]
                generated_images_gallery.append(result_image)
                temps_image = f"{(time.time() - start_time_total):.2f}" # Temps depuis le début total pour simplifier
                print(txt_color("[OK]", "ok"), f"{translate('image', module_translations)} {i+1}/{num_images} {translate('generer_en', module_translations)} {temps_image} sec")

                # --- Sauvegarde ---
                current_time_str = time.strftime("%Y%m%d_%H%M%S")
                style_filename_part = "_".join(style_names_used) if style_names_used else "NoStyle"
                style_filename_part = style_filename_part.replace(" ", "_")[:30]
                # Ajouter l'index de l'image au nom de fichier
                output_filename = f"sana_{style_filename_part}_{current_time_str}_img{i+1}_{FIXED_OUTPUT_SIZE}x{FIXED_OUTPUT_SIZE}.{self.global_config['IMAGE_FORMAT'].lower()}"
                date_str = datetime.now().strftime("%Y_%m_%d")
                save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str)
                # Créer le répertoire s'il n'existe pas
                os.makedirs(save_dir, exist_ok=True)
                chemin_image = os.path.join(save_dir, output_filename)

                xmp_data = {
                    "Module": "Sana Sprint",
                    "Creator": self.global_config.get("AUTHOR", "CyberBill"),
                    "Model": self.current_sana_model_id, # --- MODIFICATION: Utiliser le modèle actuellement chargé ---
                    "Steps": steps, # Toujours 2
                    "Styles": ", ".join(style_names_used) if style_names_used else "None",
                    "Prompt": final_prompt_text,
                    "Size": f"{FIXED_OUTPUT_SIZE} x {FIXED_OUTPUT_SIZE}",
                    "Generation Time": f"{temps_image} sec", # Temps approximatif pour cette image
                    "Seed": current_seed # Utiliser le seed courant
                }
                metadata_structure, prep_message = preparer_metadonnees_image(result_image, xmp_data, self.global_translations, chemin_image)
                print(txt_color("[INFO]", "info"), prep_message)
                # Sauvegarder l'image et le HTML pour chaque image
                enregistrer_image(result_image, chemin_image, self.global_translations, self.global_config["IMAGE_FORMAT"].upper(), metadata_to_save=metadata_structure)
                enregistrer_etiquettes_image_html(chemin_image, xmp_data, module_translations, is_last_image=(i == num_images - 1)) # Indiquer si c'est la dernière

                # Mettre à jour la galerie après chaque image réussie
                yield generated_images_gallery, last_progress_html, btn_gen_off, btn_stop_on

            elif result_container["status"] == "stopped":
                # L'arrêt a déjà été géré par la vérification au début de la boucle
                break
            else: # Erreur
                error_msg = f"{translate('sana_error_generation', module_translations)} ({image_info_text}): {result_container.get('error', 'Unknown error')}"
                print(txt_color("[ERREUR]", "erreur"), error_msg)
                final_message = f'<p style="color:red;">{error_msg}</p>'
                gr.Error(error_msg)
        # --- Fin de la boucle ---

        # Message final après la boucle (si non arrêté)
        if not self.stop_event.is_set():
            temps_total_final = f"{(time.time() - start_time_total):.2f}"
            if num_images > 1:
                # Assurez-vous que la clé 'batch_complete' existe dans vos traductions
                final_message = translate("batch_complete", module_translations).format(num_images=num_images, time=temps_total_final)
            else:
                final_message = translate("sana_generation_complete", module_translations).format(time=temps_total_final)
            print(txt_color("[OK]", "ok"), final_message)
            gr.Info(final_message, duration=3.0)
        else:
            # Le message d'arrêt a déjà été affiché
            final_message = translate("generation_arretee", module_translations)

        # --- Nettoyage et état final UI ---
        gc.collect()
        if self.model_manager.device.type == 'cuda':
            torch.cuda.empty_cache()

        yield generated_images_gallery, final_message, gr.update(interactive=True), gr.update(interactive=False)
