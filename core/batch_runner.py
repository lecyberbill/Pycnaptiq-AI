# core/batch_runner.py
import os
import json
import time
import random
import queue
import threading
import traceback # Assurez-vous que traceback est importé
from datetime import datetime
from collections import defaultdict

import torch
import gradio as gr
from PIL import Image

# --- Importations depuis les autres modules/utils ---
# Ajustez les chemins relatifs si nécessaire

from Utils.utils import (
    translate, txt_color, create_progress_bar_html,
    preparer_metadonnees_image, enregistrer_image,
    enregistrer_etiquettes_image_html, styles_fusion
)
from Utils.sampler_utils import apply_sampler_to_pipe, get_sampler_key_from_display_name
from Utils.model_manager import ModelManager
# Import de la nouvelle fonction d'exécution (assurez-vous du chemin correct)
from core.pipeline_executor import execute_pipeline_task_async
# Import du callback (assurez-vous du chemin correct)
from Utils.callback_diffuser import create_inpainting_callback as create_progress_callback



# --- Fonction principale du batch runner ---
def run_batch_from_json(model_manager: ModelManager, stop_event, json_file_obj, config, translations, device,
                        ui_status_output, ui_progress_output, ui_gallery_output,
                        ui_run_button, ui_stop_button,
                        progress=gr.Progress(track_tqdm=True)):
    """
    Exécute une série de tâches de génération d'images définies dans un fichier JSON.
    Utilise la fonction execute_pipeline_task_async pour la génération.
    Args:
        json_file_obj: Objet fichier JSON uploadé via Gradio.
        config: Dictionnaire de configuration global.
        model_manager: Instance de ModelManager.
        translations: Dictionnaire de traductions.
        device: Device PyTorch ('cuda' ou 'cpu').
        stop_event: Événement threading pour arrêter le batch.
        ui_status_output, ui_progress_output, ui_gallery_output: Composants Gradio pour les mises à jour.
        ui_run_button, ui_stop_button: Boutons Gradio à activer/désactiver.
        progress: Objet Gradio Progress pour la barre de progression tqdm.
    """

    # --- Récupérer l'état actuel depuis le ModelManager ---
    current_model_name = model_manager.current_model_name
    current_vae_name = model_manager.current_vae_name
    current_sampler_key = model_manager.current_sampler_key # Assurez-vous que ModelManager a cet attribut si nécessaire
    loras_charges_managed = model_manager.loaded_loras # Utiliser l'attribut de ModelManager

    # --- États initiaux et désactivation des boutons ---
    yield (
        translate("batch_starting", translations), # status
        create_progress_bar_html(0, 1, 0, translate("preparation", translations)), # progress
        [], # gallery
        gr.update(interactive=False), # run button
        gr.update(interactive=True) # stop button
    )

    # --- Lecture et validation du JSON ---
    if json_file_obj is None:
        yield (
            translate("erreur_aucun_fichier_json", translations), # status
            gr.update(), # progress
            gr.update(), # gallery
            gr.update(interactive=True), # run button
            gr.update(interactive=False) # stop button
        )
        return

    try:
        # Utiliser json_file_obj.name car c'est le chemin temporaire du fichier uploadé
        with open(json_file_obj.name, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except Exception as e:
        yield (
            f"{translate('erreur_lecture_json', translations)}: {e}", # status
            gr.update(), # progress
            gr.update(), # gallery
            gr.update(interactive=True), # run button (re-enable on error)
            gr.update(interactive=False) # stop button (disable on error)
        )
        return

    if not isinstance(tasks, list) or not tasks:
        yield (
            translate("erreur_json_vide_ou_invalide", translations), # status
            gr.update(), # progress
            gr.update(), # gallery
            gr.update(interactive=True), # run button (re-enable on error)
            gr.update(interactive=False) # stop button (disable on error)
        )
        return

    # --- Préparation ---
    total_tasks = len(tasks)
    generated_images = []
    start_time_batch = time.time()
    stop_event.clear()

    # --- Groupement par modèle/VAE/Sampler ---
    tasks_grouped = defaultdict(list)
    for task in tasks:
        sampler_key_task = task.get('sampler_key', 'sampler_euler') # Utiliser une clé par défaut
        group_key = (
            task.get('model'),
            task.get('vae', 'Auto'), # Utiliser "Auto" par défaut
            sampler_key_task
        )
        tasks_grouped[group_key].append(task)

    current_task_index = 0

    # --- Boucle par groupe ---
    for (model_name_req, vae_name_req, sampler_key_req), group_tasks in progress.tqdm(tasks_grouped.items(), desc=translate("batch_processing_groups", translations)):
        if not group_tasks:
             continue

        if stop_event.is_set():
            break

        # 1. Charger Modèle/VAE si nécessaire (utilise model_manager)
        if model_name_req != current_model_name or vae_name_req != current_vae_name:
            yield (
                f"{translate('batch_loading_model_vae', translations)}: {model_name_req} / {vae_name_req}", # status
                gr.update(), # progress
                gr.update(), # gallery
                gr.update(interactive=False), # run button (keep disabled during load)
                gr.update(interactive=True) # stop button (keep enabled)
            )
            try:
                # Utiliser ModelManager pour charger
                success, msg = model_manager.load_model(
                    model_name=model_name_req,
                    vae_name=vae_name_req,
                    model_type="standard", # Assumer standard, adapter si besoin
                    gradio_mode=False # Pas d'UI Gradio ici
                )
                if not success:
                    raise RuntimeError(msg)
                # Mettre à jour les variables locales après succès
                current_model_name = model_manager.current_model_name
                current_vae_name = model_manager.current_vae_name
                # Le reset des LoRAs est géré par load_model si unload=True

            except Exception as e_load:
                error_msg = f"{translate('erreur_chargement_modele', translations)}: {e_load}"
                print(f"{txt_color('[ERREUR]', 'erreur')} Exception dans le bloc de chargement de modèle (batch): {e_load}")
                print(txt_color('[ERREUR]', 'erreur'), "Traceback complet:")
                traceback.print_exc()
                yield (
                    error_msg, # status
                    gr.update(), # progress
                    gr.update(), # gallery
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Skip tasks in this group due to load error

        # Obtenir le pipe et compel actuels
        pipe = model_manager.get_current_pipe()
        compel = model_manager.get_current_compel()
        if pipe is None or compel is None: # Vérification après tentative de chargement
            error_msg = translate("erreur_pas_modele_charge_batch", translations)
            print(txt_color("[ERREUR]", "erreur"), error_msg)
            yield (
                error_msg, # status
                gr.update(), # progress
                gr.update(), # gallery
                gr.update(interactive=True), # run button (re-enable on error)
                gr.update(interactive=False) # stop button (disable on error)
            )
            continue # Skip group

        # 2. Appliquer Sampler si nécessaire (utilise le pipe obtenu)
        # Assurer que current_sampler_key est initialisé
        if current_sampler_key is None:
            # Essayer de lire depuis le scheduler actuel ou utiliser défaut
            try:
                # Tenter de retrouver la clé depuis le nom de la classe du scheduler
                scheduler_class_name = pipe.scheduler.__class__.__name__
                found = False
                for key, definition in SAMPLER_DEFINITIONS.items():
                    if definition['class_name'] == scheduler_class_name:
                        current_sampler_key = key
                        found = True
                        break
                if not found:
                    current_sampler_key = 'sampler_euler' # Fallback
            except:
                current_sampler_key = 'sampler_euler' # Fallback

        if sampler_key_req != current_sampler_key:
            yield (
                f"{translate('batch_applying_sampler', translations)}: {translate(sampler_key_req, translations)}", # status
                gr.update(), # progress
                gr.update(), # gallery
                gr.update(interactive=False), # run button (keep disabled)
                gr.update(interactive=True) # stop button (keep enabled)
            )
            sampler_message, success = apply_sampler_to_pipe(pipe, sampler_key_req, translations)
            if success:
                current_sampler_key = sampler_key_req
                # Mettre à jour l'état dans ModelManager si nécessaire
                # model_manager.current_sampler_key = sampler_key_req
            else:
                yield (
                    f"{translate('erreur_application_sampler', translations)}: {sampler_message}", # status
                    gr.update(), # progress
                    gr.update(), # gallery
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Skip tasks in this group due to sampler error

        # --- Boucle sur les tâches du groupe ---
        task_iterator = progress.tqdm(group_tasks, desc=f"{translate('batch_processing_tasks_in_group', translations)} ({model_name_req[:15]}...)")
        for task in task_iterator:
            if stop_event.is_set():
                break

            current_task_index += 1
            task_start_time = time.time()
            status_msg = f"{translate('batch_processing_task', translations)} {current_task_index}/{total_tasks}"
            yield (
                status_msg, # status
                gr.update(), # progress (will be updated by the inner loop)
                gr.update(), # gallery (will be updated later)
                gr.update(interactive=False), # run button (keep disabled)
                gr.update(interactive=True) # stop button (keep enabled)
            )

            # 3. Préparer les paramètres spécifiques
            prompt_orig = task.get('prompt', '')
            neg_prompt_orig = task.get('negative_prompt', config.get('NEGATIVE_PROMPT', ''))
            try:
                # Gérer le cas où 'styles' est une chaîne JSON ou une liste
                styles_data = task.get('styles', [])
                if isinstance(styles_data, str):
                    try: styles_list = json.loads(styles_data)
                    except json.JSONDecodeError: styles_list = []
                elif isinstance(styles_data, list):
                    styles_list = styles_data
                else: styles_list = []
                # Assurer que c'est bien une liste de strings
                if not all(isinstance(s, str) for s in styles_list):
                    styles_list = []
            except Exception: styles_list = []


            steps = task.get('steps', 30)
            guidance = task.get('guidance_scale', 7.0)
            seed = task.get('seed', -1)
            width = task.get('width', 1024)
            height = task.get('height', 1024)
            try:
                # Gérer le cas où 'loras' est une chaîne JSON ou une liste
                loras_data = task.get('loras', [])
                if isinstance(loras_data, str):
                    try: loras_list_task = json.loads(loras_data)
                    except json.JSONDecodeError: loras_list_task = []
                elif isinstance(loras_data, list):
                    loras_list_task = loras_data
                else: loras_list_task = []
                # Assurer que c'est bien une liste de dictionnaires
                if not all(isinstance(l, dict) for l in loras_list_task):
                    loras_list_task = []
            except Exception: loras_list_task = []

            output_filename_base = task.get('output_filename')

            # --- Fusion des styles ---
            final_prompt, final_neg_prompt, style_names_applied = styles_fusion(
                styles_list, prompt_orig, neg_prompt_orig, config['STYLES'], translations
            )
            print(txt_color("[INFO]", "info"), f"Batch Task {current_task_index} - Prompt Final: {final_prompt}")
            print(txt_color("[INFO]", "info"), f"Batch Task {current_task_index} - Neg Final: {final_neg_prompt}")

            # --- Gestion des LoRAs (utilise model_manager) ---
            lora_info_for_metadata = []
            try:
                # Construire la config attendue par model_manager.apply_loras
                lora_ui_config_batch = {'lora_checks': [], 'lora_dropdowns': [], 'lora_scales': []}
                for lora_info in loras_list_task:
                    lora_name = lora_info.get("name")
                    lora_weight = lora_info.get("weight")
                    if lora_name and lora_weight is not None:
                        lora_ui_config_batch['lora_checks'].append(True) # Assumer actifs
                        lora_ui_config_batch['lora_dropdowns'].append(lora_name)
                        lora_ui_config_batch['lora_scales'].append(lora_weight)
                        lora_info_for_metadata.append(f"{lora_name} ({lora_weight:.2f})")

                # Appeler apply_loras du ModelManager
                message_lora = model_manager.apply_loras(lora_ui_config_batch, gradio_mode=False) # Pas d'UI Gradio ici
                if message_lora and "erreur" in message_lora.lower():
                    print(txt_color("[AVERTISSEMENT]", "warning"), f"Erreur LoRA (non bloquante): {message_lora}")
                    # Ne pas arrêter le batch pour une erreur LoRA, juste logguer

            except Exception as e_lora:
                print(f"{txt_color('[ERREUR]', 'erreur')} {translate('erreur_lora_gestion', translations)}: {e_lora}")
                yield (
                    f"{translate('erreur_lora_gestion', translations)}: {e_lora}", # status
                    gr.update(), # progress
                    gr.update(), # gallery
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Skip this task

            # 4. Get Embeddings (utilise compel obtenu plus haut)
            try:
                conditioning, pooled = compel(final_prompt)
                neg_conditioning, neg_pooled = compel(final_neg_prompt)
            except Exception as e_compel:
                yield (
                    f"{translate('erreur_compel', translations)}: {e_compel}", # status
                    gr.update(), # progress
                    gr.update(), # gallery
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Skip this task

            # 5. Execute Pipeline via la fonction asynchrone
            image_seed = seed if seed != -1 else random.randint(1, 10**19 - 1)
            progress_update_queue = queue.Queue()

            pipeline_thread, result_container = execute_pipeline_task_async(
                pipe=pipe, # Utiliser le pipe obtenu plus haut
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=image_seed,
                width=width,
                height=height,
                device=device,
                stop_event=stop_event, # Passer l'event global
                translations=translations,
                progress_queue=progress_update_queue
                # preview_queue n'est pas utilisé ici, on ne gère pas l'aperçu dans le batch runner
            )

            # Boucle pour afficher la progression PENDANT que le thread tourne
            last_progress_html = ""
            while pipeline_thread.is_alive() or not progress_update_queue.empty():
                if stop_event.is_set():
                    break

                current_step_img, total_steps_img = None, steps
                while not progress_update_queue.empty():
                    try:
                        current_step_img, total_steps_img = progress_update_queue.get_nowait()
                    except queue.Empty: break

                new_progress_html = last_progress_html
                if current_step_img is not None:
                    progress_percent_img = int((current_step_img / total_steps_img) * 100)
                    progress_text = f"{status_msg} - Step {current_step_img}/{total_steps_img}"
                    new_progress_html = create_progress_bar_html(current_step_img, total_steps_img, progress_percent_img, progress_text)

                if new_progress_html != last_progress_html:
                    yield (
                        gr.update(), # status (keep current)
                        new_progress_html, # progress
                        gr.update(), # gallery (keep current)
                        gr.update(interactive=False), # run button (keep disabled)
                        gr.update(interactive=True) # stop button (keep enabled)
                    )
                    last_progress_html = new_progress_html

                time.sleep(0.05)

            pipeline_thread.join() # Attendre la fin du thread

            # --- Gérer le résultat après la fin du thread ---
            final_status = result_container.get("status")
            generated_image = result_container.get("final")
            error_details = result_container.get("error")

            if stop_event.is_set() or final_status == "stopped":
                print(txt_color("[INFO]", "info"), f"Batch task {current_task_index} stopped.")
                break # Sortir de la boucle des tâches

            elif final_status == "error":
                error_msg = str(error_details) if error_details else "Unknown pipeline error"
                yield (
                    f"{translate('erreur_pipeline', translations)}: {error_msg}", # status
                    "", # progress (clear)
                    gr.update(), # gallery (keep current)
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Passer à la tâche suivante

            elif generated_image is None:
                yield (
                    translate('erreur_pas_image_genere', translations), # status
                    "", # progress (clear)
                    gr.update(), # gallery (keep current)
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Passer à la tâche suivante

            # --- 6. Sauvegarde (si succès) ---
            temps_gen_img = f"{(time.time() - task_start_time):.2f} sec"
            date_str = datetime.now().strftime("%Y_%m_%d")
            heure_str = datetime.now().strftime("%H_%M_%S")
            save_dir = os.path.join(config["SAVE_DIR"], date_str)
            os.makedirs(save_dir, exist_ok=True)

            # Nom de fichier
            if output_filename_base:
                filename_final = output_filename_base.replace("{seed}", str(image_seed))
                filename_final = filename_final.replace("{index}", str(current_task_index))
                # Ajouter l'extension si manquante
                if not os.path.splitext(filename_final)[1]:
                    filename_final += f".{config['IMAGE_FORMAT'].lower()}"
            else:
                filename_final = f"batch_{date_str}_{heure_str}_{image_seed}_{width}x{height}.{config['IMAGE_FORMAT'].lower()}"
            chemin_image = os.path.join(save_dir, filename_final)

            # Métadonnées
            lora_info_str = ", ".join(lora_info_for_metadata) if lora_info_for_metadata else translate("aucun_lora", translations)
            style_info_str = ", ".join(style_names_applied) if style_names_applied else translate("Aucun_style", translations)
            sampler_display_name = translate(sampler_key_req, translations)

            donnees_xmp = {
                "Module": "SDXL Batch Generation", "Creator": config.get("AUTHOR", "CyberBill"),
                "Model": model_name_req, "VAE": vae_name_req, "Steps": steps,
                "Guidance": guidance, "Sampler": sampler_display_name,
                "Style": style_info_str,
                "Original Prompt": prompt_orig,
                "Prompt": final_prompt,
                "Negatif Prompt": final_neg_prompt,
                "Seed": image_seed, "Size": f"{width}x{height}",
                "Loras": lora_info_str, "Generation Time": temps_gen_img,
                "Batch Index": f"{current_task_index}/{total_tasks}"
            }

            metadata_structure, prep_message = preparer_metadonnees_image(
                generated_image, donnees_xmp, translations, chemin_image
            )
            print(txt_color("[INFO]", "info"), prep_message)

            # Sauvegarde (peut être mise dans un executor si besoin)
            enregistrer_image(
                generated_image, chemin_image, translations, config['IMAGE_FORMAT'],
                metadata_to_save=metadata_structure
            )
            enregistrer_etiquettes_image_html(
                chemin_image, donnees_xmp, translations, (current_task_index == total_tasks)
            )

            generated_images.append(generated_image)

            # Mise à jour UI (Galerie + Progression Globale)
            global_progress_percent = int((current_task_index / total_tasks) * 100)
            final_task_progress_html = create_progress_bar_html(current_task_index, total_tasks, global_progress_percent, status_msg)
            yield (
                status_msg, # status (keep current)
                final_task_progress_html, # progress
                generated_images, # gallery (update)
                gr.update(interactive=False), # run button (keep disabled)
                gr.update(interactive=True) # stop button (keep enabled)
            )
            # --- Fin de la gestion du résultat ---

        # --- Fin de la boucle des tâches pour ce groupe ---
        if stop_event.is_set():
             break # Sortir de la boucle des groupes si arrêt

    # --- Fin de la boucle des groupes ---
    # --- Fin du Batch ---
    final_batch_status = ""
    if stop_event.is_set():
        final_batch_status = translate("batch_stopped", translations)
    else:
        temps_total_batch = f"{(time.time() - start_time_batch):.2f} sec"
        # Utiliser current_task_index car il représente le nombre de tâches réellement traitées
        final_batch_status = translate("batch_completed", translations).format(total_tasks=current_task_index, total_time=temps_total_batch)

    # --- Nettoyage final / Restauration état initial ? (Optionnel) ---
    # Décharger les LoRAs utilisés spécifiquement pour le batch ?
    # model_manager.apply_loras({'lora_checks': [], 'lora_dropdowns': [], 'lora_scales': []}) # Pour décharger tous les LoRAs

    # Yield final
    yield (
        final_batch_status, # status
        "", # progress (clear)
        generated_images, # gallery (final state)
        gr.update(interactive=True), # run button (re-enable)
        gr.update(interactive=False) # stop button (disable)
    )

# --- Fin de core/batch_runner.py ---
