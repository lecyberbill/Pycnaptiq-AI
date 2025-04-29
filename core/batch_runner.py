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
from Utils.model_loader import charger_modele, charger_lora, decharge_lora
# Import de la nouvelle fonction d'exécution (assurez-vous du chemin correct)
from core.pipeline_executor import execute_pipeline_task_async
# Import du callback (assurez-vous du chemin correct)
from Utils.callback_diffuser import create_inpainting_callback as create_progress_callback



# --- Fonction principale du batch runner ---
def run_batch_from_json(
    gestionnaire,
    stop_event,
    # --- Dépendances & Configuration (passées via gr.State) ---
    json_file_obj,
    config,
    translations,
    device,
    # --- Composants UI à mettre à jour (passés directement) ---
    ui_status_output,
    ui_progress_output,
    ui_gallery_output,
    ui_run_button,
    ui_stop_button,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Exécute une série de tâches de génération d'images définies dans un fichier JSON.
    Utilise la fonction execute_pipeline_task_async pour la génération.
    Accède aux instances globales 'gestionnaire' et 'stop_event'.
    """
    # --- Vérification des globales importées ---
    try:
        if gestionnaire is None: # Accès direct à la globale
             raise NameError("Instance globale 'gestionnaire' est None.")
    except NameError:
         # Yield a tuple of 5 elements
         yield (
             "[ERREUR] Instance globale 'gestionnaire' non définie ou non accessible.", # status
             gr.update(), # progress
             gr.update(), # gallery
             gr.update(interactive=True), # run button
             gr.update(interactive=False) # stop button
         )
         return

    # --- Nettoyage automatique des LoRAs au début du batch ---
    try:
        # This line should now work if utils.py is updated and reloaded
        gestionnaire.verifier_et_nettoyer_loras()
    except Exception as e_auto_lora:
        error_msg = f"{translate('erreur_auto_lora', translations)}: {e_auto_lora}"
        # --- CORRECTION PRINT ---
        print(txt_color("[ERREUR]", "erreur"), error_msg)
        # --- FIN CORRECTION ---
        # --- MODIFIED YIELD ---
        # Yield updates for all 5 expected outputs
        yield (
            error_msg,                  # 1. ui_status_output
            gr.update(),                # 2. ui_progress_output (no change)
            gr.update(),                # 3. ui_gallery_output (no change)
            gr.update(interactive=True),# 4. ui_run_button (re-enable)
            gr.update(interactive=False)# 5. ui_stop_button (disable)
        )
        # --- END MODIFIED YIELD ---
        return

    # --- Récupérer l'état actuel depuis le gestionnaire ---
    current_model_name = gestionnaire.current_model_name
    current_vae_name = gestionnaire.current_vae_name
    current_sampler_key = gestionnaire.current_sampler_key
    loras_charges_managed = gestionnaire.loras_charges

    # --- États initiaux et désactivation des boutons ---
    # Yield a tuple of 5 elements
    yield (
        translate("batch_starting", translations), # status
        create_progress_bar_html(0, 1, 0, translate("preparation", translations)), # progress
        [], # gallery
        gr.update(interactive=False), # run button
        gr.update(interactive=True) # stop button
    )

    # --- Lecture et validation du JSON ---
    if json_file_obj is None:
        # Yield a tuple of 5 elements
        yield (
            translate("erreur_aucun_fichier_json", translations), # status
            gr.update(), # progress
            gr.update(), # gallery
            gr.update(interactive=True), # run button
            gr.update(interactive=False) # stop button
        )
        return

    try:
        with open(json_file_obj.name, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except Exception as e:
        # Yield a tuple of 5 elements
        yield (
            f"{translate('erreur_lecture_json', translations)}: {e}", # status
            gr.update(), # progress
            gr.update(), # gallery
            gr.update(interactive=True), # run button (re-enable on error)
            gr.update(interactive=False) # stop button (disable on error)
        )
        return

    if not isinstance(tasks, list) or not tasks:
        # Yield a tuple of 5 elements
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
    stop_event.clear() # Utilise la globale importée

    # --- Groupement par modèle/VAE/Sampler ---
    tasks_grouped = defaultdict(list)
    for task in tasks:
        sampler_key_task = task.get('sampler_key', 'sampler_euler')
        group_key = (
            task.get('model'),
            task.get('vae', 'Défaut VAE'),
            sampler_key_task
        )
        tasks_grouped[group_key].append(task)

    current_task_index = 0

    # --- Boucle par groupe ---
    for (model_name_req, vae_name_req, sampler_key_req), group_tasks in progress.tqdm(tasks_grouped.items(), desc=translate("batch_processing_groups", translations)):
        if not group_tasks:
             continue # Important de passer au groupe suivant si vide


        if stop_event.is_set():
            break

        # 1. Charger Modèle/VAE si nécessaire (utilise globale 'gestionnaire')
        if model_name_req != current_model_name or vae_name_req != current_vae_name:
            # Yield a tuple of 5 elements
            yield (
                f"{translate('batch_loading_model_vae', translations)}: {model_name_req} / {vae_name_req}", # status
                gr.update(), # progress
                gr.update(), # gallery
                gr.update(interactive=False), # run button (keep disabled during load)
                gr.update(interactive=True) # stop button (keep enabled)
            )
            try:
                pipe_new, compel_new, message, message_detail = charger_modele(
                    model_name_req, vae_name_req, translations,
                    config['MODELS_DIR'], config['VAE_DIR'],
                    device, gestionnaire.torch_dtype,
                    gestionnaire.vram_total_gb,
                    gestionnaire.global_pipe, gestionnaire.global_compel,
                    gradio_mode=False
                )
                print(f"Message principal: {message}, Détail: {message_detail}")
                if pipe_new is None: raise RuntimeError(message)
                gestionnaire.update_global_pipe(pipe_new)
                gestionnaire.update_global_compel(compel_new)
                current_model_name = model_name_req
                current_vae_name = vae_name_req
                try:
                    gestionnaire.reset_loras_charges()
                except Exception as e_reset_lora:
                    print(txt_color("[ERREUR]", "erreur"), f"Erreur pendant le reset automatique des LoRAs: {e_reset_lora}")

            except Exception as e_load:
                error_msg = f"{translate('erreur_chargement_modele', translations)}: {e_load}"
            # --- AJOUT DE LOGS ---
                print(f"{txt_color('[ERREUR]', 'erreur')} Exception dans le bloc de chargement de modèle (batch): {e_load}")
                print(txt_color('[ERREUR]', 'erreur'), "Traceback complet:")
                traceback.print_exc() 
                # Yield a tuple of 5 elements
                yield (
                    f"{translate('erreur_chargement_modele', translations)}: {e_load}", # status
                    gr.update(), # progress
                    gr.update(), # gallery
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Skip tasks in this group due to load error

        # 2. Appliquer Sampler si nécessaire (utilise globale 'gestionnaire')
        if sampler_key_req != current_sampler_key:
            # Yield a tuple of 5 elements
            yield (
                f"{translate('batch_applying_sampler', translations)}: {translate(sampler_key_req, translations)}", # status
                gr.update(), # progress
                gr.update(), # gallery
                gr.update(interactive=False), # run button (keep disabled)
                gr.update(interactive=True) # stop button (keep enabled)
            )
            # --- ASSUREZ-VOUS QUE CETTE LIGNE EST PRÉSENTE ---
            sampler_message, success = apply_sampler_to_pipe(gestionnaire.global_pipe, sampler_key_req, translations)
            # --- FIN ASSURANCE ---
            if success:
                current_sampler_key = sampler_key_req
                gestionnaire.current_sampler_key = sampler_key_req # Mettre à jour aussi dans gestionnaire si besoin
            else:
                # Yield a tuple of 5 elements
                yield (
                    f"{translate('erreur_application_sampler', translations)}: {sampler_message}", # status
                    gr.update(), # progress
                    gr.update(), # gallery
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Skip tasks in this group due to sampler error

        # try:
        # Isoler la création de l'itérateur tqdm
        task_iterator = progress.tqdm(group_tasks, desc=f"{translate('batch_processing_tasks_in_group', translations)} ({model_name_req[:15]}...)")
        # --- Boucle sur les tâches du groupe ---
        for task in task_iterator:
            # --- Le code original de la boucle des tâches commence ici ---
            if stop_event.is_set():
                break

            current_task_index += 1
            task_start_time = time.time()
            status_msg = f"{translate('batch_processing_task', translations)} {current_task_index}/{total_tasks}"
            # Yield a tuple of 5 elements
            yield (
                status_msg, # status
                gr.update(), # progress (will be updated by the inner loop)
                gr.update(), # gallery (will be updated later)
                gr.update(interactive=False), # run button (keep disabled)
                gr.update(interactive=True) # stop button (keep enabled)
            )

            # 3. Préparer les paramètres spécifiques
            prompt_orig = task.get('prompt', '') # Original user prompt
            neg_prompt_orig = task.get('negative_prompt', config.get('NEGATIVE_PROMPT', ''))
            try:
                styles_list = task.get('styles')
                if not isinstance(styles_list, list): styles_list = []
            except json.JSONDecodeError: styles_list = []

            steps = task.get('steps', 30)
            guidance = task.get('guidance_scale', 7.0)
            seed = task.get('seed', -1)
            width = task.get('width', 1024)
            height = task.get('height', 1024)
            try:
                loras_list_task = task.get('loras')
                if not isinstance(loras_list_task, list): loras_list_task = []
            except json.JSONDecodeError: loras_list_task = []

            output_filename_base = task.get('output_filename')

            # --- Fusion des styles ---
            # Call styles_fusion to get final prompts
            final_prompt, final_neg_prompt, style_names_applied = styles_fusion(
                styles_list, prompt_orig, neg_prompt_orig, config['STYLES'], translations
            )
            print(txt_color("[INFO]", "info"), f"Batch Task {current_task_index} - Prompt Final: {final_prompt}")
            print(txt_color("[INFO]", "info"), f"Batch Task {current_task_index} - Neg Final: {final_neg_prompt}")


            # --- Gestion des LoRAs (uses global 'gestionnaire' and loras_charges_managed) ---
            lora_info_for_metadata = []
            try:
                # Unload those not needed
                adapters_to_unload = set(loras_charges_managed.keys())
                required_adapters_task = set()
                for lora_task in loras_list_task:
                    lora_name = lora_task.get('name')
                    if lora_name:
                        adapter_name = lora_name.replace('.safetensors', '').replace('.', '_')
                        required_adapters_task.add(adapter_name)
                for adapter_name in adapters_to_unload - required_adapters_task:
                    decharge_lora(gestionnaire.global_pipe, translations, adapter_name)
                    if adapter_name in loras_charges_managed: del loras_charges_managed[adapter_name]

                # Load/Update those needed
                for lora_task in loras_list_task:
                    lora_name = lora_task.get('name')
                    lora_weight = lora_task.get('weight')
                    if lora_name and lora_weight is not None:
                        adapter_name = lora_name.replace('.safetensors', '').replace('.', '_')
                        if adapter_name not in loras_charges_managed:
                            charger_lora(lora_name, gestionnaire.global_pipe, config['LORAS_DIR'], translations, lora_weight)
                            loras_charges_managed[adapter_name] = lora_weight
                        elif loras_charges_managed[adapter_name] != lora_weight:
                            gestionnaire.global_pipe.set_adapters([adapter_name], adapter_weights=[lora_weight]) # Use set_adapters
                            loras_charges_managed[adapter_name] = lora_weight
                        lora_info_for_metadata.append(f"{lora_name} ({lora_weight:.2f})")
            except Exception as e_lora:
                    print(f"{txt_color('[ERREUR]', 'erreur')} {translate('erreur_lora_gestion', translations)}: {e_lora}")
                    # Yield a tuple of 5 elements
                    yield (
                        f"{translate('erreur_lora_gestion', translations)}: {e_lora}", # status
                        gr.update(), # progress
                        gr.update(), # gallery
                        gr.update(interactive=True), # run button (re-enable on error)
                        gr.update(interactive=False) # stop button (disable on error)
                    )
                    continue # Skip this task

            # 4. Get Embeddings (uses global 'gestionnaire')
            try:
                conditioning, pooled = gestionnaire.global_compel(final_prompt)
                neg_conditioning, neg_pooled = gestionnaire.global_compel(final_neg_prompt)
            except Exception as e_compel:
                # Yield a tuple of 5 elements
                yield (
                    f"{translate('erreur_compel', translations)}: {e_compel}", # status
                    gr.update(), # progress
                    gr.update(), # gallery
                    gr.update(interactive=True), # run button (re-enable on error)
                    gr.update(interactive=False) # stop button (disable on error)
                )
                continue # Skip this task

            # 5. Execute Pipeline via the async helper function
            image_seed = seed if seed != -1 else random.randint(1, 10**19 - 1)
            progress_update_queue = queue.Queue() # Queue specific to this task

            # Call the non-blocking helper function
            pipeline_thread, result_container = execute_pipeline_task_async(
                pipe=gestionnaire.global_pipe,
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
                stop_event=stop_event, # Pass the global event
                translations=translations,
                progress_queue=progress_update_queue
            )

            # Loop to display progress WHILE the thread is running
            last_progress_html = ""
            while pipeline_thread.is_alive() or not progress_update_queue.empty():
                if stop_event.is_set():
                    # The event is set, the callback in the thread should stop it.
                    # We exit the waiting loop here.
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

                # Yield only if progress has changed
                if new_progress_html != last_progress_html:
                    # Yield a tuple of 5 elements
                    yield (
                        gr.update(), # status (keep current)
                        new_progress_html, # progress
                        gr.update(), # gallery (keep current)
                        gr.update(interactive=False), # run button (keep disabled)
                        gr.update(interactive=True) # stop button (keep enabled)
                    )
                    last_progress_html = new_progress_html

                time.sleep(0.05) # Short wait

            pipeline_thread.join() # Ensure the thread is finished

            # --- Handle the result after the thread finishes ---
            final_status = result_container.get("status")
            generated_image = result_container.get("final")
            error_details = result_container.get("error")

            if stop_event.is_set() or final_status == "stopped":
                    # If stop was requested globally OR if the thread marked itself as stopped
                    print(txt_color("[INFO]", "info"), f"Batch task {current_task_index} stopped.")
                    break # Exit the task loop

            elif final_status == "error":
                    error_msg = str(error_details) if error_details else "Unknown pipeline error"
                    # Yield a tuple of 5 elements
                    yield (
                        f"{translate('erreur_pipeline', translations)}: {error_msg}", # status
                        "", # progress (clear)
                        gr.update(), # gallery (keep current)
                        gr.update(interactive=True), # run button (re-enable on error)
                        gr.update(interactive=False) # stop button (disable on error)
                    )
                    continue # Skip to the next task

            elif generated_image is None:
                    # Yield a tuple of 5 elements
                    yield (
                        translate('erreur_pas_image_genere', translations), # status
                        "", # progress (clear)
                        gr.update(), # gallery (keep current)
                        gr.update(interactive=True), # run button (re-enable on error)
                        gr.update(interactive=False) # stop button (disable on error)
                    )
                    continue # Skip to the next task

            # --- 6. Save (if successful) ---
            temps_gen_img = f"{(time.time() - task_start_time):.2f} sec"
            date_str = datetime.now().strftime("%Y_%m_%d")
            heure_str = datetime.now().strftime("%H_%M_%S")
            save_dir = os.path.join(config["SAVE_DIR"], date_str)
            os.makedirs(save_dir, exist_ok=True)

            # Filename
            if output_filename_base:
                    filename_final = output_filename_base.replace("{seed}", str(image_seed))
                    filename_final = filename_final.replace("{index}", str(current_task_index))
                    if not os.path.splitext(filename_final)[1]:
                        filename_final += f".{config['IMAGE_FORMAT'].lower()}"
            else:
                    filename_final = f"batch_{date_str}_{heure_str}_{image_seed}_{width}x{height}.{config['IMAGE_FORMAT'].lower()}"
            chemin_image = os.path.join(save_dir, filename_final)

            # Metadata
            lora_info_str = ", ".join(lora_info_for_metadata) if lora_info_for_metadata else translate("aucun_lora", translations)
            # Use the actual style names applied returned by styles_fusion
            style_info_str = ", ".join(style_names_applied) if style_names_applied else translate("Aucun_style", translations)
            sampler_display_name = translate(sampler_key_req, translations)

            donnees_xmp = {
                "Module": "SDXL Batch Generation", "Creator": config["AUTHOR"],
                "Model": model_name_req, "VAE": vae_name_req, "Steps": steps,
                "Guidance": guidance, "Sampler": sampler_display_name,
                "Style": style_info_str,
                "Original Prompt": prompt_orig, # Keep the original prompt
                "Prompt": final_prompt, # Prompt after style fusion
                "Negatif Prompt": final_neg_prompt, # Negative after style fusion
                "Seed": image_seed, "Size": f"{width}x{height}",
                "Loras": lora_info_str, "Generation Time": temps_gen_img,
                "Batch Index": f"{current_task_index}/{total_tasks}"
            }

            metadata_structure, prep_message = preparer_metadonnees_image(
                generated_image, donnees_xmp, translations, chemin_image
            )
            print(txt_color("[INFO]", "info"), prep_message)

            # Save (can be put in an executor if needed)
            enregistrer_image(
                generated_image, chemin_image, translations, config['IMAGE_FORMAT'],
                metadata_to_save=metadata_structure
            )
            enregistrer_etiquettes_image_html(
                chemin_image, donnees_xmp, translations, (current_task_index == total_tasks)
            )

            generated_images.append(generated_image)

            # Update UI (Gallery + Global Progress)
            global_progress_percent = int((current_task_index / total_tasks) * 100)
            # Display the final global progress for this task
            final_task_progress_html = create_progress_bar_html(current_task_index, total_tasks, global_progress_percent, status_msg)
            # Yield a tuple of 5 elements
            yield (
                status_msg, # status (keep current)
                final_task_progress_html, # progress
                generated_images, # gallery (update)
                gr.update(interactive=False), # run button (keep disabled)
                gr.update(interactive=True) # stop button (keep enabled)
            )
            # --- End of result handling ---


        # This print executes *after* the loop finishes OR breaks

        # except Exception as inner_loop_error:
        #     # Catch any unexpected error happening before or during the inner loop
        #     print(txt_color("[ERREUR]", "erreur"), f"--- ERREUR INATTENDUE avant/pendant la boucle interne: {inner_loop_error} ---")
        #     traceback.print_exc() # Imprimer la trace complète de l'erreur
        #     # Yield an error state to the UI
        #     yield (
        #         f"Erreur boucle interne: {inner_loop_error}", # status
        #         "", # progress (clear)
        #         gr.update(), # gallery (keep current)
        #         gr.update(interactive=True), # run button (re-enable on error)
        #         gr.update(interactive=False) # stop button (disable on error)
        #     )
        #     # --- CHANGE 'continue' TO 'break' ---
        #     break # Stop the entire batch process if this critical part fails
        #     # --- END CHANGE ---

        # --- Fin de la boucle des tâches pour ce groupe ---
        # This print should now be equivalent to Point F if the loop completed normally
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
    # ...

    # Final yield - tuple of 5 elements
    yield (
        final_batch_status, # status
        "", # progress (clear)
        generated_images, # gallery (final state)
        gr.update(interactive=True), # run button (re-enable)
        gr.update(interactive=False) # stop button (disable)
    )

# --- Fin de core/batch_runner.py ---
