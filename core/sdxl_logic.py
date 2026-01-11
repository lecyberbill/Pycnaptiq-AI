import os
import time
import random
import threading
import queue
import json
import math
import gc
import traceback
from datetime import datetime
from PIL import Image
import torch
import gradio as gr
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetInpaintPipeline
from core.translator import translate_prompt
from core.inpainting_utils import create_opaque_mask_from_editor
from Utils.utils import (
    txt_color, translate, create_progress_bar_html, 
    enregistrer_image, enregistrer_etiquettes_image_html, 
    preparer_metadonnees_image, 
    finalize_html_report_if_needed, styles_fusion
)
from Utils.gest_mem import empty_working_set
from Utils.callback_diffuser import create_callback_on_step_end, create_inpainting_callback
from Utils.sampler_utils import apply_sampler_to_pipe, get_sampler_key_from_display_name
from core.pipeline_executor import execute_pipeline_task_async
from core.logger import logger
from core.upscale_utils import tiled_upscale_process

def generate_image(text, style_selection, guidance_scale, num_steps, selected_format, traduire, seed_input, num_images, pag_enabled, pag_scale, pag_applied_layers_str, 
                   original_user_prompt_for_cycle, prompt_is_currently_enhanced, enhancement_cycle_active_state,
                   controlnet_image, controlnet_model_name, controlnet_scale, ip_adapter_image, ip_adapter_scale,
                   ip_adapter_model_name,
                   upscale_tiles_enable, upscale_tiles_factor, upscale_tiles_size, upscale_tiles_overlap, upscale_tiles_denoising,
                   model_manager, translations, config, device, stop_event, stop_gen, SAVE_DIR, AUTHOR, IMAGE_FORMAT, 
                   image_executor, html_executor, STYLES, NEGATIVE_PROMPT, PREVIEW_QUEUE, *lora_inputs):
    """Génère des images avec Stable Diffusion en utilisant execute_pipeline_task_async."""
    
    lora_checks = lora_inputs[:4]
    lora_dropdowns = lora_inputs[4:8]
    lora_scales = lora_inputs[8:]

    bouton_charger_update_off = gr.update(interactive=False)
    btn_generate_update_off = gr.update(interactive=False)
    bouton_charger_update_on = gr.update(interactive=True)
    btn_generate_update_on = gr.update(interactive=True)
    btn_save_preset_off = gr.update(interactive=False)

    initial_images = []
    initial_seeds = ""
    initial_time = translate("preparation", translations)
    initial_html = ""
    initial_preview = None
    initial_progress = ""
    output_gen_data_json = None
    output_preview_image = None
    final_save_button_state = gr.update(interactive=False)

    prompt_to_use_for_sdxl = text
    prompt_to_log_as_original = text

    if enhancement_cycle_active_state or prompt_is_currently_enhanced:
        prompt_to_log_as_original = original_user_prompt_for_cycle

    if traduire:
        translated_version = translate_prompt(prompt_to_use_for_sdxl, translations)
        if translated_version != prompt_to_use_for_sdxl:
            gr.Info(translate("prompt_traduit_pour_generation", translations), 2.0)
        prompt_to_use_for_sdxl = translated_version

    yield (
        initial_images, initial_seeds, initial_time, initial_html, initial_preview, initial_progress,
        bouton_charger_update_off, btn_generate_update_off,
        btn_save_preset_off,
        output_gen_data_json,
        output_preview_image
    )

    final_images = []
    final_seeds = ""
    final_message = ""
    final_html_msg = ""
    final_preview_img = None
    final_progress_html = ""

    try:
        if model_manager.get_current_pipe() is None:
            print(txt_color("[ERREUR] ","erreur"), translate("erreur_pas_modele", translations))
            gr.Warning(translate("erreur_pas_modele", translations), 4.0)
            final_message = translate("erreur_pas_modele", translations)
            final_html_msg = f"<p style='color: red;'>{final_message}</p>"
            yield (
                [], "", final_message, final_html_msg, 
                None, "",                             
                bouton_charger_update_on, btn_generate_update_on, 
                gr.update(interactive=False),         
                None, None                            
            )
            return

        if model_manager.current_model_type != 'standard':
            error_message = translate("erreur_mauvais_type_modele", translations)
            print(txt_color("[ERREUR] ","erreur"), error_message)
            gr.Warning(error_message, 4.0)
            final_message = error_message
            final_html_msg = f"<p style='color: red;'>{error_message}</p>"
            yield (
                [], "", final_message, final_html_msg,
                None, "",
                bouton_charger_update_on, btn_generate_update_on,
                gr.update(interactive=False),
                None, None
            )
            return

        start_time = time.time()
        stop_event.clear()
        stop_gen.clear()

        seeds = [random.randint(1, 10**19 - 1) for _ in range(num_images)] if seed_input == -1 else [seed_input] * num_images

        prompt_en, negative_prompt_str, selected_style_display_names = styles_fusion(
            style_selection,
            prompt_to_use_for_sdxl,
            NEGATIVE_PROMPT,
            STYLES,
            translations
        )

        logger.info(f"Prompt Positif Final (string): {prompt_en}")
        logger.info(f"Prompt Négatif Final (string): {negative_prompt_str}")

        compel = model_manager.get_current_compel()      
        conditioning, pooled = compel(prompt_en)
        neg_conditioning, neg_pooled = compel(negative_prompt_str)

        if isinstance(selected_format, dict):
            # If it's a dict from config.json or a preset
            selected_format_parts = selected_format.get("dimensions", "1024*1024")
        else:
            # If it's a string from the UI "1024*1024 : orientation"
            selected_format_parts = str(selected_format).split(":")[0].strip()
            
        width, height = map(int, selected_format_parts.split("*"))
        image_paths = []
        seed_strings = []
        formatted_seeds = ""

        # --- PREPARATION CONTROLNET ---
        control_image_prepared = None
        if controlnet_image is not None and controlnet_model_name:
            logger.info("Préparation de l'image ControlNet...")
            control_image_prepared = controlnet_image.resize((width, height), Image.LANCZOS)
            
            cn_model = model_manager.get_controlnet(controlnet_model_name)
            if cn_model:
                # On "upgrade" le pipe actuel temporairement
                current_pipe = model_manager.get_current_pipe()
                new_pipe = model_manager.prepare_controlnet_pipeline(current_pipe, [cn_model])
                # Note: On ne met pas à jour model_manager.current_pipe pour ne pas changer l'état global
                # si l'utilisateur change de modèle de base après.
                # On utilise ce pipe juste pour cette génération.
                pipe_to_use = new_pipe
            else:
                pipe_to_use = model_manager.get_current_pipe()
        else:
            pipe_to_use = model_manager.get_current_pipe()

        # --- PREPARATION IP-ADAPTER ---
        ip_adapter_active = False
        # On appelle toujours prepare_ip_adapter_pipeline. 
        # Si ip_adapter_image est None ou scale <= 0, model_manager gérera le déchargement si nécessaire.
        pipe_to_use = model_manager.prepare_ip_adapter_pipeline(pipe_to_use, ip_adapter_model_name if (ip_adapter_image is not None and ip_adapter_scale > 0) else None)
        
        if ip_adapter_image is not None and ip_adapter_scale > 0:
             # Vérifier si l'IP-Adapter est opérationnel (déjà chargé par l'appel ci-dessus si nécessaire)
             if hasattr(pipe_to_use, "unet") and hasattr(pipe_to_use.unet, "encoder_hid_proj") and pipe_to_use.unet.encoder_hid_proj is not None:
                 ip_adapter_active = True
                 if hasattr(pipe_to_use, "set_ip_adapter_scale"):
                     pipe_to_use.set_ip_adapter_scale(ip_adapter_scale)
             else:
                 logger.warning("L'IP-Adapter n'a pas pu être activé sur le pipeline. Génération sans IP-Adapter.")
        
        current_data_for_preset = None
        preview_image_for_preset = None

        lora_ui_config = {
            'lora_checks': lora_checks,
            'lora_dropdowns': lora_dropdowns,
            'lora_scales': lora_scales
        }
        message_lora = model_manager.apply_loras(lora_ui_config, gradio_mode=True)
        erreur_keyword = translate("erreur", translations).lower()
        if message_lora and erreur_keyword in message_lora.lower():
            gr.Warning(message_lora, duration=4.0)
            final_message = message_lora
            final_html_msg = f"<p style='color: red;'>{final_message}</p>"
            yield (
                [], "", final_message, final_html_msg,
                None, "",
                bouton_charger_update_on, btn_generate_update_on,
                gr.update(interactive=False),
                None, None
            )
            return
        elif message_lora:
            print(txt_color("[INFO]", "info"), f"Message LoRA: {message_lora}")

        pag_applied_layers = []
        if pag_enabled and pag_applied_layers_str:
            pag_applied_layers = [s.strip() for s in pag_applied_layers_str.split(',') if s.strip()]

        for idx, seed in enumerate(seeds):
            depart_time = time.time()
            PREVIEW_QUEUE.clear()
            html_message_result = translate("generation_en_cours", translations)

            if stop_event.is_set():
                logger.info(f"{translate('arrete_demande_apres', translations)} {idx} {translate('images', translations)}.")
                gr.Info(translate("arrete_demande_apres", translations) + f"{idx} {translate('images', translations)}.", 3.0)
                final_message = translate("generation_arretee", translations)
                break

            logger.info(f"{translate('generation_image', translations)} {idx+1} {translate('seed_utilise', translations)} {seed}")
            gr.Info(translate('generation_image', translations) + f"{idx+1} {translate('seed_utilise', translations)} {seed}", 3.0)

            progress_update_queue = queue.Queue() 
            pipe = pipe_to_use 
            pipeline_thread, result_container = execute_pipeline_task_async(
                pipe=pipe,
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                width=width,
                height=height,
                device=device,
                stop_event=stop_gen, 
                translations=translations,
                progress_queue=progress_update_queue,
                preview_queue=PREVIEW_QUEUE, 
                pag_enabled=pag_enabled,
                pag_scale=pag_scale,
                pag_applied_layers=pag_applied_layers,
                control_image=control_image_prepared,
                controlnet_conditioning_scale=controlnet_scale,
                ip_adapter_image=ip_adapter_image if ip_adapter_active else None
            )

            last_preview_index = 0
            last_progress_html = ""
            last_yielded_preview = None
            while pipeline_thread.is_alive() or last_preview_index < len(PREVIEW_QUEUE) or not progress_update_queue.empty():
                current_step, total_steps = None, num_steps
                while not progress_update_queue.empty():
                    try:
                        current_step, total_steps = progress_update_queue.get_nowait()
                    except queue.Empty: break
                new_progress_html = last_progress_html
                if current_step is not None:
                    progress_percent = int((current_step / total_steps) * 100)
                    new_progress_html = create_progress_bar_html(current_step, total_steps, progress_percent)

                preview_img_to_yield = None
                preview_yielded_in_loop = False
                while last_preview_index < len(PREVIEW_QUEUE):
                    preview_img_to_yield = PREVIEW_QUEUE[last_preview_index]
                    last_preview_index += 1
                    last_yielded_preview = preview_img_to_yield
                    yield (
                        image_paths, formatted_seeds, f"{idx+1}/{num_images}...", html_message_result,
                        preview_img_to_yield, new_progress_html, 
                        bouton_charger_update_off, btn_generate_update_off,
                        gr.update(interactive=False), 
                        output_gen_data_json,
                        output_preview_image
                    )
                    preview_yielded_in_loop = True
                
                if not preview_yielded_in_loop and new_progress_html != last_progress_html:
                     yield (
                         image_paths, formatted_seeds, f"{idx+1}/{num_images}...", html_message_result,
                         last_yielded_preview, 
                         new_progress_html, 
                         bouton_charger_update_off, btn_generate_update_off,
                         gr.update(interactive=False), 
                         output_gen_data_json,
                         output_preview_image
                     )
                last_progress_html = new_progress_html 
                time.sleep(0.05)
            pipeline_thread.join() 
            PREVIEW_QUEUE.clear() 

            final_status = result_container.get("status")
            final_image = result_container.get("final")
            error_details = result_container.get("error")

            final_progress_html = ""
            if final_status == "success":
                final_progress_html = create_progress_bar_html(num_steps, num_steps, 100)
            elif final_status == "error":
                final_progress_html = f'<p style="color: red;">{translate("erreur_lors_generation", translations)}</p>'

            if stop_event.is_set() or stop_gen.is_set() or final_status == "stopped":
                print(txt_color("[INFO]", "info"), translate("generation_arretee_pas_sauvegarde", translations))
                gr.Info(translate("generation_arretee_pas_sauvegarde", translations), 3.0)
                final_message = translate("generation_arretee", translations)
                yield (
                    image_paths, " ".join(seed_strings), translate("generation_arretee", translations),
                    translate("generation_arretee_pas_sauvegarde", translations), None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, 
                    gr.update(interactive=False), 
                    None, 
                    None  
                )
                break 
            elif final_status == "error":
                error_msg = str(error_details) if error_details else "Unknown pipeline error"
                print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_pipeline', translations)}: {error_msg}")
                gr.Warning(f"{translate('erreur_pipeline', translations)}: {error_msg}", 4.0)
                yield (
                    image_paths, " ".join(seed_strings), translate("erreur_pipeline", translations),
                    error_msg, None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, 
                    gr.update(interactive=False), 
                    None, 
                    None  
                )
                continue 
            elif final_image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image_genere", translations))
                gr.Warning(translate("erreur_pas_image_genere", translations), 4.0)
                yield (
                    image_paths, " ".join(seed_strings), translate("erreur_pas_image_genere", translations),
                    translate("erreur_pas_image_genere", translations), None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, 
                    gr.update(interactive=False), 
                    None, 
                    None  
                )
                continue 
            
            # --- POST-PROCESS: Tiled Upscale ---
            if upscale_tiles_enable and final_image and not (stop_event.is_set() or stop_gen.is_set()):
                def progress_upscale(current, total):
                    msg = f"{translate('upscale_tiles_processing', translations)} {current+1}/{total}"
                    progress_html = create_progress_bar_html(current, total, int((current/total)*100))
                    # On ne peut pas facilement yielder depuis ici car on est dans une fonction synchrone appelée par tiled_upscale_process
                    # Mais tiled_upscale_process est appelé dans le thread principal de generate_image (qui est un générateur)
                    # Donc on peut tricher en stockant le message pour le yield suivant ou juste logger
                    print(f"[Upscale] {msg}")

                try:
                    # On utilise le pipe actuel pour l'Img2Img par morceaux
                    # Note: on passe les embeds déjà calculés
                    final_image = tiled_upscale_process(
                        image=final_image,
                        pipe=pipe,
                        upscale_factor=upscale_tiles_factor,
                        tile_size=int(upscale_tiles_size),
                        overlap=int(upscale_tiles_overlap),
                        denoising_strength=upscale_tiles_denoising,
                        prompt_embeds=conditioning,
                        pooled_prompt_embeds=pooled,
                        neg_prompt_embeds=neg_conditioning,
                        neg_pooled_prompt_embeds=neg_pooled,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        device=device,
                        progress_callback=progress_upscale
                    )
                    html_message_result = translate("upscale_tiles_finished", translations)
                    width, height = final_image.size
                    selected_format_parts = f"{width}*{height}" # Mettre à jour pour les métadonnées
                except Exception as e_upscale:
                    print(f"[ERREUR] Tiled Upscale failed: {e_upscale}")
                    gr.Warning(f"Upscale failed: {e_upscale}", 5.0)

            if final_image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image_genere", translations))
                gr.Warning(translate("erreur_pas_image_genere", translations), 4.0)
                yield (
                    image_paths, " ".join(seed_strings), translate("erreur_pas_image_genere", translations),
                    translate("erreur_pas_image_genere", translations), None, final_progress_html,
                    bouton_charger_update_off, btn_generate_update_off, 
                    gr.update(interactive=False), 
                    None, 
                    None  
                )
                continue 

            current_data_for_preset = {
                 "model": model_manager.current_model_name,  
                 "vae": model_manager.current_vae_name,  
                 "original_user_prompt": prompt_to_log_as_original,
                 "prompt": prompt_en,
                 "current_prompt_is_enhanced": prompt_is_currently_enhanced,
                 "enhancement_cycle_active": enhancement_cycle_active_state,
                 "negative_prompt": negative_prompt_str,
                 "styles": json.dumps(selected_style_display_names if selected_style_display_names else []),
                 "guidance_scale": guidance_scale,
                 "num_steps": num_steps,
                 "sampler_key": model_manager.get_current_sampler_key() if hasattr(model_manager, 'get_current_sampler_key') else "sampler_euler", 
                 "seed": seed,
                 "width": width,
                 "height": height,
                 "loras": json.dumps([{"name": name, "weight": weight['scale']} for name, weight in model_manager.loaded_loras.items()]),
                 "pag_enabled": pag_enabled,
                 "pag_scale": pag_scale,
                 "custom_pipeline_id": "hyoungwoncho/sd_perturbed_attention_guidance" if pag_enabled else None,
                 "pag_applied_layers": pag_applied_layers_str,
                 "controlnet_model_name": controlnet_model_name,
                 "controlnet_scale": controlnet_scale,
                 "ip_adapter_model_name": ip_adapter_model_name,
                 "ip_adapter_scale": ip_adapter_scale,
                 "rating": 0,
                 "notes": ""
            }
            preview_image_for_preset = final_image.copy()
            output_gen_data_json = json.dumps(current_data_for_preset)
            output_preview_image = preview_image_for_preset

            temps_generation_image = f"{(time.time() - depart_time):.2f} sec"
            date_str = datetime.now().strftime("%Y_%m_%d")
            heure_str = datetime.now().strftime("%H_%M_%S")
            save_dir = os.path.join(SAVE_DIR, date_str)
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{date_str}_{heure_str}_{seed}_{width}x{height}_{idx+1}.{IMAGE_FORMAT.lower()}"
            chemin_image = os.path.join(save_dir, filename)

            lora_info_str = ", ".join([f"{name}({weight['scale']:.2f})" for name, weight in model_manager.loaded_loras.items()]) if model_manager.loaded_loras else translate("aucun_lora", translations)
            style_info_str = ", ".join(selected_style_display_names) if selected_style_display_names else translate("Aucun_style", translations)

            donnees_xmp = {
                 "Module": "SDXL Image Generation", "Creator": AUTHOR,
                 "Model": os.path.splitext(model_manager.current_model_name)[0] if model_manager.current_model_name else "N/A", 
                 "VAE": model_manager.current_vae_name, 
                 "Steps": num_steps, "Guidance": guidance_scale,
                 "Sampler": pipe.scheduler.__class__.__name__,
                 "IMAGE": f"{idx+1} {translate('image_sur',translations)} {num_images}",
                 "Inference": num_steps, "Style": style_info_str,
                 "original_prompt (User)": prompt_to_log_as_original,
                 "Prompt": prompt_en,
                 "LLM_Enhanced": prompt_is_currently_enhanced,
                 "Negatif Prompt": negative_prompt_str, "Seed": seed,
                 "Size": selected_format_parts, 
                 "Loras": lora_info_str,
                 "Generation Time": temps_generation_image,
                 "PAG Enabled": pag_enabled,
                 "PAG Scale": f"{pag_scale:.2f}" if pag_enabled else "N/A", 
                 "PAG Custom Pipeline": "hyoungwoncho/sd_perturbed_attention_guidance" if pag_enabled else "N/A", 
                 "PAG Applied Layers": pag_applied_layers_str if pag_enabled else "N/A"
             }

            metadata_structure, prep_message = preparer_metadonnees_image(
                final_image, donnees_xmp, translations, chemin_image
            )
            print(txt_color("[INFO]", "info"), prep_message)

            gr.Info(translate("image_sauvegarder", translations) + " " + chemin_image, 3.0)
            image_future = image_executor.submit(enregistrer_image, final_image, chemin_image, translations, IMAGE_FORMAT, metadata_to_save=metadata_structure)
            html_future = html_executor.submit(enregistrer_etiquettes_image_html, chemin_image, donnees_xmp, translations, is_last_image=(idx == num_images - 1))

            print(txt_color("[OK] ","ok"),translate("image_sauvegarder", translations), txt_color(f"{filename}","ok"))
            image_paths.append(chemin_image)
            
            del final_image
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            empty_working_set(translations)

            seed_strings.append(f"[{seed}]")
            formatted_seeds = " ".join(seed_strings)
 
            try:
                image_future.result(timeout=30)
                html_message_result = html_future.result(timeout=10)
            except Exception as e:
                 print(txt_color("[ERREUR]", "erreur"), f"Erreur lors de la finalisation de la sauvegarde: {e}")
                 html_message_result = translate("erreur_lors_generation_html", translations)

            yield (
                image_paths, formatted_seeds, f"{idx+1}{translate('image_sur', translations)} {num_images} {translate('images_generees', translations)}",
                html_message_result, None, final_progress_html,
                bouton_charger_update_off, btn_generate_update_off,
                gr.update(interactive=False), 
                output_gen_data_json,
                output_preview_image
            )

        final_images = image_paths
        final_seeds = formatted_seeds
        final_html_msg = html_message_result
        final_preview_img = None 
        final_progress_html = "" 

        if not final_message: 
            elapsed_time = f"{(time.time() - start_time):.2f} sec"
            final_message = translate('temps_total_generation', translations) + " : " + elapsed_time
            print(txt_color("[INFO] ","info"), final_message)
            gr.Info(final_message, 3.0)

        final_save_button_state = gr.update(interactive=False)
        if not stop_event.is_set() and not stop_gen.is_set() and final_images:
             final_save_button_state = gr.update(interactive=True)

    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f"{translate('erreur_lors_generation', translations)} : {e}")
        traceback.print_exc()
        final_message = f"{translate('erreur_lors_generation', translations)} : {str(e)}"
        final_images = []
        final_seeds = ""
        final_html_msg = f"<p style='color: red;'>{final_message}</p>"
        gr.Error(final_message)
        final_preview_img = None
        final_progress_html = ""
        final_save_button_state = gr.update(interactive=False)
        output_gen_data_json = None 
        output_preview_image = None

    finally: 
        if (stop_event.is_set() or stop_gen.is_set()) and final_images:
            if 'date_str' in locals() and date_str: 
                chemin_rapport_html_actuel = os.path.join(SAVE_DIR, date_str, "rapport.html")
                finalize_html_report_if_needed(chemin_rapport_html_actuel, translations)
                
        yield (
            final_images, final_seeds, final_message, final_html_msg,
            final_preview_img, final_progress_html,
            bouton_charger_update_on, btn_generate_update_on,
            final_save_button_state, 
            output_gen_data_json, 
            output_preview_image 
        )

def generate_inpainted_image(text, original_image_pil, image_mask_editor_dict, num_steps, strength, guidance_scale, traduire,
                             model_manager, translations, stop_gen, SAVE_DIR, AUTHOR, IMAGE_FORMAT, image_executor, html_executor):
    """Génère une image inpainted avec Stable Diffusion XL."""

    btn_gen_inp_off = gr.update(interactive=False)
    btn_load_inp_off = gr.update(interactive=False)
    btn_gen_inp_on = gr.update(interactive=True)
    btn_load_inp_on = gr.update(interactive=True)

    initial_slider_output = [None, None]
    initial_msg_load = "" 
    initial_msg_status = translate("preparation", translations) 
    initial_progress = "" 

    yield initial_slider_output, initial_msg_load, initial_msg_status, initial_progress, btn_load_inp_off, btn_gen_inp_off

    final_slider_output_result = [None, None]
    final_msg_load_result = ""
    final_msg_status_result = ""
    final_progress_result = ""

    try:
        start_time = time.time()
        stop_gen.clear()
        if model_manager.get_current_pipe() is None:
            msg = translate("erreur_pas_modele_inpainting", translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, 4.0)
            final_msg_status_result = msg
            yield (
                [None, None], "", final_msg_status_result, "",
                btn_load_inp_on, btn_gen_inp_on
            )
            return
            
        pipe = model_manager.get_current_pipe() 
        if not isinstance(pipe, StableDiffusionXLInpaintPipeline):
            error_message = translate("erreur_mauvais_type_modele_inpainting", translations)
            print(txt_color("[ERREUR] ", "erreur"), error_message)
            gr.Warning(error_message, 4.0)
            final_msg_status_result = error_message
            final_msg_load_result = f"<p style='color: red;'>{error_message}</p>"
            yield (
                [None, None], final_msg_load_result, final_msg_status_result, "",
                btn_load_inp_on, btn_gen_inp_on
            )
            return

        if original_image_pil is None or image_mask_editor_dict is None:
            msg = translate("erreur_image_mask_manquant", translations)
            print(txt_color("[ERREUR] ", "erreur"), msg)
            gr.Warning(msg, 4.0)
            final_msg_status_result = msg
            yield (
                [None, None], "", final_msg_status_result, "",
                btn_load_inp_on, btn_gen_inp_on
            )
            return

        pipeline_mask_pil = create_opaque_mask_from_editor(image_mask_editor_dict, original_image_pil.size, translations)

        actual_total_steps = math.ceil(num_steps * strength)
        if actual_total_steps <= 0:
            actual_total_steps = 1
            final_msg_status_result = translate("erreur_strength_trop_faible", translations)
            gr.Warning(final_msg_status_result, 3.0)
            yield (
                [original_image_pil, None], "", final_msg_status_result, "",
                btn_load_inp_on, btn_gen_inp_on
            )
            return

        prompt_text = translate_prompt(text, translations) if traduire else text
        compel = model_manager.get_current_compel()
        conditioning, pooled = compel(prompt_text)
        
        active_adapters = pipe.get_active_adapters()
        for adapter_name in active_adapters:
             if hasattr(pipe, 'set_adapters'):
                pipe.set_adapters(adapter_name, 0)
        
        image_rgb = original_image_pil.convert("RGB")
        final_image_container = {}

        progress_update_queue = queue.Queue()

        inpainting_callback = create_inpainting_callback(
            stop_gen,
            actual_total_steps, 
            translations,
            progress_update_queue 
        )

        def run_pipeline():
            print(txt_color("[INFO] ", "info"), translate("debut_inpainting", translations))
            gr.Info(translate("debut_inpainting", translations), 3.0)
            try:
                inpainted_image_result = pipe(
                    pooled_prompt_embeds=pooled,
                    prompt_embeds=conditioning,
                    image=image_rgb,
                    mask_image=pipeline_mask_pil,
                    width=original_image_pil.width,
                    height=original_image_pil.height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    callback_on_step_end=inpainting_callback
                ).images[0]
                if not stop_gen.is_set():
                    final_image_container["final"] = inpainted_image_result
            except InterruptedError:
                 print(txt_color("[INFO]", "info"), translate("inpainting_interrompu_interne", translations))
            except Exception as e:
                 if not (hasattr(pipe, '_interrupt') and pipe._interrupt):
                     print(txt_color("[ERREUR]", "erreur"), f"Erreur dans run_pipeline (inpainting): {e}")
                     final_image_container["error"] = e

        thread = threading.Thread(target=run_pipeline)
        thread.start()

        last_progress_html = "" 

        while thread.is_alive() or not progress_update_queue.empty():
            current_step, total_steps = None, actual_total_steps
            while not progress_update_queue.empty():
                try:
                    current_step, total_steps = progress_update_queue.get_nowait()
                except queue.Empty: break
            new_progress_html = last_progress_html 
            if current_step is not None:
                progress_percent = int((current_step / total_steps) * 100)
                new_progress_html = create_progress_bar_html(current_step, total_steps, progress_percent)
                yield [original_image_pil, None], gr.update(), gr.update(), new_progress_html, btn_load_inp_off, btn_gen_inp_off
            time.sleep(0.05)  
        thread.join()

        current_final_progress_html = ""

        if not stop_gen.is_set() and "error" not in final_image_container:
             final_progress_html = create_progress_bar_html(actual_total_steps, actual_total_steps, 100)
        elif "error" in final_image_container:
             final_progress_html = f'<p style="color: red;">{translate("erreur_lors_inpainting", translations)}</p>'
        
        if hasattr(pipe, '_interrupt'):
            pipe._interrupt = False
        
        if "error" in final_image_container:
             error_message = f"{translate('erreur_lors_inpainting', translations)}: {final_image_container['error']}"
             print(txt_color("[ERREUR]", "erreur"), error_message)
             final_msg_status_result = translate('erreur_lors_inpainting', translations)
             final_msg_load_result = error_message
             current_final_progress_html = f'<p style="color: red;">{translate("erreur_lors_inpainting", translations)}</p>'

        elif stop_gen.is_set():
            print(txt_color("[INFO]", "info"), translate("inpainting_arrete_pas_sauvegarde", translations))
            gr.Info(translate("inpainting_arrete_pas_sauvegarde", translations), 3.0)
            final_msg_status_result = translate("inpainting_arrete", translations)
            final_msg_load_result = translate("inpainting_arrete_pas_sauvegarde", translations)
        
        else:
            inpainted_image = final_image_container.get("final", None)
            if inpainted_image is None:
                 err_msg = translate("erreur_pas_image_inpainting", translations)
                 print(txt_color("[ERREUR]", "erreur"), err_msg)
                 gr.Warning(err_msg, 4.0)
                 final_msg_status_result = translate("erreur_lors_inpainting", translations)
                 final_msg_load_result = err_msg
            else:
                final_slider_output_result = [original_image_pil, inpainted_image]
                current_final_progress_html = create_progress_bar_html(actual_total_steps, actual_total_steps, 100)
       
                temps_generation_image = f"{(time.time() - start_time):.2f} sec"
                date_str = datetime.now().strftime("%Y_%m_%d")
                heure_str = datetime.now().strftime("%H_%M_%S")
                save_dir = os.path.join(SAVE_DIR, date_str)
                os.makedirs(save_dir, exist_ok=True)
                filename = f"inpainting_{date_str}_{heure_str}_{original_image_pil.width}x{original_image_pil.height}.{IMAGE_FORMAT.lower()}"
                chemin_image = os.path.join(save_dir, filename)
                donnees_xmp = {
                    "Module": "SDXL Inpainting",
                    "Creator": AUTHOR,
                    "Model": os.path.splitext(model_manager.current_model_name)[0] if model_manager.current_model_name else "N/A",
                    "Steps": num_steps, "Guidance": guidance_scale, "Strength": strength,
                    "Prompt": prompt_text, "Size": f"{original_image_pil.width}x{original_image_pil.height}",
                    "Generation Time": temps_generation_image
                }

                metadata_structure, prep_message = preparer_metadonnees_image(
                    inpainted_image, donnees_xmp, translations, chemin_image
                )

                print(txt_color("[INFO]", "info"), prep_message)

                image_future = image_executor.submit(enregistrer_image, inpainted_image, chemin_image, translations, IMAGE_FORMAT, metadata_to_save=metadata_structure)
                html_future = html_executor.submit(enregistrer_etiquettes_image_html, chemin_image, donnees_xmp, translations, is_last_image=True)
        
                try:
                    image_future.result(timeout=30)
                    html_message_result = html_future.result(timeout=10)
                    final_msg_load_result = html_message_result
                except Exception as e:
                    print(txt_color("[ERREUR]", "erreur"), f"Erreur lors de la finalisation de la sauvegarde (inpainting): {e}")
                    final_msg_load_result = translate("erreur_lors_generation_html", translations)

                final_msg_status_result = translate("inpainting_reussi", translations)
                print(txt_color("[OK] ", "ok"), translate("fin_inpainting", translations))
                gr.Info(translate("fin_inpainting", translations), 3.0)

        final_progress_result = current_final_progress_html

    except Exception as e:
        err_msg = f"{translate('erreur_lors_inpainting', translations)}: {e}"
        print(txt_color("[ERREUR] ", "erreur"), err_msg)
        traceback.print_exc()
        final_msg_status_result = translate('erreur_lors_inpainting', translations)
        final_msg_load_result = str(e)
        gr.Error(err_msg)

    finally:
        empty_working_set(translations)
        yield final_slider_output_result, final_msg_load_result, final_msg_status_result, final_progress_result, btn_load_inp_on, btn_gen_inp_on
