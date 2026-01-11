import math
import json
import traceback
import gradio as gr
from PIL import Image
from io import BytesIO
from functools import partial
from Utils.utils import txt_color, translate
from Utils.sampler_utils import apply_sampler_to_pipe

def get_filter_options(preset_manager, translations):
    """Récupère les options de filtrage pour les presets."""
    filter_data = preset_manager.get_distinct_preset_filters()
    models = filter_data.get('models', [])
    sampler_keys_in_presets = filter_data.get('samplers', [])
    loras = filter_data.get('loras', [])
    sampler_display_names = [translate(s_key, translations) for s_key in sampler_keys_in_presets]
    sampler_display_names = list(set(sampler_display_names))
    sampler_display_names.sort()
    return models, sampler_display_names, loras

def update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras, preset_manager, presets_per_page):
    """Met à jour le Dropdown de pagination."""
    all_presets_data = preset_manager.load_presets_for_display(
        preset_type='gen_image', search_term=search, sort_by=sort,
        selected_models=filter_models or None, selected_samplers=filter_samplers or None, selected_loras=filter_loras or None
    )
    total_presets = len(all_presets_data)
    total_pages = math.ceil(total_presets / presets_per_page) if total_presets > 0 else 1
    current_page = max(1, min(page, total_pages))

    page_choices = list(range(1, total_pages + 1))

    return gr.update(
        choices=page_choices,
        value=current_page,
        interactive=(total_pages > 1)
    )

def handle_preset_rename_click(preset_id, current_trigger, translations): 
    print(txt_color("[INFO]", "info"), f"{translate('preset_action_rename_click', translations)} {translate('current_trigger_log', translations)}: {current_trigger}")
    return gr.update(value=preset_id), gr.update(value=current_trigger + 1)

def handle_preset_cancel_click(current_trigger): 
    print("[Action Preset] Clic Annuler Édition.")
    return gr.update(value=None), gr.update(value=current_trigger + 1)

def handle_preset_rename_submit(preset_id, new_name, current_trigger, preset_manager, translations):
    """Soumet le nouveau nom pour le preset."""
    print(f"[Action Preset {preset_id}] Submit Renommer vers '{new_name}'.")
    if not preset_id or not new_name:
        gr.Warning(translate("erreur_nouveau_nom_vide", translations))
        return gr.update(value=preset_id), gr.update() 

    success, message = preset_manager.rename_preset(preset_id, new_name)
    if success:
        gr.Info(message)
        return gr.update(value=None), gr.update(value=current_trigger + 1)
    else:
        gr.Warning(message)
        return gr.update(value=preset_id), gr.update()

def handle_preset_delete_click(preset_id, current_trigger, page, search, sort, filter_models, filter_samplers, filter_loras, current_search_value, preset_manager, translations, presets_per_page):
    """Supprime le preset, déclenche un refresh ET met à jour la pagination."""
    trigger_update_on_error = gr.update()
    pagination_update_on_error = gr.update()
    search_update_on_error = gr.update()

    success, message = preset_manager.delete_preset(preset_id)
    if success:
        gr.Info(message)
        pagination_update = update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras, preset_manager, presets_per_page)
        current_search_str = current_search_value if current_search_value else ""
        if current_search_str.endswith(" "):
            temp_search_value = current_search_str[:-1] 
        else:
            temp_search_value = current_search_str + " " 
        
        search_update_hack = gr.update(value=temp_search_value)
        return gr.update(value=current_trigger + 1), pagination_update, search_update_hack
    else:
        gr.Warning(message)
        return trigger_update_on_error, pagination_update_on_error, search_update_on_error

def handle_preset_rating_change(preset_id, new_rating_value, preset_manager):
    """Met à jour la note du preset."""
    if preset_id is not None and new_rating_value is not None:
        success, message = preset_manager.update_preset_rating(preset_id, int(new_rating_value))
        if not success:
            gr.Warning(message)

def update_pagination_and_trigger_refresh(page, search, sort, filter_models, filter_samplers, filter_loras, current_trigger, preset_manager, presets_per_page):
    """Met à jour l'UI de pagination ET incrémente le trigger de refresh."""
    pagination_updates = update_pagination_display(page, search, sort, filter_models, filter_samplers, filter_loras, preset_manager, presets_per_page)
    trigger_update = gr.update(value=current_trigger + 1)
    return list(pagination_updates) + [trigger_update]

def handle_page_change(direction, current_page):
    """Calcule la nouvelle page."""
    new_page = current_page + direction
    return gr.update(value=new_page)

def update_filter_choices_after_save(preset_manager, translations):
    models, samplers, loras = get_filter_options(preset_manager, translations)
    return gr.update(choices=models), gr.update(choices=samplers), gr.update(choices=loras)

def handle_save_preset(preset_name, preset_notes, current_gen_data_json, preview_image_pil, current_trigger, current_search_value, preset_manager, translations):
    """
    Gère l'appel à preset_manager pour sauvegarder le preset.
    """
    filter_updates_on_error = [gr.update(), gr.update(), gr.update()]
    trigger_update_on_error = gr.update()
    search_update_on_error = gr.update()

    if not preset_name:
        gr.Warning(translate("erreur_nom_preset_vide", translations), 3.0)
        return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error

    current_gen_data = None
    if isinstance(current_gen_data_json, str):
        try:
            current_gen_data = json.loads(current_gen_data_json)
        except json.JSONDecodeError as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_decodage_json_preset', translations)}: {e}")
            gr.Warning(translate("erreur_interne_decodage_json", translations), 3.0)
            return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error
    else:
         if isinstance(current_gen_data_json, dict):
             current_gen_data = current_gen_data_json
         else:
             print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_type_inattendu_json_preset', translations)}: {type(current_gen_data_json)}")
             gr.Warning(translate("erreur_interne_donnees_generation_invalides", translations), 3.0)
             return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error

    if not isinstance(preview_image_pil, Image.Image):
         if isinstance(preview_image_pil, bytes):
             try:
                 preview_image_pil = Image.open(BytesIO(preview_image_pil))
                 print(txt_color("[INFO]", "info"), translate("info_image_preview_chargee_bytes", translations))
             except Exception as img_err:
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_chargement_image_preview_bytes', translations)}: {img_err}")
                 preview_image_pil = None 
         else:
             preview_image_pil = None 

    if not isinstance(current_gen_data, dict) or preview_image_pil is None:
         print(txt_color("[ERREUR]", "erreur"), translate("erreur_donnees_generation_ou_image_manquantes", translations))
         gr.Warning(translate("erreur_pas_donnees_generation", translations), 3.0)
         return gr.update(), gr.update(), *filter_updates_on_error,  trigger_update_on_error, search_update_on_error

    data_to_save = current_gen_data.copy()
    data_to_save['notes'] = preset_notes
    data_to_save['original_user_prompt'] = current_gen_data.get('original_user_prompt', data_to_save.get('prompt', ''))
    data_to_save['current_prompt_is_enhanced'] = current_gen_data.get('current_prompt_is_enhanced', False)
    data_to_save['enhancement_cycle_active'] = current_gen_data.get('enhancement_cycle_active', False)

    try:
        success, message = preset_manager.save_gen_image_preset(preset_name, data_to_save, preview_image_pil)
    except Exception as save_err:
        print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_appel_sauvegarde_preset', translations)}: {save_err}")
        traceback.print_exc()
        success = False
        message = translate("erreur_interne_sauvegarde_preset", translations)

    if success: 
        gr.Info(message, 3.0)
        try:
            update_model, update_sampler, update_lora = update_filter_choices_after_save(preset_manager, translations)
            current_search_str = current_search_value if current_search_value else ""
            if current_search_str.endswith(" "):
                temp_search_value = current_search_str[:-1] 
            else:
                temp_search_value = current_search_str + " " 
            search_update_hack = gr.update(value=temp_search_value)
            return gr.update(value=""), gr.update(value=""), update_model, update_sampler, update_lora, gr.update(value=current_trigger + 1), search_update_hack
        except Exception as filter_err:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_update_filtres_preset', translations)}: {filter_err}")
            return gr.update(value=""), gr.update(value=""), *filter_updates_on_error, gr.update(value=current_trigger + 1), search_update_on_error
    else:
        gr.Warning(message, 4.0)
        return gr.update(), gr.update(), *filter_updates_on_error, trigger_update_on_error, search_update_on_error

def handle_preset_load_click(preset_id, preset_manager, model_manager, translations, config, STYLES, FORMATS):
    """
    Charge les données d'un preset et met à jour les contrôles de l'UI de génération.
    """
    print(f"[Action Preset {preset_id}] Clic Charger.")
    preset_data = preset_manager.load_preset_data(preset_id)

    if preset_data is None:
        msg = translate("erreur_chargement_preset_introuvable", translations).format(preset_id)
        gr.Warning(msg)
        num_lora_slots = 4
        return [gr.update()] * (6 + 7 + 3 * num_lora_slots + 3 + 1)

    try:
        prompt_to_display = preset_data.get('prompt', '')
        original_user_prompt_from_preset = preset_data.get('original_user_prompt', prompt_to_display)
        current_prompt_is_enhanced_from_preset = preset_data.get('current_prompt_is_enhanced', False)
        enhancement_cycle_active_from_preset = preset_data.get('enhancement_cycle_active', False)
        last_ai_output_from_preset = preset_data.get('last_ai_enhanced_output', None)

        styles_data = preset_data.get('styles', [])
        loaded_style_names = []
        if isinstance(styles_data, str):
            try:
                loaded_style_names = json.loads(styles_data)
            except json.JSONDecodeError:
                loaded_style_names = [] 
        elif isinstance(styles_data, list):
            loaded_style_names = styles_data

        model_name = preset_data.get('model', None)
        raw_vae_name_from_preset = preset_data.get('vae')

        vae_to_attempt_loading = "Auto" if raw_vae_name_from_preset in [None, "Défaut VAE"] else raw_vae_name_from_preset

        guidance = preset_data.get('guidance_scale', 7.0)
        steps = preset_data.get('num_steps', 30)
        sampler_key = preset_data.get('sampler_key') or preset_data.get('sampler') or 'sampler_euler'
        sampler_display = translate(sampler_key, translations) 
        seed_val = preset_data.get('seed', -1)
        width = preset_data.get('width', 1024)
        height = preset_data.get('height', 1024)
        loras_data = preset_data.get('loras', [])
        
        custom_pipeline_id_preset = preset_data.get('custom_pipeline_id')
        pag_enabled_preset = bool(custom_pipeline_id_preset) 
        pag_scale_preset = preset_data.get('pag_scale', 1.5)
        pag_applied_layers_preset = preset_data.get('pag_applied_layers', "m0") 

        available_models = model_manager.list_models(model_type="standard", gradio_mode=True)
        model_update = gr.update()
        if model_name and model_name in available_models:
            model_update = gr.update(value=model_name)
        elif model_name:
            print(txt_color("[AVERTISSEMENT]", "warning"), f"{translate('preset_load_model_not_found_warn', translations)}")

        available_vae_choices_for_ui = model_manager.list_vaes()
        final_vae_for_ui_dropdown = "Auto"
        if vae_to_attempt_loading in available_vae_choices_for_ui:
            final_vae_for_ui_dropdown = vae_to_attempt_loading
        elif vae_to_attempt_loading != "Auto":
            gr.Warning(translate("erreur_vae_preset_introuvable", translations).format(vae_to_attempt_loading))
        
        vae_update = gr.update(value=final_vae_for_ui_dropdown)

        loaded_loras = []
        if isinstance(loras_data, str):
            try:
                loaded_loras = json.loads(loras_data)
            except json.JSONDecodeError:
                loaded_loras = [] 
        elif isinstance(loras_data, list):
            loaded_loras = loras_data

        format_string = f"{width}*{height}"
        orientation_key = None
        for fmt in config["FORMATS"]:
            if fmt.get("dimensions") == f"{width}*{height}":
                orientation_key = fmt.get("orientation")
                break
        if orientation_key:
            format_string = f"{width}*{height} : {translate(orientation_key, translations)}"
        else:
            format_string = f"{width}*{height} : {translate('orientation_inconnue', translations)}" 

        # FORMATS is now format_choices (list of strings)
        if format_string not in FORMATS:
             # Try to find a match by dimensions only if exact match fails
             found_match = False
             for choice in FORMATS:
                 if choice.startswith(f"{width}*{height}"):
                     format_string = choice
                     found_match = True
                     break
             if not found_match:
                 format_string = FORMATS[3] if len(FORMATS) > 3 else FORMATS[0]

        num_lora_slots = 4
        lora_check_updates = [gr.update(value=False) for _ in range(num_lora_slots)]
        lora_dd_updates = [gr.update(value=None, interactive=False, choices=[translate("aucun_lora_disponible", translations)]) for _ in range(num_lora_slots)]
        lora_scale_updates = [gr.update(value=0) for _ in range(num_lora_slots)]

        available_loras = model_manager.list_loras(gradio_mode=True)
        has_available_loras = bool(available_loras) and translate("aucun_modele_trouve", translations) not in available_loras
        lora_choices = available_loras if has_available_loras else [translate("aucun_lora_disponible", translations)]

        for i, lora_info in enumerate(loaded_loras):
            if i >= num_lora_slots: break 
            l_name = lora_info.get('name')
            l_weight = lora_info.get('weight')
            if l_name in lora_choices:
                lora_check_updates[i] = gr.update(value=True)
                lora_dd_updates[i] = gr.update(choices=lora_choices, value=l_name, interactive=True)
                lora_scale_updates[i] = gr.update(value=l_weight)

        sampler_update_msg, success = apply_sampler_to_pipe(model_manager.get_current_pipe(), sampler_key, translations)
        if success:
            sampler_update = gr.update(value=sampler_display) 
            gr.Info(sampler_update_msg, 2.0) 
        else:
            gr.Warning(sampler_update_msg)
            sampler_update = gr.update(value=translate("sampler_euler", translations))

        valid_style_choices = [style["name"] for style in STYLES if style["name"] != translate("Aucun_style", translations)]
        final_style_selection = [s_name for s_name in loaded_style_names if s_name in valid_style_choices]

        enhance_button_text_update = translate("refaire_amelioration_btn", translations) if enhancement_cycle_active_from_preset else translate("ameliorer_prompt_ia_btn", translations)
        enhance_button_interactive_update = bool(prompt_to_display.strip())
        validate_button_visible_update = enhancement_cycle_active_from_preset
        validate_button_interactive_update = enhancement_cycle_active_from_preset

        outputs_list = [
            model_update, 
            vae_update, 
            gr.update(value=prompt_to_display), # text_input
            gr.update(value=final_style_selection),  
            gr.update(value=guidance),               
            gr.update(value=steps),                  
            gr.update(value=format_string),          
            sampler_update,                          
            gr.update(value=seed_val),               
            *lora_check_updates,                     
            *lora_dd_updates,                        
            *lora_scale_updates,                     
            gr.update(value=pag_enabled_preset),     
            gr.update(value=pag_scale_preset),       
            gr.update(value=pag_applied_layers_preset), 
            gr.update(value=original_user_prompt_from_preset), # original_user_prompt_state
            gr.update(value=current_prompt_is_enhanced_from_preset), # current_prompt_is_enhanced_state
            gr.update(value=enhancement_cycle_active_from_preset), # enhancement_cycle_active_state
            gr.update(value=last_ai_output_from_preset), # last_ai_enhanced_output_state
            gr.update(value=enhance_button_text_update, interactive=enhance_button_interactive_update), # enhance_or_redo_button
            gr.update(visible=validate_button_visible_update, interactive=validate_button_interactive_update), # validate_prompt_button
            gr.update(value=translate("preset_charge_succes", translations).format(preset_data.get('name', f'ID: {preset_id}')))
        ]
        
        return outputs_list
    except Exception as e:
        print(txt_color("[ERREUR]", "erreur"), f"Erreur lors du chargement du preset {preset_id}: {e}")
        traceback.print_exc()
        gr.Warning(translate("erreur_generale_chargement_preset", translations))
        num_lora_slots = 4 
        return [gr.update()] * (29) # 6 + 7 + 12 + 3 + 1 = 29

def render_presets_with_decorator(
    page, search, sort, filter_models, filter_samplers, filter_loras, trigger_val,
    preset_manager, model_manager, translations, config, STYLES, FORMATS, 
    PRESETS_PER_PAGE, PRESET_COLS_PER_ROW,
    gen_ui_outputs_for_preset_load, delete_inputs, delete_outputs
):
    """
    Récupère les presets et crée l'UI dynamiquement.
    """
    def safe_get_from_row(row, key, default=None):
        try:
            return row[key] if key in row.keys() else default
        except (IndexError, TypeError): return default

    sampler_keys_to_filter = []
    if filter_samplers:
        reverse_sampler_map = {v: k for k, v in translations.items() if k.startswith("sampler_")}
        for sampler_display_name in filter_samplers:
            internal_key = reverse_sampler_map.get(sampler_display_name)
            if internal_key:
                sampler_keys_to_filter.append(internal_key)
            else:
                print(txt_color("[ERREUR]", "erreur"), translate("gestion_samplers_erreur", translations).format(sampler_display_name))

    all_presets_data = preset_manager.load_presets_for_display(
        preset_type='gen_image', search_term=search, sort_by=sort,
        selected_models=filter_models or None,
        selected_samplers=sampler_keys_to_filter or None,
        selected_loras=filter_loras or None
    )

    total_presets = len(all_presets_data)
    total_pages = math.ceil(total_presets / PRESETS_PER_PAGE) if total_presets > 0 else 1
    current_page = max(1, min(page, total_pages))
    start_index = (current_page - 1) * PRESETS_PER_PAGE
    end_index = start_index + PRESETS_PER_PAGE
    presets_for_page = all_presets_data[start_index:end_index]

    if not presets_for_page:
        gr.Markdown(f"*{translate('aucun_preset_trouve', translations)}*", key="no_presets_found_md")
    else:
        num_rows_for_page = math.ceil(len(presets_for_page) / PRESET_COLS_PER_ROW)
        preset_idx_on_page = 0
        for r in range(num_rows_for_page):
            with gr.Row(equal_height=False):
                for c in range(PRESET_COLS_PER_ROW):
                    if preset_idx_on_page < len(presets_for_page):
                        preset_data = presets_for_page[preset_idx_on_page]
                        preset_id = safe_get_from_row(preset_data, "id", f"ERREUR_ID_{preset_idx_on_page}")
                        preset_name = safe_get_from_row(preset_data, "name", "ERREUR_NOM")

                        with gr.Column(scale=0, min_width=200):
                            image_bytes = safe_get_from_row(preset_data, "preview_image")
                            preview_img = None
                            if image_bytes:
                                try: preview_img = Image.open(BytesIO(image_bytes))
                                except Exception: pass
                            gr.Image(value=preview_img, height=128, width=128, show_label=True, interactive=False, show_download_button=False, key=f"preset_img_{preset_id}")

                            gr.Textbox(value=preset_name, show_label=False, interactive=False, key=f"preset_name_display_{preset_id}")

                            preset_notes = safe_get_from_row(preset_data, 'notes')
                            if preset_notes:
                                with gr.Accordion(translate("voir_notes", translations), open=False):
                                    gr.Markdown(preset_notes, key=f"preset_notes_md_{preset_id}")

                            rating_value = safe_get_from_row(preset_data, "rating", 0)
                            rating_comp = gr.Radio(
                                choices=[str(r) for r in range(1, 6)], value=str(rating_value) if rating_value > 0 else None,
                                label=translate("evaluation", translations), interactive=True, key=f"preset_rating_{preset_id}"
                            )

                            try:
                                model_name_disp = safe_get_from_row(preset_data, 'model', '?')
                                sampler_key_name = safe_get_from_row(preset_data, 'sampler_key', '?')

                                details_md = f"- **Modèle:** {model_name_disp}\n- **Sampler:** {sampler_key_name}"
                                with gr.Accordion(translate("details_techniques", translations), open=False):
                                    gr.Markdown(details_md, key=f"preset_details_md_{preset_id}")
                            except Exception: pass

                            load_btn = gr.Button(translate("charger", translations) + " 💾", size="sm", key=f"preset_load_{preset_id}")
                            delete_btn = gr.Button(translate("supprimer", translations) + " 🗑️", variant="stop", size="sm", key=f"preset_delete_{preset_id}")

                            if isinstance(preset_id, int):
                                load_btn.click(
                                    fn=partial(handle_preset_load_click,
                                               preset_manager=preset_manager,
                                               model_manager=model_manager,
                                               translations=translations,
                                               config=config,
                                               STYLES=STYLES,
                                               FORMATS=FORMATS),
                                    inputs=[gr.State(preset_id)],
                                    outputs=gen_ui_outputs_for_preset_load
                                )
                                delete_btn.click(
                                    fn=partial(handle_preset_delete_click,
                                               preset_manager=preset_manager,
                                               translations=translations,
                                               presets_per_page=PRESETS_PER_PAGE),
                                    inputs=[gr.State(preset_id)] + delete_inputs,
                                    outputs=delete_outputs
                                )
                                rating_comp.change(
                                    fn=partial(handle_preset_rating_change,
                                               preset_manager=preset_manager),
                                    inputs=[gr.State(preset_id), rating_comp],
                                    outputs=[]
                                )

                        preset_idx_on_page += 1

def reset_page_state_only(translations):
    """Retourne simplement une mise à jour pour mettre l'état de la page à 1."""
    print(txt_color("[INFO]", "info"), translate('reset_page_state_log', translations))
    return gr.update(value=1)

def handle_page_dropdown_change(page_selection, translations):
    """Gère le changement du dropdown de page."""
    print(txt_color("[INFO]", "info"), f"{translate('page_dropdown_change_log', translations).format(page=page_selection)}")
    return page_selection
