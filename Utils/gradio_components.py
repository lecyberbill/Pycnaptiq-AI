import gradio as gr
from Utils.utils import translate, txt_color
from core.translator import translate_prompt

def _smart_translate(key, module_translations, prefix=None):
    """
    Tente de traduire avec le préfixe, puis sans le préfixe.
    """
    if prefix:
        prefixed_key = f"{prefix}{key}"
        result = translate(prefixed_key, module_translations)
        # La fonction translate renvoie [key] si non trouvé
        if result != f"[{prefixed_key}]":
            return result
    return translate(key, module_translations)

def create_prompt_interface(module_instance, module_translations, prefix=None):
    """
    Crée l'interface de prompt standard avec amélioration LLM et traduction.
    """
    prompt = gr.Textbox(
        label=_smart_translate("prompt_label", module_translations, prefix),
        info=_smart_translate("prompt_info", module_translations, prefix),
        placeholder=_smart_translate("prompt_placeholder", module_translations, prefix),
        lines=3,
    )
    with gr.Row():
        enhance_button = gr.Button(
            translate("ameliorer_prompt_ia_btn", module_translations), # Souvent global ou commun
            interactive=True
        )
        validate_button = gr.Button(
            translate("valider_prompt_btn", module_translations),
            interactive=False,
            visible=False
        )
    traduire_checkbox = gr.Checkbox(
        label=translate("traduire_en_anglais", module_translations),
        value=False,
        info=translate("traduire_prompt_libre", module_translations),
    )
    style_dropdown = gr.Dropdown(
        label=translate("selectionner_styles", module_translations),
        choices=[style["name"] for style in module_instance.styles if style["name"] != translate("Aucun_style", module_instance.global_translations)],
        value=[],
        multiselect=True,
        info=translate("selectionner_styles_info", module_translations),
    )
    
    return prompt, enhance_button, validate_button, traduire_checkbox, style_dropdown

def create_generation_settings(module_instance, module_translations, prefix=None, allowed_resolutions=None, default_steps=4, max_steps=50):
    """
    Crée les réglages de génération standards (résolution, steps, guidance, seed, npm_images).
    """
    with gr.Row():
        resolution = gr.Dropdown(
            label=translate("resolution_label", module_instance.global_translations),
            choices=allowed_resolutions or ["1024x1024"],
            value=allowed_resolutions[0] if allowed_resolutions else "1024x1024",
            interactive=True
        )
        steps = gr.Slider(
            minimum=1, maximum=max_steps, value=default_steps, step=1,
            label=_smart_translate("steps_label", module_translations, prefix)
        )
        guidance_scale = gr.Slider(
            minimum=0.0, maximum=20.0, value=0.0, step=0.1,
            label=_smart_translate("guidance_scale_label", module_translations, prefix)
        )
    with gr.Row():
        seed = gr.Number(
            label=translate("seed_label", module_instance.global_translations),
            value=-1,
            info=translate("seed_info_neg_one_random", module_instance.global_translations)
        )
        num_images = gr.Slider(
            minimum=1, maximum=20, value=1, step=1,
            label=translate("nombre_images", module_translations),
            interactive=True
        )
    
    return resolution, steps, guidance_scale, seed, num_images

def create_lora_interface(module_instance, module_translations, num_slots=2):
    """
    Crée l'interface de gestion des LoRA.
    """
    with gr.Accordion(translate("lora_section_title", module_translations), open=False):
        lora_checks = []
        lora_dropdowns = []
        lora_scales = []
        
        for i in range(1, num_slots + 1):
            with gr.Group():
                l_check = gr.Checkbox(label=f"LoRA {i}", value=False)
                l_dropdown = gr.Dropdown(
                    choices=module_instance.lora_choices_for_ui,
                    label=translate("selectionner_lora", module_instance.global_translations),
                    interactive=module_instance.has_loras
                )
                l_scale = gr.Slider(0, 1, value=0.8, label=translate("poids_lora", module_instance.global_translations))
                
                lora_checks.append(l_check)
                lora_dropdowns.append(l_dropdown)
                lora_scales.append(l_scale)

                l_check.change(
                    fn=lambda chk, has_loras_flag: gr.update(interactive=chk and has_loras_flag),
                    inputs=[l_check, gr.State(module_instance.has_loras)],
                    outputs=[l_dropdown]
                )
        
        lora_message = gr.Textbox(label=translate("message_lora", module_instance.global_translations), interactive=False)
        refresh_button = gr.Button(translate("refresh_lora_list", module_translations), variant="secondary")

    return lora_checks, lora_dropdowns, lora_scales, lora_message, refresh_button

def create_output_interface(module_translations, prefix=None):
    """
    Crée l'interface de sortie (Galerie, Boutons Gen/Stop, Progress).
    """
    status_box = gr.Textbox(
        label=_smart_translate("model_status", module_translations, prefix),
        value=_smart_translate("model_not_loaded", module_translations, prefix),
        interactive=False,
    )
    result_gallery = gr.Gallery(
        label=translate("output_image", module_translations),
        elem_id="result_gallery",
        columns=2, height="auto"
    )
    with gr.Row():
        gen_button = gr.Button(
            value=_smart_translate("generate_button", module_translations, prefix),
            interactive=True,
            variant="primary"
        )
        stop_button = gr.Button(
            translate("arreter", module_translations),
            interactive=False, variant="stop",
        )
    progress_html = gr.HTML()
    
    return status_box, result_gallery, gen_button, stop_button, progress_html
