import gradio as gr
import time
from Utils.utils import translate
from Utils.sampler_utils import get_sampler_key_from_display_name, apply_sampler_to_pipe
from core.batch_runner import run_batch_from_json
from Utils.gest_mem import update_memory_stats
from core.image_prompter import generate_prompt_from_image, DEFAULT_FLORENCE2_TASK
from core.sdxl_logic import generate_image, generate_inpainted_image
from Utils.preset_handlers import handle_save_preset

def handle_sampler_change_logic(selected_display_name, model_manager, translations):
    """
    Logic for changing the sampler.
    Returns: (message, success, sampler_key)
    """
    sampler_key = get_sampler_key_from_display_name(selected_display_name, translations)
    
    if sampler_key and model_manager.get_current_pipe() is not None:
        message, success = apply_sampler_to_pipe(model_manager.get_current_pipe(), sampler_key, translations)
        if success:
            model_manager.current_sampler_key = sampler_key
            gr.Info(message, 3.0)
        else:
            gr.Warning(message, 4.0)
        return message, success, sampler_key
    else:
        if model_manager.get_current_pipe() is None:
            error_msg = translate("erreur_pas_modele_pour_sampler", translations)
            gr.Warning(error_msg, 4.0)
        else:
            error_msg = f"{translate('erreur_sampler_inconnu', translations)}: {selected_display_name}"
            gr.Warning(error_msg, 4.0)
        return error_msg, False, None

def batch_runner_wrapper_logic(model_manager, stop_event, *args, progress=gr.Progress(track_tqdm=True)):
    """Wrapper that calls run_batch_from_json with yield from."""
    yield from run_batch_from_json(
        model_manager,
        stop_event,
        *args,
        progress=progress
    )

def toggle_pag_scale_visibility_logic(pag_enabled):
    """Updates visibility of PAG-related sliders."""
    return gr.update(visible=pag_enabled), gr.update(visible=pag_enabled)

def generate_prompt_ui_logic(image, current_translations, florence2_model_id):
    """Wraps Florence-2 prompt generation for the UI."""
    return generate_prompt_from_image(
        image,
        current_translations,
        task=DEFAULT_FLORENCE2_TASK,
        unload_after=True,
        model_id=florence2_model_id
    )

def stream_live_memory_stats_logic(enable_live, update_interval, translations_state, device_state):
    """Generator for streaming memory stats to the UI."""
    if not enable_live:
        yield update_memory_stats(translations_state, device_state)
        return

    last_update_time = 0
    while True:
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            yield update_memory_stats(translations_state, device_state)
            last_update_time = current_time
        time.sleep(0.2)

def generate_image_ui_wrapper(model_manager, translations, config, device, stop_event, stop_gen, SAVE_DIR, AUTHOR, IMAGE_FORMAT, image_executor, html_executor, STYLES, NEGATIVE_PROMPT, PREVIEW_QUEUE, *args):
    """Bridge for standard image generation."""
    ui_inputs = list(args[:25])
    lora_inputs = list(args[25:])
    dependencies = [
        model_manager, translations, config, device, stop_event, stop_gen, SAVE_DIR, AUTHOR, IMAGE_FORMAT, 
        image_executor, html_executor, STYLES, NEGATIVE_PROMPT, PREVIEW_QUEUE
    ]
    yield from generate_image(*(ui_inputs + dependencies + lora_inputs))

def generate_inpainted_image_ui_wrapper(model_manager, translations, stop_gen, SAVE_DIR, AUTHOR, IMAGE_FORMAT, image_executor, html_executor, *args):
    """Bridge for inpainting image generation."""
    ui_inputs = list(args[:7])
    dependencies = [
        model_manager, translations, stop_gen, SAVE_DIR, AUTHOR, IMAGE_FORMAT, 
        image_executor, html_executor
    ]
    yield from generate_inpainted_image(*(ui_inputs + dependencies))

def handle_save_preset_ui_wrapper(preset_manager, translations, *args):
    """Bridge for saving presets."""
    return handle_save_preset(
        *args,
        preset_manager=preset_manager,
        translations=translations
    )
