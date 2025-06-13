import psutil
import torch
import gradio as gr
import traceback
import gc
import math
import time

# Importations nécessaires depuis utils.py
from .utils import txt_color, translate

# Try importing pynvml for NVIDIA GPU stats
# Install with: pip install pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    _pynvml_available = True
except pynvml.NVMLError as e:
    print(f"[WARN] pynvml not available or failed to initialize: {e}. VRAM stats will be limited.")
    _pynvml_available = False
except ImportError:
    print("[WARN] pynvml not installed. VRAM stats will not be available for NVIDIA GPUs. Install with 'pip install pynvml'.")
    _pynvml_available = False


def get_ram_usage(translations):
    """Gets current RAM usage in GB."""
    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        used_gb = mem.used / (1024**3)
        mem_percent = mem.percent
        cpu_percent_val = psutil.cpu_percent(interval=None) # Non-blocking
        return used_gb, total_gb, mem_percent, cpu_percent_val
    except Exception as e:
        print(f"[ERROR] Failed to get RAM usage: {e}")
        return 0, 0, 0, 0 # Return zeros in case of error

def get_vram_usage(device, translations):
    """Gets current VRAM usage and GPU temperature/utilization for the specified device."""
    gpu_util_percent = 0  # Default to 0 if not available
    # gpu_temp_celsius = None # Default to None - REMOVED

    if device.type == 'cuda' and _pynvml_available:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = mem_info.total / (1024**3)
            used_gb = mem_info.used / (1024**3)
            percent = (used_gb / total_gb) * 100 if total_gb > 0 else 0
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util_percent = util_rates.gpu
            # gpu_temp_celsius = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMP_GPU) # REMOVED
            return used_gb, total_gb, percent, gpu_util_percent # MODIFIED RETURN
        except pynvml.NVMLError as e:
            print(f"[ERROR] Failed to get VRAM/GPU stats for {device}: {e}")
            return 0, 0, 0, 0 # MODIFIED RETURN
        except Exception as e:
            print(f"[ERROR] Unexpected error getting VRAM/GPU stats for {device}: {e}")
            return 0, 0, 0, 0 # MODIFIED RETURN
    elif device.type == 'cuda':
         try:
             total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
             # memory_reserved is often closer to actual usage than memory_allocated
             used_gb = torch.cuda.memory_reserved(device) / (1024**3)
             percent = (used_gb / total_gb) * 100 if total_gb > 0 else 0
             print(f"[WARN] Using torch.cuda.memory_reserved for VRAM ({used_gb:.2f} GB used). GPU utilization not available via torch. Install pynvml for more accurate stats.")
             return used_gb, total_gb, percent, 0 # MODIFIED RETURN
         except Exception as e:
             print(f"[ERROR] Failed to get VRAM usage via torch.cuda: {e}")
             return 0, 0, 0, 0 # MODIFIED RETURN
    else:
        # Not a CUDA device
        return 0, 0, 0, 0 # MODIFIED RETURN

# --- SVG Generation ---
SVG_VIEWBOX = "0 0 100 100"
SVG_WIDTH = "65"  # Adjusted for combined HTML
SVG_HEIGHT = "65" # Adjusted for combined HTML
CIRCLE_CX = 50
CIRCLE_CY = 50
CIRCLE_R = 40
STROKE_WIDTH = 9 # Adjusted for combined HTML
BG_STROKE_COLOR = "#e5e7eb"  # Tailwind gray-200
COLOR_GREEN = "#4ade80" # Tailwind green-400
COLOR_ORANGE = "#fb923c" # Tailwind orange-400
COLOR_RED = "#f87171" # Tailwind red-400
TEXT_FILL_COLOR = "#4b5563"  # Tailwind gray-600
PERCENT_TEXT_FILL_COLOR = "#eb750e"
FONT_SIZE_PERCENT = 20 # Adjusted for combined HTML
FONT_SIZE_DETAILS = 12 # Adjusted for combined HTML
DETAILS_BOX_BG_COLOR = "#f3f4f6" # Tailwind gray-100

CIRCUMFERENCE = 2 * math.pi * CIRCLE_R

def _create_memory_svg(percent_value, resource_name_translated, translations, used_gb=None, total_gb=None):
    """
    Helper function to create an SVG string for memory, utilization, or temperature display.
    - percent_value: The percentage (0-100) or temperature value.
    - resource_name_translated: The pre-translated name of the resource.
    - translations: Dictionary for "Stats N/A".
    - used_gb, total_gb: Provided for memory types (RAM/VRAM).
    """
    translate_func = translations.get

    # N/A condition
    is_na = percent_value is None or \
              (used_gb is not None and total_gb is not None and total_gb <= 0 and not (resource_name_translated == translate_func("ram_label", "RAM") and psutil.virtual_memory().total > 0))

    if is_na:
        # Clé de traduction: "memory_stats_not_available"
        return f"""
        <div style="width: {SVG_WIDTH}px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid {BG_STROKE_COLOR}; border-radius: 8px; background-color: #f9fafb; padding: 5px 0;">
            <p style="margin: 0; font-size: {FONT_SIZE_DETAILS + 2}px; color: {TEXT_FILL_COLOR}; font-weight: bold; text-align: center;">{resource_name_translated}</p>
            <p style="margin: 0; font-size: {FONT_SIZE_DETAILS}px; color: #6b7280;">{translate_func("memory_stats_not_available", "Stats N/A")}</p>
        </div>
        """

    clamped_percent_for_circle = max(0, min(100, percent_value))
    display_value = percent_value # This will be temperature or percentage

    offset = CIRCUMFERENCE * (1 - (clamped_percent_for_circle / 100))

    # Determine foreground color based on percentage (or temperature for color, but not for circle fill)
    # For temperature, we might want different thresholds or a fixed color.
    # For simplicity, using the same logic for color.
    color_value_for_thresholds = percent_value
    if resource_name_translated == translate_func("gpu_temp_label", "GPU Temp"):
        # Example: Green < 60C, Orange < 80C, Red >= 80C
        if color_value_for_thresholds < 60: fg_color = COLOR_GREEN
        elif color_value_for_thresholds < 80: fg_color = COLOR_ORANGE
        else: fg_color = COLOR_RED
    else: # RAM, VRAM, CPU, GPU Util
        if color_value_for_thresholds < 50: fg_color = COLOR_GREEN
        elif color_value_for_thresholds < 75: fg_color = COLOR_ORANGE
        else: fg_color = COLOR_RED

    # Determine the text content
    if used_gb is not None and total_gb is not None: # Memory type (RAM/VRAM)
        details_text_content = f"{resource_name_translated}: {used_gb:.1f} GB / {total_gb:.1f} GB"
        display_text_in_circle = f"{display_value:.0f}%"
    else: # Utilization type (CPU/GPU Util)
        details_text_content = f"{resource_name_translated}: {display_value:.0f} %"
        display_text_in_circle = f"{display_value:.0f}%"


    svg_html = f"""
    <div style="text-align: center;">
        <svg viewBox="{SVG_VIEWBOX}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" style="display: block; margin: 0 auto 3px auto;">
            <circle cx="{CIRCLE_CX}" cy="{CIRCLE_CY}" r="{CIRCLE_R}" fill="none" stroke="{BG_STROKE_COLOR}" stroke-width="{STROKE_WIDTH}"/>
            <circle cx="{CIRCLE_CX}" cy="{CIRCLE_CY}" r="{CIRCLE_R}" fill="none" stroke="{fg_color}" stroke-width="{STROKE_WIDTH}"
                    stroke-dasharray="{CIRCUMFERENCE:.2f}"
                    stroke-dashoffset="{offset:.2f}"
                    stroke-linecap="round"
                    transform="rotate(-90 {CIRCLE_CX} {CIRCLE_CY})"/>
            <text x="{CIRCLE_CX}" y="{CIRCLE_CY + (FONT_SIZE_PERCENT / 3)}" text-anchor="middle" fill="{PERCENT_TEXT_FILL_COLOR}" font-weight="bold" font-size="{FONT_SIZE_PERCENT}px" style="paint-order: stroke; stroke: #000000; stroke-width: 0.5px; stroke-linecap: butt; stroke-linejoin: miter;">
                {display_text_in_circle}
            </text>
        </svg>
        <div style="font-size: {FONT_SIZE_DETAILS}px;
                    color: {TEXT_FILL_COLOR};
                    background-color: {DETAILS_BOX_BG_COLOR};
                    padding: 2px 5px;
                    border-radius: 4px;">
            {details_text_content}
        </div>
    </div>
    """
    return svg_html

def _create_combined_memory_html(translations, device_for_vram):
    """
    Creates a single HTML string containing all memory/CPU/GPU stats.
    """
    translate_func = translations.get

    # Get all stats
    ram_used, ram_total, ram_percent, cpu_util = get_ram_usage(translations)
    vram_used, vram_total, vram_percent, gpu_util = get_vram_usage(device_for_vram, translations) # MODIFIED: gpu_temp removed

    # Create individual SVG items
    ram_svg_item = _create_memory_svg(ram_percent, translate_func("ram_label", "RAM"), translations, ram_used, ram_total)
    cpu_svg_item = _create_memory_svg(cpu_util, translate_func("cpu_usage_label", "CPU Usage"), translations)
    vram_svg_item = _create_memory_svg(vram_percent, translate_func("vram_label", "VRAM"), translations, vram_used, vram_total)
    gpu_util_svg_item = _create_memory_svg(gpu_util, translate_func("gpu_usage_label", "GPU Usage"), translations)
    # gpu_temp_svg_item = _create_memory_svg(gpu_temp, translate_func("gpu_temp_label", "GPU Temp"), translations) # REMOVED


    # Combine into a single HTML structure using flexbox
    # Adjusted for 4 items
    combined_html = f"""
    <div style="display: flex; flex-direction: row; justify-content: space-around; align-items: flex-start; width: 100%; padding: 5px 0;">
        <div style="flex: 1; min-width: {SVG_WIDTH}px; margin: 0 2px;">{ram_svg_item}</div>
        <div style="flex: 1; min-width: {SVG_WIDTH}px; margin: 0 2px;">{cpu_svg_item}</div>
        <div style="flex: 1; min-width: {SVG_WIDTH}px; margin: 0 2px;">{vram_svg_item}</div>
        <div style="flex: 1; min-width: {SVG_WIDTH}px; margin: 0 2px;">{gpu_util_svg_item}</div>
    </div>
    """
    return combined_html

def create_memory_accordion_ui(translations, model_manager_instance):
    """
    Creates the Gradio UI components for the Memory Management accordion.
    Returns a dictionary of Gradio components.
    """
    # Translation keys needed:
    # memory_management_accordion_title
    # ram_label
    # cpu_usage_label
    # vram_label
    # gpu_usage_label 
    # gpu_temp_label
    # unload_all_models_button_label
    # statut
    # memory_stats_not_available (used in _create_memory_svg)
    # unload_process_completed_check_stats (used in unload_all_models_action)
    # performing_explicit_memory_cleanup (used in unload_all_models_action)
    # cuda_cache_emptied_after_unload (used in unload_all_models_action)
    # cuda_cache_emptied_after_unload_fallback (used in unload_all_models_action)
    # cleanup_warning_suffix (used in unload_all_models_action)
    # cleanup_warning_occurred (used in unload_all_models_action)

    translate_func = translations.get

    with gr.Accordion(translate_func("memory_management_accordion_title", "Memory Management"), open=False) as memory_accordion:
        # Single HTML component to hold all stats
        all_stats_html = gr.HTML(
            value=_create_combined_memory_html(translations, model_manager_instance.device)
        )

        with gr.Row(equal_height=False): # Mettre tous les contrôles sur une seule ligne
            enable_live_stats_checkbox = gr.Checkbox(
                label=translate_func("enable_live_memory_stats_label", "Activer MàJ auto Stats"),
                value=False, # Par défaut désactivé
                elem_id="enable_live_memory_stats",
                scale=2 # Donner un peu plus de largeur à la checkbox et son label
            )
            memory_interval_slider = gr.Slider(
                minimum=1, maximum=60, value=5, step=1,
                label=translate_func("memory_stats_update_interval_label", "Intervalle MàJ (sec)"),
                elem_id="memory_stats_update_interval",
                interactive=True,
                scale=3 # Donner plus de largeur au slider
            )
            unload_button = gr.Button(translate_func("unload_all_models_button_label", "Unload All Models"), variant="stop", scale=1)
            unload_status = gr.Textbox(label=translate_func("statut", "Status"), interactive=False, value="", scale=2)

    def unload_all_models_action(translations_dict):
        translate_action = translations_dict.get
        status_message = ""

        try:
            # model_manager_instance.unload_model should handle its own UI feedback (gr.Info/Error)
            # and return success status and a translated message.
            success, message = model_manager_instance.unload_model(gradio_mode=True)
            status_message = message # Use the message from model_manager

        except Exception as e:
            status_message = f"{translate_action('unload_error_message', 'Error unloading models')}: {e}"
            print(f"{txt_color('[ERROR]', 'erreur')} Unexpected error during ModelManager unload action: {e}")
            traceback.print_exc()
            gr.Error(status_message, 4.0) # Ensure UI feedback for unexpected errors

        # Explicit memory cleanup
        print(txt_color("[INFO]", "info"), translate_action("performing_explicit_memory_cleanup", "Performing explicit memory cleanup..."))
        try:
            gc.collect()
            if hasattr(model_manager_instance, 'device') and model_manager_instance.device.type == 'cuda':
                torch.cuda.empty_cache()
                print(txt_color("[INFO]", "info"), translate_action("cuda_cache_emptied_after_unload", "CUDA cache emptied after unload attempt."))
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(txt_color("[INFO]", "info"), translate_action("cuda_cache_emptied_after_unload_fallback", "CUDA cache emptied (fallback check)."))
            gc.collect()
        except Exception as e_cleanup:
            print(f"{txt_color('[WARN]', 'warning')} Error during explicit memory cleanup: {e_cleanup}")
            # Append warning to status message if it's not already an error
            # Check if 'success' variable is defined and True from the try block
            if 'success' in locals() and success:
                 status_message += f" ({translate_action('cleanup_warning_suffix', 'Cleanup warning logged')})"
            elif not status_message: # If status_message is empty, set it to the cleanup warning
                 status_message = translate_action('cleanup_warning_occurred', 'Warning during memory cleanup.')


        # Update memory stats HTML
        # Utiliser update_memory_stats pour obtenir l'objet gr.update
        updated_html_component = update_memory_stats(translations_dict, model_manager_instance.device)

        if not status_message: # Fallback status message
            status_message = translate_action('unload_process_completed_check_stats', "Unload process completed. Check memory stats.")
        return status_message, updated_html_component

    unload_button.click(
        fn=unload_all_models_action,
        inputs=[gr.State(translations)],
        outputs=[unload_status, all_stats_html]
    )

    return {
        "accordion": memory_accordion,
        "all_stats_html": all_stats_html,
        "unload_button": unload_button,
        "unload_status": unload_status,
        "enable_live_stats_checkbox": enable_live_stats_checkbox,
        "memory_interval_slider": memory_interval_slider,
    }

def update_memory_stats(translations_dict, device):
    """
    Updates all memory stats and returns a Gradio update for the combined HTML component.
    """
    combined_html_updated = _create_combined_memory_html(translations_dict, device)
    return gr.update(value=combined_html_updated)
