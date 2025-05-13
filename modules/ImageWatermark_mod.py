import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import os
import json
import traceback # Import traceback for detailed error logging
# Importer vos utilitaires (translate, GestionModule, etc.)
from Utils.utils import translate, GestionModule, enregistrer_image # et autres nécessaires

MODULE_NAME = "ImageWatermark"
# ... chargement du JSON de traduction spécifique au module ...

def initialize(global_translations, model_manager_instance, gestionnaire_instance: GestionModule, global_config=None):
    return ImageWatermarkModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class ImageWatermarkModule:
    def __init__(self, global_translations, model_manager_instance, gestionnaire_instance: GestionModule, global_config=None):
        self.gestionnaire = gestionnaire_instance
        self.global_config = global_config
        self.global_translations = global_translations
        self.module_translations = {} # Sera rempli par create_tab
        # Pas besoin de model_manager ici a priori

    def _apply_watermark_logic(self, base_image_pil: Image.Image, watermark_type: str,
                               text_content: str, text_font_path: str, text_size: int, text_color_hex: str,
                               watermark_image_pil: Image.Image, image_scale_percent: int,
                               opacity_percent: int, position_str: str, margin: int, rotation_degrees: int) -> Image.Image | None:
        """
        Applies the watermark to the base image.
        Returns the watermarked PIL Image, or None if an error occurs.
        """
        try:
            if base_image_pil is None:
                # This case should be caught by the wrapper, but as a safeguard:
                print("Error in _apply_watermark_logic: base_image_pil is None.")
                return None

            base_img_rgba = base_image_pil.copy().convert("RGBA")
            watermark_layer = None
            watermark_width, watermark_height = 0, 0

            # --- 1. Prepare Watermark Layer ---
            # Ensure opacity_percent is within a valid range (0-100)
            opacity_percent = max(0, min(100, opacity_percent))
            opacity_value = int(255 * (opacity_percent / 100.0))

            if watermark_type == translate("watermark_type_text", self.module_translations):
                if not text_content:
                    gr.Warning(translate("error_no_watermark_text", self.module_translations))
                    return None

                try:
                    font = ImageFont.truetype(text_font_path, text_size) if text_font_path else ImageFont.load_default()
                except IOError:
                    print(f"Warning: Font {text_font_path} not found. Using default font.")
                    font = ImageFont.load_default()
                
                # Determine text size
                # Create a dummy image and draw object to get text bounding box
                # This is more reliable than textsize for multi-line text or complex fonts
                dummy_img = Image.new("RGBA", (1,1)) # Small dummy image
                draw_temp = ImageDraw.Draw(dummy_img)
                
                try:
                    # For Pillow >= 9.2.0, textbbox is preferred.
                    # It returns (left, top, right, bottom)
                    bbox = draw_temp.textbbox((0, 0), text_content, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError: 
                    # Fallback for older Pillow versions that don't have textbbox
                    # textsize might not be as accurate for complex cases
                    text_width, text_height = draw_temp.textsize(text_content, font=font)
                
                # Ensure dimensions are at least 1 pixel
                watermark_width = max(1, text_width)
                watermark_height = max(1, text_height)

                watermark_layer = Image.new("RGBA", (watermark_width, watermark_height), (255, 255, 255, 0)) # Transparent background
                draw = ImageDraw.Draw(watermark_layer)

                # Convert hex color to RGBA tuple
                text_rgba_color = (255, 255, 255, opacity_value) # Default to white with opacity
                if isinstance(text_color_hex, str):
                    if text_color_hex.startswith("rgba("):
                        try:
                            parts = text_color_hex.replace("rgba(", "").replace(")", "").split(',')
                            r, g, b = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
                            # Alpha from rgba string is 0-1, opacity_value is 0-255 from slider
                            # We'll prioritize the slider's opacity for consistency
                            text_rgba_color = (r, g, b, opacity_value)
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse rgba color '{text_color_hex}': {e}. Defaulting to white.")
                    elif text_color_hex.startswith("#"):
                        hex_color_val = text_color_hex.lstrip('#')
                        if len(hex_color_val) == 6: # Standard RRGGBB
                            rgb_color = tuple(int(hex_color_val[i:i+2], 16) for i in (0, 2, 4))
                            text_rgba_color = (*rgb_color, opacity_value)
                        elif len(hex_color_val) == 8: # RRGGBBAA
                            rgb_color = tuple(int(hex_color_val[i:i+2], 16) for i in (0, 2, 4))
                            alpha_from_hex = int(hex_color_val[6:8], 16)
                            # Modulate hex alpha with slider opacity
                            text_rgba_color = (*rgb_color, int(alpha_from_hex * (opacity_percent / 100.0)))
                        else:
                            print(f"Warning: Invalid hex color format '{text_color_hex}'. Defaulting to white.")
                else:
                    print(f"Warning: Invalid color type '{type(text_color_hex)}'. Defaulting to white.")

                # Draw text. For Pillow >= 9.2.0, text anchor can be used.
                # Default anchor "la" (left, ascender) is usually fine for (0,0) origin.
                draw.text((0,0), text_content, font=font, fill=text_rgba_color)

            elif watermark_type == translate("watermark_type_image", self.module_translations):
                if watermark_image_pil is None:
                    gr.Warning(translate("error_no_watermark_image", self.module_translations))
                    return None
                
                watermark_img_rgba = watermark_image_pil.copy().convert("RGBA")
                
                # Scale watermark image
                base_w, base_h = base_img_rgba.size
                scale_factor = image_scale_percent / 100.0
                
                # Scale based on base image width, maintain aspect ratio
                # Ensure watermark is not scaled to 0 width/height
                new_wm_width = max(1, int(watermark_img_rgba.width * scale_factor))
                wm_aspect_ratio = watermark_img_rgba.height / watermark_img_rgba.width if watermark_img_rgba.width > 0 else 1
                new_wm_height = max(1, int(new_wm_width * wm_aspect_ratio))

                watermark_width, watermark_height = new_wm_width, new_wm_height
                watermark_layer = watermark_img_rgba.resize((watermark_width, watermark_height), Image.Resampling.LANCZOS)

                # Apply opacity to image watermark
                alpha = watermark_layer.split()[3]
                # Ensure opacity_value is used correctly here
                final_alpha_for_image = alpha.point(lambda p: int(p * (opacity_percent / 100.0)))
                watermark_layer.putalpha(final_alpha_for_image)
            else:
                print(f"Error: Unknown watermark type '{watermark_type}'")
                return None

            if watermark_layer is None:
                print("Error: Watermark layer was not created.")
                return None 

            # --- 2. Apply Rotation (if any) ---
            if rotation_degrees != 0:
                # Rotate the watermark layer, expand to fit, use bicubic for better quality
                watermark_layer = watermark_layer.rotate(rotation_degrees, expand=True, resample=Image.Resampling.BICUBIC)
                # Update dimensions after rotation as they might have changed
                watermark_width, watermark_height = watermark_layer.size

            # --- 3. Calculate Position and Paste ---
            base_w, base_h = base_img_rgba.size

            if position_str == translate("position_tile", self.module_translations):
                if watermark_width == 0 or watermark_height == 0: # Avoid division by zero
                    print("Error: Watermark for tiling has zero dimension.")
                    return base_img_rgba.convert("RGB") # Return original if tile is invalid

                # Ensure margin for tiling is not excessively large
                tile_margin_x = margin if watermark_width + margin > 0 else 0
                tile_margin_y = margin if watermark_height + margin > 0 else 0
                
                step_x = watermark_width + tile_margin_x
                step_y = watermark_height + tile_margin_y

                if step_x <= 0 or step_y <=0: # Prevent infinite loop
                    print("Error: Tiling step is zero or negative due to large negative margin.")
                    return base_img_rgba.convert("RGB")


                for y_offset in range(0, base_h, step_y):
                    for x_offset in range(0, base_w, step_x):
                        base_img_rgba.alpha_composite(watermark_layer, (x_offset, y_offset))
            else:
                x, y = 0, 0 # Default to top-left if no match
                if position_str == translate("position_top_left", self.module_translations):
                    x, y = margin, margin
                elif position_str == translate("position_top_right", self.module_translations):
                    x, y = base_w - watermark_width - margin, margin
                elif position_str == translate("position_bottom_left", self.module_translations):
                    x, y = margin, base_h - watermark_height - margin
                elif position_str == translate("position_bottom_right", self.module_translations):
                    x, y = base_w - watermark_width - margin, base_h - watermark_height - margin
                elif position_str == translate("position_center", self.module_translations):
                    x, y = (base_w - watermark_width) // 2, (base_h - watermark_height) // 2
                
                # Ensure coordinates are within bounds, especially after rotation
                x = max(0, min(x, base_w - watermark_width))
                y = max(0, min(y, base_h - watermark_height))

                base_img_rgba.alpha_composite(watermark_layer, (int(x), int(y)))

            return base_img_rgba.convert("RGB") # Return RGB for general compatibility

        except Exception as e:
            print(f"Error in _apply_watermark_logic: {e}")
            traceback.print_exc() # Print full traceback for debugging
            gr.Error(f"Error applying watermark: {str(e)[:100]}...") # Show a concise error in UI
            return None

    def create_tab(self, module_translations):
        self.module_translations = module_translations # Stocker les traductions pour ce module

        with gr.Tab(translate("watermark_tab", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('watermark_title', self.module_translations)}")

            with gr.Row():
                with gr.Column(scale=1): # Colonne des inputs
                    input_mode_radio = gr.Radio(
                        choices=[
                            translate("input_mode_single", self.module_translations),
                            translate("input_mode_batch", self.module_translations)
                        ],
                        label=translate("input_mode_label", self.module_translations),
                        value=translate("input_mode_single", self.module_translations)
                    )

                    with gr.Group(visible=True) as single_image_group:
                        input_image = gr.Image(label=translate("image_input_label", self.module_translations), type="pil")

                    with gr.Group(visible=False) as batch_folder_group:
                        input_folder_textbox = gr.Textbox(label=translate("input_folder_label", self.module_translations), placeholder="/path/to/input_folder")
                        output_folder_textbox = gr.Textbox(label=translate("output_folder_label", self.module_translations), placeholder="/path/to/output_folder")

                    gr.Markdown("---") # Separator

                    watermark_type_radio = gr.Radio(
                        choices=[
                            translate("watermark_type_text", self.module_translations),
                            translate("watermark_type_image", self.module_translations)
                        ],
                        label=translate("watermark_type_label", self.module_translations),
                        value=translate("watermark_type_text", self.module_translations)
                    )

                    # --- Options Filigrane Texte ---
                    with gr.Group(visible=True) as text_watermark_group:
                        text_content = gr.Textbox(label=translate("text_content_label", self.module_translations), value="Cyberbill SDXL")
                        # TODO: Add font selection dropdown if desired
                        # text_font_path_input = gr.Textbox(label="Chemin Police (laisser vide pour défaut)", value="arial.ttf") 
                        text_size = gr.Slider(minimum=8, maximum=300, value=36, step=1, label=translate("text_size_label", self.module_translations))
                        text_color = gr.ColorPicker(label=translate("text_color_label", self.module_translations), value="#FFFFFF")

                    # --- Options Filigrane Image ---
                    with gr.Group(visible=False) as image_watermark_group:
                        watermark_image_upload = gr.Image(label=translate("image_upload_watermark_label", self.module_translations), type="pil")
                        image_scale = gr.Slider(minimum=1, maximum=100, value=20, step=1, label=translate("image_scale_label", self.module_translations))

                    # --- Options Communes ---
                    opacity = gr.Slider(minimum=0, maximum=100, value=50, step=1, label=translate("opacity_label", self.module_translations))
                    
                    positions = [
                        translate("position_top_left", self.module_translations),
                        translate("position_top_right", self.module_translations),
                        translate("position_bottom_left", self.module_translations),
                        translate("position_bottom_right", self.module_translations),
                        translate("position_center", self.module_translations),
                        translate("position_tile", self.module_translations)
                    ]
                    position_dropdown = gr.Dropdown(choices=positions, value=positions[3], label=translate("position_label", self.module_translations)) # Renamed to avoid conflict
                    margin_number = gr.Number(value=10, label=translate("margin_label", self.module_translations), minimum=0, precision=0) # Renamed
                    rotation_slider = gr.Slider(minimum=-180, maximum=180, value=0, step=1, label=translate("rotation_label", self.module_translations)) # Renamed

                    apply_btn = gr.Button(translate("apply_button", self.module_translations))

                with gr.Column(scale=1): # Colonne de sortie
                    # Le composant output_image affichera l'image unique ou un message pour le lot
                    output_display = gr.Image(label=translate("output_image_label", self.module_translations), type="pil", show_download_button=True)
                    batch_status_textbox = gr.Textbox(label="Batch Log", lines=5, interactive=False, visible=False)

            # Logique de visibilité des groupes
            def toggle_input_mode_visibility(mode_choice):
                is_single = (mode_choice == translate("input_mode_single", self.module_translations))
                return gr.update(visible=is_single), gr.update(visible=not is_single), gr.update(visible=not is_single)

            input_mode_radio.change(
                fn=toggle_input_mode_visibility,
                inputs=input_mode_radio,
                outputs=[single_image_group, batch_folder_group, batch_status_textbox]
            )

            def toggle_watermark_options(choice):
                is_text = (choice == translate("watermark_type_text", self.module_translations))
                return gr.update(visible=is_text), gr.update(visible=not is_text)
            
            watermark_type_radio.change(
                fn=toggle_watermark_options,
                inputs=watermark_type_radio,
                outputs=[text_watermark_group, image_watermark_group]
            )
            
            # Logique du bouton "Appliquer"
            def process_watermark_wrapper(
                input_mode, img_in, input_folder, output_folder, # Nouveaux inputs pour le mode
                wm_type_str, 
                txt_content, txt_size, txt_color_hex,
                wm_img_in, img_scl_percent,
                opcty_percent, pos_str, mrg_val, rot_degrees,
                progress=gr.Progress(track_tqdm=True)
            ):
                progress(0, desc=translate("processing_watermark", self.module_translations))
                batch_log_messages = []

                # Determine font path (this is a simplification)
                # For a real app, you'd want a more robust font selection/management system
                font_path = "arial.ttf" # Default, might not exist on all systems
                # Attempt to use a system font if arial.ttf is not found (very basic check)
                if not os.path.exists(font_path):
                    # This is OS-dependent and not guaranteed.
                    # On Windows, 'arial.ttf' is common. On Linux, 'DejaVuSans.ttf' might be.
                    # For true cross-platform, bundle fonts or use a font discovery library.
                    if os.name == 'nt': # Windows
                        font_path_system = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arial.ttf')
                        if os.path.exists(font_path_system):
                            font_path = font_path_system
                        else:
                            font_path = None # Fallback to Pillow's default
                    else: # Linux/macOS (very rough guess)
                        common_linux_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                        if os.path.exists(common_linux_font):
                            font_path = common_linux_font
                        else:
                            font_path = None # Fallback to Pillow's default
                
                if font_path is None:
                     print("Warning: Defaulting to Pillow's internal basic font as no suitable system font was found.")

                if input_mode == translate("input_mode_single", self.module_translations):
                    if img_in is None:
                        gr.Warning(translate("error_no_image", self.module_translations))
                        return None, "" # output_display, batch_status_textbox
                    
                    result_img = self._apply_watermark_logic(
                        img_in, wm_type_str,
                        txt_content, font_path, int(txt_size), txt_color_hex,
                        wm_img_in, int(img_scl_percent),
                        int(opcty_percent), pos_str, int(mrg_val), int(rot_degrees)
                    )
                    if result_img is not None and isinstance(result_img, Image.Image):
                        gr.Info(translate("success_watermark", self.module_translations))
                    progress(1.0)
                    return result_img, "\n".join(batch_log_messages)

                elif input_mode == translate("input_mode_batch", self.module_translations):
                    if not input_folder:
                        gr.Warning(translate("error_no_input_folder", self.module_translations))
                        return None, translate("error_no_input_folder", self.module_translations)
                    if not output_folder:
                        gr.Warning(translate("error_no_output_folder", self.module_translations))
                        return None, translate("error_no_output_folder", self.module_translations)
                    if not os.path.isdir(input_folder):
                        gr.Warning(translate("error_input_folder_not_found", self.module_translations))
                        return None, translate("error_input_folder_not_found", self.module_translations)
                    
                    try:
                        os.makedirs(output_folder, exist_ok=True)
                    except OSError as e:
                        err_msg = translate("error_output_folder_not_creatable", self.module_translations) + f": {e}"
                        gr.Error(err_msg)
                        return None, err_msg

                    batch_log_messages.append(translate("batch_processing_start", self.module_translations))
                    
                    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                    total_images = len(image_files)
                    processed_count = 0

                    if total_images == 0:
                        msg = translate("batch_no_images_found", self.module_translations)
                        gr.Info(msg)
                        batch_log_messages.append(msg)
                        return None, "\n".join(batch_log_messages)

                    last_processed_image_for_display = None

                    for i, filename in enumerate(image_files):
                        progress((i + 1) / total_images, desc=translate("batch_processing_image_x_of_y", self.module_translations).format(current=i+1, total=total_images, filename=filename))
                        batch_log_messages.append(f"Processing: {filename}")
                        try:
                            img_path = os.path.join(input_folder, filename)
                            current_img_pil = Image.open(img_path)

                            watermarked_img = self._apply_watermark_logic(
                                current_img_pil, wm_type_str,
                                txt_content, font_path, int(txt_size), txt_color_hex,
                                wm_img_in, int(img_scl_percent),
                                int(opcty_percent), pos_str, int(mrg_val), int(rot_degrees)
                            )

                            if watermarked_img:
                                output_path = os.path.join(output_folder, filename)
                                # Utiliser la fonction globale enregistrer_image
                                enregistrer_image(watermarked_img, output_path, self.global_translations, os.path.splitext(filename)[1].lstrip('.').upper() or "PNG")
                                batch_log_messages.append(f"  Saved: {output_path}")
                                processed_count += 1
                                last_processed_image_for_display = watermarked_img # Pour l'affichage final
                            else:
                                batch_log_messages.append(f"  Skipped (error during watermarking): {filename}")

                        except Exception as e:
                            batch_log_messages.append(f"  Error processing {filename}: {e}")
                            print(f"Error processing {filename} in batch: {e}")
                            traceback.print_exc()
                    
                    final_batch_msg = translate("batch_processing_complete", self.module_translations).format(count=processed_count)
                    gr.Info(final_batch_msg)
                    batch_log_messages.append(final_batch_msg)
                    return last_processed_image_for_display, "\n".join(batch_log_messages)
                return None, "\n".join(batch_log_messages) # Fallback

            apply_btn.click(
                fn=process_watermark_wrapper,
                inputs=[
                    input_mode_radio, input_image, input_folder_textbox, output_folder_textbox,
                    watermark_type_radio,
                    text_content, text_size, text_color,
                    watermark_image_upload, image_scale,
                    opacity, position_dropdown, margin_number, rotation_slider # Use renamed variables
                ],
                outputs=[output_display, batch_status_textbox]
            )
        return tab
