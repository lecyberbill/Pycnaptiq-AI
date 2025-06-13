# d:\image_to_text\cyberbill_SDXL\cyberbill_image_generator\modules\civitai_downloader_mod.py
import os
import json
import requests
import gradio as gr
from pathlib import Path
import traceback
import math
from tqdm import tqdm
import functools # Pour partial

# Importer les utilitaires nécessaires
from Utils.utils import txt_color, translate, GestionModule, telechargement_modele
from Utils.model_manager import ModelManager

# --- Chargement des métadonnées et initialisation ---
MODULE_NAME = "civitai_downloader"
module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

try:
    with open(module_json_path, 'r', encoding="utf-8") as f:
        module_data_json = json.load(f)
except FileNotFoundError:
    print(txt_color("[ERREUR]", "erreur"), f"Fichier JSON non trouvé pour {MODULE_NAME}: {module_json_path}")
    module_data_json = {"name": MODULE_NAME, "language": {"fr": {}}}
except json.JSONDecodeError:
    print(txt_color("[ERREUR]", "erreur"), f"Erreur de décodage JSON pour {MODULE_NAME}: {module_json_path}")
    module_data_json = {"name": MODULE_NAME, "language": {"fr": {}}}

def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
    print(txt_color("[OK] ", "ok"), f"Initialisation du module: {module_data_json.get('name', MODULE_NAME)}")
    return CivitaiDownloaderModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class CivitaiDownloaderModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.gestionnaire = gestionnaire_instance
        self.global_config = global_config
        self.module_translations = module_data_json.get("language", {}).get("fr", {})

        self.civitai_api_url = "https://civitai.com/api/v1/models"
        self.current_search_results = []
        self.current_model_details = None # Stocke les détails du modèle sélectionné (via JS ou dropdown)

        self.models_dir = self.global_config.get("MODELS_DIR", "models/Stable-diffusion")
        self.loras_dir = self.global_config.get("LORAS_DIR", "models/Lora")
        self.inpaint_models_dir = self.global_config.get("INPAINT_MODELS_DIR", "models/Inpainting")
        self.vae_dir = self.global_config.get("VAE_DIR", "models/VAE")
        self.civitai_api_key_from_config = self.global_config.get("CIVITAI_API_KEY", "") # Lire depuis config
        self.embeddings_dir = self.global_config.get("EMBEDDINGS_DIR", os.path.join(self.models_dir, "embeddings"))
        self.hypernetworks_dir = self.global_config.get("HYPERNETWORKS_DIR", os.path.join(self.models_dir, "hypernetworks"))
        self.controlnet_dir = self.global_config.get("CONTROLNET_DIR", os.path.join(self.models_dir, "controlnet"))


    def _get_target_directory(self, model_type_str):
        model_type_map = {
            "Checkpoint": self.models_dir,
            "LORA": self.loras_dir,
            "DoRA": self.loras_dir,
            "TextualInversion": self.embeddings_dir,
            "Hypernetwork": self.hypernetworks_dir,
            "Controlnet": self.controlnet_dir,
            "VAE": self.vae_dir,
        }
        target_dir = model_type_map.get(model_type_str)
        if not target_dir:
            gr.Warning(translate("unsupported_model_type_for_download", self.module_translations).format(model_type=model_type_str))
            return self.models_dir
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            err_msg = translate("error_creating_directory", self.module_translations).format(directory=target_dir, error=e)
            print(txt_color("[ERREUR]", "erreur"), err_msg)
            gr.Error(err_msg)
            return None
        return target_dir

    def _fetch_from_civitai(self, endpoint_suffix="", params=None, api_key=None):
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = requests.get(f"{self.civitai_api_url}{endpoint_suffix}", params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(txt_color("[ERREUR]", "erreur"), f"Erreur API Civitai: {e}")
            gr.Error(translate("error_fetching_models", self.module_translations).format(error=str(e)))
            return None
        except json.JSONDecodeError as e:
            print(txt_color("[ERREUR]", "erreur"), f"Erreur décodage JSON Civitai: {e}")
            err_msg = translate("error_civitai_json_decode", self.module_translations).format(error=str(e))
            gr.Error(err_msg)
            return None

    def search_models(self, query, model_type, sort_by, period, nsfw_filter, limit, page, civitai_api_key, filter_sdxl):
        params = {
            "limit": int(limit),
            "page": int(page),
            "query": query if query else None,
            "types": [model_type] if model_type != translate("all_types", self.module_translations) else None,
            "nsfw": None
        }
        
        def get_api_key_from_ui_value(ui_value, translation_key_to_api_map):
            for translation_key, api_value in translation_key_to_api_map.items():
                if translate(translation_key, self.module_translations) == ui_value:
                    return api_value
            print(f"Warning: Could not map UI value '{ui_value}' to API key. Using UI value directly or None.")
            if ui_value in translation_key_to_api_map.values():
                return ui_value
            return None 

        sort_translation_map = {
            "sort_highest_rated": "Highest Rated",
            "sort_most_downloaded": "Most Downloaded",
            "sort_newest": "Newest"
        }
        params["sort"] = get_api_key_from_ui_value(sort_by, sort_translation_map)

        period_translation_map = {
            "period_alltime": "AllTime", "period_year": "Year", "period_month": "Month",
            "period_week": "Week", "period_day": "Day"
        }
        params["period"] = get_api_key_from_ui_value(period, period_translation_map)

        if nsfw_filter == self.module_translations.get("nsfw_none"):
            params["nsfw"] = "false"
        elif nsfw_filter == self.module_translations.get("nsfw_soft"):
            params["nsfwLevel"] = 2
        elif nsfw_filter == self.module_translations.get("nsfw_mature"):
            params["nsfwLevel"] = 4
        elif nsfw_filter == self.module_translations.get("nsfw_x"):
            params["nsfwLevel"] = 8
        elif nsfw_filter == self.module_translations.get("nsfw_all"):
            params["nsfwLevel"] = 1 | 2 | 4 | 8

        if filter_sdxl:
            params["tag"] = "sdxl"
        params = {k: v for k, v in params.items() if v is not None}
        data = self._fetch_from_civitai(params=params, api_key=civitai_api_key)

        if data and "items" in data:
            self.current_search_results = data["items"]
            metadata = data.get("metadata", {})
            current_page_api = metadata.get("currentPage", 1)
            total_pages_api = metadata.get("totalPages", 1)
            results_html = self.format_search_results_html(self.current_search_results)
            page_info = translate("page_display", self.module_translations).format(current=current_page_api, total=total_pages_api)
            prev_btn_update = gr.update(interactive=(current_page_api > 1))
            next_btn_update = gr.update(interactive=(current_page_api < total_pages_api))
            
            # Réinitialiser les champs de détails
            description_clear_update = gr.update(value="")
            version_dd_clear_update = gr.update(
                choices=[translate("select_version_prompt", self.module_translations)], 
                value=translate("select_version_prompt", self.module_translations),
                interactive=False  # Inactif après une nouvelle recherche
            )
            file_dd_clear_update = gr.update(
                choices=[translate("select_file_prompt", self.module_translations)], 
                value=translate("select_file_prompt", self.module_translations),
                interactive=False  # Inactif après une nouvelle recherche
            )
            status_clear_update = gr.update(value="")

            return results_html, page_info, current_page_api, prev_btn_update, next_btn_update, description_clear_update, version_dd_clear_update, file_dd_clear_update, status_clear_update
        else:
            self.current_search_results = []
            return translate("no_models_found", self.module_translations), \
                   translate("page_display", self.module_translations).format(current=1, total=1), 1, \
                   gr.update(interactive=False), gr.update(interactive=False), \
                   gr.update(value=""), \
                   gr.update(choices=[translate("select_version_prompt", self.module_translations)], value=translate("select_version_prompt", self.module_translations), interactive=False), \
                   gr.update(choices=[translate("select_file_prompt", self.module_translations)], value=translate("select_file_prompt", self.module_translations), interactive=False), \
                   gr.update(value="")

    def format_search_results_html(self, items):
        if not items:
            return f"<p>{translate('no_models_found', self.module_translations)}</p>"
        
        styles_html = """
        <style>
            .civitai-downloader-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 15px; /* Spacing between cards */
            }
            .model-card-container {
                border: 1px solid #ffffff;
                border-radius: 8px;
                padding: 10px;
                width: 200px; /* Fixed width for each card */
                display: flex;
                flex-direction: column;
                justify-content: space-between; /* Pushes button to the bottom */
                background-color: #000000;  
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Subtle shadow */
                box-sizing: border-box;
            }
            .model-card-image {
                width: 100%;
                height: 150px; /* Fixed height for image */
                object-fit: cover; /* Ensures image covers the area, might crop */
                border-radius: 4px; /* Rounded corners for the image */
                margin-bottom: 8px; /* Space below the image */
            }
            .model-card-title {
                margin: 5px 0;
                font-size: 0.95em; /* Slightly larger for better readability */
                font-weight: bold;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis; /* Shows '...' if text is too long */
                color: #333; /* Darker text color */
            }
            .model-card-text {
                font-size: 0.8em;
                margin: 3px 0; /* Consistent margin for text paragraphs */
                color: #555; /* Medium-dark text color */
                line-height: 1.3; /* Improved line spacing */
            }
            .civitai-downloader-details-btn {
                background: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 8px; /* Correspond au style .meta-button et à la carte */
                padding: 8px; /* Correspond au style .meta-button */
                width: 100%;
                cursor: pointer;
                font-weight: bold; /* Correspond au style .meta-button */
                margin-top: 4px; /* Correspond au style .meta-button */
                text-align: center;
                font-size: 0.85em;
                transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out; /* Transition pour fond et couleur */
            }
            .civitai-downloader-details-btn:hover {
                background: #000; /* Fond noir au survol, comme .meta-button:hover */
                color: white; /* Texte blanc au survol pour contraster avec le fond noir */
            }
        </style>
        """        
        
        
        html_content = styles_html + "<div class='civitai-downloader-grid'>"



        for item in items:
            model_id = item.get("id")
            name = item.get("name", translate("not_available_abbreviation", self.module_translations))
            creator_name = item.get("creator", {}).get("username", translate("not_available_abbreviation", self.module_translations))
            model_type_raw = item.get("type", translate("not_available_abbreviation", self.module_translations))
            normalized_type_key_part = model_type_raw.lower().replace(' ', '_').replace('-', '_')
            model_type_key = f"model_type_{normalized_type_key_part}"
            model_type_display_candidate = translate(model_type_key, self.module_translations)
            model_type_display = model_type_raw if model_type_display_candidate == f"[{model_type_key}]" else model_type_display_candidate
            preview_image_url = ""
            if item.get("modelVersions") and item["modelVersions"][0].get("images"):
                images_data = item["modelVersions"][0]["images"]
                sorted_images = sorted(
                    [img for img in images_data if img.get("url")], 
                    key=lambda img: img.get("width") if img.get("width") is not None else float('inf')
                )
                if sorted_images:
                    preview_image_url = sorted_images[0].get("url", "") # Default to smallest
                    for img_data in sorted_images: # Try to find one <= 512
                        if img_data.get("width") and img_data["width"] <= 512:
                            preview_image_url = img_data["url"]
                            break
            stats = item.get("stats", {})
            downloads = stats.get("downloadCount", 0)
            rating = stats.get("rating", 0)
            rating_count = stats.get("ratingCount", 0)
            html_content += f"""
            <div class='model-card-container'>
                <img src='{preview_image_url}' alt='{name}' class='model-card-image' onerror="this.onerror=null; this.src='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'; this.style.objectFit='scale-down';">
                <h4 class='model-card-title' title='{name}'>{name}</h4>
                <p class='model-card-text'>{translate('model_type_label', self.module_translations)}: {model_type_display}</p>
                <p class='model-card-text'>{translate('model_card_creator_prefix', self.module_translations)} {creator_name}</p>
                <p class='model-card-text'>
                    {translate('model_card_stats_downloads_prefix', self.module_translations)} {downloads} 
                    |
                    {translate('model_card_stats_rating_prefix', self.module_translations)} {rating:.1f} ({rating_count})
                </p>
                <button class="civitai-downloader-details-btn" data-model-name="{name.replace('"', '&quot;')}">
                    {translate('view_details_button', self.module_translations)}
                </button>
            </div>
            """

        html_content += "</div>"
        return html_content

    def update_version_selection(self, selected_version_name):
        if selected_version_name == translate("select_version_prompt", self.module_translations) or not self.current_model_details:
            return gr.update(
                choices=[translate("select_file_prompt", self.module_translations)], 
                value=translate("select_file_prompt", self.module_translations),
                interactive=False # Le dropdown des fichiers devient inactif
            )
        versions = self.current_model_details.get("modelVersions", [])
        selected_version = next((v for v in versions if v.get("name", f"Version ID: {v.get('id')}") == selected_version_name), None)
        if selected_version:
            files = selected_version.get("files", [])
            file_choices = [translate("select_file_prompt", self.module_translations)]
            for f in files:
                file_display_name = translate("file_info_format_string", self.module_translations).format(
                    name=f.get('name'),
                    size_kb=f.get('sizeKB', 0),
                    type=f.get('type', translate("not_available_abbreviation", self.module_translations))
                )
                file_choices.append(file_display_name)
            # Le dropdown des fichiers devient actif s'il y a des fichiers
            is_interactive = len(file_choices) > 1 
            return gr.update(
                choices=file_choices, 
                value=translate("select_file_prompt", self.module_translations),
                interactive=is_interactive
            )
        return gr.update(
            choices=[translate("select_file_prompt", self.module_translations)], 
            value=translate("select_file_prompt", self.module_translations),
            interactive=False # Le dropdown des fichiers devient inactif
        )

    def download_selected_file(self, selected_file_info_str, civitai_api_key):
        if not self.current_model_details or \
           selected_file_info_str == translate("select_file_prompt", self.module_translations) or \
           not self.current_model_details.get("modelVersions"):
            gr.Warning(translate("error_no_file_selected", self.module_translations))
            return translate("error_no_file_selected", self.module_translations)
        file_name_to_match = selected_file_info_str.split(" (")[0]
        selected_file_obj = None
        selected_version_obj = None
        for version in self.current_model_details.get("modelVersions", []):
            for file_obj in version.get("files", []):
                if file_obj.get("name") == file_name_to_match:
                    selected_file_obj = file_obj
                    selected_version_obj = version
                    break
            if selected_file_obj:
                break
        if not selected_file_obj or not selected_version_obj:
            warn_msg = translate("error_file_not_found_in_model_warn", self.module_translations).format(filename=file_name_to_match)
            gr.Warning(warn_msg)
            return translate("error_file_not_found_in_model_return", self.module_translations).format(filename=file_name_to_match)
        download_url = selected_version_obj.get("downloadUrl")
        if not download_url:
            warn_msg = translate("error_download_url_missing_warn", self.module_translations)
            gr.Warning(warn_msg)
            return translate("error_download_url_missing_return", self.module_translations)
        if civitai_api_key:
            download_url += f"&token={civitai_api_key}" if "?" in download_url else f"?token={civitai_api_key}"
        
        # Determine the model type, with a fallback mechanism if not specified in model details
        # or if the translation for the default type key is missing.
        default_type_translation_key = "model_type_Checkpoint"
        hardcoded_fallback_type = "Checkpoint" # Fallback if "model_type_Checkpoint" is not translated

        translated_default_type_name = translate(default_type_translation_key, self.module_translations)
        
        actual_default_model_type_for_get = hardcoded_fallback_type \
            if translated_default_type_name == f"[{default_type_translation_key}]" \
            else translated_default_type_name
            
        model_type_civitai = self.current_model_details.get("type", actual_default_model_type_for_get)
        target_dir = self._get_target_directory(model_type_civitai)
        if not target_dir:
            return translate("error_cannot_determine_save_folder", self.module_translations)
        file_name_on_disk = selected_file_obj.get("name", "downloaded_model")
        gr.Info(translate("download_starting", self.module_translations).format(filename=file_name_on_disk, directory=target_dir))
        final_file_path = os.path.join(target_dir, file_name_on_disk)
        if os.path.exists(final_file_path):
            msg = translate("file_already_exists", self.module_translations).format(filename=file_name_on_disk, directory=target_dir)
            gr.Warning(msg)
            return msg
        download_successful = telechargement_modele(download_url, file_name_on_disk, target_dir, self.module_translations)
        if download_successful:
            success_msg = translate("download_complete", self.module_translations).format(filepath=final_file_path)
            gr.Info(success_msg)
            if self.gestionnaire:
                 pass
            return success_msg
        else:
            return translate("download_failed", self.module_translations).format(error=translate("download_error_see_logs", self.module_translations))

    def handle_js_model_selection_trigger(self, model_name_from_js):
        if not model_name_from_js or not self.current_search_results:
            self.current_model_details = None
            return gr.update(value=""), \
                   gr.update(choices=[translate("select_version_prompt", self.module_translations)], 
                             value=translate("select_version_prompt", self.module_translations), 
                             interactive=False), \
                   gr.update(choices=[translate("select_file_prompt", self.module_translations)], 
                             value=translate("select_file_prompt", self.module_translations), 
                             interactive=False), \
                   gr.update(value="")

        selected_model = next((m for m in self.current_search_results if m["name"] == model_name_from_js), None)
        if selected_model:
            self.current_model_details = selected_model
            description_html = self.current_model_details.get("description", "")
            if description_html is None: description_html = ""
            versions = self.current_model_details.get("modelVersions", [])
            version_choices = [translate("select_version_prompt", self.module_translations)] + [v.get("name", f"Version ID: {v.get('id')}") for v in versions]
            # Le dropdown des versions devient actif s'il y a des versions
            versions_interactive = len(version_choices) > 1
            return gr.update(value=description_html), \
                   gr.update(choices=version_choices, 
                             value=translate("select_version_prompt", self.module_translations), 
                             interactive=versions_interactive), \
                   gr.update(choices=[translate("select_file_prompt", self.module_translations)], 
                             value=translate("select_file_prompt", self.module_translations), 
                             interactive=False), gr.update(value="")

        error_msg = translate("error_model_not_found_details", self.module_translations)
        return gr.update(value=error_msg), \
               gr.update(choices=[translate("select_version_prompt", self.module_translations)], value=translate("select_version_prompt", self.module_translations), interactive=False), \
               gr.update(choices=[translate("select_file_prompt", self.module_translations)], value=translate("select_file_prompt", self.module_translations), interactive=False), \
               gr.update(value="")

    def create_tab(self, module_translations_override=None):
        if module_translations_override:
            self.module_translations = module_translations_override

        # Vérifier si la clé API est la valeur par défaut et informer l'utilisateur
        if self.civitai_api_key_from_config == "your_key_here":
            gr.Info(translate("civitai_api_key_placeholder_notice", self.module_translations))


        with gr.Tab(translate("civitai_downloader_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('civitai_downloader_tab_title', self.module_translations)}")
            current_page_state = gr.State(1)
            hidden_model_name_trigger_for_js = gr.Textbox(
                label="HiddenModelTrigger",
                visible=False, 
                elem_id="civitai_downloader_hidden_model_trigger")
            with gr.Row():
                with gr.Column(scale=1):
                    civitai_api_key_input = gr.Textbox(
                        label=translate("api_key_label", self.module_translations), 
                        type="password", 
                        info=translate("api_key_info", self.module_translations),
                        value=self.civitai_api_key_from_config) # Utiliser la valeur de config
                    search_query_input = gr.Textbox(label=translate("search_query_label", self.module_translations))
                    model_type_choices_map = {
                        "all_types": "All Types", "checkpoint_type": "Checkpoint", "lora_type": "LORA",
                        "textual_inversion_type": "TextualInversion", "hypernetwork_type": "Hypernetwork",
                        "controlnet_type": "Controlnet", "vae_type": "VAE", "dora_type": "DoRA"
                    }
                    model_type_dropdown = gr.Dropdown(
                        label=translate("model_type_label", self.module_translations),
                        choices=[translate(k, self.module_translations) for k in model_type_choices_map.keys()],
                        value=translate("all_types", self.module_translations)
                    )
                    sort_by_choices_map = {
                        "sort_newest": "Newest", "sort_most_downloaded": "Most Downloaded", "sort_highest_rated": "Highest Rated",
                    }
                    sort_by_dropdown = gr.Dropdown(
                        label=translate("sort_by_label", self.module_translations),
                        choices=[translate(k, self.module_translations) for k in sort_by_choices_map.keys()],
                        value=translate("sort_newest", self.module_translations)
                    )
                    period_choices_map = {
                        "period_alltime": "AllTime", "period_year": "Year", "period_month": "Month",
                        "period_week": "Week", "period_day": "Day"
                    }
                    period_dropdown = gr.Dropdown(
                        label=translate("period_label", self.module_translations),
                        choices=[translate(k, self.module_translations) for k in period_choices_map.keys()],
                        value=translate("period_alltime", self.module_translations)
                    )
                    nsfw_choices_map = {
                        "nsfw_none": "None", "nsfw_soft": "Soft", 
                        "nsfw_mature": "Mature", "nsfw_x": "X", "nsfw_all": "All"
                    }
                    nsfw_dropdown = gr.Dropdown(
                        label=translate("nsfw_label", self.module_translations),
                        choices=[translate(k, self.module_translations) for k in nsfw_choices_map.keys()],
                        value=translate("nsfw_soft", self.module_translations)
                    )
                    sdxl_filter_checkbox = gr.Checkbox(label=translate("sdxl_filter_label", self.module_translations), value=True)
                    limit_slider = gr.Slider(minimum=5, maximum=100, value=20, step=5, label=translate("limit_label", self.module_translations))
                    search_button = gr.Button(translate("search_button_label", self.module_translations), variant="primary")
                
                with gr.Column(scale=3):
                    with gr.Row():
                        page_info_display = gr.Textbox(label=translate("page_info_label", self.module_translations), interactive=False, value=translate("page_display", self.module_translations).format(current=1, total=1))
                        prev_page_button = gr.Button(translate("previous_page", self.module_translations), interactive=False)
                        next_page_button = gr.Button(translate("next_page", self.module_translations), interactive=False)
                    search_results_html = gr.HTML(f"<p>{translate('select_model_prompt', self.module_translations)}</p>")
                    with gr.Group(elem_id="civitai_downloader_details_container"): # Remplacer gr.Box par gr.Group
                        gr.Markdown("---")
                        gr.Markdown(f"### {translate('model_details_label', self.module_translations)}")
                        model_description_html = gr.HTML()
                        model_version_dropdown = gr.Dropdown(
                            label=translate("model_versions_label", self.module_translations), 
                            choices=[translate("select_version_prompt", self.module_translations)], 
                            value=translate("select_version_prompt", self.module_translations), 
                            interactive=False) # Initialement inactif
                        model_file_dropdown = gr.Dropdown(
                            label=translate("model_files_label", self.module_translations), 
                            choices=[translate("select_file_prompt", self.module_translations)], 
                            value=translate("select_file_prompt", self.module_translations), 
                            interactive=False) # Initialement inactif
                        download_button = gr.Button(translate("download_button_label", self.module_translations))
                        download_status_textbox = gr.Textbox(label=translate("download_status_label", self.module_translations), interactive=False)

            search_inputs = [search_query_input, model_type_dropdown, sort_by_dropdown, period_dropdown, nsfw_dropdown, limit_slider, current_page_state, civitai_api_key_input, sdxl_filter_checkbox]
            search_outputs = [
                search_results_html, page_info_display, current_page_state, 
                prev_page_button, next_page_button,
                model_description_html, model_version_dropdown, 
                model_file_dropdown, download_status_textbox
            ]

            def reset_page_and_search(*args):
                return self.search_models(*args[:6], page=1, civitai_api_key=args[-2], filter_sdxl=args[-1])
            search_button.click(fn=reset_page_and_search, inputs=search_inputs, outputs=search_outputs)
            
            def go_to_prev_page(*args):
                current_page = args[-3]
                if current_page > 1:
                    return self.search_models(*args[:-3], page=current_page - 1, civitai_api_key=args[-2], filter_sdxl=args[-1])
                return [gr.update()] * len(search_outputs)
            prev_page_button.click(fn=go_to_prev_page, inputs=search_inputs, outputs=search_outputs)

            def go_to_next_page(*args):
                current_page = args[-3]
                return self.search_models(*args[:-3], page=current_page + 1, civitai_api_key=args[-2], filter_sdxl=args[-1])
            next_page_button.click(fn=go_to_next_page, inputs=search_inputs, outputs=search_outputs)
            
            model_version_dropdown.change(fn=self.update_version_selection, inputs=[model_version_dropdown], outputs=[model_file_dropdown])
            download_button.click(fn=self.download_selected_file, inputs=[model_file_dropdown, civitai_api_key_input], outputs=[download_status_textbox])
            hidden_model_name_trigger_for_js.change(
                fn=self.handle_js_model_selection_trigger,
                inputs=[hidden_model_name_trigger_for_js],
                outputs=[model_description_html, model_version_dropdown, model_file_dropdown, download_status_textbox]
            )
        
        return tab
