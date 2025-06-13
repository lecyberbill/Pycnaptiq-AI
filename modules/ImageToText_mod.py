#\modules\ImageToText_mod.py
import os
import json
import time
import traceback
from PIL import Image
import gradio as gr
import threading
import fnmatch # Pour le filtrage des noms de fichiers
import os # Assurer que os est importé

# Imports depuis l'application principale
from Utils.utils import txt_color, translate, GestionModule
from Utils.model_manager import ModelManager
from core.image_prompter import generate_prompt_from_image, unload_caption_model as unload_florence2_model, FLORENCE2_TASKS, MODEL_ID_FLORENCE2 as DEFAULT_FLORENCE2_MODEL_ID, DEFAULT_FLORENCE2_TASK


# --- Configuration du Module ---
MODULE_NAME = "ImageToText"
# Le chemin du JSON est utilisé par GestionModule, pas besoin de le charger ici directement
# si GestionModule fusionne correctement les traductions.

# --- Constantes ---
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']

def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
    """Initialise le module ImageToText."""
    print(txt_color("[OK] ", "ok"), f"Initialisation du module {MODULE_NAME}")
    return ImageToTextModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class ImageToTextModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
        self.gestionnaire = gestionnaire_instance
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.module_translations = {} # Sera peuplé par create_tab
        self.stop_event = threading.Event()
        self.is_processing = False # Drapeau pour éviter les exécutions multiples
        self.status_log_list = []
        self.max_log_entries = 50

    def _translate(self, key, **kwargs):
        """Helper pour utiliser les traductions du module."""
        # Assure que module_translations est utilisé s'il est peuplé, sinon fallback sur global
        active_translations = self.module_translations if self.module_translations else self.global_translations
        return translate(key, active_translations).format(**kwargs)

    def create_tab(self, module_translations_from_gestionnaire):
        """Crée l'onglet Gradio pour ce module."""
        self.module_translations = module_translations_from_gestionnaire

        with gr.Tab(self._translate("img2txt_tab_title")) as tab:
            gr.Markdown(f"## {self._translate('img2txt_module_title')}")

            with gr.Row():
                with gr.Column(scale=2):
                    directory_input = gr.Textbox(
                        label=self._translate("img2txt_directory_input_label"),
                        info=self._translate("img2txt_directory_input_info"),
                        placeholder="/chemin/vers/vos/images"
                    )
                    # Nouveau dropdown pour la tâche Florence-2
                    florence2_task_dropdown = gr.Dropdown(
                        label=self._translate("img2txt_florence2_task_label"),
                        choices=[self._translate(f"florence2_task_{task_key.strip('<>').lower()}") for task_key in FLORENCE2_TASKS],
                        value=self._translate(f"florence2_task_{FLORENCE2_TASKS[0].strip('<>').lower()}"), # Valeur par défaut
                        info=self._translate("img2txt_florence2_task_info")
                    )
                    filename_filter_input = gr.Textbox(
                        label=self._translate("img2txt_filter_label"),
                        placeholder="*.png (laisser vide pour tous)",
                        value=""
                    )
                    recursive_checkbox = gr.Checkbox(
                        label=self._translate("img2txt_recursive_label"),
                        value=True, # Par défaut, scanner récursivement
                        info=self._translate("img2txt_recursive_info")
                    )
                    overwrite_checkbox = gr.Checkbox(
                        label=self._translate("img2txt_overwrite_checkbox_label"),
                        value=False
                    )
                    with gr.Row():
                        start_button = gr.Button(self._translate("img2txt_start_button"), variant="primary")
                        stop_button = gr.Button(self._translate("img2txt_stop_button_label"), variant="stop", interactive=False)
                    with gr.Row():
                        unload_model_button = gr.Button(self._translate("img2txt_unload_model_button"), variant="secondary")


                with gr.Column(scale=3):
                    status_output_html = gr.HTML(label=self._translate("img2txt_status_label"), value="<p>Prêt.</p>")

            def process_directory_wrapper(dir_path, selected_task_display_name, filename_filter, recursive, overwrite, progress=gr.Progress(track_tqdm=True)):
                if self.is_processing:
                    gr.Warning(self._translate("img2txt_already_processing"))
                    yield status_output_html.value, gr.update(interactive=False), gr.update(interactive=True)
                    return

                self.is_processing = True
                self.stop_event.clear()
                # Mettre à jour les boutons : désactiver start, activer stop
                yield status_output_html.value, gr.update(interactive=False), gr.update(interactive=True)
                
                status_log_list = []
                summary_report_data = [] # Pour stocker les données du rapport JSON

                def log_status_and_update_ui(message_key, type="info", **kwargs):
                    message = self._translate(message_key, **kwargs)
                    color_map = {"ok": "green", "warning": "orange", "error": "red", "info": "#ADD8E6"} # Light blue for info
                    html_color = color_map.get(type, "white")
                    timestamp = time.strftime("%H:%M:%S")
                    
                    # Ajoute le nouveau message au début de la liste
                    self.status_log_list.insert(0, f"<span style='color:{html_color};'>[{timestamp}] {message}</span>")
                    
                    # Garde seulement les N derniers messages
                    
                    if len(self.status_log_list) > self.max_log_entries:
                        self.status_log_list.pop() # Enlève le plus ancien
                        
                    return "<br>".join(self.status_log_list)


                if not dir_path:
                    new_html = log_status_and_update_ui("img2txt_no_directory_selected", "error")
                    self.is_processing = False
                    yield new_html, gr.update(interactive=True), gr.update(interactive=False)
                    return

                if not os.path.isdir(dir_path):
                    new_html = log_status_and_update_ui("img2txt_directory_not_found", "error", directory=dir_path)
                    self.is_processing = False
                    yield new_html, gr.update(interactive=True), gr.update(interactive=False)
                    return
                
                current_html = log_status_and_update_ui("img2txt_processing_directory", "info", directory=dir_path)
                yield current_html, gr.update(), gr.update() # This yield is for the "processing directory" message

                # Mapper le nom affiché de la tâche à sa clé interne
                # Construire le map basé sur FLORENCE2_TASKS importé et les traductions du module
                task_display_to_key_map = {}
                for task_key_iter in FLORENCE2_TASKS:
                    # La clé de traduction est construite comme: "florence2_task_" + la version en minuscules de la tâche sans les chevrons
                    # Exemple: "<MORE_DETAILED_CAPTION>" -> "florence2_task_more_detailed_caption"
                    translation_suffix = task_key_iter.strip('<>').lower().replace(' ', '_')
                    display_name = self._translate(f"florence2_task_{translation_suffix}")
                    task_display_to_key_map[display_name] = task_key_iter
                
                selected_internal_task = None
                selected_internal_task = task_display_to_key_map.get(selected_task_display_name)

                if selected_internal_task is None: # Fallback si non trouvé (ne devrait pas arriver)
                    current_html = log_status_and_update_ui("img2txt_task_mapping_error", "warning", display_name=selected_task_display_name, default_task=DEFAULT_FLORENCE2_TASK)
                    yield current_html, gr.update(), gr.update()
                    selected_internal_task = DEFAULT_FLORENCE2_TASK

                image_files = []
                for root, _, files in os.walk(dir_path, topdown=True): # topdown=True est la valeur par défaut
                    if self.stop_event.is_set(): break
                    for f_name in files:
                        if self.stop_event.is_set(): break
                        ext = os.path.splitext(f_name)[1].lower()
                        if ext in SUPPORTED_IMAGE_EXTENSIONS:
                            if filename_filter and filename_filter.strip():
                                if fnmatch.fnmatch(f_name, filename_filter.strip()):
                                    image_files.append(os.path.join(root, f_name))
                            else:
                                image_files.append(os.path.join(root, f_name))
                    if not recursive: # Si non récursif, arrêter après le répertoire principal
                        break

                # Filtrer les fichiers pour s'assurer qu'ils existent (peut arriver avec des liens symboliques ou erreurs)
                image_files = [f for f in image_files if os.path.exists(f)]

                if self.stop_event.is_set():
                    current_html = log_status_and_update_ui("img2txt_stopped", "warning")
                    self.is_processing = False
                    yield current_html, gr.update(interactive=True), gr.update(interactive=False)
                    return

                if not image_files:
                    current_html = log_status_and_update_ui("img2txt_no_images_found", "warning", directory=dir_path)
                    self.is_processing = False
                    yield current_html, gr.update(interactive=True), gr.update(interactive=False)
                    return

                current_html = log_status_and_update_ui("img2txt_found_images", "info", count=len(image_files))
                yield current_html, gr.update(), gr.update()

                processed_count = 0
                error_count = 0
                total_images = len(image_files)
                successful_count = 0 # Pour compter les succès
                skipped_count = 0 # Pour compter les fichiers ignorés

                for i, img_path in enumerate(progress.tqdm(image_files, desc=self._translate("img2txt_module_title"))):
                    if self.stop_event.is_set():
                        current_html = log_status_and_update_ui("img2txt_stopped", "warning")
                        break
                    
                    img_name = os.path.basename(img_path)
                    txt_filename = os.path.splitext(img_name)[0] + ".txt"
                    txt_filepath = os.path.join(os.path.dirname(img_path), txt_filename)
                    file_process_start_time = time.time()

                    current_html = log_status_and_update_ui("img2txt_processing_image", "info", image_name=img_name, current=i + 1, total=total_images)
                    yield current_html, gr.update(), gr.update()

                    if not overwrite and os.path.exists(txt_filepath):
                        current_html = log_status_and_update_ui("img2txt_skipped_exists", "info", txt_filename=txt_filename)
                        processed_count += 1
                        file_processing_time = time.time() - file_process_start_time
                        summary_report_data.append({
                            "image_filename": img_name,
                            "status": "skipped",
                            "method_used": selected_internal_task,
                            "output_text": self._translate("img2txt_skipped_reason_exists"), # Nouvelle clé
                            "processing_time_seconds": round(file_processing_time, 2),
                            "output_txt_filename": txt_filename,
                            "output_txt_path": txt_filepath,
                            "error_details": None # Pas d'erreur, juste ignoré
                        })
                        yield current_html, gr.update(), gr.update()
                        continue
                    
                    try:
                        pil_image = Image.open(img_path).convert("RGB")
                        # Passer la tâche sélectionnée à la fonction de génération de prompt
                        prompt_text = generate_prompt_from_image(
                            pil_image,
                            self.module_translations, # Utiliser les traductions du module pour les logs internes de generate_prompt
                            task=selected_internal_task,
                            unload_after=False, # Important: Ne pas décharger après chaque image pour le lot
                            model_id=DEFAULT_FLORENCE2_MODEL_ID # Utiliser le modèle par défaut ou un configurable
                        )
                        error_marker = f"[{translate('erreur', self.global_translations).upper()}]"
                        if prompt_text.startswith(error_marker):
                            error_msg_from_prompter = prompt_text.replace(error_marker, "").strip()
                            current_html = log_status_and_update_ui("img2txt_error_processing_image", "error", image_name=img_name, error=error_msg_from_prompter)
                            error_count += 1
                            file_processing_time = time.time() - file_process_start_time
                            summary_report_data.append({
                                "image_filename": img_name,
                                "status": "error",
                                "method_used": selected_internal_task,
                                "output_text": None, # Pas de texte généré en cas d'erreur
                                "processing_time_seconds": round(file_processing_time, 2),
                                "output_txt_filename": None,
                                "output_txt_path": None,
                                "error_details": error_msg_from_prompter
                            })
                        else:
                            with open(txt_filepath, "w", encoding="utf-8") as f_out:
                                f_out.write(prompt_text)
                            current_html = log_status_and_update_ui("img2txt_prompt_saved", "ok", txt_file_path=txt_filepath)
                            processed_count += 1
                            successful_count += 1
                            file_processing_time = time.time() - file_process_start_time
                            summary_report_data.append({
                                "image_filename": img_name,
                                "status": "processed",
                                "method_used": selected_internal_task,
                                "output_text": prompt_text,
                                "processing_time_seconds": round(file_processing_time, 2),
                                "output_txt_filename": txt_filename,
                                "output_txt_path": txt_filepath,
                                "error_details": None
                            })
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        current_html = log_status_and_update_ui("img2txt_error_processing_image", "error", image_name=img_name, error=str(e))
                        print(txt_color(f"[ERREUR] Traitement {img_name}: {e}\n{tb_str}", "erreur"))
                        error_count += 1
                        file_processing_time = time.time() - file_process_start_time
                        summary_report_data.append({
                            "image_filename": img_name,
                            "status": "error",
                            "method_used": selected_internal_task,
                            "output_text": None,
                            "processing_time_seconds": round(file_processing_time, 2),
                            "output_txt_filename": None,
                            "output_txt_path": None,
                            "error_details": str(e) + "\n" + tb_str # Inclure la trace complète dans le rapport
                        })
                    
                    yield current_html, gr.update(), gr.update()
                    time.sleep(0.01) # Petit délai pour l'affichage Gradio

                final_message_key = "img2txt_stopped" if self.stop_event.is_set() else "img2txt_completed"
                final_status_type = "warning" if self.stop_event.is_set() or error_count > 0 else "ok"
                current_html = log_status_and_update_ui(final_message_key, final_status_type, processed_count=processed_count, error_count=error_count, skipped_count=skipped_count)

                # --- Sauvegarde du rapport JSON ---
                if summary_report_data: # Sauvegarder seulement s'il y a des données
                    try:
                        # Utiliser le répertoire d'entrée comme base pour le répertoire du rapport
                        report_base_dir = dir_path 
                        reports_dir = os.path.join(report_base_dir, "reports_img2txt") # Sous-répertoire spécifique dans le dossier d'entrée
                        os.makedirs(reports_dir, exist_ok=True)

                        timestamp_report = time.strftime("%Y%m%d_%H%M%S")
                        report_filename = f"img2txt_report_{timestamp_report}.json"
                        report_filepath = os.path.join(reports_dir, report_filename)

                        with open(report_filepath, "w", encoding="utf-8") as f_report:
                            json.dump(summary_report_data, f_report, indent=4, ensure_ascii=False)

                        report_save_msg = self._translate("img2txt_report_saved", report_path=report_filepath)
                        current_html += f"<br><span style='color:green;'>[{time.strftime('%H:%M:%S')}] {report_save_msg}</span>"
                        print(txt_color("[OK]", "ok"), report_save_msg)
                        gr.Info(report_save_msg, 3.0)

                    except Exception as e_report:
                        report_error_msg = self._translate("img2txt_report_save_error", error=str(e_report))
                        current_html += f"<br><span style='color:red;'>[{time.strftime('%H:%M:%S')}] {report_error_msg}</span>"
                        print(txt_color("[ERREUR]", "erreur"), report_error_msg)

                self.is_processing = False
                yield current_html, gr.update(interactive=True), gr.update(interactive=False)

            start_button.click(
                fn=process_directory_wrapper,
                inputs=[directory_input, florence2_task_dropdown, filename_filter_input, recursive_checkbox, overwrite_checkbox],
                outputs=[status_output_html, start_button, stop_button]
            )

            def stop_processing_func():
                if self.is_processing:
                    self.stop_event.set()
                    gr.Info(self._translate("img2txt_stopping"))
                # Le bouton stop sera réactivé par le wrapper à la fin du traitement
                return gr.update(interactive=False) 

            stop_button.click(
                fn=stop_processing_func,
                inputs=[],
                outputs=[stop_button]
            )
        return tab

    def handle_unload_model(self):
        if self.is_processing:
            updated_html = self._log_status_and_update_ui_direct("img2txt_cannot_unload_processing", "warning")
            gr.Warning(self._translate("img2txt_cannot_unload_processing"))
            return updated_html, gr.update(interactive=False) # Garder le bouton start désactivé

        status_message_from_core = unload_florence2_model(self.module_translations)

        log_type = "ok"
        if "aucun modèle" in status_message_from_core.lower() or \
           "no model" in status_message_from_core.lower() or \
           "not loaded" in status_message_from_core.lower():
            log_type = "info"
        elif "erreur" in status_message_from_core.lower() or "error" in status_message_from_core.lower():
            log_type = "error"

        updated_html = self._log_status_and_update_ui_direct("img2txt_unload_status_message", log_type, message=status_message_from_core)
        gr.Info(status_message_from_core)
        return updated_html, gr.update(interactive=True) # Réactiver le bouton start

    def _log_status_and_update_ui_direct(self, message_key_or_raw_message, type="info", is_raw=False, **kwargs):
        message = message_key_or_raw_message if is_raw else self._translate(message_key_or_raw_message, **kwargs)
        color_map = {"ok": "green", "warning": "orange", "error": "red", "info": "#ADD8E6"}
        html_color = color_map.get(type, "white")
        timestamp = time.strftime("%H:%M:%S")
        self.status_log_list.insert(0, f"<span style='color:{html_color};'>[{timestamp}] {message}</span>")
        if len(self.status_log_list) > self.max_log_entries:
            self.status_log_list.pop()
        return "<br>".join(self.status_log_list)
