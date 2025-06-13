#\image_to_text\cyberbill_SDXL\cyberbill_image_generator\modules\LoRATraining_mod.py
import os
import json
import time
import traceback
from PIL import Image
import gradio as gr
import torch
import gc
import contextlib # Ajout pour contextlib.nullcontext
import threading
import shutil 
import queue # Ajout de l'import manquant
from pathlib import Path
from tqdm.auto import tqdm # Pour la barre de progression dans la console
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline, # Utilisé pour save_lora_weights et potentiellement le chargement initial simplifié
)
from diffusers.optimization import get_scheduler
import safetensors.torch # AJOUT: Pour la sauvegarde en fichier unique
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model_state_dict, get_peft_model
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers.training_utils import cast_training_params
import random

# Imports depuis l'application principale
from Utils.utils import txt_color, translate, GestionModule
from Utils.model_manager import ModelManager
from core.image_prompter import generate_prompt_from_image, unload_caption_model as unload_image_prompter_model, FLORENCE2_TASKS, DEFAULT_FLORENCE2_TASK

MODULE_NAME = "LoRATraining" 
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp']

# --- Dataset Class ---
class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt_fallback, tokenizer_one, tokenizer_two, target_size=1024, center_crop=False):
        self.instance_data_root = instance_data_root
        self.instance_prompt_fallback = instance_prompt_fallback # Utilisé si .txt manque
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.target_size = target_size # Taille cible pour l'entraînement
        self.center_crop = center_crop

        self.instance_images_path = []
        self.original_sizes = [] # Pour stocker les tailles originales
        self.crop_coords_top_left = [] # Pour stocker les coordonnées de crop
        self.instance_captions = []

        for root, _, files in os.walk(instance_data_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
                    img_path = os.path.join(root, file)
                    caption_path = os.path.join(root, os.path.splitext(file)[0] + ".txt")
                    if os.path.exists(caption_path):
                        with open(caption_path, "r", encoding="utf-8") as f_cap:
                            caption = f_cap.read().strip()
                        self.instance_images_path.append(img_path)
                        # Charger l'image pour obtenir sa taille originale
                        with Image.open(img_path) as img_orig:
                            self.original_sizes.append(img_orig.size) # (width, height)
                        self.instance_captions.append(caption)
                        self.crop_coords_top_left.append((0,0)) # Par défaut (0,0) si pas de crop spécifique
                    else:
                        # Fallback si pas de caption, utiliser le prompt d'instance
                        self.instance_images_path.append(img_path)
                        with Image.open(img_path) as img_orig:
                            self.original_sizes.append(img_orig.size)
                        self.instance_captions.append(self.instance_prompt_fallback)
                        self.crop_coords_top_left.append((0,0))
        
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        # Transformations d'image
        self.image_transforms_resize = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.image_transforms_crop = transforms.CenterCrop(self.target_size) if self.center_crop else transforms.RandomCrop(self.target_size)
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalise entre -1 et 1
        ])
    
    def _tokenize_prompt(self, tokenizer, prompt):
        return tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids.squeeze()

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_caption = self.instance_captions[index % self.num_instance_images]
        original_size_hw = self.original_sizes[index % self.num_instance_images][::-1] # (height, width)
        
        try:
            instance_image = Image.open(instance_image_path)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            
            # Appliquer resize
            resized_image = self.image_transforms_resize(instance_image)

            # Calculer crop_coords_top_left par rapport à l'image redimensionnée
            # avant le crop final. Pour CenterCrop, c'est simple. Pour RandomCrop, c'est plus complexe.
            # Le script de référence calcule y1, x1 après resize.
            if self.center_crop:
                y1 = max(0, int(round((resized_image.height - self.target_size) / 2.0)))
                x1 = max(0, int(round((resized_image.width - self.target_size) / 2.0)))
            else: # RandomCrop
                y1, x1, _, _ = self.image_transforms_crop.get_params(resized_image, (self.target_size, self.target_size))
            
            cropped_image = self.image_transforms_crop(resized_image)
            example["instance_images"] = self.image_transforms(cropped_image) # Applique ToTensor et Normalize

            example["input_ids_one"] = self._tokenize_prompt(self.tokenizer_one, instance_caption)
            example["input_ids_two"] = self._tokenize_prompt(self.tokenizer_two, instance_caption)
            
            example["original_size_hw"] = original_size_hw # (height, width)
            example["crop_coords_top_left_yx"] = (y1, x1) # (top, left) ou (y, x)

        except Exception as e:
            print(f"Error loading/processing image or caption {instance_image_path}: {e}")
            # Return a dummy item or skip
            return self.__getitem__(random.randint(0, len(self) -1))


        return example

def collate_fn(examples):
    pixel_values = torch.stack([example["instance_images"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
    input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
    
    original_sizes_hw = [example["original_size_hw"] for example in examples]
    crop_coords_top_left_yx = [example["crop_coords_top_left_yx"] for example in examples]

    # Les légendes brutes ne sont plus nécessaires ici si on a déjà les input_ids
    # captions = [example["instance_captions"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
        "original_sizes_hw": original_sizes_hw,
        "crop_coords_top_left_yx": crop_coords_top_left_yx,
        # "captions": captions 
    }



def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
    print(txt_color("[OK] ", "ok"), f"Initialisation du module {MODULE_NAME}")
    return LoRATrainingModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)

class LoRATrainingModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
        self.gestionnaire = gestionnaire_instance
        self.global_config = global_config
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.module_translations = {}
        self.is_training = False
        self.stop_event = threading.Event()
        self.is_preparing = False # Ajout d'un drapeau pour la préparation
        self.status_log_list = []
        self.max_log_entries = 100 # Pour l'affichage des logs dans l'UI

        self.default_lora_project_dir = global_config.get("LORA_PROJECTS_DIR", "LoRA_Projects_Internal")
        # Chemin où les LoRAs finaux seront sauvegardés pour utilisation par l'application
        self.final_loras_save_dir = self._get_absolute_path_from_config("LORAS_DIR", "models/loras")

        # Pour le dropdown du modèle de base
        default_model_path_from_config = global_config.get("DEFAULT_SDXL_MODEL_FOR_LORA", "")
        self.default_sdxl_base_model_name = os.path.basename(default_model_path_from_config) if default_model_path_from_config else None

        self.device = model_manager_instance.device if model_manager_instance else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

    def _get_absolute_path_from_config(self, config_key, default_relative_path):
        path_from_config = self.global_config.get(config_key, default_relative_path)
        if not os.path.isabs(path_from_config):
            # Supposer que global_config contient des chemins relatifs au dossier racine du projet
            # ou que ModelManager peut résoudre cela. Pour l'instant, on le fait par rapport à la racine.
            project_root = Path(__file__).resolve().parent.parent # cyberbill_image_generator
            return str(project_root / path_from_config)
        return path_from_config

    def _translate(self, key, **kwargs):
        active_translations = self.module_translations if self.module_translations else self.global_translations
        return translate(key, active_translations).format(**kwargs)

    def _log_status_and_update_ui(self, message_key, type="info", **kwargs):
        message = self._translate(message_key, **kwargs)
        color_map = {"ok": "green", "warning": "orange", "error": "red", "info": "#ADD8E6"}
        html_color = color_map.get(type, "white")
        timestamp = time.strftime("%H:%M:%S")
        
        self.status_log_list.insert(0, f"<span style='color:{html_color};'>[{timestamp}] {message}</span>")
        if len(self.status_log_list) > self.max_log_entries:
            self.status_log_list.pop()
        return "<br>".join(self.status_log_list)

    def create_tab(self, module_translations_from_gestionnaire):
        self.module_translations = module_translations_from_gestionnaire
        
        florence2_task_choices_translated = [self._translate(f"florence2_task_{tk.strip('<>').lower().replace(' ', '_')}") for tk in FLORENCE2_TASKS]
        default_florence2_task_translated = self._translate(f"florence2_task_{DEFAULT_FLORENCE2_TASK.strip('<>').lower().replace(' ', '_')}")
        
        optimizer_choices = ["AdamW", "AdamW8bit", "Lion"] # Exemple
        
        # Définir les choix pour le learning rate avec explications
        learning_rate_choices_with_desc = {
            "5e-4": self._translate("lora_lr_5e_4_desc"), # "Très rapide, pour tests, risque élevé"
            "1e-4": self._translate("lora_lr_1e_4_desc"), # "Rapide, bon point de départ"
            "5e-5": self._translate("lora_lr_5e_5_desc"), # "Modéré, plus stable"
            "2e-5": self._translate("lora_lr_2e_5_desc"), # "Lent, pour affinage fin"
            "1e-5": self._translate("lora_lr_1e_5_desc")  # "Très lent, pour affinage très fin"
        }
        scheduler_choices = ["linear", "cosine", "constant", "constant_with_warmup"] # Exemple
        mixed_precision_choices = ["no", "fp16", "bf16"] # bf16 si supporté

        # Liste des modèles pour le dropdown
        available_base_models = self.model_manager.list_models(model_type="standard")
        # S'assurer que la valeur par défaut est valide ou None
        default_base_model_for_ui = self.default_sdxl_base_model_name
        if default_base_model_for_ui not in available_base_models:
            if available_base_models and available_base_models[0] != translate("aucun_modele_trouve", self.global_translations):
                default_base_model_for_ui = available_base_models[0]
            else:
                default_base_model_for_ui = None

        with gr.Tab(self._translate("lora_training_tab_title")) as tab:
            gr.Markdown(f"## {self._translate('lora_training_module_title')}")
            with gr.Row():
                with gr.Column(scale=1): # Colonne de gauche pour la préparation des données
                    gr.Markdown(f"### {self._translate('lora_data_preparation_title')}")
                    input_images_dir = gr.Textbox(label=self._translate("lora_input_images_dir_label"), info=self._translate("lora_input_images_dir_info"), placeholder="C:/chemin/vers/vos/images_sources")
                    trigger_word = gr.Textbox(label=self._translate("lora_trigger_word_label"), placeholder=self._translate("lora_trigger_word_placeholder"))
                    concept = gr.Textbox(label=self._translate("lora_concept_label"), placeholder=self._translate("lora_concept_placeholder"), value="my_concept_data")
                    caption_task_dropdown = gr.Dropdown(
                        label=self._translate("lora_caption_task_label"),
                        choices=florence2_task_choices_translated,
                        value=default_florence2_task_translated,
                        info=self._translate("lora_caption_task_info")
                    )
                    auto_captioning_checkbox = gr.Checkbox( # AJOUT DE LA CASE À COCHER
                        label=self._translate("lora_auto_captioning_label"), # Nouvelle clé de traduction
                        value=True, # Coché par défaut (comportement actuel)
                        info=self._translate("lora_auto_captioning_info") # Nouvelle clé de traduction
                    )
                    
                    gr.Markdown(f"### {self._translate('lora_training_parameters_title')}")
                    lora_output_name = gr.Textbox(label=self._translate("lora_output_name_label"), placeholder=self._translate("lora_output_name_placeholder"))
                    base_model_dropdown = gr.Dropdown( # Changé en Dropdown
                        label=self._translate("lora_base_model_label"),
                        choices=available_base_models,
                        value=default_base_model_for_ui,
                        info=self._translate("lora_base_model_info")
                    )
                    training_project_dir = gr.Textbox(
                        label=self._translate("lora_training_project_dir_label"),
                        info=self._translate("lora_training_project_dir_info"),
                        placeholder=self._translate("lora_training_project_dir_placeholder"),
                        value=self.default_lora_project_dir
                    )
                    with gr.Row():
                        epochs = gr.Slider(label=self._translate("lora_epochs_label"), minimum=1, maximum=200, value=5, step=1)
                        # Remplacer Textbox par Dropdown pour le learning rate
                        learning_rate_dropdown = gr.Dropdown(
                            label=self._translate("lora_learning_rate_label"),
                            choices=[f"{val} - {desc}" for val, desc in learning_rate_choices_with_desc.items()],
                            value=f"1e-5 - {learning_rate_choices_with_desc['1e-5']}", # Valeur par défaut
                            info=self._translate("lora_learning_rate_info_dropdown")) # Info générale pour le dropdown
                    with gr.Row():
                        batch_size = gr.Slider(label=self._translate("lora_batch_size_label"), minimum=1, maximum=16, value=1, step=1)
                        resolution = gr.Slider(label=self._translate("lora_resolution_label"), minimum=512, maximum=2048, value=1024, step=64)
                    
                    with gr.Accordion(self._translate("lora_advanced_network_options_title"), open=False):
                        network_dim = gr.Slider(label=self._translate("lora_network_dim_label"), minimum=1, maximum=256, value=16, step=1)
                        network_alpha = gr.Slider(label=self._translate("lora_network_alpha_label"), minimum=1, maximum=256, value=8, step=1)
                        train_unet_only = gr.Checkbox(label=self._translate("lora_train_unet_only_label"), value=True)
                        train_text_encoder = gr.Checkbox(label=self._translate("lora_train_text_encoder_label"), value=False)
                    mixed_precision_choices_ui = ["fp16", "bf16", "fp32"] # "no" est remplacé par "fp32"
                    with gr.Accordion(self._translate("lora_optimizer_options_title"), open=False):
                        optimizer_dropdown = gr.Dropdown(label=self._translate("lora_optimizer_label"), choices=optimizer_choices, value="AdamW8bit")
                        lr_scheduler_dropdown = gr.Dropdown(label=self._translate("lora_lr_scheduler_label"), choices=scheduler_choices, value="constant")
                        mixed_precision_dropdown = gr.Dropdown(label=self._translate("lora_mixed_precision_label"), choices=mixed_precision_choices_ui, value="fp16" if self.device.type == "cuda" else "fp32")
                        save_every_n_epochs = gr.Slider(label=self._translate("lora_save_every_n_epochs_label"), minimum=0, maximum=50, value=1, step=1)

                with gr.Column(scale=2): # Colonne de droite pour les logs et boutons
                    with gr.Row():
                        prepare_data_button = gr.Button(self._translate("lora_prepare_data_button"), variant="primary") # Renommé
                        stop_button = gr.Button(self._translate("img2txt_stop_button_label"), variant="stop", interactive=False)
                    start_training_button = gr.Button(self._translate("lora_start_training_button"), variant="primary", interactive=False) # Nouveau bouton
                    
                    # Préparer le message de log initial
                    # S'assurer que self.status_log_list est vide pour ce premier message
                    self.status_log_list = [] 
                    initial_log_value = self._log_status_and_update_ui('lora_training_ready', 'info')

                    status_output_html = gr.HTML(
                        label=self._translate("lora_training_status_label"),
                        value=initial_log_value, # Utiliser le message formaté
                        max_height=500,
                        container=True
                    )
                    # log_output_textbox a été supprimé

            # Inputs pour la préparation des données
            prepare_data_inputs = [
                input_images_dir, trigger_word, concept, caption_task_dropdown, auto_captioning_checkbox,
                lora_output_name, training_project_dir, resolution
            ]
            # Inputs pour l'entraînement (certains sont communs, d'autres spécifiques)
            training_inputs = [
                input_images_dir, trigger_word, concept, caption_task_dropdown, auto_captioning_checkbox, # AJOUT auto_captioning_checkbox
                lora_output_name, base_model_dropdown, training_project_dir, # Utiliser base_model_dropdown
                epochs, learning_rate_dropdown, batch_size, resolution, # Utiliser learning_rate_dropdown
                network_dim, network_alpha, train_unet_only, train_text_encoder,
                optimizer_dropdown, lr_scheduler_dropdown, mixed_precision_dropdown, save_every_n_epochs
            ]

            prepare_data_button.click(
                fn=self.run_preparation_thread,
                inputs=prepare_data_inputs,
                outputs=[status_output_html, prepare_data_button, stop_button, start_training_button]
            )

            start_training_button.click(
                fn=self.run_training_thread, # Nouvelle fonction pour démarrer l'entraînement
                inputs=training_inputs, # Utiliser tous les inputs nécessaires pour l'entraînement
                outputs=[status_output_html, start_training_button, stop_button, prepare_data_button]
            )

            def stop_training_wrapper():
                # S'arrête si la préparation OU l'entraînement est en cours
                if self.is_preparing or self.is_training:
                    self.stop_event.set()
                    if self.is_training:
                        gr.Info(self._translate("lora_stopping_training"))
                    if self.is_preparing:
                        gr.Info(self._translate("lora_stopping_preparation")) # Nouvelle clé
                return gr.update(interactive=False)

            stop_button.click(fn=stop_training_wrapper, inputs=None, outputs=[stop_button])
        return tab

    def run_preparation_thread(self, *args):
        # Wrapper pour lancer dans un thread et mettre à jour l'UI
        if self.is_preparing or self.is_training: # Vérifier les deux états
            gr.Warning(self._translate("lora_process_already_running")) # Nouvelle clé plus générique
            # Le yield doit correspondre aux outputs du bouton prepare_data_button
            yield self._log_status_and_update_ui("lora_process_already_running", "warning"), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)
            return

        self.is_preparing = True # Correction: utiliser is_preparing
        self.stop_event.clear()
        self.status_log_list = [] # Réinitialiser les logs UI
        
        # Mettre à jour l'UI pour indiquer le démarrage
        initial_status_html = self._log_status_and_update_ui("lora_preparing_data", "info")
        yield initial_status_html, gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False) # start_training_button reste False

        # Créer et démarrer le thread
        # Le générateur doit être consommé dans le thread principal pour les mises à jour UI
        # Donc, la fonction de thread ne doit pas être un générateur.
        # On va utiliser une queue pour les logs du thread vers le thread principal.
        log_queue = queue.Queue()

        training_thread = threading.Thread(
            target=self._actual_preparation_logic, # Correction ici
            args=(log_queue, *args) # Passer la queue et les autres args
        )
        training_thread.start()

        # Boucle pour récupérer les logs de la queue et mettre à jour l'UI
        # jusqu'à ce que le thread soit terminé.
        final_status_message = initial_status_html # Initialiser avec le premier message
        preparation_really_succeeded_flag = False # Nouveau drapeau pour le succès
        while training_thread.is_alive() or not log_queue.empty():
            try:
                log_entry = log_queue.get(timeout=0.1) # Timeout pour ne pas bloquer indéfiniment
                log_type, log_key, log_kwargs = log_entry
                current_log_html = self._log_status_and_update_ui(log_key, log_type, **log_kwargs)
                if log_key == "lora_data_preparation_complete" and log_type == "ok":
                    preparation_really_succeeded_flag = True
                final_status_message = current_log_html # Garder le dernier message pour le statut final
                yield current_log_html, gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)
            except queue.Empty:
                pass # Pas de nouveau log, on continue
            time.sleep(0.05) # Petit délai pour ne pas surcharger

        training_thread.join() # S'assurer que le thread est bien terminé

        # Récupérer les derniers logs (au cas où)
        while not log_queue.empty():
            log_entry = log_queue.get_nowait()
            log_type, log_key, log_kwargs = log_entry
            if log_key == "lora_data_preparation_complete" and log_type == "ok": # Vérifier aussi ici
                preparation_really_succeeded_flag = True
            current_log_html = self._log_status_and_update_ui(log_key, log_type, **log_kwargs)
            final_status_message = current_log_html # S'assurer que le dernier message est bien capturé
            # Pas besoin de yield ici si la boucle principale a déjà yieldé ce message

        self.is_preparing = False # Correction: utiliser is_preparing
        # Utiliser le nouveau drapeau pour déterminer l'état du bouton
        yield final_status_message, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=preparation_really_succeeded_flag)

    def run_training_thread(self, *args):
        if self.is_training or self.is_preparing:
            gr.Warning(self._translate("lora_process_already_running"))
            # Outputs: status_output_html, start_training_button, stop_button, prepare_data_button
            yield self._log_status_and_update_ui("lora_process_already_running", "warning"), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)
            return

        self.is_training = True
        self.is_preparing = False # Assurer que is_preparing est false
        self.stop_event.clear()
        # Ne pas réinitialiser status_log_list ici pour conserver les logs de préparation

        initial_status_html = self._log_status_and_update_ui("lora_starting_training_phase", "info") # Nouvelle clé
        # Outputs: status_output_html, start_training_button, stop_button, prepare_data_button
        yield initial_status_html, gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)

        log_queue = queue.Queue()
        
        # Le nom de la fonction cible pour le thread d'entraînement
        training_logic_thread = threading.Thread(
            target=self._actual_training_logic, # Nouvelle fonction pour la logique d'entraînement
            args=(log_queue, *args)
        )
        training_logic_thread.start()

        final_status_message = initial_status_html
        while training_logic_thread.is_alive() or not log_queue.empty():
            try:
                log_entry = log_queue.get(timeout=0.1)
                log_type, log_key, log_kwargs = log_entry
                current_log_html = self._log_status_and_update_ui(log_key, log_type, **log_kwargs)
                final_status_message = current_log_html
                # Outputs: status_output_html, start_training_button, stop_button, prepare_data_button
                yield current_log_html, gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)
            except queue.Empty:
                pass
            time.sleep(0.05)

        training_logic_thread.join()

        while not log_queue.empty():
            log_entry = log_queue.get_nowait()
            log_type, log_key, log_kwargs = log_entry
            current_log_html = self._log_status_and_update_ui(log_key, log_type, **log_kwargs)
            final_status_message = current_log_html

        self.is_training = False
        # Outputs: status_output_html, start_training_button, stop_button, prepare_data_button
        yield final_status_message, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)


    def _actual_preparation_logic(
        self, log_queue,
        input_images_dir, trigger_word, concept, caption_task_display_name, auto_captioning_enabled,
        lora_output_name, training_project_dir, resolution
    ):
        if not all([input_images_dir, trigger_word, concept, lora_output_name, training_project_dir]):
            log_queue.put(("error", "lora_error_missing_input_prep", {})) # Nouvelle clé
            self.is_preparing = False
            return False # Indiquer l'échec

    def _encode_prompt(self, text_encoders, tokenizers, prompt_text_list, text_input_ids_list=None):
        """ Similaire à la fonction encode_prompt du script de référence.
            Prend une liste de prompts textuels ou une liste de text_input_ids pré-tokenisés.
        """
        prompt_embeds_list = []
        pooled_prompt_embeds = None # Sera défini par le deuxième encodeur

        for i, text_encoder in enumerate(text_encoders):
            tokenizer = tokenizers[i]
            if text_input_ids_list is None:
                # Tokenize les prompts si non fournis
                text_inputs = tokenizer(
                    prompt_text_list,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(text_encoder.device)
            else:
                text_input_ids = text_input_ids_list[i].to(text_encoder.device)

            prompt_embeds_out = text_encoder(text_input_ids, output_hidden_states=True, return_dict=False)
            
            # Utiliser l'avant-dernier état caché (penultimate)
            current_prompt_embeds = prompt_embeds_out[2][-2] # hidden_states est à l'index 2 du tuple
            prompt_embeds_list.append(current_prompt_embeds)
            if i == 1: # Le deuxième text encoder fournit le pooled output
                pooled_prompt_embeds = prompt_embeds_out[0] # text_embeds (pooled) est à l'index 0 pour CLIPTextModelWithProjection
        
        final_prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return final_prompt_embeds, pooled_prompt_embeds

    def _compute_time_ids(self, original_size_hw, crop_coords_top_left_yx, target_size_hw, device, dtype=torch.float16): # dtype changé pour correspondre à unet
        add_time_ids = list(original_size_hw + crop_coords_top_left_yx + target_size_hw)
        add_time_ids = torch.tensor([add_time_ids], device=device, dtype=dtype)
        return add_time_ids

    # Renommer et adapter la fonction principale
    def _actual_training_logic(
        self, log_queue,
        input_images_dir, trigger_word, concept, caption_task_display_name, auto_captioning_enabled, # AJOUT auto_captioning_enabled
        lora_output_name, base_model_name_selected, training_project_dir,
        epochs, learning_rate_str, batch_size, resolution,
        network_dim, network_alpha, train_unet_only, train_text_encoder,
        optimizer_choice, lr_scheduler_choice, mixed_precision_choice, save_every_n_epochs,
        # progress=gr.Progress(track_tqdm=True) # Gradio Progress n'est plus passé directement
    ):
        # --- Validation des entrées (simplifiée, à compléter) ---
        # La validation des chemins d'images sources n'est plus faite ici, mais dans la préparation.
        # On valide ici les paramètres spécifiques à l'entraînement.
        if not all([lora_output_name, base_model_name_selected, training_project_dir]):
            log_queue.put(("error", "lora_error_missing_training_params", {})) # Nouvelle clé
            return


        start_time_total_training = time.time() # <-- Démarrer le minuteur global ici

        try:
            # Extraire la valeur numérique du learning rate depuis la chaîne du dropdown
            actual_lr_value_str = learning_rate_str.split(" - ")[0]
            learning_rate = float(actual_lr_value_str)
        except ValueError:
            log_queue.put(("error", "lora_error_invalid_lr", {"lr": actual_lr_value_str}))
            return

        # Construire le chemin complet du modèle de base à partir du nom sélectionné.
        # self.model_manager.models_dir devrait être le chemin absolu vers le dossier contenant les modèles.
        # base_model_name_selected devrait être le nom du fichier modèle.
        actual_base_model_path = os.path.join(self.model_manager.models_dir, base_model_name_selected)

        if not os.path.exists(actual_base_model_path):
            log_queue.put(("error", "lora_error_base_model_path_not_exist", {"path": actual_base_model_path})) # Nouvelle clé pour "n'existe pas"
            return

        if not os.path.isfile(actual_base_model_path):
            # Le chemin existe mais n'est pas un fichier (c'est peut-être un dossier)
            log_queue.put(("error", "lora_error_base_model_not_a_file", {"path": actual_base_model_path})) # Nouvelle clé pour "n'est pas un fichier"
            return

        # Si nous arrivons ici, le chemin existe et est un fichier.
        log_queue.put(("info", "lora_base_model_path_validated", {"path": actual_base_model_path})) # Log de validation
        # --- 1. Préparation des données (Dataset) ---
        # Cette partie est maintenant gérée par _actual_preparation_logic
        # On a seulement besoin de reconstruire instance_data_dir
        log_queue.put(("info", "lora_verifying_prepared_data", {})) # Nouvelle clé

        project_path = Path(training_project_dir) / lora_output_name
        # Le nom du dossier de données inclut maintenant le trigger word et le concept
        # Exemple: 10_mylora_person_concept
        # Kohya utilise N_trigger_concept, ici on va faire N_concept (trigger est dans les captions)
        # Ou on peut garder la structure N_trigger_concept si on veut
        # Pour l'instant, on crée un dossier simple pour les images et captions
        instance_data_dir = project_path / "data" / f"{concept.strip()}_images"

        if not instance_data_dir.exists() or not any(instance_data_dir.iterdir()):
            log_queue.put(("error", "lora_error_prepared_data_not_found", {"path": str(instance_data_dir)})) # Nouvelle clé
            return
        log_queue.put(("ok", "lora_prepared_data_verified", {"path": str(instance_data_dir)})) # Nouvelle clé

        if self.stop_event.is_set():
            log_queue.put(("warning", "lora_training_stopped_by_user", {}))
            return

        # --- 2. Chargement du Modèle de Base et Configuration PEFT ---
        log_queue.put(("info", "lora_model_loading_for_train", {}))
        try:
            # Déterminer le dtype de chargement basé sur mixed_precision_choice
            if mixed_precision_choice == "fp16" and self.device.type == "cuda":
                loading_dtype = torch.float16
            elif mixed_precision_choice == "bf16" and self.device.type == "cuda":
                loading_dtype = torch.bfloat16
            else: # "fp32" ou device non-cuda
                loading_dtype = torch.float32
            
            log_queue.put(("info", "lora_loading_pipeline_from_single_file", {"path": actual_base_model_path}))
            # Charger le pipeline complet à partir du fichier unique .safetensors
            pipeline = StableDiffusionXLPipeline.from_single_file(
                actual_base_model_path,
                torch_dtype=loading_dtype, # Utiliser le loading_dtype déterminé
                variant="fp16" if loading_dtype == torch.float16 else None, # Ajuster variant
                use_safetensors=True,
                # local_files_only=True # Décommentez si sûr que le chemin est toujours local
            )
            log_queue.put(("ok", "lora_pipeline_loaded_successfully", {}))

            # Extraire les composants
            log_queue.put(("info", "lora_extracting_components_from_pipeline", {}))
            tokenizer_one = pipeline.tokenizer
            tokenizer_two = pipeline.tokenizer_2
            text_encoder_one = pipeline.text_encoder
            text_encoder_two = pipeline.text_encoder_2
            vae = pipeline.vae
            unet = pipeline.unet
            noise_scheduler = pipeline.scheduler

            # Les composants sont déjà sur loading_dtype grâce à from_single_file.
            # VAE toujours en fp32 pour l'encodage.
            vae.to(self.device, dtype=torch.float32)
            text_encoder_one.to(self.device, dtype=loading_dtype)
            text_encoder_two.to(self.device, dtype=loading_dtype)
            unet.to(self.device, dtype=loading_dtype)
            # Les tokenizers n'ont pas de .to() ou de dtype

            # Mettre les modèles non entraînés en mode eval et désactiver les gradients
            vae.requires_grad_(False)
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
            unet.requires_grad_(False) # Sera réactivé pour les couches LoRA
            
            text_encoder_one.eval()
            text_encoder_two.eval()
            vae.eval()
            # unet.eval() # UNet sera mis en .train() plus tard

            log_queue.put(("ok", "lora_components_extracted_and_configured", {}))
            
            # Supprimer le pipeline complet pour libérer de la mémoire
            del pipeline
            gc.collect()
            if self.device.type == "cuda": torch.cuda.empty_cache()
            log_queue.put(("info", "lora_full_pipeline_object_deleted", {}))

            # Configuration LoRA
            log_queue.put(("info", "lora_configuring_peft_model", {}))
            
            # Ajout des adaptateurs LoRA à l'UNet
            lora_config = LoraConfig(
                r=network_dim,
                lora_alpha=network_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"], # Cibles communes pour SDXL UNet
                lora_dropout=0.05, # Exemple
                bias="none",
                init_lora_weights="gaussian",
            )
            # unet = get_peft_model(unet, lora_config) # Remplacé par add_adapter
            unet.add_adapter(lora_config, adapter_name="default") # Utiliser add_adapter
            # Cast LoRA params to fp32 if training precision is fp16
            if mixed_precision_choice == "fp16":
                cast_training_params(unet, dtype=torch.float32)

            # Calculer et afficher les paramètres entraînables manuellement
            trainable_params = 0
            all_param = 0
            for _, param in unet.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            log_queue.put(("info", "lora_trainable_parameters_unet", {"trainable": trainable_params, "total": all_param, "percentage": 100 * trainable_params / all_param if all_param > 0 else 0}))

            if train_text_encoder:
                # TODO: Ajouter la configuration LoRA pour les text encoders si désiré.
                # Cela nécessiterait de créer des LoraConfig distinctes pour chaque encodeur de texte
                # et d'appliquer get_peft_model à text_encoder_one et text_encoder_two.
                # text_encoder_one = get_peft_model(text_encoder_one, lora_config_te1)
                # text_encoder_two = get_peft_model(text_encoder_two, lora_config_te2) # Remplacé par add_adapter
                text_lora_config = LoraConfig(
                    r=network_dim, lora_alpha=network_alpha, init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], # Cibles communes pour CLIPTextModel
                )
                text_encoder_one.add_adapter(text_lora_config, adapter_name="default")
                text_encoder_two.add_adapter(text_lora_config, adapter_name="default")
                # Cast LoRA params to fp32 if training precision is fp16
                if mixed_precision_choice == "fp16":
                    cast_training_params(text_encoder_one, dtype=torch.float32)
                    cast_training_params(text_encoder_two, dtype=torch.float32)
                
                # Afficher les paramètres entraînables pour les encodeurs de texte
                for i, te in enumerate([text_encoder_one, text_encoder_two]):
                    te_trainable_params = 0
                    te_all_param = 0
                    for _, param in te.named_parameters():
                        te_all_param += param.numel()
                        if param.requires_grad:
                            te_trainable_params += param.numel()
                    log_queue.put(("info", f"lora_trainable_parameters_te{i+1}", {"trainable": te_trainable_params, "total": te_all_param, "percentage": 100 * te_trainable_params / te_all_param if te_all_param > 0 else 0}))
                log_queue.put(("info", "lora_text_encoders_lora_added", {}))
            log_queue.put(("ok", "lora_peft_model_configured", {}))

        except Exception as e_load:
            log_queue.put(("error", "lora_error_loading_base_model_components", {"path": actual_base_model_path, "error": str(e_load)})) # Clé d'erreur plus spécifique
            return

        # --- 3. Création du Dataset et DataLoader ---
        try:
            train_dataset = DreamBoothDataset(
                instance_data_root=str(instance_data_dir),
                instance_prompt_fallback=trigger_word, # Fallback si un .txt manque
                tokenizer_one=tokenizer_one,
                tokenizer_two=tokenizer_two,
                target_size=resolution, # Utiliser target_size
                center_crop=True # Exemple: Activer le center crop
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
            )
        except Exception as e_dataset:
            log_queue.put(("error", "lora_dataset_creation_failed", {"error": str(e_dataset)}))
            return

        # --- 4. Configuration de l'Optimiseur et du Scheduler ---
        try:
            params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
            if train_text_encoder:
                # params_to_optimize.extend(list(filter(lambda p: p.requires_grad, text_encoder_one.parameters())))
                # params_to_optimize.extend(list(filter(lambda p: p.requires_grad, text_encoder_two.parameters())))
                params_to_optimize.extend(list(text_encoder_one.parameters()))
                params_to_optimize.extend(list(text_encoder_two.parameters()))

            if optimizer_choice == "AdamW8bit":
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(params_to_optimize, lr=learning_rate)
                except ImportError:
                    log_queue.put(("warning", "lora_bnb_not_found_adamw", {})) # Nouvelle clé
                    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
            elif optimizer_choice == "Lion":
                try:
                    from lion_pytorch import Lion
                    optimizer = Lion(params_to_optimize, lr=learning_rate)
                except ImportError:
                    log_queue.put(("warning", "lora_lion_not_found_adamw", {})) # Nouvelle clé
                    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
            else: # AdamW par défaut
                optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

            lr_scheduler = get_scheduler(
                lr_scheduler_choice,
                optimizer=optimizer,
                num_warmup_steps=0, # Exemple, à configurer
                num_training_steps=len(train_dataloader) * epochs,
            )
        except Exception as e_optim:
            log_queue.put(("error", "lora_optimizer_setup_error", {"error": str(e_optim)}))
            return

        log_queue.put(("info", "lora_starting_training", {}))
        
        # Mixed precision scaler
        scaler = None
        actual_training_precision_dtype = None # Pour autocast
        if mixed_precision_choice == "fp16" and self.device.type == "cuda":
            scaler = torch.amp.GradScaler()
            actual_training_precision_dtype = torch.float16 # autocast vers fp16
        elif mixed_precision_choice == "bf16" and self.device.type == "cuda":
            actual_training_precision_dtype = torch.bfloat16 # autocast vers bf16, scaler n'est pas utilisé
        # Si "fp32", scaler et actual_training_precision_dtype restent None

        # Mettre les modèles en mode entraînement
        unet.train()
        if train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

        global_step = 0
        for epoch in range(epochs):
            if self.stop_event.is_set(): break
            
            epoch_loss = 0.0
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(train_dataloader):
                if self.stop_event.is_set(): break

                with torch.set_grad_enabled(True), (torch.amp.autocast(device_type=self.device.type, dtype=actual_training_precision_dtype) if scaler else contextlib.nullcontext()):
                    optimizer.zero_grad()

                    # Convertir les images en latents
                    # pixel_values sont float32 du dataloader. Pas besoin de re-caster ici avant VAE.
                    pixel_values = batch["pixel_values"].to(self.device)
                    # VAE encoding should be outside autocast if VAE is in fp32 and training in fp16/bf16
                    with torch.no_grad(): # Toujours no_grad pour l'encodage VAE
                        latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample() # Assurer que l'input du VAE est fp32
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=unet.dtype) # S'assurer que les latents sont au dtype de l'UNet

                    # Bruit aléatoire
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
                    timesteps = timesteps.long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # type: ignore

                    # Obtenir les embeddings de texte
                    # Utiliser les input_ids du batch
                    with torch.no_grad() if not train_text_encoder else contextlib.nullcontext(): # No grad si TE non entraînés
                        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[tokenizer_one, tokenizer_two], # Pass tokenizers for consistency, though not used if text_input_ids_list is provided
                            prompt_text_list=None, # On utilise les IDs pré-tokenisés
                            text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]]
                        )
                    prompt_embeds = prompt_embeds.to(dtype=unet.dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=unet.dtype)
                                        
                    # Calculer add_time_ids
                    add_time_ids_list = []
                    for i_batch in range(bsz):
                        add_time_ids_list.append(self._compute_time_ids(
                            batch["original_sizes_hw"][i_batch], 
                            batch["crop_coords_top_left_yx"][i_batch], 
                            (resolution, resolution), # target_size_hw
                            device=self.device, dtype=prompt_embeds.dtype # Utiliser le dtype des embeddings pour add_time_ids
                        ))
                    add_time_ids = torch.stack(add_time_ids_list).to(self.device)
                    # add_time_ids est déjà créé avec unet.dtype, pas besoin de re-caster ici
                    # if scaler: add_time_ids = add_time_ids.to(unet.dtype) 

                    # Préparer added_cond_kwargs pour l'UNet
                    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                       encoder_hidden_states=prompt_embeds,
                       added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    # Compute reconstruction loss
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    # Backward and optimization step (with AMP if enabled)
                    if scaler:
                        scaler.scale(loss).backward() # type: ignore
                        # ADD: Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        # ADD: Clip gradients
                        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0) # max_norm est une valeur typique
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        # ADD: Clip gradients
                        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0) # max_norm est une valeur typique
                        optimizer.step()

                    # Update learning rate scheduler
                    lr_scheduler.step()
                    # Aggregate metrics
                    epoch_loss += loss.item()
                    global_step += 1
                    progress_bar.set_postfix(loss=loss.item())
                    progress_bar.update(1)

                    log_queue.put(("info", "lora_training_epoch_progress", {
                        "current_epoch": epoch + 1, "total_epochs": epochs,
                        "current_step": step + 1, "total_steps": len(train_dataloader),
                        "loss": loss.item()
                    }))

            progress_bar.close()
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            log_queue.put(("info", "lora_epoch_end_summary", {"epoch": epoch + 1, "avg_loss": avg_epoch_loss})) # Nouvelle clé

            # Sauvegarde du checkpoint
            if save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
                log_queue.put(("info", "lora_saving_checkpoint", {"epoch": epoch + 1}))
                checkpoint_dir = project_path / "checkpoints" / f"epoch_{epoch+1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Sauvegarder les poids LoRA de l'UNet
                    unwrapped_unet = unet.module if hasattr(unet, "module") else unet
                    unwrapped_unet.save_pretrained(str(checkpoint_dir / "unet"), adapter_name="default")

                    if train_text_encoder:
                        unwrapped_text_encoder_one = text_encoder_one.module if hasattr(text_encoder_one, "module") else text_encoder_one
                        unwrapped_text_encoder_two = text_encoder_two.module if hasattr(text_encoder_two, "module") else text_encoder_two
                        unwrapped_text_encoder_one.save_pretrained(str(checkpoint_dir / "text_encoder"), adapter_name="default")
                        unwrapped_text_encoder_two.save_pretrained(str(checkpoint_dir / "text_encoder_2"), adapter_name="default")

                    log_queue.put(("ok", "lora_checkpoint_saved", {"path": str(checkpoint_dir)}))
                except Exception as e_save_cp:
                    log_queue.put(("error", "lora_checkpoint_save_error", {"epoch": epoch + 1, "error": str(e_save_cp)})) # Nouvelle clé

        # --- Fin de l'entraînement ---
        if self.stop_event.is_set():
            log_queue.put(("warning", "lora_training_stopped_by_user", {}))
            elapsed_time_total_training = time.time() - start_time_total_training
            log_queue.put(("info", "lora_total_training_time", {"time": f"{elapsed_time_total_training:.2f}"}))
            # Nettoyage même si arrêté
            del tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, unet, noise_scheduler
            del train_dataset, train_dataloader, optimizer, lr_scheduler, params_to_optimize
            if scaler: del scaler
            gc.collect()
            if self.device.type == "cuda": torch.cuda.empty_cache()
            # self.is_training = False # Déjà géré dans run_preparation_and_training_thread
            return

        # Sauvegarde finale du LoRA en utilisant le format diffusers
        # MODIFICATION: Sauvegarder en un seul fichier .safetensors
        final_lora_filename = f"{lora_output_name.strip()}.safetensors"
        final_lora_filepath = Path(self.final_loras_save_dir) / final_lora_filename
        # S'assurer que le dossier parent existe
        final_lora_filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            log_queue.put(("info", "lora_preparing_final_safetensors", {"filename": final_lora_filename}))

            final_lora_state_dict = {}
            
            # UNet LoRA layers
            unet_lora_layers_state_dict = get_peft_model_state_dict(unet) # adapter_name="default" is implicit if only one
            for k, v in unet_lora_layers_state_dict.items():
                final_lora_state_dict[f"unet.{k}"] = v # Standard diffusers prefix

            if train_text_encoder:
                # Text Encoder 1 LoRA layers
                text_encoder_lora_layers_state_dict = get_peft_model_state_dict(text_encoder_one)
                for k, v in text_encoder_lora_layers_state_dict.items():
                    final_lora_state_dict[f"text_encoder.{k}"] = v # Standard diffusers prefix
                
                # Text Encoder 2 LoRA layers
                text_encoder_2_lora_layers_state_dict = get_peft_model_state_dict(text_encoder_two)
                for k, v in text_encoder_2_lora_layers_state_dict.items():
                    final_lora_state_dict[f"text_encoder_2.{k}"] = v # Standard diffusers prefix
            
            safetensors.torch.save_file(final_lora_state_dict, str(final_lora_filepath))
            
            elapsed_time_total_training = time.time() - start_time_total_training
            log_queue.put(("ok", "lora_training_complete_single_file", {"filepath": str(final_lora_filepath), "time": f"{elapsed_time_total_training:.2f}"})) # Clé de log modifiée
            # Le message de temps total est maintenant inclus dans "lora_training_complete"
            # log_queue.put(("info", "lora_total_training_time", {"time": f"{elapsed_time_total_training:.2f}"}))
        except Exception as e_final_save:
            log_queue.put(("error", "lora_final_save_error", {"error": str(e_final_save), "path": str(final_lora_filepath)})) # Path mis à jour

        # Nettoyage
        del tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, unet, noise_scheduler
        del train_dataset, train_dataloader, optimizer, lr_scheduler, params_to_optimize
        if scaler: del scaler
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        # self.is_training = False # Déjà géré dans run_preparation_and_training_thread

    # La fonction _actual_preparation_and_training est maintenant divisée en:
    # _actual_preparation_logic (ci-dessous)
    # _actual_training_logic (ci-dessus, adaptée de l'originale)

    def _actual_preparation_logic(
        self, log_queue,
        input_images_dir, trigger_word, concept, caption_task_display_name, auto_captioning_enabled,
        lora_output_name, training_project_dir, resolution
    ):
        self.is_preparing = True # Mettre le drapeau de préparation
        try:
            if not all([input_images_dir, trigger_word, concept, lora_output_name, training_project_dir]):
                log_queue.put(("error", "lora_error_missing_input_prep", {}))
                self.is_preparing = False
                return False

            log_queue.put(("info", "lora_preparing_data", {}))
            project_path = Path(training_project_dir) / lora_output_name
            instance_data_dir = project_path / "data" / f"{concept.strip()}_images"
            instance_data_dir.mkdir(parents=True, exist_ok=True)
            log_queue.put(("info", "lora_data_prepared_in", {"path": str(instance_data_dir)}))

            target_extensions_for_processing = ['.jpeg', '.jpg', '.png']
            source_image_files = [f for f in os.listdir(input_images_dir) if os.path.splitext(f)[1].lower() in target_extensions_for_processing]
            if not source_image_files:
                log_queue.put(("error", "lora_error_no_compatible_images_in_dir", {"path": input_images_dir, "extensions": ", ".join(target_extensions_for_processing)}))
                self.is_preparing = False
                return False

            task_display_to_key_map = {self._translate(f"florence2_task_{tk.strip('<>').lower().replace(' ', '_')}"): tk for tk in FLORENCE2_TASKS}
            selected_caption_task_key = task_display_to_key_map.get(caption_task_display_name, DEFAULT_FLORENCE2_TASK)
            target_size_for_crop = resolution

            for img_filename in tqdm(source_image_files, desc=self._translate("lora_captioning_images")):
                if self.stop_event.is_set(): break
                source_img_path = os.path.join(input_images_dir, img_filename)
                dest_img_path = instance_data_dir / img_filename
                base_img_filename_no_ext = os.path.splitext(img_filename)[0]
                dest_txt_path = instance_data_dir / (base_img_filename_no_ext + ".txt")
                try:
                    pil_image = Image.open(source_img_path).convert("RGB")
                    width, height = pil_image.size
                    if width < target_size_for_crop or height < target_size_for_crop:
                        log_queue.put(("warning", "lora_image_too_small_skipped", {"image": img_filename, "original_size": f"{width}x{height}", "min_size": target_size_for_crop}))
                        continue
                    if width != target_size_for_crop or height != target_size_for_crop:
                        left, top = (width - target_size_for_crop) / 2, (height - target_size_for_crop) / 2
                        pil_image_cropped = pil_image.crop((left, top, left + target_size_for_crop, top + target_size_for_crop))
                    else:
                        pil_image_cropped = pil_image
                    pil_image_cropped.save(dest_img_path)
                    if auto_captioning_enabled:
                        caption = generate_prompt_from_image(pil_image_cropped, self.module_translations, task=selected_caption_task_key, unload_after=False)
                        error_marker = f"[{translate('erreur', self.global_translations).upper()}]"
                        if caption.startswith(error_marker): raise ValueError(caption.replace(error_marker, "").strip())
                        final_caption = f"{trigger_word.strip()}, {caption}"
                        with open(dest_txt_path, "w", encoding="utf-8") as f_cap: f_cap.write(final_caption)
                        log_queue.put(("info", "lora_caption_saved_auto", {"image": img_filename, "txt_file": str(dest_txt_path)}))
                    else: # Copie manuelle des .txt
                        source_txt_path = Path(input_images_dir) / (base_img_filename_no_ext + ".txt")
                        if source_txt_path.exists(): shutil.copy2(source_txt_path, dest_txt_path)
                        else: log_queue.put(("info", "lora_caption_skipped_manual_no_source_txt", {"image": img_filename, "trigger": trigger_word}))
                except Exception as e_cap:
                    log_queue.put(("error", "lora_error_caption_generation", {"image": img_filename, "error": str(e_cap)}))
            if auto_captioning_enabled: unload_image_prompter_model(self.module_translations); log_queue.put(("info", "lora_caption_model_unloaded", {}))
            if self.stop_event.is_set(): log_queue.put(("warning", "lora_preparation_stopped_by_user", {})); self.is_preparing = False; return False # Nouvelle clé
            
            # --- Début: Renommage séquentiel des images et des fichiers .txt ---
            log_queue.put(("info", "lora_renaming_files_sequentially", {})) # Nouvelle clé de log
            image_files_in_instance_dir = sorted([
                f for f in instance_data_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in target_extensions_for_processing
            ])

            renamed_count = 0
            for i, old_image_path in enumerate(image_files_in_instance_dir):
                if self.stop_event.is_set(): break
                
                original_extension = old_image_path.suffix # Ex: .jpg
                # Nouveau nom de base pour l'image (ex: image_001)
                # Utiliser un nom de base cohérent, par exemple le nom du concept ou un nom générique
                # Ici, on utilise le nom du concept pour plus de clarté si plusieurs datasets sont générés
                new_base_name = f"{concept.strip()}_{i+1:04d}" # concept_0001, concept_0002, etc.
                
                new_image_path = instance_data_dir / f"{new_base_name}{original_extension}"
                old_txt_path = instance_data_dir / f"{old_image_path.stem}.txt"
                new_txt_path = instance_data_dir / f"{new_base_name}.txt"

                # Renommer l'image
                if old_image_path != new_image_path: # Éviter de renommer si le nom est déjà correct (peu probable ici)
                    old_image_path.rename(new_image_path)
                    log_queue.put(("info", "lora_file_renamed", {"old": str(old_image_path.name), "new": str(new_image_path.name)})) # Nouvelle clé
                    renamed_count +=1
                # Renommer le fichier .txt associé s'il existe
                if old_txt_path.exists() and old_txt_path != new_txt_path:
                    old_txt_path.rename(new_txt_path)
                    log_queue.put(("info", "lora_file_renamed", {"old": str(old_txt_path.name), "new": str(new_txt_path.name)})) # Nouvelle clé
            log_queue.put(("ok", "lora_renaming_complete", {"count": renamed_count})) # Nouvelle clé
            # --- Fin: Renommage séquentiel ---

            log_queue.put(("ok", "lora_data_preparation_complete", {})) # Nouvelle clé pour succès de préparation
            self.is_preparing = False
            return True # Indiquer le succès
        except Exception as e_prep:
            log_queue.put(("error", "lora_data_preparation_failed_unexpectedly", {"error": str(e_prep)})) # Nouvelle clé
            traceback.print_exc()
            self.is_preparing = False
            return False # Indiquer l'échec
