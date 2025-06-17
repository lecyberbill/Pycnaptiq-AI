# model_manager.py
import torch
import sys # Ajout de sys pour modifier le path
import time
import os
import gc
import traceback
from pathlib import Path
import gradio as gr
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler, # Exemple, ajoute les schedulers nécessaires
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    SanaSprintPipeline,
    CogView4Pipeline,
    FluxPipeline, 
    FluxImg2ImgPipeline, # <-- AJOUT POUR FLUX Img2Img
    CogView3PlusPipeline,
)  
from compel import Compel, ReturnedEmbeddingsType

# Importer les fonctions utilitaires nécessaires (ajuster si besoin)
from .utils import txt_color, translate, lister_fichiers, str_to_bool # Importer lister_fichiers et str_to_bool

# Définir les devices ici ou les passer via config/init
cpu = torch.device("cpu")
gpu = torch.device( # Garder cette initialisation pour gpu
    f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
)

# Ajouter le répertoire 'modules' au sys.path pour permettre l'importation de bibliothèques locales
project_root_dir = Path(__file__).resolve().parent.parent # Remonte au dossier principal du projet (cyberbill_image_generator)
modules_dir_abs = project_root_dir / "modules"
if str(modules_dir_abs) not in sys.path:
    sys.path.insert(0, str(modules_dir_abs))

SANA_MODEL_TYPE_KEY = "sana_sprint" 
COGVIEW4_MODEL_ID = "THUDM/CogView4-6B"
COGVIEW4_MODEL_TYPE_KEY = "cogview4"
COGVIEW3PLUS_MODEL_ID = "THUDM/CogView3-Plus-3B"
COGVIEW3PLUS_MODEL_TYPE_KEY = "cogview3plus"
FLUX_SCHNELL_MODEL_ID = "black-forest-labs/FLUX.1-schnell" 
FLUX_SCHNELL_MODEL_TYPE_KEY = "flux_schnell" 
FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY = "flux_schnell_img2img" # <-- NOUVEAU pour FLUX Img2Img

class ModelManager:
    def __init__(self, config, translations, device, torch_dtype, vram_total_gb):
        self.config = config
        self.translations = translations
        self.device = device
        self.torch_dtype = torch_dtype
        self.vram_total_gb = vram_total_gb

        self.current_pipe = None
        self.current_compel = None
        self.current_model_name = None
        self.current_vae_name = None 
        self.current_model_type = None  
        self.current_sampler_key = None 
        self.loaded_loras = {}  

        self.models_dir = self._get_absolute_path(config.get("MODELS_DIR", "models/checkpoints"))
        self.vae_dir = self._get_absolute_path(config.get("VAE_DIR", "models/vae"))
        self.loras_dir = self._get_absolute_path(config.get("LORAS_DIR", "models/loras"))
        self.inpaint_models_dir = self._get_absolute_path(config.get("INPAINT_MODELS_DIR", "models/inpainting"))

    def _get_absolute_path(self, path_from_config):
        root_dir = Path(__file__).parent.parent 
        if not os.path.isabs(path_from_config):
            return os.path.abspath(os.path.join(root_dir, path_from_config))
        return path_from_config

    def list_models(self, model_type="standard", gradio_mode=False):
        if model_type == "inpainting":
            dir_path = self.inpaint_models_dir
        else: 
            dir_path = self.models_dir
        model_files_raw = lister_fichiers(dir_path, self.translations, ext=".safetensors", gradio_mode=gradio_mode)
        placeholder_model = "your_default_modele.safetensors"
        filtered_model_files = [f for f in model_files_raw if f != placeholder_model]
        no_model_msg = translate("aucun_modele_trouve", self.translations)
        not_found_msg = translate("repertoire_not_found", self.translations) 
        if not filtered_model_files:
            if model_files_raw and model_files_raw[0] != no_model_msg and model_files_raw[0] != not_found_msg:
                return [no_model_msg]
            else:
                return model_files_raw
        else:
            return filtered_model_files

    def list_vaes(self, gradio_mode=False):
        return ["Auto"] + lister_fichiers(self.vae_dir, self.translations, ext=".safetensors", gradio_mode=gradio_mode)

    def list_loras(self, gradio_mode=False):
        if not os.path.isdir(self.loras_dir):
            msg = translate("repertoire_not_found", self.translations)
            if gradio_mode: gr.Warning(msg + f": {self.loras_dir}", 3.0)
            return [msg]
        lora_items = []
        for item_name in os.listdir(self.loras_dir):
            item_path = os.path.join(self.loras_dir, item_name)
            if os.path.isdir(item_path) or (os.path.isfile(item_path) and item_name.lower().endswith(".safetensors")):
                lora_items.append(item_name)
        if not lora_items:
            msg = translate("aucun_lora_disponible", self.translations) 
            if gradio_mode: gr.Info(msg, 3.0)
            return [msg]
        return sorted(lora_items) 

    def get_cuda_free_memory_gb(self):
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return 0
        try:
            memory_stats = torch.cuda.memory_stats(self.device)
            bytes_active = memory_stats["active_bytes.all.current"]
            bytes_reserved = memory_stats["reserved_bytes.all.current"]
            bytes_free_cuda, _ = torch.cuda.mem_get_info(self.device)
            bytes_inactive_reserved = bytes_reserved - bytes_active
            bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
            return bytes_total_available / (1024**3)
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"Erreur récupération mémoire CUDA: {e}")
            return 0

    def unload_model(self, gradio_mode=False):
        if self.current_pipe is None and self.current_compel is None:
            msg = translate("aucun_modele_a_decharger", self.translations)
            print(txt_color("[INFO]", "info"), msg)
            if gradio_mode: gr.Info(msg, duration=2.0)
            return True, msg

        print(txt_color("[INFO]", "info"), translate("dechargement_modele_en_cours", self.translations))
        if gradio_mode: gr.Info(translate("dechargement_modele_en_cours", self.translations), duration=2.0)

        try:
            if self.current_pipe is not None:
                if not isinstance(self.current_pipe, dict): 
                    # self.current_pipe.to(cpu) # <-- SUPPRIMER CETTE LIGNE
                    # Déplacer le pipeline vers le CPU avant de supprimer les attributs peut causer
                    # des erreurs avec les modèles offloadés qui utilisent des tenseurs "meta".
                    attrs_to_delete = ['vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'scheduler', 'feature_extractor', 'safety_checker']
                    for attr in attrs_to_delete:
                        if hasattr(self.current_pipe, attr):
                            try: 
                                component_to_delete = getattr(self.current_pipe, attr)
                                delattr(self.current_pipe, attr)
                                del component_to_delete 
                            except Exception as e_del:
                                print(txt_color("[WARN]", "warning"), f"Impossible de supprimer l'attribut {attr}: {e_del}")
                del self.current_pipe
                self.current_pipe = None

            if self.current_compel is not None:
                if hasattr(self.current_compel, 'tokenizer'): del self.current_compel.tokenizer
                if hasattr(self.current_compel, 'text_encoder'): del self.current_compel.text_encoder
                del self.current_compel
                self.current_compel = None

            self.current_model_name = None
            self.current_vae_name = None
            self.current_model_type = None
            self.current_sampler_key = None 
            self.loaded_loras.clear()

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()

            print(txt_color("[OK]", "ok"), translate("modele_precedent_decharge", self.translations))
            if gradio_mode: gr.Info(translate("modele_precedent_decharge", self.translations), duration=3.0)
            return True, translate("modele_precedent_decharge", self.translations)
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_dechargement_modele', self.translations)}: {e}")
            traceback.print_exc()
            self.current_pipe = None
            self.current_compel = None
            self.current_model_name = None
            self.current_vae_name = None
            self.current_model_type = None
            self.current_sampler_key = None
            self.loaded_loras.clear()
            gc.collect()
            if self.device.type == "cuda": torch.cuda.empty_cache()
            error_message = f"{translate('erreur_dechargement_modele', self.translations)}: {e}"
            if gradio_mode: gr.Error(error_message)
            return False, error_message

    def load_model(self, model_name, vae_name="Auto", model_type="standard", gradio_mode=False, custom_pipeline_id=None): 
        if not model_name or model_name == translate("aucun_modele", self.translations) or model_name == translate("aucun_modele_trouve", self.translations):
            msg = translate("aucun_modele_selectionne", self.translations)
            print(txt_color("[ERREUR]", "erreur"), msg)
            if gradio_mode: gr.Warning(msg, duration=4.0)
            return False, msg

        if self.current_pipe is not None and \
           (self.current_model_name != model_name or self.current_model_type != model_type):
            print(txt_color("[INFO]", "info"), translate("dechargement_modele_precedent_avant_chargement", self.translations)) 
            unload_success, unload_msg = self.unload_model(gradio_mode=gradio_mode)
            if not unload_success:
                return False, unload_msg

        pipeline_loader = None
        pipeline_class = None  
        model_dir = self.models_dir 
        is_from_single_file = True 

        if model_type == SANA_MODEL_TYPE_KEY: 
            pipeline_loader = SanaSprintPipeline.from_pretrained
            is_from_single_file = False 
            chemin_modele = model_name 
        elif model_type == COGVIEW4_MODEL_TYPE_KEY:
            pipeline_loader = CogView4Pipeline.from_pretrained
            is_from_single_file = False
            specific_torch_dtype = torch.bfloat16 
            chemin_modele = model_name 
        elif model_type == COGVIEW3PLUS_MODEL_TYPE_KEY:
            pipeline_loader = CogView3PlusPipeline.from_pretrained
            is_from_single_file = False
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                specific_torch_dtype = torch.bfloat16
                print(txt_color("[INFO]", "info"), "Utilisation de torch.bfloat16 pour CogView3-Plus (supporté).")
            else:
                specific_torch_dtype = torch.float16
                print(txt_color("[INFO]", "info"), "Utilisation de torch.float16 pour CogView3-Plus (bfloat16 non supporté ou CUDA non disponible).")
            chemin_modele = model_name 
        elif model_type == FLUX_SCHNELL_MODEL_TYPE_KEY: 
            pipeline_loader = FluxPipeline.from_pretrained
            is_from_single_file = False
            specific_torch_dtype = torch.bfloat16 
            chemin_modele = model_name 
        elif model_type == FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY: # <-- NOUVELLE BRANCHE
            pipeline_loader = FluxImg2ImgPipeline.from_pretrained # Utiliser FluxImg2ImgPipeline
            is_from_single_file = False
            specific_torch_dtype = torch.bfloat16 
            chemin_modele = model_name # Devrait être FLUX_SCHNELL_MODEL_ID
        elif model_type == "inpainting":
            pipeline_class = StableDiffusionXLInpaintPipeline 
            model_dir = self.inpaint_models_dir 
        elif model_type == "img2img":
            pipeline_class = StableDiffusionXLImg2ImgPipeline
        else: # standard
            pipeline_class = StableDiffusionXLPipeline
            model_dir = self.models_dir

        if is_from_single_file: 
            chemin_modele = os.path.join(model_dir, model_name)
            if not os.path.exists(chemin_modele):
                msg = f"{translate('modele_non_trouve', self.translations)}: {chemin_modele}"
                print(txt_color("[ERREUR]", "erreur"), msg)
                if gradio_mode: gr.Warning(msg, duration=4.0)
                return False, msg
        chemin_vae = os.path.join(self.vae_dir, vae_name) if vae_name and vae_name != "Auto" else None

        print(txt_color("[INFO]", "info"), f"{translate('chargement_modele', self.translations)}: {model_name} ({model_type})")
        if gradio_mode: gr.Info(f"{translate('chargement_modele', self.translations)}: {model_name} ({model_type})", duration=3.0)

        try:
            pipe = None
            try:
                pipeline_kwargs = {
                    "torch_dtype": specific_torch_dtype if 'specific_torch_dtype' in locals() else self.torch_dtype,
                    "use_safetensors": True,
                    "safety_checker": None,
                }
                if custom_pipeline_id: 
                    pipeline_kwargs["custom_pipeline"] = custom_pipeline_id
                    print(txt_color("[INFO]", "info"), f"Tentative de chargement avec custom_pipeline: {custom_pipeline_id}")

                if is_from_single_file:
                    pipe = pipeline_class.from_single_file(chemin_modele, **pipeline_kwargs)
                else: 
                    if model_type == SANA_MODEL_TYPE_KEY:
                        dtype_to_use = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else self.torch_dtype
                        print(f"[INFO] Chargement {model_type} ({chemin_modele}) avec dtype: {dtype_to_use}")
                        pipe = pipeline_loader(chemin_modele, torch_dtype=dtype_to_use)
                    elif model_type == COGVIEW4_MODEL_TYPE_KEY:
                        print(f"[INFO] Chargement {model_type} ({chemin_modele}) avec dtype: {specific_torch_dtype}")
                        pipe = pipeline_loader(chemin_modele, torch_dtype=specific_torch_dtype)
                    elif model_type == COGVIEW3PLUS_MODEL_TYPE_KEY:
                        print(f"[INFO] Chargement {model_type} ({chemin_modele}) avec dtype: {specific_torch_dtype}")
                        pipe = pipeline_loader(chemin_modele, torch_dtype=specific_torch_dtype)
                    elif model_type == FLUX_SCHNELL_MODEL_TYPE_KEY: 
                        print(f"[INFO] Chargement {model_type} ({chemin_modele}) avec dtype: {specific_torch_dtype}")
                        pipe = pipeline_loader(chemin_modele, torch_dtype=specific_torch_dtype)
                    elif model_type == FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY: # <-- NOUVELLE BRANCHE
                        print(f"[INFO] Chargement {model_type} ({chemin_modele}) avec dtype: {specific_torch_dtype}")
                        pipe = pipeline_loader(chemin_modele, torch_dtype=specific_torch_dtype)
            except Exception as e_pipe_load:
                raise RuntimeError(f"{translate('erreur_chargement_pipeline', self.translations)}: {e_pipe_load}")
            
            vae_message = ""
            loaded_vae_name = "Auto"
            if model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY] \
               and chemin_vae and os.path.exists(chemin_vae):
                print(txt_color("[INFO]", "info"), f"{translate('chargement_vae', self.translations)}: {vae_name}")
                try:
                    vae = AutoencoderKL.from_single_file(chemin_vae, torch_dtype=self.torch_dtype)
                    if hasattr(pipe, 'vae') and pipe.vae is not None: del pipe.vae
                    pipe.vae = vae.to(self.device) 
                    vae_message = f" + VAE: {vae_name}"
                    loaded_vae_name = vae_name
                    print(txt_color("[OK]", "ok"), f"{translate('vae_charge', self.translations)}: {vae_name}")
                except Exception as e_vae:
                    vae_message = f" + VAE: {translate('erreur_chargement_vae_court', self.translations)}"
                    loaded_vae_name = f"Auto ({translate('erreur', self.translations)})"
                    print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_chargement_vae', self.translations)}: {vae_name} - {e_vae}")
                    if gradio_mode: gr.Warning(f"{translate('erreur_chargement_vae', self.translations)}: {vae_name} - {e_vae}", duration=4.0) 
            elif model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY] and chemin_vae: 
                 vae_message = f" + VAE: {translate('vae_non_trouve_court', self.translations)}"
                 loaded_vae_name = f"Auto ({translate('non_trouve', self.translations)})"
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('vae_non_trouve', self.translations)}: {chemin_vae}")
                 if gradio_mode: gr.Warning(f"{translate('vae_non_trouve', self.translations)}: {chemin_vae}", duration=4.0) 
            elif model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]: 
                 vae_message = f" + VAE: {translate('auto_label', self.translations)}" 
                 print(txt_color("[INFO]", "info"), translate("utilisation_vae_integre", self.translations))
            elif model_type == COGVIEW4_MODEL_TYPE_KEY:
                vae_message = f" (VAE intégré à CogView4)"
                print(txt_color("[INFO]", "info"), "CogView4 utilise son VAE interne.")
            elif model_type == COGVIEW3PLUS_MODEL_TYPE_KEY:
                vae_message = f" (VAE intégré à CogView3-Plus)"
                print(txt_color("[INFO]", "info"), "CogView3-Plus utilise son VAE interne.")
            elif model_type == FLUX_SCHNELL_MODEL_TYPE_KEY: 
                vae_message = f" (VAE intégré à FLUX)"
                print(txt_color("[INFO]", "info"), "FLUX.1-schnell utilise son VAE interne.")
            elif model_type == FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY: # <-- NOUVELLE BRANCHE
                vae_message = f" (VAE intégré à FLUX Img2Img)"
                print(txt_color("[INFO]", "info"), "FLUX.1-schnell (Img2Img) utilise son VAE interne.")

            if model_type in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY] and pipe is not None and not isinstance(pipe, dict): 
                print(txt_color("[INFO]", "info"), "Application des configurations spécifiques à CogView4 (CPU offload, slicing, tiling)...")
                if gradio_mode: gr.Info(f"Application des configurations spécifiques à {model_type}...", duration=2.0) 
                try:
                    pipe.enable_model_cpu_offload() 
                    if hasattr(pipe, 'vae') and pipe.vae is not None:
                        pipe.vae.enable_slicing()
                        pipe.vae.enable_tiling()
                        print(txt_color("[OK]", "ok"), "Slicing et Tiling VAE activés pour CogView4.")
                    else:
                        print(txt_color("[AVERTISSEMENT]", "warning"), f"Le pipe {model_type} n'a pas d'attribut 'vae' ou VAE est None. Slicing/Tiling VAE non appliqué.")
                except Exception as e_cog_config:
                    print(txt_color("[ERREUR]", "erreur"), f"Erreur lors de la configuration spécifique de {model_type}: {e_cog_config}")
                    if gradio_mode: gr.Warning(f"Erreur configuration {model_type}: {e_cog_config}", duration=4.0) 
            elif model_type in [FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY] and pipe is not None: # <-- MODIFIÉ POUR INCLURE IMG2IMG
                print(txt_color("[INFO]", "info"), "Application des configurations spécifiques à FLUX.1-schnell...")
                if gradio_mode: gr.Info(f"Application des configurations spécifiques à {model_type}...", duration=2.0)
                try:
                    pipe.enable_sequential_cpu_offload() 
                    if hasattr(pipe, 'vae') and pipe.vae is not None:
                        pipe.vae.enable_slicing()
                        pipe.vae.enable_tiling()
                        print(txt_color("[OK]", "ok"), "Slicing et Tiling VAE activés pour FLUX.")
                except Exception as e_flux_config:
                    print(txt_color("[ERREUR]", "erreur"), f"Erreur lors de la configuration spécifique de {model_type}: {e_flux_config}")
                    if gradio_mode: gr.Warning(f"Erreur configuration {model_type}: {e_flux_config}", duration=4.0)
            elif pipe is not None and not isinstance(pipe, dict): 
                force_cpu_offload_config = str_to_bool(str(self.config.get("FORCE_CPU_OFFLOAD", "False")))
                automatic_offload_condition = self.device.type == "cuda" and self.vram_total_gb < 8
                should_enable_offload = (force_cpu_offload_config or automatic_offload_condition) and model_type not in [SANA_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY] 

                if should_enable_offload:
                    if force_cpu_offload_config:
                        reason_message = translate("activation_cpu_offload_forced_config", self.translations)
                    else: 
                        reason_message = translate("activation_cpu_offload", self.translations).format(vram=self.vram_total_gb)
                    print(txt_color("[INFO]", "info"), reason_message)
                    if gradio_mode: gr.Info(reason_message, duration=3.0) 
                    pipe.enable_model_cpu_offload()
                else:
                    print(txt_color("[INFO]", "info"), f"{translate('deplacement_modele_device', self.translations)} {self.device}...")
                    pipe.to(self.device)

            if pipe is not None and not isinstance(pipe, dict): 
                try:
                    if model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]: 
                        pipe.enable_vae_slicing()
                        pipe.enable_vae_tiling()

                    if self.device.type == "cuda" and self.vram_total_gb < 10 and model_type not in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]:
                        if hasattr(pipe, 'enable_attention_slicing'): 
                            pipe.enable_attention_slicing()
                            print(txt_color("[INFO]", "info"), translate("optimisation_memoire_activee", self.translations))
                        else:
                            print(txt_color("[INFO]", "info"), f"enable_attention_slicing non disponible pour {model_type}.")
 
                    if self.device.type == "cuda" and model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]:
                        try: 
                            pipe.enable_xformers_memory_efficient_attention()
                            print(txt_color("[INFO]", "info"), "XFormers activé.")
                        except ImportError:
                            print(txt_color("[INFO]", "info"), "XFormers non disponible, ignoré.")
                        except Exception as e_xformers:
                            print(txt_color("[AVERTISSEMENT]", "warning"), f"Erreur activation XFormers: {e_xformers}")
                    elif model_type in [SANA_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]: 
                        print(txt_color("[INFO]", "info"), f"Optimisations XFormers/attention slicing non applicables ou désactivées pour {model_type}.") 
                except Exception as e_optim:
                    print(txt_color("[AVERTISSEMENT]", "warning"), f"Erreur application optimisations: {e_optim}")

            if model_type in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]: # <-- AJOUT FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY
                self.current_compel = None 
                print(txt_color("[INFO]", "info"), f"Compel n'est pas utilisé pour {model_type}.")
            elif pipe is not None: 
                has_tokenizer_2 = hasattr(pipe, 'tokenizer_2') and pipe.tokenizer_2 is not None
                has_encoder_2 = hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None
                if model_type == SANA_MODEL_TYPE_KEY:
                    compel_returned_embeddings_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
                    compel_requires_pooled = False
                else: 
                    compel_returned_embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                    compel_requires_pooled = [False, True] if has_encoder_2 else False
                compel_instance = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2] if has_tokenizer_2 else pipe.tokenizer,
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2] if has_encoder_2 else pipe.text_encoder,
                    returned_embeddings_type=compel_returned_embeddings_type,
                    requires_pooled=compel_requires_pooled,
                    device=self.device
                )
                self.current_compel = compel_instance
                print(txt_color("[INFO]", "info"), f"Compel initialisé pour {model_type}.")
            else:
                self.current_compel = None 
                if model_type not in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]:
                    print(txt_color("[AVERTISSEMENT]", "warning"), f"Compel non initialisé pour {model_type} car le pipeline est None ou non géré.")

            self.current_pipe = pipe
            self.current_model_name = model_name 
            self.current_vae_name = loaded_vae_name
            self.current_model_type = model_type
            self.loaded_loras.clear() 
            self.current_sampler_key = None 

            final_message = f"{translate('modele_charge', self.translations)}: {model_name}{vae_message}"
            print(txt_color("[OK]", "ok"), final_message)
            if gradio_mode: gr.Info(final_message, duration=3.0) 
            return True, final_message
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_generale_chargement_modele', self.translations)}: {e}") 
            traceback.print_exc()
            self.unload_model() 
            error_msg = f"{translate('erreur_chargement_modele', self.translations)}: {e}"
            if gradio_mode: gr.Error(error_msg) 
            return False, error_msg

    def get_current_pipe(self):
        return self.current_pipe

    def get_current_compel(self):
        return self.current_compel

    def load_lora(self, lora_item_name, scale): 
        if self.current_pipe is None:
            msg = translate("erreur_pas_modele_pour_lora", self.translations) 
            print(txt_color("[ERREUR]", "erreur"), msg)
            return False, msg
        lora_full_path = os.path.join(self.loras_dir, lora_item_name)
        if not os.path.exists(lora_full_path): 
            msg = f"{translate('erreur_lora_introuvable', self.translations)}: {lora_full_path}" 
            print(txt_color("[ERREUR]", "erreur"), msg)
            return False, msg
        adapter_base_name = os.path.splitext(lora_item_name)[0] if lora_item_name.lower().endswith(".safetensors") else lora_item_name
        adapter_name = adapter_base_name.replace(".", "_").replace(" ", "_")
        try:
            print(txt_color("[INFO]", "info"), f"{translate('lora_charge_depuis_chemin', self.translations)} {lora_full_path}") 
            self.current_pipe.load_lora_weights(lora_full_path, adapter_name=adapter_name)
            self.loaded_loras[adapter_name] = scale 
            msg = f"{translate('lora_charge', self.translations)}: {adapter_name} (Scale: {scale})"
            print(txt_color("[OK]", "ok"), msg)
            return True, msg
        except Exception as e:
            msg = f"{translate('erreur_lora_chargement', self.translations)}: {e}"
            print(txt_color("[ERREUR]", "erreur"), msg)
            traceback.print_exc()
            try:
                if hasattr(self.current_pipe, 'delete_adapter'):
                    self.current_pipe.delete_adapter(adapter_name)
                    print(txt_color("[INFO]", "info"), f"Tentative de suppression de l'adaptateur '{adapter_name}' après échec de chargement.")
            except Exception as e_unload_fail:
                print(txt_color("[WARN]", "warning"), f"Échec du nettoyage après erreur chargement LoRA {adapter_name}: {e_unload_fail}")
            return False, msg

    def unload_lora(self, adapter_name):
        if self.current_pipe is None: return False, translate("erreur_pas_modele", self.translations)
        try:
            if hasattr(self.current_pipe, "unload_lora_weights"):
                print(txt_color("[INFO]", "info"), f"Déchargement des poids LoRA: {adapter_name}")
                self.current_pipe.unload_lora_weights()
            else:
                print(txt_color("[WARN]", "warning"), f"La méthode 'unload_lora_weights' n'existe pas pour ce pipeline ({self.current_pipe.__class__.__name__}). Désactivation via set_adapters tentée.")
                current_active_adapters = list(self.loaded_loras.keys())
                current_active_weights = list(self.loaded_loras.values())
                if adapter_name in current_active_adapters:
                    idx = current_active_adapters.index(adapter_name)
                    current_active_adapters.pop(idx)
                    current_active_weights.pop(idx)
                    self.current_pipe.set_adapters(current_active_adapters, adapter_weights=current_active_weights)
            if adapter_name in self.loaded_loras:
                del self.loaded_loras[adapter_name] 
            msg = f"{translate('lora_decharge_nom', self.translations)}: {adapter_name}"
            print(txt_color("[INFO]", "info"), msg)
            return True, msg
        except Exception as e:
            msg = f"{translate('erreur_lora_dechargement', self.translations)}: {e}" 
            print(txt_color("[ERREUR]", "erreur"), msg)
            traceback.print_exc() 
            if adapter_name in self.loaded_loras:
                del self.loaded_loras[adapter_name]
            return False, msg

    def apply_loras(self, lora_ui_config, gradio_mode=False):
        if self.current_pipe is None:
            msg = translate("erreur_pas_modele_pour_lora", self.translations)
            if gradio_mode: gr.Warning(msg, duration=4.0)
            return msg
        messages = [] 
        lora_checks = lora_ui_config.get('lora_checks', [])
        lora_dropdowns = lora_ui_config.get('lora_dropdowns', [])
        lora_scales = lora_ui_config.get('lora_scales', [])
        requested_loras = {} 
        for i, is_checked in enumerate(lora_checks):
            if is_checked and i < len(lora_dropdowns) and i < len(lora_scales):
                lora_filename = lora_dropdowns[i]
                scale = lora_scales[i]
                if lora_filename and lora_filename != translate("aucun_lora_disponible", self.translations):
                    adapter_name = os.path.splitext(lora_filename)[0].replace(".", "_").replace(" ", "_")
                    requested_loras[adapter_name] = (lora_filename, scale)
        loras_to_unload = [name for name in self.loaded_loras if name not in requested_loras]
        for name in loras_to_unload:
            self.unload_lora(name)
        needs_set_adapters_call = False 
        for adapter_name, (lora_filename, scale) in requested_loras.items():
            if adapter_name not in self.loaded_loras:
                success, msg = self.load_lora(lora_filename, scale)
                messages.append(msg)
                if success:
                    needs_set_adapters_call = True 
            elif self.loaded_loras[adapter_name] != scale:
                self.loaded_loras[adapter_name] = scale 
                messages.append(f"{translate('lora_poids_maj', self.translations)}: {adapter_name} -> {scale}") 
                needs_set_adapters_call = True 
                print(txt_color("[INFO]", "info"), f"Mise à jour poids LoRA {adapter_name} -> {scale}")
        if needs_set_adapters_call or loras_to_unload:
            try:
                active_adapters = list(self.loaded_loras.keys())
                active_weights = [self.loaded_loras[name] for name in active_adapters]
                if active_adapters:
                     print(txt_color("[INFO]", "info"), f"Application des adaptateurs actifs: {active_adapters} avec poids {active_weights}")
                     try:
                         self.current_pipe.set_adapters(active_adapters, adapter_weights=active_weights)
                     except KeyError as e_key: 
                         msg = f"Erreur (KeyError) lors de l'application des poids LoRA: {e_key}. L'état interne du pipeline pourrait être incohérent."
                         print(txt_color("[ERREUR]", "erreur"), msg)
                         messages.append(msg)
                     except Exception as e_set: 
                         msg = f"Erreur lors de l'application des poids LoRA: {e_set}"
                         print(txt_color("[ERREUR]", "erreur"), msg)
                         messages.append(msg)
                else:
                     print(txt_color("[INFO]", "info"), "Aucun LoRA actif à appliquer, désactivation.")
                     self.current_pipe.set_adapters([], adapter_weights=[])
            except Exception as e_final_set:
                 msg = f"Erreur finale lors de set_adapters: {e_final_set}"
                 print(txt_color("[ERREUR]", "erreur"), msg)
                 messages.append(msg)
        final_message = "\n".join(filter(None, messages))
        return final_message if final_message else translate("loras_geres_succes", self.translations)
