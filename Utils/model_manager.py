# model_manager.py
import torch
import sys # Ajout de sys pour modifier le path
import time
import os
import gc
import traceback
from contextlib import contextmanager
from pathlib import Path
import safetensors
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
    FluxTransformer2DModel,
    FluxPipeline,
    FluxImg2ImgPipeline, # <-- AJOUT POUR FLUX Img2Img
    CogView3PlusPipeline,
    StableDiffusionInstructPix2PixPipeline,
)
# --- AJOUT POUR SD3.5 TURBO ---
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers import EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, DPMSolverSDEScheduler, DPMSolverSinglestepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, T5EncoderModel, T5TokenizerFast, CLIPTextModelWithProjection, CLIPTokenizer, SiglipVisionModel, SiglipImageProcessor
from compel import Compel, ReturnedEmbeddingsType

# Importer les fonctions utilitaires nécessaires (ajuster si besoin)
from huggingface_hub import login 
from .utils import txt_color, translate, lister_fichiers, str_to_bool # Importer lister_fichiers et str_to_bool

# Définir les devices ici ou les passer via config/init
cpu = torch.device("cpu")
gpu = torch.device( # Garder cette initialisation pour gpu
    f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
)

# Custom exception for Hugging Face authentication issues
class HuggingFaceAuthError(Exception):
    """Custom exception for Hugging Face authentication/access issues."""

@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull."""
    with open(os.devnull, 'w') as fnull:
        old_stderr = sys.stderr
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

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
STARVECTOR_MODEL_TYPE_KEY = "starvector" # New model type key
REALEDIT_MODEL_TYPE_KEY = "realedit_instructpix2pix" # <-- AJOUT POUR REALEDIT
SD3_5_TURBO_MODEL_TYPE_KEY = "sd3_5_turbo" # <-- AJOUT POUR SD3.5 TURBO

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
        self.current_fp8_state = False # <-- AJOUT: Mémoriser l'état FP8
        self.is_ip_adapter_loaded = False # Pour suivre l'état de l'IP-Adapter
        self.current_vae_pipe = None # Pour stocker le VAE chargé séparément

        # StarVector specific
        self.current_processor = None
        # self.current_tokenizer = None # Tokenizer might not be needed for generate_im2svg

        self.models_dir = self._get_absolute_path(config.get("MODELS_DIR", "models/checkpoints"))
        self.vae_dir = self._get_absolute_path(config.get("VAE_DIR", "models/vae"))
        self.loras_dir = self._get_absolute_path(config.get("LORAS_DIR", "models/loras"))
        self.inpaint_models_dir = self._get_absolute_path(config.get("INPAINT_MODELS_DIR", "models/inpainting"))
        self.sd3_models_dir = self._get_absolute_path(config.get("SD3_MODELS_DIR", "models/Stable-diffusion-3"))
        self.flux_models_dir = self._get_absolute_path(config.get("FLUX_MODELS_DIR", "models/flux"))
        self.flux_models_dir = self._get_absolute_path(config.get("FLUX_MODELS_DIR", "models/flux"))


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

    def list_sd3_models(self, gradio_mode=False):
        """Liste les fichiers de modèles dans le répertoire SD3."""
        model_files = lister_fichiers(self.sd3_models_dir, self.translations, ext=".safetensors", gradio_mode=gradio_mode)
        return model_files

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
            self.current_fp8_state = False # <-- AJOUT: Réinitialiser l'état FP8
            self.current_vae_pipe = None # Clear separate VAE
            self.is_ip_adapter_loaded = False # Réinitialiser l'état de l'IP-Adapter
            # Clear StarVector specific components
            self.current_processor = None
            # self.current_tokenizer = None

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
            self.current_fp8_state = False # <-- AJOUT: Réinitialiser l'état FP8
            self.is_ip_adapter_loaded = False
            self.current_vae_pipe = None
            self.current_processor = None
            # self.current_tokenizer = None
            gc.collect()
            if self.device.type == "cuda": torch.cuda.empty_cache()
            error_message = f"{translate('erreur_dechargement_modele', self.translations)}: {e}"
            if gradio_mode: gr.Error(error_message)
            return False, error_message


    def load_model(self, model_name, vae_name="Auto", model_type="standard", gradio_mode=False, custom_pipeline_id=None, from_single_file=False, use_ip_adapter=False, use_fp8=False):
        # --- GARDE-FOU pour éviter de recharger un modèle déjà chargé avec les mêmes paramètres ---
        if (self.current_pipe is not None and
            self.current_model_name == model_name and
            self.current_model_type == model_type and
            self.current_vae_name == vae_name and
            self.is_ip_adapter_loaded == use_ip_adapter and
            self.current_fp8_state == use_fp8):
            
            msg = translate("modele_deja_charge", self.translations).format(model_name=model_name)
            print(txt_color("[INFO]", "info"), msg)
            if gradio_mode:
                gr.Info(msg, duration=2.0)
            return True, msg
        # --- FIN DU GARDE-FOU ---

        if not model_name or model_name == translate("aucun_modele", self.translations) or model_name == translate("aucun_modele_trouve", self.translations):
            msg = translate("aucun_modele_selectionne", self.translations)
            print(txt_color("[ERREUR]", "erreur"), msg)
            if gradio_mode: gr.Warning(msg, duration=4.0)
            return False, msg

        if self.current_pipe is not None and (self.current_model_name != model_name or self.current_model_type != model_type or self.current_vae_name != vae_name or self.is_ip_adapter_loaded != use_ip_adapter or self.current_fp8_state != use_fp8):
            print(txt_color("[INFO]", "info"), translate("dechargement_modele_precedent_avant_chargement", self.translations))
            unload_success, unload_msg = self.unload_model(gradio_mode=gradio_mode)
            if not unload_success:
                return False, unload_msg

        pipeline_loader = None
        pipeline_class = None
        model_dir = self.models_dir
        is_from_single_file = True
        specific_torch_dtype = self.torch_dtype # Default to manager's dtype

        if model_type == SANA_MODEL_TYPE_KEY:
            pipeline_loader = SanaSprintPipeline.from_pretrained
            is_from_single_file = False
            chemin_modele = model_name
            specific_torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else self.torch_dtype
        elif model_type == COGVIEW4_MODEL_TYPE_KEY:
            pipeline_loader = CogView4Pipeline.from_pretrained
            is_from_single_file = False
            specific_torch_dtype = torch.bfloat16
            chemin_modele = model_name
        elif model_type == COGVIEW3PLUS_MODEL_TYPE_KEY:
            pipeline_loader = CogView3PlusPipeline.from_pretrained
            is_from_single_file = False
            specific_torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            chemin_modele = model_name

        elif model_type == SD3_5_TURBO_MODEL_TYPE_KEY:
            is_from_single_file = from_single_file # Utiliser l'argument passé
            chemin_modele = model_name
            specific_torch_dtype = torch.bfloat16 # Recommandé pour SD3.5
            pipeline_loader = None # Pas de loader simple, logique custom
        elif model_type == REALEDIT_MODEL_TYPE_KEY:
            pipeline_class = StableDiffusionInstructPix2PixPipeline
            is_from_single_file = False
            chemin_modele = model_name # This is the Hugging Face ID "peter-sushko/RealEdit"
            specific_torch_dtype = torch.float16 # RealEdit example uses float16
            # Le scheduler EulerAncestralDiscreteScheduler est géré par RealEdit_mod.py après le chargement
        elif model_type == STARVECTOR_MODEL_TYPE_KEY:
            # StarVector uses AutoModelForCausalLM, not a diffusers pipeline directly for loading
            is_from_single_file = False # Not a single file pipeline in the diffusers sense
            chemin_modele = model_name # This is the Hugging Face model ID
            # StarVector example uses float16, let's allow override or use manager's
            specific_torch_dtype = torch.float16 # As per example
        elif model_type == "inpainting":
            pipeline_class = StableDiffusionXLInpaintPipeline
            model_dir = self.inpaint_models_dir
        elif model_type == "img2img":
            pipeline_class = StableDiffusionXLImg2ImgPipeline
        else: # standard
            pipeline_class = StableDiffusionXLPipeline
            model_dir = self.models_dir

        if is_from_single_file and model_type not in [STARVECTOR_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]:
            chemin_modele = os.path.join(model_dir, model_name)
            if not os.path.exists(chemin_modele):
                msg = f"{translate('modele_non_trouve', self.translations)}: {chemin_modele}"
                print(txt_color("[ERREUR]", "erreur"), msg)
                if gradio_mode: gr.Warning(msg, duration=4.0)
                return False, msg
        elif model_type not in [STARVECTOR_MODEL_TYPE_KEY]: # For HF ID based models, chemin_modele is already model_name
             pass


        chemin_vae = os.path.join(self.vae_dir, vae_name) if vae_name and vae_name != "Auto" else None

        print(txt_color("[INFO]", "info"), f"{translate('chargement_modele', self.translations)}: {model_name} ({model_type})")
        if gradio_mode: gr.Info(f"{translate('chargement_modele', self.translations)}: {model_name} ({model_type})", duration=3.0)

        pipe = None
        compel_instance = None
        vae_pipe_instance = None # For separate VAE

        try:
            if model_type == STARVECTOR_MODEL_TYPE_KEY:
                print(txt_color("[INFO]", "info"), translate("loading_starvector_model_console", self.translations).format(model_name=chemin_modele))
                # Chargement spécifique à StarVector
                model_instance = AutoModelForCausalLM.from_pretrained(
                    chemin_modele, # chemin_modele is model_name (HF ID)
                    torch_dtype=specific_torch_dtype,
                    trust_remote_code=True
                )
                pipe = model_instance # Store the model instance as 'pipe' for consistency
            elif model_type in [FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]:
                chemin_modele = model_name
                is_local_file = from_single_file

                if is_local_file:
                    # Logique pour les fichiers locaux .safetensors
                    pipeline_class_flux = FluxImg2ImgPipeline if model_type == FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY else FluxPipeline
                    bfl_repo = "black-forest-labs/FLUX.1-schnell"
                    local_model_path = os.path.join(self.flux_models_dir, chemin_modele)

                    if not os.path.exists(local_model_path):
                        raise FileNotFoundError(f"{translate('modele_non_trouve', self.translations)}: {local_model_path}")

                    # For local files, we now prioritize loading without FP8 due to persistent errors.
                    # The FP8 logic is commented out but kept for future reference if the underlying library issues are resolved.
                    
                    # if use_fp8:
                    #     from optimum.quanto import qfloat8, quantize, freeze
                    #     print(txt_color("[INFO]", "info"), translate("loading_fp8_model_log", self.translations))
                    #     # The following block is the source of the errors and is disabled.
                    #     # ... (previous FP8 attempts) ...
                    #     gr.Warning("FP8 quantization for local FLUX models is currently disabled due to loading errors.", 5.0)

                    transformer = FluxTransformer2DModel.from_single_file(local_model_path, torch_dtype=specific_torch_dtype)
                    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=specific_torch_dtype)
                    
                    if use_fp8:
                        gr.Warning("FP8 quantization for local FLUX models is currently disabled due to persistent errors. The model will be loaded in its native precision.", 5.0)

                    pipe = pipeline_class_flux.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=specific_torch_dtype)
                    pipe.transformer = transformer
                    pipe.text_encoder_2 = text_encoder_2
                else:
                    # Logique pour le chargement depuis Hugging Face
                    pipeline_loader = FluxImg2ImgPipeline.from_pretrained if model_type == FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY else FluxPipeline.from_pretrained
                    try:
                        pipe = pipeline_loader(chemin_modele, torch_dtype=specific_torch_dtype)
                    except Exception as e:
                        error_message = str(e)
                        if "restricted" in error_message.lower() or "authentication" in error_message.lower():
                            raise HuggingFaceAuthError(error_message)
                        else:
                            raise
            elif model_type == SD3_5_TURBO_MODEL_TYPE_KEY and use_ip_adapter:
                print(txt_color("[INFO]", "info"), "Chargement du pipeline SD3.5 Turbo avec IP-Adapter...")
                image_encoder_id = "google/siglip-so400m-patch14-384"
                ip_adapter_id = "InstantX/SD3.5-Large-IP-Adapter"
                
                print(txt_color("[INFO]", "info"), f"  -> Chargement du feature extractor (SiglipImageProcessor) depuis : {image_encoder_id}")
                feature_extractor = SiglipImageProcessor.from_pretrained(
                    image_encoder_id,
                    torch_dtype=specific_torch_dtype
                )
                print(txt_color("[INFO]", "info"), f"  -> Chargement de l'encodeur d'image (SiglipVisionModel) depuis : {image_encoder_id}")
                image_encoder = SiglipVisionModel.from_pretrained(
                    image_encoder_id,
                    torch_dtype=specific_torch_dtype
                ).to(self.device)

                print(txt_color("[INFO]", "info"), f"  -> Chargement du pipeline de base 'stabilityai/stable-diffusion-3.5-large-turbo' avec les composants image...")
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3.5-large-turbo",
                    torch_dtype=specific_torch_dtype,
                    feature_extractor=feature_extractor,
                    image_encoder=image_encoder,
                )
                print(txt_color("[INFO]", "info"), f"  -> Chargement des poids de l'IP-Adapter depuis : {ip_adapter_id}")
                # CORRECTION: Le nom du fichier est 'ip-adapter.bin' et non '.safetensors' pour ce modèle.
                pipe.load_ip_adapter(
                    ip_adapter_id, subfolder="", weight_name="ip-adapter.bin"
                )
                pipe.safety_checker = None
            elif model_type == SD3_5_TURBO_MODEL_TYPE_KEY:
                base_model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
                if from_single_file:
                    # LOGIQUE EXISTANTE POUR LES MODÈLES LOCAUX NON-QUANTIFIÉS
                    print(txt_color("[INFO]", "info"), f"Chargement du transformateur SD3 (full precision) depuis le fichier local...")
                    transformer = SD3Transformer2DModel.from_single_file(
                        chemin_modele,
                        torch_dtype=specific_torch_dtype
                    )
                    # Charger le reste du pipeline depuis le modèle de base, en remplaçant le transformateur
                    pipe = StableDiffusion3Pipeline.from_pretrained(
                        base_model_id,
                        transformer=transformer,
                        torch_dtype=specific_torch_dtype,
                    )
                    pipe.safety_checker = None
                    

                else:
                    # Logique existante pour charger depuis Hugging Face avec quantification
                    print(txt_color("[INFO]", "info"), f"Chargement du pipeline SD3 depuis Hugging Face : {chemin_modele}")
                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    transformer = SD3Transformer2DModel.from_pretrained(
                        chemin_modele,
                        subfolder="transformer",
                        quantization_config=nf4_config,
                        torch_dtype=specific_torch_dtype
                    )
                    text_encoder_3 = T5EncoderModel.from_pretrained(
                        "diffusers/t5-nf4", torch_dtype=specific_torch_dtype
                    )

                    pipe = StableDiffusion3Pipeline.from_pretrained(
                        chemin_modele, 
                        transformer=transformer,
                        text_encoder_3=text_encoder_3,
                        torch_dtype=specific_torch_dtype,
                    )
                    pipe.safety_checker = None
            elif is_from_single_file and model_type not in [FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY]:
                # Handles standard, inpainting, img2img if they are single files
                pipeline_kwargs = {
                    "torch_dtype": specific_torch_dtype,
                    "use_safetensors": True,
                    "safety_checker": None,
                }
                if custom_pipeline_id:
                    pipeline_kwargs["custom_pipeline"] = custom_pipeline_id
                pipe = pipeline_class.from_single_file(chemin_modele, **pipeline_kwargs)
            elif pipeline_loader is not None: # For HF ID models that explicitly set pipeline_loader (Sana, CogView, FLUX)
                pipe = pipeline_loader(chemin_modele, torch_dtype=specific_torch_dtype)
            elif pipeline_class is not None: # For HF ID models that set pipeline_class (e.g., RealEdit, or standard/inpainting/img2img from HF ID)
                # This block handles models like RealEdit or if standard SDXL pipelines are loaded by HF ID.
                from_pretrained_kwargs = {
                    "torch_dtype": specific_torch_dtype, # Keep the specified dtype
                    "safety_checker": None,  # Often set to None for custom use, and RealEdit example uses it
                }
                # For RealEdit, explicitly set use_safetensors=False as it uses .bin files
                if model_type == REALEDIT_MODEL_TYPE_KEY:
                    from_pretrained_kwargs["use_safetensors"] = False

                pipe = pipeline_class.from_pretrained(chemin_modele, **from_pretrained_kwargs)
            else:
                # This case should ideally not be reached if logic for all model types is correct
                msg = f"Model loading configuration error for {model_name} ({model_type}). Neither pipeline_loader nor pipeline_class is appropriately set for non-single-file loading."
                print(txt_color("[ERREUR]", "erreur"), msg)
                if gradio_mode: gr.Error(msg)
                return False, msg


            vae_message = ""
            loaded_vae_name = "Auto"
            # VAE loading logic for non-StarVector, non-integrated VAE models
            if model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY, STARVECTOR_MODEL_TYPE_KEY, SD3_5_TURBO_MODEL_TYPE_KEY]:
                if chemin_vae and os.path.exists(chemin_vae):
                    print(txt_color("[INFO]", "info"), f"{translate('chargement_vae', self.translations)}: {vae_name}")
                    try:
                        vae_pipe_instance = AutoencoderKL.from_single_file(chemin_vae, torch_dtype=self.torch_dtype)
                        if hasattr(pipe, 'vae') and pipe.vae is not None: del pipe.vae # Remove built-in VAE
                        pipe.vae = vae_pipe_instance.to(self.device)
                        vae_message = f" + VAE: {vae_name}"
                        loaded_vae_name = vae_name
                        print(txt_color("[OK]", "ok"), f"{translate('vae_charge', self.translations)}: {vae_name}")
                    except Exception as e_vae:
                        vae_message = f" + VAE: {translate('erreur_chargement_vae_court', self.translations)}"
                        loaded_vae_name = f"Auto ({translate('erreur', self.translations)})"
                        print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_chargement_vae', self.translations)}: {vae_name} - {e_vae}")
                        if gradio_mode: gr.Warning(f"{translate('erreur_chargement_vae', self.translations)}: {vae_name} - {e_vae}", duration=4.0)
                elif chemin_vae: # Path provided but not found
                    vae_message = f" + VAE: {translate('vae_non_trouve_court', self.translations)}"
                    loaded_vae_name = f"Auto ({translate('non_trouve', self.translations)})"
                    print(txt_color("[ERREUR]", "erreur"), f"{translate('vae_non_trouve', self.translations)}: {chemin_vae}")
                    if gradio_mode: gr.Warning(f"{translate('vae_non_trouve', self.translations)}: {chemin_vae}", duration=4.0)
                else: # "Auto" or no VAE specified, use built-in
                    vae_message = f" + VAE: {translate('auto_label', self.translations)}"
                    print(txt_color("[INFO]", "info"), translate("utilisation_vae_integre", self.translations))
            elif model_type in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY, SD3_5_TURBO_MODEL_TYPE_KEY]:
                vae_message = translate("vae_integrated_with_model_type", self.translations).format(model_type=model_type)
                print(txt_color("[INFO]", "info"), f"{model_type} {translate('uses_internal_vae', self.translations)}")
            elif model_type == STARVECTOR_MODEL_TYPE_KEY:
                vae_message = translate("vae_not_applicable_for_starvector", self.translations)


            # Optimizations and device placement
            if model_type == STARVECTOR_MODEL_TYPE_KEY and pipe is not None:
                pipe.to(self.device).eval()
                if hasattr(pipe, 'model') and hasattr(pipe.model, 'processor'):
                    self.current_processor = pipe.model.processor
                else:
                    print(txt_color("[ERREUR]", "erreur"), translate("starvector_processor_not_found_log", self.translations))
                    # Optionally, fail loading if processor is critical
                    # return False, translate("starvector_processor_not_found_error", self.translations)
            elif model_type in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY] and pipe is not None:
                pipe.enable_model_cpu_offload()
                if hasattr(pipe, 'vae') and pipe.vae is not None:
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()
            elif model_type in [FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY] and pipe is not None:
                pipe.enable_sequential_cpu_offload()
                if hasattr(pipe, 'vae') and pipe.vae is not None:
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()
            elif model_type == SD3_5_TURBO_MODEL_TYPE_KEY and pipe is not None:
                # Il supporte enable_model_cpu_offload.
                pipe.enable_model_cpu_offload()
                pipe.safety_checker = None
            elif pipe is not None: # Standard SDXL, Inpainting, Img2Img
                force_cpu_offload_config = str_to_bool(str(self.config.get("FORCE_CPU_OFFLOAD", "False")))
                automatic_offload_condition = self.device.type == "cuda" and self.vram_total_gb < 8
                if (force_cpu_offload_config or automatic_offload_condition):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(self.device)

                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
                if self.device.type == "cuda" and self.vram_total_gb < 10:
                    if hasattr(pipe, 'enable_attention_slicing'):
                        pipe.enable_attention_slicing()
                if self.device.type == "cuda":
                    # --- MODIFICATION: Gérer xformers spécifiquement pour Sana Sprint ---
                    if model_type == SANA_MODEL_TYPE_KEY:
                        # Sana Sprint's attention head dimension (72) is not compatible with xformers.
                        # We must explicitly disable it to prevent errors on compatible hardware.
                        pipe.disable_xformers_memory_efficient_attention()
                        print(txt_color("[INFO]", "info"), "xformers memory efficient attention disabled for Sana Sprint pipeline due to incompatibility.")
                    else:
                        try:
                            pipe.enable_xformers_memory_efficient_attention()
                        except Exception as e_xformers:
                            print(txt_color("[AVERTISSEMENT]", "warning"), f"Failed to enable xformers: {e_xformers}")


            # Compel initialization for non-StarVector models that use it
            if model_type not in [STARVECTOR_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY, FLUX_SCHNELL_MODEL_TYPE_KEY, FLUX_SCHNELL_IMG2IMG_MODEL_TYPE_KEY, SD3_5_TURBO_MODEL_TYPE_KEY] and pipe is not None:
                has_tokenizer_2 = hasattr(pipe, 'tokenizer_2') and pipe.tokenizer_2 is not None
                has_encoder_2 = hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None
                compel_returned_embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                compel_requires_pooled = [False, True] if has_encoder_2 else False
                compel_instance = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2] if has_tokenizer_2 else pipe.tokenizer,
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2] if has_encoder_2 else pipe.text_encoder,
                    returned_embeddings_type=compel_returned_embeddings_type,
                    requires_pooled=compel_requires_pooled,
                    device=self.device
                )
            elif model_type == SANA_MODEL_TYPE_KEY and pipe is not None: # Sana specific Compel
                 compel_instance = Compel(
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=False, # Sana doesn't use pooled output in the same way
                    device=self.device
                )

            self.current_pipe = pipe
            self.current_compel = compel_instance
            self.current_model_name = model_name
            self.current_vae_name = loaded_vae_name
            self.current_model_type = model_type
            self.current_fp8_state = use_fp8 # <-- AJOUT: Sauvegarder l'état FP8
            self.is_ip_adapter_loaded = use_ip_adapter # Mettre à jour l'état de l'IP-Adapter
            self.current_vae_pipe = vae_pipe_instance # Store separate VAE
            self.loaded_loras.clear()
            self.current_sampler_key = None

            final_message = f"{translate('modele_charge', self.translations)}: {model_name}{vae_message}"
            print(txt_color("[OK]", "ok"), final_message)
            if gradio_mode: gr.Info(final_message, duration=3.0)
            return True, final_message

        except HuggingFaceAuthError as e: # AJOUT: Gérer l'erreur d'authentification Hugging Face
            print(txt_color("[ERREUR]", "erreur"), f"{translate('error_hf_auth_required', self.translations)}: {e}")
            traceback.print_exc()
            self.unload_model() # Nettoyer tout chargement partiel
            error_msg = f"{translate('error_hf_auth_required', self.translations)}: {e}"
            if gradio_mode: gr.Error(error_msg)
            return False, error_msg
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

    def get_current_processor(self):
        """Returns the currently loaded processor (for models like StarVector)."""
        return self.current_processor

    def get_current_vae(self):
        """Returns the currently loaded VAE pipeline or the VAE from the main pipe."""
        if self.current_vae_pipe:
            return self.current_vae_pipe
        elif self.current_pipe and hasattr(self.current_pipe, 'vae'):
            return self.current_pipe.vae
        return None

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
            self.loaded_loras[adapter_name] = {'filename': lora_item_name, 'scale': scale}
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
                    print(txt_color("[INFO]", "info"), translate("attempting_adapter_delete_after_fail_log", self.translations).format(adapter_name=adapter_name))
            except Exception as e_unload_fail:
                print(txt_color("[WARN]", "warning"), translate("lora_cleanup_failed_after_error_log", self.translations).format(adapter_name=adapter_name, error=e_unload_fail))
            return False, msg

    def unload_lora(self, adapter_name):
        if self.current_pipe is None: return False, translate("erreur_pas_modele", self.translations)
        try:
            if hasattr(self.current_pipe, "unload_lora_weights"): # Diffusers standard way
                print(txt_color("[INFO]", "info"), f"Déchargement des poids LoRA (unload_lora_weights): {adapter_name}")
                print(txt_color("[INFO]", "info"), translate("unloading_lora_weights_log", self.translations).format(adapter_name=adapter_name))
                self.current_pipe.unload_lora_weights() # This unloads ALL LoRAs

                temp_loaded_loras_info = self.loaded_loras.copy()
                if adapter_name in temp_loaded_loras_info:
                    del temp_loaded_loras_info[adapter_name]

                self.loaded_loras.clear() # Clear all before re-applying

                # Re-apply remaining LoRAs using load_lora_weights
                if temp_loaded_loras_info:
                    print(txt_color("[INFO]", "info"), translate("reapplying_remaining_loras_log", self.translations).format(adapter_name=adapter_name, lora_list=list(temp_loaded_loras_info.keys())))
                    for remaining_adapter_name, lora_info in temp_loaded_loras_info.items():
                        lora_filename = lora_info['filename']
                        scale = lora_info['scale']
                        lora_full_path = os.path.join(self.loras_dir, lora_filename)
                        try:
                            self.current_pipe.load_lora_weights(lora_full_path, adapter_name=remaining_adapter_name)
                            self.loaded_loras[remaining_adapter_name] = {'filename': lora_filename, 'scale': scale} # Store back with filename
                        except Exception as e_reapply:
                            print(txt_color("[WARN]", "warning"), f"Failed to re-apply LoRA {remaining_adapter_name}: {e_reapply}")
                            # If re-application fails, it's better to remove it from loaded_loras
                            if remaining_adapter_name in self.loaded_loras:
                                del self.loaded_loras[remaining_adapter_name]
                
            elif hasattr(self.current_pipe, 'delete_adapter'): # PEFT way
                print(txt_color("[INFO]", "info"), translate("deleting_lora_adapter_log", self.translations).format(adapter_name=adapter_name))
                self.current_pipe.delete_adapter(adapter_name)
            else:
                print(txt_color("[WARN]", "warning"), translate("lora_unload_method_not_found_log", self.translations).format(pipeline_class=self.current_pipe.__class__.__name__))

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
        
        # 1. Analyser les LoRAs demandés depuis l'interface utilisateur
        requested_loras_config = {} # adapter_name: {'filename': str, 'scale': float}
        lora_checks = lora_ui_config.get('lora_checks', [])
        lora_dropdowns = lora_ui_config.get('lora_dropdowns', [])
        lora_scales = lora_ui_config.get('lora_scales', [])

        for i, is_checked in enumerate(lora_checks):
            if is_checked and i < len(lora_dropdowns) and i < len(lora_scales):
                lora_filename = lora_dropdowns[i]
                scale = lora_scales[i]
                if lora_filename and lora_filename not in [
                    translate("aucun_lora_disponible", self.translations),
                    translate("aucun_modele_trouve", self.translations),
                    translate("repertoire_not_found", self.translations)
                ]:
                    adapter_name = os.path.splitext(lora_filename)[0].replace(".", "_").replace(" ", "_")
                    requested_loras_config[adapter_name] = {'filename': lora_filename, 'scale': scale}

        # 2. Décharger tous les LoRAs existants pour garantir un état propre.
        if hasattr(self.current_pipe, "unload_lora_weights"):
            self.current_pipe.unload_lora_weights()
            print(txt_color("[INFO]", "info"), translate("loras_unloaded_before_new_config", self.translations))
        elif hasattr(self.current_pipe, "set_adapters"):
            self.current_pipe.set_adapters([], adapter_weights=[])
        self.loaded_loras.clear()

        # 3. Charger tous les LoRAs demandés
        if not requested_loras_config:
            return translate("no_lora_selected_all_disabled", self.translations)

        for name, config in requested_loras_config.items():
            lora_path = os.path.join(self.loras_dir, config['filename'])
            try:
                # Suppress the harmless "No LoRA keys associated to CLIPTextModel" warning
                with suppress_stderr():
                    self.current_pipe.load_lora_weights(lora_path, adapter_name=name)
                self.loaded_loras[name] = config
                messages.append(translate("lora_loaded_name", self.translations).format(name=name))
            except Exception as e:
                msg = translate("lora_load_error_name", self.translations).format(name=name, e=e)
                print(txt_color("[ERREUR]", "erreur"), msg)
                messages.append(msg)
                if name in self.loaded_loras:
                    del self.loaded_loras[name]
        
        # 4. Activer l'ensemble final d'adaptateurs sur le pipeline
        final_adapters_to_activate = list(self.loaded_loras.keys())
        final_weights = [self.loaded_loras[name]['scale'] for name in final_adapters_to_activate]

        if final_adapters_to_activate:
            try:
                self.current_pipe.set_adapters(final_adapters_to_activate, adapter_weights=final_weights)
                
                weight_part_str = translate("lora_weight_log_part", self.translations)
                activated_loras_details = [f"{name}{weight_part_str.format(weight=self.loaded_loras[name]['scale'])}" for name in final_adapters_to_activate]
                
                msg = translate("loras_activated_details", self.translations).format(details=', '.join(activated_loras_details))
                print(txt_color("[OK]", "ok"), msg)
                messages.append(msg)
            except Exception as e:
                msg = translate("loras_activation_error", self.translations).format(e=e)
                print(txt_color("[ERREUR]", "erreur"), msg)
                messages.append(msg)
                traceback.print_exc()

        final_message = "\n".join(messages)
        return final_message if final_message else translate("no_lora_to_apply", self.translations)
