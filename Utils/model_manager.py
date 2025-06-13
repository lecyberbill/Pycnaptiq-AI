# d:\image_to_text\cyberbill_SDXL\cyberbill_image_generator\Utils\model_manager.py
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

SANA_MODEL_TYPE_KEY = "sana_sprint" # <-- AJOUT DEFINITION CONSTANTE
COGVIEW4_MODEL_ID = "THUDM/CogView4-6B"
COGVIEW4_MODEL_TYPE_KEY = "cogview4"
COGVIEW3PLUS_MODEL_ID = "THUDM/CogView3-Plus-3B"
COGVIEW3PLUS_MODEL_TYPE_KEY = "cogview3plus"

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
        self.current_vae_name = None # Garder pour la cohérence, même si Sana ne l'utilise pas
        self.current_model_type = None  # 'standard', 'inpainting', 'img2img'
        self.current_sampler_key = None # Ajout pour suivre le sampler
        self.loaded_loras = {}  # adapter_name: scale

        # Obtenir les chemins depuis la config
        self.models_dir = self._get_absolute_path(config.get("MODELS_DIR", "models/checkpoints"))
        self.vae_dir = self._get_absolute_path(config.get("VAE_DIR", "models/vae"))
        self.loras_dir = self._get_absolute_path(config.get("LORAS_DIR", "models/loras"))
        self.inpaint_models_dir = self._get_absolute_path(config.get("INPAINT_MODELS_DIR", "models/inpainting"))
        # Le modèle img2img utilise souvent le même checkpoint que le modèle standard
        # self.img2img_models_dir = self._get_absolute_path(config.get("IMG2IMG_MODELS_DIR", self.models_dir)) # Ou un dossier dédié

    def _get_absolute_path(self, path_from_config):
        """Convertit un chemin relatif de la config en chemin absolu."""
        root_dir = Path(__file__).parent.parent # Chemin racine du projet
        if not os.path.isabs(path_from_config):
            return os.path.abspath(os.path.join(root_dir, path_from_config))
        return path_from_config

    # --- Méthodes pour lister les fichiers ---
    # --- Méthodes pour lister les fichiers ---
    def list_models(self, model_type="standard", gradio_mode=False):
        """Liste les modèles disponibles pour un type donné, en filtrant le placeholder."""
        if model_type == "inpainting":
            dir_path = self.inpaint_models_dir
        else: 
            dir_path = self.models_dir

        # 1. Obtenir la liste brute des fichiers
        model_files_raw = lister_fichiers(dir_path, self.translations, ext=".safetensors", gradio_mode=gradio_mode)

        # 2. Définir et filtrer le placeholder
        placeholder_model = "your_default_modele.safetensors"
        filtered_model_files = [f for f in model_files_raw if f != placeholder_model]

        # 3. Gérer les cas où la liste filtrée est vide
        no_model_msg = translate("aucun_modele_trouve", self.translations)
        not_found_msg = translate("repertoire_not_found", self.translations) 

        if not filtered_model_files:
            # Si la liste filtrée est vide, vérifier la liste brute
            if model_files_raw and model_files_raw[0] != no_model_msg and model_files_raw[0] != not_found_msg:
                return [no_model_msg]
            else:
                return model_files_raw
        else:
            return filtered_model_files


    def list_vaes(self, gradio_mode=False):
        """Liste les VAEs disponibles."""
        # Retourne ["Auto"] + liste réelle pour l'UI
        return ["Auto"] + lister_fichiers(self.vae_dir, self.translations, ext=".safetensors", gradio_mode=gradio_mode)

    def list_loras(self, gradio_mode=False):
        """Liste les LoRAs disponibles."""
        # Modifié pour lister les dossiers LoRA ET les fichiers .safetensors individuels
        if not os.path.isdir(self.loras_dir):
            msg = translate("repertoire_not_found", self.translations)
            if gradio_mode: gr.Warning(msg + f": {self.loras_dir}", 3.0)
            return [msg]
        
        # Lister les dossiers ET les fichiers .safetensors
        lora_items = []
        for item_name in os.listdir(self.loras_dir):
            item_path = os.path.join(self.loras_dir, item_name)
            if os.path.isdir(item_path) or (os.path.isfile(item_path) and item_name.lower().endswith(".safetensors")):
                lora_items.append(item_name)
        
        if not lora_items:
            msg = translate("aucun_lora_disponible", self.translations) # Assurez-vous que cette clé existe
            if gradio_mode: gr.Info(msg, 3.0)
            return [msg]
        return sorted(lora_items) # Trier pour la cohérence

    def get_cuda_free_memory_gb(self):
        """Calcule la mémoire VRAM libre + inactive réservée en Go."""
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
            print(
                txt_color("[ERREUR]", "erreur"),
                f"Erreur récupération mémoire CUDA: {e}",
            )
            return 0

    def unload_model(self, gradio_mode=False):
        """Décharge le modèle et Compel actuellement chargés."""
        if self.current_pipe is None and self.current_compel is None:
            msg = translate("aucun_modele_a_decharger", self.translations)
            print(txt_color("[INFO]", "info"), msg)
            if gradio_mode:
                gr.Info(msg, duration=2.0)
            return True, msg


        print(
            txt_color("[INFO]", "info"),
            translate("dechargement_modele_en_cours", self.translations), # Nouvelle clé
        )
        if gradio_mode:
            gr.Info(translate("dechargement_modele_en_cours", self.translations), duration=2.0) # Utiliser duration

        try:
            if self.current_pipe is not None:
                if not isinstance(self.current_pipe, dict): # Cas pipeline Diffusers standard (pas Janus)
                    self.current_pipe.to(cpu)
                    # Supprimer les composants explicitement pour aider GC
                    attrs_to_delete = ['vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'scheduler', 'feature_extractor', 'safety_checker']
                    for attr in attrs_to_delete:
                        if hasattr(self.current_pipe, attr):
                            try: # Ajouter try/except pour la suppression
                                component_to_delete = getattr(self.current_pipe, attr)
                                delattr(self.current_pipe, attr)
                                del component_to_delete # Essayer de supprimer la référence
                            except Exception as e_del:
                                print(txt_color("[WARN]", "warning"), f"Impossible de supprimer l'attribut {attr}: {e_del}")
                else: # Cas d'un dictionnaire (anciennement Janus, maintenant on le supprime juste)
                    # On ne fait rien de spécifique ici car on ne s'attend plus à ce type
                    pass
                del self.current_pipe
                self.current_pipe = None

            if self.current_compel is not None:
                if hasattr(self.current_compel, 'tokenizer'): del self.current_compel.tokenizer
                if hasattr(self.current_compel, 'text_encoder'): del self.current_compel.text_encoder
                del self.current_compel
                self.current_compel = None

            # Réinitialiser les infos
            self.current_model_name = None
            self.current_vae_name = None
            self.current_model_type = None
            self.current_sampler_key = None # Réinitialiser aussi le sampler
            self.loaded_loras.clear()

            # Nettoyage mémoire
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()

            print(
                txt_color("[OK]", "ok"),
                translate("modele_precedent_decharge", self.translations),
            )
            if gradio_mode:
                gr.Info(translate("modele_precedent_decharge", self.translations), duration=3.0)
            return True, translate("modele_precedent_decharge", self.translations)
        except Exception as e:
            print(
                txt_color("[ERREUR]", "erreur"),
                f"{translate('erreur_dechargement_modele', self.translations)}: {e}",
            )
            traceback.print_exc()
            # Forcer la réinitialisation même en cas d'erreur
            self.current_pipe = None
            self.current_compel = None
            self.current_model_name = None
            self.current_vae_name = None
            self.current_model_type = None
            self.current_sampler_key = None
            self.loaded_loras.clear()
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            error_message = f"{translate('erreur_dechargement_modele', self.translations)}: {e}"
            if gradio_mode: gr.Error(error_message)
            return False, error_message


    def load_model(self, model_name, vae_name="Auto", model_type="standard", gradio_mode=False, custom_pipeline_id=None): # <-- AJOUT custom_pipeline_id

        if not model_name or model_name == translate("aucun_modele", self.translations) or model_name == translate("aucun_modele_trouve", self.translations):
            msg = translate("aucun_modele_selectionne", self.translations)
            print(txt_color("[ERREUR]", "erreur"), msg)
            if gradio_mode: gr.Warning(msg, duration=4.0)
            return False, msg

        # Déterminer le chemin et le type de pipeline
        pipeline_loader = None
        pipeline_class = None  # Initialiser pipeline_class ici
        model_dir = self.models_dir # Le répertoire par défaut pour les modèles
        is_from_single_file = True # Par défaut

        if model_type == SANA_MODEL_TYPE_KEY: # <-- AJOUT GESTION SANA
            pipeline_loader = SanaSprintPipeline.from_pretrained
            is_from_single_file = False # Chargé depuis ID Hugging Face
            # model_name est l'ID Hugging Face ici
            chemin_modele = model_name # Utiliser l'ID directement
        elif model_type == COGVIEW4_MODEL_TYPE_KEY:
            pipeline_loader = CogView4Pipeline.from_pretrained
            is_from_single_file = False
            # model_name est l'ID Hugging Face pour CogView4
            specific_torch_dtype = torch.bfloat16 # CogView4 spécifique
            chemin_modele = model_name # Utiliser l'ID directement
        elif model_type == COGVIEW3PLUS_MODEL_TYPE_KEY:
            pipeline_loader = CogView3PlusPipeline.from_pretrained
            is_from_single_file = False
            # Choisir bfloat16 si supporté, sinon float16 pour CogView3-Plus
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                specific_torch_dtype = torch.bfloat16
                print(txt_color("[INFO]", "info"), "Utilisation de torch.bfloat16 pour CogView3-Plus (supporté).")
            else:
                specific_torch_dtype = torch.float16
                print(txt_color("[INFO]", "info"), "Utilisation de torch.float16 pour CogView3-Plus (bfloat16 non supporté ou CUDA non disponible).")
            chemin_modele = model_name # Utiliser l'ID directement
        elif model_type == "inpainting":
            # Décharger l'ancien modèle AVANT de potentiellement échouer sur le chemin
            unload_success, unload_msg = self.unload_model(gradio_mode=gradio_mode)
            if not unload_success: return False, unload_msg
            pipeline_class = StableDiffusionXLInpaintPipeline # Correction: pipeline_class doit être défini ici
            model_dir = self.inpaint_models_dir # Correction: model_dir doit être défini ici
        elif model_type == "img2img":
            pipeline_class = StableDiffusionXLImg2ImgPipeline
            # Utiliser models_dir par défaut pour img2img, car ils utilisent souvent les mêmes checkpoints
            # Si un dossier dédié est configuré, décommenter la ligne suivante
            # model_dir = self.img2img_models_dir
        else: # standard
            pipeline_class = StableDiffusionXLPipeline
            # Décharger l'ancien modèle AVANT de potentiellement échouer sur le chemin
            unload_success, unload_msg = self.unload_model(gradio_mode=gradio_mode)
            if not unload_success: return False, unload_msg
            model_dir = self.models_dir

        # Définir le chemin seulement si c'est un fichier local
        if is_from_single_file: # On ne gère plus JANUS_MODEL_TYPE_KEY
            chemin_modele = os.path.join(model_dir, model_name)
            if not os.path.exists(chemin_modele):
                msg = f"{translate('modele_non_trouve', self.translations)}: {chemin_modele}"
                print(txt_color("[ERREUR]", "erreur"), msg)
                if gradio_mode: gr.Warning(msg, duration=4.0)
                return False, msg
        chemin_vae = os.path.join(self.vae_dir, vae_name) if vae_name and vae_name != "Auto" else None

        # Le déchargement est maintenant fait au début de chaque bloc if/elif pour les types non-HF
        # ou au début du bloc JANUS_MODEL_TYPE_KEY.

        print(txt_color("[INFO]", "info"), f"{translate('chargement_modele', self.translations)}: {model_name} ({model_type})")
        if gradio_mode: gr.Info(f"{translate('chargement_modele', self.translations)}: {model_name} ({model_type})", duration=3.0)

        try:
            # Charger le pipeline avec gestion d'erreur plus fine
            pipe = None

            try:
                pipeline_kwargs = {
                    "torch_dtype": specific_torch_dtype if 'specific_torch_dtype' in locals() else self.torch_dtype,
                    "use_safetensors": True,
                    "safety_checker": None,
                }
                if custom_pipeline_id: # Si un custom_pipeline est fourni
                    pipeline_kwargs["custom_pipeline"] = custom_pipeline_id
                    print(txt_color("[INFO]", "info"), f"Tentative de chargement avec custom_pipeline: {custom_pipeline_id}")

                if is_from_single_file:
                    pipe = pipeline_class.from_single_file(
                        chemin_modele,
                        **pipeline_kwargs # Passer tous les kwargs
                    )
                else: # Cas from_pretrained (Sana Sprint, Cosmos)
                    if model_type == SANA_MODEL_TYPE_KEY:
                        dtype_to_use = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else self.torch_dtype
                        print(f"[INFO] Chargement {model_type} ({chemin_modele}) avec dtype: {dtype_to_use}")
                        pipe = pipeline_loader(
                            chemin_modele, # C'est l'ID HF ici
                            torch_dtype=dtype_to_use,
                        )
                    elif model_type == COGVIEW4_MODEL_TYPE_KEY:
                        print(f"[INFO] Chargement {model_type} ({chemin_modele}) avec dtype: {specific_torch_dtype}")
                        pipe = pipeline_loader( # CogView4Pipeline.from_pretrained
                            chemin_modele, # C'est l'ID HF (COGVIEW4_MODEL_ID)
                            torch_dtype=specific_torch_dtype,
                        )
                    elif model_type == COGVIEW3PLUS_MODEL_TYPE_KEY:
                        print(f"[INFO] Chargement {model_type} ({chemin_modele}) avec dtype: {specific_torch_dtype}")
                        pipe = pipeline_loader( # CogView3PlusPipeline.from_pretrained
                            chemin_modele, # C'est l'ID HF (COGVIEW3PLUS_MODEL_ID)
                            torch_dtype=specific_torch_dtype,
                        )
            except Exception as e_pipe_load:
                # Erreur spécifique au chargement du pipeline
                raise RuntimeError(f"{translate('erreur_chargement_pipeline', self.translations)}: {e_pipe_load}")
            # Charger le VAE externe si spécifié
            vae_message = ""
            loaded_vae_name = "Auto"
            if model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY] \
               and chemin_vae and os.path.exists(chemin_vae):
                print(txt_color("[INFO]", "info"), f"{translate('chargement_vae', self.translations)}: {vae_name}")
                try:
                    vae = AutoencoderKL.from_single_file(chemin_vae, torch_dtype=self.torch_dtype)
                    # Remplacer le VAE du pipeline par celui chargé
                    # S'assurer que l'ancien VAE est supprimé si possible
                    if hasattr(pipe, 'vae') and pipe.vae is not None:
                        del pipe.vae
                    pipe.vae = vae.to(self.device) # Déplacer le nouveau VAE sur le device
                    vae_message = f" + VAE: {vae_name}"
                    loaded_vae_name = vae_name
                    print(txt_color("[OK]", "ok"), f"{translate('vae_charge', self.translations)}: {vae_name}")
                except Exception as e_vae:
                    vae_message = f" + VAE: {translate('erreur_chargement_vae_court', self.translations)}"
                    loaded_vae_name = f"Auto ({translate('erreur', self.translations)})"
                    print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_chargement_vae', self.translations)}: {vae_name} - {e_vae}")
                    if gradio_mode: gr.Warning(f"{translate('erreur_chargement_vae', self.translations)}: {vae_name} - {e_vae}", duration=4.0) # type: ignore
            elif model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY] and chemin_vae: # Chemin spécifié mais non trouvé
                 vae_message = f" + VAE: {translate('vae_non_trouve_court', self.translations)}"
                 loaded_vae_name = f"Auto ({translate('non_trouve', self.translations)})"
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('vae_non_trouve', self.translations)}: {chemin_vae}")
                 if gradio_mode: gr.Warning(f"{translate('vae_non_trouve', self.translations)}: {chemin_vae}", duration=4.0) # type: ignore
            elif model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY]: # VAE Auto (intégré)
                 vae_message = f" + VAE: {translate('auto_label', self.translations)}" # Nouvelle clé pour "Auto"
                 print(txt_color("[INFO]", "info"), translate("utilisation_vae_integre", self.translations))
            elif model_type == COGVIEW4_MODEL_TYPE_KEY:
                vae_message = f" (VAE intégré à CogView4)"
                print(txt_color("[INFO]", "info"), "CogView4 utilise son VAE interne.")
            elif model_type == COGVIEW3PLUS_MODEL_TYPE_KEY:
                vae_message = f" (VAE intégré à CogView3-Plus)"
                print(txt_color("[INFO]", "info"), "CogView3-Plus utilise son VAE interne.")

            # Gestion du déplacement vers le device et de l'offloading CPU
            if model_type in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY] and pipe is not None and not isinstance(pipe, dict): # Assurer que pipe est un pipeline diffusers
                print(txt_color("[INFO]", "info"), "Application des configurations spécifiques à CogView4 (CPU offload, slicing, tiling)...")
                if gradio_mode: gr.Info(f"Application des configurations spécifiques à {model_type}...", duration=2.0) # type: ignore
                try:
                    pipe.enable_model_cpu_offload() # Gère aussi le déplacement vers device
                    if hasattr(pipe, 'vae') and pipe.vae is not None:
                        pipe.vae.enable_slicing()
                        pipe.vae.enable_tiling()
                        print(txt_color("[OK]", "ok"), "Slicing et Tiling VAE activés pour CogView4.")
                    else:
                        print(txt_color("[AVERTISSEMENT]", "warning"), f"Le pipe {model_type} n'a pas d'attribut 'vae' ou VAE est None. Slicing/Tiling VAE non appliqué.")
                except Exception as e_cog_config:
                    print(txt_color("[ERREUR]", "erreur"), f"Erreur lors de la configuration spécifique de {model_type}: {e_cog_config}")
                    if gradio_mode: gr.Warning(f"Erreur configuration {model_type}: {e_cog_config}", duration=4.0) # type: ignore
            elif pipe is not None and not isinstance(pipe, dict): # Pour les pipelines Diffusers (non Janus)
                force_cpu_offload_config = str_to_bool(str(self.config.get("FORCE_CPU_OFFLOAD", "False")))
                automatic_offload_condition = self.device.type == "cuda" and self.vram_total_gb < 8
                should_enable_offload = (force_cpu_offload_config or automatic_offload_condition) and model_type not in [SANA_MODEL_TYPE_KEY]

                if should_enable_offload:
                    if force_cpu_offload_config:
                        reason_message = translate("activation_cpu_offload_forced_config", self.translations)
                    else: # Must be automatic_offload_condition that was true
                        reason_message = translate("activation_cpu_offload", self.translations).format(vram=self.vram_total_gb)
                    print(txt_color("[INFO]", "info"), reason_message)
                    if gradio_mode: gr.Info(reason_message, duration=3.0) # type: ignore
                    pipe.enable_model_cpu_offload()
                else:
                    print(txt_color("[INFO]", "info"), f"{translate('deplacement_modele_device', self.translations)} {self.device}...")
                    pipe.to(self.device)

            # Appliquer les optimisations mémoire si nécessaire
            # Ne pas appliquer les optimisations VAE pour Sana Sprint
            if pipe is not None and not isinstance(pipe, dict): # Pour les pipelines Diffusers (non Janus)
                try:
                    if model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY]:
                        pipe.enable_vae_slicing()
                        pipe.enable_vae_tiling()

                    if self.device.type == "cuda" and self.vram_total_gb < 10 and model_type not in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY]:
                        if hasattr(pipe, 'enable_attention_slicing'): # Vérifier si la méthode existe
                            pipe.enable_attention_slicing()
                            print(txt_color("[INFO]", "info"), translate("optimisation_memoire_activee", self.translations))
                        else:
                            print(txt_color("[INFO]", "info"), f"enable_attention_slicing non disponible pour {model_type}.")
 
                    if self.device.type == "cuda" and model_type not in [SANA_MODEL_TYPE_KEY, COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY]:
                        try: # Ajouter try/except pour xformers
                            pipe.enable_xformers_memory_efficient_attention()
                            print(txt_color("[INFO]", "info"), "XFormers activé.")
                        except ImportError:
                            print(txt_color("[INFO]", "info"), "XFormers non disponible, ignoré.")
                        except Exception as e_xformers:
                            print(txt_color("[AVERTISSEMENT]", "warning"), f"Erreur activation XFormers: {e_xformers}")
                    elif model_type in [SANA_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY]:
                        print(txt_color("[INFO]", "info"), f"Optimisations XFormers/attention slicing non applicables ou désactivées pour {model_type}.")

                except Exception as e_optim:
                    print(txt_color("[AVERTISSEMENT]", "warning"), f"Erreur application optimisations: {e_optim}")


            # Créer Compel
            if model_type in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY]: # On ne gère plus JANUS_MODEL_TYPE_KEY
                self.current_compel = None
                print(txt_color("[INFO]", "info"), f"Compel n'est pas utilisé pour {model_type}.")
            elif pipe is not None: # Pour les pipelines Diffusers qui utilisent Compel (SDXL, Inpainting, Img2Img, Sana)
                has_tokenizer_2 = hasattr(pipe, 'tokenizer_2') and pipe.tokenizer_2 is not None
                has_encoder_2 = hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None

                if model_type == SANA_MODEL_TYPE_KEY:
                    # Configuration spécifique pour Sana Sprint
                    compel_returned_embeddings_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
                    compel_requires_pooled = False
                else: # Cas standard SDXL (standard, inpainting, img2img)
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
                # Ce cas se produit si pipe est None pour un modèle qui devrait avoir un pipe,
                # ou pour un type de modèle non géré qui ne devrait pas avoir de Compel.
                self.current_compel = None
                if model_type not in [COGVIEW4_MODEL_TYPE_KEY, COGVIEW3PLUS_MODEL_TYPE_KEY]:
                    print(txt_color("[AVERTISSEMENT]", "warning"), f"Compel non initialisé pour {model_type} car le pipeline est None ou non géré.")

            # Mettre à jour l'état interne
            self.current_pipe = pipe
            # Pour les modèles HF, model_name est l'ID HF. Pour les single_file, c'est le nom de fichier.
            self.current_model_name = model_name 
            self.current_vae_name = loaded_vae_name
            self.current_model_type = model_type
            self.loaded_loras.clear() # Effacer les LoRAs lors du changement de modèle
            # Réinitialiser le sampler lors du chargement du modèle
            self.current_sampler_key = None # Ou définir un sampler par défaut ici

            final_message = f"{translate('modele_charge', self.translations)}: {model_name}{vae_message}"
            print(txt_color("[OK]", "ok"), final_message)
            if gradio_mode: gr.Info(final_message, duration=3.0) # type: ignore
            return True, final_message

        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_generale_chargement_modele', self.translations)}: {e}") # Nouvelle clé plus générique
            traceback.print_exc()
            self.unload_model() # Assurer le nettoyage en cas d'erreur
            error_msg = f"{translate('erreur_chargement_modele', self.translations)}: {e}"
            if gradio_mode: gr.Error(error_msg) # type: ignore
            return False, error_msg

    def get_current_pipe(self):
        """Retourne le pipeline actuellement chargé."""
        return self.current_pipe

    def get_current_compel(self):
        """Retourne l'instance Compel actuelle."""
        return self.current_compel

    # --- Méthodes pour LoRA ---

    def load_lora(self, lora_item_name, scale): # lora_folder_name devient lora_item_name
        """Charge un LoRA spécifique depuis son dossier ou son fichier."""
        if self.current_pipe is None:
            msg = translate("erreur_pas_modele_pour_lora", self.translations) # Nouvelle clé
            print(txt_color("[ERREUR]", "erreur"), msg)
            return False, msg

        # lora_item_name peut être un nom de dossier ou un nom de fichier .safetensors
        lora_full_path = os.path.join(self.loras_dir, lora_item_name)
        if not os.path.exists(lora_full_path): # Vérifier si le chemin (dossier ou fichier) existe
            msg = f"{translate('erreur_lora_introuvable', self.translations)}: {lora_full_path}" # Clé plus générique
            print(txt_color("[ERREUR]", "erreur"), msg)
            return False, msg

        # Créer un nom d'adaptateur unique et valide
        # Utiliser le nom de l'item (dossier ou fichier sans extension) comme base
        adapter_base_name = os.path.splitext(lora_item_name)[0] if lora_item_name.lower().endswith(".safetensors") else lora_item_name
        adapter_name = adapter_base_name.replace(".", "_").replace(" ", "_")

        try:
            print(txt_color("[INFO]", "info"), f"{translate('lora_charge_depuis_chemin', self.translations)} {lora_full_path}") # Clé plus générique

            # Charger les poids LoRA depuis le dossier.
            # load_lora_weights prend le chemin du dossier LoRA.
            self.current_pipe.load_lora_weights(lora_full_path, adapter_name=adapter_name)
            # Note: load_lora_weights peut aussi prendre un state_dict ou un chemin vers un fichier .safetensors unique.
            # lora_full_path peut maintenant être un dossier OU un fichier.
            
            self.loaded_loras[adapter_name] = scale # Mettre à jour l'état interne
            msg = f"{translate('lora_charge', self.translations)}: {adapter_name} (Scale: {scale})"
            print(txt_color("[OK]", "ok"), msg)
            return True, msg
        except Exception as e:
            msg = f"{translate('erreur_lora_chargement', self.translations)}: {e}"
            print(txt_color("[ERREUR]", "erreur"), msg)
            traceback.print_exc()
            # Tentative de nettoyage si l'adaptateur a été partiellement ajouté
            try:
                # delete_adapter devrait gérer le cas où l'adaptateur n'existe pas.
                if hasattr(self.current_pipe, 'delete_adapter'):
                    self.current_pipe.delete_adapter(adapter_name)
                    print(txt_color("[INFO]", "info"), f"Tentative de suppression de l'adaptateur '{adapter_name}' après échec de chargement.")
            except Exception as e_unload_fail:
                print(txt_color("[WARN]", "warning"), f"Échec du nettoyage après erreur chargement LoRA {adapter_name}: {e_unload_fail}")
            return False, msg

    def unload_lora(self, adapter_name):
        """Décharge un LoRA spécifique par son nom d'adaptateur."""
        if self.current_pipe is None: return False, translate("erreur_pas_modele", self.translations)

        try:
            # --- Utiliser unload_lora_weights ---
            if hasattr(self.current_pipe, "unload_lora_weights"):
                print(txt_color("[INFO]", "info"), f"Déchargement des poids LoRA: {adapter_name}")
                self.current_pipe.unload_lora_weights()
            else:
                print(txt_color("[WARN]", "warning"), f"La méthode 'unload_lora_weights' n'existe pas pour ce pipeline ({self.current_pipe.__class__.__name__}). Désactivation via set_adapters tentée.")
                # Fallback: tenter de désactiver via set_adapters (peut échouer avec 'unet')
                current_active_adapters = list(self.loaded_loras.keys())
                current_active_weights = list(self.loaded_loras.values())
                if adapter_name in current_active_adapters:
                    idx = current_active_adapters.index(adapter_name)
                    current_active_adapters.pop(idx)
                    current_active_weights.pop(idx)
                    self.current_pipe.set_adapters(current_active_adapters, adapter_weights=current_active_weights)

            if adapter_name in self.loaded_loras:
                del self.loaded_loras[adapter_name] # Mettre à jour l'état interne

            msg = f"{translate('lora_decharge_nom', self.translations)}: {adapter_name}"
            print(txt_color("[INFO]", "info"), msg)
            return True, msg
        except Exception as e:
            msg = f"{translate('erreur_lora_dechargement', self.translations)}: {e}" # Nouvelle clé
            # --- AJOUT: Afficher l'erreur correctement ---
            print(txt_color("[ERREUR]", "erreur"), msg)
            traceback.print_exc() # Imprimer la trace complète
            # Retirer de l'état interne même si le déchargement API échoue
            if adapter_name in self.loaded_loras:
                del self.loaded_loras[adapter_name]
            return False, msg

    def apply_loras(self, lora_ui_config, gradio_mode=False):
        """
        Gère le chargement/déchargement/mise à jour des LoRAs basé sur la configuration de l'UI.
        lora_ui_config: {'lora_checks': [], 'lora_dropdowns': [], 'lora_scales': []}
        """
        if self.current_pipe is None:
            msg = translate("erreur_pas_modele_pour_lora", self.translations)
            if gradio_mode: gr.Warning(msg, duration=4.0)
            return msg

        # --- Initialiser messages ici ---
        messages = [] # Pour collecter les messages de chargement/erreur

        lora_checks = lora_ui_config.get('lora_checks', [])
        lora_dropdowns = lora_ui_config.get('lora_dropdowns', [])
        lora_scales = lora_ui_config.get('lora_scales', [])

        requested_loras = {} # adapter_name: (lora_filename, scale)
        for i, is_checked in enumerate(lora_checks):
            if is_checked and i < len(lora_dropdowns) and i < len(lora_scales):
                lora_filename = lora_dropdowns[i]
                scale = lora_scales[i]
                if lora_filename and lora_filename != translate("aucun_lora_disponible", self.translations):
                    adapter_name = os.path.splitext(lora_filename)[0].replace(".", "_").replace(" ", "_")
                    requested_loras[adapter_name] = (lora_filename, scale)

        # LORAs à décharger (utilise unload_lora qui appelle unload_lora_weights)
        loras_to_unload = [name for name in self.loaded_loras if name not in requested_loras]
        for name in loras_to_unload:
            self.unload_lora(name)

        # --- SUPPRESSION: La désactivation globale avant chargement est retirée ---
        # On se fie maintenant à unload_lora_weights dans unload_lora

        # LORAs à charger ou mettre à jour
        needs_set_adapters_call = False # Flag pour savoir si on doit appeler set_adapters

        for adapter_name, (lora_filename, scale) in requested_loras.items():
            if adapter_name not in self.loaded_loras:
                # Charger le nouveau LoRA (utilise load_lora qui appelle load_lora_weights)
                success, msg = self.load_lora(lora_filename, scale)
                messages.append(msg)
                if success:
                    needs_set_adapters_call = True # On devra appeler set_adapters à la fin
            elif self.loaded_loras[adapter_name] != scale:
                # Mettre à jour le poids d'un LoRA existant
                self.loaded_loras[adapter_name] = scale # Mettre à jour l'état interne
                messages.append(f"{translate('lora_poids_maj', self.translations)}: {adapter_name} -> {scale}") # Nouvelle clé
                needs_set_adapters_call = True # On devra réappliquer tous les poids
                print(txt_color("[INFO]", "info"), f"Mise à jour poids LoRA {adapter_name} -> {scale}")
            # else: LoRA déjà chargé avec le bon poids, rien à faire

        # --- Appeler set_adapters à la fin pour définir l'état actif ---
        # On l'appelle si on a chargé/déchargé ou changé un poids
        if needs_set_adapters_call or loras_to_unload:
            try:
                active_adapters = list(self.loaded_loras.keys())
                active_weights = [self.loaded_loras[name] for name in active_adapters]

                if active_adapters:
                     print(txt_color("[INFO]", "info"), f"Application des adaptateurs actifs: {active_adapters} avec poids {active_weights}")
                     try:
                         self.current_pipe.set_adapters(active_adapters, adapter_weights=active_weights)
                     except KeyError as e_key: # Intercepter spécifiquement KeyError ('unet')
                         msg = f"Erreur (KeyError) lors de l'application des poids LoRA: {e_key}. L'état interne du pipeline pourrait être incohérent."
                         print(txt_color("[ERREUR]", "erreur"), msg)
                         messages.append(msg)
                     except Exception as e_set: # Autres erreurs
                         msg = f"Erreur lors de l'application des poids LoRA: {e_set}"
                         print(txt_color("[ERREUR]", "erreur"), msg)
                         messages.append(msg)
                else:
                     # S'il n'y a plus de LoRAs actifs, désactiver explicitement
                     print(txt_color("[INFO]", "info"), "Aucun LoRA actif à appliquer, désactivation.")
                     self.current_pipe.set_adapters([], adapter_weights=[])
            except Exception as e_final_set:
                 msg = f"Erreur finale lors de set_adapters: {e_final_set}"
                 print(txt_color("[ERREUR]", "erreur"), msg)
                 messages.append(msg)

        final_message = "\n".join(filter(None, messages))
        return final_message if final_message else translate("loras_geres_succes", self.translations) 