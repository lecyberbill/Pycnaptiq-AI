# d:\image_to_text\cyberbill_SDXL\cyberbill_image_generator\Utils\model_manager.py
import torch
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
    SanaSprintPipeline # <-- AJOUT SANA
)
from compel import Compel, ReturnedEmbeddingsType

# Importer les fonctions utilitaires nécessaires (ajuster si besoin)
from .utils import txt_color, translate, lister_fichiers # Importer lister_fichiers

# Définir les devices ici ou les passer via config/init
cpu = torch.device("cpu")
gpu = torch.device(
    f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
)
SANA_MODEL_TYPE_KEY = "sana_sprint" # <-- AJOUT DEFINITION CONSTANTE


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
        return lister_fichiers(self.loras_dir, self.translations, ext=".safetensors", gradio_mode=gradio_mode)

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
            print(
                txt_color("[INFO]", "info"),
                translate("aucun_modele_a_decharger", self.translations),
            )
            return

        print(
            txt_color("[INFO]", "info"),
            translate("dechargement_modele_en_cours", self.translations), # Nouvelle clé
        )
        if gradio_mode:
            gr.Info(translate("dechargement_modele_en_cours", self.translations), duration=2.0) # Utiliser duration

        try:
            if self.current_pipe is not None:
                self.current_pipe.to(cpu)
                # Supprimer les composants explicitement pour aider GC
                attrs_to_delete = ['vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'scheduler', 'feature_extractor', 'safety_checker']
                for attr in attrs_to_delete:
                    if hasattr(self.current_pipe, attr):
                        try: # Ajouter try/except pour la suppression
                            delattr(self.current_pipe, attr)
                        except Exception as e_del:
                            print(txt_color("[WARN]", "warning"), f"Impossible de supprimer l'attribut {attr}: {e_del}")
                del self.current_pipe
                self.current_pipe = None

            if self.current_compel is not None:
                # Supprimer les références internes de Compel si possible (dépend de l'implémentation)
                # del self.current_compel.tokenizer, self.current_compel.text_encoder # Exemple
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
            if gradio_mode:
                gr.Error(f"{translate('erreur_dechargement_modele', self.translations)}: {e}")


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
        elif model_type == "inpainting":
            pipeline_class = StableDiffusionXLInpaintPipeline
            model_dir = self.inpaint_models_dir
        elif model_type == "img2img":
            pipeline_class = StableDiffusionXLImg2ImgPipeline
            # Utiliser models_dir par défaut pour img2img, car ils utilisent souvent les mêmes checkpoints
            # Si un dossier dédié est configuré, décommenter la ligne suivante
            # model_dir = self.img2img_models_dir
        else: # standard
            pipeline_class = StableDiffusionXLPipeline
            model_dir = self.models_dir

        # Définir le chemin seulement si c'est un fichier local
        if is_from_single_file:
            chemin_modele = os.path.join(model_dir, model_name)
            if not os.path.exists(chemin_modele):
                msg = f"{translate('modele_non_trouve', self.translations)}: {chemin_modele}"
                print(txt_color("[ERREUR]", "erreur"), msg)
                if gradio_mode: gr.Warning(msg, duration=4.0)
                return False, msg
        chemin_vae = os.path.join(self.vae_dir, vae_name) if vae_name and vae_name != "Auto" else None

        # Décharger l'ancien modèle
        self.unload_model(gradio_mode=gradio_mode)

        print(txt_color("[INFO]", "info"), f"{translate('chargement_modele', self.translations)}: {model_name} ({model_type})")
        if gradio_mode: gr.Info(f"{translate('chargement_modele', self.translations)}: {model_name} ({model_type})", duration=3.0)

        try:
            # Charger le pipeline avec gestion d'erreur plus fine
            pipe = None
            try:
                pipeline_kwargs = {
                    "torch_dtype": self.torch_dtype,
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
                else: # Cas Sana Sprint (from_pretrained)
                    # Utiliser bfloat16 si dispo et recommandé par Sana, sinon self.torch_dtype
                    dtype_to_use = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else self.torch_dtype
                    print(f"[INFO] Chargement Sana Sprint avec dtype: {dtype_to_use}")
                    pipe = pipeline_loader( # Utilise SanaSprintPipeline.from_pretrained
                        chemin_modele, # C'est l'ID HF ici
                        torch_dtype=dtype_to_use,
                        # Ajouter d'autres args si Sana le requiert
                    )
            except Exception as e_pipe_load:
                # Erreur spécifique au chargement du pipeline
                raise RuntimeError(f"{translate('erreur_chargement_pipeline', self.translations)}: {e_pipe_load}") # Nouvelle clé

            # Charger le VAE externe si spécifié
            vae_message = ""
            loaded_vae_name = "Auto"
            # Ne charger le VAE externe que si ce n'est PAS Sana Sprint
            if model_type != SANA_MODEL_TYPE_KEY and chemin_vae and os.path.exists(chemin_vae):
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
                    if gradio_mode: gr.Warning(f"{translate('erreur_chargement_vae', self.translations)}: {vae_name} - {e_vae}", duration=4.0)
            elif model_type != SANA_MODEL_TYPE_KEY and chemin_vae: # Chemin spécifié mais non trouvé (et pas Sana)
                 vae_message = f" + VAE: {translate('vae_non_trouve_court', self.translations)}"
                 loaded_vae_name = f"Auto ({translate('non_trouve', self.translations)})"
                 print(txt_color("[ERREUR]", "erreur"), f"{translate('vae_non_trouve', self.translations)}: {chemin_vae}")
                 if gradio_mode: gr.Warning(f"{translate('vae_non_trouve', self.translations)}: {chemin_vae}", duration=4.0)
            elif model_type != SANA_MODEL_TYPE_KEY: # VAE Auto (intégré) (et pas Sana)
                 vae_message = f" + VAE: Auto"
                 print(txt_color("[INFO]", "info"), translate("utilisation_vae_integre", self.translations))

            # Déplacer le pipeline sur le device
            print(txt_color("[INFO]", "info"), f"{translate('deplacement_modele_device', self.translations)} {self.device}...") # Nouvelle clé
            pipe.to(self.device)

            # Appliquer les optimisations mémoire si nécessaire
            # Ne pas appliquer les optimisations VAE pour Sana Sprint
            try:
                if model_type != SANA_MODEL_TYPE_KEY:
                    pipe.enable_vae_slicing()
                    pipe.enable_vae_tiling()
                if self.device.type == "cuda" and self.vram_total_gb < 10:
                     pipe.enable_attention_slicing()
                     print(txt_color("[INFO]", "info"), translate("optimisation_memoire_activee", self.translations)) # Nouvelle clé
                # XFormers (si installé et compatible)
                # --- MODIFICATION: Ne pas activer xformers pour Sana Sprint ---
                if self.device.type == "cuda" and model_type != SANA_MODEL_TYPE_KEY:
                    try: # Ajouter try/except pour xformers
                        pipe.enable_xformers_memory_efficient_attention()
                        print(txt_color("[INFO]", "info"), "XFormers activé.")
                    except ImportError:
                        print(txt_color("[INFO]", "info"), "XFormers non disponible, ignoré.")
                    except Exception as e_xformers:
                        print(txt_color("[AVERTISSEMENT]", "warning"), f"Erreur activation XFormers: {e_xformers}")
                elif model_type == SANA_MODEL_TYPE_KEY:
                    print(txt_color("[INFO]", "info"), "XFormers désactivé pour Sana Sprint.")

            except Exception as e_optim:
                 print(txt_color("[AVERTISSEMENT]", "warning"), f"Erreur application optimisations: {e_optim}") # Utiliser warning


            # Créer Compel
            # Adapter pour Sana si nécessaire (ex: si un seul tokenizer/encoder)
            has_tokenizer_2 = hasattr(pipe, 'tokenizer_2') and pipe.tokenizer_2 is not None
            has_encoder_2 = hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None

            if model_type == SANA_MODEL_TYPE_KEY:
                # Configuration spécifique pour Sana Sprint (probablement pas de pooled)
                compel_returned_embeddings_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED # Ou PENULTIMATE si ça marche mieux pour Sana
                compel_requires_pooled = False
            else:
                # Configuration standard pour SDXL (avec pooled)
                compel_returned_embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                compel_requires_pooled = [False, True] if has_encoder_2 else False

            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2] if has_tokenizer_2 else pipe.tokenizer,
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2] if has_encoder_2 else pipe.text_encoder,
                returned_embeddings_type=compel_returned_embeddings_type,
                requires_pooled=compel_requires_pooled,
                device=self.device # Passer le device explicitement peut aider

            )

            # Mettre à jour l'état interne
            self.current_pipe = pipe
            self.current_compel = compel
            self.current_model_name = model_name
            self.current_vae_name = loaded_vae_name
            self.current_model_type = model_type
            self.loaded_loras.clear() # Effacer les LoRAs lors du changement de modèle
            # Réinitialiser le sampler lors du chargement du modèle
            self.current_sampler_key = None # Ou définir un sampler par défaut ici

            final_message = f"{translate('modele_charge', self.translations)}: {model_name}{vae_message}"
            print(txt_color("[OK]", "ok"), final_message)
            if gradio_mode: gr.Info(final_message, duration=3.0)
            return True, final_message

        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_generale_chargement_modele', self.translations)}: {e}") # Nouvelle clé plus générique
            traceback.print_exc()
            self.unload_model() # Assurer le nettoyage en cas d'erreur
            error_msg = f"{translate('erreur_chargement_modele', self.translations)}: {e}"
            if gradio_mode: gr.Error(error_msg)
            return False, error_msg

    def get_current_pipe(self):
        """Retourne le pipeline actuellement chargé."""
        return self.current_pipe

    def get_current_compel(self):
        """Retourne l'instance Compel actuelle."""
        return self.current_compel

    # --- Méthodes pour LoRA ---

    def load_lora(self, lora_name, scale):
        """Charge un LoRA spécifique."""
        if self.current_pipe is None:
            msg = translate("erreur_pas_modele_pour_lora", self.translations) # Nouvelle clé
            print(txt_color("[ERREUR]", "erreur"), msg)
            return False, msg

        lora_path = os.path.join(self.loras_dir, lora_name)
        if not os.path.exists(lora_path):
            msg = f"{translate('erreur_fichier_lora', self.translations)}: {lora_path}"
            print(txt_color("[ERREUR]", "erreur"), msg)
            return False, msg

        # Créer un nom d'adaptateur unique et valide
        adapter_name = os.path.splitext(lora_name)[0].replace(".", "_").replace(" ", "_")

        try:
            print(txt_color("[INFO]", "info"), f"{translate('lora_charge_depuis', self.translations)} {lora_path}")

            # Charger les poids du LoRA. Diffusers gère l'ajout à la config PEFT.
            self.current_pipe.load_lora_weights(self.loras_dir, weight_name=lora_name, adapter_name=adapter_name)
            self.loaded_loras[adapter_name] = scale # Mettre à jour l'état interne
            msg = f"{translate('lora_charge', self.translations)}: {adapter_name} (Scale: {scale})"
            print(txt_color("[OK]", "ok"), msg)
            return True, msg
        except Exception as e:
            msg = f"{translate('erreur_lora_chargement', self.translations)}: {e}"
            print(txt_color("[ERREUR]", "erreur"), msg)
            traceback.print_exc()
            # Essayer de décharger si le chargement a échoué à mi-chemin
            try:
                # Vérifier si l'adaptateur existe avant de tenter de le supprimer
                if hasattr(self.current_pipe, 'delete_adapters') and adapter_name in self.current_pipe.peft_config:
                    self.unload_lora(adapter_name) # Tenter de nettoyer
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
        return final_message if final_message else translate("loras_geres_succes", self.translations) # Nouvelle clé
