import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from torch.hub import download_url_to_file
from enum import Enum
import time
import json # For loading module JSON
import threading
import gc
import traceback


from Utils.utils import (
    txt_color,
    translate,
    enregistrer_image,
    preparer_metadonnees_image,
    enregistrer_etiquettes_image_html,
    create_progress_bar_html,
    ImageSDXLchecker
)
from Utils.model_manager import ModelManager
from Utils.utils import GestionModule
from core.translator import translate_prompt

try:
    from modules.modules_utils.briarmbg import BriaRMBG
except ImportError as e:
    print(txt_color("[ERREUR CRITIQUE]", "erreur"),
          f"Impossible d'importer BriaRMBG depuis 'modules.modules_utils.briarmbg'. "
          f"Vérifiez que briarmbg.py se trouve bien dans 'modules/modules_utils' et que la structure du projet est correcte. Erreur: {e}")
    raise


# --- Module Metadata Loading ---
MODULE_NAME = "reLighting"
module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

try:
    with open(module_json_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)
except FileNotFoundError:
    print(f"[ERREUR] Fichier JSON du module {MODULE_NAME} introuvable: {module_json_path}")
    module_data = {"name": MODULE_NAME, "language": {"fr": {}}} # Fallback
except json.JSONDecodeError:
    print(f"[ERREUR] Erreur de décodage JSON pour le module {MODULE_NAME}.")
    module_data = {"name": MODULE_NAME, "language": {"fr": {}}} # Fallback


# --- Initialization Function ---
def initialize(global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
    """Initializes the ReLighting module."""
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME)}")
    return ReLightingModule(global_translations, model_manager_instance, gestionnaire_instance, global_config)


class BGSourceFC(Enum):
    NONE_FC = "None (FC)"
    LEFT_FC = "Left Light (FC)"
    RIGHT_FC = "Right Light (FC)"
    TOP_FC = "Top Light (FC)"
    BOTTOM_FC = "Bottom Light (FC)"

    def get_translated_value(self, translations):
        key_map = {
            "None (FC)": "bg_source_none_fc",
            "Left Light (FC)": "bg_source_left_fc",
            "Right Light (FC)": "bg_source_right_fc",
            "Top Light (FC)": "bg_source_top_fc",
            "Bottom Light (FC)": "bg_source_bottom_fc",
        }
        return translate(key_map.get(self.value, self.value), translations)

class BGSourceFBC(Enum):
    UPLOAD_FBC = "Upload Background (FBC)"
    UPLOAD_FLIP_FBC = "Upload Flipped Background (FBC)"
    LEFT_FBC = "Left Light (FBC)"
    RIGHT_FBC = "Right Light (FBC)"
    TOP_FBC = "Top Light (FBC)"
    BOTTOM_FBC = "Bottom Light (FBC)"
    GREY_FBC = "Ambient Grey (FBC)"

    def get_translated_value(self, translations):
        key_map = {
            "Upload Background (FBC)": "bg_source_upload_fbc",
            "Upload Flipped Background (FBC)": "bg_source_upload_flip_fbc",
            "Left Light (FBC)": "bg_source_left_fbc",
            "Right Light (FBC)": "bg_source_right_fbc",
            "Top Light (FBC)": "bg_source_top_fbc",
            "Bottom Light (FBC)": "bg_source_bottom_fbc",
            "Ambient Grey (FBC)": "bg_source_grey_fbc",
        }
        return translate(key_map.get(self.value, self.value), translations)


class ReLightingModule:
    def __init__(self, global_translations, model_manager_instance: ModelManager, gestionnaire_instance: GestionModule, global_config=None):
        self.global_translations = global_translations
        self.model_manager = model_manager_instance
        self.gestionnaire = gestionnaire_instance
        self.global_config = global_config
        self.module_translations = {}

        self.pipe_t2i = None
        self.pipe_i2i = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.tokenizer = None
        self.rmbg = None
        self.device = model_manager_instance.device if model_manager_instance else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stop_event = threading.Event()
        
        self.models_loaded = False
        self.current_relight_mode = "FC" # Default mode: "FC" or "FBC"

        self.sd15_name = 'stablediffusionapi/realistic-vision-v51'
        self.ic_light_model_dir = self.global_config.get("IC_LIGHT_MODEL_DIR", "./models")
        os.makedirs(self.ic_light_model_dir, exist_ok=True)
        self.ic_light_fc_model_path = os.path.join(self.ic_light_model_dir, 'iclight_sd15_fc.safetensors')
        self.ic_light_fbc_model_path = os.path.join(self.ic_light_model_dir, 'iclight_sd15_fbc.safetensors')
        self.default_negative_prompt = self.global_config.get("NEGATIVE_PROMPT", "lowres, bad anatomy, bad hands, cropped, worst quality")

        self.NO_CHOICE_KEY = "aucun_choix_relight"
        self.quick_prompts_data = [
            self.NO_CHOICE_KEY, 'sunshine_from_window_key', 'neon_light_city_key', 'sunset_over_sea_key', 'golden_time_key',
            'scifi_rgb_cyberpunk_key', 'natural_lighting_key', 'warm_atmosphere_home_bedroom_key',
            'magic_lit_key', 'evil_gothic_yharnam_key', 'light_and_shadow_key', 'shadow_from_window_key',
            'soft_studio_lighting_key', 'home_atmosphere_cozy_bedroom_key', 'neon_wongkarwai_warm_key',
            'dramatic_lighting_key', 'moonlight_key', 'forest_at_night_key', 'underwater_lighting_key', 'fireworks_key'
        ]
        self.quick_subjects_data = [
            self.NO_CHOICE_KEY, 'beautiful_woman_face_key', 'handsome_man_face_key', 'cute_cat_key', 'majestic_dog_key',
            'mythical_creature_key', 'futuristic_robot_key', 'ancient_warrior_key', 'serene_landscape_key',
            'bustling_cityscape_key', 'abstract_concept_key'
        ]

    def _load_models(self, mode_to_load):
        if self.models_loaded and self.current_relight_mode == mode_to_load:
            success_msg = translate("models_already_loaded_for_mode", self.module_translations).format(mode=mode_to_load)
            yield gr.update(value=success_msg, interactive=False), gr.update(interactive=True)
            return

        if self.model_manager.get_current_pipe() is not None:
            unload_msg = translate("unloading_main_model_before_relight", self.module_translations)
            print(txt_color("[INFO]", "info"), unload_msg)
            yield gr.update(value=unload_msg, interactive=False), gr.update(interactive=False)
            self.model_manager.unload_model(gradio_mode=False)
            print(txt_color("[OK]", "ok"), translate("main_model_unloaded_relight", self.module_translations))

        loading_msg_formatted = translate("model_loading_for_mode_wait", self.module_translations).format(mode=mode_to_load)
        print(txt_color("[INFO]", "info"), loading_msg_formatted)
        yield gr.update(value=loading_msg_formatted, interactive=False), gr.update(interactive=False)
        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.sd15_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.sd15_name, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(self.sd15_name, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(self.sd15_name, subfolder="unet")
            self.rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

            num_input_channels_unet = 0
            ic_light_model_to_load_path = ""

            if mode_to_load == "FC":
                num_input_channels_unet = 8
                ic_light_model_to_load_path = self.ic_light_fc_model_path
                if not os.path.exists(ic_light_model_to_load_path):
                    print(f"Downloading IC-Light FC model to {ic_light_model_to_load_path}...")
                    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=ic_light_model_to_load_path)
            elif mode_to_load == "FBC":
                num_input_channels_unet = 12
                ic_light_model_to_load_path = self.ic_light_fbc_model_path
                if not os.path.exists(ic_light_model_to_load_path):
                    print(f"Downloading IC-Light FBC model to {ic_light_model_to_load_path}...")
                    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=ic_light_model_to_load_path)
            else:
                raise ValueError(f"Unknown relight mode: {mode_to_load}")

            with torch.no_grad():
                new_conv_in = torch.nn.Conv2d(num_input_channels_unet, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding)
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
                new_conv_in.bias = self.unet.conv_in.bias
                self.unet.conv_in = new_conv_in

            self.unet_original_forward = self.unet.forward
            self.unet.forward = self._hooked_unet_forward

            sd_offset = sf.load_file(ic_light_model_to_load_path)
            sd_origin = self.unet.state_dict()
            keys = sd_origin.keys()
            sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
            self.unet.load_state_dict(sd_merged, strict=True)
            del sd_offset, sd_origin, sd_merged, keys

            self.text_encoder = self.text_encoder.to(device=self.device, dtype=torch.float16)
            self.vae = self.vae.to(device=self.device, dtype=torch.bfloat16)
            self.unet = self.unet.to(device=self.device, dtype=torch.float16)
            self.rmbg = self.rmbg.to(device=self.device, dtype=torch.float32)

            self.unet.set_attn_processor(AttnProcessor2_0())
            self.vae.set_attn_processor(AttnProcessor2_0())

            dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1
            )

            self.pipe_t2i = StableDiffusionPipeline(
                vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                unet=self.unet, scheduler=dpmpp_2m_sde_karras_scheduler,
                safety_checker=None, requires_safety_checker=False,
                feature_extractor=None, image_encoder=None
            )
            self.pipe_i2i = StableDiffusionImg2ImgPipeline(
                vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                unet=self.unet, scheduler=dpmpp_2m_sde_karras_scheduler,
                safety_checker=None, requires_safety_checker=False,
                feature_extractor=None, image_encoder=None
            )
            self.models_loaded = True
            self.current_relight_mode = mode_to_load
            success_msg = translate("models_loaded_successfully_for_mode", self.module_translations).format(mode=mode_to_load)
            print(txt_color("[OK]", "ok"), success_msg)
            yield gr.update(value=success_msg, interactive=False), gr.update(interactive=True)
        except Exception as e:
            self.models_loaded = False
            error_msg = f"{translate('error_loading_models_for_mode', self.module_translations).format(mode=mode_to_load)}: {e}"
            print(txt_color("[ERREUR]", "erreur"), error_msg)
            traceback.print_exc()
            yield gr.update(value=error_msg, interactive=True), gr.update(interactive=False)
    
    def _hooked_unet_forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return self.unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    @torch.inference_mode()
    def _encode_prompt_inner(self, txt: str):
        max_length = self.tokenizer.model_max_length
        chunk_length = self.tokenizer.model_max_length - 2
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        id_pad = id_end
        def pad(x, p, i): return x[:i] if len(x) >= i else x + [p] * (i - len(x))
        tokens = self.tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]
        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = self.text_encoder(token_ids).last_hidden_state
        return conds

    @torch.inference_mode()
    def _encode_prompt_pair(self, positive_prompt, negative_prompt):
        c = self._encode_prompt_inner(positive_prompt)
        uc = self._encode_prompt_inner(negative_prompt)
        c_len, uc_len = float(len(c)), float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat, uc_repeat = int(math.ceil(max_count / c_len)), int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))
        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]
        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)
        return c, uc

    @torch.inference_mode()
    def _pytorch2numpy(self, imgs, quant=True):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)
            if quant:
                y = y * 127.5 + 127.5
                y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
            else:
                y = y * 0.5 + 0.5
                y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)
            results.append(y)
        return results

    @torch.inference_mode()
    def _numpy2pytorch(self, imgs):
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
        h = h.movedim(-1, 1)
        return h

    def _resize_and_center_crop(self, image, target_width, target_height):
        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        scale_factor = max(target_width / original_width, target_height / original_height)
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        cropped_image = resized_image.crop((left, top, right, bottom))
        return np.array(cropped_image)

    def _resize_without_crop(self, image, target_width, target_height):
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        return np.array(resized_image)

    @torch.inference_mode()
    def _run_rmbg(self, img_np, sigma=0.0):
        H, W, C = img_np.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = self._resize_without_crop(img_np, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = self._numpy2pytorch([feed]).to(device=self.device, dtype=torch.float32)
        alpha_outputs, _ = self.rmbg(feed)
        alpha_mask_tensor = alpha_outputs[0]
        if alpha_mask_tensor.ndim == 3: alpha_mask_tensor = alpha_mask_tensor.unsqueeze(0)
        elif alpha_mask_tensor.ndim == 2: alpha_mask_tensor = alpha_mask_tensor.unsqueeze(0).unsqueeze(0)
        alpha = torch.nn.functional.interpolate(alpha_mask_tensor, size=(H, W), mode="bilinear", align_corners=False)
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (img_np.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha

    @torch.inference_mode()
    def _process_internal(self, current_relight_mode,
                          input_fg_np, prompt, traduire_prompt, 
                          image_width, image_height, num_samples, seed, steps, a_prompt, cfg, 
                          highres_scale, highres_denoise, 
                          fc_lowres_denoise, fc_bg_source_str,
                          fbc_input_bg_ui_np, fbc_bg_source_str
                          ):
        final_prompt = prompt
        if traduire_prompt:
            try:
                final_prompt = translate_prompt(prompt, self.module_translations)
                print(f"{txt_color('[INFO]', 'info')} Prompt original: {prompt} -> Traduit: {final_prompt}")
            except Exception as e_translate:
                print(txt_color("[ERREUR]", "erreur"), f"Échec traduction prompt: {e_translate}")
                gr.Warning(f"Échec de la traduction du prompt : {e_translate}. Utilisation du prompt original.")

        actual_seed_used = int(seed)
        if actual_seed_used == -1:
            actual_seed_used = torch.randint(0, 2**32 - 1, (1,)).item()
            print(txt_color("[INFO]", "info"), f"Seed -1 détectée, utilisation d'une seed aléatoire: {actual_seed_used}")
        rng = torch.Generator(device=self.device).manual_seed(actual_seed_used)

        conds, unconds = self._encode_prompt_pair(positive_prompt=final_prompt + ', ' + a_prompt, negative_prompt=self.default_negative_prompt)
        fg = self._resize_and_center_crop(input_fg_np, image_width, image_height)
        
        unet_concat_conds = None
        latents = None

        if current_relight_mode == "FC":
            fc_bg_source_enum_val = next((item for item in BGSourceFC if item.get_translated_value(self.module_translations) == fc_bg_source_str), BGSourceFC.NONE_FC)
            fc_input_bg_for_initial_latent = None
            if fc_bg_source_enum_val == BGSourceFC.LEFT_FC:
                gradient = np.linspace(255, 0, image_width); image = np.tile(gradient, (image_height, 1))
                fc_input_bg_for_initial_latent = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            elif fc_bg_source_enum_val == BGSourceFC.RIGHT_FC:
                gradient = np.linspace(0, 255, image_width); image = np.tile(gradient, (image_height, 1))
                fc_input_bg_for_initial_latent = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            elif fc_bg_source_enum_val == BGSourceFC.TOP_FC:
                gradient = np.linspace(255, 0, image_height)[:, None]; image = np.tile(gradient, (1, image_width))
                fc_input_bg_for_initial_latent = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            elif fc_bg_source_enum_val == BGSourceFC.BOTTOM_FC:
                gradient = np.linspace(0, 255, image_height)[:, None]; image = np.tile(gradient, (1, image_width))
                fc_input_bg_for_initial_latent = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            unet_concat_conds = self.vae.encode(self._numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor

            if fc_input_bg_for_initial_latent is None:
                latents = self.pipe_t2i(
                    prompt_embeds=conds, negative_prompt_embeds=unconds, width=image_width, height=image_height,
                    num_inference_steps=steps, num_images_per_prompt=num_samples, generator=rng, output_type='latent',
                    guidance_scale=cfg, cross_attention_kwargs={'concat_conds': unet_concat_conds},
                ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
            else:
                bg_for_latent = self._resize_and_center_crop(fc_input_bg_for_initial_latent, image_width, image_height)
                bg_latent_i2i = self.vae.encode(self._numpy2pytorch([bg_for_latent]).to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor
                latents = self.pipe_i2i(
                    image=bg_latent_i2i, strength=fc_lowres_denoise, prompt_embeds=conds, negative_prompt_embeds=unconds,
                    width=image_width, height=image_height, num_inference_steps=int(round(steps / fc_lowres_denoise)),
                    num_images_per_prompt=num_samples, generator=rng, output_type='latent', guidance_scale=cfg,
                    cross_attention_kwargs={'concat_conds': unet_concat_conds},
                ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
        
        elif current_relight_mode == "FBC":
            fbc_bg_source_enum_val = next((item for item in BGSourceFBC if item.get_translated_value(self.module_translations) == fbc_bg_source_str), BGSourceFBC.GREY_FBC)
            actual_fbc_bg_cond_np = None
            if fbc_bg_source_enum_val == BGSourceFBC.UPLOAD_FBC:
                if fbc_input_bg_ui_np is None: raise ValueError(translate("error_no_fbc_bg_image", self.module_translations))
                actual_fbc_bg_cond_np = fbc_input_bg_ui_np
            elif fbc_bg_source_enum_val == BGSourceFBC.UPLOAD_FLIP_FBC:
                if fbc_input_bg_ui_np is None: raise ValueError(translate("error_no_fbc_bg_image", self.module_translations))
                actual_fbc_bg_cond_np = np.fliplr(fbc_input_bg_ui_np)
            elif fbc_bg_source_enum_val == BGSourceFBC.GREY_FBC:
                actual_fbc_bg_cond_np = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 127
            elif fbc_bg_source_enum_val == BGSourceFBC.LEFT_FBC:
                gradient = np.linspace(224, 32, image_width); image = np.tile(gradient, (image_height, 1)) # From FBC script
                actual_fbc_bg_cond_np = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            elif fbc_bg_source_enum_val == BGSourceFBC.RIGHT_FBC:
                gradient = np.linspace(32, 224, image_width); image = np.tile(gradient, (image_height, 1))
                actual_fbc_bg_cond_np = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            elif fbc_bg_source_enum_val == BGSourceFBC.TOP_FBC:
                gradient = np.linspace(224, 32, image_height)[:, None]; image = np.tile(gradient, (1, image_width))
                actual_fbc_bg_cond_np = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            elif fbc_bg_source_enum_val == BGSourceFBC.BOTTOM_FBC:
                gradient = np.linspace(32, 224, image_height)[:, None]; image = np.tile(gradient, (1, image_width))
                actual_fbc_bg_cond_np = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported FBC background source: {fbc_bg_source_enum_val}")

            bg_fbc = self._resize_and_center_crop(actual_fbc_bg_cond_np, image_width, image_height)
            fg_latent_pt = self.vae.encode(self._numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor
            bg_latent_pt = self.vae.encode(self._numpy2pytorch([bg_fbc]).to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor
            unet_concat_conds = torch.cat([fg_latent_pt, bg_latent_pt], dim=1)

            latents = self.pipe_t2i(
                prompt_embeds=conds, negative_prompt_embeds=unconds, width=image_width, height=image_height,
                num_inference_steps=steps, num_images_per_prompt=num_samples, generator=rng, output_type='latent',
                guidance_scale=cfg, cross_attention_kwargs={'concat_conds': unet_concat_conds},
            ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
        else:
            raise ValueError(f"Unknown relight_mode in _process_internal: {current_relight_mode}")

        # --- Common High-Res Pass Preparation ---
        pixels_lowres = self.vae.decode(latents).sample
        pixels_lowres_np = self._pytorch2numpy(pixels_lowres)
        pixels_resized_for_hr = [self._resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64))
        for p in pixels_lowres_np]

        pixels_pt_hr = self._numpy2pytorch(pixels_resized_for_hr).to(device=self.vae.device, dtype=self.vae.dtype)
        latents_for_hr = self.vae.encode(pixels_pt_hr).latent_dist.mode() * self.vae.config.scaling_factor
        latents_for_hr = latents_for_hr.to(device=self.unet.device, dtype=self.unet.dtype)
        
        image_height_hr, image_width_hr = latents_for_hr.shape[2] * 8, latents_for_hr.shape[3] * 8
        
        unet_concat_conds_hr = None
        if current_relight_mode == "FC":
            fg_hr = self._resize_and_center_crop(input_fg_np, image_width_hr, image_height_hr)
            unet_concat_conds_hr = self.vae.encode(self._numpy2pytorch([fg_hr]).to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor
        elif current_relight_mode == "FBC":
            fg_hr = self._resize_and_center_crop(input_fg_np, image_width_hr, image_height_hr)
            # actual_fbc_bg_cond_np should be defined from the FBC block above
            bg_fbc_hr = self._resize_and_center_crop(actual_fbc_bg_cond_np, image_width_hr, image_height_hr)
            fg_latent_pt_hr = self.vae.encode(self._numpy2pytorch([fg_hr]).to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor
            bg_latent_pt_hr = self.vae.encode(self._numpy2pytorch([bg_fbc_hr]).to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor
            unet_concat_conds_hr = torch.cat([fg_latent_pt_hr, bg_latent_pt_hr], dim=1)
        
        # --- Common High-Res Img2Img Pass ---
        latents_final_hr = self.pipe_i2i(
            image=latents_for_hr, strength=highres_denoise, prompt_embeds=conds, negative_prompt_embeds=unconds,
            width=image_width_hr, height=image_height_hr, num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples, generator=rng, output_type='latent', guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': unet_concat_conds_hr},
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        pixels_final = self.vae.decode(latents_final_hr).sample
        return self._pytorch2numpy(pixels_final), actual_seed_used

    def preprocess_input_image(self, input_fg_np):
        # Déterminer l'interactivité de fbc_input_bg_ui en fonction de la présence de input_fg_np
        # et si le mode FBC est actuellement sélectionné (ce dernier sera géré par update_ui_for_mode)
        fbc_bg_interactive_update = gr.update(interactive=(input_fg_np is not None))

        if input_fg_np is None:
            # Si pas d'image sujet, le bouton relight est off, load_models est on (si pas chargés), et fbc_bg est off
            return None, gr.update(interactive=False), gr.update(interactive=not self.models_loaded), fbc_bg_interactive_update

        image_for_preview_pil = Image.fromarray(input_fg_np) # Commencer avec l'image originale

        try:
            if self.models_loaded and self.rmbg is not None:
                gr.Info(translate("preprocessing_input_image_start", self.module_translations))
                processed_fg_np_after_rmbg, _ = self._run_rmbg(input_fg_np) # RMBG sur l'image originale
                if processed_fg_np_after_rmbg is not None:
                    image_for_preview_pil = Image.fromarray(processed_fg_np_after_rmbg)
                    gr.Info(translate("preprocessing_input_image_done", self.module_translations))
                else:
                    gr.Error(translate("error_rmbg_failed", self.module_translations))
                    # En cas d'échec de RMBG, image_for_preview_pil reste l'originale.
            elif not self.models_loaded:
                gr.Info(translate("models_not_loaded_preview_no_rmbg", self.module_translations))
            elif self.models_loaded and self.rmbg is None: # Cas d'erreur où les modèles sont chargés mais rmbg ne l'est pas
                gr.Error(translate("error_rmbg_model_not_initialized_unexpected", self.module_translations))
                # image_for_preview_pil reste l'originale

            # Redimensionnement unique pour la prévisualisation finale
            max_pixels_val = self.global_config.get("MAX_IMAGE_PIXELS_RELIGHTING", 1024*1024*1.5)
            if not isinstance(max_pixels_val, (int, float)): max_pixels_val = 1024*1024*1.5
            checker = ImageSDXLchecker(image_for_preview_pil, self.module_translations, max_pixels=max_pixels_val)
            final_preview_np = np.array(checker.redimensionner_image()) # Un seul redimensionnement pour la preview ici

            relight_btn_interactive = self.models_loaded and (self.current_relight_mode is not None) and (input_fg_np is not None)
            load_btn_interactive = not self.models_loaded # Le bouton de chargement est interactif si les modèles ne sont pas chargés

            return final_preview_np, gr.update(interactive=relight_btn_interactive), gr.update(interactive=load_btn_interactive), fbc_bg_interactive_update

        except Exception as e:
            gr.Error(f"{translate('error_converting_to_pil', self.module_translations)}: {e}")
            return None, gr.update(interactive=False), gr.update(interactive=not self.models_loaded), gr.update(interactive=False)

    def process_relight_wrapper(self, 
                                relight_mode_selected_str, input_fg_np, prompt, traduire_prompt_checkbox_value, 
                                num_samples, seed, steps, a_prompt, cfg, 
                                highres_scale, highres_denoise,
                                fc_lowres_denoise_slider_val, fc_bg_source_radio_val,
                                fbc_input_bg_ui_np_val, fbc_bg_source_radio_val
                                ):
        self.stop_event.clear()
        start_time = time.time()
        
        # Determine actual mode key ("FC" or "FBC") from translated UI string
        relight_mode_key = "FC" # Default
        if relight_mode_selected_str == translate("relight_mode_fbc_label", self.module_translations):
            relight_mode_key = "FBC"

        yield None, [], gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)

        if not self.models_loaded or self.current_relight_mode != relight_mode_key:
            gr.Warning(translate("error_models_not_loaded_for_mode_try_load", self.module_translations).format(mode=relight_mode_key))
            load_gen = self._load_models(relight_mode_key)
            try:
                while True: next(load_gen)
            except StopIteration: pass
            if not self.models_loaded or self.current_relight_mode != relight_mode_key: # Check again
                yield None, [], gr.update(interactive=True), gr.update(interactive=False), gr.update(value=translate("error_loading_models_for_mode", self.module_translations).format(mode=relight_mode_key), interactive=True)
                return
        
        if input_fg_np is None:
            gr.Warning(translate("error_no_input_image_relight", self.module_translations))
            yield None, [], gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=not self.models_loaded)
            return

        if relight_mode_key == "FBC":
            fbc_bg_enum_val = next((item for item in BGSourceFBC if item.get_translated_value(self.module_translations) == fbc_bg_source_radio_val), None)
            if fbc_bg_enum_val in [BGSourceFBC.UPLOAD_FBC, BGSourceFBC.UPLOAD_FLIP_FBC] and fbc_input_bg_ui_np_val is None:
                gr.Warning(translate("error_no_fbc_bg_image_for_upload_mode", self.module_translations))
                yield None, [], gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=not self.models_loaded)
                return
        
        processed_fg_np, _ = self._run_rmbg(input_fg_np)
        if processed_fg_np is None:
            gr.Error(translate("error_rmbg_failed", self.module_translations))
            yield None, [], gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=not self.models_loaded)
            return

        output_gallery_images = []
        try:
            pil_processed_fg = Image.fromarray(processed_fg_np)
            max_pixels_val = self.global_config.get("MAX_IMAGE_PIXELS_RELIGHTING", 1024*1024*1.5)
            if not isinstance(max_pixels_val, (int, float)): max_pixels_val = 1024*1024*1.5
            
            original_w, original_h = pil_processed_fg.size
            checker = ImageSDXLchecker(pil_processed_fg, self.module_translations, max_pixels=max_pixels_val)
            checked_image_pil = checker.redimensionner_image()
            checked_w, checked_h = checked_image_pil.size
            if (original_w, original_h) != (checked_w, checked_h):
                print(txt_color("[INFO]", "info"), translate("redimensionnement_image", self.module_translations).format(original_w, original_h, original_w*original_h, checked_w, checked_h))

            image_width = max(256, int(round(checked_w / 64.0) * 64))
            image_height = max(256, int(round(checked_h / 64.0) * 64))
            print(f"[INFO] Relighting with target dimensions: {image_width}x{image_height}")

            if self.stop_event.is_set(): raise InterruptedError(translate("generation_stopped_by_user", self.module_translations))

            output_gallery_images, final_seed_used = self._process_internal(
                current_relight_mode=relight_mode_key,
                input_fg_np=processed_fg_np, prompt=prompt, traduire_prompt=traduire_prompt_checkbox_value,
                image_width=image_width, image_height=image_height, num_samples=num_samples, seed=seed, steps=steps,
                a_prompt=a_prompt, cfg=cfg, highres_scale=highres_scale, highres_denoise=highres_denoise,
                fc_lowres_denoise=fc_lowres_denoise_slider_val, fc_bg_source_str=fc_bg_source_radio_val,
                fbc_input_bg_ui_np=fbc_input_bg_ui_np_val, fbc_bg_source_str=fbc_bg_source_radio_val
            )
            if self.stop_event.is_set(): raise InterruptedError(translate("generation_stopped_by_user", self.module_translations))

            for i, img_data_np in enumerate(output_gallery_images):
                img_to_save = Image.fromarray(img_data_np)
                current_time_str = time.strftime("%Y%m%d_%H%M%S")
                # Récupérer le format d'image depuis la configuration globale
                image_format = self.global_config.get("IMAGE_FORMAT", "PNG").lower()
                if not image_format: # Fallback si la clé existe mais est vide
                    image_format = "png"
                filename = f"relighted_{relight_mode_key.lower()}_{current_time_str}_{i}.{image_format}"
                save_dir_base = self.global_config.get("SAVE_DIR", "Output")
                date_str = time.strftime("%Y_%m_%d")
                save_dir = os.path.join(save_dir_base, date_str)
                os.makedirs(save_dir, exist_ok=True)
                image_path = os.path.join(save_dir, filename)
                
                xmp_data = {
                    "Module": f"ReLighting IC-Light ({relight_mode_key})",
                    "Creator": self.global_config.get("AUTHOR", "CyberBill"),
                    "Prompt": prompt,
                    "EffectivePrompt": (translate_prompt(prompt, self.module_translations) if traduire_prompt_checkbox_value else prompt) + ', ' + a_prompt,
                    "NegativePrompt": self.default_negative_prompt, "Seed": final_seed_used, "Steps": steps, "CFG": cfg,
                    "BG_Source_FC": fc_bg_source_radio_val if relight_mode_key == "FC" else "N/A",
                    "BG_Source_FBC": fbc_bg_source_radio_val if relight_mode_key == "FBC" else "N/A",
                    "Size": f"{img_to_save.width}x{img_to_save.height}",
                    "GenerationTime": f"{(time.time() - start_time):.2f} sec" 
                }
                metadata_structure, prep_message = preparer_metadonnees_image(img_to_save, xmp_data, self.global_translations, image_path)
                print(txt_color("[INFO]", "info"), prep_message)
                enregistrer_image(img_to_save, image_path, self.global_translations, image_format.upper(), metadata_to_save=metadata_structure)
                enregistrer_etiquettes_image_html(image_path, xmp_data, self.module_translations, is_last_image=(i == len(output_gallery_images) -1))

            elapsed_time = time.time() - start_time
            final_message = translate("relight_complete", self.module_translations).format(time=f"{elapsed_time:.2f}")
            gr.Info(final_message)
            yield processed_fg_np, output_gallery_images, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=not self.models_loaded)

        except InterruptedError as e:
            print(txt_color("[INFO]", "info"), str(e))
            yield processed_fg_np, output_gallery_images, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=not self.models_loaded)
        except Exception as e:
            error_msg = f"{translate('error_processing_relight', self.module_translations)}: {e}"
            print(txt_color("[ERREUR]", "erreur"), error_msg)
            traceback.print_exc()
            yield processed_fg_np, output_gallery_images, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=not self.models_loaded)
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def stop_generation(self):
        self.stop_event.set()
        print(txt_color("[INFO]", "info"), translate("generation_stopped_by_user", self.module_translations))

    def create_tab(self, module_translations_arg):
        self.module_translations = module_translations_arg
        translated_no_choice = translate(self.NO_CHOICE_KEY, self.module_translations)
        quick_prompts_choices = [translate(p_key, self.module_translations) if p_key != self.NO_CHOICE_KEY else translated_no_choice for p_key in self.quick_prompts_data]
        quick_subjects_choices = [translate(s_key, self.module_translations) if s_key != self.NO_CHOICE_KEY else translated_no_choice for s_key in self.quick_subjects_data]
        
        fc_bg_source_choices = [e.get_translated_value(self.module_translations) for e in BGSourceFC]
        fc_default_bg_source = BGSourceFC.NONE_FC.get_translated_value(self.module_translations)
        fbc_bg_source_choices = [e.get_translated_value(self.module_translations) for e in BGSourceFBC]
        fbc_default_bg_source = BGSourceFBC.UPLOAD_FBC.get_translated_value(self.module_translations)

        with gr.Tab(translate("relight_tab_name", self.module_translations)) as tab:
            current_relight_mode_state = gr.State(self.current_relight_mode) # Stores "FC" or "FBC"
            gr.Markdown(f"## {translate('relight_tab_title', self.module_translations)}")
            
            with gr.Row():
                # --- Colonne de Gauche (Entrées Images) ---
                with gr.Column(scale=1):
                    input_fg = gr.Image(type="numpy", label=translate("input_foreground_label", self.module_translations), height=400)
                    output_preview_fg = gr.Image(type="numpy", label=translate("output_preprocessed_label", self.module_translations), height=400, interactive=False)
                    
                    # Group for FBC background inputs (image upload)
                    with gr.Group(visible=(self.current_relight_mode == "FBC")) as fbc_input_image_group_ui_visible_state: # Renamed for clarity
                        fbc_input_bg_ui = gr.Image(type="numpy", label=translate("fbc_background_input_label", self.module_translations), height=240, interactive=False) # Initialement non interactif

                # --- Colonne Centrale (Paramètres) ---
                with gr.Column(scale=2):
                    relight_mode_radio = gr.Radio(
                        choices=[translate("relight_mode_fc_label", self.module_translations), translate("relight_mode_fbc_label", self.module_translations)],
                        value=translate(f"relight_mode_{self.current_relight_mode.lower()}_label", self.module_translations),
                        label=translate("relight_mode_selector_label", self.module_translations)
                    )
                    prompt_textbox = gr.Textbox(label=translate("prompt_label", self.module_translations), lines=2)
                    traduire_prompt_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", self.module_translations), value=False, info=translate("traduire_prompt_libre", self.module_translations))
                    
                    with gr.Group(visible=(self.current_relight_mode == "FC")) as fc_specific_options_group_ui_visible_state: # Renamed for clarity
                        fc_bg_source_radio = gr.Radio(choices=fc_bg_source_choices, value=fc_default_bg_source, label=translate("fc_lighting_preference_label", self.module_translations), type='value')
                    
                    # Group for FBC background source options (radio buttons)
                    with gr.Group(visible=(self.current_relight_mode == "FBC")) as fbc_bg_source_options_group_ui_visible_state:
                        fbc_bg_source_radio = gr.Radio(choices=fbc_bg_source_choices, value=fbc_default_bg_source, label=translate("fbc_background_source_label", self.module_translations), type='value')

                    subject_dropdown = gr.Dropdown(choices=quick_subjects_choices, value=translated_no_choice, label=translate("subject_quick_list_label", self.module_translations), allow_custom_value=False)
                    lighting_dropdown = gr.Dropdown(choices=quick_prompts_choices, value=translated_no_choice, label=translate("lighting_quick_list_label", self.module_translations), allow_custom_value=False)
                    
                    with gr.Row():
                        load_models_button = gr.Button(value=translate("sana_load_button", self.module_translations), interactive=not self.models_loaded)
                        relight_button = gr.Button(value=translate("relight_button", self.module_translations), variant="primary", interactive=self.models_loaded)
                        stop_button = gr.Button(value=translate("stop_button", self.module_translations), variant="stop", interactive=False)

                    with gr.Group(): # Common sliders
                        with gr.Row():
                            num_samples_slider = gr.Slider(label=translate("num_samples_label", self.module_translations), minimum=1, maximum=12, value=1, step=1)
                            seed_number = gr.Number(label=translate("seed_label", self.module_translations), value=-1, precision=0)
                    
                    with gr.Accordion(translate("advanced_options_label", self.module_translations), open=False):
                        steps_slider = gr.Slider(label=translate("steps_label", self.module_translations), minimum=1, maximum=100, value=25, step=1)
                        cfg_slider = gr.Slider(label=translate("cfg_scale_label", self.module_translations), minimum=1.0, maximum=32.0, value=2.0, step=0.01)
                        fc_lowres_denoise_slider = gr.Slider(label=translate("fc_lowres_denoise_label", self.module_translations), minimum=0.1, maximum=1.0, value=0.9, step=0.01, visible=(self.current_relight_mode == "FC"))
                        highres_scale_slider = gr.Slider(label=translate("highres_scale_label", self.module_translations), minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                        highres_denoise_slider = gr.Slider(label=translate("highres_denoise_label", self.module_translations), minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                        a_prompt_textbox = gr.Textbox(label=translate("added_prompt_label", self.module_translations), value='best quality')
                
                # --- Colonne de Droite (Sorties Images) ---
                with gr.Column(scale=1):
                    result_gallery = gr.Gallery(height=832, object_fit='contain', label=translate("output_gallery_label", self.module_translations))

            ips = [
                relight_mode_radio, input_fg, prompt_textbox, traduire_prompt_checkbox,
                num_samples_slider, seed_number, steps_slider, a_prompt_textbox, cfg_slider,
                highres_scale_slider, highres_denoise_slider,
                fc_lowres_denoise_slider, fc_bg_source_radio,
                fbc_input_bg_ui, fbc_bg_source_radio
            ]
            
            def handle_load_models_click(selected_mode_str_ui):
                mode_key = "FC" if selected_mode_str_ui == translate("relight_mode_fc_label", self.module_translations) else "FBC"
                # This will yield updates for load_models_button and relight_button
                # Need to consume the generator properly
                gen = self._load_models(mode_key)
                final_load_update, final_relight_update = None, None
                try:
                    while True:
                        final_load_update, final_relight_update = next(gen)
                except StopIteration:
                    pass # Generator finished
                
                # Update current_relight_mode_state if loading was successful for the new mode
                if self.models_loaded and self.current_relight_mode == mode_key:
                    current_relight_mode_state.value = mode_key # Update the state directly
                
                return final_load_update, final_relight_update, gr.update(value=self.current_relight_mode)


            load_models_button.click(
                fn=handle_load_models_click,
                inputs=[relight_mode_radio],
                outputs=[load_models_button, relight_button, current_relight_mode_state]
            )

            def update_ui_for_mode(selected_mode_str_ui, current_models_loaded_bool, current_loaded_mode_internal_state, input_fg_has_image):
                is_fc_selected = (selected_mode_str_ui == translate("relight_mode_fc_label", self.module_translations))
                is_fbc_selected = not is_fc_selected
                
                mode_key_selected_ui = "FC" if is_fc_selected else "FBC"

                relight_interactive = current_models_loaded_bool and (current_loaded_mode_internal_state == mode_key_selected_ui)
                load_interactive = not current_models_loaded_bool or (current_loaded_mode_internal_state != mode_key_selected_ui)
                
                load_button_text_key = "sana_load_button"
                if not load_interactive and current_models_loaded_bool : # Models are loaded for the selected mode
                    load_button_text_key = "models_loaded_successfully_for_mode"
                
                load_button_text = translate(load_button_text_key, self.module_translations)
                if load_button_text_key == "models_loaded_successfully_for_mode":
                    load_button_text = load_button_text.format(mode=current_loaded_mode_internal_state)

                # L'input de fond FBC est interactif seulement si le mode FBC est sélectionné ET une image sujet est chargée
                fbc_bg_input_interactive = is_fbc_selected and input_fg_has_image

                return gr.update(visible=is_fc_selected), gr.update(visible=is_fbc_selected), \
                       gr.update(visible=is_fbc_selected), gr.update(visible=is_fc_selected), \
                       gr.update(interactive=relight_interactive), \
                       gr.update(interactive=load_interactive, value=load_button_text), \
                       gr.update(interactive=fbc_bg_input_interactive) # Interactivité de fbc_input_bg_ui

            relight_mode_radio.change(
                fn=update_ui_for_mode,
                inputs=[relight_mode_radio, gr.State(self.models_loaded), current_relight_mode_state, input_fg], # Ajout de input_fg comme input pour son état
                outputs=[fc_specific_options_group_ui_visible_state, fbc_input_image_group_ui_visible_state, fbc_bg_source_options_group_ui_visible_state, fc_lowres_denoise_slider,
                         relight_button, load_models_button, fbc_input_bg_ui] # Ajout de fbc_input_bg_ui aux outputs
            )
            
            relight_button.click(
                fn=self.process_relight_wrapper,
                inputs=ips,
                outputs=[output_preview_fg, result_gallery, relight_button, stop_button, load_models_button]
            )
            stop_button.click(fn=self.stop_generation, inputs=None, outputs=[stop_button], api_name=False) # Disable stop button after click

            def build_prompt_from_dropdown_values(subject_dd_value, lighting_dd_value):
                translated_no_choice_val = translate(self.NO_CHOICE_KEY, self.module_translations)
                final_subject_part = subject_dd_value if subject_dd_value and subject_dd_value != translated_no_choice_val else ""
                final_lighting_part = lighting_dd_value if lighting_dd_value and lighting_dd_value != translated_no_choice_val else ""
                if final_subject_part and final_lighting_part: return f"{final_subject_part}, {final_lighting_part}"
                return final_subject_part or final_lighting_part or ""

            lighting_dropdown.change(fn=build_prompt_from_dropdown_values, inputs=[subject_dropdown, lighting_dropdown], outputs=prompt_textbox)
            subject_dropdown.change(fn=build_prompt_from_dropdown_values, inputs=[subject_dropdown, lighting_dropdown], outputs=prompt_textbox) # Changed input to lighting_dropdown

            input_fg.change(
                fn=self.preprocess_input_image,
                inputs=[input_fg],
                outputs=[output_preview_fg, relight_button, load_models_button, fbc_input_bg_ui] # fbc_input_bg_ui est maintenant un output
            )

            # --- Add change handler for FBC background image input ---
            def handle_fbc_bg_upload(input_bg_np):
                """Handles the upload of the FBC background image."""
                # This function simply receives the uploaded image and returns it.
                # Its main purpose is to ensure Gradio processes the change event for this component.
                # We don't need to return the image itself if it causes a loop.
                # When outputs=None, the function should ideally return None or nothing.
                # Gradio will handle the internal state update of the image component.
                pass # Explicitly do nothing and return None implicitly.

            fbc_input_bg_ui.change(
                fn=handle_fbc_bg_upload,
                inputs=[fbc_input_bg_ui],
                outputs=None # Ou outputs=[] si gr.update() seul ne suffit pas. Testez outputs=None en premier.
            )
            # --- End change handler for FBC background image input ---

            
            # Initial UI setup based on default mode
            # This is a bit tricky with Gradio's initial rendering.
            # We might need to trigger the change event once after the UI is built if direct gr.update doesn't work as expected on launch.
            # For now, the visibility is set directly in the gr.Group definitions.
            # And the load_models_button interactivity is set based on self.models_loaded.
            # The relight_button interactivity is also set based on self.models_loaded.

        return tab
