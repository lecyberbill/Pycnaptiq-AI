# d:\image_to_text\cyberbill_SDXL\cyberbill_image_generator\Utils\callback_diffuser.py
import os
import torch
from PIL import Image
import glob
import time
import threading
import gradio as gr 
import queue

# Fonction pour convertir les latents en image RGB (reste inchangée)
def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    # S'assurer que l'image est dans le bon range et format avant conversion NumPy
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy() # Utiliser permute avant numpy

    return Image.fromarray(image_array)

# --- MODIFICATION 1 : create_callback_on_step_end ---
def create_callback_on_step_end(PREVIEW_QUEUE, stop_gen, total_steps: int, translations: dict, progress_queue: queue.Queue):
    """
    Crée une fonction de callback pour les aperçus ET la progression.

    Args:
        PREVIEW_QUEUE (list): La queue pour stocker les aperçus.
        stop_gen (threading.Event): L'événement pour arrêter la génération.
        progress_bar (gr.Progress): L'objet Gradio Progress à mettre à jour.
        total_steps (int): Le nombre total d'étapes d'inférence.
        translations (dict): Le dictionnaire de traductions.
    """
    # Utilise l'objet gr.Progress passé en argument pour suivre la progression
    # Pas besoin de créer un nouveau tracker ici si on utilise track_tqdm=True

    def callback_on_step_end(pipe, step: int, timestep: float, callback_kwargs: dict):
        # 1. Vérifier l'arrêt
        if stop_gen.is_set():
            pipe._interrupt = True # Signaler l'interruption au pipeline
            # On peut lever une exception ici aussi si nécessaire, mais _interrupt est souvent suffisant
            # raise InterruptedError("Génération interrompue via callback")

        # 2. Mettre à jour la barre de progression
        #    On utilise step+1 car les étapes sont souvent 0-indexées
        #    et on veut afficher 1/total_steps à la première étape.
        current_step_display = step + 1
        try:
            progress_queue.put_nowait((current_step_display, total_steps))
        except queue.Full:
            pass # Ignorer si la queue est pleine (évite le blocage)       

        # 3. Gérer l'aperçu (logique existante)
        try:
            # Assurez-vous que latents est bien dans callback_kwargs et a la bonne structure
            if "latents" in callback_kwargs:
                latents = callback_kwargs["latents"]
                # S'assurer qu'on prend le premier latent du batch s'il y en a plusieurs
                latent_to_preview = latents[0] if latents.ndim == 4 else latents
                if latent_to_preview.ndim == 3: # S'assurer qu'il a 3 dimensions (C, H, W)
                    image = latents_to_rgb(latent_to_preview)
                    if image:
                        PREVIEW_QUEUE.append(image)
                else:
                     print(f"[Callback Warning] Latent shape unexpected for preview: {latent_to_preview.shape}")
            else:
                print("[Callback Warning] 'latents' not found in callback_kwargs for preview.")

        except Exception as e:
            print(f"[Callback Error] Erreur lors de la génération de l'aperçu: {e}")

        # 4. Retourner les kwargs (important pour le pipeline)
        return callback_kwargs

    return callback_on_step_end

# --- MODIFICATION 2 : interrupt_diffusers_callback ---
# Renommée pour plus de clarté et ajout de la progression
def create_inpainting_callback(stop_gen, total_steps: int, translations: dict, progress_queue: queue.Queue):
    """
    Crée une fonction de callback pour l'inpainting gérant l'arrêt ET la progression.

    Args:
        stop_gen (threading.Event): L'événement pour arrêter la génération.
        progress_bar (gr.Progress): L'objet Gradio Progress à mettre à jour.
        total_steps (int): Le nombre total d'étapes d'inférence.
        translations (dict): Le dictionnaire de traductions.
    """
    def interrupt_callback(pipe, step: int, timestep: float, callback_kwargs: dict):
        # 1. Vérifier l'arrêt
        if stop_gen.is_set():
            pipe._interrupt = True

        # 2. Mettre à jour la barre de progression
        current_step_display = step + 1
        try:
            progress_queue.put_nowait((current_step_display, total_steps))
        except queue.Full:
            pass #

        # 3. Retourner les kwargs
        return callback_kwargs

    return interrupt_callback

# --- Helper function pour la traduction (si besoin ici, sinon passer 'translations') ---
# Vous avez déjà 'translate' dans cyberbill_SDXL.py, il faudra passer 'translations'
def translate(key, translations, default=None):
    return translations.get(key, default if default is not None else key)
