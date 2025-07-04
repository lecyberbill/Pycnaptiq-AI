import torch
import threading
import queue
import time
import traceback

# --- Définir les fallbacks AVANT le try ---
def _dummy_callback(*a, **kw): pass

def translate(key, t_dict, default=None): return t_dict.get(key, default or key)
def txt_color(text, _): return text

def create_callback_on_step_end(*args, **kwargs):
    print("[ERREUR] create_callback_on_step_end non importé ou échec import!")
    return _dummy_callback

def create_inpainting_callback(*args, **kwargs):
    print("[ERREUR] create_inpainting_callback non importé ou échec import!")
    return _dummy_callback

# --- MODIFICATION: Importer les deux types de callbacks ---
try:
    from Utils.callback_diffuser import create_callback_on_step_end, create_inpainting_callback # Importer et écraser les fallbacks si succès
    from Utils.utils import translate, txt_color
except ImportError as e:
    # --- Utilisation de print standard ici car translate n'est peut-être pas dispo ---
    print(f"[ERREUR pipeline_executor] Impossible d'importer les dépendances Utils: {e}")

def execute_pipeline_task_async(
    pipe,
    # --- Arguments sans défauts en premier ---
    # Réorganisation: les arguments sans défauts d'abord
    num_inference_steps,
    guidance_scale,
    seed,
    width,
    height,
    device,
    stop_event, # Garder stop_event ici
    translations, # Garder translations ici
    progress_queue, # Garder progress_queue ici
    # --- Arguments avec défauts ensuite ---
    image=None, 
    image_guidance_scale=None, 
    # --- Arguments PAG (avec défauts) ---
    pag_enabled=False,
    pag_scale=1.5,
    pag_applied_layers=None, # Liste des couches pour PAG
    prompt=None, # Optionnel pour le texte brut
    negative_prompt=None, # Optionnel pour le texte brut
    prompt_embeds=None,
    pooled_prompt_embeds=None,
    negative_prompt_embeds=None,
    negative_pooled_prompt_embeds=None,
    preview_queue=None,
    **kwargs # Accepter les arguments supplémentaires
 ):
    """
    Exécute une seule tâche de génération dans un thread séparé,
    gère la progression et les aperçus via des queues.
    Retourne le thread et un dictionnaire pour le résultat.
    """
    result_container = {"final": None, "error": None, "status": "running"}
    generator = torch.Generator(device=device).manual_seed(seed)

    # --- MODIFICATION: Choisir le callback en fonction de preview_queue ---
    if preview_queue is not None:
        callback_combined = create_callback_on_step_end( # Callback avec aperçu
            preview_queue,
            stop_event,
            num_inference_steps,
            translations,
            progress_queue,
            preview_frequency=1 # Ou une autre fréquence
        )
    else:
        callback_combined = create_inpainting_callback( # Callback sans aperçu
            stop_event,
            num_inference_steps,
            translations,
            progress_queue
        )
    # --- FIN MODIFICATION ---


    def run_pipeline_thread():
        try:
            # Vérifier si le callback est valide avant l'appel
            if not callable(callback_combined):
                 # Utiliser translate pour le message d'erreur
                 raise TypeError(translate("erreur_callback_non_appelable", translations))

            # --- Construire les arguments du pipeline dynamiquement ---
            pipeline_args = {
                "num_inference_steps": num_inference_steps, # Use : instead of =
                "guidance_scale": guidance_scale,
                "generator": generator,
                "width": width, # Keep width
                "height": height,
                "callback_on_step_end": callback_combined,
                "callback_on_step_end_tensor_inputs": ["latents"] # Nécessaire pour le callback
            }

            # Condition pour choisir entre texte brut et embeddings
            if prompt is not None: # Si le texte est fourni (cas spécifique comme Sana Sprint)
                pipeline_args["prompt"] = prompt
                # Ne PAS ajouter negative_prompt ici car Sana ne le prend pas
                # pipeline_args["negative_prompt"] = negative_prompt
            else: # Sinon, utiliser les embeddings (cas par défaut pour SDXL, etc.)
                pipeline_args["prompt_embeds"] = prompt_embeds
                pipeline_args["pooled_prompt_embeds"] = pooled_prompt_embeds
                pipeline_args["negative_prompt_embeds"] = negative_prompt_embeds
                pipeline_args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

            # --- AJOUT: Passer l'image si elle est fournie ---
            if image is not None:
                pipeline_args["image"] = image

            # --- AJOUT: Passer image_guidance_scale si elle est fournie ---
            if image_guidance_scale is not None:
                pipeline_args["image_guidance_scale"] = image_guidance_scale

            # --- AJOUT PAG ---
            if pag_enabled:
                pipeline_args["pag_scale"] = pag_scale
                if pag_applied_layers: # S'assurer que la liste n'est pas vide
                    pipeline_args["pag_applied_layers_index"] = pag_applied_layers # Nom du paramètre de l'exemple
                # Si PAG modifie guidance_scale, ajustez-le ici ou assurez-vous que le pipeline le fait.
            
            # Ajouter les arguments supplémentaires (comme max_sequence_length)
            pipeline_args.update(kwargs)
            # --- FIN AJOUT PAG ---
            # Vérifier l'arrêt APRÈS l'appel au pipeline
            if not stop_event.is_set():
                result = pipe(**pipeline_args) # Appeler le pipeline avec les arguments préparés
                # --- FIN MODIFICATION ---
                result_container["final"] = result.images[0]
                result_container["status"] = "success"
            else:
                result_container["status"] = "stopped"
                # Utiliser translate pour le message d'info
                print(txt_color("[INFO]", "info"), translate("info_pipeline_arrete_pendant_apres", translations))

        # Gérer l'interruption spécifique levée par le callback
        except InterruptedError:
             result_container["status"] = "stopped"
             # Utiliser translate pour le message d'info
             print(txt_color("[INFO]", "info"), translate("info_pipeline_interrompu_callback", translations))
        except Exception as e_pipe:
             if not stop_event.is_set():
                 result_container["error"] = e_pipe
                 result_container["status"] = "error"
                 # Utiliser translate pour le message d'erreur
                 print(f"{txt_color('[ERREUR]', 'erreur')} {translate('erreur_pipeline_thread', translations)}: {e_pipe}")
                 traceback.print_exc()
             else:
                 result_container["status"] = "stopped"
                 # Utiliser translate pour le message d'info
                 print(txt_color("[INFO]", "info"), f"{translate('info_pipeline_arrete_exception', translations)}: {e_pipe}")
        finally:
             if result_container["status"] == "running":
                 if stop_event.is_set():
                     result_container["status"] = "stopped"
                 else:
                     result_container["status"] = "unknown"
                     # Utiliser translate pour le message d'avertissement
                     print(txt_color("[WARN]", "warning"), translate("warn_pipeline_statut_inconnu", translations))


    thread = threading.Thread(target=run_pipeline_thread)
    thread.start()


    return thread, result_container
