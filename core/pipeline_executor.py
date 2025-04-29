import torch
import threading
import queue
import time
import traceback

try:
    # --- CORRECTION DE L'IMPORT/ALIAS ---
    # Importer la fonction qui gère les aperçus
    from Utils.callback_diffuser import create_callback_on_step_end as create_progress_callback
    # --- FIN CORRECTION ---
    from Utils.utils import translate, txt_color
except ImportError as e:
    # --- Utilisation de print standard ici car translate n'est peut-être pas dispo ---
    print(f"[ERREUR pipeline_executor] Impossible d'importer les dépendances Utils: {e}")
    # Fallbacks basiques
    def translate(key, t_dict, default=None): return t_dict.get(key, default or key)
    def txt_color(text, _): return text
    def create_progress_callback(*args, **kwargs):
        print("[ERREUR] create_progress_callback non importé!")
        # Retourner une fonction factice qui ne fait rien
        def dummy_callback(*a, **kw): pass
        return dummy_callback

def execute_pipeline_task_async(
    pipe,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    num_inference_steps,
    guidance_scale,
    seed,
    width,
    height,
    device,
    stop_event, 
    translations, 
    progress_queue, 
    preview_queue=None
):
    """
    Exécute une seule tâche de génération dans un thread séparé,
    gère la progression et les aperçus via des queues.
    Retourne le thread et un dictionnaire pour le résultat.
    """
    result_container = {"final": None, "error": None, "status": "running"}
    generator = torch.Generator(device=device).manual_seed(seed)


    callback_combined = create_progress_callback( 
        preview_queue, 
        stop_event,    
        num_inference_steps,
        translations, 
        progress_queue,
        preview_frequency=1
    )


    def run_pipeline_thread():
        try:
            # Vérifier si le callback est valide avant l'appel
            if not callable(callback_combined):
                 # Utiliser translate pour le message d'erreur
                 raise TypeError(translate("erreur_callback_non_appelable", translations))

            result = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=width,
                height=height,
                callback_on_step_end=callback_combined,
                callback_on_step_end_tensor_inputs=["latents"] # Nécessaire pour le callback
            )
            # Vérifier l'arrêt APRÈS l'appel au pipeline
            if not stop_event.is_set():
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