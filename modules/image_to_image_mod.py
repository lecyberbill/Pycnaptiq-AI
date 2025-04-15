#i2i_mod
import os
import json
import queue
import threading 
from Utils.utils import txt_color, translate, GestionModule, lister_fichiers, decharger_modele, check_gpu_availability, enregistrer_image, enregistrer_etiquettes_image_html, ImageSDXLchecker, styles_fusion, create_progress_bar_html
from Utils.callback_diffuser import create_inpainting_callback
from core.trannslator import translate_prompt
from datetime import datetime
#JSON associated at this module
module_json_path = os.path.join(os.path.dirname(__file__), "image_to_image_mod.json")


# Create a ManagementModule instance to manage dependencies
with open(module_json_path, 'r', encoding="utf-8") as f:
    module_data = json.load(f)


# Now we can do imports safely

import gradio as gr
import time
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
import torch
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
import gc


def initialize(global_translations, gestionnaire,  global_config=None):
    """Initialise le module test."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    return Image2imageSDXLModule(global_translations, gestionnaire, global_config)

class Image2imageSDXLModule:
    def __init__(self, global_translations, gestionnaire, global_config=None):
        """Initialise la classe Image2imageSDXLModule."""
        self.global_config = global_config
        self.global_translations = global_translations
        self.Image2ImageModel_path = global_config["MODELS_DIR"]
        self.vae_dir = global_config.get("VAE_DIR", "models/vae")
        self.liste_vaes = ["Auto"] + lister_fichiers(self.vae_dir, self.global_translations, ext=".safetensors", gradio_mode=False)
        self.liste_modeles = lister_fichiers(self.Image2ImageModel_path, self.global_translations)
        self.device, self.torch_dtype, self.vram_total_gb = check_gpu_availability(self.global_translations)
        self.gestionnaire = gestionnaire
        # Load styles from styles.json
        self.styles_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "styles.json"
        )
        self.styles = self.load_styles()
        self.default_negative_prompt = self.global_config.get("NEGATIVE_PROMPT", "")
        self.current_model_name = None
        self.current_vae_name = "Auto"
        self.stop_event = threading.Event()
    
    
    def stop_generation(self):
        """Active l'événement pour arrêter la génération en cours."""
        self.stop_event.set()
        print(txt_color("[INFO]", "info"), translate("stop_requested", self.global_translations))


    def load_styles(self):
        """Loads styles from styles.json."""
        try:
            with open(self.styles_path, "r", encoding="utf-8") as f:
                styles_data = json.load(f)
            return styles_data
        except FileNotFoundError:
            print(
                txt_color("[ERREUR] ", "erreur"),
                f"styles.json not found at {self.styles_path}",
            )
            return []
        except json.JSONDecodeError:
            print(
                txt_color("[ERREUR] ", "erreur"),
                f"Invalid JSON format in styles.json at {self.styles_path}",
            )
            return []   

    def create_tab(self, module_translations):
        """
        Crée l'onglet Gradio pour ce module.

        Args:
        module_translations (dict): Le dictionnaire de traductions du module.

        Returns:
        gr.Tab: L'onglet Gradio créé.
        """
        with gr.Tab(translate("i2i_tab_name", module_translations)) as tab:  
            gr.Markdown(f"## {translate('i2i_tab_title', module_translations)}")
            with gr.Row():
                with gr.Column():
                    self.image_input = gr.Image(label=translate("input_image", module_translations), type="pil")
                with gr.Column():
                    self.modele_i2i_dropdown = gr.Dropdown(label=translate("selectionner_modele", module_translations), choices=self.liste_modeles, value=self.liste_modeles[0] if self.liste_modeles else None)
                    self.vae_dropdown = gr.Dropdown(
                        label=translate("selectionner_vae", module_translations), # Needs translation key
                        choices=self.liste_vaes,
                        value="Auto", # Default to Auto
                        info=translate("selectionner_vae_info", module_translations) # Needs translation key
                    )
                    self.bouton_lister = gr.Button(translate("lister_modeles", module_translations))
                    self.bouton_charger_i2i = gr.Button(translate("charger_modele", module_translations))
                    self.message_chargement_i2i = gr.Textbox(label=translate("statut", module_translations), value=translate("aucun_modele_charge", module_translations))
                    
                    self.i2i_prompt_libre = gr.Textbox(label=translate("prompt_libre", module_translations), info=translate("prompt_libre_info", module_translations), placeholder=translate("prompt_libre_placeholder", module_translations))
                    self.i2i_traduire_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", module_translations), value=False, info=translate("traduire_prompt_libre", module_translations))
                    
                    self.style_dropdown = gr.Dropdown(
                        label=translate("selectionner_styles", module_translations),
                        choices=[style["name"] for style in self.styles],
                        value=[],
                        multiselect=True,
                        info=translate("selectionner_styles_info", module_translations),
                    )
                    with gr.Row():
                        self.i2i_progress_html = gr.HTML()
                    with gr.Row():
                        self.step_slider = gr.Slider(minimum=1, maximum=100, value=45, step=1, label=translate("etapes", module_translations))
                        self.guidance_slider = gr.Slider(minimum=1, maximum=20, value=9, step=0.5, label=translate("guidage", module_translations))
                        self.strength_slider = gr.Slider(minimum=0, maximum=1, value=0.6, step=0.01, label=translate("strength", module_translations))
                        texte_bouton_initial = translate("charger_modele_pour_commencer", module_translations)
                        self.bouton_i2i_gen = gr.Button(value=texte_bouton_initial, interactive=False)
                        self.i2i_bouton_stop = gr.Button(translate("arreter", module_translations), interactive=False, variant="stop", scale=1) # Ajouter "arreter" aux locales

                with gr.Column():
                    self.result_image_output = gr.Image(label=translate("output_image", module_translations), type="pil")


            self.bouton_lister.click(
                fn=self.mettre_a_jour_listes,
                outputs=[self.modele_i2i_dropdown, self.vae_dropdown]
            )

            self.bouton_charger_i2i.click(
                fn=lambda nom_f, nom_v: self.charger_modele_i2i_gradio(nom_f, nom_v, module_translations),
                inputs=[self.modele_i2i_dropdown, self.vae_dropdown],
                outputs=[self.message_chargement_i2i, self.bouton_i2i_gen, self.bouton_i2i_gen]
            )
            self.bouton_i2i_gen.click(
                fn=self.image_to_image_gen, # Appelle directement la méthode (qui est maintenant un générateur)
                inputs=[
                    self.i2i_prompt_libre,
                    self.i2i_traduire_checkbox,
                    self.style_dropdown,
                    self.image_input,
                    self.step_slider,
                    self.guidance_slider,
                    self.strength_slider
                ],
                # --- CORRECTION: Mettre les outputs dans une liste ---
                outputs=[
                    self.result_image_output,
                    self.i2i_progress_html,
                    self.bouton_i2i_gen,
                    self.i2i_bouton_stop
                ]
                # --- FIN CORRECTION ---
            )
            self.i2i_bouton_stop.click(
                fn=self.stop_generation,
                inputs=None,
                outputs=None
            )

        return tab
    
    def mettre_a_jour_listes(self):
        """Met à jour la liste des modèles et des VAEs disponibles."""
        self.liste_modeles = lister_fichiers(self.Image2ImageModel_path, self.global_translations, gradio_mode=True)
        self.liste_vaes = ["Auto"] + lister_fichiers(self.vae_dir, self.global_translations, ext=".safetensors", gradio_mode=True)
        return gr.update(choices=self.liste_modeles), gr.update(choices=self.liste_vaes)

    def charger_modele_i2i(self, nom_fichier, nom_vae, Image2ImageModel_path, module_translations):
        """
        Charge un modèle d'image_to_image à partir d'un fichier et le stocke dans self.global_pipe.

        Args:
            nom_fichier (str): Le nom du fichier contenant le modèle à charger.
            Image2ImageModel_path(str): Le chemin du dossier contenant le modèle à charger.
            nom_vae (str): Le nom du fichier VAE à charger, ou "Auto" pour utiliser celui du modèle.
            module_translations (dict): Le dictionnaire de traductions du module.

        Returns:
            tuple: A tuple containing:
                - self.global_pipe (StableDiffusionUpscalePipeline or None): The loaded pipeline or None if an error occurred.
                - message (str): A message to display in Gradio.
        """
        try:
            if not nom_fichier or nom_fichier == translate("aucun_modele", module_translations):
                print(txt_color("[ERREUR] ", "erreur"), translate("aucun_modele_selectionne", module_translations))
                gr.Warning(translate("aucun_modele_selectionne", module_translations), 4.0)
                return None, translate("aucun_modele_selectionne", module_translations)
            
            chemin_modele = os.path.join(Image2ImageModel_path, nom_fichier)
            
            if not os.path.exists(chemin_modele):
                print(txt_color("[ERREUR] ", "erreur"), f"{translate('modele_non_trouve', module_translations)}: {chemin_modele}")
                gr.Warning(translate("modele_non_trouve", module_translations) + f": {chemin_modele}", 4.0)
                return None, translate("modele_non_trouve", module_translations)

            
            # Décharger l'ancien modèle avant de charger le nouveau
            if self.gestionnaire.global_pipe is not None:
                self.decharger_modele(self.gestionnaire.global_pipe, self.gestionnaire.global_compel, self.global_translations)
            else:
                print(txt_color("[INFO] ", "info"), translate("aucun_modele_a_decharger", module_translations))

            self.gestionnaire.global_pipe = None
            self.gestionnaire.global_compel = None
            gc.collect()
            torch.cuda.empty_cache()

            print(txt_color("[INFO] ", "info"), f"{translate('chargement_modele', module_translations)} : {nom_fichier}")
            gr.Info(translate("chargement_modele", module_translations) + f" : {nom_fichier}", 3.0)
            self.gestionnaire.global_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                chemin_modele,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
            ).to(self.device)

            vae_message = ""
            if nom_vae and nom_vae != "Auto":
                chemin_vae = os.path.join(self.vae_dir, nom_vae)
                if os.path.exists(chemin_vae):
                    print(txt_color("[INFO] ", "info"), f"{translate('chargement_vae', module_translations)}: {nom_vae}") # Needs translation key
                    try:
                        vae = AutoencoderKL.from_single_file(
                            chemin_vae,
                            torch_dtype=self.torch_dtype # Use same dtype
                        ).to(self.device)
                        # Move VAE to device before assigning
                        self.current_vae_name = nom_vae
                        vae_message = f" + VAE: {nom_vae}"
                        print(txt_color("[OK] ", "ok"), f"{translate('vae_charge', module_translations)}: {nom_vae}") # Needs translation key
                    except Exception as e_vae:
                        print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_chargement_vae', module_translations)}: {nom_vae} - {e_vae}") # Needs translation key
                        gr.Warning(f"{translate('erreur_chargement_vae', module_translations)}: {nom_vae} - {e_vae}", 4.0)
                        # Continue without the external VAE, using the embedded one
                        self.current_vae_name = "Auto (Erreur)"
                        vae_message = f" + VAE: {translate('erreur_chargement_vae_court', module_translations)}" # Needs translation key
                else:
                    print(txt_color("[ERREUR] ", "erreur"), f"{translate('vae_non_trouve', module_translations)}: {chemin_vae}") # Needs translation key
                    gr.Warning(f"{translate('vae_non_trouve', module_translations)}: {chemin_vae}", 4.0)
                    self.current_vae_name = "Auto (Non trouvé)"
                    vae_message = f" + VAE: {translate('vae_non_trouve_court', module_translations)}" # Needs translation key
            else:
                # Using embedded VAE
                self.current_vae_name = "Auto"
                print(txt_color("[INFO] ", "info"), translate("utilisation_vae_integre", module_translations)) # Needs translation key
                vae_message = f" + VAE: Auto"

            try:
                self.gestionnaire.global_pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                 print(txt_color("[INFO] ", "info"), "xformers not available. Skipping memory efficient attention.")
            except Exception as e_optim:
                 print(txt_color("[AVERTISSEMENT] ", "erreur"), f"Could not enable xformers: {e_optim}")

            self.gestionnaire.global_pipe.enable_vae_slicing()
            self.gestionnaire.global_pipe.enable_vae_tiling()
            self.gestionnaire.global_pipe.enable_attention_slicing()
            self.gestionnaire.global_pipe.enable_model_cpu_offload()


            self.gestionnaire.global_compel = Compel(
            tokenizer=[self.gestionnaire.global_pipe.tokenizer, self.gestionnaire.global_pipe.tokenizer_2],
            text_encoder=[self.gestionnaire.global_pipe.text_encoder, self.gestionnaire.global_pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.device
            )    

            self.current_model_name = os.path.splitext(nom_fichier)[0]
            final_message = f"{translate('modele_charge', module_translations)}: {nom_fichier}{vae_message}"

            print(txt_color("[OK] ", "ok"), final_message)
            gr.Info(final_message, 3.0)
            return self.gestionnaire.global_pipe, final_message

        except Exception as e:
            # Clean up pipe if loading failed partially
            if 'pipe' in locals() and pipe is not None: del pipe
            if 'loaded_vae' in locals() and loaded_vae is not None: del loaded_vae
            self.gestionnaire.global_pipe = None
            self.gestionnaire.global_compel = None
            gc.collect()
            torch.cuda.empty_cache()
            error_msg = f"{translate('erreur_chargement_modele', module_translations)}: {e}"
            print(txt_color("[ERREUR] ", "erreur"), error_msg)
            # Use raise gr.Error for better UI feedback
            raise gr.Error(error_msg)
    
    def charger_modele_i2i_gradio(self, nom_fichier, nom_vae, module_translations):
        """
        Charge un modèle d'image_to_image et VAE, retourne un message pour Gradio
        et met à jour l'état interactif ET le texte du bouton de génération.
        """
        self.gestionnaire.global_pipe, message = self.charger_modele_i2i(nom_fichier, nom_vae, self.Image2ImageModel_path, self.global_translations)
        
        if self.gestionnaire.global_pipe is not None:
            etat_interactif = True
            texte_bouton = translate("image_to_image_gen", module_translations)
        else:
            etat_interactif = False
            texte_bouton = translate("charger_modele_pour_commencer", module_translations)
            
        
        update_interactif = gr.update(interactive=etat_interactif)
        update_texte = gr.update(value=texte_bouton)
        
        return message, update_interactif, update_texte
    

    def image_to_image_gen(self, prompt_libre, traduire, selected_styles, image, step, guidance, strength, module_translations):
        """
        Generates an image using the loaded i2i model.

        Args:
            prompt (str): The prompt to guide the image_to_image process.
            negative_prompt (str): The negative prompt to guide the image_to_image process.
            image (PIL.Image.Image): The image to use as a base.
            step (int): The number of inference steps.
            guidance (float): The guidance scale.
            strength (float): How much to transform the reference image.

        Returns:
            PIL.Image.Image: The generated image, or None if an error occurs.
        """
        module_translations = self.global_translations 
        try:
            if self.gestionnaire.global_pipe is None or self.gestionnaire.global_compel is None: # Also check compel
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_modele_i2i", module_translations))
                gr.Warning(translate("erreur_pas_modele_i2i", self.global_translations), 4.0)
                yield None, "", gr.update(interactive=False), gr.update(interactive=False)
                return

            if image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image", module_translations))
                gr.Warning(translate("erreur_pas_image", self.global_translations), 4.0)
                yield None, "", gr.update(interactive=False), gr.update(interactive=False)
                return

            start_time = time.time()
            self.stop_event.clear() # Réinitialiser l'événement d'arrêt
            final_image_container = {} # Pour récupérer l'image du thread
            progress_queue = queue.Queue()

            actual_steps = max(1, int(step * strength))

            yield None, create_progress_bar_html(0, actual_steps, 0), gr.update(interactive=False), gr.update(interactive=True)

             # --- PROMPT HANDLING ---
            base_user_prompt = ""
            if prompt_libre and prompt_libre.strip():
                print(txt_color("[INFO]", "info"), translate("utilisation_prompt_libre", module_translations))
                base_user_prompt = translate_prompt(prompt_libre, module_translations) if traduire else prompt_libre


            final_prompt_text, final_negative_prompt, style_names_used = styles_fusion(
                selected_styles,
                base_user_prompt,
                self.default_negative_prompt,
                self.styles,
                module_translations
            )
            # --- END PROMPT HANDLING ---

            #redimensionnement de l'image si besoin
            image_checker = ImageSDXLchecker(image, module_translations)
            image = image_checker.redimensionner_image()
            


            # 'final_prompt_text' contient maintenant le prompt à utiliser pour Compel
            if not final_prompt_text.strip():
                print(txt_color("[AVERTISSEMENT]", "erreur"), translate("erreur_aucun_prompt_final", module_translations))
                gr.Warning(translate("erreur_aucun_prompt_final", module_translations), 4.0)
                yield None, f'<p style="color:red;">{msg}</p>', gr.update(interactive=True), gr.update(interactive=False)

                return Non

            # Utiliser Compel avec le prompt final
            conditioning, pooled = self.gestionnaire.global_compel(final_prompt_text)
            neg_conditioning, neg_pooled = self.gestionnaire.global_compel(final_negative_prompt)
            generator = torch.Generator(device=self.device).manual_seed(int(time.time()))


            callback = create_inpainting_callback(
                self.stop_event,
                actual_steps,
                module_translations,
                progress_queue
            )

            def run_pipeline():
                try:
                    result = self.gestionnaire.global_pipe(
                        pooled_prompt_embeds=pooled,
                        prompt_embeds=conditioning,
                        negative_prompt_embeds=neg_conditioning,         
                        negative_pooled_prompt_embeds=neg_pooled,  
                        image=image,
                        num_inference_steps=step,
                        guidance_scale=guidance,
                        strength=strength,
                        generator=generator,
                        callback_on_step_end=callback
                    )
                    if not self.stop_event.is_set():
                                final_image_container["final"] = result.images[0]
                except Exception as e:
                    # Ne pas afficher l'erreur si c'est juste une interruption
                    if not (hasattr(self.gestionnaire.global_pipe, '_interrupt') and self.gestionnaire.global_pipe._interrupt):
                        print(txt_color("[ERREUR]", "erreur"), f"Erreur dans run_pipeline (i2i): {e}")
                        final_image_container["error"] = e

            thread = threading.Thread(target=run_pipeline)
            thread.start()


            last_progress_html = ""
            while thread.is_alive() or not progress_queue.empty():
                current_step_prog, total_steps_prog = None, step
                while not progress_queue.empty():
                    try:
                        current_step_prog, total_steps_prog = progress_queue.get_nowait()
                    except queue.Empty:
                        break       


                if current_step_prog is not None:
                    progress_percent = int((current_step_prog / total_steps_prog) * 100)
                    last_progress_html = create_progress_bar_html(current_step_prog, total_steps_prog, progress_percent)
                    # Yield la progression (Image=None, Progress HTML, Bouton Gen=inactive, Bouton Stop=active)
                    yield None, last_progress_html, gr.update(interactive=False), gr.update(interactive=True)

                time.sleep(0.05)

            thread.join()


            if hasattr(self.gestionnaire.global_pipe, '_interrupt'):
                self.gestionnaire.global_pipe._interrupt = False


            if "error" in final_image_container:
                error_msg = f"{translate('erreur_image_to_image', module_translations)}: {final_image_container['error']}"
                print(txt_color("[ERREUR]", "erreur"), error_msg)
                # Yield état d'erreur
                yield None, f'<p style="color:red;">{error_msg}</p>', gr.update(interactive=True), gr.update(interactive=False)
                return

            if self.stop_event.is_set():
                msg = translate("generation_arretee", module_translations) # Ajouter "generation_arretee" aux locales
                print(txt_color("[INFO]", "info"), msg)
                gr.Info(msg, 3.0)
                # Yield état arrêté
                yield None, msg, gr.update(interactive=True), gr.update(interactive=False)
                return

            result_image = final_image_container.get("final")
            if result_image is None:
                # Sécurité, ne devrait pas arriver si pas d'erreur/arrêt
                msg = translate("erreur_pas_image_genere", module_translations) # Ajouter "erreur_pas_image_genere"
                print(txt_color("[ERREUR]", "erreur"), msg)
                yield None, f'<p style="color:red;">{msg}</p>', gr.update(interactive=True), gr.update(interactive=False)
                return

            temps_generation_image = f"{(time.time() - start_time):.2f} sec"
            current_time = time.strftime("%Y_%m_%d_%H_%M_%S")
            width, height = result_image.size
            style_filename_part = "_".join(style_names_used) if style_names_used else "NoStyle"
            style_filename_part = style_filename_part.replace(" ", "_")[:30] # Limit length
            filename = f"i2i_{style_filename_part}_{current_time}_{height}_{width}.{self.global_config['IMAGE_FORMAT']}"
            date_str = datetime.now().strftime("%Y_%m_%d")
            save_dir = os.path.join(self.global_config["SAVE_DIR"], date_str)
            os.makedirs(save_dir, exist_ok=True)
            chemin_image = os.path.join(save_dir, filename)
            enregistrer_image(result_image, chemin_image, self.global_translations, self.global_config["IMAGE_FORMAT"])
            #crétation du rapport HTML ou Mise à jour de celui-ci
            xmp_data = {
                "Module": "Image to Image SDXL module",
                "Creator": self.global_config['AUTHOR'],
                "Modèle": self.current_model_name,
                "VAE": self.current_vae_name,
                "Inference": step,
                "Guidance": guidance,
                "Styles": ", ".join(style_names_used) if style_names_used else "None", # List styles used
                "Prompt": final_prompt_text,
                "Negative prompt:": final_negative_prompt, # Use combined negative
                "strength": strength,
                "Dimensions": f"{width} x {height}",
                "Temps de génération": temps_generation_image
            }


            enregistrer_etiquettes_image_html(chemin_image, xmp_data, module_translations, True)

            print(txt_color("[OK]", "ok"), f"Génération I2I terminée en {temps_generation_image}")
            # Yield final (Image, Progress HTML vide, Bouton Gen=active, Bouton Stop=inactive)
            yield result_image, "", gr.update(interactive=True), gr.update(interactive=False)

        except Exception as e:
            # Gestion d'erreur globale pour la fonction génératrice
            error_message = f"{translate('erreur_image_to_image', module_translations)}: {e}"
            print(txt_color("[ERREUR] ", "erreur"), error_message)
            import traceback
            traceback.print_exc() # Afficher la trace complète pour le débogage
            # Yield état d'erreur
            yield None, f'<p style="color:red;">{error_message}</p>', gr.update(interactive=True), gr.update(interactive=False)
        finally:
            pass


    def decharger_modele(self, pipe, compel, translations):
        """Libère proprement la mémoire GPU en déplaçant temporairement le modèle sur CPU."""
        # This method seems duplicated from utils.py. Consider calling the one from utils.
        # For now, keeping the local implementation as requested.
        try:
            if pipe is not None:
                print(txt_color("[INFO] ", "info"), translate("dechargement_modele", translations))

                # Déplacer le modèle vers CPU avant suppression pour libérer la VRAM
                try:
                    pipe.to("cpu")
                    print(txt_color("[INFO] ", "info"), translate("modele_deplace_cpu", translations))
                except Exception as e:
                    print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_deplacement_cpu', translations)}: {e}")

                torch.cuda.synchronize() # Ensure move is complete

                if hasattr(pipe, 'vae'): del pipe.vae
                if hasattr(pipe, 'text_encoder'): del pipe.text_encoder
                if hasattr(pipe, 'text_encoder_2'): del pipe.text_encoder_2
                if hasattr(pipe, 'tokenizer'): del pipe.tokenizer
                if hasattr(pipe, 'tokenizer_2'): del pipe.tokenizer_2
                if hasattr(pipe, 'unet'): del pipe.unet
                if hasattr(pipe, 'scheduler'): del pipe.scheduler
                if hasattr(pipe, 'feature_extractor'): del pipe.feature_extractor # If exists

                del pipe
                # pipe = None # Not needed, variable goes out of scope or is reassigned

                if compel is not None:
                    # Compel doesn't hold large models directly, but good practice
                    del compel
                    # compel = None # Not needed

                # Explicitly set manager's references to None
                self.gestionnaire.global_pipe = None
                self.gestionnaire.global_compel = None
                self.current_model_name = None
                self.current_vae_name = "Auto"


                # Nettoyage de la mémoire GPU et RAM
                gc.collect() # Suggest garbage collection
                torch.cuda.empty_cache() # Release cached memory on GPU
                torch.cuda.ipc_collect() # More aggressive cleanup (if needed)
                torch.cuda.synchronize() # Wait for cleanup

                print(txt_color("[OK] ", "ok"), translate("modele_precedent_decharge", translations))
            else:
                print(txt_color("[INFO] ", "info"), translate("aucun_modele_a_decharger", translations))

        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_dechargement_modele', translations)}: {e}")
            # Ensure references are cleared even on error
            self.gestionnaire.global_pipe = None
            self.gestionnaire.global_compel = None
            self.current_model_name = None
            self.current_vae_name = "Auto"
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            pass



