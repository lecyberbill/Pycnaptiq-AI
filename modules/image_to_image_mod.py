#i2i_mod
import os
import json
from Utils.utils import txt_color, translate, GestionModule, lister_fichiers, decharger_modele, check_gpu_availability, enregistrer_image, enregistrer_etiquettes_image_html, ImageSDXLchecker
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
from diffusers import StableDiffusionXLImg2ImgPipeline
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
        self.Image2ImageModel_path = global_config["MODELS_DIR"]
        self.global_translations = global_translations
        self.liste_modeles = lister_fichiers(self.Image2ImageModel_path, self.global_translations)
        self.device, self.torch_dtype, self.vram_total_gb = check_gpu_availability(self.global_translations)
        self.gestionnaire = gestionnaire
        # Load styles from styles.json
        self.styles_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "styles.json"
        )
        self.styles = self.load_styles()

        self.current_model_name = None
    
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
                    self.bouton_lister = gr.Button(translate("lister_modeles", module_translations))
                    self.bouton_charger_i2i = gr.Button(translate("charger_modele", module_translations))
                    self.message_chargement_i2i = gr.Textbox(label=translate("statut", module_translations), value=translate("aucun_modele_charge", module_translations))
                    
                    self.i2i_prompt_libre = gr.Textbox(label=translate("prompt_libre", module_translations), info=translate("prompt_libre_info", module_translations), placeholder=translate("prompt_libre_placeholder", module_translations))
                    self.i2i_traduire_checkbox = gr.Checkbox(label=translate("traduire_en_anglais", module_translations), value=False, info=translate("traduire_prompt_libre", module_translations))
                    
                    self.style_dropdown = gr.Dropdown(
                        label=translate("selectionner_style", module_translations),
                        choices=[style["name"] for style in self.styles],
                        value=self.styles[0]["name"] if self.styles else None,
                    )
                    with gr.Row():
                        self.step_slider = gr.Slider(minimum=1, maximum=100, value=75, step=1, label=translate("etapes", module_translations))
                        self.guidance_slider = gr.Slider(minimum=1, maximum=20, value=9, step=0.5, label=translate("guidage", module_translations))
                        self.strength_slider = gr.Slider(minimum=0, maximum=1, value=0.6, step=0.01, label=translate("strength", module_translations))
                        texte_bouton_initial = translate("charger_modele_pour_commencer", module_translations)
                        self.bouton_i2i_gen = gr.Button(value=texte_bouton_initial, interactive=False)

                with gr.Column():
                    self.result_image_output = gr.Image(label=translate("output_image", module_translations), type="pil")


            self.bouton_lister.click(
                fn=self.mettre_a_jour_listes,
                outputs=self.modele_i2i_dropdown
            )

            self.bouton_charger_i2i.click(
                fn=lambda nom_f: self.charger_modele_i2i_gradio(nom_f, module_translations),
                inputs=self.modele_i2i_dropdown,
                outputs=[self.message_chargement_i2i, self.bouton_i2i_gen, self.bouton_i2i_gen]
            )
            self.bouton_i2i_gen.click(
                # Utiliser une lambda pour passer les arguments UI + module_translations
                fn=lambda prompt, traduire, style, img, steps, guide, strength: self.image_to_image_gen(
                    prompt, traduire, style, img, steps, guide, strength, module_translations
                ),
                inputs=[
                    self.i2i_prompt_libre,
                    self.i2i_traduire_checkbox,
                    self.style_dropdown,
                    self.image_input,
                    self.step_slider,
                    self.guidance_slider,
                    self.strength_slider
                ],
                outputs=self.result_image_output
            )

        return tab
    
    def mettre_a_jour_listes(self):
        """Met à jour la liste des modèles disponibles."""
        self.liste_modeles = lister_fichiers(self.Image2ImageModel_path, self.global_translations)
        return gr.update(choices=self.liste_modeles)

    def charger_modele_i2i(self, nom_fichier, Image2ImageModel_path, module_translations):
        """
        Charge un modèle d'image_to_image à partir d'un fichier et le stocke dans self.global_pipe.

        Args:
            nom_fichier (str): Le nom du fichier contenant le modèle à charger.
            Image2ImageModel_path(str): Le chemin du dossier contenant le modèle à charger.
            module_translations (dict): Le dictionnaire de traductions du module.

        Returns:
            tuple: A tuple containing:
                - self.global_pipe (StableDiffusionUpscalePipeline or None): The loaded pipeline or None if an error occurred.
                - message (str): A message to display in Gradio.
        """
        try:
            if not nom_fichier:
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

            print(txt_color("[INFO] ", "info"), f"{translate('chargement_modele', module_translations)} : {nom_fichier}")
            gr.Info(translate("chargement_modele", module_translations) + f" : {nom_fichier}", 3.0)
            self.gestionnaire.global_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                chemin_modele,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
            ).to(self.device)

            self.gestionnaire.global_pipe.enable_xformers_memory_efficient_attention()
            self.gestionnaire.global_pipe.enable_model_cpu_offload()
            self.gestionnaire.global_pipe.enable_vae_slicing()
            self.gestionnaire.global_pipe.enable_vae_tiling()

            # Initialisation de compel

            self.gestionnaire.global_compel = Compel(
            tokenizer=[self.gestionnaire.global_pipe.tokenizer, self.gestionnaire.global_pipe.tokenizer_2],
            text_encoder=[self.gestionnaire.global_pipe.text_encoder, self.gestionnaire.global_pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
            )    

            self.current_model_name = os.path.splitext(nom_fichier)[0]

            print(txt_color("[OK] ", "ok"), translate("modele_charge", module_translations), f": {nom_fichier}")
            gr.Info(translate("modele_charge", module_translations) + f": {nom_fichier}", 3.0)
            return self.gestionnaire.global_pipe, translate("modele_charge", module_translations)

        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_chargement_modele", module_translations), f": {e}")
            raise gr.Error(translate("erreur_chargement_modele", module_translations), f": {e}", 4.0)
            return None, translate("erreur_chargement_modele", module_translations) + f": {e}"
    
    def charger_modele_i2i_gradio(self, nom_fichier, module_translations):
        """
        Charge un modèle d'image_to_image, retourne un message pour Gradio
        et met à jour l'état interactif ET le texte du bouton de génération.
        """
        self.gestionnaire.global_pipe, message = self.charger_modele_i2i(nom_fichier, self.Image2ImageModel_path, self.global_translations)
        
        if self.gestionnaire.global_pipe is not None:
            etat_interactif = True
            texte_bouton = translate("image_to_image_gen", module_translations)
        else:
            etat_interactif = False
            texte_bouton = translate("charger_modele_pour_commencer", module_translations)
            
        
        update_interactif = gr.update(interactive=etat_interactif)
        update_texte = gr.update(value=texte_bouton)
        
        return message, update_interactif, update_texte
    

    def image_to_image_gen(self, prompt_libre, traduire, selected_style, image, step, guidance, strength, module_translations):
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
        try:
            if self.gestionnaire.global_pipe is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_modele_i2i", module_translations))
                gr.Warning(translate("erreur_pas_modele_i2i", self.global_translations), 4.0)
                return None

            if image is None:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_pas_image", module_translations))
                gr.Warning(translate("erreur_pas_image", self.global_translations), 4.0)
                return None

            # Get the selected style
            selected_style_data = next(
                (
                    style
                    for style in self.styles
                    if style["name"] == selected_style
                ),
                None,
            )

            if selected_style_data is None:
                print(
                    txt_color("[ERREUR] ", "erreur"),
                    translate("erreur_style_non_trouve", module_translations),
                    f": {selected_style}",
                )
                gr.Warning(
                    translate("erreur_style_non_trouve", module_translations)
                    + f": {selected_style}",
                    4.0,
                )
                return None
            start_time = time.time()

            #redimensionnement de l'image si besoin
            image_checker = ImageSDXLchecker(image, module_translations)
            image = image_checker.redimensionner_image()
            
            final_prompt_text = "" # Initialiser le prompt final

            if prompt_libre and prompt_libre.strip():
                # Un prompt libre a été fourni
                print(txt_color("[INFO]", "info"), translate("utilisation_prompt_libre", module_translations))

                # 1. Obtenir le prompt utilisateur de base (traduit ou non)
                core_user_prompt = translate_prompt(prompt_libre, module_translations) if traduire else prompt_libre

                # 2. Essayer de l'intégrer dans la structure du style sélectionné
                if selected_style_data and "{prompt}" in selected_style_data["prompt"]:
                    # Si le style existe et contient le marqueur, on remplace
                    final_prompt_text = selected_style_data["prompt"].replace("{prompt}", core_user_prompt)
                    print(txt_color("[INFO]", "info"), translate("prompt_libre_integre_style", module_translations).format(selected_style))
                else:
                    # Sinon (pas de style sélectionné ou pas de marqueur dans le style), utiliser le prompt utilisateur directement
                    final_prompt_text = core_user_prompt
                    print(txt_color("[INFO]", "info"), translate("prompt_libre_utilise_directement", module_translations))

            else:
                # Aucun prompt libre fourni, utiliser celui du style
                if selected_style_data:
                    print(txt_color("[INFO]", "info"), translate("utilisation_prompt_style", module_translations).format(selected_style))
                    final_prompt_text = selected_style_data["prompt"]
                    # Optionnel : Nettoyer le marqueur {prompt} s'il existe mais n'a pas été remplacé
                    final_prompt_text = final_prompt_text.replace("{prompt}", "").strip()
                # Si selected_style_data est None aussi, final_prompt_text restera "" (à gérer plus loin si nécessaire)

            # 'final_prompt_text' contient maintenant le prompt à utiliser pour Compel
            if not final_prompt_text:
                 print(txt_color("[AVERTISSEMENT]", "erreur"), translate("erreur_aucun_prompt_final", module_translations))
                 # Gérer l'erreur : peut-être retourner None ou lever une exception Gradio
                 raise gr.Error(translate("erreur_aucun_prompt_final", module_translations), 4.0)
                 # return None # Si la fonction doit retourner quelque chose en cas d'erreur

            # Utiliser Compel avec le prompt final
            conditioning, pooled = self.gestionnaire.global_compel(final_prompt_text)

            # Generate the image using the pipeline
            result_image = self.gestionnaire.global_pipe(
                pooled_prompt_embeds=pooled,
                prompt_embeds=conditioning,
                negative_prompt=selected_style_data["negative_prompt"],
                image=image,
                num_inference_steps=step,
                guidance_scale=guidance,
                strength=strength,
            ).images[0]

            temps_generation_image = f"{(time.time() - start_time):.2f} sec"
            # Generate filename
            current_time = time.strftime("%Y_%m_%d_%H_%M_%S")
            width, height = result_image.size
            filename = f"i2i_{selected_style}_{current_time}_{height}_{width}.{self.global_config['IMAGE_FORMAT']}"
            # Save the image
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
                "Inference": step,
                "Guidance": guidance,
                "Style": selected_style,
                "Prompt": final_prompt_text,
                "Negative prompt:": selected_style_data["negative_prompt"],
                "strength": strength,
                "Dimensions": f"{width} x {height}",
                "Temps de génération": temps_generation_image
            }


            enregistrer_etiquettes_image_html(chemin_image, xmp_data, self.global_translations, True)

            return result_image

        except Exception as e:
            print(
                txt_color("[ERREUR] ", "erreur"),
                translate("erreur_image_to_image", module_translations),
                f": {e}",
            )
            raise gr.Error(
                translate("erreur_image_to_image", module_translations),
                f": {e}",
                4.0,
            )
            return None


            return result_image

        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_image_to_image", module_translations), f": {e}")
            raise gr.Error(translate("erreur_image_to_image", module_translations), f": {e}", 4.0)
            return None



    def decharger_modele(self, pipe, compel, translations):
        """Libère proprement la mémoire GPU en déplaçant temporairement le modèle sur CPU."""
        try:
            if pipe is not None:
                print(txt_color("[INFO] ", "info"), translate("dechargement_modele", translations))

                # Déplacer le modèle vers CPU avant suppression pour libérer la VRAM
                try:
                    pipe.to("cpu")
                    print(txt_color("[INFO] ", "info"), translate("modele_deplace_cpu", translations))
                except Exception as e:
                    print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_deplacement_cpu', translations)}: {e}")

                torch.cuda.synchronize()

                # Désactiver les optimisations mémoire
                if hasattr(pipe, 'disable_vae_slicing'):
                    pipe.disable_vae_slicing()
                if hasattr(pipe, 'disable_vae_tiling'):
                    pipe.disable_vae_tiling()
                if hasattr(pipe, 'disable_attention_slicing'):
                    pipe.disable_attention_slicing()

                # Supprimer proprement chaque composant
                if hasattr(pipe, 'vae'):
                    del pipe.vae
                if hasattr(pipe, 'text_encoder'):
                    del pipe.text_encoder
                if hasattr(pipe, 'text_encoder_2'):
                    del pipe.text_encoder_2
                if hasattr(pipe, 'tokenizer'):
                    del pipe.tokenizer
                if hasattr(pipe, 'tokenizer_2'):
                    del pipe.tokenizer_2
                if hasattr(pipe, 'unet'):
                    del pipe.unet
                if hasattr(pipe, 'scheduler'):
                    del pipe.scheduler

                del pipe
                pipe = None  # ajout de la suppression de pipe

                if compel is not None:
                    del compel
                    compel = None  # ajout de la suppression de compel

                # Nettoyage de la mémoire GPU et RAM
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                gc.collect()

                print(txt_color("[OK] ", "ok"), translate("modele_precedent_decharge", translations))
            else:
                print(txt_color("[INFO] ", "info"), translate("aucun_modele_a_decharger", translations))

        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_dechargement_modele', translations)}: {e}")
        finally:
            pass



