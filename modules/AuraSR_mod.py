# AuraSR_mod.py
import os
import json
import time
from datetime import datetime
from Utils.utils import txt_color, translate, GestionModule, decharger_modele, enregistrer_image  
# Obtenir le chemin du fichier JSON du module
module_json_path = os.path.join(os.path.dirname(__file__), "AuraSR_mod.json")

# Créer une instance de GestionModule pour gérer les dépendances
with open(module_json_path, 'r', encoding="utf-8") as f:
    module_data = json.load(f)
module_manager = GestionModule(translations=module_data["language"]["fr"])


# Maintenant, on peut faire les imports en toute sécurité
import gradio as gr
from aura_sr import AuraSR
import torch
import gc


def initialize(global_translations, gestionnaire, global_config=None):
    """Initialise le module AuraSR."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    return AuraSRModule(global_translations, gestionnaire, global_config)

class AuraSRModule:
    def __init__(self, global_translations, gestionnaire, global_config=None):
        """Initialise la classe TestModule."""
        self.global_translations = global_translations
        self.gestionnaire = gestionnaire
        self.global_config = global_config
            
        
        if self.global_config is None:
            print(txt_color("[ERREUR]", "erreur"), "La configuration globale (global_config) n'a pas été reçue par AuraSRModule.")   
    
    def create_tab(self, module_translations):
        """Crée l'onglet Gradio pour ce module."""
        # Stocker les traductions spécifiques au module si nécessaire
        self.module_translations = module_translations

        with gr.Tab(translate("aura_sr_tab", module_translations)) as tab:
            gr.Markdown(f"## {translate('aura_sr_title', module_translations)}")

            with gr.Row():
                image_input = gr.Image(label=translate("image_input", module_translations), type="pil")
                image_output = gr.Image(label=translate("image_output", module_translations), type="pil")

            upscale_button = gr.Button(translate("upscale_button", module_translations))

            def upscale_image(image):
                # --- AJOUT : Vérification de global_config ---
                if self.global_config is None:
                    error_msg = "Erreur: La configuration globale n'est pas disponible."
                    print(txt_color("[ERREUR]", "erreur"), error_msg)
                    raise gr.Error(error_msg) # Lever une erreur Gradio
                    # return None # Ou retourner None si vous préférez

                if image is None:
                    gr.Warning(translate("veuillez_fournir_image", module_translations), 4.0) # Utiliser module_translations
                    return None

                start_time = time.time()
                try:
                    # Utiliser module_translations pour les messages spécifiques à cette fonction
                    upscaled_image = self.upscaleImage_BY_AuraSR(image, module_translations)

                    if upscaled_image is None: # Vérifier si l'upscale a réussi
                         return None

                    date_str = datetime.now().strftime("%Y_%m_%d")
                    heure_str = datetime.now().strftime("%H_%M_%S")

                    # Accès sécurisé à SAVE_DIR
                    save_dir_base = self.global_config.get("SAVE_DIR")
                    if not save_dir_base:
                         error_msg = "Erreur: Le chemin de sauvegarde (SAVE_DIR) n'est pas défini dans la configuration."
                         print(txt_color("[ERREUR]", "erreur"), error_msg)
                         raise gr.Error(error_msg)
                         # return None

                    save_dir = os.path.join(save_dir_base, date_str)
                    os.makedirs(save_dir, exist_ok=True)
                    # Utiliser module_translations pour le nom de fichier si pertinent
                    filename_key = 'upscaled_image' # Clé pour "image_upscalee"
                    filename_part = translate(filename_key, module_translations)
                    chemin_image = os.path.join(save_dir, f"aura_sr_{filename_part}_{date_str}_{heure_str}.jpg")

                    # Utiliser global_translations pour la fonction globale enregistrer_image
                    enregistrer_image(upscaled_image, chemin_image, self.global_translations, "JPEG")

                    elapsed_time = f"{(time.time() - start_time):.2f} sec"
                    # Utiliser global_translations pour les messages globaux
                    print(txt_color("[INFO] ","info"),f"{translate('temps_total_generation', self.global_translations)} : {elapsed_time}")
                    gr.Info(f"{translate('temps_total_generation', self.global_translations)} : {elapsed_time}", 3.0)

                    return upscaled_image

                except Exception as e:
                     # Utiliser module_translations pour l'erreur spécifique à l'upscale
                     error_msg = translate("erreur_upscale", module_translations) # Nouvelle clé
                     print(txt_color("[ERREUR]", "erreur"), f"{error_msg}: {e}")
                     import traceback
                     traceback.print_exc() # Afficher la trace complète pour le débogage
                     raise gr.Error(f"{error_msg}: {e}")
                     # return None

            upscale_button.click(fn=upscale_image, inputs=image_input, outputs=image_output, api_name="upscale_image")

        return tab
    
    def upscaleImage_BY_AuraSR(self, image, module_translations):
        """
        Upscale an image using AuraSR.
        """
        # Utiliser global_translations pour les messages liés au modèle principal
        if self.gestionnaire.global_pipe is not None:
            print(txt_color("[INFO]", "info"), translate("dechargement_modele_principal_avant_aurasr", self.global_translations)) # Nouvelle clé
            decharger_modele(self.gestionnaire.global_pipe, self.gestionnaire.global_compel, self.global_translations)
            self.gestionnaire.update_global_pipe(None)
            self.gestionnaire.update_global_compel(None)
            print(txt_color("[OK]", "ok"), translate("modele_principal_decharge", self.global_translations)) # Nouvelle clé
        else:
            print(txt_color("[INFO]", "info"), translate("aucun_modele_principal_a_decharger", self.global_translations)) # Nouvelle clé
        # --- FIN Déchargement ---

        try:
            # Utiliser module_translations pour les messages spécifiques à AuraSR
            print(txt_color("[INFO]", "info"), translate('chargement_aurasr', module_translations))
            gr.Info(translate('chargement_aurasr', module_translations), 3.0)
            aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

            # Utiliser module_translations pour les messages de device
            if torch.cuda.is_available():
                 print(txt_color("[INFO]", "info"), translate("tentative_utilisation_cuda_aurasr", module_translations)) # Nouvelle clé
            else:
                 print(txt_color("[INFO]", "info"), translate("utilisation_cpu_aurasr", module_translations)) # Nouvelle clé

            print(txt_color("[INFO]", "info"), translate('Upscaling_image_with_AuraSR', module_translations))
            gr.Info(translate('Upscaling_image_with_AuraSR', module_translations), 3.0)
            upscaled_image = aura_sr.upscale_4x_overlapped(image)

            # Utiliser module_translations pour les messages de déchargement AuraSR
            print(txt_color("[INFO]", "info"), translate("dechargement_aurasr", module_translations)) # Nouvelle clé
            del aura_sr
            torch.cuda.empty_cache() # Nettoyer la VRAM
            gc.collect()
            print(txt_color("[OK]", "ok"), translate("aurasr_decharge", module_translations)) # Nouvelle clé
            # --- FIN Déchargement ---

            return upscaled_image

        except Exception as e:
            error_msg = translate("erreur_chargement_ou_upscale_aurasr", module_translations)
            print(txt_color("[ERREUR]", "erreur"), f"{error_msg}: {e}")
            import traceback
            traceback.print_exc()
            gr.Error(f"{error_msg}: {e}")
            return None