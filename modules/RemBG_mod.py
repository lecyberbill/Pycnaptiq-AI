# RemBG_mod.py
import os
import json
import time
from datetime import datetime
# Importez les fonctions nécessaires de utils
from Utils.utils import txt_color, translate, GestionModule, decharger_modele, enregistrer_image
# Obtenir le chemin du fichier JSON du module
module_json_path = os.path.join(os.path.dirname(__file__), "RemBG_mod.json")

# Charger les données du module
with open(module_json_path, 'r', encoding="utf-8") as f:
    module_data = json.load(f)

# Imports spécifiques au module
import gradio as gr
from rembg import remove # Importer la fonction remove
from PIL import Image
import torch # Importer torch
import gc # Importer gc

# --- MODIFICATION : Signature de initialize ---
# Elle reçoit global_translations, gestionnaire, et global_config
def initialize(global_translations, gestionnaire, global_config=None):
    """Initialise le module RemBG."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    # Passer les arguments corrects à __init__
    return RemBGModule(global_translations, gestionnaire, global_config)

class RemBGModule:
    # --- MODIFICATION : Signature de __init__ ---
    def __init__(self, global_translations, gestionnaire, global_config=None):
        """Initialise la classe RemBGModule."""
        self.global_translations = global_translations
        self.gestionnaire = gestionnaire # Stocker l'instance du gestionnaire
        self.global_config = global_config # Stocker la configuration globale

        if self.global_config is None:
            print(txt_color("[ERREUR]", "erreur"), translate("erreur_config_globale_non_recue", self.global_translations).format("RemBGModule")) # Nouvelle clé
            # self.global_config = {} # Optionnel: définir un dict vide

    def create_tab(self, module_translations):
        """Crée l'onglet Gradio pour ce module."""
        self.module_translations = module_translations

        with gr.Tab(translate("rembg_tab", module_translations)) as tab: # Nouvelle clé
            gr.Markdown(f"## {translate('rembg_title', module_translations)}") # Nouvelle clé

            with gr.Row():
                image_input = gr.Image(label=translate("image_input", module_translations), type="pil")
                image_output = gr.Image(label=translate("image_output", module_translations), type="pil")

            remove_button = gr.Button(translate("remove_bg_button", module_translations)) # Nouvelle clé

            def removeBG(image):
                # --- AJOUT : Vérification de global_config ---
                if self.global_config is None:
                    error_msg = translate("erreur_config_globale_non_dispo", self.global_translations) # Nouvelle clé
                    print(txt_color("[ERREUR]", "erreur"), error_msg)
                    raise gr.Error(error_msg)
                    # return None

                if image is None:
                    gr.Warning(translate("veuillez_fournir_image", module_translations), 4.0)
                    return None

                start_time = time.time()
                try:
                    # Utiliser module_translations pour les messages spécifiques
                    processed_image = self.remove_backgroud(image, module_translations)

                    if processed_image is None: # Vérifier si remove_backgroud a réussi
                        return None

                    date_str = datetime.now().strftime("%Y_%m_%d")
                    heure_str = datetime.now().strftime("%H_%M_%S")

                    # Accès sécurisé à SAVE_DIR
                    save_dir_base = self.global_config.get("SAVE_DIR")
                    if not save_dir_base:
                         error_msg = translate("erreur_chemin_sauvegarde_non_defini", self.global_translations) # Nouvelle clé
                         print(txt_color("[ERREUR]", "erreur"), error_msg)
                         raise gr.Error(error_msg)
                         # return None

                    save_dir = os.path.join(save_dir_base, date_str)
                    os.makedirs(save_dir, exist_ok=True)
                    filename_part = translate('image_sans_fond', module_translations) # Nouvelle clé
                    # Sauvegarder en PNG pour conserver la transparence
                    chemin_image = os.path.join(save_dir, f"rembg_{filename_part}_{date_str}_{heure_str}.png")

                    # Utiliser global_translations pour la fonction globale enregistrer_image
                    # S'assurer que enregistrer_image gère bien le format PNG
                    enregistrer_image(processed_image, chemin_image, self.global_translations, "PNG")

                    elapsed_time = f"{(time.time() - start_time):.2f} sec"
                    # Utiliser global_translations pour les messages globaux
                    print(txt_color("[INFO] ","info"),f"{translate('temps_total_traitement', self.global_translations)} : {elapsed_time}") # Nouvelle clé
                    gr.Info(f"{translate('temps_total_traitement', self.global_translations)} : {elapsed_time}", 3.0)

                    return processed_image

                except Exception as e:
                     # Utiliser module_translations pour l'erreur spécifique
                     error_msg = translate("erreur_remove_bg", module_translations) # Nouvelle clé
                     print(txt_color("[ERREUR]", "erreur"), f"{error_msg}: {e}")
                     import traceback
                     traceback.print_exc()
                     raise gr.Error(f"{error_msg}: {e}")
                     # return None

            remove_button.click(fn=removeBG, inputs=image_input, outputs=image_output, api_name="remove_background")

        return tab

    def remove_backgroud(self, image, module_translations):
        """
        Removes the background from an image using rembg.
        """
        # --- MODIFICATION : Utiliser le gestionnaire pour décharger ---
        if self.gestionnaire.global_pipe is not None:
            print(txt_color("[INFO]", "info"), translate("dechargement_modele_principal_avant_rembg", self.global_translations)) # Nouvelle clé
            # Appeler decharger_modele avec les objets du gestionnaire
            decharger_modele(self.gestionnaire.global_pipe, self.gestionnaire.global_compel, self.global_translations)
            # Mettre à jour les références dans le gestionnaire après déchargement
            self.gestionnaire.update_global_pipe(None)
            self.gestionnaire.update_global_compel(None)
            print(txt_color("[OK]", "ok"), translate("modele_principal_decharge", self.global_translations))
        else:
            print(txt_color("[INFO]", "info"), translate("aucun_modele_principal_a_decharger", self.global_translations))
        # --- FIN MODIFICATION ---

        try:
            # Utiliser rembg
            print(txt_color("[INFO]", "info"), translate('debut_remove_bg', module_translations)) # Nouvelle clé
            gr.Info(translate('debut_remove_bg', module_translations), 3.0)

            # La fonction remove de rembg prend une image PIL et retourne une image PIL
            output_image = remove(image)

            # --- AJOUT : Nettoyage mémoire après rembg (optionnel mais recommandé) ---
            # rembg peut utiliser des modèles en arrière-plan (ex: u2net)
            print(txt_color("[INFO]", "info"), translate("nettoyage_memoire_apres_rembg", self.global_translations)) # Nouvelle clé
            torch.cuda.empty_cache()
            gc.collect()
            print(txt_color("[OK]", "ok"), translate("nettoyage_memoire_termine", self.global_translations))
            # --- FIN AJOUT ---

            return output_image

        except Exception as e:
            error_msg = translate("erreur_lib_rembg", module_translations) # Nouvelle clé
            print(txt_color("[ERREUR]", "erreur"), f"{error_msg}: {e}")
            import traceback
            traceback.print_exc()
            gr.Error(f"{error_msg}: {e}") # Afficher l'erreur dans Gradio
            return None # Retourner None en cas d'échec
