# test_module_mod.py
import os
import json
import time
from datetime import datetime
# Importe les fonctions/classes nécessaires de utils
# English: Import necessary functions/classes from utils
# Adapte cette liste selon les besoins réels du module
# English: Adapt this list according to the actual needs of the module
from Utils.utils import txt_color, translate, GestionModule, enregistrer_image

# Importe Gradio et d'autres bibliothèques nécessaires au module
# English: Import Gradio and other libraries necessary for the module
import gradio as gr
import torch
import gc
from PIL import Image

# --- Chargement des métadonnées du module (pour le nom, la description, etc.) ---
# English: Loading module metadata (for name, description, etc.) ---
module_json_path = os.path.join(os.path.dirname(__file__), "test_module_mod.json") # Assure-toi d'avoir ce fichier JSON / English: Make sure you have this JSON file

try:
    with open(module_json_path, 'r', encoding="utf-8") as f:
        module_data = json.load(f)
except FileNotFoundError:
    print(f"[ERREUR] Fichier JSON non trouvé pour test_module: {module_json_path}")
    module_data = {"name": "Test Module (Erreur JSON)"} # Valeur par défaut / English: Default value

# --- Fonction d'initialisation (appelée par GestionModule) ---
# English: Initialization function (called by GestionModule) ---
# Elle reçoit les traductions globales, l'instance du gestionnaire, et la config globale
# English: It receives global translations, the manager instance, and the global config
def initialize(global_translations, model_manager_instance, gestionnaire, global_config=None):
    """Initialise le module de test."""
    # English: Initializes the test module.
    print(txt_color("[OK] ", "ok"), f"Initialisation du module: {module_data.get('name', 'Test Module')}")
    # Crée et retourne une instance de la classe principale du module
    # English: Creates and returns an instance of the main module class
    return TestModule(global_translations, model_manager_instance, gestionnaire, global_config)

# --- Classe principale du module ---
# English: Main module class ---
class TestModule:
    # Le constructeur reçoit les arguments de la fonction initialize
    # English: The constructor receives arguments from the initialize function
    def __init__(self, global_translations, model_manager_instance, gestionnaire, global_config=None):
        """Initialise la classe TestModule."""
        # English: Initializes the TestModule class.
        self.global_translations = global_translations # Traductions globales / English: Global translations
        self.gestionnaire = gestionnaire # Instance de GestionModule / English: Instance of GestionModule
        self.global_config = global_config # Configuration globale (dict) / English: Global configuration (dict)
        self.model_manager = model_manager_instance # Stocke l'instance de ModelManager / English: Store the ModelManager instance
        self.module_translations = {} # Sera rempli dans create_tab / English: Will be filled in create_tab

        print(txt_color("[INFO]", "info"), f"{module_data.get('name', 'Test Module')}: Configuration reçue: {'Oui' if self.global_config else 'Non'}")
        # English: Configuration received: Yes/No
        print(txt_color("[INFO]", "info"), f"{module_data.get('name', 'Test Module')}: Gestionnaire reçu: {'Oui' if self.gestionnaire else 'Non'}")
        # English: Manager received: Yes/No

        # Tu peux initialiser d'autres variables spécifiques au module ici
        # English: You can initialize other module-specific variables here
        # Exemple: charger des ressources, définir des états, etc.
        # English: Example: load resources, define states, etc.
        # self.device, _, _ = check_gpu_availability(self.global_translations) # Si besoin du device / English: If device is needed

    def create_tab(self, module_translations):
        """Crée l'onglet Gradio pour ce module."""
        # English: Creates the Gradio tab for this module.
        # Stocke les traductions spécifiques à ce module
        # English: Stores the translations specific to this module
        self.module_translations = module_translations

        # Utilise module_translations pour l'interface de cet onglet
        # English: Use module_translations for the interface of this tab
        tab_name = translate("test_module_tab_name", self.module_translations) # Clé à définir dans le JSON / English: Key to define in JSON
        tab_title = translate("test_module_tab_title", self.module_translations) # Clé à définir / English: Key to define

        with gr.Tab(tab_name) as tab:
            gr.Markdown(f"## {tab_title}")
            gr.Markdown(translate("test_module_description", self.module_translations)) # Clé à définir / English: Key to define

            with gr.Row():
                input_text = gr.Textbox(label=translate("test_input_label", self.module_translations)) # Clé à définir / English: Key to define
                output_text = gr.Textbox(label=translate("test_output_label", self.module_translations)) # Clé à définir / English: Key to define

            process_button = gr.Button(translate("test_process_button", self.module_translations)) # Clé à définir / English: Key to define

            # Exemple de fonction de rappel utilisant les éléments stockés
            # English: Example callback function using stored elements
            def process_text(text_in):
                # Exemple d'accès à la config globale
                # English: Example of accessing global config
                save_dir = self.global_config.get("SAVE_DIR", ".") if self.global_config else "."
                print(f"Répertoire de sauvegarde (depuis config): {save_dir}")
                # English: Save directory (from config): ...

                # Exemple d'accès au pipe global (vérifier s'il existe)
                # English: Example of accessing global pipe (check if it exists)
                if self.model_manager and self.model_manager.get_current_pipe(): # Utilise model_manager / English: Use model_manager
                    print(txt_color("[INFO]", "info"), translate("test_pipe_disponible", self.module_translations)) # Clé à définir / English: Key to define
                    # Attention: Ne pas faire d'opération lourde ici sans décharger/recharger
                    # English: Warning: Do not perform heavy operations here without unloading/reloading
                else:
                    print(txt_color("[AVERTISSEMENT]", "erreur"), translate("test_pipe_non_disponible", self.module_translations)) # Clé à définir / English: Key to define

                # Logique de traitement simple
                # English: Simple processing logic
                processed = f"{translate('test_processed_prefix', self.module_translations)}: {text_in.upper()}" # Clé à définir / English: Key to define
                return processed

            process_button.click(
                fn=process_text,
                inputs=[input_text],
                outputs=[output_text]
            )

        return tab

    # Tu peux ajouter d'autres méthodes spécifiques au module ici
    # English: You can add other module-specific methods here
    # Par exemple, une méthode qui interagit avec le pipe global
    # English: For example, a method that interacts with the global pipe
    def exemple_utilisation_pipe(self):
        if self.model_manager and self.model_manager.get_current_pipe(): # Utilise model_manager / English: Use model_manager
            # Décharger d'abord si nécessaire
            # English: Unload first if necessary
            # ...
            # Utiliser self.model_manager.get_current_pipe() et self.model_manager.get_current_compel()
            # English: Use self.model_manager.get_current_pipe() and self.model_manager.get_current_compel()
            # ...
            # Recharger le modèle principal si besoin
            # English: Reload the main model if needed
            pass
        else:
            print(txt_color("[ERREUR]", "erreur"), translate("test_erreur_pipe_inaccessible", self.module_translations)) # Clé à définir / English: Key to define
