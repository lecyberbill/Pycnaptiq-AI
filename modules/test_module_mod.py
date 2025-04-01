# test_module_mod.py
import os
import json
from Utils.utils import txt_color, translate, GestionModule  # Import GestionModule

# Obtenir le chemin du fichier JSON du module
module_json_path = os.path.join(os.path.dirname(__file__), "test_module_mod.json")

# Créer une instance de GestionModule pour gérer les dépendances
with open(module_json_path, 'r', encoding="utf-8") as f:
    module_data = json.load(f)
module_manager = GestionModule(translations=module_data["language"]["fr"])


# Maintenant, on peut faire les imports en toute sécurité
import gradio as gr

def initialize(global_translations, global_pipe = None, global_compel=None,  global_config=None):
    """Initialise le module test."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    return TestModule(global_translations, global_pipe, global_compel,  global_config=None)

class TestModule:
    def __init__(self, global_translations, global_pipe=None, global_compel=None, global_config=None):
        """Initialise la classe TestModule."""
        self.global_translations = global_translations
        self.global_config = None
        if global_pipe is not None:
            self.global_pipe = global_pipe
        else:
            self.global_pie = None
        
        if global_compel is not None:
            self.global_combel = global_compel 
        else:
            self.global_compel = None
            
        
    
    
    def create_tab(self, module_translations):
        """
        Crée l'onglet Gradio pour ce module.

        Args:
            module_translations (dict): Le dictionnaire de traductions du module.

        Returns:
            gr.Tab: L'onglet Gradio créé.
        """
        with gr.Tab(translate("mon_nouveau_module", module_translations)) as tab:  # Crée un onglet avec un nom traduit
            gr.Markdown(f"## {translate('mon_nouveau_module_titre', module_translations)}")  # Titre de l'onglet (traduit)
            # Ici, vous ajouterez vos composants Gradio (gr.Textbox, gr.Button, etc.)
            # Pour l'instant, l'onglet est vide.
            
            # Exemple d'ajout d'un composant
            my_text_input = gr.Textbox(label=translate("my_text_input", module_translations))
            my_button = gr.Button(translate("my_button", module_translations))
            my_text_output = gr.Textbox(label=translate("my_text_output", module_translations))

            def my_function(text):
                return f"You entered: {text}"

            my_button.click(my_function, inputs=my_text_input, outputs=my_text_output)
            
        return tab
