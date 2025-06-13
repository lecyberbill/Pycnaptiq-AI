# civitai_browser_mod.py
import os
import json
from Utils.utils import txt_color, translate, GestionModule  # Import GestionModule

# Obtenir le chemin du fichier JSON du module
module_json_path = os.path.join(os.path.dirname(__file__), "civitai_browser_mod.json")

# Créer une instance de GestionModule pour gérer les dépendances
with open(module_json_path, 'r', encoding="utf-8") as f:
    module_data = json.load(f)
module_manager = GestionModule(translations=module_data["language"]["fr"])

# Maintenant, on peut faire les imports en toute sécurité
import gradio as gr
import requests
from tqdm import tqdm
import pandas as pd
from PIL import Image
import io
import numpy as np


def initialize(global_translations, global_pipe=None, global_compel=None, global_config=None):
    """Initialise le module Civitai Browser."""
    print(txt_color("[OK] ", "ok"), module_data["name"])
    return CivitaiBrowser(global_translations, global_pipe, global_compel, global_config)


class CivitaiBrowser:
    def __init__(self, global_translations, global_pipe=None, global_compel=None, global_config=None):
        self.current_page = 1  # Variable pour suivre la page actuelle
        self.global_translations = global_translations
        self.global_pipe = None
        self.global_compel = None
        self.global_config = None

    def create_tab(self, module_translations):
        """Crée l'onglet Gradio pour le navigateur Civitai."""
        with gr.Tab(translate("civitai_browser", module_translations)) as tab:
            gr.Markdown(f"## {translate('civitai_browser_title', module_translations)}")
            with gr.Row():
                with gr.Column(scale=1):
                    page_display = gr.Textbox(
                        label=translate("current_page", module_translations), 
                        interactive=False, 
                        value="Page 1"
                    )
                    prev_button = gr.Button(translate("previous", module_translations) + " ⬅️")
                    next_button = gr.Button("Suivant ➡️" + translate("next", module_translations))
                with gr.Column(scale=4):
                    with gr.Accordion(translate("advanced_search", module_translations), open=True):
                        limit = gr.Slider(
                            label=translate("limit", module_translations),
                            value=10,
                            step=1,
                            minimum=1,
                            maximum=200
                        )
                        nsfw = gr.Dropdown(
                            choices=["Soft", "Mature", "X", "All"],
                            value="Soft",
                            label=translate("nsfw", module_translations)
                        )
                        sort = gr.Dropdown(
                            choices=["Most Reactions", "Most Comments", "Newest", "Oldest"],
                            value="Newest",
                            label=translate("sort", module_translations)
                        )
                        period = gr.Dropdown(
                            choices=["All Time", "Year", "Month", "Week", "Day"],
                            value="Day",
                            label=translate("period", module_translations)
                        )
                    # Bouton pour lancer la recherche avec les paramètres avancés
                    refresh_button = gr.Button(translate("load", module_translations))
                    # Affichage de la galerie sous forme de HTML (grille)
            image_gallery = gr.HTML()
            state_module_translations = gr.State(module_translations)
            refresh_button.click(
                self.search_civitai,
                inputs=[limit, nsfw, sort, period, state_module_translations],
                outputs=[image_gallery, page_display],             
            )
            prev_button.click(
                self.previous_page,
                inputs=[limit, nsfw, sort, period, state_module_translations],
                outputs=[image_gallery, page_display]
            )
            next_button.click(
                self.next_page,
                inputs=[limit, nsfw, sort, period, state_module_translations],
                outputs=[image_gallery, page_display]
            )         
        return tab

    def search_civitai(self, limit, nsfw, sort, period, module_translations):
        """Effectue une recherche et affiche les résultats sur la première page."""
        self.current_page = 1  # Réinitialisation à la première page
        return self.fetch_images(limit, nsfw, sort, period, self.current_page, module_translations)

    def previous_page(self, limit, nsfw, sort, period, module_translations):
        """Passe à la page précédente si possible."""
        if self.current_page > 1:
            self.current_page -= 1
        return self.fetch_images(limit, nsfw, sort, period, self.current_page, module_translations)

    def next_page(self, limit, nsfw, sort, period, module_translations):
        """Passe à la page suivante."""
        self.current_page += 1
        return self.fetch_images(limit, nsfw, sort, period, self.current_page, module_translations)

    def fetch_images(self, limit, nsfw, sort, period, page, module_translations):
        base_url = "https://civitai.com/api/v1/images"
        params = {
            "limit": int(limit),
            "page": int(page),
            "nsfw": nsfw,
            "sort": sort,
            "period": period
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            images = data.get("items", [])

            html_gallery = """
            <style>
            /* Conteneur principal pour la grille et le défilement */
            .gallery-container {
                max-height: 80vh; /* Hauteur maximale du conteneur */
                overflow-y: auto; /* Active le défilement vertical */
                margin-top: 20px; /* Marge au-dessus de la grille */
                padding-right: 10px; /* Espace pour la scrollbar */
            }
            /* Grille d'images */
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            /* Conteneur de la carte avec hauteur fixe */
            .image-container {
                position: relative;
                height: 350px;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                background: #000;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            /* L'image occupe la partie supérieure, s'ouvre dans un nouvel onglet */
            .image-container a {
                display: block;
                height: 70%;
            }
            .image_card {
                width: 100%;
                height: 100%;
                object-fit: cover;
                cursor: pointer;
            }
            /* Bouton "Voir métadonnées" stylé */
            .meta-button {
                background: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 8px;
                width: 100%;
                cursor: pointer;
                font-weight: bold;
                margin-top: 4px;
            }
            .meta-button:hover {
                background: #000;
            }
            /* Overlay pour les métadonnées, couvrant toute la carte */
            .overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                color: #fff;
                display: none; /* Conservé - pour le toggle JS */
                flex-direction: column;
                /* Changé/Supprimé: justify-content et align-items pour permettre le scroll depuis le haut */
                justify-content: flex-start; /* Aligne le contenu en haut */
                align-items: center; /* Garde le centrage horizontal */
                padding: 20px; /* Conserve le padding */
                /* Ajouté: Espace pour le bouton fermer en haut peut être géré par le padding ou la position du contenu */
                padding-top: 50px; /* Augmente le padding haut pour laisser de la place au bouton fermer */
                box-sizing: border-box;
                z-index: 10;
            }
            .overlay.active {
                display: flex;
            }
            .overlay .close-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                background: none;
                border: none;
                font-size: 20px;
                color: #fff;
                cursor: pointer;
            }
            .meta-content {
                text-align: left;
                width: 90%; /* Limite la largeur (ou 100  pour cent i vous préférez utiliser tout l'espace horizontal) */
                max-width: 100%; /* Assurez-vous qu'il ne dépasse pas le parent */
            /* margin-top: 30px; */ /* Peut être retiré si padding-top sur .overlay gère l'espacement */
                margin-bottom: 20px; /* Ajoute un peu d'espace en bas */

                /* --- C'est la partie clé pour le scrolling --- */
                max-height: calc(100% - 20px); /* Hauteur max = 100 pour cent la hauteur restante - marge basse */
                        /* Ajustez ce calcul si nécessaire (ex: 100% - 40px si vous avez du padding bas aussi) */
                        /* Ou une valeur fixe: max-height: 250px; */
                overflow-y: auto; /* Affiche la barre de défilement verticale si besoin */
                /* --- Fin de la partie clé --- */

                /* Amélioration visuelle pour la barre de scroll (optionnel, dépend du navigateur) */
                scrollbar-width: thin; /* Pour Firefox */
                scrollbar-color: #666 #333; /* Couleur pouce/piste pour Firefox */
            }

            .meta-content::-webkit-scrollbar {
                width: 8px;
            }

            .meta-content::-webkit-scrollbar-track {
                background: #333;
                border-radius: 4px;
            }

            .meta-content::-webkit-scrollbar-thumb {
                background-color: #666;
                border-radius: 4px;
                border: 2px solid #333;
            }

            .overlay .close-btn {
                position: absolute;
                top: 15px; /* Ajusté pour être dans le nouveau padding-top */
                right: 15px;
                background: none;
                border: none;
                font-size: 24px; /* Légèrement plus grand peut-être */
                color: #fff;
                cursor: pointer;
                z-index: 11; /* Au-dessus du contenu scrollable */
            }

            .copy-btn {
                background: #007bff;
                color: white;
                padding: 5px 10px;
                border: none;
                cursor: pointer;
                margin-top: 5px;
                border-radius: 4px;
            }
            .copy-btn:hover {
                background: #0056b3;
            }
            </style>
            <div class="gallery-container">
            <div class="image-grid">
            """

            for i, image in enumerate(images):
                if image is None:
                    continue

                image_url = image.get("url", "")
                meta = image.get("meta") or {}
                image_size = image.get("Size", "N/A")
                model_used = meta.get("Model", "N/A")
                model_used = meta.get("Model", "N/A")
                model_used_tab = model_used.split(":")
                cfgScale = meta.get("cfgScale", "N/A")
                prompt = meta.get("prompt", "Aucun prompt")
                negative_prompt = meta.get("negativePrompt", "Aucun negative prompt")
                steps = meta.get("steps", "N/A")
                sampler = meta.get("sampler", "N/A")
                seed = meta.get("seed", "N/A")
                clip_skip = meta.get("Clip skip", "N/A")
                html_liste_model = "<ul>"
                if model_used_tab:
                    html_liste_model_items = []
                    for element in model_used_tab:
                        html_liste_model_items.append(f"<li>{element}</li>")
                    html_liste_model = "<ul>" + "".join(html_liste_model_items) + "</ul>"
                else:
                    html_liste_model = "N/A"

                html_gallery += f"""
                <div class="image-container">
                <a href="{image_url}" target="_blank">
                    <img src="{image_url}" class="image_card">
                </a>
                <button class="meta-button" data-overlay-id="overlay-{i}">
                    {translate("see_metadata", module_translations)}
                </button>
                <div class="overlay" id="overlay-{i}">
                    <button class="close-btn" data-overlay-id="overlay-{i}">✖ {translate("close", module_translations)}</button>
                    <div class="meta-content">
                    <h3>{translate("prompt", module_translations)}</h3>
                    <p id="prompt-{i}">{prompt}</p>
                    <button class="copy-btn" data-text-id="prompt-{i}">📋 {translate("copy", module_translations)}</button>
                    <h3>{translate("no_negative_prompt", module_translations)}</h3>
                    <p>{negative_prompt}</p>
                    <h3>{translate("other_metadata", module_translations)}</h3>
                    <ul>
                        <li><strong>{translate("image_size", module_translations)}:</strong> {image_size}</li>
                        <li><strong>{translate("model_used", module_translations)}:</strong> {html_liste_model}</li>
                        <li><strong>{translate("cfgScale", module_translations)}:</strong> {cfgScale}</li>
                        <li><strong>{translate("steps", module_translations)}:</strong> {steps}</li>
                        <li><strong>{translate("sampler", module_translations)}:</strong> {sampler}</li>
                        <li><strong>{translate("seed", module_translations)}:</strong> {seed}</li>
                        <li><strong>{translate("clip_skip", module_translations)}:</strong> {clip_skip}</li>
                    </ul>
                    </div>
                </div>
                </div>
                """

            html_gallery += "</div></div>"
            return html_gallery, f"Page {page}"

        except requests.exceptions.RequestException as e:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_recherche_civitai", module_translations), f": {e}")
            raise gr.Error(translate("erreur_recherche_civitai", module_translations) + f": {e}", 4.0)
            return "<p>Erreur lors de la recherche.</p>", f"Page {page}"
