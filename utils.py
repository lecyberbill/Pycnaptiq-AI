import os
import gradio as gr
import torch
from pathlib import Path
import re
import html
import json
import requests
from tqdm import tqdm
import torch
from colorama import init, Fore, Style
from collections import defaultdict

init()

def load_locales(lang="fr"):
    """Charge les traductions depuis un fichier JSON."""
    chemin_dossier_locales = Path(__file__).parent / "locales"
    chemin_fichier_langue = chemin_dossier_locales / f"{lang}.json"

    try:
        with open(chemin_fichier_langue, "r", encoding="utf-8") as f:
            translations = json.load(f)
        print(txt_color("[OK] ","ok"),translate("langue_charge",translations),f" {lang}")
        return translations
    except FileNotFoundError:
        print(txt_color("[ERREUR] ","erreur"),translate("erreur_fichier_langue",translations), f": {chemin_fichier_langue}")
        return {}
    except json.JSONDecodeError:
        print(txt_color("[ERREUR] ","erreur"),translate("erreur_decodage_json",translations), f": {chemin_fichier_langue}")
        return {}
    

def translate(key, translations):
    """Traduit une clé en utilisant le dictionnaire de traductions."""
    return translations.get(key, f"[{key}]") #Si il ne trouve pas la valeur on affiche la clé

def get_language_options(translations):
    """Récupère la liste des langues disponibles."""
    chemin_dossier_locales = Path(__file__).parent / "locales"
    
    languages = []
    for filename in os.listdir(chemin_dossier_locales):
        if filename.endswith(".json"):
            languages.append(filename[:-5])  # Retire l'extension .json
    try:
        languages.remove("template")
    except:
        pass
    print(txt_color("[INFO] ","info"), translate("langue_disponible",translations), f": {languages}")
    return languages


def enregistrer_etiquettes_image_html(chemin_image, etiquettes, translations, is_last_image=False):
    """
    Enregistre les étiquettes d'une image dans un fichier HTML avec affichage de l'image et tableau stylisé (sans jQuery UI).
    Gère la réouverture du fichier HTML pour ajouter de nouvelles images.

    Args:
        chemin_image (str): Chemin vers le fichier image .jpg.
        etiquettes (dict): Dictionnaire d'étiquettes et de leurs valeurs.
        is_last_image (bool): Indique si c'est la dernière image à traiter.
    """
    chemin_dossier_utils = Path(__file__).parent / "html_util"
    chemin_jquery = chemin_dossier_utils / "jquery.min.js"
    chemin_magnific_popupCSS = chemin_dossier_utils / "magnific-popup.css"
    chemin_magnific_popupJS = chemin_dossier_utils / "jquery.magnific-popup.min.js"

    with open(chemin_jquery, 'r', encoding='utf-8') as f:
        contenu_jquery = f.read()

    with open(chemin_magnific_popupCSS, 'r', encoding='utf-8') as f:
        contenu_CSS = f.read()

    with open(chemin_magnific_popupJS, 'r', encoding='utf-8') as f:
        contenu_popupJS = f.read()

    title_lien_html = html.escape(etiquettes.get("Prompt"))

    try:
        nom_fichier_html = "rapport.html"
        chemin_fichier_html = os.path.join(os.path.dirname(chemin_image), nom_fichier_html)

        # Contenu HTML à ajouter pour chaque image
        image_html = ""

        # Ajouter les informations de l'image, l'image et les étiquettes dans un div avec un tableau
        image_html += "<div class='image-item'>\n"  # Début du div pour l'image
        image_html += "    <div class='image-container'>\n"  # Conteneur flex pour l'image et le tableau
        image_html += f"   <a class='image-popup' href='{os.path.basename(chemin_image)}' title='{title_lien_html}' target='_blank'><img src='{os.path.basename(chemin_image)}' alt='Image'></a>\n"  # Afficher l'image
        image_html += "        <div class='etiquettes'>\n"  # Début du div pour les étiquettes
        image_html += "             <table border='1'>\n"
        for etiquette, valeur in etiquettes.items():
            image_html += f"             <tr><td>{etiquette}</td><td>{valeur}</td></tr>\n"
        image_html += "             </table>\n"
        image_html += "       </div>\n"  # Fin du div pour les étiquettes
        image_html += "    </div>\n"  # Fin du conteneur flex
        image_html += "</div>\n\n"  # Fin du div pour l'image

        # Contenu à écrire dans le fichier HTML
        # Utilisation d'un dictionnaire global par chemin de fichier
        global html_contenu_buffer
        if 'html_contenu_buffer' not in globals():
            html_contenu_buffer = {}

        if chemin_fichier_html not in html_contenu_buffer:
            html_contenu_buffer[chemin_fichier_html] = []  # Initialise une liste pour chaque fichier

        html_contenu_buffer[chemin_fichier_html].append(image_html)  # Ajoute le contenu à la liste

        # Gestion de l'ouverture et de la fermeture du fichier HTML (seulement si c'est la dernière image)
        if is_last_image:
            # Si le fichier existe déjà
            if os.path.exists(chemin_fichier_html):
                with open(chemin_fichier_html, "r", encoding='utf-8') as f:
                    contenu = f.read()

                position_body = contenu.rfind("</body>")
                position_html = contenu.rfind("</html>")

                if position_body != -1 and position_html != -1 and position_body < position_html:
                    # Insérer le nouveau contenu avant </body> et avant </html>
                    nouveau_contenu = (
                            contenu[:position_body]
                            + "".join(html_contenu_buffer[chemin_fichier_html])
                            + contenu[position_body:position_html]
                            + contenu[position_html:]
                    )

                    with open(chemin_fichier_html, "w", encoding='utf-8') as f:
                        f.write(nouveau_contenu)

                    print(txt_color("[OK] ", "ok"), translate("mise_a_jour_du", translations),
                          txt_color(f"{chemin_fichier_html}", "ok"))
                    # réinitialise le buffer
                    html_contenu_buffer.pop(chemin_fichier_html, None)
                    return translate("mise_a_jour_du", translations) + f": {chemin_fichier_html}"

            else:  # Fichier n'existe pas
                with open(chemin_fichier_html, 'w', encoding='utf-8') as f:
                    f.write("<!DOCTYPE html>\n")
                    f.write("<html>\n")
                    f.write("<head>\n")
                    f.write("<title>Recutecapitulatif des images</title>\n")
                    f.write(f"<script>{contenu_jquery}</script>\n")
                    f.write(f"<script>{contenu_popupJS}</script>\n")
                    f.write("<script>\n")
                    f.write("$(document).ready(function() {\n")
                    f.write("  $('.image-popup').magnificPopup({\n")
                    f.write("    type: 'image',\n")
                    f.write("    closeOnContentClick: true,  // Ferme la popup en cliquant sur l'image\n")
                    f.write("    closeBtnInside: false,      // Affiche le bouton de fermeture à l'extérieur de l'image\n")
                    f.write("    mainClass: 'mfp-with-zoom', // Ajoute une classe pour une animation de zoom\n")
                    f.write("    image: {\n")
                    f.write("      verticalFit: true, // Ajuste l'image à la hauteur de la fenêtre\n")
                    f.write("      titleSrc: 'title' // Affiche l'attribut 'title' comme titre de l'image dans la popup\n")
                    f.write("    },\n")
                    f.write("    zoom: {\n")
                    f.write("      enabled: true, // Active l'animation de zoom\n")
                    f.write("      duration: 300 // Durée de l'animation de zoom en millisecondes\n")
                    f.write("    }\n")
                    f.write("  });\n")
                    f.write("});\n")
                    f.write("</script>\n")
                    f.write("<style>\n")  # Style CSS personnalisé
                    f.write("body {\n")
                    f.write("  background-color: black;\n")  # Fond noir
                    f.write("  color: white;\n")  # Texte en blanc
                    f.write("  font-family: Arial, sans-serif;\n")  # Police
                    f.write("}\n")
                    f.write(".image-item {\n")
                    f.write("  margin-bottom: 20px;\n")  # Espacement entre les items
                    f.write("}\n")
                    f.write(".image-container {\n")
                    f.write("  display: flex;\n")  # Utilisation de flexbox
                    f.write("  flex-wrap: wrap;\n")  # Pour gérer les débordements
                    f.write("  margin-bottom: 10px;\n")
                    f.write("  padding: 10px;\n")
                    f.write("  background-color: #222;\n")  # Fond sombre pour la zone image
                    f.write("  border-radius: 8px;\n")
                    f.write("}\n")
                    f.write("img {\n")
                    f.write("  max-width: 300px;\n")
                    f.write("  height: auto;\n")
                    f.write("  margin-right: 20px;\n")  # Espacement entre l'image et le tableau
                    f.write("}\n")
                    f.write(".etiquettes {\n")
                    f.write("  flex: 1;\n")  # Permet à la section des étiquettes de prendre le reste de l'espace
                    f.write("}\n")
                    f.write("table {\n")
                    f.write("  width: 100%;\n")
                    f.write("  border-collapse: collapse;\n")
                    f.write("}\n")
                    f.write("th, td {\n")
                    f.write("  padding: 8px;\n")
                    f.write("  border: 1px solid #ddd;\n")
                    f.write("  text-align: left;\n")
                    f.write("}\n")
                    f.write(f"{contenu_CSS}\n")
                    f.write("</style>\n")
                    f.write("</head>\n")
                    f.write("<body>\n")  # Début du body
                    f.write("".join(html_contenu_buffer[chemin_fichier_html]))  # Ajouter le contenu de la première image
                    f.write("</body>\n")  # Fermeture du body
                    f.write("</html>\n")
                print(txt_color("[OK] ", "ok"), translate("fichier_cree", translations), f": {chemin_fichier_html}")
                # reset buffer
                html_contenu_buffer.pop(chemin_fichier_html, None)
                return translate("fichier_cree", translations) + f": {chemin_fichier_html}"

    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_lors_generation_html", translations), f": {e}")


def charger_configuration():
    """Loads the configuration from the config.json file.
        Args:
        chemin_image (str): config.json file
        Return:
        Return a dictionary with the configuration values
    """

    try:
        # Get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = Path(__file__).parent / "config"
        config_json = config_dir / "config.json"
        chemin_styles = config_dir / "styles.json"
        
        with open(config_json, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Convert relative paths to absolute paths
        for key in ["MODELS_DIR", "VAE_DIR", "SAVE_DIR", "LORAS_DIR"]:
            if not os.path.isabs(config[key]):
                config[key] = os.path.join(script_dir, config[key])
        
        
        # Chargement des styles
        if os.path.exists(chemin_styles):
             with open(chemin_styles, "r", encoding="utf-8") as fichier_styles:
                config["STYLES"] = json.load(fichier_styles)

        else:
             print(f"{txt_color('[ERREUR]','erreur')}", f"Error: styles.json not found at {chemin_styles}")
        
        print(txt_color("[OK] ","ok"),"Configuration successfully loaded")       
        return config

    except FileNotFoundError:
        print(txt_color("[ERREUR] ","erreur"), f"Error loading configuration: config file not found")
        return {}
    except json.JSONDecodeError as e:
        print(txt_color("[ERREUR] ","erreur"), f"Error loading configuration: JSON decode error: {e}")
        return {}
    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"), f"Error loading configuration: An error occurred: {e}")
        return {}

        
        
#fonction pour changer le thème de gradio
def gradio_change_theme(theme):
  """
  Fonction pour choisir un thème Gradio 5 avec match-case.

  Args:
    nom_theme: Le nom du thème à appliquer (str).

  Returns:
    Le thème Gradio 5 correspondant (gr.theme.Theme) ou None si le thème n'existe pas.
  """

  theme = theme.lower() # Pour ignorer la casse

  match theme:
    case "base":
      return gr.themes.Base()
    case "default":
      return gr.themes.Default()
    case "origin":
      return gr.themes.Origin()
    case "citrus":
      return gr.themes.Citrus()
    case "monochrome":
      return gr.themes.Monochrome()
    case "soft":
      return gr.themes.Soft()
    case "glass":
      return gr.themes.Glass()
    case "ocean":
      return gr.themes.Ocean()
    case _:  # Cas par défaut (si aucun thème ne correspond)
      return gr.themes.Default()
      
      
# liste fichiers .safetensors
def lister_fichiers(dir, translations, ext=".safetensors"):
    """List files in a directory with a specific extension."""
    
    # Try to get the list of files from the specified directory. 
    try:
        fichiers = [f for f in os.listdir(dir) if f.endswith(ext)]
        
        # If no files are found, print a specific message and return an empty list.
        if not fichiers:
            print(txt_color("[INFO] ","info"), translate("aucun_modele",translations))
            return [translate("aucun_modele",translations)]
            
    except FileNotFoundError:
        # If the directory doesn't exist, print a specific error message and return an empty list. 
        print(txt_color("[ERREUR] ","erreur"),translate("directory_not_found",translations),f" {dir}")
        return [translate("repertoire_not_found", translations)]
        
    else:
        # If files are found, print them out and return the file_list. 
        print(txt_color("[INFO] ","info"),translate("files_found_in",translations),f" {dir}: {fichiers}")
        return fichiers
        
        
def telechargement_modele(lien_modele, nom_fichier, models_dir,translations):
    """Télécharge un modèle depuis un lien et l'enregistre dans models_dir."""
    try:
        print(txt_color("[INFO] ","info"),translate("telechargement_modele_commence",translations))
        response = requests.get(lien_modele, stream=True)
        response.raise_for_status()  # Vérifie si le téléchargement a réussi
        taille_totale = int(response.headers.get('content-length', 0))  # Taille du fichier
        
        chemin_destination = os.path.join(models_dir, nom_fichier)  # Chemin complet
        with open(chemin_destination, "wb") as f:
            with tqdm(total=taille_totale, unit='B', unit_scale=True, desc=nom_fichier) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filtre les chunks vides
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(txt_color("[ok] ","ok"),translate("modele_telecharge",translations), f": {nom_fichier}")
        return True  # Indique que le téléchargement a réussi

    except requests.exceptions.RequestException as e:
        print(txt_color("[ERREUR] ","erreur"),translate("erreur_telechargement_modele",translations), f" : {e}")
        return False  # Indique que le téléchargement a échoué
    except Exception as e:
        print(txt_color("[ERREUR] ","erreur"),translate("erreur_telechargement_modele",translations), f" : {e}")
        return False
        

def txt_color(texte, statut):
    """
    Ajoute de la couleur à un texte en fonction du statut spécifié.

    Args:
        texte (str): La chaîne de caractères à colorer.
        statut (str): Le statut du message, parmi 'info', 'erreur', 'ok'.

    Returns:
        str: La chaîne de caractères colorée en fonction du statut.
             - 'info': retourne le texte en bleu.
             - 'erreur': retourne le texte en rouge.
             - 'ok': retourne le texte en vert.
             Pour tout autre statut ou si le statut n'est pas reconnu,
             retourne le texte sans couleur.
    """
    if statut == "erreur":
        return Fore.RED + texte + Style.RESET_ALL
    elif statut == "ok":
        return Fore.GREEN + texte + Style.RESET_ALL
    elif statut == "info":
        return Fore.CYAN + texte + Style.RESET_ALL
    elif statut == "debug":
        return Fore.MAGENTA + texte + Style.RESET_ALL
    else:
        return texte
        
        
def cprint(*args, statut=None, **kwargs):
    """
    Fonction d'impression personnalisée qui ajoute de la couleur si le paramètre 'statut' est fourni.

    Args:
        *args: Arguments à passer à la fonction print originale.
        statut (str, optional): Le statut pour la coloration ('info', 'erreur', 'ok'). Par défaut: None (pas de couleur).
        **kwargs: Arguments additionnels à passer à la fonction print originale (sep, end, file, flush).
    """
    if statut:
        colored_args = [txt_color(str(arg), statut) for arg in args] # Utilisation d'une compréhension de liste pour simplifier
        print(*colored_args, **kwargs) # Utiliser la fonction print originale (sans la redéfinir)
    else:
        print(*args, **kwargs) # Utiliser la fonction print originale sans coloration 
        # cprint("VRAM détectée :", f'{vram_total_gb:.2f} Go', statut='info')
        
def str_to_bool(s):
    """
    Convertit une chaîne de caractères en une valeur booléenne.

    Cette fonction convertit la chaîne d'entrée en minuscules et vérifie si celle-ci
    correspond à une des valeurs considérées comme représentant True.
    Les valeurs reconnues comme True sont : "true", "1", "yes", "oui", "o", "ok" et "y".
    Toute autre valeur sera évaluée à False.

    Paramètres :
      s (str) : La chaîne de caractères à convertir.

    Retourne :
      bool : True si la chaîne représente une valeur vraie, sinon False.

    Exemples :
      >>> str_to_bool("True")
      True
      >>> str_to_bool("false")
      False
      >>> str_to_bool("1")
      True
      >>> str_to_bool("0")
      False
      False
    """
    return s.lower() in ("true", "1", "yes", "y", "ok", "oui", "o")

def enregistrer_image(image, chemin_image, donnees_xmp, translations, IMAGE_FORMAT):
    """Enregistre l'image et écrit les métadonnées."""
    try:
        image.save(chemin_image, format=IMAGE_FORMAT)
        
        print(txt_color("[OK] ", "ok"), translate("image_sauvegarder", translations), f" {chemin_image}")
    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_sauvegarde_image", translations), f" {e}")


    