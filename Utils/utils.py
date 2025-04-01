import os, sys
import importlib
import shutil
import gradio as gr
import torch
from pathlib import Path
import re
import html
import json
import requests
from tqdm import tqdm
from colorama import init, Fore, Style
from collections import defaultdict
import subprocess
import inspect
from compel import Compel, ReturnedEmbeddingsType
import gc


init()

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


def load_locales(lang="fr"):
    """Charge les traductions depuis un fichier JSON."""
    root_dir = Path(__file__).parent.parent
    chemin_dossier_locales = root_dir / "locales"
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
    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent
    chemin_dossier_locales = root_dir / "locales"
    
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
    root_dir = Path(__file__).parent.parent
    chemin_dossier_utils = root_dir / "html_util"
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
                    gr.Info(translate("mise_a_jour_du", translations) + f": {chemin_fichier_html}", 3.0)
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
            return translate("mise_a_jour_du", translations) + f": {chemin_fichier_html}"

    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_lors_generation_html", translations), f": {e}")
        raise gr.Error(translate("erreur_lors_generation_html", translations) + f": {e}")
        return translate("erreur_lors_generation_html", translations) + f": {e}"


def charger_configuration():
    """Loads the configuration from the config.json file.
        Args:
        chemin_image (str): config.json file
        Return:
        Return a dictionary with the configuration values
    """

    try:
        # Get the script's directory
        root_dir = Path(__file__).parent.parent
        config_dir = root_dir / "config"
        config_json = config_dir / "config.json"
        chemin_styles = config_dir / "styles.json"
        
        with open(config_json, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Convert relative paths to absolute paths if necessary
        for key in ["MODELS_DIR", "VAE_DIR", "SAVE_DIR", "LORAS_DIR", "INPAINT_MODELS_DIR"]:
            if key in config:  # Check if the key exists
                if not os.path.isabs(config[key]):
                    # If it's a relative path, join it with the root directory
                    config[key] = os.path.abspath(os.path.join(root_dir, config[key]))

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
def lister_fichiers(dir, translations, ext=".safetensors", gradio_mode=False):
    """List files in a directory with a specific extension."""
    
    root_dir = Path(__file__).parent.parent
    # Try to get the list of files from the specified directory.

    if not os.path.isabs(dir):
        dir = os.path.abspath(os.path.join(root_dir, dir))

        

    try:
        fichiers = [f for f in os.listdir(dir) if f.endswith(ext)]
        
        # If no files are found, print a specific message and return an empty list.
        if not fichiers:
            print(txt_color("[INFO] ","info"), translate("aucun_modele",translations))
            if gradio_mode:
                raise gr.Error(translate("aucun_modele",translations), 4.0)
            return [translate("aucun_modele",translations)]
            
    except FileNotFoundError:
        # If the directory doesn't exist, print a specific error message and return an empty list. 
        print(txt_color("[ERREUR] ","erreur"),translate("directory_not_found",translations),f" {dir}")
        if gradio_mode:
            raise gr.Error(translate("repertoire_not_found",translations), 4.0)
        return [translate("repertoire_not_found", translations)]
        
    else:
        # If files are found, print them out and return the file_list. 
        print(txt_color("[INFO] ","info"),translate("files_found_in",translations),f" {dir}: {fichiers}")
        if gradio_mode:
            gr.Info(translate("files_found_in",translations) + f": {dir}: {fichiers}", 3.0)
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

def enregistrer_image(image, chemin_image, translations, IMAGE_FORMAT):
    """Enregistre l'image et écrit les métadonnées."""
    try:
        image.save(chemin_image, format=IMAGE_FORMAT)
        
        print(txt_color("[OK] ", "ok"), translate("image_sauvegarder", translations), f" {chemin_image}")
        gr.Info(translate("image_sauvegarder", translations) + f": {chemin_image}", 3.0)

    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), translate("erreur_sauvegarde_image", translations), f" {e}")
        raise gr.Error(translate("erreur_sauvegarde_image", translations) + f" {e}", 4.0)
        

class GestionModule:
    """
    Classe pour gérer le chargement et l'initialisation des modules.
    """

    def __init__(self, modules_dir="modules", translations=None, language="fr", global_pipe=None, global_compel=None, config=None):
        """
        Initialise le gestionnaire de modules.

        Args:
            modules_dir (str): Le répertoire contenant les modules.
            translations (dict): Le dictionnaire de traductions.
        """
        self.modules_dir = modules_dir
        self.language = language
        self.translations = translations
        self.modules = {}
        self.tabs = {}
        self.modules_names = []
        self.global_pipe = global_pipe
        self.global_compel = global_compel
        self.config = config
        self.js_code = ""
        

    def verifier_version(self, package_name, min_version):
        """
        Vérifie si le package est installé et satisfait la version minimale requise.
        
        Args:
            package_name (str): Le nom du package tel qu'il est utilisé pour l'installation (ex: "Pillow").
            min_version (str): La version minimale requise (ex: "8.0.0").
        
        Returns:
            (bool, str): Un tuple indiquant si la version est satisfaisante et la version installée (ou None si non installé).
        """
        try:
            installed_version = get_version(package_name)
            if pkg_version.parse(installed_version) >= pkg_version.parse(min_version):
                return True, installed_version
            else:
                return False, installed_version
        except PackageNotFoundError:
            return False, None

    def check_and_install_dependencies(self, module_json_path):
        """
        Vérifie et installe les dépendances d'un module à partir de son fichier JSON.
        La configuration des dépendances dans le JSON peut être soit une liste de chaînes (nom du package)
        ou une liste de dictionnaires avec des clés 'package', 'import' (optionnel) et 'min_version'.

        Exemple de dépendance dans le JSON :
        [
            "numpy",
            {
                "package": "Pillow",
                "import": "PIL",
                "min_version": "8.0.0"
            }
        ]
        """
        try:
            with open(module_json_path, 'r', encoding="utf-8") as f:
                module_data = json.load(f)
        except FileNotFoundError:
            print(txt_color("[ERREUR] ", "erreur"), translate("module_json_not_found", self.translations).format(module_json_path))
            return False
        except json.JSONDecodeError:
            print(txt_color("[ERREUR] ", "erreur"), translate("module_json_decode_error", self.translations).format(module_json_path))
            return False

        if "dependencies" not in module_data:
            print(txt_color("[INFO] ", "info"), translate("module_no_dependencies", self.translations).format(module_json_path))
            return True

        # Chemin vers l'exécutable pip dans l'environnement virtuel
        venv_pip_path = sys.executable.replace("python", "pip")


        dependencies = module_data["dependencies"]

        print(txt_color("[INFO] ", "info"), translate("module_checking_dependencies", self.translations).format(module_data.get('name', 'module')))

        for dep in dependencies:
            # On supporte deux formats : une chaîne ou un dictionnaire
            if isinstance(dep, str):
                package_name = dep
                min_version = None
            elif isinstance(dep, dict):
                package_name = dep.get("package")
                min_version = dep.get("min_version")
                # Optionnellement, on peut utiliser 'import' pour le nom d'import si nécessaire,
                # mais ici on vérifie directement via le nom du package.
            else:
                print(txt_color("[ERREUR] ", "erreur"), f"Dépendance au format inconnu: {dep}")
                continue

            if min_version:
                valid, installed_version = self.verifier_version(package_name, min_version)
                if valid:
                    print(txt_color("[INFO] ", "info"), translate("dependency_already_installed", self.translations).format(f"{package_name} (v{installed_version})"))
                else:
                    if installed_version:
                        print(txt_color("[INFO] ", "info"), translate("dependency_outdated", self.translations).format(package_name, installed_version, min_version))
                    else:
                        print(txt_color("[INFO] ", "info"), translate("dependency_missing", self.translations).format(package_name))
                    try:
                        subprocess.check_call([venv_pip_path, "install", f"{package_name}>={min_version}"])
                        importlib.invalidate_caches()
                        # Vérifier de nouveau
                        valid, installed_version = self.verifier_version(package_name, min_version)
                        if valid:
                            print(txt_color("[INFO] ", "info"), translate("dependency_installed_success", self.translations).format(f"{package_name} (v{installed_version})"))
                        else:
                            print(txt_color("[ERREUR] ", "erreur"), translate("dependency_install_error", self.translations).format(package_name, f"version {installed_version}"))
                            return False
                    except subprocess.CalledProcessError as e:
                        print(txt_color("[ERREUR] ", "erreur"), translate("dependency_install_error", self.translations).format(package_name, e))
                        return False
            else:  # No min_version specified
                spec = importlib.util.find_spec(package_name)
                if spec is None:
                    print(txt_color("[INFO] ", "info"), f"Dependency '{package_name}' missing. Attempting installation...")
                    try:
                        result = subprocess.run([venv_pip_path, "install", package_name], capture_output=True, text=True, check=True)
                       
                        importlib.invalidate_caches()
                        print(txt_color("[INFO] ", "info"), f"Dependency '{package_name}' installed successfully.")
                    except subprocess.CalledProcessError as e:
                        print(txt_color("[ERREUR] ", "erreur"), f"Error installing {package_name}: return code {e.returncode}, stderr:\n{e.stderr}") # Improved error message
                        print(txt_color("[ERREUR] ", "erreur"), translate("dependency_install_manual", self.translations).format(package_name, sys.executable.replace("python", "pip")))
                        return False
                    except FileNotFoundError:
                        print(txt_color("[ERREUR] ", "erreur"), f"pip executable not found at: {venv_pip_path}")
                        print(txt_color("[ERREUR] ", "erreur"), translate("dependency_install_manual", self.translations).format(package_name, sys.executable.replace("python", "pip")))
                        return False
                    except Exception as e:
                        print(txt_color("[ERREUR] ", "erreur"), f"An unexpected error occurred while installing {package_name}: {e}")
                        print(txt_color("[ERREUR] ", "erreur"), translate("dependency_install_manual", self.translations).format(package_name, sys.executable.replace("python", "pip")))
                        return False
                else:
                    print(txt_color("[INFO] ", "info"), f"Dependency '{package_name}' already installed.")

        return True

    def charger_module(self, module_name):
        """
        Charge un module à partir de son nom de fichier et charge ses traductions.

        Args:
            module_name (str): Le nom du fichier du module (sans l'extension .py).

        Returns:
            module: Le module chargé ou None si le chargement a échoué.
        """
        try:
            module_path = os.path.join(self.modules_dir, module_name + "_mod.py")
            metadata_path = os.path.join(self.modules_dir, module_name + "_mod.json")

            if not os.path.exists(module_path):
                print(txt_color("[ERREUR] ", "erreur"), translate("module_not_exist", self.translations).format(module_name))
                return None

            # Check and install dependencies BEFORE importing the module
            if os.path.exists(metadata_path):
                if not self.check_and_install_dependencies(metadata_path):
                    print(txt_color("[ERREUR] ", "erreur"), translate("dependency_install_failed", self.translations).format(module_name))
                    return None


            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)



            # Vérifier si le module a un fichier de métadonnées
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    module.metadata = metadata
            else:
                module.metadata = {}

            module.translations = self.charger_traductions_module(module, module_name, self.language)

            self.modules[module_name] = module
            print(txt_color("[OK] ", "ok"), translate("module_loaded", self.translations).format(module_name))
            return module
        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), translate("module_load_error", self.translations).format(module_name, e))
            return None

    def charger_traductions_module(self, module, module_name, language):
        """
        Charge les traductions d'un module à partir de son fichier JSON de métadonnées.

        Args:
            module: Le module chargé.
            module_name (str): Le nom du module.
            language (str): La langue à charger (par exemple, "fr", "en").
        """
        if not hasattr(module, "metadata"):
            print(txt_color("[ERREUR] ", "erreur"), translate("module_no_metadata", self.translations).format(module_name))
            return self.translations

        metadata = module.metadata
        module_translations = {}

        if "language" in metadata and language in metadata["language"]:
            return metadata["language"][language]
        else:
            print(txt_color("[ERREUR] ", "erreur"), translate("module_translations_not_found", self.translations).format(module_name, language))
        
        merged_translations = self.translations.copy()
        merged_translations.update(module_translations)

        return merged_translations      

    def initialiser_module(self, module_name, *args, **kwargs):
        """Initialise un module chargé."""
        module = self.modules.get(module_name)
        if module is None:
            print(txt_color("[ERREUR] ", "erreur"), translate("module_not_loaded", self.translations).format(module_name))
            return None

        try:
            init_func_name = module.metadata.get("init_function", "initialize")
            if hasattr(module, init_func_name):
                init_func = getattr(module, init_func_name)
                # Correctly pass only four arguments
                instance = init_func(self.translations, self.global_pipe, self.global_compel, self.config, *args, **kwargs)
                module.instance = instance
                print(txt_color("[OK] ", "ok"), translate("module_initialized", self.translations).format(module_name))
                # Collect JavaScript code if the module has it
                if hasattr(module.instance, "get_module_js"):
                    self.js_code += module.instance.get_module_js() 
                return instance
            else:
                print(txt_color("[ERREUR] ", "erreur"), translate("module_no_init_function", self.translations).format(module_name, init_func_name))
                return None
        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), translate("module_init_error", self.translations).format(module_name, e))
            return None

    def charger_tous_les_modules(self):
        """Charge tous les modules dans le répertoire spécifié."""
        for filename in os.listdir(self.modules_dir):
            if filename.endswith("_mod.py"):
                print(txt_color("[INFO] ", "info"), translate("module_loading_attempt", self.translations).format(filename))
                # Charger le module
                module_name = filename[:-7]  # Supprimer l'extension _mod.py
                module = self.charger_module(module_name)
                if module:
                    module.translations = self.charger_traductions_module(module, module_name, self.language)
                    # Initialize the module and store the instance
                    instance = self.initialiser_module(module_name)
                    if instance:
                        module.instance = instance
                        self.modules_names.append(module_name)

    def creer_onglet_module(self, module_name, translations):
        """Crée un onglet Gradio à partir d'un module."""
        module = self.modules.get(module_name)
        if module is None:
            print(txt_color("[ERREUR] ", "erreur"), translate("module_not_loaded", self.translations).format(module_name))
            return None

        try:
            module_translations = module.translations if hasattr(module, "translations") else {}
            tab_name = module.metadata.get("tab_name", module_name)

            if hasattr(module, "instance") and hasattr(module.instance, "create_tab"):
                tab = module.instance.create_tab(module_translations)
                self.tabs[module_name] = tab
                print(txt_color("[OK] ", "ok"), translate("tab_created_for_module", self.translations).format(tab_name, module_name))
                return tab
            else:
                print(txt_color("[ERREUR] ", "erreur"), translate("module_no_create_tab", self.translations).format(module_name))
                return None
        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), translate("tab_creation_error", self.translations).format(module_name, e))
            return None

    def creer_tous_les_onglets(self, translations):
        """Crée tous les onglets Gradio pour les modules chargés."""
        print(txt_color("[INFO] ", "info"), translate("creating_all_tabs", self.translations))
        for module_name in self.modules:
            self.creer_onglet_module(module_name, self.translations)
    
    def get_loaded_modules(self):
        """Returns a list of the names of the loaded modules."""
        return self.modules_names
    
    def get_js_code(self):
        """Return the javascript code"""
        return self.js_code

def decharger_modele(pipe, compel, translations):
    """Libère proprement la mémoire GPU en déplaçant temporairement le modèle sur CPU."""

    if pipe is not None:
        try:
            print(txt_color("[INFO] ", "info"), translate("dechargement_modele", translations))

            # Déplacer le modèle vers CPU avant suppression pour libérer la VRAM
            pipe.to("cpu")
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
            if compel is not None:
                del compel

            # Nettoyage de la mémoire GPU et RAM
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            gc.collect()

            print(txt_color("[OK] ", "ok"), translate("modele_precedent_decharge", translations))
        except Exception as e:
            print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_dechargement_modele', translations)}: {e}")
        finally:
            pipe, compel = None, None
    else:
        return None, None
    return pipe, compel


def check_gpu_availability(translations):
    """
    Checks if a CUDA-enabled GPU is available and configures PyTorch accordingly.

    Args:
        translations (dict): The translation dictionary for localized messages.

    Returns:
        tuple: A tuple containing:
            - device (str): The device to use ("cuda" or "cpu").
            - torch_dtype (torch.dtype): The recommended data type (torch.float16 or torch.float32).
            - vram_total_gb (float): The total VRAM in GB (or 0 if no GPU is available).
    """
    if torch.cuda.is_available():
        gpu_id = 0  # GPU ID (adjust if needed)
        vram_total = torch.cuda.get_device_properties(gpu_id).total_memory  # in bytes
        vram_total_gb = vram_total / (1024 ** 3)  # Convert to GB

        print(translate("vram_detecte", translations), f"{txt_color(f'{vram_total_gb:.2f} Go', 'info')}")

        # Enable expandable_segments if VRAM < 10 GB
        if vram_total_gb < 10:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True max_split_size_mb:512"
            print(translate("pytroch_active", translations))

        device = "cuda"
        torch_dtype = torch.float16
    else:
        print(txt_color(translate("cuda_dispo", translations), "erreur"))
        device = "cpu"
        torch_dtype = torch.float32
        vram_total_gb = 0

    print(txt_color(f'{translate("utilistation_device", translations)} : {device} + dtype {torch_dtype}', 'info'))
    return device, torch_dtype, vram_total_gb