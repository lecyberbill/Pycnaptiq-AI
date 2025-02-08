import os
import torch



def fichier_recap(chemin_image, etiquettes):
    """
    Enregistre les étiquettes d'une image dans un fichier texte.

    Args:
        chemin_image (str): Chemin vers le fichier image .jpg.
        etiquettes (dict): Dictionnaire d'étiquettes et de leurs valeurs.
    """

    try:
        # 1. Créer le chemin du fichier texte
        nom_fichier_txt = os.path.splitext(os.path.basename(chemin_image))[0] + ".txt"
        chemin_fichier_txt = os.path.join(os.path.dirname(chemin_image), nom_fichier_txt)

        # 2. Écrire les informations dans le fichier texte
        with open(chemin_fichier_txt, 'w') as f:
            f.write(f"Image: {chemin_image}\n")
            for etiquette, valeur in etiquettes.items():
                f.write(f"{etiquette}: {valeur}\n")

        print(f"Les étiquettes ont été enregistrées dans : {chemin_fichier_txt}")

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        
  

def enregistrer_etiquettes_image_html(chemin_image, etiquettes):
    """
    Enregistre les étiquettes d'une image dans un fichier HTML avec affichage de l'image et tableau stylisé (sans jQuery UI).
    Gère la réouverture du fichier HTML pour ajouter de nouvelles images.

    Args:
        chemin_image (str): Chemin vers le fichier image .jpg.
        etiquettes (dict): Dictionnaire d'étiquettes et de leurs valeurs.
    """
    chemin
    with open(chemin_jquery, 'r') as f:
      contenu_jquery = f.read() 
    
    try:
        nom_fichier_html = "rapport.html"
        chemin_fichier_html = os.path.join(os.path.dirname(chemin_image), nom_fichier_html)

        # Contenu HTML à ajouter pour chaque image
        image_html = ""

        # Ajouter les informations de l'image, l'image et les étiquettes dans un div avec un tableau
        image_html += "<div class='image-item'>\n"  # Début du div pour l'image
        image_html += "    <div class='image-container'>\n"  # Conteneur flex pour l'image et le tableau
        image_html += f"   <a class='image-popup' href='{os.path.basename(chemin_image)}' title='{etiquette.Prompt}' target='_blank'><img src='{os.path.basename(chemin_image)}' alt='Image'></a>\n"  # Afficher l'image
        image_html += "        <div class='etiquettes'>\n"  # Début du div pour les étiquettes
        image_html += "             <table border='1'>\n"
        for etiquette, valeur in etiquettes.items():
            image_html += f"             <tr><td>{etiquette}</td><td>{valeur}</td></tr>\n"
        image_html += "             </table>\n"
        image_html += "       </div>\n"  # Fin du div pour les étiquettes
        image_html += "    </div>\n"  # Fin du conteneur flex
        image_html += "</div>\n\n"  # Fin du div pour l'image

        # Gestion de l'ouverture et de la fermeture du fichier HTML
        if os.path.exists(chemin_fichier_html):  # Fichier existe déjà
            with open(chemin_fichier_html, 'r+') as f:  # Lecture et écriture
                content = f.read()
                match = re.search(r"</body>\s*", content, re.IGNORECASE)  # \s* pour gérer les espaces
                if match:
                    position_debut = match.start()
                    position_fin = match.end()
                    f.seek(position_debut)
                    f.truncate(position_debut)
                    f.write(image_html)  # Ajouter le contenu de l'image
                    f.write("</body>\n")  # Réécrire </body>
            print(f"Mise à jour du fichier rapport : {chemin_fichier_html}")
        else:  # Fichier n'existe pas
            with open(chemin_fichier_html, 'w') as f:
                f.write("<!DOCTYPE html>\n")
                f.write("<html>\n")
                f.write("<head>\n")
                f.write("<title>Récapitulatif des images</title>\n")
                f.write("<link rel='stylesheet' href='magnific-popup.css'>\n")
                f.write("<script src='jquery.min.js'></script>\n")
                f.write("<script src='jquery.magnific-popup.min.js'></script>\n")
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
                f.write("</style>\n")
                f.write("</head>\n")
                f.write("<body>\n")  # Début du body
                f.write(image_html)  # Ajouter le contenu de la première image
                f.write("</body>\n")  # Fermeture du body
            print(f"Création du fichier rapport : {chemin_fichier_html}")
    
    except Exception as e:
        print(f"Erreur lors de la génération : {e}")



