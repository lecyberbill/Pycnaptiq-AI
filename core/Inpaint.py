import gradio as gr
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import os, sys
from Utils.utils import txt_color, translate


def apply_mask_effects(image_and_mask, translations, opacity=1.0, blur_radius=0):
    """
    Extrait le masque, applique l'opacité et le flou,
    et retourne le masque traité comme une image PIL (convertie en RGB pour l'affichage).
    """
    if image_and_mask is None:
        print(txt_color("[AVERTISSEMENT]", "erreur"), translate("erreur_image_mask_none", translations))
        return None

    # Initialiser opacity et blur_radius si ils sont None
    opacity = 1.0 if opacity is None else opacity
    blur_radius = 0 if blur_radius is None else blur_radius

    try:
        # Extraire les données du masque
        mask_data = None
        if isinstance(image_and_mask, dict):
            layers = image_and_mask.get("layers")
            if layers and len(layers) > 0:
                mask_data = layers[0] # Peut être ndarray ou PIL Image
            else:
                print(txt_color("[ERREUR]", "erreur"), translate("erreur_mask_manquant", translations))
                gr.Warning(translate("erreur_mask_manquant", translations), 4.0)
                return None
        elif isinstance(image_and_mask, Image.Image):
             # Si l'entrée est déjà une image PIL (peut arriver lors des updates)
             mask_data = image_and_mask
        else:
             print(txt_color("[ERREUR]", "erreur"), translate("erreur_image_mask_invalide", translations))
             gr.Warning(translate("erreur_image_mask_invalide", translations), 4.0)
             return None

        # Convertir en tableau NumPy si ce n'est pas déjà le cas
        if isinstance(mask_data, Image.Image):
            if mask_data.mode == 'RGBA':
                mask_array = np.array(mask_data)
            else:
                mask_array = np.array(mask_data.convert('L')) # Assurer 2D si pas RGBA
        elif isinstance(mask_data, np.ndarray):
            mask_array = mask_data
        else:
            print(txt_color("[ERREUR]", "erreur"), translate("erreur_type_mask_inattendu", translations).format(type(mask_data)))
            gr.Warning(translate("erreur_type_mask_inattendu", translations).format(type(mask_data)), 4.0)
            return None

        # --- CORRECTION : Logique pour extraire le masque (alpha ou niveaux de gris) ---
        alpha_channel = None
        # Vérifier le nombre de dimensions AVANT d'accéder à shape[2]
        if mask_array.ndim == 3 and mask_array.shape[2] == 4:
            # C'est RGBA, prendre le canal alpha
            alpha_channel = mask_array[:, :, 3].astype(np.uint8)
        elif mask_array.ndim == 2:
            # C'est déjà en niveaux de gris (ou a été converti)
            alpha_channel = mask_array.astype(np.uint8)
        elif mask_array.ndim == 3 and mask_array.shape[2] == 3:
             # C'est RGB, convertir en niveaux de gris
             temp_pil_img = Image.fromarray(mask_array)
             alpha_channel = np.array(temp_pil_img.convert('L')).astype(np.uint8)
        else:
            # Format inattendu
            print(txt_color("[ERREUR]", "erreur"), translate("erreur_format_mask_inattendu", translations).format(mask_array.shape))
            raise gr.Error(translate("erreur_format_mask_inattendu", translations).format(mask_array.shape), 4.0)
            # return None
        # --- FIN CORRECTION ---

        # --- Application des effets ---
        if alpha_channel is None:
             print(txt_color("[ERREUR]", "erreur"), "alpha_channel est None après extraction.")
             return None # Ou gérer l'erreur autrement

        # Appliquer l'opacité
        if opacity < 1.0:
            alpha_channel = (alpha_channel * opacity).astype(np.uint8)

        # Reconvertir en image PIL (mode 'L') pour le flou
        mask_image = Image.fromarray(alpha_channel, mode='L')

        # Appliquer le flou
        if blur_radius > 0:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Retourner l'image PIL convertie en RGB pour l'affichage Gradio
        return mask_image.convert('RGB')

    except Exception as e:
        print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_mask', translations)}: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{translate('erreur_mask', translations)}: {e}", 4.0)
        # return None

def extract_image(image_and_mask, translations):
    """Extrait l'image de l'image avec le masque."""
    if image_and_mask is None:
        return None

    try:
        if "background" in image_and_mask:
            image = image_and_mask["background"]
            image = Image.fromarray(image).convert("RGB")
            return image
        else:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_image_manquant", translations))
            gr.Warning(translate("erreur_image_manquant", translations), 4.0)
            return None
    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_extract_image', translations)}: {e}")
        raise gr.Error(translate("erreur_extract_image", translations), f": {e}", 4.0)
        return None