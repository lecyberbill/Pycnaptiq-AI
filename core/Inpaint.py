import gradio as gr
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import os, sys
from Utils.utils import txt_color, translate


def apply_mask_effects(image_and_mask, translations, opacity, blur_radius):
    """
    Extracts the mask from the gr.ImageMask component, 
    ensures the background is black and the drawn area is white, 
    and returns it as a PIL Image.
    """
    if image_and_mask is None:
        return None

    # Initialiser opacity et blur_radius si ils sont None
    if opacity is None:
        opacity = 1.0
    if blur_radius is None:
        blur_radius = 0

    try:
        if isinstance(image_and_mask, dict):
            layers = image_and_mask.get("layers")
            if layers and len(layers) > 0:
                mask_array = layers[0]
            else:
                print(txt_color("[ERREUR] ", "erreur"), translate("erreur_mask_manquant", translations))
                gr.Warning(translate("erreur_mask_manquant", translations), 4.0)
                return None
        else:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_image_mask_invalide", translations))
            gr.Warning(translate("erreur_image_mask_invalide", translations), 4.0)
            return None

        # Ensure mask_array is a NumPy array
        mask_array = np.array(mask_array)

        # Check if mask_array has an alpha channel
        if mask_array.shape[2] < 4:
            print(txt_color("[ERREUR] ", "erreur"), translate("erreur_mask_pas_rgba", translations))
            gr.Warning(translate("erreur_mask_pas_rgba", translations), 4.0)
            return None

        # Extract the alpha channel (mask)
        alpha_channel = mask_array[:, :, 3].astype(np.uint8)

        # Create a PIL Image from the alpha channel
        mask_image = Image.fromarray(alpha_channel).convert("L")

        # Invert the mask (black background, white drawing)
        mask_image = ImageOps.invert(mask_image)

        # Appliquer l'opacité
        if opacity < 1.0:
            mask_array = np.array(mask_image)
            mask_array = (mask_array * opacity).astype(np.uint8)
            mask_image = Image.fromarray(mask_array)

        # Appliquer le flou
        if blur_radius > 0:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))


        return mask_image

    except Exception as e:
        print(txt_color("[ERREUR] ", "erreur"), f"{translate('erreur_mask', translations)}: {e}")
        raise gr.Error(translate("erreur_mask", translations) + f": {e}", 4.0)
        return None

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
        raise gr.Error(translate("erreur_extract_image", translations) + f": {e}", 4.0)
        return None