from PIL import Image
import gradio as gr
from Utils.utils import txt_color, translate, ImageSDXLchecker

def handle_image_mask_interaction(image_mask_value_dict, current_original_bg_props, translations):
    """
    Gère les interactions avec ImageMask (chargement d'image, dessin).
    Met à jour validated_image_state si l'image de fond change.
    Met à jour original_editor_background_props_state avec les props de l'image de fond de l'éditeur.
    """
    if image_mask_value_dict is None:
        return None, None 

    current_background_pil = image_mask_value_dict.get("background") 
    if current_background_pil is None:
        if current_original_bg_props is not None: 
            return None, None 
        else: 
            return gr.update(), gr.update() 

    if current_background_pil is not None and not isinstance(current_background_pil, Image.Image):
        print(txt_color("[AVERTISSEMENT]", "warning"), f"handle_image_mask_interaction: Le fond de ImageMask n'est pas une PIL.Image. Type: {type(current_background_pil)}. Retour sans mise à jour.")
        return gr.update(), gr.update()  

    if current_background_pil is None:
        return None, None

    new_bg_props = (current_background_pil.width, current_background_pil.height, current_background_pil.mode)

    if current_original_bg_props is None or new_bg_props != current_original_bg_props:
        print(txt_color("[INFO]", "info"), "Nouvelle image de fond détectée dans ImageMask, validation en cours.")
        image_checker = ImageSDXLchecker(current_background_pil, translations)
        processed_background_for_pipeline = image_checker.redimensionner_image()

        if isinstance(processed_background_for_pipeline, Image.Image):
            return processed_background_for_pipeline, new_bg_props
        else:
            print(txt_color("[AVERTISSEMENT]", "warning"), "L'image de fond de ImageMask n'est pas valide après traitement par ImageSDXLchecker.")
            return None, None
    else:
        return gr.update(), gr.update()

def create_opaque_mask_from_editor(editor_dict, target_size, translations):
    """
    Crée un masque binaire opaque (PIL, mode 'L') à partir des layers dessinés
    dans ImageEditor. Les zones dessinées (non transparentes dans les layers)
    deviennent blanches (255), le reste noir (0).
    """
    if editor_dict is None or not isinstance(editor_dict, dict):
        print(txt_color("[AVERTISSEMENT]", "warning"), translate("erreur_donnees_editeur_mask_invalides", translations))
        return Image.new('L', target_size, 0) 

    layers_pil_from_editor = editor_dict.get("layers", []) 
    
    if not layers_pil_from_editor: 
        return Image.new('L', target_size, 0) 

    composite_rgba_mask = Image.new('RGBA', target_size, (0, 0, 0, 0))

    for layer_pil in layers_pil_from_editor:
        if layer_pil is None:
            continue
        try:
            if not isinstance(layer_pil, Image.Image):
                print(txt_color("[AVERTISSEMENT]", "warning"), f"Un élément dans 'layers' n'est pas une image PIL: {type(layer_pil)}")
                continue
            layer_to_composite = layer_pil.convert('RGBA')
            if layer_to_composite.size != target_size:
                print(txt_color("[INFO]", "info"), f"Redimensionnement du layer de masque de {layer_to_composite.size} à {target_size}")
                layer_to_composite = layer_to_composite.resize(target_size, Image.Resampling.NEAREST)

            composite_rgba_mask.alpha_composite(layer_to_composite)
        except Exception as e:
            print(txt_color("[ERREUR]", "erreur"), f"{translate('erreur_traitement_layer_mask', translations)}: {e}")
            continue

    alpha_channel = composite_rgba_mask.split()[-1] 
    binary_mask_pil = alpha_channel.point(lambda p: 255 if p > 0 else 0, mode='L')
    
    return binary_mask_pil
