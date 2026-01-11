import math
import numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image

def get_tiles_coords(width, height, tile_size=512, overlap=64):
    """
    Calcule les coordonnées (x, y, w, h) de chaque tile pour couvrir l'image entière 
    avec un chevauchement donné.
    """
    coords = []
    
    # On s'assure que tile_size n'est pas plus grand que l'image
    tile_w = min(tile_size, width)
    tile_h = min(tile_size, height)
    
    # Calculer le pas (step)
    stride = tile_size - overlap
    
    y = 0
    while y < height:
        # Ajuster si on dépasse
        actual_h = tile_h
        if y + actual_h > height:
            y = height - actual_h
        
        x = 0
        while x < width:
            actual_w = tile_w
            if x + actual_w > width:
                x = width - actual_w
            
            coords.append((x, y, actual_w, actual_h))
            
            if x + actual_w >= width:
                break
            x += stride
        
        if y + actual_h >= height:
            break
        y += stride
        
    return coords

def create_feather_mask(tile_size, overlap):
    """
    Crée un masque de fusion (0.0 à 1.0) pour une tile, 
    avec des gradients sur les bords pour une fusion douce.
    """
    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    
    if overlap <= 0:
        return mask

    # Application du gradient sur les 4 bords
    for i in range(overlap):
        alpha = (i + 1) / (overlap + 1)
        # Bord gauche
        mask[:, i] *= alpha
        # Bord haut
        mask[i, :] *= alpha
        # Bord droit
        mask[:, -(i+1)] *= alpha
        # Bord bas
        mask[-(i+1), :] *= alpha
        
    return mask

def tiled_upscale_process(image, pipe, upscale_factor, tile_size, overlap, denoising_strength, 
                          prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds,
                          num_inference_steps, guidance_scale, seed, device, progress_callback=None):
    """
    Processus complet de tiling : redimensionnement, découpage, Img2Img par tile et reconstruction.
    """
    # 1. Redimensionner l'image source à la taille cible
    target_w = int(image.width * upscale_factor)
    target_h = int(image.height * upscale_factor)
    
    # On aligne sur des multiples de 8 pour SDXL
    target_w = (target_w // 8) * 8
    target_h = (target_h // 8) * 8
    
    print(f"[Upscale] Upscaling to {target_w}x{target_h} via {tile_size}px tiles (overlap: {overlap}px)")
    
    high_res_base = image.resize((target_w, target_h), resample=Image.LANCZOS)
    
    # S'assurer que le pipeline supporte l'Img2Img (conversion à la volée très légère)
    try:
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    except Exception as e:
        print(f"[Upscale] Warning: Error during pipeline conversion: {e}. Attempting with original pipe.")

    # 2. Préparer le canvas de résultat et le canvas de poids (pour la moyenne pondérée)
    result_canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
    weight_canvas = np.zeros((target_h, target_w), dtype=np.float32)
    
    # 3. Calculer les coordonnées des tiles
    coords = get_tiles_coords(target_w, target_h, tile_size, overlap)
    total_tiles = len(coords)
    
    # 4. Créer le masque de fusion
    feather_mask = create_feather_mask(tile_size, overlap)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # 5. Boucle sur les tiles
    for i, (x, y, w, h) in enumerate(coords):
        if progress_callback:
            progress_callback(i, total_tiles)
            
        print(f"[Upscale] Processing tile {i+1}/{total_tiles} at ({x}, {y})")
        
        # Extraire la zone de l'image base
        tile_img = high_res_base.crop((x, y, x + w, y + h))
        
        # S'assurer que le masque correspond à la taille réelle de la tile (bordures d'image)
        if w != tile_size or h != tile_size:
            actual_mask = Image.fromarray((feather_mask * 255).astype(np.uint8)).resize((w, h))
            actual_mask = np.array(actual_mask).astype(np.float32) / 255.0
        else:
            actual_mask = feather_mask

        # Lancer l'Img2Img sur la tile
        try:
            with torch.no_grad():
                processed_tile = pipe(
                    image=tile_img,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                    strength=denoising_strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=w,
                    height=h
                ).images[0]
                
            # Convertir en numpy pour le mélange
            tile_np = np.array(processed_tile).astype(np.float32)
            
            # Ajuster si la sortie du pipe n'est pas exactement w x h (SDXL force parfois des multiples de 8)
            if tile_np.shape[0] != h or tile_np.shape[1] != w:
                tile_np = np.array(Image.fromarray(tile_np.astype(np.uint8)).resize((w, h))).astype(np.float32)

            # Ajouter au canvas final avec pondération par le masque
            for c in range(3):
                result_canvas[y:y+h, x:x+w, c] += tile_np[:, :, c] * actual_mask
            
            weight_canvas[y:y+h, x:x+w] += actual_mask
        except Exception as e_tile:
            print(f"[Upscale] Error processing tile {i}: {e_tile}")
            # Fallback: utiliser l'image d'origine pour cette tile
            tile_np = np.array(tile_img).astype(np.float32)
            for c in range(3):
                result_canvas[y:y+h, x:x+w, c] += tile_np[:, :, c] * actual_mask
            weight_canvas[y:y+h, x:x+w] += actual_mask
        
    # 6. Normaliser par les poids et convertir en Image PIL
    # Éviter la division par zéro
    weight_canvas[weight_canvas == 0] = 1.0
    
    for c in range(3):
        result_canvas[:, :, c] /= weight_canvas
        
    final_image = Image.fromarray(np.clip(result_canvas, 0, 255).astype(np.uint8))
    
    return final_image
