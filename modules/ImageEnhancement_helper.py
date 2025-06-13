import os
import torch
from modules.ImageEnhancement_definitions import Embedder, OneRestore
try:
    # This relative import might work if the main script runs from the parent directory
    from Utils.utils import txt_color
except ImportError:
    # Fallback or define a simple txt_color if needed for standalone testing
    def txt_color(text, _): return text
    print("[WARN] Could not import txt_color from Utils.utils in ImageEnhancement_helper.py")


def freeze(model):
    """Sets requires_grad to False for all model parameters."""
    if model is not None: # Check if model is not None before iterating
        for param in model.parameters():
            param.requires_grad = False

# Les fonctions load_word_embeddings et initialize_wordembedding_matrix ont été déplacées
# vers ImageEnhancement_models.py pour casser la dépendance circulaire.

def load_embedder_ckpt(device, freeze_model=True, ckpt_name=None,
                                  combine_type = ['clear', 'low', 'haze', 'rain', 'snow',
                                            'low_haze', 'low_rain', 'low_snow', 'haze_rain',
                                                    'haze_snow', 'low_haze_rain', 'low_haze_snow']):
    """Loads the Embedder checkpoint."""
    print(f'> Loading Embedder Checkpoint from {ckpt_name}')
    # Embedder est importé depuis ImageEnhancement_models
    model = Embedder(type_name=combine_type)

    if ckpt_name is not None and os.path.exists(ckpt_name):
        print(f'==> loading existing Embedder model weights from: {ckpt_name}')
        model_info = torch.load(ckpt_name, map_location=device)
        model.load_state_dict(model_info, strict=False)
    else:
        print('==> Checkpoint not found or ckpt_name is None. Initializing Embedder model with random weights.')

    if freeze_model:
        freeze(model)
    model.eval()
    return model.to(device)

def load_restore_ckpt(device, freeze_model=False, ckpt_name=None):
    """Loads the Restorer checkpoint."""
    print(f'> Loading Restorer Checkpoint from {ckpt_name}')
    # OneRestore est importé depuis ImageEnhancement_models
    model = OneRestore() # Uses default channel=32

    if ckpt_name is not None and os.path.exists(ckpt_name):
        print(f'==> loading existing OneRestore model weights from: {ckpt_name}')
        model_info = torch.load(ckpt_name, map_location=device)
        model.load_state_dict(model_info, strict=False)
    else:
         print('==> Checkpoint not found or ckpt_name is None. Initializing OneRestore model with random weights.')

    if freeze_model:
        freeze(model)
    model.eval()
    return model.to(device)
