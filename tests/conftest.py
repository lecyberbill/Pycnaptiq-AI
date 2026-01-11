import pytest
import os
import json
from unittest.mock import MagicMock

@pytest.fixture
def mock_translations():
    """Fixture to provide a minimal set of translations for testing."""
    return {
        "attention": "Attention",
        "erreur": "Erreur",
        "chargement": "Chargement...",
        "ok": "OK",
        "generer": "Générer",
        "preparation": "Préparation...",
        "prompt_traduit_pour_generation": "Prompt traduit",
        "image_sauvegarder": "Image sauvegardée",
        "temps_total_generation": "Temps total",
        "generation_image": "Génération image",
        "seed_utilise": "Seed utilisée",
        "generation_en_cours": "Génération en cours",
    }

@pytest.fixture
def mock_config():
    """Fixture to provide a mock configuration dictionary."""
    return {
        "MODELS_DIR": "models/checkpoints",
        "VAE_DIR": "models/vae",
        "SAVE_DIR": "outputs",
        "LORAS_DIR": "models/loras",
        "INPAINT_MODELS_DIR": "models/inpainting",
        "DEFAULT_LANGUAGE": "fr",
        "AUTHOR": "TestUser",
        "IMAGE_FORMAT": "JPEG"
    }

@pytest.fixture
def mock_model_manager():
    """Fixture to provide a mock ModelManager."""
    manager = MagicMock()
    manager.current_model_name = "test_model.safetensors"
    manager.current_vae_name = "Auto"
    manager.current_model_type = "standard"
    manager.loaded_loras = {}
    manager.get_current_pipe.return_value = MagicMock()
    manager.get_current_compel.return_value = MagicMock(return_value=(MagicMock(), MagicMock()))
    return manager
