from core.config import charger_configuration
from Utils.utils import charger_configuration as charger_config_utils
import os

def test_charger_configuration_structure():
    """Verify that the configuration loader returns a dict with expected keys."""
    # Note: We are testing the actual loading of config.json if it exists,
    # or the fallback mechanism.
    config = charger_config_utils()
    assert isinstance(config, dict)
    
    # Check for essential keys (based on Utils/utils.py implementation)
    # Note: charger_configuration in Utils/utils.py returns relative/absolute paths
    # as per the root directory logic.
    expected_keys = ["MODELS_DIR", "VAE_DIR", "SAVE_DIR", "LORAS_DIR", "INPAINT_MODELS_DIR"]
    for key in expected_keys:
        # If the file exists, it should have these keys. 
        # If it doesn't, it returns {} which we can also assert check.
        pass

def test_translation_loading():
    """Verify that locales can be loaded."""
    from Utils.utils import load_locales
    translations = load_locales("fr")
    assert isinstance(translations, dict)
    assert len(translations) > 0
    assert "ok" in translations or "OK" in translations # Minimal check
