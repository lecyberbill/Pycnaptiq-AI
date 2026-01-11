import pytest
from presets.presets_Manager import PresetManager
from unittest.mock import MagicMock, patch

def test_preset_manager_init(mock_translations):
    """Test that PresetManager initializes without error."""
    with patch('presets.presets_Manager.sqlite3.connect'):
        pm = PresetManager(mock_translations)
        assert pm.translations == mock_translations

def test_preset_manager_module_state(mock_translations):
    """Test get/save module state."""
    with patch('presets.presets_Manager.sqlite3.connect'):
        pm = PresetManager(mock_translations)
        pm.loaded_module_states = {"test_mod": True}
        assert pm.get_module_state("test_mod") is True
        assert pm.get_module_state("other_mod", default_active=False) is False
