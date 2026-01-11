import pytest
from unittest.mock import MagicMock, patch
from core.sdxl_logic import generate_image
import PIL.Image

@pytest.fixture
def mock_sdxl_deps(mock_translations, mock_model_manager):
    """Mock dependencies for SDXL generation tests."""
    with patch('core.sdxl_logic.translate', side_effect=lambda k, t: mock_translations.get(k, k)), \
         patch('core.sdxl_logic.gr'), \
         patch('core.sdxl_logic.os.makedirs'), \
         patch('core.sdxl_logic.enregistrer_image'), \
         patch('core.sdxl_logic.preparer_metadonnees_image', return_value=(None, "Success")), \
         patch('core.sdxl_logic.empty_working_set'):
        yield

def test_generate_image_basic_yields(mock_sdxl_deps, mock_translations, mock_model_manager, mock_config):
    """Test that generate_image yields expected UI updates."""
    # Setup mock pipeline
    mock_pipe = MagicMock()
    # Mock __call__ to return a mock object with 'images' attribute
    mock_pipe.return_value = MagicMock(images=[PIL.Image.new('RGB', (64, 64))])
    mock_model_manager.get_current_pipe.return_value = mock_pipe
    
    # Run generation with all required arguments (28+)
    gen = generate_image(
        "A test prompt", # text
        "None",          # style_selection
        7.5,             # guidance_scale
        11,              # num_steps (wait, I used 1 in previous, let's keep 1)
        1,               # num_steps
        "1024*1024",     # selected_format
        1234,            # seed_input
        1,               # num_images
        False,           # pag_enabled
        3.0,             # pag_scale
        "",              # pag_applied_layers_str
        "",              # original_user_prompt_for_cycle
        False,           # prompt_is_currently_enhanced
        False,           # enhancement_cycle_is_active
        mock_model_manager, # model_manager
        mock_translations,  # translations
        mock_config,        # config
        "cpu",              # device
        MagicMock(),        # stop_event
        MagicMock(),        # stop_gen
        "outputs",          # SAVE_DIR
        "TestUser",         # AUTHOR
        "JPEG",             # IMAGE_FORMAT
        MagicMock(),        # image_executor
        MagicMock(),        # html_executor
        [],                 # STYLES
        "neg",              # NEGATIVE_PROMPT
        [],                 # PREVIEW_QUEUE
        # *lora_inputs (at least 12 expected based on code)
        False, False, False, False, # lora_checks
        "None", "None", "None", "None", # lora_dropdowns
        1.0, 1.0, 1.0, 1.0 # lora_scales
    )
    
    # Collect yields
    results = list(gen)
    
    # Assertions
    assert len(results) > 0
    # The last result should be a tuple of Gradio updates
    assert isinstance(results[-1], tuple) or isinstance(results[-1], list)
    assert len(results[-1]) >= 11
