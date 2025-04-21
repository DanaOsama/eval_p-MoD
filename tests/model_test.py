import sys
import os
import pytest
from transformers import PaliGemmaForConditionalGeneration

# Add root of project to sys.path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom constructor
from registry import MODEL_REGISTRY

# Example checkpoint from Hugging Face
CHECKPOINT = "google/paligemma-3b-mix-448"  # Replace with the one you're using

@pytest.mark.parametrize("model_name", ["hf_paligemma"])  # add "pmod_paligemma" when it's ready
def test_model_loading(model_name):
    try:
        model_fn = MODEL_REGISTRY[model_name]
        model = model_fn(CHECKPOINT) if model_name == "hf_paligemma" else model_fn(None)

        assert model is not None, f"{model_name} returned None"
        
        if model_name == "hf_paligemma":
            assert isinstance(model, PaliGemmaForConditionalGeneration), \
                f"{model_name} is not an instance of PaliGemmaForConditionalGeneration"
        
        elif model_name == "pmod_paligemma":
            assert isinstance(model, CustomPaliGemma), \
                f"{model_name} is not an instance of CustomPaliGemma"

        print(f"{model_name} loaded successfully")

    except Exception as e:
        pytest.fail(f"{model_name} failed to load: {e}")

if __name__ == "__main__":
    pytest.main([__file__])