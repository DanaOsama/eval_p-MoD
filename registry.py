from models.custom_paligemma import CustomPaliGemma
from transformers import PaliGemmaForConditionalGeneration
from data.dataset_loader import (
    load_ocr_vqa,
    load_text_vqa,
    load_doc_vqa,
    load_info_vqa,
    load_st_vqa
)

# Checkpoint-based HF model loader
def create_hf_paligemma(checkpoint):
    return PaliGemmaForConditionalGeneration.from_pretrained(checkpoint)

# Model registry
MODEL_REGISTRY = {
    "pmod_paligemma": lambda checkpoint=None: CustomPaliGemma(),
    "hf_paligemma": create_hf_paligemma,
}

# Dataset registry
DATASET_REGISTRY = {
    "ocr-vqa": load_ocr_vqa,
    "text-vqa": load_text_vqa,
    "doc-vqa": load_doc_vqa,
    "info-vqa": load_info_vqa,
    "st-vqa": load_st_vqa,
}

def load_hf_dataset(dataset_name, split="train"):
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name](split)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
