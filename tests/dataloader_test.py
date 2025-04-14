# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from data import load_hf_dataset

# DATASET_REGISTRY = {
#     "ocr-vqa": load_ocr_vqa,
#     "text-vqa": load_text_vqa,
#     "doc-vqa": load_doc_vqa,
#     "info-vqa": load_info_vqa,
#     "st-vqa": load_st_vqa
# }

# def test_datasets():
#     for dataset_name in DATASET_REGISTRY:
#         try:
#             dataset = load_hf_dataset(dataset_name, split="train")
#             assert dataset is not None, f"Failed to load dataset: {dataset_name}"
#             print(f"Successfully loaded {dataset_name}")
#         except Exception as e:
#             print(f"Error loading {dataset_name}: {e}")

# # Run the test
# test_datasets()

import sys
import os
import pytest
from datasets import DatasetDict, Dataset

# Ensure the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import load_hf_dataset

@pytest.mark.parametrize("dataset_name", ["ocr-vqa", "text-vqa", "doc-vqa", "info-vqa", "st-vqa"])
def test_load_hf_dataset(dataset_name):
    dataset = load_hf_dataset(dataset_name, split="validation" if dataset_name != "st-vqa" else "test")
    
    assert dataset is not None, f"Failed to load dataset: {dataset_name}"
    print(f"Successfully loaded {dataset_name}")

if __name__ == "__main__":
    pytest.main([__file__])
