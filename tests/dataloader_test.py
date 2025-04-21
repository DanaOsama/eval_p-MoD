import sys
import os
import pytest
from datasets import DatasetDict, Dataset

# Ensure the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from registry import load_hf_dataset

@pytest.mark.parametrize("dataset_name", ["ocr-vqa", "text-vqa", "doc-vqa", "info-vqa", "st-vqa"])
def test_load_hf_dataset(dataset_name):
    dataset = load_hf_dataset(dataset_name, split="validation" if dataset_name != "st-vqa" else "test")
    
    assert dataset is not None, f"Failed to load dataset: {dataset_name}"
    print(f"Successfully loaded {dataset_name}")

if __name__ == "__main__":
    pytest.main([__file__])
