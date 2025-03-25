from datasets import load_dataset

def load_hf_dataset(dataset_name, split="train", **kwargs):
    """
    Loads a dataset from Hugging Face.
    """
    return load_dataset(dataset_name, split=split, **kwargs)
