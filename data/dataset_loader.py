from datasets import load_dataset, get_dataset_split_names, Dataset

# TODO: Check where to specify batch size
def prepare_image_ocrvqa(image):
    # Check if the image is grayscale (1 channel)
    if image.mode != 'RGB':  # If it's not RGB, convert to RGB or process as grayscale
        if image.mode == 'L':  # Grayscale
            image = image.convert("RGB")  # Convert grayscale to RGB by repeating the channel
        else:
            print(f"Warning: Image mode is {image.mode}. Converting to RGB.")
            image = image.convert("RGB")
    return image

def load_ocr_vqa(split="train"):
    """
    All three splits are available on howard-hou/OCR-VQA: 'train', 'validation', and 'test'.
    """
    dataset_name = "howard-hou/OCR-VQA"
    check_split_availability(dataset_name, dataset_subname=None, split=split)
    dataset = load_dataset(dataset_name, split=split)
    def data_generator():
        """Yield processed examples one at a time instead of storing them all in memory."""
        for item in dataset:
            image = item["image"]
            question_id = item["image_id"]
            questions = item["questions"]
            answers = item["answers"]

            for idx, (q, a) in enumerate(zip(questions, answers)):
                image = prepare_image_ocrvqa(image)
                yield {
                    "image": image,
                    "question": q,
                    "answer": a,
                    "question_id": f"{question_id}_{idx}"
                }

    # Convert generator to Hugging Face Dataset (Lazy loading)
    return Dataset.from_generator(data_generator)

def load_text_vqa(split="train"):
    """
    All three splits are available on lmm-lab/textvqa: 'train', 'validation', and 'test'.
    """
    dataset_name = "lmms-lab/textvqa"
    check_split_availability(dataset_name, dataset_subname=None, split=split)
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def load_doc_vqa(split="validation"):
    """
    Only splits available on lmm-lab/DocVQA are 'test' and 'validation'.
    """
    dataset_name = "lmms-lab/DocVQA"
    dataset_subname = "DocVQA"
    check_split_availability(dataset_name, dataset_subname=dataset_subname, split=split)
    dataset = load_dataset(dataset_name, name=dataset_subname, split=split)
    return dataset

def load_info_vqa(split="validation"):
    """
    Only splits available on lmm-lab/DocVQA for InfoVQAare 'test' and 'validation'.
    """
    dataset_name = "lmms-lab/DocVQA"
    dataset_subname = "InfographicVQA"
    check_split_availability(dataset_name, dataset_subname=dataset_subname, split=split)
    dataset = load_dataset(dataset_name, name=dataset_subname, split=split)

    return dataset

def load_st_vqa(split="test"):
    """
    Only split available on lmm-lab/ST-VQA is 'test'.
    No answer column in the test split.
    """
    dataset_name = "lmms-lab/ST-VQA"
    check_split_availability(dataset_name, dataset_subname=None, split=split)
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def check_split_availability(dataset_name, split, dataset_subname=None):
    """Check if the requested split exists for the dataset"""
    available_splits = get_dataset_split_names(dataset_name, config_name=dataset_subname) if dataset_subname else get_dataset_split_names(dataset_name)
    if split not in available_splits:
        raise ValueError(f"Error: Split '{split}' is not available for dataset '{dataset_name}' ({dataset_subname if dataset_subname else 'default'}). "
                         f"Available splits: {available_splits}")



