from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import torch
from torchvision import transforms
import os

# TODO: Model specific prompt
# TODO: how to add batch size.

# prompts.py
def get_prompt_for_model(model_name, question, num_images=1, prefix="answer en"):
    if "paligemma" in model_name.lower():
        image_token = "<image>"
        prefix = prefix
        image_tokens = " ".join([image_token] * num_images)
        return f"{image_tokens} {prefix} {question}"

    elif "blip" in model_name.lower():
        return f"Question: {question} Answer:"

    elif "llava" in model_name.lower():
        image_tokens = "\n".join(["<image>"] * num_images)
        return f"###Human: {image_tokens}\n{question}\n###Assistant:"

    else:
        return question


# Initialize the tokenizer and processor
def initialize_processor_or_tokenizer(model_name: str):
    # Load the appropriate processor or tokenizer for the model
    try:
        # This handles models like CLIP, BLIP, PaliGemma, etc.
        processor = AutoProcessor.from_pretrained(model_name)
        return processor
    except Exception:
        print(f"Failed to load processor for {model_name}.")

def preprocess_text(text, tokenizer):
    """Preprocess text (question) for tokenization."""
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

def preprocess_image(image, processor=None):
    """Preprocess image data for the model."""
    if processor:
        # Use processor for models that accept image+text input (e.g., PaliGemma, BLIP)
        return processor(images=image, return_tensors="pt")["pixel_values"]
    else:
        # For models that only need image transformations (e.g., pure vision models)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension

# def prepare_inputs(batch, model_name, processor=None):
#     """Preprocess inputs for the model based on available data in the batch."""
    
#     inputs = {}

#     # Handle image data (assuming the image is in PIL format in the batch)
#     if "image" in batch:
#         image = batch["image"]
#         inputs["pixel_values"] = preprocess_image(image, processor)
    
#     # Handle text data (e.g., question in VQA)
#     if "question" in batch:
#         question = batch["question"]  # This is now a list if batched
#         tokenized = preprocess_text(question, processor.tokenizer if processor else None)
#         inputs["input_ids"] = tokenized["input_ids"]
#         inputs["attention_mask"] = tokenized["attention_mask"]

#     # If there are other fields, like document_id, handle them (if needed)
#     if "document_id" in batch:
#         inputs["document_id"] = batch["document_id"]

#     return inputs

def prepare_inputs(batch, model_name, device, processor):
    """Preprocess inputs for the model based on available data in the batch."""
    max_length = 512
    # Handle image data
    images = batch.get("image", None)
    questions = batch.get("question", None)
    if images is not None and questions is not None:
        if isinstance(images, list):
            num_images_list = [len(img) if isinstance(img, list) else 1 for img in images]
        else:
            # Same number of <image> tokens for all if it's a single image
            num_images_list = [1] * len(questions)

        # Ensure questions and num_images_list are lists of the same length
        if isinstance(questions, str):
            questions = [questions]

        if isinstance(num_images_list, int):
            num_images_list = [num_images_list] * len(questions)

        # Generate prompts with image tokens
        prompts = [get_prompt_for_model(model_name, q, n_img) for q, n_img in zip(questions, num_images_list)]
        images = [img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGB") for img in batch["image"]]
        
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        # inputs["pixel_values"] = preprocess_image(images, processor)

    # # Handle question text and inject appropriate number of <image> tokens
    # questions = batch.get("question", None)
    # if questions is not None:
    #     # Determine number of images per example (assuming batched input)
    #     if isinstance(images, list):
    #         num_images_list = [len(img) if isinstance(img, list) else 1 for img in images]
    #     else:
    #         # Same number of <image> tokens for all if it's a single image
    #         num_images_list = [1] * len(questions)

    #     # Ensure questions and num_images_list are lists of the same length
    #     if isinstance(questions, str):
    #         questions = [questions]

    #     if isinstance(num_images_list, int):
    #         num_images_list = [num_images_list] * len(questions)

    #     # Generate prompts with image tokens
    #     prompts = [get_prompt_for_model(model_name, q, n_img) for q, n_img in zip(questions, num_images_list)]

    #     tokenized = preprocess_text(prompts, processor.tokenizer if processor else None)
    #     inputs["input_ids"] = tokenized["input_ids"]
    #     inputs["attention_mask"] = tokenized["attention_mask"]

    # # Optionally include document ID
    # if "document_id" in batch:
    #     inputs["document_id"] = batch["document_id"]

    return inputs


def process_batch(batch, model_name, processor, device="cuda"):
    """Process a batch and prepare it for the model.""" 
    # Prepare the inputs
    inputs = prepare_inputs(batch, model_name, device, processor)
    
    return inputs

import json

def save_predictions_to_json(predictions, question_ids, model_name, dataset_name):
    """
    Save predictions in the format: [{"question_id": ..., "answer": ...}, ...]
    """
    breakpoint()
    results = [{"question_id": int(qid), "answer": ans} for qid, ans in zip(question_ids, predictions)]
    
    output_path = f"predictions/{model_name}_{dataset_name}_predictions.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved predictions to {output_path}")
