import argparse
from train import train
from eval import evaluate_model

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from registry import MODEL_REGISTRY, load_hf_dataset

import torch

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a model on a dataset.")

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (e.g., train, validation, test).")
    parser.add_argument("--model", type=str, default="hf_paligemma",required=True, help="Name of the model to use.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training.")
    parser.add_argument("--task", type=str, choices=["train", "eval"], required=True, help="Task to perform: train or evaluate.")
    # parser.add_argument("--metrics", type=str, nargs="+", default=["anls", "exact_match", "vqa"], help="Metrics to evaluate.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--ckpt", type=str, default="google/paligemma-3b-ft-ocrvqa-896", help="Path to model checkpoint for evaluation.")
    parser.add_argument("--save_json", action="store_true", help="Save predictions to JSON file.")
    args = parser.parse_args()

    # Create the model instance based on model_name
    model_fn = MODEL_REGISTRY[args.model]
    model = model_fn(args.ckpt) if args.model == "hf_paligemma" else model_fn(None)

    # Load dataset with the correct split
    dataset = load_hf_dataset(args.dataset, split=args.split)

    # HARDCODED METRICS FOR EACH DATASET - From Paligemma paper's appendix
    metrics = None
    if args.dataset == "doc-vqa":
        metrics = ["anls"]
    elif args.dataset == "text-vqa":
        metrics = ["vqa"]
    elif args.dataset == "st-vqa":
        metrics = ["anls"]
    elif args.dataset == "info-vqa":
        metrics = ["anls"]
    elif args.dataset == "ocr-vqa":
        metrics = ["exact_match"]
    else:
        # Default metrics for other datasets
        metrics = ["anls", "exact_match", "vqa"]


    if args.task == "train":
        train(model=model, dataset_name=args.dataset, epochs=args.epochs, lr=args.lr)
    
    elif args.task == "eval":
        # For evaluation, we must make sure that the dataset, split, model, and checkpoint are all specifiedm - at least!
        if args.dataset is None or args.split is None or args.model is None or args.ckpt is None:
            raise ValueError("Dataset, split, and model must be specified in the command line for evaluation.")

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Call the evaluation function 
        evaluate_model(model=model, dataset_name=args.dataset, split=args.split, checkpoint=args.ckpt, metrics=metrics, device=device, save_json=args.save_json)

if __name__ == "__main__":
    main()
