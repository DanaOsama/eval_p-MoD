import argparse
from train import train
from evaluate import evaluate

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from data.dataset_loader import load_hf_dataset

# Dictionary mapping model names to constructors
def create_hf_paligemma(checkpoint):
    return PaliGemmaForConditionalGeneration.from_pretrained(checkpoint)

# Dictionary to map model names to their instances
MODEL_REGISTRY = {
    "pmod_paligemma": lambda checkpoint: CustomPaliGemma(),,
    "hf_paligemma": create_hf_paligemma  # Hugging Face PaliGemma with a checkpoint
}

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a model on a dataset.")

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (e.g., train, validation, test).")
    parser.add_argument("--model", type=str, default="paligemma_pmod",required=True, help="Name of the model to use.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training.")
    parser.add_argument("--task", type=str, choices=["train", "evaluate"], required=True, help="Task to perform: train or evaluate.")
    parser.add_argument("--metrics", type=str, nargs="+", default=["anls", "exact_match", "vqa"], help="Metrics to evaluate.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--ckpt", type=str, default="google/paligemma-3b-ft-ocrvqa-896", help="Path to model checkpoint for evaluation.")
    
    args = parser.parse_args()

    # Create the model instance based on model_name
    model = MODEL_REGISTRY[args.model_name](args.checkpoint)

    # Load dataset with the correct split
    dataset = load_hf_dataset(args.dataset, split=args.split)

    if args.task == "train":
        model, dataset_name, epochs=3, lr=5e-5
        train(model=model, dataset_name=args.dataset, epochs=args.epochs, lr=args.lr)
    elif args.task == "evaluate":
        evaluate(dataset_name=args.dataset, model_name=args.model, metrics=args.metrics)


if __name__ == "__main__":
    main()
