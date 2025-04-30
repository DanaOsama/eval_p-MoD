import torch
# from transformers import AdamW
# from transformers.optimization import AdamW
from models.custom_paligemma import CustomPaliGemma
from tabulate import tabulate
from registry import load_hf_dataset
from registry import MODEL_REGISTRY
from utils.metrics import anls, exact_match, vqa_accuracy
from utils.model_preprocess import process_batch, initialize_processor_or_tokenizer, save_predictions_to_json
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, dataset_name, epochs=3, lr=5e-5):
    print(f"Training {model.__class__.__name__} on {dataset_name} for {epochs} epochs...")

    dataset = load_hf_dataset(dataset)
    # optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in dataset:
            inputs, labels = batch["input_ids"], batch["labels"]
            outputs = model(inputs, attention_mask=batch["attention_mask"])
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} completed.")

if __name__ == "__main__":
    train()
