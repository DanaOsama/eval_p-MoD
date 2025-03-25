import torch
from transformers import AdamW
from data.dataset_loader import load_hf_dataset
from models.paligemma_custom import CustomPaliGemma

def train():
    model = CustomPaliGemma()
    dataset = load_hf_dataset("your-dataset-name")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
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
