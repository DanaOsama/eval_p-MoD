import torch
from transformers import AdamW
from data.dataset_loader import load_hf_dataset
from models.paligemma_custom import CustomPaliGemma

epochs = 1
lr = 5e-5
dataset = "howard-hou/OCR-VQA"
model_name = "paligemma_pmod"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(model, dataset_name, epochs=3, lr=5e-5):
    print(f"Training {model.__class__.__name__} on {dataset_name} for {epochs} epochs...")

    dataset = load_hf_dataset(dataset)
    optimizer = AdamW(model.parameters(), lr=lr)

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
