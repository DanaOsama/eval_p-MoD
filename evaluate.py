import torch
from tabulate import tabulate
from data.dataset_loader import load_hf_dataset
from models.paligemma_custom import CustomPaliGemma
from utils.metrics import anls_score, exact_match, vqa_accuracy

# Dictionary mapping metric names to functions
METRICS = {
    "anls": anls_score,
    "exact_match": exact_match,
    "vqa": vqa_accuracy
}

def evaluate(metrics=["anls", "exact_match", "vqa"]):
    model = CustomPaliGemma()
    dataset = load_hf_dataset("your-dataset-name", split="test")

    predictions, ground_truths, ground_truth_lists = [], [], []
    for batch in dataset:
        inputs = batch["input_ids"]
        outputs = model(inputs, attention_mask=batch["attention_mask"])
        preds = torch.argmax(outputs.logits, dim=-1).tolist()
        
        predictions.extend(preds)
        ground_truths.extend(batch["labels"])
        ground_truth_lists.extend(batch["all_labels"])  # For VQA metric

    # Compute selected metrics
    results = {}
    for metric in metrics:
        if metric == "vqa":
            results["VQA Accuracy"] = METRICS[metric](predictions, ground_truth_lists)
        else:
            results[metric.replace("_", " ").title()] = METRICS[metric](predictions, ground_truths)

    # Print summary table
    table = [[metric, f"{value:.4f}"] for metric, value in results.items()]
    print(tabulate(table, headers=["Metric", "Score"], tablefmt="grid"))

if __name__ == "__main__":
    evaluate(metrics=["anls", "exact_match", "vqa"])
