import torch
from tabulate import tabulate
from registry import load_hf_dataset
from registry import MODEL_REGISTRY
from utils.metrics import anls, exact_match, vqa_accuracy
from utils.model_preprocess import process_batch, initialize_processor_or_tokenizer, save_predictions_to_json
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.cuda.empty_cache()

# Dictionary mapping metric names to functions
METRICS = {
    "anls": anls,
    "exact_match": exact_match,
    "vqa": vqa_accuracy
}

# ---------------------------------------------

import sys
import traceback
import pdb

def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    print("\n⚠️ Uncaught exception! Dropping into debugger...\n")
    pdb.post_mortem(tb)

sys.excepthook = info

# ---------------------------------------------

def collate_fn(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}

def evaluate_model(model, dataset_name, split, batch_size=2, checkpoint=None, metrics=["anls", "exact_match", "vqa"], device="cuda", save_json=False):
    print(f"Evaluating {model.__class__.__name__} on {dataset_name} with metrics: {metrics}")
    # Check that split is either "validation" or "test"
    if split not in ["validation", "test"]:
        raise ValueError("Split must be either 'validation' or 'test'.")
    else:
        print(f"Using split: {split}")
        dataset = load_hf_dataset(dataset_name, split=split)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    processor = initialize_processor_or_tokenizer(checkpoint)
    # breakpoint()

    model.to(device)
    # prediction_records saves the question_id or questionId
    predictions, ground_truths, prediction_records = [], [], []
    calculate_metrics = True
    question_id_not_found = False

    for batch in tqdm(data_loader, desc="Evaluating"):
        inputs = process_batch(batch, model.name_or_path, processor, device)
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).tolist()
        
        decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        predictions.extend(decoded_preds)

        # Check if key "question_id" or "questionId" and use the appropriate one
        if "question_id" in batch:
            prediction_records.extend(batch["question_id"])
        elif "questionId" in batch: 
            prediction_records.extend(batch["questionId"])
        else:
            question_id_not_found = True
            print("No question_id or questionId found in batch.")

        # Check if key "answer" or "answers" and use the appropriate one
        if "answer" in batch:
            ground_truths.extend(batch["answer"])
            # Check if the ground_truths contains null values
            if any(gt is None for gt in ground_truths):
                print("Ground truths contain null values. Setting calculate_metrics to False.")
                calculate_metrics = False
                save_json = True
        elif "answers" in batch:
            ground_truths.extend(batch["answers"])
        else:
            print("No answer or answers found in batch. Saving predictions only.")
            save_json = True
            calculate_metrics = False
            # raise ValueError("Batch must contain either 'answer' or 'answers' key.")
        # ground_truth_lists.append(batch.get("answers", []))  # Optional for VQA-style evaluation


    # Convert predictions to the same format as ground truths
    if isinstance(predictions[0], list):
        predictions = [pred[0] for pred in predictions]
    elif isinstance(predictions[0], str):
        predictions = [[pred] for pred in predictions]
    else:
        raise ValueError("Predictions must be a list of lists or a list of dictionaries.")

    # Compute selected metrics
    if calculate_metrics:
        results = {}
        for metric in metrics:
            if metric == "vqa":
                # results["VQA Accuracy"] = METRICS[metric](predictions, ground_truth_lists)
                results["VQA Accuracy"] = METRICS[metric](predictions, ground_truths)
            else:
                results[metric.replace("_", " ").title()] = METRICS[metric](predictions, ground_truths)

        table = [[metric, f"{value:.4f}"] for metric, value in results.items()]
        print(tabulate(table, headers=["Metric", "Score"], tablefmt="grid"))

    if save_json and not question_id_not_found:
        # Save predictions to JSON
        save_predictions_to_json(predictions, prediction_records, model.name_or_path, dataset_name)

if __name__ == "__main__":
    from registry import MODEL_REGISTRY
    model_name = "hf_paligemma"  # Change to "pmod_paligemma" for custom model
    # CHECKPOINT = "google/paligemma-3b-mix-448"
    # CHECKPOINT = "google/paligemma-3b-ft-ocrvqa-448"
    CHECKPOINT = "google/paligemma-3b-ft-docvqa-448"
    # CHECKPOINT = "google/paligemma-3b-ft-infovqa-448"
    model_fn = MODEL_REGISTRY[model_name]
    model = model_fn(CHECKPOINT) if model_name == "hf_paligemma" else model_fn(None)
    # dataset_name = "info-vqa" # tmux 4
    dataset_name = "doc-vqa" # tmux 3
    # dataset_name = "text-vqa" # tmux 2
    # dataset_name = "st-vqa" # tmux 1
    split = "validation"
    evaluate_model(model,dataset_name, split, checkpoint=CHECKPOINT, metrics=["anls", "exact_match", "vqa"], save_json=True)
    