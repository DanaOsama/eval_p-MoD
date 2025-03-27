import numpy as np
from preprocess import preprocess_text_inflect, preprocess_text_word2number
from typing import List, Callable
import numpy as np

import re
import string



def levenshtein_distance(s1, s2):
    """
    Levenshtein distance is the minimum number of single-character edits 
    (insertions, deletions, or substitutions) required to change one string 
    into another.
    """

    # s2 is always the longer string
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def anls_single(
    references,
    predictions,
    thresh_hold=0.5,
    preprocess=False
):
    """
    Calculates anls for a single data point
    https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    # Unwrap predictions if it's a nested list
    pred = predictions[0] if isinstance(predictions[0], str) else predictions[0][0]

    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        if preprocess:
            gt_answer = preprocess_text_inflect(gt_answer)
            det_answer = preprocess_text_inflect(det_answer)

        dist = levenshtein_distance(gt_answer, det_answer)

        # Normalizes the distance by dividing by the maximum string length.
        # Ensures the score is between 0 (exact match) and 1 (completely different).
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    # 1 - min(values) converts distance into similarity
    # If the predicted answer is identical to a ground truth → ANLS = 1
    # If completely different → ANLS = 0
    # Partial match gives a score between 0 and 1
    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return {"anls": question_result}

def compute_anls_for_dataset(
    all_references: List[List[str]],
    all_predictions: List[List[str]],
    aggregation_method: Callable[[List[float]], float] = np.mean,
    threshold: float = 0.5
):
    """
    Computes ANLS for an entire dataset.
    
    :param all_references: List of lists of reference answers (each question has multiple references)
    :param all_predictions: List of lists of predicted answers (one prediction per question)
    :param aggregation_method: Function to aggregate ANLS scores (default: np.mean)
    :param threshold: Threshold for ANLS scoring (default 0.5)
    :return: Aggregated ANLS score over dataset
    """
    anls_scores = []

    for references, prediction in zip(all_references, all_predictions):
        score = anls_single(references, prediction, threshold)["anls"]  # Compute ANLS per question
        anls_scores.append(score)

    return {"anls": aggregation_method(anls_scores) if anls_scores else 0.0}

def mean_aggregation(values):
    """Returns the mean of the list."""
    return np.mean(values) if values else 0.0

def median_aggregation(values):
    """Returns the median of the list."""
    return np.median(values) if values else 0.0

def max_aggregation(values):
    """Returns the maximum value from the list."""
    return max(values) if values else 0.0

def min_aggregation(values):
    """Returns the minimum value from the list."""
    return min(values) if values else 0.0

def weighted_mean_aggregation(values, weights):
    """
    Returns the weighted mean of the list.

    :param values: List of ANLS scores.
    :param weights: List of corresponding weights (must be same length as values).
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0
    return np.average(values, weights=weights)

def percentile_aggregation(values, percentile=75):
    """
    Returns the specified percentile (e.g., 75th percentile) of the list.

    :param values: List of ANLS scores.
    :param percentile: The percentile to compute (default: 75).
    """
    return np.percentile(values, percentile) if values else 0.0


### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    """
    Computes the exact match (EM) score over an entire dataset.
    
    :param predictions: List of predicted answers (one per question)
    :param references: List of lists of reference answers (each question may have multiple correct answers)
    :param regexes_to_ignore: List of regex patterns to ignore in both predictions and references
    :param ignore_case: If True, converts text to lowercase before comparison
    :param ignore_punctuation: If True, removes punctuation before comparison
    :param ignore_numbers: If True, removes digits before comparison
    :return: Dictionary with "exact_match" score
    """
    
    def preprocess_text(text):
        """Preprocesses text based on given settings."""
        if regexes_to_ignore:
            for s in regexes_to_ignore:
                text = re.sub(s, "", text)
        
        if ignore_case:
            text = text.lower()
        
        if ignore_punctuation:
            repl_table = str.maketrans("", "", string.punctuation)
            text = text.translate(repl_table)

        if ignore_numbers:
            repl_table = str.maketrans("", "", string.digits)
            text = text.translate(repl_table)

        return text.strip()
    
    total_score = 0
    num_samples = len(predictions)

    for pred, ref_list in zip(predictions, references):
        # Preprocess prediction
        processed_pred = preprocess_text(pred)
        
        # Preprocess all reference answers
        processed_refs = [preprocess_text(ref) for ref in ref_list]

        # Check if the prediction matches any of the references
        match = int(processed_pred in processed_refs)
        total_score += match

    return {"exact_match": total_score / num_samples if num_samples > 0 else 0.0}


def vqa_accuracy(predictions, references):
    """
    Computes VQA accuracy as per the official VQA evaluation metric.
    
    Args:
        predictions (list): List of predicted answers (one per sample).
        references (list): List of lists, where each inner list contains human-annotated answers.
    
    Returns:
        float: VQA accuracy score.
    """
    if preprocess:
        predictions = [preprocess_text_word2number(pred) for pred in predictions]
        references = [[preprocess_text_word2number(ans) for ans in ans_list] for ans_list in references]
    scores = []
    for pred, ground_truths in zip(predictions, references):
        answer_counts = Counter(ground_truths)
        vqa_score = min(answer_counts[pred] / 3.0, 1.0) if pred in answer_counts else 0
        scores.append(vqa_score)
    
    return np.mean(scores)
