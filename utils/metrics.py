import numpy as np

def anls_score(predictions, ground_truths, threshold=0.5):
    """
    Computes the Average Normalized Levenshtein Similarity (ANLS) score.
    """
    def levenshtein(a, b):
        dp = np.zeros((len(a)+1, len(b)+1))
        for i in range(len(a)+1):
            for j in range(len(b)+1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[len(a)][len(b)]

    scores = []
    for pred, gt in zip(predictions, ground_truths):
        distance = levenshtein(pred, gt)
        normalized_score = 1 - (distance / max(len(pred), len(gt)))
        scores.append(normalized_score if normalized_score >= threshold else 0)
    return np.mean(scores)

def exact_match(predictions, ground_truths):
    """
    Computes Exact Match (EM) accuracy.
    """
    return np.mean([pred == gt for pred, gt in zip(predictions, ground_truths)])


def vqa_accuracy(predictions, ground_truths_list):
    """
    Computes VQA accuracy as per the official VQA evaluation metric.
    
    Args:
        predictions (list): List of predicted answers (one per sample).
        ground_truths_list (list): List of lists, where each inner list contains human-annotated answers.
    
    Returns:
        float: VQA accuracy score.
    """
    scores = []
    for pred, ground_truths in zip(predictions, ground_truths_list):
        answer_counts = Counter(ground_truths)
        vqa_score = min(answer_counts[pred] / 3.0, 1.0) if pred in answer_counts else 0
        scores.append(vqa_score)
    
    return np.mean(scores)
