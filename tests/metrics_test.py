import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import exact_match, anls, vqa_accuracy

test_cases = [
    # Case 1: Single reference, exact match
    {
        "references": [["The Eiffel Tower"]],
        "prediction": "The Eiffel Tower",  # Should get high ANLS & exact match
    },
    
    # Case 2: Single reference, minor difference
    {
        "references": [["Mount Everest"]],
        "prediction": "Mt. Everest",  # Should get high ANLS, but no exact match
    },

    # Case 3: Multiple references, one exact match
    {
        "references": [["dog"], ["canine"], ["puppy"]],
        "prediction": "dog",  # Should get exact match and high ANLS
    },

    # Case 4: Multiple references, close but not exact
    {
        "references": [["a red apple"], ["an apple"], ["a juicy apple"]],
        "prediction": "red apple",  # Should get decent ANLS but no exact match
    },

    # Case 5: Multiple references, all different from prediction
    {
        "references": [["New York"], ["NYC"], ["The Big Apple"]],
        "prediction": "Los Angeles",  # Should get very low ANLS & no exact match
    },

    # Case 6: prediction matches 2 references 
    # still outputted 0.333 for vqa_accuracy.
    # TODO: Double check why this is the case
    {
        "references": [["New York"], ["New York"], ["The Big Apple"]],
        "prediction": "New York",  # Should get very low ANLS & no exact match
    },

    # Case 7: prediction contains 2 references
    {
        "references": [["New York"], ["NYC"], ["The Big Apple"]],
        "prediction": "New York NYC",  # Should get very low ANLS & no exact match
    }
]

for i, test_case in enumerate(test_cases):
    references = test_case["references"]
    prediction = test_case["prediction"]
    print(f"Running test case {i + 1}...")
    print(f"References: {references}")
    print(f"Prediction: {prediction}")
    print(f"Exact match: {exact_match(prediction, references)}")
    anls_score = anls(prediction, references, aggregation_method="mean")
    print(f"ANLS: {anls_score}")
    print(f"VQA accuracy: {vqa_accuracy(references, prediction, True)}")
    print()