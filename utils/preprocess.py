import re
import inflect
from word2number import w2n

from typing import List, Union

# Initialize inflect engine for number word conversion
p = inflect.engine()

def preprocess_text_word2number(text):
    """Cleans and normalizes text for fair comparison in VQA evaluation."""
    
    # Lowercasing
    text = text.lower()

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Convert number words to digits (e.g., "twenty five" → "25")
    try:
        text = w2n.word_to_num(text)
        text = str(text)
    except ValueError:
        pass  # Keep original text if it's not a number

    # Handle missing apostrophes in contractions (e.g., "dont" → "don't")
    contractions = {"dont": "don't", "wont": "won't", "isnt": "isn't", "arent": "aren't"}
    words = text.split()
    words = [contractions[word] if word in contractions else word for word in words]
    text = " ".join(words)

    # Remove punctuation (except apostrophes and colons)
    text = re.sub(r'[^a-zA-Z0-9:\'\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_text_inflect(text):
    """
    Preprocesses text with the following steps:
    - Converts to lowercase
    - Removes periods except in decimal numbers
    - Converts number words to digits
    - Removes articles (a, an, the)
    - Adds apostrophes to contractions
    - Replaces punctuation (except apostrophes and colons) with space
    - Preserves commas in numbers
    """
    # Convert to lowercase
    text = text.lower()

    # Remove periods except if part of a decimal number
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)  

    # Convert number words to digits
    words = text.split()
    for i, word in enumerate(words):
        if p.singular_noun(word):  # Check if word is a singular noun
            continue
        num_word = p.number_to_words(word)
        if num_word != "":  
            words[i] = num_word
    text = " ".join(words)

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # Add apostrophes for contractions
    contractions = {
        "dont": "don't", "doesnt": "doesn't", "cant": "can't", "wont": "won't", "isnt": "isn't",
        "arent": "aren't", "wasnt": "wasn't", "werent": "weren't", "havent": "haven't",
        "hasnt": "hasn't", "hadnt": "hadn't", "shouldnt": "shouldn't", "wouldnt": "wouldn't",
        "couldnt": "couldn't", "mustnt": "mustn't", "mightnt": "mightn't"
    }
    text = " ".join([contractions.get(word, word) for word in text.split()])

    # Replace punctuation with space, keeping apostrophes and colons
    text = re.sub(r"(?<!\d),(?!\d)", " ", text)  # Remove commas except between digits
    text = re.sub(r"[^\w\s':]", " ", text)  # Replace other punctuation with space

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# def normalize_predictions_references(predictions, references):
#     """
#     Ensures predictions are a list of strings and references are a list of lists of strings.
    
#     Args:
#         predictions (list): List of predicted answers (strings or lists).
#         references (list): List of reference answers (strings or lists of lists).
    
#     Returns:
#         tuple: (processed_predictions, processed_references)
#     """
#     normalized_predictions = []
#     normalized_references = []

#     for i, (pred, ref_list) in enumerate(zip(predictions, references)):
#         # Ensure prediction is always a string
#         if not isinstance(pred, str):
#             if isinstance(pred, list) and pred:  # If it's a non-empty list, take the first item
#                 pred = pred[0]
#             else:
#                 raise ValueError(f"Invalid prediction format at index {i}: {pred}")
#         normalized_predictions.append(pred)

#         # Ensure reference is a list of lists of strings
#         if isinstance(ref_list, str):  
#             ref_list = [[ref_list]]  # Convert single string to a list of lists
#         elif isinstance(ref_list, list):  
#             if all(isinstance(ref, str) for ref in ref_list):  # Case where ref_list = ["answer1", "answer2"]
#                 ref_list = [ref_list]  # Wrap in another list
#             elif all(isinstance(ref, list) for ref in ref_list):  # Case where ref_list = [["answer1"], ["answer2"]]
#                 pass  # Keep as is
#             else:
#                 raise ValueError(f"Invalid reference format at index {i}: {ref_list}")
#         else:
#             raise ValueError(f"Invalid reference format at index {i}: {ref_list}")

#         normalized_references.append(ref_list)

#     return normalized_predictions, normalized_references

def normalize_predictions_references(predictions, references):
    """
    Ensures predictions are a list of strings and references are a list of lists of strings.
    
    Args:
        predictions (str or list): Either a single string or a list of predicted answers.
        references (str or list): Either a single string, a list of strings, or a list of lists of strings.

    Returns:
        tuple: (processed_predictions, processed_references)
    """
    # Ensure predictions is a list of strings
    if isinstance(predictions, str):
        predictions = [predictions]  # Convert single string to a list of strings
    elif isinstance(predictions, list):
        predictions = [pred[0] if isinstance(pred, list) and pred else pred for pred in predictions]
    else:
        raise ValueError(f"Invalid predictions format: {predictions}")

    # Ensure references is a list of lists of strings
    if isinstance(references, str):
        references = [[references]]  # Convert single string to a list of lists
    elif isinstance(references, list):
        if all(isinstance(ref, str) for ref in references):  
            references = [references]  # Wrap in another list to make it a list of lists
        elif all(isinstance(ref, list) for ref in references):
            pass  # Already in correct format
        else:
            raise ValueError(f"Invalid reference format: {references}")
    else:
        raise ValueError(f"Invalid reference format: {references}")

    return predictions, references

def ensure_type(value: Union[str, List[Union[str, List[str]]]], target_type: type) -> Union[str, List[str]]:
    """
    Ensures that the given value is returned in the specified type (either str or List[str]).
    
    - If target_type is `str`, flattens nested lists and joins elements into a single string.
    - If target_type is `list`, flattens nested lists and ensures a list of strings.

    :param value: The input value, which can be a string, a list of strings, or a nested list.
    :param target_type: The desired output type (str or List[str]).
    :return: The value converted to the specified type.
    """

    # Helper function to flatten a nested list
    def flatten(lst):
        flat_list = []
        for item in lst:
            if isinstance(item, list):
                flat_list.extend(flatten(item))  # Recursively flatten
            else:
                flat_list.append(item)
        return flat_list
    
    if target_type == str:
        if isinstance(value, list):
            flat_value = flatten(value)  # Flatten if nested
            return " ".join(flat_value)  # Convert list to a single string
        return str(value)  # Ensure it's a string

    elif target_type == list:
        if isinstance(value, str):
            return [value]  # Convert single string to list
        elif isinstance(value, list):
            return flatten(value)  # Flatten nested lists and return a list of strings

    else:
        raise ValueError("target_type must be either 'str' or 'list'.")

