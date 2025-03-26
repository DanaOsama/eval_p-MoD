import re
import inflect
import word2number


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
        text = word2number.w2n.word_to_num(text)
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
