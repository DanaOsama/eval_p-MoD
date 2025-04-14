import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import train
from evaluate import evaluate

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from data.dataset_loader import load_hf_dataset

# Dictionary mapping model names to constructors
def create_hf_paligemma(checkpoint):
    return PaliGemmaForConditionalGeneration.from_pretrained(checkpoint)

