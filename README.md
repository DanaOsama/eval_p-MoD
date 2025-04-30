# Evaluation of p-MoD

To run this repository, create a new environment using the requirements.txt. Follow the steps below:
```
git clone <repository_URL>
cd eval_p-MoD
conda create --name eval_pmod python=3.10
conda activate eval_pmod
pip install -r requirements.txt
```

To run the evaluation script in this repository, you need to specify the dataset name, split, model name, and model checkpoint (for the huggingface models). 

Here is an example command of running the evaluation task on the [Doc-VQA](https://huggingface.co/datasets/lmms-lab/DocVQA) dataset using the validation split and the PaliGemma model on huggingface:

`python main.py --task eval --ckpt "google/paligemma-3b-ft-docvqa-448" --model "hf_paligemma" --dataset doc-vqa --split validation`

## Integrated Datasets

### 1. [OCR-VQA](https://huggingface.co/datasets/howard-hou/OCR-VQA)
All three splits are available on howard-hou/OCR-VQA: 'train', 'validation', and 'test'.
There is an error here related to grayscale images that I am currently working on fixing.

**Metric used**: exact_match

### 2. [Text-VQA](https://huggingface.co/datasets/lmms-lab/textvqa)
All three splits are available on lmm-lab/textvqa: 'train', 'validation', and 'test'. The test set's answers column has empty values. Also, there is no available platform to get results on the test set for this challenge as it is closed (as of 30 April 2025, this is still true). Therefore, you can report results on the validation set. 

**Metric used**: [vqa](https://visualqa.org/evaluation.html)

### 3. [Doc-VQA](https://huggingface.co/datasets/lmms-lab/DocVQA)
Only splits available on lmm-lab/DocVQA are 'test' and 'validation'.The validation set has an "answers" column, whereas the test set's is full of null values.

You can submit the results you generate from the test set to this [link](https://rrc.cvc.uab.es/?ch=17&com=tasks) for evaluation. 

**Metric used**: anls

### 4. [Info-VQA](https://huggingface.co/datasets/lmms-lab/DocVQA/viewer/InfographicVQA)
Only splits available on lmm-lab/DocVQA for InfoVQAare 'test' and 'validation'. The validation set has an "answers" column, whereas the test set's is full of null values.

You can submit the results you generate from the test set to this [link](https://rrc.cvc.uab.es/?ch=17&com=tasks) for evaluation. 

**Metric used**: anls

### 5. [ST-VQA](https://huggingface.co/datasets/lmms-lab/ST-VQA)
Only split available on lmm-lab/ST-VQA is 'test'. There is no "answers" column in the test set.

Only splits available on lmm-lab/DocVQA for InfoVQAare 'test' and 'validation'.
You can submit the results you generate from the test set to this [link](https://rrc.cvc.uab.es/?ch=11&com=tasks) for evaluation. 

**Metric used**: anls

**Note**: A json file of the results is automatically generated if the the metrics were not generated. You can save the json results in other cases by adding the `--save_json` flag to your evaluation command. 

### Integrating Another Dataset
To integrate another dataset into this repository, you would need to edit two main things:
1. In `data/dataset_loader.py`, create a new function for your dataset with the naming convention `load_<dataset_name>`. In this function, you need to define the name of the dataset from Hugging Face (HF), call the `check_split_availability` function, then simple use HF's `load_dataset` method. You can follow the same logic as `load_st_vqa`.

2. In `registry.py`, import your new loading function `load_<dataset_name>`. Then, add the name of your dataset and your loading function to the `DATASET_REGISTRY` dictionary as: `"dataset_name": load_<dataset_name>`.

## Integration of p-MoD into PaliGemma

This work aims to evaluate the performance of the p-MoD method [1] on the PaliGemma model [2]. The choice of model is made because of the presence of different resolutions in the vision encoder of PaliGemma, which makes it easy to evaluate how well p-MoD is performing when being used on a higher resolution model versus on a model with a lower resolution.

## Evaluation Results

| Dataset/Checkpoint | 224 | 448 | 896 |
|--------------------|-----|-----|-----|
| OCR-VQA            | 1   | 2   | 3   |
| Text-VQA           | 4   | 5   | 6   |
| Doc-VQA            | 7   | 8   | 9   |
| Info-VQA           | 10  | 11  | 12  |
| ST-VQA             | 13  | 14  | 15  |

## References

[1]	J. Zhang, D. Meng, J. Qi, Z. Huang, T. Wu, and L. Wang, ‘p-MoD: Building Mixture-of-Depths MLLMs via Progressive Ratio Decay’, *arXiv [cs.CV]*. 2024.

[2]	L. Beyer *et al.*, ‘PaliGemma: A versatile 3B VLM for transfer’, *arXiv [cs.CV]*. 2024.
