import pandas as pd
import torch
from typing import List, Dict
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes ROUGE, BLEU-4, and METEOR scores for a set of predictions and references.
    """
    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": [], "meteor": []}
    rouge = Rouge()

    for pred, ref in zip(predictions, references):
        hypothesis = pred.split()
        reference = ref.split()
        if len(pred) == 0 or len(ref) == 0:
            result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        else:
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))

        bleu_score = sentence_bleu([list(ref)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        meteor = meteor_score([reference], hypothesis)
        score_dict["meteor"].append(round(meteor * 100, 4))

    # Calculate the average of the scores across all predictions and references
    return {k: torch.tensor(v, device=device).mean().item() for k, v in score_dict.items()}

def calculate_metrics_from_csv(predictions_file: str, references_file: str):
    """
    Calculate and print the BLEU-4, ROUGE, and METEOR scores using data from CSV files.
    """
    # Load CSV files
    preds_df = pd.read_csv(predictions_file)
    refs_df = pd.read_csv(references_file)

    # Extract 'release' column data
    preds = preds_df['release'].fillna('').tolist()
    refs = refs_df['release'].fillna('').tolist()

    # Compute and print metrics
    metrics = compute_metrics(preds, refs)
    print(f"Predictions file: {predictions_file}, References file: {references_file}, Metrics: {metrics}")

# Specify paths to your CSV files
predictions_file = 'predictions.csv'
references_file = '/dataset/graph_rn/test.csv'

# Call the function to compute and print metrics
calculate_metrics_from_csv(predictions_file, references_file)

