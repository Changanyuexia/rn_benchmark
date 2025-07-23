import pandas as pd
import torch
from typing import List, Dict
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
import concurrent.futures

# Download WordNet and Punkt (only needed for the first run)
nltk.download('wordnet')
nltk.download('punkt')

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Compute evaluation metrics: ROUGE-L, BLEU-4, METEOR. Each pair is evaluated in parallel and averaged.
    """
    score_dict = {"rouge-l": [], "bleu-4": [], "meteor": []}
    rouge = Rouge()
    smoothing = SmoothingFunction().method3

    def single_eval(args):
        pred, ref = args
        if not pred.strip() or not ref.strip():
            return 0.0, 0.0, 0.0
        # Lowercase and tokenize
        hypothesis = nltk.word_tokenize(pred.lower())
        reference = nltk.word_tokenize(ref.lower())
        # ROUGE-L
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        rouge_l = round(scores[0]["rouge-l"]["f"] * 100, 4)
        # BLEU-4 (corpus-level for this pair)
        try:
            bleu_score = nltk.translate.bleu_score.corpus_bleu(
                [[reference]], [hypothesis],
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            bleu = round(bleu_score * 100, 4)
        except Exception:
            bleu = 0.0
        # METEOR
        try:
            meteor = meteor_score([reference], hypothesis)
            meteor = round(meteor * 100, 4)
        except Exception:
            meteor = 0.0
        return rouge_l, bleu, meteor

    # Parallel computation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(single_eval, zip(preds, refs)))

    # Collect results
    for rouge_l, bleu, meteor in results:
        score_dict["rouge-l"].append(rouge_l)
        score_dict["bleu-4"].append(bleu)
        score_dict["meteor"].append(meteor)

    # Compute average and return
    return {k: round(sum(v) / len(v), 4) for k, v in score_dict.items()}


def find_column(df, candidates):
    """
    Find the first matching column name in candidates from the DataFrame.
    """
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of the columns {candidates} found in DataFrame columns: {df.columns.tolist()}")

def calculate_metrics_from_csv(predictions_file: str, references_file: str):
    """
    Read predictions and references from CSV files, auto-detect column names, and compute metrics.
    """
    # Load CSV files
    preds_df = pd.read_csv(predictions_file)
    refs_df  = pd.read_csv(references_file)

    # Candidate column names for predictions and references
    pred_candidates = ['release', 'predict', 'prediction', 'pred', 'output']
    ref_candidates  = ['release', 'reference', 'ground_truth', 'target', 'ref', 'gt']

    pred_col = find_column(preds_df, pred_candidates)
    ref_col  = find_column(refs_df, ref_candidates)

    preds = preds_df[pred_col].fillna('').tolist()
    refs  = refs_df[ref_col].fillna('').tolist()

    # Compute and print metrics
    metrics = compute_metrics(preds, refs)
    print(f"Predictions: {predictions_file}\nReferences:  {references_file}\nMetrics:     {metrics}")

if __name__ == "__main__":
    # Please replace the following with your actual file paths
    predictions_file = 'predictions.csv'  # Path to your predictions CSV
    references_file = 'test.csv'    # Path to your references CSV
    calculate_metrics_from_csv(predictions_file, references_file)