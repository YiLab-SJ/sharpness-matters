import os
import numpy as np
from pathlib import Path

# from MLstatkit import Delong_test
from MLstatkit.stats import Delong_test
from scipy.stats import norm
import pandas as pd
import itertools
import click
from statsmodels.stats.multitest import multipletests

from sharpness_matters.ptx.utils.logging_utils import initialize_logger

logger = initialize_logger(level="info")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def delong_test(file1: str, file2: str, labels_file: str) -> tuple:
    """
    Perform DeLong test to compare AUC of two models on the same dataset.

    Args:
        file1 (str): Path to first model's prediction logits (.npy file)
        file2 (str): Path to second model's prediction logits (.npy file)
        labels_file (str): Path to ground truth binary labels (.npy file)

    Returns:
        tuple: Z-score and p-value from DeLong test
    """
    # Load prediction logits
    logits1 = np.load(file1)
    logits2 = np.load(file2)
    labels = np.load(labels_file)
    logger.debug(
        f"Len of files: {len(labels)}, {len(logits1)}, {len(logits2)}"
    )  # Sanity len of loaded logits and labels
    # Ensure labels are binary (0 or 1)
    assert set(np.unique(labels)) == {0, 1}, "Labels must be binary (0 or 1)."

    z_score, p_value = Delong_test(labels, logits1, logits2)

    return z_score, p_value


def process_delong_tests(
    saved_pred_path: str, model: str, resolutions: list, fn: str
) -> pd.DataFrame:
    """
    Process DeLong tests for all resolution pairs and apply multiple testing corrections.

    Args:
        saved_pred_path (str): Path to directory containing prediction files
        model (str): Model name (resnet or densenet)
        resolutions (list): List of resolution strings to compare
        fn (str): File prefix (oodpred or holdoutpred)

    Returns:
        pd.DataFrame: Results with Z-scores, p-values, and corrected p-values
    """
    results = []

    # Generate all pairs of resolutions to compare
    for res1, res2 in itertools.combinations(resolutions, 2):
        # Construct file paths for predictions
        pred_res1 = os.path.join(saved_pred_path, f"{fn}_{model}_{res1}.npy")
        pred_res2 = os.path.join(saved_pred_path, f"{fn}_{model}_{res2}.npy")
        if "ood" in fn:
            gt_file = os.path.join(saved_pred_path, f"oodgt.npy")
        else:
            gt_file = os.path.join(saved_pred_path, f"holdoutgt.npy")

        # Ensure all files exist
        if not all(os.path.exists(f) for f in [pred_res1, pred_res2, gt_file]):
            logger.warning(f"Skipping comparison {res1} vs {res2}: Missing files.")
            continue

        # Compute DeLong test
        z_score, p_value = delong_test(
            pred_res1, pred_res2, gt_file
        )  # Assuming gt files are the same across resolutions

        # Store results
        results.append(
            {
                "Model": model,
                "Resolution 1": res1,
                "Resolution 2": res2,
                "Z-score": z_score,
                "P-value": p_value,
            }
        )

    results_df = pd.DataFrame(results)

    # Apply multiple testing correction
    if not results_df.empty:
        raw_p_values = results_df["P-value"].values

        # Bonferroni correction
        results_df["P-value Bonferroni"] = multipletests(
            raw_p_values, method="bonferroni"
        )[1]

        # Holm-Bonferroni correction (step-down method)
        results_df["P-value Holm"] = multipletests(raw_p_values, method="holm")[1]

        # False Discovery Rate (Benjamini-Hochberg)
        results_df["P-value FDR (BH)"] = multipletests(raw_p_values, method="fdr_bh")[1]

    # Ensure the directory exists before saving
    out_dir = os.path.join(PROJECT_ROOT, "output", "p_values")
    os.makedirs(out_dir, exist_ok=True)

    # Save to CSV file
    output_file = os.path.join(out_dir, f"delong_test_results_{fn}_{model}.csv")
    results_df.to_csv(output_file, index=False)

    return results_df


@click.command()
@click.option(
    "--model", "-m", type=str, required=True, help="Choose between resnet or densenet"
)
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help="Choose between external or holdout",
)
def main(model: str, dataset: str):
    """
    Main function to run DeLong statistical tests comparing model performance across resolutions.

    Args:
        model (str): Model architecture (resnet or densenet)
        dataset (str): Dataset type (external or holdout)
    """
    if dataset == "external":
        fn = "oodpred"
    elif dataset == "holdout":
        fn = "holdoutpred"
    else:
        raise ValueError("Invalid dataset choice. Choose 'external' or 'holdout'.")
    saved_pred_path = os.path.join(PROJECT_ROOT, "output", "classif_preds")
    if not os.path.isdir(saved_pred_path):
        raise FileNotFoundError(
            f"Predictions directory not found: {saved_pred_path}. "
            "Run evaluate_kfold.py first to generate prediction files."
        )
    resolutions = ["64", "128", "224", "512", "768", "1024"]
    delong_results = process_delong_tests(saved_pred_path, model, resolutions, fn)


if __name__ == "__main__":
    main()
