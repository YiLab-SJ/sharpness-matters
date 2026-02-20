import os
import numpy as np
from MLstatkit.stats import Delong_test
from scipy.stats import norm
import pandas as pd
import itertools
import click
from pathlib import Path
from statsmodels.stats.multitest import multipletests

from sharpness_matters.ptx.utils.logging_utils import initialize_logger

HOME = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="info")


def delong_test(file1: str, file2: str, labels_file: str) -> tuple:
    """
    Perform DeLong test for two sets of prediction logits.
    """
    # Load prediction logits
    logits1 = np.load(file1)
    logits2 = np.load(file2)
    labels = np.load(labels_file)
    print(
        f"Len of files: {len(labels)}, {len(logits1)}, {len(logits2)}"
    )  # Sanity len of loaded logits and labels
    # Ensure labels are binary (0 or 1)
    assert set(np.unique(labels)) == {0, 1}, "Labels must be binary (0 or 1)."

    z_score, p_value = Delong_test(labels, logits1, logits2)

    return z_score, p_value


def process_delong_tests(folder_path: str, resolutions: list, fn: str) -> pd.DataFrame:
    """
    Process DeLong tests for all resolution pairs within a specified folder.
    """
    # Store results separately for each model
    results_model_1 = []
    results_model_2 = []
    models = ["resnet", "densenet"]

    # Loop through each model
    for model in models:  # Assuming `models` is a list with 2 model names
        results = []  # Temporary storage for this model

        # Generate all pairs of resolutions to compare
        for res1, res2 in itertools.combinations(resolutions, 2):
            # Construct file paths for predictions
            pred_res1 = os.path.join(folder_path, f"{fn}_{model}_{res1}.npy")
            pred_res2 = os.path.join(folder_path, f"{fn}_{model}_{res2}.npy")
            gt_file = os.path.join(
                folder_path, "oodgt.npy" if "ood" in fn else "gt.npy"
            )

            # Ensure all files exist
            if not all(os.path.exists(f) for f in [pred_res1, pred_res2, gt_file]):
                print(
                    f"Skipping comparison {res1} vs {res2} for {model}: Missing files."
                )
                continue

            # Compute DeLong test
            z_score, p_value = delong_test(
                pred_res1, pred_res2, gt_file
            )  # Assuming gt files are the same across resolutions

            # Store raw results
            results.append(
                {
                    "Model": model,
                    "Resolution 1": res1,
                    "Resolution 2": res2,
                    "Z-score": z_score,
                    "P-value": p_value,
                }
            )

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Apply multiple testing correction **across all resolution comparisons for this model**
        if not results_df.empty:
            raw_p_values = results_df["P-value"].values

            # Bonferroni correction
            results_df["P-value Bonferroni"] = multipletests(
                raw_p_values, method="bonferroni"
            )[1]

            # Holm-Bonferroni correction
            results_df["P-value Holm"] = multipletests(raw_p_values, method="holm")[1]

            # False Discovery Rate (Benjamini-Hochberg)
            results_df["P-value FDR (BH)"] = multipletests(
                raw_p_values, method="fdr_bh"
            )[1]

            # Store results separately for each model
            if model == models[0]:
                results_model_1 = results_df
            else:
                results_model_2 = results_df

    # Ensure the directory exists before saving
    os.makedirs(folder_path, exist_ok=True)

    # Save to CSV file
    output_file = os.path.join(folder_path, f"delong_test_results_{fn}_resnet.csv")
    results_model_1.to_csv(output_file, index=False)
    output_file = os.path.join(folder_path, f"delong_test_results_{fn}_densenet.csv")
    results_model_2.to_csv(output_file, index=False)

    return results_df


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help="Choose between external or holdout",
)
def main(dataset):
    if dataset == "external":
        fn = "oodpred"
    elif dataset == "holdout":
        fn = "pred"
    folder_path = f"{HOME}/output/classif_preds"
    resolutions = ["64", "128", "224", "512", "768", "1024"]
    delong_results = process_delong_tests(folder_path, resolutions, fn)


if __name__ == "__main__":
    main()
