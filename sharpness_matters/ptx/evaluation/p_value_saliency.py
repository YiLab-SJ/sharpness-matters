import numpy as np
from pathlib import Path
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
import pandas as pd
from scipy.stats import shapiro
import os
import click

from sharpness_matters.ptx.utils.logging_utils import initialize_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="debug")


def check_normality(data: list, alpha: float) -> bool:
    """
    Returns True if data passes Shapiro–Wilk test for normality (p > alpha),
    False otherwise.
    """
    stat, p_value = shapiro(data)
    return p_value > alpha


def compare_groups(data1: list, data2: list, normal1: bool, normal2: bool) -> float:
    """
    Compare two independent samples using either:
      - t-test if both appear normal
      - Mann–Whitney U if not
    Returns the p-value from the chosen test.
    """
    if normal1 and normal2:
        # Use two-sample t-test (assuming equal variances for demonstration)
        _, p_value = ttest_ind(data1, data2, equal_var=True)
    else:
        # Use Mann–Whitney U test for nonparametric comparison
        # _, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        if len(data1) == len(data2):
            _, p_value = wilcoxon(data1, data2)
        else:
            _, p_value = mannwhitneyu(data1, data2, alternative="two-sided")
    return p_value


def load_data(
    model_name: str, metric: str, threshold: float, resolution: str, dataset: str
) -> list:
    """
    Load metric predictions (.npy) for a model/resolution/threshold from external (OOD) or holdout dataset.
    """
    if dataset == "external":
        pred_path = f"{PROJECT_ROOT}/output/saliency_eval/{model_name}/ood{metric}_{resolution}_{threshold}.npy"
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(
                f"Prediction file not found: {pred_path}. "
                "Run evaluate_kfold.py first to generate prediction files."
            )
        return list(np.load(pred_path, allow_pickle=True).item().values())
    elif dataset == "holdout":
        pred_path = f"{PROJECT_ROOT}/output/saliency_eval/{model_name}/{metric}_{resolution}_{threshold}.npy"
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(
                f"Prediction file not found: {pred_path}. "
                "Run evaluate_kfold.py first to generate prediction files."
            )
        return list(np.load(pred_path, allow_pickle=True).item().values())
    else:
        raise ValueError("Invalid testing mode. Choose between external and holdout")


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help="Choose between external or holdout",
)
@click.option(
    "--metric",
    type=str,
    default="precision",
    help="Choose between precision or coverage",
)
@click.option(
    "--alpha",
    type=float,
    default=0.05,
    help="Significance level for tests and FDR correction",
)
def main(dataset: str, metric: str, alpha: float):
    """
    Run pairwise statistical comparisons of saliency/evaluation metrics across image resolutions
    for each model architecture, applying multiple-testing correction (FDR).

    Workflow:
    1. For each model in the predefined list (resnet, densenet):
        - For each target resolution in ['64','128','224','512','768','1024']:
          * Load metric arrays for three fixed internal thresholds (0.25, 0.5, 0.75) via `load_data`.
          * Average the arrays across those thresholds to obtain a single metrics vector per resolution.
    2. For every unique resolution pair within the same model:
        - Assess normality of both samples using `check_normality` (alpha provided).
        - Select and execute an appropriate group comparison test via `compare_groups`
          (e.g., t-test, Mann–Whitney, etc.—implementation-dependent).
        - Collect raw p-values.
    3. Apply Benjamini–Hochberg FDR correction across all accumulated comparisons (both models combined).
    4. Assemble results (model, resolution pair, raw p, corrected p, significance flag) into a DataFrame.
    5. Log a human-readable summary and persist results to:
          {PROJECT_ROOT}/output/p_val_saliency_{dataset}_{metric}.csv

    Parameters
    ----------
    dataset : str
         Identifier/name of the dataset passed through to `load_data`.
    metric : str
         Name of the evaluation or saliency metric to load and compare.
    threshold : float
         (Currently unused in this function.) A nominal threshold parameter; the function
         instead averages over fixed internal thresholds [0.25, 0.5, 0.75]. Retained
         for interface consistency or future extension.
    alpha : float
         Significance level used both for normality testing and as the FDR control
         level in Benjamini–Hochberg correction.

    Outputs
    -------
    CSV file with columns:
         model, resolution1, resolution2, raw_p_value, corrected_p_value, significant_fdr
    """
    out_csv = os.path.join(PROJECT_ROOT, "output", "p_values")
    resolutions = ["64", "128", "224", "512", "768", "1024"]
    models = ["resnet", "densenet"]

    # We'll collect all comparisons and their raw p-values in these lists
    comparisons = []
    p_values = []

    for model in models:
        # Load data for each resolution within this model
        data_dict = {}
        for res in resolutions:
            data_dict[res] = []
            for thresh in [0.25, 0.5, 0.75]:
                d = load_data(
                    model_name=model,
                    metric=metric,
                    threshold=thresh,
                    resolution=res,
                    dataset=dataset,
                )
                data_dict[res].append(d)
            data_dict[res] = np.mean(data_dict[res], axis=0)
            logger.debug(
                f"Res: {res}, Shape of averaged metrics: {data_dict[res].shape}"
            )

        # For each pair of resolutions within this model...
        for i in range(len(resolutions)):
            for j in range(i + 1, len(resolutions)):
                res1 = resolutions[i]
                res2 = resolutions[j]

                data1 = data_dict[res1]
                data2 = data_dict[res2]

                # Check normality
                normal1 = check_normality(data1, alpha=alpha)
                normal2 = check_normality(data2, alpha=alpha)

                # Compare groups
                p_val = compare_groups(data1, data2, normal1, normal2)

                # Store results
                comparisons.append((model, res1, res2))
                p_values.append(p_val)

    # Correct for multiple testing across *all* comparisons (both models)
    # using Benjamini–Hochberg (FDR)
    reject_list, pvals_corrected, _, _ = multipletests(
        p_values, alpha=alpha, method="fdr_bh"
    )
    # Build a pandas DataFrame of the results
    results = []
    for (model, r1, r2), raw_p, corr_p, reject in zip(
        comparisons, p_values, pvals_corrected, reject_list
    ):
        results.append(
            {
                "model": model,
                "resolution1": r1,
                "resolution2": r2,
                "raw_p_value": raw_p,
                "corrected_p_value": corr_p,
                "significant_fdr": reject,
            }
        )

    df = pd.DataFrame(results)
    # Print or return results
    logger.info("=== Pairwise Resolution Comparisons Within Each Model ===")
    for (model, r1, r2), raw_p, corr_p, reject in zip(
        comparisons, p_values, pvals_corrected, reject_list
    ):
        logger.info(f"Model: {model} | Compare: {r1} vs {r2}")
        logger.info(f"  Raw p-value:          {raw_p:.4g}")
        logger.info(f"  Corrected p-value:    {corr_p:.4g}")
        logger.info(f"  Significant (FDR)?    {reject}")
        logger.info("-" * 50)
    # Save to CSV
    df.to_csv(
        os.path.join(out_csv, f"p_val_saliency_{dataset}_{metric}.csv"), index=False
    )
    logger.info(
        f"\nResults saved to {os.path.join(out_csv,f'p_val_saliency_{dataset}_{metric}.csv')}"
    )


if __name__ == "__main__":
    main()
