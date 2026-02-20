import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pandas as pd
from scipy.stats import shapiro
import os
from pathlib import Path
from typing import List, Tuple, Union

from sharpness_matters.ptx.utils.logging_utils import initialize_logger

HOME = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="info")


def check_normality(data: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Test data normality using Shapiro-Wilk test.

    Args:
        data (np.ndarray): Sample data to test
        alpha (float): Significance level for normality test

    Returns:
        bool: True if data passes normality test (p > alpha), False otherwise
    """
    stat, p_value = shapiro(data)
    return p_value > alpha


def compare_groups(
    data1: np.ndarray, data2: np.ndarray, normal1: bool, normal2: bool
) -> float:
    """
    Compare two independent samples using appropriate statistical test.

    Args:
        data1 (np.ndarray): First sample data
        data2 (np.ndarray): Second sample data
        normal1 (bool): Whether first sample is normally distributed
        normal2 (bool): Whether second sample is normally distributed

    Returns:
        float: P-value from t-test (if both normal) or Mann-Whitney U test (otherwise)
    """
    if normal1 and normal2:
        # Use two-sample t-test (assuming equal variances for demonstration)
        _, p_value = ttest_ind(data1, data2, equal_var=True)
    else:
        # Use Mann–Whitney U test for nonparametric comparison
        _, p_value = mannwhitneyu(data1, data2, alternative="two-sided")
    return p_value


def load_data(
    model_name: str, metric: str, threshold: float, resolution: str, testing_mode: str
) -> List[float]:
    """
    Load saliency evaluation data for specified parameters.

    Args:
        model_name (str): Name of the model ('resnet' or 'densenet')
        metric (str): Evaluation metric name ('iou', 'precision', etc.)
        threshold (float): Threshold value used in evaluation
        resolution (str): Image resolution as string
        testing_mode (str): Testing mode ('external' or 'holdout')

    Returns:
        List[float]: List of metric values for the specified configuration
    """
    if testing_mode == "external":
        np_path = f"{HOME}/output/saliency_eval/{model_name}/bboxood{metric}_{resolution}_{threshold}.npy"
        if not os.path.exists(np_path):
            raise FileNotFoundError(
                f"File not found: {np_path}. Please run saliency evaluation first."
            )
        return list(np.load(np_path, allow_pickle=True).item().values())
    elif testing_mode == "holdout":
        np_path = f"{HOME}/output/saliency_eval/{model_name}/bbox{metric}_{resolution}_{threshold}.npy"
        if not os.path.exists(np_path):
            raise FileNotFoundError(
                f"File not found: {np_path}. Please run saliency evaluation first."
            )
        return list(np.load(np_path, allow_pickle=True).item().values())
    else:
        raise ValueError("Invalid testing mode. Choose between external and holdout")


def main(metric: str, testing_mode: str, out_csv: str, alpha: float = 0.05) -> None:
    """
    Perform statistical comparison of saliency metrics across resolutions with multiple testing correction.

    Args:
        metric (str): Evaluation metric to analyze ('iou', 'precision', etc.)
        testing_mode (str): Testing mode ('external' or 'holdout')
        out_csv (str): Output directory path for saving results
        alpha (float): Significance level for statistical tests
    """
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
                    testing_mode=testing_mode,
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
        os.path.join(out_csv, f"p_val_saliency_{testing_mode}_{metric}.csv"),
        index=False,
    )
    logger.info(
        f"\nResults saved to {os.path.join(out_csv,f'p_val_saliency_{testing_mode}_{metric}.csv')}"
    )


if __name__ == "__main__":
    main(metric="iou", testing_mode="holdout", out_csv=f"{HOME}/output/")
