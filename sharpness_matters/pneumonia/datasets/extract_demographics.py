"""
Demographic Analysis for RSNA Pneumonia and SIIM COVID Datasets

This script analyzes the demographic distribution (age and sex) of medical imaging datasets
in the saliency_pn project. It supports RSNA pneumonia and SIIM COVID datasets.

Usage:
    # Analyze RSNA pneumonia dataset (test split by default)
    python sharpness_matters/pneumonia/datasets/extract_demographics.py --dataset rsna_pneumonia
    
    # Analyze RSNA pneumonia train split
    python sharpness_matters/pneumonia/datasets/extract_demographics.py --dataset rsna_pneumonia --split train
    
    # Analyze both train and test splits of RSNA pneumonia
    python sharpness_matters/pneumonia/datasets/extract_demographics.py --dataset rsna_pneumonia --split both
    
    # Analyze SIIM COVID dataset  
    python sharpness_matters/pneumonia/datasets/extract_demographics.py --dataset siim_covid
    
    # Specify output directory
    python sharpness_matters/pneumonia/datasets/extract_demographics.py --dataset rsna_pneumonia --split both --output-dir /path/to/output
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import glob
from pathlib import Path
from typing import Union, Any, Dict

from sharpness_matters.pneumonia.datasets.rsna_pneumonia import ChestXrayHoldout
from sharpness_matters.pneumonia.datasets.siim_covid import ChestXrayOOD
from sharpness_matters.pneumonia.utils.logger_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="info")
HOME = Path(__file__).resolve().parent.parent
RSNA_PNEUMONIA_ROOT_DIR = cfg.pneumonia.rsna_pneumonia.root_dir
SIIM_COVID_ROOT_DIR = cfg.pneumonia.siim_covid.root_dir


def analyze_demographics(
    dataset: Union[ChestXrayHoldout, ChestXrayOOD],
    dataset_name: str = "Unknown",
    output_dir: str = f"{HOME}/output",
):
    """
    Analyze the demographic distribution of a medical imaging dataset.

    Parameters:
    - dataset: Dataset instance (ChestXrayHoldout or ChestXrayOOD)
    - dataset_name: Name of the dataset for labeling outputs
    - output_dir: Directory to save results and plots

    Returns:
    - dict: Dictionary containing all demographic statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Extracting demographics from {dataset_name} dataset...")
    demographics = dataset.get_all_demographics()

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(demographics)

    # Ensure age column is properly typed
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Samples with age data: {df['age'].notna().sum()}")
    logger.info(f"Samples with sex data: {df['sex'].notna().sum()}")

    # Basic statistics
    stats = {
        "dataset_name": dataset_name,
        "total_samples": len(df),
        "samples_with_age": int(df["age"].notna().sum()),
        "samples_with_sex": int(df["sex"].notna().sum()),
        "samples_with_both": int(df[["age", "sex"]].notna().all(axis=1).sum()),
        "positive_samples": int(df["label"].sum()),
        "negative_samples": int((df["label"] == 0).sum()),
    }

    logger.info(f"Basic stats: {stats}")

    # Age statistics
    age_stats = analyze_age_distribution(df, dataset_name, output_dir)
    stats.update(age_stats)

    # Sex statistics
    sex_stats = analyze_sex_distribution(df, dataset_name, output_dir)
    stats.update(sex_stats)

    # Cross-tabulation analysis
    crosstab_stats = analyze_crosstabs(df, dataset_name, output_dir)
    stats.update(crosstab_stats)

    # Save detailed demographics CSV
    output_csv = os.path.join(
        output_dir,
        f'{dataset_name.lower().replace(" ", "_").replace("-", "_")}_demographics.csv',
    )
    df.to_csv(output_csv, index=False)
    logger.info(f"Demographics saved to {output_csv}")

    # Save summary statistics
    output_json = os.path.join(
        output_dir,
        f'{dataset_name.lower().replace(" ", "_").replace("-", "_")}_demographic_stats.json',
    )
    with open(output_json, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    return stats, df


def analyze_age_distribution(
    df: pd.DataFrame, dataset_name: str, output_dir: str
) -> Dict[str, Any]:
    """
    Analyze age distribution and create visualizations.
    Parameters:
    - df: DataFrame containing demographic data
    - dataset_name: Name of the dataset for labeling outputs
    - output_dir: Directory to save results and plots
    Returns:
    - dict: Dictionary containing age statistics
    """
    age_stats = {}

    # Filter out missing age data
    age_data = df[df["age"].notna()]

    if len(age_data) > 0:
        age_stats["age_mean"] = float(age_data["age"].mean())
        age_stats["age_median"] = float(age_data["age"].median())
        age_stats["age_std"] = float(age_data["age"].std())
        age_stats["age_min"] = int(age_data["age"].min())
        age_stats["age_max"] = int(age_data["age"].max())
        age_stats["age_q25"] = float(age_data["age"].quantile(0.25))
        age_stats["age_q75"] = float(age_data["age"].quantile(0.75))

        # Age groups
        age_bins = [0, 18, 30, 45, 60, 75, 100]
        age_labels = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
        age_data = age_data.copy()
        age_data["age_group"] = pd.cut(
            age_data["age"], bins=age_bins, labels=age_labels, right=False
        )
        age_group_counts = age_data["age_group"].value_counts()
        age_stats["age_groups"] = {str(k): int(v) for k, v in age_group_counts.items()}

        # Visualizations
        plt.figure(figsize=(15, 5))

        # Age histogram
        plt.subplot(1, 3, 1)
        plt.hist(age_data["age"], bins=20, edgecolor="black", alpha=0.7)
        plt.title(f"{dataset_name} Age Distribution")
        plt.xlabel("Age (years)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Age by label
        plt.subplot(1, 3, 2)
        age_by_label = age_data.groupby("label")["age"].apply(list)
        labels = ["Negative", "Positive"]
        ages_by_label = [age_by_label.get(0, []), age_by_label.get(1, [])]
        plt.boxplot(ages_by_label, labels=labels)
        plt.title(f"Age Distribution by {dataset_name} Label")
        plt.ylabel("Age (years)")
        plt.grid(True, alpha=0.3)

        # Age groups bar plot
        plt.subplot(1, 3, 3)
        age_group_counts.plot(kind="bar")
        plt.title("Age Groups Distribution")
        plt.xlabel("Age Group")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f'{dataset_name.lower().replace(" ", "_").replace("-", "_")}_age_distribution.png',
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(
            f"Age statistics: Mean={age_stats['age_mean']:.1f}, Median={age_stats['age_median']:.1f}"
        )
    else:
        logger.warning("No age data available for analysis")

    return age_stats


def analyze_sex_distribution(
    df: pd.DataFrame, dataset_name: str, output_dir: str
) -> Dict[str, Any]:
    """
    Analyze sex distribution and create visualizations.

    Parameters:
    - df: DataFrame containing demographic data
    - dataset_name: Name of the dataset for labeling outputs
    - output_dir: Directory to save results and plots

    Returns:
    - dict: Dictionary containing sex statistics
    """
    sex_stats = {}

    # Filter out missing sex data
    sex_data = df[df["sex"].notna()]

    if len(sex_data) > 0:
        sex_counts = sex_data["sex"].value_counts()
        sex_stats["sex_distribution"] = sex_counts.to_dict()
        sex_stats["sex_percentages"] = {
            k: float(v / len(sex_data) * 100) for k, v in sex_counts.items()
        }

        # Visualizations
        plt.figure(figsize=(12, 4))

        # Sex distribution pie chart
        plt.subplot(1, 3, 1)
        plt.pie(
            sex_counts.values, labels=sex_counts.index, autopct="%1.1f%%", startangle=90
        )
        plt.title(f"{dataset_name} Sex Distribution")

        # Sex by label
        plt.subplot(1, 3, 2)
        sex_label_crosstab = pd.crosstab(sex_data["sex"], sex_data["label"])
        sex_label_crosstab.plot(kind="bar")
        plt.title(f"Sex Distribution by {dataset_name} Label")
        plt.xlabel("Sex")
        plt.ylabel("Count")
        plt.legend(["Negative", "Positive"])
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)

        # Sex proportions by label
        plt.subplot(1, 3, 3)
        sex_label_proportions = pd.crosstab(
            sex_data["sex"], sex_data["label"], normalize="columns"
        )
        sex_label_proportions.plot(kind="bar")
        plt.title(f"Sex Proportions by {dataset_name} Label")
        plt.xlabel("Sex")
        plt.ylabel("Proportion")
        plt.legend(["Negative", "Positive"])
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f'{dataset_name.lower().replace(" ", "_").replace("-", "_")}_sex_distribution.png',
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Sex distribution: {sex_stats['sex_distribution']}")
    else:
        logger.warning("No sex data available for analysis")

    return sex_stats


def analyze_crosstabs(
    df: pd.DataFrame, dataset_name: str, output_dir: str
) -> Dict[str, Any]:
    """
    Analyze cross-tabulations between demographics and labels.
    Parameters:
    - df: DataFrame containing demographic data
    - dataset_name: Name of the dataset for labeling outputs
    - output_dir: Directory to save results and plots
    Returns:
    - dict: Dictionary containing crosstab statistics
    """
    crosstab_stats = {}

    # Filter data with both age and sex
    complete_data = df[df[["age", "sex"]].notna().all(axis=1)]

    if len(complete_data) > 0:
        # Age groups for crosstab
        age_bins = [0, 30, 50, 70, 100]
        age_labels = ["0-29", "30-49", "50-69", "70+"]
        complete_data = complete_data.copy()
        complete_data["age_group"] = pd.cut(
            complete_data["age"], bins=age_bins, labels=age_labels, right=False
        )

        # Crosstabs
        age_sex_crosstab = pd.crosstab(complete_data["age_group"], complete_data["sex"])
        age_label_crosstab = pd.crosstab(
            complete_data["age_group"], complete_data["label"]
        )
        sex_label_crosstab = pd.crosstab(complete_data["sex"], complete_data["label"])

        # Store as dictionaries
        crosstab_stats["age_sex_crosstab"] = age_sex_crosstab.to_dict()
        crosstab_stats["age_label_crosstab"] = age_label_crosstab.to_dict()
        crosstab_stats["sex_label_crosstab"] = sex_label_crosstab.to_dict()

        # Visualizations
        plt.figure(figsize=(15, 5))

        # Age-Sex heatmap
        plt.subplot(1, 3, 1)
        sns.heatmap(age_sex_crosstab, annot=True, fmt="d", cmap="Blues")
        plt.title("Age Group × Sex")
        plt.ylabel("Age Group")
        plt.xlabel("Sex")

        # Age-Label heatmap
        plt.subplot(1, 3, 2)
        sns.heatmap(age_label_crosstab, annot=True, fmt="d", cmap="Oranges")
        plt.title(f"Age Group × {dataset_name} Label")
        plt.ylabel("Age Group")
        plt.xlabel("Label (0=Neg, 1=Pos)")

        # Sex-Label heatmap
        plt.subplot(1, 3, 3)
        sns.heatmap(sex_label_crosstab, annot=True, fmt="d", cmap="Greens")
        plt.title(f"Sex × {dataset_name} Label")
        plt.ylabel("Sex")
        plt.xlabel("Label (0=Neg, 1=Pos)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f'{dataset_name.lower().replace(" ", "_").replace("-", "_")}_demographic_crosstabs.png',
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info("Cross-tabulation analysis completed")
    else:
        logger.warning("Insufficient data for cross-tabulation analysis")

    return crosstab_stats


def print_summary(stats: Dict[str, Any]) -> None:
    """Print a formatted summary of the demographic analysis."""
    dataset_name = stats.get("dataset_name", "Unknown")
    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} DATASET DEMOGRAPHIC ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nDATASET OVERVIEW:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Positive samples: {stats['positive_samples']:,}")
    print(f"  Negative samples: {stats['negative_samples']:,}")
    print(
        f"  Samples with age data: {stats['samples_with_age']:,} ({stats['samples_with_age']/stats['total_samples']*100:.1f}%)"
    )
    print(
        f"  Samples with sex data: {stats['samples_with_sex']:,} ({stats['samples_with_sex']/stats['total_samples']*100:.1f}%)"
    )
    print(f"  Samples with both age & sex: {stats['samples_with_both']:,}")

    if "age_mean" in stats:
        print(f"\nAGE STATISTICS:")
        print(f"  Mean age: {stats['age_mean']:.1f} years")
        print(f"  Median age: {stats['age_median']:.1f} years")
        print(f"  Age range: {stats['age_min']} - {stats['age_max']} years")
        print(f"  Standard deviation: {stats['age_std']:.1f} years")

        if "age_groups" in stats:
            print(f"\nAGE GROUP DISTRIBUTION:")
            for group, count in stats["age_groups"].items():
                if group != "nan":
                    print(f"  {group}: {count:,}")

    if "sex_distribution" in stats:
        print(f"\nSEX DISTRIBUTION:")
        for sex, count in stats["sex_distribution"].items():
            percentage = stats["sex_percentages"][sex]
            print(f"  {sex}: {count:,} ({percentage:.1f}%)")

    print("\n" + "=" * 60)


def create_split_comparison(combined_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create comparison visualizations between train and test splits.
    Parameters:
    - combined_df: DataFrame containing combined demographics with 'split' column
    - output_dir: Directory to save the comparison plots
    """
    plt.figure(figsize=(15, 10))

    # Age distribution comparison
    plt.subplot(2, 3, 1)
    train_ages = combined_df[combined_df["split"] == "train"]["age"].dropna()
    test_ages = combined_df[combined_df["split"] == "test"]["age"].dropna()

    plt.hist(
        [train_ages, test_ages],
        bins=20,
        alpha=0.7,
        label=["Train", "Test"],
        edgecolor="black",
    )
    plt.title("Age Distribution: Train vs Test")
    plt.xlabel("Age (years)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Age boxplot comparison
    plt.subplot(2, 3, 2)
    age_data_by_split = [train_ages, test_ages]
    plt.boxplot(age_data_by_split, labels=["Train", "Test"])
    plt.title("Age Distribution: Train vs Test")
    plt.ylabel("Age (years)")
    plt.grid(True, alpha=0.3)

    # Sex distribution comparison
    plt.subplot(2, 3, 3)
    sex_split_crosstab = pd.crosstab(
        combined_df["sex"], combined_df["split"], normalize="columns"
    )
    sex_split_crosstab.plot(kind="bar", ax=plt.gca())
    plt.title("Sex Distribution: Train vs Test")
    plt.xlabel("Sex")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(["Test", "Train"])
    plt.grid(True, alpha=0.3)

    # Label distribution comparison
    plt.subplot(2, 3, 4)
    label_split_crosstab = pd.crosstab(
        combined_df["label"], combined_df["split"], normalize="columns"
    )
    label_split_crosstab.plot(kind="bar", ax=plt.gca())
    plt.title("Label Distribution: Train vs Test")
    plt.xlabel("Label (0=Neg, 1=Pos)")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(["Test", "Train"])
    plt.grid(True, alpha=0.3)

    # Age by label and split
    plt.subplot(2, 3, 5)
    for split in ["train", "test"]:
        split_data = combined_df[combined_df["split"] == split]
        age_by_label = split_data.groupby("label")["age"].apply(list)
        ages_neg = age_by_label.get(0, [])
        ages_pos = age_by_label.get(1, [])

        if ages_neg:
            plt.boxplot(
                [ages_neg],
                positions=[0 if split == "train" else 0.4],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(
                    facecolor="lightblue" if split == "train" else "lightcoral"
                ),
            )
        if ages_pos:
            plt.boxplot(
                [ages_pos],
                positions=[1 if split == "train" else 1.4],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(
                    facecolor="lightblue" if split == "train" else "lightcoral"
                ),
            )

    plt.title("Age by Label and Split")
    plt.ylabel("Age (years)")
    plt.xticks([0.2, 1.2], ["Negative", "Positive"])
    plt.grid(True, alpha=0.3)

    # Sample size comparison
    plt.subplot(2, 3, 6)
    split_counts = combined_df["split"].value_counts()
    plt.bar(split_counts.index, split_counts.values, color=["lightblue", "lightcoral"])
    plt.title("Sample Size: Train vs Test")
    plt.xlabel("Split")
    plt.ylabel("Count")
    for i, v in enumerate(split_counts.values):
        plt.text(i, v + max(split_counts.values) * 0.01, str(v), ha="center")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "rsna_pneumonia_split_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("Split comparison visualization saved")


def load_rsna_dataset(split: str = "test") -> ChestXrayHoldout:
    """
    Load RSNA pneumonia dataset with default parameters.
    """
    # Default paths for RSNA dataset - you may need to adjust these
    dicom_dir = RSNA_PNEUMONIA_ROOT_DIR
    label_dir = os.path.join(RSNA_PNEUMONIA_ROOT_DIR, "stage_2_train_labels.csv")

    # Get all DICOM files
    file_paths = glob.glob(os.path.join(dicom_dir, "**/*.dcm"), recursive=True)
    if not file_paths:
        file_paths = glob.glob(os.path.join(dicom_dir, "**/*.dicom"), recursive=True)

    if not file_paths:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label file not found at {label_dir}")

    logger.info(f"Found {len(file_paths)} DICOM files")

    dataset = ChestXrayHoldout(
        file_paths=file_paths,
        label_dir=label_dir,
        img_size=224,
        split=split,  # Use specified split
        test_size=0.1,
    )

    return dataset


def load_siim_covid_dataset() -> ChestXrayOOD:
    """Load SIIM COVID dataset with default parameters."""
    # Default paths for SIIM COVID dataset - you may need to adjust these
    dicom_dir = SIIM_COVID_ROOT_DIR
    study_label_dir = os.path.join(SIIM_COVID_ROOT_DIR, "covid_study_level.csv")
    image_label_dir = os.path.join(SIIM_COVID_ROOT_DIR, "covid_image_level.csv")

    # Get all DICOM files
    file_paths = glob.glob(os.path.join(dicom_dir, "**/*.dcm"), recursive=True)
    if not file_paths:
        file_paths = glob.glob(os.path.join(dicom_dir, "**/*.dicom"), recursive=True)

    if not file_paths:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    if not os.path.exists(study_label_dir) or not os.path.exists(image_label_dir):
        raise FileNotFoundError(
            f"Label files not found at {study_label_dir} or {image_label_dir}"
        )

    logger.info(f"Found {len(file_paths)} DICOM files")

    dataset = ChestXrayOOD(
        file_paths=file_paths,
        label_dirs=[study_label_dir, image_label_dir],
        img_size=224,
    )

    return dataset


def main():
    """Main function to run the demographic analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze demographics of pneumonia/COVID datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["rsna_pneumonia", "siim_covid"],
        default="rsna_pneumonia",
        help="Dataset to analyze: rsna_pneumonia or siim_covid",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="test",
        help="Data split to analyze: train, test, or both (only for RSNA pneumonia)",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{HOME}/output/demographics",
        help="Output directory for results",
    )

    args = parser.parse_args()

    logger.info(f"Starting {args.dataset} dataset demographic analysis...")

    try:
        # Initialize dataset based on choice
        if args.dataset == "rsna_pneumonia":
            if args.split == "both":
                # Analyze both train and test splits
                logger.info("Analyzing both train and test splits...")

                # Train split
                logger.info("Processing training split...")
                train_dataset = load_rsna_dataset(split="train")
                train_dataset_name = "RSNA Pneumonia Train"
                train_stats, train_df = analyze_demographics(
                    train_dataset, train_dataset_name, args.output_dir
                )

                # Test split
                logger.info("Processing test split...")
                test_dataset = load_rsna_dataset(split="test")
                test_dataset_name = "RSNA Pneumonia Test"
                test_stats, test_df = analyze_demographics(
                    test_dataset, test_dataset_name, args.output_dir
                )

                # Print summaries for both
                print_summary(train_stats)
                print_summary(test_stats)

                # Combined analysis
                logger.info("Creating combined analysis...")
                combined_df = pd.concat(
                    [train_df.assign(split="train"), test_df.assign(split="test")],
                    ignore_index=True,
                )

                # Save combined demographics
                combined_csv = os.path.join(
                    args.output_dir, "rsna_pneumonia_combined_demographics.csv"
                )
                combined_df.to_csv(combined_csv, index=False)
                logger.info(f"Combined demographics saved to {combined_csv}")

                # Create split comparison visualization
                create_split_comparison(combined_df, args.output_dir)

                logger.info("Analysis completed for both splits!")
                return

            else:
                dataset = load_rsna_dataset(split=args.split)
                dataset_name = f"RSNA Pneumonia {args.split.title()}"

        elif args.dataset == "siim_covid":
            dataset = load_siim_covid_dataset()
            dataset_name = "SIIM COVID"

        # Run analysis
        stats, df = analyze_demographics(dataset, dataset_name, args.output_dir)

        # Print summary
        print_summary(stats)

        logger.info("Analysis completed successfully!")
        logger.info("Results saved in output directory:")
        dataset_filename = dataset_name.lower().replace(" ", "_").replace("-", "_")
        logger.info(
            f"  - {dataset_filename}_demographics.csv: Detailed demographics data"
        )
        logger.info(
            f"  - {dataset_filename}_demographic_stats.json: Summary statistics"
        )
        logger.info(f"  - {dataset_filename}_age_distribution.png: Age analysis plots")
        logger.info(f"  - {dataset_filename}_sex_distribution.png: Sex analysis plots")
        logger.info(
            f"  - {dataset_filename}_demographic_crosstabs.png: Cross-tabulation heatmaps"
        )

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
