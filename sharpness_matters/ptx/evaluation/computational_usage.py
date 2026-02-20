import os
import glob
import time
import click
import torch
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
import pandas as pd
from pathlib import Path

from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="info")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = cfg.ptx.model.checkpoint_dir


def measure_peak_memory(
    model: torch.nn.Module, dummy_input: torch.Tensor, device: torch.device
) -> float:
    """
    Measures peak GPU memory usage (in MB) during a forward pass.

    Args:
        model: PyTorch model to evaluate
        dummy_input: Input tensor for forward pass
        device: PyTorch device (cuda/cpu)

    Returns:
        float: Peak memory usage in MB
    """
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    # Get peak memory allocated in bytes and convert to MB
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    return peak_memory


def measure_inference_time(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    num_runs=100,
    device=torch.device("cuda"),
) -> float:
    """
    Measures average inference time per forward pass (per batch).

    Args:
        model: PyTorch model to evaluate
        dummy_input: Input tensor for forward pass
        num_runs (int): Number of inference runs for averaging
        device: PyTorch device (cuda/cpu)

    Returns:
        float: Average inference time in seconds
    """
    model.eval()
    # Warm-up runs (important for GPU timing)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    return avg_time


def measure_throughput(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    num_runs=100,
    device=torch.device("cuda"),
) -> float:
    """
    Computes throughput in samples per second.

    Args:
        model: PyTorch model to evaluate
        dummy_input: Input tensor for forward pass
        num_runs (int): Number of inference runs for averaging
        device: PyTorch device (cuda/cpu)

    Returns:
        float: Throughput in samples per second
    """
    avg_time = measure_inference_time(model, dummy_input, num_runs, device)
    batch_size = dummy_input.size(0)
    throughput = batch_size / avg_time
    return throughput


def calculate_flops(model: torch.nn.Module, input_size: tuple) -> tuple:
    """
    Uses ptflops to calculate FLOPs (as MACs) and the number of parameters.

    Args:
        model: PyTorch model to analyze
        input_size (tuple): Input dimensions (channels, height, width)

    Returns:
        tuple: (MACs as string, Parameters as string)
    """
    macs, params = get_model_complexity_info(
        model, input_size, as_strings=True, print_per_layer_stat=False, verbose=False
    )
    return macs, params


@click.command()
@click.option(
    "--model_name",
    type=str,
    required=True,
    help="Choose a model type between densenet and resnet",
)
def main(model_name: str):
    """
    Main function to benchmark model performance across different image sizes.

    Args:
        model_name (str): Model architecture (densenet or resnet)
    """
    IMG_SIZES = [64, 128, 224, 512, 768, 1024]
    MODEL_NAME = model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_list = []
    for img_size in IMG_SIZES:
        logger.info(f"-----------Img Size: {img_size}-----------")
        CKPT_PATH = f"{CHECKPOINT_DIR}/{MODEL_NAME}/{MODEL_NAME}_{img_size}_logs/lightning_logs/version_0/checkpoints/"
        CKPT_PATH = os.path.join(CKPT_PATH, os.listdir(CKPT_PATH)[0])
        if not os.path.exists(CKPT_PATH):
            logger.error(f"Checkpoint not found: {CKPT_PATH}")
            continue
        logger.info(f"Loading model from checkpoint: {CKPT_PATH}")
        model = CNNBinaryClassifier.load_from_checkpoint(CKPT_PATH)
        input_size = (1, img_size, img_size)
        dummy_input = torch.randn(16, *input_size).to(device)

        # Measure peak memory usage (only works on GPU)
        if device.type == "cuda":
            peak_memory = measure_peak_memory(model, dummy_input, device)
            logger.debug(f"Peak Memory Usage: {peak_memory:.2f} MB")
        else:
            logger.debug("Peak memory measurement requires a CUDA-enabled GPU.")

        # Measure average inference time per batch
        avg_inference_time = measure_inference_time(
            model, dummy_input, num_runs=100, device=device
        )
        logger.debug(
            f"Average Inference Time per Batch: {avg_inference_time * 1000:.2f} ms"
        )

        # Measure throughput (samples per second)
        throughput = measure_throughput(model, dummy_input, num_runs=100, device=device)
        logger.debug(f"Throughput: {throughput:.2f} samples/sec")
        # Note: ptflops expects input_size as (channels, height, width)
        macs, params = calculate_flops(model, input_size)
        results_list.append(
            {
                "img_size": img_size,
                "Peak Memory Usage (MB)": peak_memory,
                "Avg Inference Time (ms)": avg_inference_time
                * 1000,  # convert seconds to ms
                "Throughput (samples/s)": throughput,
                "FLOPs (MACs)": macs,
                "Parameters": params,
            }
        )
        logger.debug(f"FLOPs (MACs): {macs}")
        logger.debug(f"Number of Parameters: {params}")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f"{PROJECT_ROOT}/output/{MODEL_NAME}_performance_benchmarks.csv")


if __name__ == "__main__":
    main()
