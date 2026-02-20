import os
import time
import click
import torch
from ptflops import get_model_complexity_info
import pandas as pd
from pathlib import Path

from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.pneumonia.models.cnn import CNNBinaryClassifier
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="info")
CHECKPOINT_DIR = cfg.pneumonia.model.checkpoint_dir


def measure_peak_memory(
    model: CNNBinaryClassifier, dummy_input: torch.Tensor, device: torch.device
) -> float:
    """
    Measures peak GPU memory usage (in MB) during a forward pass.
    Parameters:
        model (CNNBinaryClassifier): The model to evaluate.
        dummy_input (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        device (torch.device): Device to run the model on.
    Returns:
        float: Peak memory usage in MB.
    """
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    # Get peak memory allocated in bytes and convert to MB
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    return peak_memory


def measure_inference_time(
    model: CNNBinaryClassifier,
    dummy_input: torch.Tensor,
    num_runs: int = 100,
    device: torch.device = torch.device("cuda"),
) -> float:
    """
    Measures average inference time per forward pass (per batch).
    Parameters:
        model (CNNBinaryClassifier): The model to evaluate.
        dummy_input (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        num_runs (int): Number of runs to average over.
        device (torch.device): Device to run the model on.
    Returns:
        float: Average inference time per batch in seconds.
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
    model: CNNBinaryClassifier,
    dummy_input: torch.Tensor,
    num_runs: int = 100,
    device: torch.device = torch.device("cuda"),
) -> float:
    """
    Computes throughput in samples per second.
    Parameters:
        model (CNNBinaryClassifier): The model to evaluate.
        dummy_input (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        num_runs (int): Number of runs to average over.
        device (torch.device): Device to run the model on.
    Returns:
        float: Throughput in samples per second.
    """
    avg_time = measure_inference_time(model, dummy_input, num_runs, device)
    batch_size = dummy_input.size(0)
    throughput = batch_size / avg_time
    return throughput


def calculate_flops(model: CNNBinaryClassifier, input_size: tuple) -> tuple:
    """
    Uses ptflops to calculate FLOPs (as MACs) and the number of parameters.
    Parameters:
        model (CNNBinaryClassifier): The model to evaluate.
        input_size (tuple): Input size as (channels, height, width).
    Returns:
        tuple: (FLOPs in MACs as string, number of parameters as string)
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
    IMG_SIZES = [64, 128, 224, 512, 768, 1024]
    MODEL_NAME = model_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results_list = []
    for img_size in IMG_SIZES:
        logger.info(f"-----------Img Size: {img_size}-----------")
        CKPT_PATH = f"{CHECKPOINT_DIR}/{MODEL_NAME}/{MODEL_NAME}_{img_size}_logs/lightning_logs/version_0/checkpoints/"  # Pick the first fold model for the sake of benchmarking
        CKPT_PATH = os.path.join(CKPT_PATH, os.listdir(CKPT_PATH)[0])
        model = CNNBinaryClassifier.load_from_checkpoint(CKPT_PATH, map_location=device)
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
    results_df.to_csv(f"{HOME}/output/{MODEL_NAME}_performance_benchmarks.csv")


if __name__ == "__main__":
    main()
