import numpy as np
import torch
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os
import json


def benchmark(ground_truth:torch.Tensor, prediction:torch.Tensor, file_path:str) -> dict():
    """
    Computes all benchmark metrics we use to evaluate our model's performance.
    idea: maybe just input the model and then generate ground_truth and prediction within this file. #todo

    Parameters:
        ground_truth (torch.tensor): Ground truth data of shape (m, n), where n is the number of dimensions.
        prediction (torch.tensor): Predicted data of shape (m, n), where n is the number of dimensions.
        file_path (String): File path to which to save the summary & pdf plots.

        Returns:
        dict: Concats all dicts from each metric to one dict which has all the computed values.

    Example usage:
        benchmark(ground_truth,predicted, plot=False)
    """
    print(len(ground_truth))
    print(len(prediction))
    
    d1 = pdf(ground_truth, prediction, file_path)
    d2 = mse(ground_truth,prediction)
    d3 = rmse(ground_truth,prediction)
    d4 = mae(ground_truth, prediction)
    
    d5 = dict(d1)
    d5.update(d2)
    d5.update(d3)
    d5.update(d4)
    save_benchmarks(d5, file_path)
    return d5

def pdf(ground_truth:torch.Tensor, prediction:torch.Tensor, file_path:str) -> dict():
    """
    Compares probability density functions (PDFs) for the ground truth and predictions for each dimension.
    
    Parameters:
        ground_truth (torch.tensor): Ground truth data of shape (m, n), where n is the number of dimensions.
        prediction (torch.tensor): Predicted data of shape (m, n), where n is the number of dimensions.
        file_path (String): File path to which to save the pdf plots.
        
    Returns:
        dict: KL divergence for each dimension.
    """

    if prediction.shape != ground_truth.shape:
        raise ValueError("Predictions and ground truth must have the same shape.")

    _, dimension = ground_truth.shape
    dimensions = [i for i in range(1, dimension + 1)]
    
    ground_truth = ground_truth.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    
    kl_divergences = {}
    sum_kl = 0
    c = 0

    # Create a figure for subplots
    fig, axes = plt.subplots(nrows=dimension, ncols=1, figsize=(8, 5 * dimension))
    if dimension == 1:  # Handle a single subplot case
        axes = [axes]
    
    for i, dim in enumerate(dimensions):
        # KDE for PDFs
        kde_truth = gaussian_kde(ground_truth[:, i])
        kde_model = gaussian_kde(prediction[:, i])
        
        # Generate x values for the PDF
        x_min = min(ground_truth[:, i].min(), prediction[:, i].min())
        x_max = max(ground_truth[:, i].max(), prediction[:, i].max())
        x_vals = np.linspace(x_min, x_max, 500)
        
        # Calculate KL Divergence
        kl_div = entropy(kde_truth(x_vals), kde_model(x_vals))
        kl_divergences['pdf_dim'+str(dim)+' kl_divergence'] = float(kl_div)
        sum_kl += kl_div
        c += 1
        
        # Plotting on the respective subplot
        ax = axes[i]
        ax.plot(x_vals, kde_truth(x_vals), label=f"Ground Truth PDF ({dim})")
        ax.plot(x_vals, kde_model(x_vals), label=f"Model PDF ({dim})", linestyle='--')
        ax.legend()
        ax.set_title(f"Comparison of PDFs for Dimension {dim}")
        ax.set_xlabel(f"Dimension {dim}")
        ax.set_ylabel("Density")

    plt.tight_layout()
    # Save the entire figure
    save_path = os.path.join(file_path, "pdf_comparison_all_dimensions.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    kl_divergences['pdf_kl_divergence_avg'] = float(sum_kl/c)
    return kl_divergences

def mse(ground_truth:torch.Tensor, prediction:torch.Tensor) -> dict():
    """
    Calculates the Mean Squared Error (MSE) using Scikit-learn.
    
    Parameters:
        ground_truth (torch.tensor): Ground truth data of shape (m, n), where n is the number of dimensions.
        prediction (torch.tensor): Predicted data of shape (m, n), where n is the number of dimensions.
    Returns:
        dict: The key is the name of the metric. The value of the dict is the value computed corresponding to the metric.
    """
    return {'mse': float(mean_squared_error(ground_truth, prediction))}

def rmse(ground_truth:torch.Tensor, prediction:torch.Tensor) -> dict():
    """
    Calculates the Root Mean Squared Error (RMSE) using Scikit-learn.
    
    Parameters:
        ground_truth (torch.tensor): Ground truth data of shape (m, n), where n is the number of dimensions.
        prediction (torch.tensor): Predicted data of shape (m, n), where n is the number of dimensions.
    Returns:
        dict: The key is the name of the metric. The value of the dict is the value computed corresponding to the metric.
    """
    mse = float(mean_squared_error(ground_truth, prediction))
    return {'rmse': math.sqrt(mse)}

def mae(ground_truth:torch.Tensor, prediction:torch.Tensor) -> dict():
    """
    Calculates the Mean Absolute Error (MAE) using Scikit-learn.
    
    Parameters:
        ground_truth (torch.tensor): Ground truth data of shape (m, n), where n is the number of dimensions.
        prediction (torch.tensor): Predicted data of shape (m, n), where n is the number of dimensions.
    Returns:
        dict: The key is the name of the metric. The value of the dict is the value computed corresponding to the metric.
    """
    return {'mae': float(mean_absolute_error(ground_truth,prediction))}

def save_benchmarks(benchmarks:dict(), file_path: str) -> None:
    """
    Saves the benchmark metrics dictionary to a file in JSON format.

    Parameters:
        benchmarks (dict): Dictionary containing all computed benchmark metrics.
        file_path (str): Path to the file where the benchmarks will be saved.
    
    Returns:
        None
    """
    file_path = file_path + '/benchmarks'

    # Save the dictionary to a JSON file
    with open(file_path, 'w') as file:
        json.dump(benchmarks, file, indent=4)

    print(f"Benchmarks successfully saved to {file_path}")




def save_all_benchmarks(benchmarks:dict(), file_path: str) -> None:
    """
    Saves the benchmark metrics dictionary to a file in JSON format.

    Parameters:
        benchmarks (dict): Dictionary containing all computed benchmark metrics.
        file_path (str): Path to the file where the benchmarks will be saved.
    
    Returns:
        None
    """
    print(benchmarks)
    file_path = file_path + '/benchmarks'

    # Save the dictionary to a JSON file
    with open(file_path, 'w') as file:
        json.dump(benchmarks, file, indent=4)

    print(f"ALL benchmarks successfully saved to {file_path}")




    
