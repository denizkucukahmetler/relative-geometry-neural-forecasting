import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import seaborn as sns
import os
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster


def cosine_heatmap(dir_path:str) -> None:
    """Takes the cosine similarity matrix of the experiment and computes the average cosine similarity of each models seed to
    all the other models seeds."""

    file_path = f"{dir_path}similarities_cosine.csv"
    df = pd.read_csv(file_path)

    df = df.set_index("Unnamed: 0")
    def get_family(name: str) -> str:
        return ''.join([c for c in name if not c.isdigit()])

    # Map models to families and seeds
    families = {}
    for model in df.index:
        family = get_family(model)
        seed = int(''.join([c for c in model if c.isdigit()])) if any(c.isdigit() for c in model) else None
        if family not in families:
            families[family] = []
        families[family].append(model)

    # Ensure consistent ordering by seed number
    for f in families:
        families[f] = sorted(families[f], key=lambda x: int(''.join([c for c in x if c.isdigit()])))

    # Compute family-to-family averages
    family_names = list(families.keys())
    matrix = pd.DataFrame(index=family_names, columns=family_names, dtype=float)

    for famA in family_names:
        for famB in family_names:
            sims = []
            for modelA in families[famA]:
                row = df.loc[modelA, families[famB]].copy()
                # Exclude self-similarity if same family and same model
                if famA == famB:
                    row = row.drop(modelA)
                sims.append(row.mean())
            matrix.loc[famA, famB] = np.mean(sims)

    # Drop "none" family
    try:
        matrix_clean = matrix.drop(index="none", columns="none")
    except KeyError:
        matrix_clean = matrix
    # Round for readability
    matrix_clean_rounded = matrix_clean.round(3)
    matrix_clean_rounded

    # Define custom order
    custom_order = [
        "mlp", "nodemlp", "koopmanmlp",  
        "transformer", "nodetransformer", "koopmantransformer"
    ]

    # Reorder rows and columns
    matrix_ordered = matrix_clean.loc[custom_order, custom_order].round(3)

    # Plot heatmap
    colors = ["#2166ac", "#f7f7f7", "#b2182b"]  # dark blue, white, dark red # Custom diverging colormap: blue -> white -> red
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)     # Fix range from 0.0 to 1.0

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_ordered, annot=True, fmt=".2f", cmap="coolwarm", norm=norm, cbar=True,
                xticklabels=matrix_ordered.columns, yticklabels=matrix_ordered.index, vmin=0, vmax=1)

    #plt.title("Average Cosine Similarity Heatmap (Custom Order)", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(dir_path, "cosine_similarity_heatmap.png")
    print(f"Saved cosine similarity heatmap in {save_path}.")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # dpi=300 gives high quality

def eval_stitching_loss(dir_path:str, encoder_name:str) -> pd.DataFrame:
    """This method takes an encoder name and computes the average sitching loss (mse loss)
    for the encoder to each decoder family."""
    file_path = f"{dir_path}{encoder_name}_stitching_results.csv"
    df = pd.read_csv(file_path)

    # Drop "none" decoders
    df = df[df["decoder"] != "none"].copy()

    # Families = non-numeric prefix
    df["decoder_family"] = df["decoder"].str.replace(r"\d+$", "", regex=True)
    encoder_family = re.sub(r"\d+$", "", encoder_name)

    # Exclude only the exact self-pair when averaging within the same family
    mask_self_pair = (df["decoder"] == encoder_name) & (df["decoder_family"] == encoder_family)
    df_filtered = df.loc[~mask_self_pair].copy()

    # Average per decoder family
    # Compute mean and std per decoder family
    stats = (
        df_filtered.groupby("decoder_family")["mse_loss"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("decoder_family")
    )
    stats["mean"] = stats["mean"].round(3)
    stats["std"] = stats["std"].round(3)

    # Round float precision
    avg_df = stats[["decoder_family", "mean"]].rename(columns={"mean": "mse_loss"})
    std_df = stats[["decoder_family", "std"]].rename(columns={"std": "mse_loss_std"})
    return avg_df, std_df

def eval_all_models_stitching(
    dir_path: str, 
    model_names: list, 
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds encoder-decoder pair stitching loss matrices (mean & std)."""

    custom_order = [
        "mlp", "nodemlp", "koopmanmlp", 
        "oldrnn", "rnnautoreg", "nodernnautoreg", "koopmanrnnautoreg", 
        "transformer", "nodetransformer", "koopmantransformer"
    ]

    all_means, all_stds = [], []
    for encoder_name in model_names:
        avg_df, std_df = eval_stitching_loss(dir_path=dir_path, encoder_name=encoder_name)
        encoder_family = re.sub(r"\d+$", "", encoder_name)

        avg_df["encoder_family"] = encoder_family
        std_df["encoder_family"] = encoder_family

        all_means.append(avg_df)
        all_stds.append(std_df)
    
    merged_means = pd.concat(all_means, ignore_index=True)
    merged_stds = pd.concat(all_stds, ignore_index=True)

    matrix_mean = merged_means.pivot(index="encoder_family", columns="decoder_family", values="mse_loss")
    matrix_std  = merged_stds.pivot(index="encoder_family", columns="decoder_family", values="mse_loss_std")

    # Enforce order
    matrix_mean = matrix_mean.reindex(index=custom_order, columns=custom_order)
    matrix_std  = matrix_std.reindex(index=custom_order, columns=custom_order)

    # Save results
    matrix_mean.to_csv(f"{dir_path}stitching_matrix_mean.csv")
    matrix_std.to_csv(f"{dir_path}stitching_matrix_std.csv")

    print(f"Saved mean stitching loss matrix in {dir_path}stitching_matrix_mean.csv")
    print(f"Saved std stitching loss matrix in {dir_path}stitching_matrix_std.csv")

    return matrix_mean, matrix_std



def plot_grouped_similarity(
    csv_path: str | Path,
    *,
    group_name_fn: Callable[[str], str] = lambda s: re.sub(r"\d+$", "", s),
    label_map: Optional[Dict[str, str]] = None,   # pretty labels for grouped names (seed or base ok)
    linkage_method: str = "average",
    threshold: float = 0.1,
    annotate: bool = True,
    show_dendrogram: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, int], np.ndarray]:
    """
    Read a similarities.csv whose rows/cols are seed-suffixed model names (e.g., mlp1..mlp10),
    average over seeds by grouping names via `group_name_fn`, and plot a clustered heatmap.
    Parameters
    ----------
    csv_path : str | Path
        Path to the similarity matrix CSV (square, symmetric).
    group_name_fn : callable
        Function mapping raw names -> group name (default strips trailing digits).
    label_map : dict[str, str] | None
        Mapping base or seed names -> pretty label.
    linkage_method : str
        Linkage method for hierarchical clustering.
    threshold : float
        Distance threshold for flat clustering.
    annotate : bool
        Whether to annotate values in the heatmap.
    show_dendrogram : bool
        Whether to plot a dendrogram before the heatmap.
    Returns
    -------
    grouped_sim : pd.DataFrame
        Similarity matrix averaged over seeds (group × group).
    clusters : dict[str, int]
        Mapping base_model -> flat cluster ID.
    Z : np.ndarray
        Linkage matrix from hierarchical clustering.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col=0).apply(pd.to_numeric, errors="coerce")
    if df.isnull().values.any():
        raise ValueError(f"Non-numeric entries detected in {csv_path}")
    if list(df.index) != list(df.columns):
        raise ValueError("Index/columns mismatch in similarity matrix.")
    # make strictly symmetric & clamp
    df = ((df + df.T) / 2).clip(lower=0.0, upper=1.0)
    groups: Dict[str, list[str]] = {}
    for name in df.index:
        g = group_name_fn(name)
        groups.setdefault(g, []).append(name)
    base_models = sorted(groups.keys())
    grouped_sim = pd.DataFrame(index=base_models, columns=base_models, dtype=float)
    for gi in base_models:
        rows = groups[gi]
        for gj in base_models:
            cols = groups[gj]
            sub = df.loc[rows, cols].to_numpy()
            grouped_sim.loc[gi, gj] = float(np.nanmean(sub))
    # enforce symmetry & clean diagonal
    grouped_sim = ((grouped_sim + grouped_sim.T) / 2).astype(float)
    np.fill_diagonal(grouped_sim.values, 1.0)
    dist = sch.distance.squareform(1.0 - grouped_sim.to_numpy())
    Z = sch.linkage(dist, method=linkage_method)
    flat = fcluster(Z, t=threshold, criterion="distance")
    clusters = dict(zip(grouped_sim.index, flat))
    # normalize label_map to base names, accepting seed-suffixed keys too
    normalized_label_map: Dict[str, str] = {}
    if label_map:
        for k, v in label_map.items():
            normalized_label_map.setdefault(group_name_fn(k), v)
    if show_dendrogram:
        plt.figure(figsize=(max(10, 0.35 * len(base_models)), 4.2))
        dendro = sch.dendrogram(Z, labels=grouped_sim.index.tolist(), leaf_rotation=45)
        plt.title(f"Dendrogram (threshold={threshold})")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()
        plt.close()
        leaves = dendro["leaves"]
    else:
        leaves = sch.leaves_list(Z)
    ordered_ids = [grouped_sim.index[i] for i in leaves]
    pretty = [normalized_label_map.get(k, k) for k in ordered_ids]
    ordered = grouped_sim.loc[ordered_ids, ordered_ids]
    # proper figsize as (width, height)
    n = len(ordered_ids)
    fig_w = min(18, 2 + 0.6 * n)
    fig_h = min(18, 2 + 0.6 * n)
    plt.figure(figsize=(fig_w, fig_h), dpi=120)
    ax = sns.heatmap(
        ordered, annot=annotate, fmt=".2f", cmap="coolwarm",
        vmin=0, vmax=1, linewidths=0.4, cbar_kws={"label": "Average similarity"}
    )
    ax.set_xticks(np.arange(len(pretty)) + 0.5)
    ax.set_xticklabels(pretty, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pretty)) + 0.5)
    ax.set_yticklabels(pretty, rotation=0)
    plt.title("Average Similarity (seeds collapsed by base model)")
    plt.tight_layout()
    plt.savefig("/home/max/Coding/Python/MPI_Research/git/REU-ML-Summer-Project-24/results/abs_all_20epochs/new_heatmap.png")
    plt.close()
    return grouped_sim, clusters, Z


def plot_std_heatmap(file_path:str, save_path:str):
    # Column names (encoder families)
    columns = ['mlp', 'nodemlp', 'koopmanmlp', 'oldrnn', 'rnnautoreg', 
            'nodernnautoreg', 'koopmanrnnautoreg', 'transformer', 
            'nodetransformer', 'koopmantransformer']

    # Create DataFrame
    df = pd.read_csv(file_path)
    df.columns = ['model'] + columns
    df = df.set_index('model')

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, 
                annot=True,  # Show values in cells
                cmap='coolwarm',  # Color scheme
                center=0.2,  # Center the colormap around this value
                fmt='.3f',  # Format numbers to 3 decimal places
                linewidths=0.5,  # Add lines between cells
                #cbar_kws={'label': 'Standard Deviation'}
                )

    #plt.title('Standard Deviation Heatmap: Model vs Encoder Family', fontsize=14, pad=20)
    plt.xlabel('Decoder', fontsize=12)
    plt.ylabel('Encoder', fontsize=12)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.savefig(save_path)
