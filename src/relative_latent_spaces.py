from __future__ import annotations

"""Relative-latent-space evaluation — minimal clean-up, API intact."""

import json
import random
import time
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from src.encoder.MLPEncoder import MLPEncoder
from src.encoder.RNNEncoder import RNNEncoder
from src.encoder.RNNAutoRegEncoder import RNNAutoRegEncoder
from src.encoder.NoneModelEncoder import NoneModelEncoder
from src.encoder.TransformerEncoder import TransformerEncoder
from src.model.ESN import ESNModel
from src.model.model_interface import BaseModel

try:
    from src.encoder.KoopmanEncoder import KoopmanEncoder  # optional
except ImportError:
    KoopmanEncoder = None  # type: ignore

from src.data_handlers.dynamical_dataset import DynamicalDataset
from src.data_handlers.data_utils import sequence_collate_fn


def _subset(ds: DynamicalDataset, idx):
    return DynamicalDataset(
        ds.enc_inputs[idx],
        ds.dec_inputs[idx],
        ds.dec_targets[idx],
        ds.timesteps[idx],
        [ds.scalers[i] for i in idx],
    )


def _z(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-8)

def _top1_agreement(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A, B: [n_pts, n_anc]
    a1 = A.argmax(dim=1)
    b1 = B.argmax(dim=1)
    return (a1 == b1).float().mean()

def _rowwise_spearman(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Convert each row to ranks (1 = highest score). Break ties by stable sort.
    def ranks(x: torch.Tensor) -> torch.Tensor:
        # Indices of descending order per row
        order = torch.argsort(x, dim=1, descending=True, stable=True)
        r = torch.empty_like(x, dtype=torch.float)

        # 1..n_anc as floats, then repeat for each row to match index dims
        row_ranks = torch.arange(1, x.size(1) + 1, device=x.device, dtype=torch.float)
        row_ranks = row_ranks.unsqueeze(0).expand(x.size(0), -1)  # [n_pts, n_anc]

        # Scatter ranks into the sorted positions
        r.scatter_(1, order, row_ranks)
        return r

    rA, rB = ranks(A), ranks(B)

    # Pearson corr per row between ranks
    rA = rA - rA.mean(dim=1, keepdim=True)
    rB = rB - rB.mean(dim=1, keepdim=True)
    denom = (rA.norm(dim=1) * rB.norm(dim=1)).clamp_min(1e-12)
    row_corr = (rA * rB).sum(dim=1) / denom
    return row_corr.mean()

def _similarity_matrix(rel_lat: List[torch.Tensor], kind: str) -> torch.Tensor:
    """Build an n×n similarity matrix across models for the requested metric."""
    n = len(rel_lat)
    sim = torch.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if kind == "cosine":
                # Row-wise cosine between relative-emb matrices, averaged across points
                v = F.cosine_similarity(rel_lat[i], rel_lat[j], dim=1).mean()
            elif kind == "rank":
                # Row-wise Spearman (via Pearson on ranks), averaged across points
                v = _rowwise_spearman(rel_lat[i], rel_lat[j])
            elif kind == "top1":
                # Top-1 anchor agreement rate
                v = _top1_agreement(rel_lat[i], rel_lat[j])
            else:
                raise ValueError(f"Unknown similarity kind: {kind}")
            sim[i, j] = sim[j, i] = v
    return sim

def _similarity_matrix_windowed(
    rel_lat: List[torch.Tensor],
    kind: str = "cosine",
    win: int = 5,
    stride: int = 1,
) -> torch.Tensor:
    """
    Compute model-vs-model similarity over sliding time windows.
    rel_lat: list of length n_models; each tensor shape [T, D_rel] (rows in temporal order)
    Returns: S of shape [Nwin, n_models, n_models], where S[k] is the similarity
             matrix computed on rows [t0:t0+win] for window k.
    """
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")

    # Ensure common length across models (truncate to min, just like you already do elsewhere)
    #T = min(x.size(0) for x in rel_lat)
   
    T = 300
    rel_lat = [x[:T] for x in rel_lat]
    
    

    if T < win:
        raise ValueError(f"Window size {win} exceeds sequence length {T}")

    n_models = len(rel_lat)
    windows = list(range(0, T - win + 1, stride))
    out = torch.zeros((len(windows), n_models, n_models), dtype=rel_lat[0].dtype)

    for k, t0 in enumerate(windows):
        # Slice an ordered window of rows
        rel_w = [x[t0:t0 + win] for x in rel_lat]
        # Reuse your existing function – its row-wise mean now averages over 'win' time steps
        out[k] = _similarity_matrix(rel_w, kind)

    return out  # [Nwin, n, n]

def save_windowed_long_csv(S_win: torch.Tensor, names: list[str], path: Path):
    rows = []
    Nwin, n, _ = S_win.shape
    for w in range(Nwin):
        for i in range(n):
            for j in range(n):
                rows.append((w, names[i], names[j], float(S_win[w, i, j])))
    pd.DataFrame(rows, columns=["window_index", "model_i", "model_j", "similarity"]).to_csv(
        path, index=False, float_format="%.6f"
    )

def evaluate_models(
    models: Sequence[BaseModel],
    datasets: Sequence[Dataset],
    file_path: str | Path,
    hyperparams: dict[str, Any],
) -> Tuple[List[BaseModel], List[torch.Tensor], List[torch.Tensor]]:
    out_dir = Path(file_path)
    device, n_pts, n_anc = (
        hyperparams["device"],
        hyperparams["num_samples"],
        hyperparams["num_anchors"],
    )
    if n_anc >= n_pts:
        raise ValueError("num_anchors must be < num_samples")
    max_len = min(len(d) for d in datasets)
    if n_pts > max_len:
        raise ValueError("num_samples exceeds dataset length")

    rnd_flag = hyperparams.get("random_anchors", False)
    if rnd_flag:
        seed_anchor = random.SystemRandom().randint(0, 2**32 - 1)
    else:
        seed_anchor = 42
    rng_anc = random.Random(seed_anchor)
        
    seed_idx = 42
    rng_idx = random.Random(seed_idx) #
    idx_sample = rng_idx.sample(range(max_len), n_pts)

    # absolute embeddings
    latent_abs = []
    for m, ds in zip(models, datasets):
        
        enc = m if isinstance(m, ESNModel) else m.encoder
        loader = DataLoader(
            _subset(ds, idx_sample),
            batch_size=1 if isinstance(m, ESNModel) else 32,
            shuffle=False,
            collate_fn=sequence_collate_fn,
        )
        
        enc.eval()
        buf = []
        with torch.no_grad():
            for enc_in, _, dec_tgt, *_ in loader:
                enc_in, dec_tgt = enc_in.to(device), dec_tgt.to(device)
                if isinstance(enc, MLPEncoder):
                    dec_tgt = dec_tgt.reshape(dec_tgt.size(0), -1)
                    out = enc(enc_in)
                elif isinstance(enc, (RNNEncoder, RNNAutoRegEncoder)):
                    out, _ = enc(enc_in)
                elif isinstance(enc, TransformerEncoder):
                    out = enc(enc_in)
                elif isinstance(enc, ESNModel):
                    out = enc.embed(enc_in)
                elif isinstance(enc, NoneModelEncoder):
                    out = enc(enc_in)
                elif KoopmanEncoder and isinstance(enc, KoopmanEncoder):
                    out = enc(enc_in)
                else:
                    raise TypeError(f"Unsupported encoder {type(enc)}")
                if out.ndim == 3:
                    out = out[:, -1, :]
                buf.append(out.cpu())
        latent_abs.append(_z(torch.cat(buf)))

    # relative embeddings
    anchor_idx = rng_anc.sample(range(n_pts), n_anc)
    rel_lat = [
        F.cosine_similarity(lat.unsqueeze(1), lat[anchor_idx].unsqueeze(0), dim=-1)
        for lat in latent_abs
    ]

    
    
    pca_latent_abs = []
    for lat in latent_abs:
        lat_np = lat.detach().cpu().numpy()
        n_components = min(3, lat_np.shape[0], lat_np.shape[1])
        if n_components == 0:
            reduced = torch.zeros(lat.size(0), 3, dtype=lat.dtype, device=lat.device)
        else:
            pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
            reduced_np = pca.fit_transform(lat_np)
            if n_components < 3:
                reduced_np = np.pad(
                    reduced_np,
                    ((0, 0), (0, 3 - n_components)),
                    mode="constant",
                )
            reduced = torch.from_numpy(reduced_np).to(lat.device, dtype=lat.dtype)
        pca_latent_abs.append(_z(reduced))


    names = [m.hyperparams["model_name"] for m in models]
    prefix_mode = str(out_dir).endswith("_")

    if prefix_mode:
        # '.../sim_tmp_' is a PREFIX, not a directory: write ONLY the legacy cosine file.
        prefix = str(out_dir)
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        rel_paths = {"cosine": Path(prefix + "similarities.csv")}
        abs_paths = {"cosine": Path(prefix + "similarities_abs.csv")}
        json_path = Path(prefix + "metadata.json")
        write_location = Path(prefix).parent
        kinds = ["cosine"]
    else:
        # Directory mode: write exactly three files.
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_paths = {
            "cosine": out_dir / "similarities_cosine.csv",
            "rank":   out_dir / "similarities_rank.csv",
            "top1":   out_dir / "similarities_top1.csv",
        }
        abs_paths = (
            {
                "cosine": out_dir / "similarities_abs_cosine.csv",
                "rank":   out_dir / "similarities_abs_rank.csv",
                "top1":   out_dir / "similarities_abs_top1.csv",
            }
        )
        json_path = out_dir / "metadata.json"
        write_location = out_dir
        kinds = ["cosine", "rank", "top1"]
        
   

    for kind in kinds:
        sim_rel = _similarity_matrix(rel_lat, kind)
        pd.DataFrame(sim_rel.numpy(), index=names, columns=names).to_csv(
            rel_paths[kind], float_format="%.4f"
        )

        abs_target = abs_paths.get(kind)
        if abs_target is not None:
            sim_abs = _similarity_matrix(pca_latent_abs, kind)
            pd.DataFrame(sim_abs.detach().numpy(), index=names, columns=names).to_csv(
                abs_target, float_format="%.4f"
            )

    # metadata
    with open(json_path, "w", encoding="utf-8") as fh:
        metadata = {
            "seed_used": seed_anchor,
            "num_anchors": n_anc,
            "num_points": n_pts,
            "device": device,
            "similarity_files": {k: str(v.name) for k, v in rel_paths.items()},
        }
        if abs_paths:
            metadata["similarity_files_absolute"] = {
                k: str(v.name) for k, v in abs_paths.items()
            }
        json.dump(metadata, fh, indent=4)

    print("Similarity matrices + metadata written to", write_location)
    return list(models), rel_lat, latent_abs

def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)

def save_rel_lat_time_csv(
    rel_lat: List[torch.Tensor],
    names: List[str],
    anchor_idx: List[int],
    temporal_start: int,
    path_long: Path,
    dir_per_model: Path,
):
    """
    rel_lat[i] shape: [T, n_anc] (row=time, col=anchor) for model i
    Saves:
      - ONE long-form CSV with columns: time_index, anchor_index, model, value
      - ONE wide CSV per model: columns: time_index, anc_<global_anchor_id>...
    """
    rows = []
    for i, (name, mat) in enumerate(zip(names, rel_lat)):
        mat_np = mat.detach().cpu().numpy()  # [T, n_anc]
        T, A = mat_np.shape

        # Per-model wide CSV
        df_wide = pd.DataFrame(mat_np, columns=[f"anc_{a}" for a in anchor_idx])
        df_wide.insert(0, "time_index", np.arange(T, dtype=int) + temporal_start)
        fname = f"rel_lat_{_sanitize_filename(name)}.csv"
        df_wide.to_csv(dir_per_model / fname, index=False, float_format="%.6f")

        # Long-form rows
        for t in range(T):
            t_abs = t + temporal_start
            for a, anch in enumerate(anchor_idx):
                rows.append((t_abs, int(anch), name, float(mat_np[t, a])))

    # Single long-form CSV
    pd.DataFrame(
        rows, columns=["time_index", "anchor_index", "model", "value"]
    ).to_csv(path_long, index=False, float_format="%.6f")

    
def evaluate_models_time(
    models: Sequence[BaseModel],
    datasets: Sequence[Dataset],
    file_path: str | Path,
    hyperparams: dict[str, Any],
) -> Tuple[List[BaseModel], List[torch.Tensor], List[torch.Tensor]]:
    """
    Time-resolved evaluation that ONLY saves sliding-window cosine similarities
    in the RELATIVE latent space.

    Hyperparams used:
        device            : str (e.g., "cuda")
        num_samples       : int
        num_anchors       : int
        random_anchors    : bool (default False)
        temporal_window   : int > 0 (e.g., 5)
        temporal_stride   : int > 0 (default 1)
        temporal_start    : int >= 0 (default 0)

    Outputs:
        - similarities_temporal_cosine.csv  (long-form: window_index, model_i, model_j, similarity)
        - metadata.json
    """
    t0 = time.time()
    out_dir = Path(file_path)
    device = hyperparams["device"]
    n_pts  = int(hyperparams["num_samples"])
    n_anc  = int(hyperparams["num_anchors"])

    temporal_win    = int(hyperparams.get("temporal_window", 5))
    temporal_stride = int(hyperparams.get("temporal_stride", 1))
    temporal_start  = int(hyperparams.get("temporal_start", 0))

    if temporal_win <= 0 or temporal_stride <= 0:
        raise ValueError("temporal_window and temporal_stride must be positive.")
    if n_anc >= n_pts:
        raise ValueError("num_anchors must be < num_samples")

    max_len = min(len(d) for d in datasets)
    if n_pts > max_len:
        raise ValueError("num_samples exceeds dataset length")
    if temporal_start + n_pts > max_len:
        raise ValueError(
            f"temporal_start ({temporal_start}) + num_samples ({n_pts}) exceeds dataset length ({max_len})"
        )
    if temporal_win > n_pts:
        raise ValueError(f"temporal_window ({temporal_win}) exceeds num_samples ({n_pts})")

    rnd_flag = hyperparams.get("random_anchors", False)
    seed_anchor = random.SystemRandom().randint(0, 2**32 - 1) if rnd_flag else 42
    rng_anc = random.Random(seed_anchor)

    idx_sample = list(range(temporal_start, temporal_start + n_pts))

    latent_abs: List[torch.Tensor] = []
    for m, ds in zip(models, datasets):
        enc = m if isinstance(m, ESNModel) else m.encoder
        loader = DataLoader(
            _subset(ds, idx_sample),
            batch_size=1 if isinstance(m, ESNModel) else 32,
            shuffle=False,
            collate_fn=sequence_collate_fn,
        )

        enc.eval()
        buf = []
        with torch.no_grad():
            for enc_in, _, dec_tgt, *_ in loader:
                enc_in, dec_tgt = enc_in.to(device), dec_tgt.to(device)
                if isinstance(enc, MLPEncoder):
                    dec_tgt = dec_tgt.reshape(dec_tgt.size(0), -1)
                    out = enc(enc_in)
                elif isinstance(enc, (RNNEncoder, RNNAutoRegEncoder)):
                    out, _ = enc(enc_in)
                elif isinstance(enc, TransformerEncoder):
                    out = enc(enc_in)
                elif isinstance(enc, ESNModel):
                    out = enc.embed(enc_in)
                elif isinstance(enc, NoneModelEncoder):
                    out = enc(enc_in)
                elif KoopmanEncoder and isinstance(enc, KoopmanEncoder):
                    out = enc(enc_in)
                else:
                    raise TypeError(f"Unsupported encoder {type(enc)}")
                if out.ndim == 3:
                    out = out[:, -1, :]
                buf.append(out.cpu())
        latent_abs.append(_z(torch.cat(buf)))

    anchor_idx = rng_anc.sample(range(n_pts), n_anc)
    rel_lat: List[torch.Tensor] = [
        F.cosine_similarity(lat.unsqueeze(1), lat[anchor_idx].unsqueeze(0), dim=-1)
        for lat in latent_abs
    ]

    names = [m.hyperparams["model_name"] for m in models]
    prefix_mode = str(out_dir).endswith("_")
    if prefix_mode:
        prefix = str(out_dir)
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        temporal_path = Path(prefix + "similarities_temporal_cosine.csv")
        rel_long_path = Path(prefix + "relative_embeddings_time.csv")                 
        per_model_dir = Path(prefix).parent                                           
       
        json_path = Path(prefix + "metadata.json")
        write_location = Path(prefix).parent
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        temporal_path = out_dir / "similarities_temporal_cosine.csv"
        rel_long_path = out_dir / "relative_embeddings_time.csv"                      
        per_model_dir = out_dir                                                       
        
        json_path = out_dir / "metadata.json"
        write_location = out_dir

    S_win_cos = _similarity_matrix_windowed(rel_lat, kind="cosine", win=temporal_win, stride=temporal_stride)
    save_windowed_long_csv(S_win_cos, names, temporal_path)
    
    save_rel_lat_time_csv(
        rel_lat=rel_lat,
        names=names,
        anchor_idx=anchor_idx,
        temporal_start=temporal_start,
        path_long=rel_long_path,
        dir_per_model=per_model_dir,
    )

    print("Temporal cosine similarities  written to", write_location)
    return list(models), rel_lat, latent_abs
