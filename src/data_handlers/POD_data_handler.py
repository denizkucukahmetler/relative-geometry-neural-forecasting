# src/data_handlers/pod_coeff_data_handler.py
import torch
from torch import tensor
import numpy as np
from scipy.io import loadmat     
import random
from pathlib import Path
from typing import List, Tuple

from src.data_handlers.base_data_handler import BaseDataHandler
from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams


class PODDataHandler(BaseDataHandler):
    """
    Loads pre-computed POD coefficients (shape: n_modes × T) and slices them
    into equal-length trajectories that look just like the outputs of your
    ODE-based data-handlers.
    """

    def __init__(
        self,
        hyperparams: BaseDataHandlerParams,
        experiment_name: str,
        is_eval_data: bool,
    ):
        super().__init__(hyperparams)

        self._traj_dir = Path(f"results/{experiment_name}/trajectories")
        random.seed(hyperparams.get("seed", 42))
        np.random.seed(hyperparams.get("seed", 42))

        data_files = hyperparams.get(
            "data_files",
            [
                "data/PODcoefficients.mat",
                "data/PODcoefficients_run1.mat",
            ],
        )
        
        r = hyperparams.get("r", 2) #convective modes

        coeff_series = np.concatenate(
            [self._load_coeff_file(Path(f), r) for f in data_files],
            axis=1,
        )

        # Store for later use by _build_trajectories
        self._coeff_series = coeff_series  # numpy array, shape (n_modes, T)
        

        # Convenience
        self._T_full = coeff_series.shape[1]
        self.traj_len = hyperparams.get("trajectory_length", 500)
        self.stride = hyperparams.get("stride", self.traj_len)  # window hop

        split_approach = hyperparams.get("split_approach", "trajectory")

        if split_approach == "trajectory":
            n_traj = hyperparams.get("num_trajectories", 100)
            self.train_trajectories = self._build_random_windows(
                n_traj, section="train"
            )
            self.val_trajectories = self._build_random_windows(
                n_traj, section="val"
            )
            self.test_trajectories = self._build_random_windows(
                n_traj, section="test"
            )

        elif split_approach == "time":
            splits = [0.8, 0.1, 0.1]  # train, val, test
            idx1 = int(splits[0] * self._T_full)
            idx2 = int((splits[0] + splits[1]) * self._T_full)

            self.train_trajectories = self._build_sequential_windows(
                0, idx1
            )
            self.val_trajectories = self._build_sequential_windows(
                idx1, idx2
            )
            self.test_trajectories = self._build_sequential_windows(
                idx2, self._T_full
            )
        else:
            raise ValueError(f"Unknown split_approach: {split_approach!r}")

        # Optional: persist generated tensors
        if is_eval_data:
            self.save_trajectories_to_file(
                self._traj_dir / "train", self.train_trajectories
            )
            self.save_trajectories_to_file(
                self._traj_dir / "val", self.val_trajectories
            )
            self.save_trajectories_to_file(
                self._traj_dir / "test", self.test_trajectories
            )

    
    def _load_coeff_file(self, path: Path, r: int) -> np.ndarray:
        """
        Returns an array with shape (n_modes, T) ready to concatenate
        along the *time* axis.
        """
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=True)      # in case you convert later
        elif path.suffix == ".mat":
            mat = loadmat(path)
            # keep the first `r` convective modes and ONE shift mode
            alpha   = mat["alpha"][:, :r]                # shape (T, r)
            alphaS  = mat["alphaS"][:, :1]               # shape (T, 1)
            coeff   = np.concatenate((alpha, alphaS), axis=1)   # (T, r+1)
            return coeff.T                              # (r+1, T) – match .npy orientation
        else:
            raise ValueError(f"Unknown file type: {path}")

        
    def _build_random_windows(
        self, n_traj: int, section: str
    ) -> List[torch.Tensor]:
        """Sample n_traj windows of length traj_len at random start indices."""
        max_start = self._T_full - self.traj_len
        starts = np.random.randint(0, max_start + 1, size=n_traj)

        return [
            self._slice_to_tensor(s, s + self.traj_len) for s in starts
        ]

    def _build_sequential_windows(
        self, start_idx: int, end_idx: int
    ) -> List[torch.Tensor]:
        """Tile the [start_idx, end_idx) segment by sliding a window."""
        out = []
        idx = start_idx
        while idx + self.traj_len <= end_idx:
            out.append(self._slice_to_tensor(idx, idx + self.traj_len))
            idx += self.stride
        return out

    def _slice_to_tensor(self, beg: int, end: int) -> torch.Tensor:
        """
        Convert coeff_series[:, beg:end] → (1, traj_len, n_modes)  **on device**.
        Matches the layout produced by your ODE handlers
        (batch=1, time, dim).
        """
        seg = self._coeff_series[:, beg:end]  # shape (n_modes, traj_len)
        # Reorder to (time, n_modes) and wrap batch dim
        seg = seg.T.astype(np.float32)
        return tensor(seg, dtype=torch.float32).unsqueeze(0).to(self.device)
