import torch
import numpy as np
import random
from torch import tensor
from pathlib import Path
import re
import numpy as np

from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams
from src.data_handlers.base_data_handler import BaseDataHandler

class LogisticMapDataHandler(BaseDataHandler):
    '''
    Extends BaseDataHandler.
    A DataHandler for generating Logistic Map trajectories.
    '''

    def __init__(self, hyperparams: BaseDataHandlerParams, experiment_name, is_eval_data):
        '''
        Constructor. Will generate the trajectories and store them inside the class.
        Args:
            hyperparams: dict[str, any] contains hyperparameters for data generation. Should contain:
                - num_trajectories: Number of trajectories to generate (default 100)
                - trajectory_length: Length of each trajectory (default 1000)
                - mus: List of mu parameters for the logistic map
                - device: The device where the Tensors should be stored (default cpu)
        '''
        super().__init__(hyperparams)
        
        self.mu  = float(hyperparams.get("mu",3.57))            # ← fixed control parameter
        self.noise_pct = float(hyperparams.get("noise_pct", 0.0))  # measurement noise % of std


        self._traj_dir = Path(f"results/{experiment_name}/trajectories")

        random.seed(hyperparams.get("seed", 42))
        np.random.seed(hyperparams.get("seed", 42))

        approach  = hyperparams.get("split_approach", "trajectory")
        n_traj    = self.num_trajectories
        rand_ic   = lambda n: [random.uniform(0.1, 0.9) for _ in range(n)]

        
        
        if approach == 'trajectory':
            train_ic = hyperparams.get("initial_conditions_train") or rand_ic(n_traj)
            val_ic   = hyperparams.get("initial_conditions_val")   or rand_ic(n_traj)
            test_ic  = hyperparams.get("initial_conditions_test")  or rand_ic(n_traj)
            
            self.train_trajectories = self._build_trajectories(train_ic)
            self.val_trajectories   = self._build_trajectories(val_ic)
            self.test_trajectories  = self._build_trajectories(test_ic)


        elif approach == "time":
            all_ic   = hyperparams.get("initial_conditions_all") or rand_ic(n_traj)
            raw_traj = self._build_trajectories(all_ic)
            t, v, te = self.time_split(raw_traj)
            self.train_trajectories, self.val_trajectories, self.test_trajectories = t, v, te
        else:
            raise ValueError(f"Unknown split_approach: {approach!r}")

        
        if is_eval_data:
            self.save_trajectories_to_file(self._traj_dir / "train", self.train_trajectories)
            self.save_trajectories_to_file(self._traj_dir / "val",   self.val_trajectories)
            self.save_trajectories_to_file(self._traj_dir / "test",  self.test_trajectories)
            
    def _build_trajectories(self, x0_list: list[float]) -> list[torch.Tensor]:
        """
        Parameters
        x0_list : list of float
            Initial x-values for each trajectory.

        Returns
        list[Tensor]
            Each tensor is (1, T, 2) with channels [x, μ].
        """
        T   = len(self.t_eval)                    # inherited from BaseDataHandler
        out = []

        for x0 in x0_list:
            traj = np.empty((1, T), dtype=np.float32)
            traj[0, 0] = x0

            for k in range(1, T):
                x_next  = self.mu * traj[0, k-1] * (1.0 - traj[0, k-1])
                traj[0, k] = np.clip(x_next, 0.0, 1.0)  # still safe with noise

            tensor_traj = (
                torch.from_numpy(traj)
                     .float()
                     .to(self.device)
                     .T                 # (T, 1)
                     .unsqueeze(0)      # (1, T, 1)
            )
            
            if self.noise_pct > 0.0:
                # per-trajectory std across time, keep dims for broadcasting
                scale = tensor_traj.std(dim=1, keepdim=True)       # (1, 1, 1)
                noise = torch.randn_like(tensor_traj) * (self.noise_pct * scale)
                tensor_traj = tensor_traj + noise
                # keep logistic map values bounded
                tensor_traj = torch.clamp(tensor_traj, 0.0, 1.0)
            out.append(tensor_traj)

        return out

    def _load_csv_traj(self, path: Path) -> torch.Tensor:
        """
        Load a trajectory saved by `save_trajectories_to_file`.
        Resulting tensor shape: (1, T, F) where
            F = 1 for the logistic map (x only)
            F = 3 for Lorenz, 4 for double pendulum, …
        """
        data = np.loadtxt(path, delimiter=",", skiprows=1)   # (T, F)

        # handle the corner-case where np.loadtxt returns a 1-D array if F == 1
        if data.ndim == 1:
            data = data[:, None]                             # make it (T, 1)

        return (
            torch.from_numpy(data)
                 .float()
                 .unsqueeze(0)                               # (1, T, F)
                 .to(self.device)
        )

            
            
            
            
            
        