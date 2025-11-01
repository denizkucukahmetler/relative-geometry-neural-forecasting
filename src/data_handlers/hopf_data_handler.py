import torch
import random
import numpy as np
from torch import tensor
from pathlib import Path
import re
import numpy as np

from scipy.integrate import solve_ivp

from src.data_handlers.base_data_handler import BaseDataHandler
from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams


class HopfDataHandler(BaseDataHandler):
    '''
    Extends BaseDataHandler.
    A DataHandler for generating Hopf Normal Form data.
    Based on: https://pysindy.readthedocs.io/en/stable/examples/3_original_paper/example.html
    '''

    def __init__(self, hyperparams: BaseDataHandlerParams, experiment_name, is_eval_data):
        '''
        Constructor. Will generate the trajectories and store them inside the class.
        Args:
            hyperparams: dict[str, any] contains hyperparameters for data generation. Should contain:
                - num_trajectories: Number of trajectories to generate (default 10)
                - trajectory_length: Length of each trajectory (default 1000)
                - dt: Timestep between points (default 0.01)
                - rand_start_bounds: Trajectories start points will be random in all dimensions
                  between -rand_start_bounds and +rand_start_bounds. Default 1.
                - device: The device where the Tensors should be stored (default cpu)
                - mu: Hopf parameter mu (default 0.1)
                - omega: Hopf parameter omega (default 1.0)
                - A: Hopf parameter A (default 0.0) # Set to -1.0 for limit cycle example in PySINDy
        '''
        super().__init__(hyperparams)

        
        # Store Hopf specific parameters
        self.mu = hyperparams.get('mu', 0.1)
        self.omega = hyperparams.get('omega', 1.0)
        self.A = hyperparams.get('A', 0.0) # Set to -1.0 for limit cycle
        self.noise_pct  = float(hyperparams.get('noise_pct', 0.0))    # measurement noise (post)


        
        # Ensure input dim matches 2
        if hyperparams.get('in_dim') != 2:
            print("Warning: overriding in_dim to 2 for Hopf DataHandler.")
            hyperparams['in_dim'] = 2

            
            
        # Trajectory directory
        self._traj_dir = Path(f'results/{experiment_name}/trajectories')

        # Seeds
        seed = hyperparams.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        
        
        # Initial conditions splits
        train_ic = hyperparams.get('initial_conditions_train')
        val_ic = hyperparams.get('initial_conditions_val')
        test_ic = hyperparams.get('initial_conditions_test')

        approach = hyperparams.get('split_approach', 'trajectory')
        
        
        n = self.num_trajectories
        bounds = self.rand_start_bounds
        rand_ic = lambda n: [(
            random.uniform(-bounds, bounds),
            random.uniform(-bounds, bounds)
        ) for _ in range(n)]

        if approach == 'trajectory':
            train_ic = train_ic or rand_ic(n)
            val_ic = val_ic or rand_ic(n)
            test_ic = test_ic or rand_ic(n)

            self.train_trajectories = self._build_trajectories(train_ic)
            self.val_trajectories = self._build_trajectories(val_ic)
            self.test_trajectories = self._build_trajectories(test_ic)

        elif approach == 'time':
            all_ic = hyperparams.get('initial_conditions_all') or rand_ic(n)
            raw = self._build_trajectories(all_ic)
            t, v, te = self.time_split(raw)
            self.train_trajectories, self.val_trajectories, self.test_trajectories = t, v, te

        else:
            raise ValueError(f"Unknown split_approach: {approach!r}")

        # Optionally save trajectories
        if is_eval_data:
            self.save_trajectories_to_file(self._traj_dir / 'train', self.train_trajectories)
            self.save_trajectories_to_file(self._traj_dir / 'val', self.val_trajectories)
            self.save_trajectories_to_file(self._traj_dir / 'test', self.test_trajectories)

    def _build_trajectories(self, initials: list[tuple[float, float]]) -> list[torch.Tensor]:
        """
        Build trajectories given initial (x1, x2) states.
        Returns list of tensors shaped (1, T, 2).
        """
        out = []
        for x1_0, x2_0 in initials:
            sol = solve_ivp(
                fun=self.__hopf_ode,
                t_span=self.t_span,
                y0=[x1_0, x2_0],
                t_eval=self.t_eval,
                method='RK45'
            )
            # sol.y: shape (2, T)
            traj = tensor(sol.y, dtype=torch.float32).to(self.device)
            traj = traj.T.unsqueeze(0)  # (1, T, 2)
            
            if self.noise_pct > 0.0:
                scale = traj.std(dim=1, keepdim=True)            # (1, 1, 2)
                noise = torch.randn_like(traj) * (self.noise_pct * scale)
                traj = traj + noise
                
            out.append(traj)
        
        
        return out

    def __hopf_ode(self, t, state):
        x1, x2 = state
        μ, ω, A = self.mu, self.omega, self.A
        dx1dt =  μ*x1 - ω*x2 - A*x1*(x1**2 + x2**2)
        dx2dt =  ω*x1 + μ*x2 - A*x2*(x1**2 + x2**2)
        return [dx1dt, dx2dt]

    def _load_csv_traj(self, path: Path) -> torch.Tensor:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        # ensure (T, F) shape
        if data.ndim == 1:
            data = data[:, None]
        # return (1, T, F)
        return (
            torch.from_numpy(data)
                 .float()
                 .unsqueeze(0)
                 .to(self.device)
        )

    