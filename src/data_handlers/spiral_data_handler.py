import torch
import random
from scipy.integrate import solve_ivp
from pathlib import Path
import numpy as np
import re

from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams
from src.data_handlers.base_data_handler import BaseDataHandler

class SpiralDataHandler(BaseDataHandler):
    '''
    Extends BaseDataHandler.
    A DataHandler for generating 2D limit‐cycle data (converging to a circle).
    '''
    
    def __init__(self, hyperparams: BaseDataHandlerParams, experiment_name, is_eval_data):
        
        super().__init__(hyperparams)
        
        self._traj_dir = Path(f'results/{experiment_name}/trajectories')
        
        random.seed(hyperparams.get('seed', 42))
        np.random.seed(hyperparams.get('seed', 42))

        approach = hyperparams.get('split_approach', 'trajectory')
        
        # limit‐cycle parameters (can be set in hyperparams)
        self.radius = getattr(hyperparams, 'limit_cycle_radius', 1.0)
        self.mu     = getattr(hyperparams, 'limit_cycle_mu',     1.0)
        self.omega  = getattr(hyperparams, 'limit_cycle_omega',  1.0)
        self.noise_pct  = float(hyperparams.get('noise_pct', 0.0))    # measurement noise (Cartesian)

        
        if approach == 'trajectory':
            # 1) if user didn't supply any train-ICs, sample them at random

            train_ic = hyperparams.get('initial_conditions_train', None)
            val_ic   = hyperparams.get('initial_conditions_val',None)
            test_ic  = hyperparams.get('initial_conditions_test',None)
        
            if not train_ic:
                train_ic = [
                    (
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds)
                    )
                    for _ in range(self.num_trajectories)
                ]

            # 2) if no explicit val/test lists, auto-perturb or reuse
            if not val_ic:
                 val_ic = [
                    (
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds)
                    )
                    for _ in range(self.num_trajectories)
                ]
            if not test_ic:
                test_ic = [
                    (
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds)
                    )
                    for _ in range(self.num_trajectories)
                ]

            # 3) now generate each split
            self.train_trajectories = self._build_trajectories(train_ic)
            self.val_trajectories   = self._build_trajectories(val_ic)
            self.test_trajectories  = self._build_trajectories(test_ic)
            
        elif approach == 'time':
            all_ic    = hyperparams.get('initial_conditions_all', None)
            if not all_ic:
                all_ic = [
                    (
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds)
                    )
                    for _ in range(self.num_trajectories)
                ]
            raw_trajs = self._build_trajectories(all_ic)

            t, v, te = self.time_split(raw_trajs)
            self.train_trajectories = t
            self.val_trajectories   = v
            self.test_trajectories  = te

        else:
            raise ValueError(f"Unknown split_approach: {approach!r}")

            
        # optionally write them out
        if is_eval_data:
            self.save_trajectories_to_file(self._traj_dir / "train", self.train_trajectories)
            self.save_trajectories_to_file(self._traj_dir / "val",   self.val_trajectories)
            self.save_trajectories_to_file(self._traj_dir / "test",  self.test_trajectories)

        

    def _build_trajectories(self, initials: list[tuple[float, float]]) -> list[torch.Tensor]:
            out = []
            for x0, y0 in initials:
                # convert Cartesian → polar for solver
                r0     = np.hypot(x0, y0)
                theta0 = np.arctan2(y0, x0)
                sol = solve_ivp(
                    self.__limit_cycle,
                    self.t_span,
                    [r0, theta0],
                    t_eval=self.t_eval,
                    method='RK45'
                )
                # polar → Cartesian trajectory (2, T)
                traj = self.__generate_limit_cycle_trajectory(sol.y[0], sol.y[1])
                # make it (1, T, 2) on the right device
                traj = traj.T.unsqueeze(0).to(self.device)
                
                if self.noise_pct > 0.0:
                    scale = traj.std(dim=1, keepdim=True)            # (1, 1, 2)
                    noise = torch.randn_like(traj) * (self.noise_pct * scale)
                    traj = traj + noise
                out.append(traj)
            return out



    def __limit_cycle(self, t, state):
        '''
        Polar‐form limit‐cycle ODE:
          dr/dt     = mu * (radius - r)
          dθ/dt     = omega
        '''
        r, theta = state
        drdt     = self.mu * (self.radius - r)
        dthetadt = self.omega
        return [drdt, dthetadt]

    def __generate_limit_cycle_trajectory(self, radii, thetas):
        '''
        Convert polar coords back to Cartesian (x, y).
        Returns a torch.Tensor of shape (2, len(radii)).
        '''
        r     = torch.tensor(radii, dtype=torch.float, device=self.device)
        theta = torch.tensor(thetas, dtype=torch.float, device=self.device)

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack((x, y), dim=0)
