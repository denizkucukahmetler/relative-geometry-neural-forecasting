import torch
import random
from torch import tensor
from pathlib import Path
from scipy.integrate import solve_ivp
import re
import json
import numpy as np


from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams
from src.data_handlers.base_data_handler import BaseDataHandler



class LorenzDataHandler(BaseDataHandler):
    '''
    Extends BaseDataHandler.
    A DataHandler for generating Lorenz System data.
    '''

    def __init__(self, hyperparams: BaseDataHandlerParams,  experiment_name, is_eval_data):
        '''
        Constructor. Will generate the trajectories and store them inside the class.
        Args:
            hyperparams: dict[str, any] contains hyperparameters for data generation. Should contain:
                - num_trajectories: Number of trajectories to generate (default 100)
                - trajectory_length: Length of each trajectory (default 2000)
                - dt: Timestep between points (default 0.01)
                - rand_start_bounds: Trajectories start points will be random in all dimensions between -rand_start_bounds and +rand_start_bounds
                    if 0, all trajectories will start at (1,1,1). Default 0
                - device: The device where the Tensors should be stored (default cpu)

        '''
        super().__init__(hyperparams)
        
        self._traj_dir = Path(f'results/{experiment_name}/trajectories')
        
        random.seed(hyperparams.get('seed', 42))
        np.random.seed(hyperparams.get('seed', 42))
        
        
        approach = hyperparams.get('split_approach', 'trajectory')
        if approach == 'trajectory':

            train_ic = hyperparams.get('initial_conditions_train', None)
            val_ic   = hyperparams.get('initial_conditions_val',None)
            test_ic  = hyperparams.get('initial_conditions_test',None)
            # 1) if user didn't supply any train-ICs, sample them at random
            if not train_ic:
                train_ic = [
                    (
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
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
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds)
                    )
                    for _ in range(self.num_trajectories)
                ]
            if not test_ic:
                test_ic = [
                    (
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
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
        noise_pct = self.hyperparams.get("noise_pct", 0.0)  # e.g. 0.05 = 5%


        for initial_state in initials:
            sol = solve_ivp(self.__lorenz, self.t_span, initial_state, t_eval=self.t_eval, method='RK45',rtol=1e-9, atol=1e-12)
            trajectory = tensor(sol.y, dtype=torch.float).to(self.device)  # shape: (3, len(t_eval))
            trajectory = trajectory.T.unsqueeze(0)
            if noise_pct > 0.0:
                scale = trajectory.std(dim=1, keepdim=True)  # (1,1,3)
                noise = torch.randn_like(trajectory) * (noise_pct * scale)
                trajectory = trajectory + noise

            
            out.append(trajectory)
        return out


    def __lorenz(self, t, state):
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    
   