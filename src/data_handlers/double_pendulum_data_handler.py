import torch
import random
from torch import tensor
from pathlib import Path
import re
import numpy as np

from scipy.integrate import solve_ivp

from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams
from src.data_handlers.base_data_handler import BaseDataHandler



class DoublePendulumDataHandler(BaseDataHandler):
    '''
    Extends BaseDataHandler.
    A DataHandler for generating double pendulum system data.
    '''

    def __init__(self, hyperparams: BaseDataHandlerParams,  experiment_name, is_eval_data):
        '''
        Constructor. Will generate the trajectories and store them inside the class.
        Args:
            hyperparams: dict[str, any] contains hyperparameters for data generation. Should contain:
                - num_trajectories: Number of trajectories to generate (default 100)
                - trajectory_length: Length of each trajectory (default 2000)
                - dt: Timestep between points (default 0.01)
                - rand_start_bounds: Trajectories start points will be random within bounds for angles and velocities
                - device: The device where the Tensors should be stored (default cpu)
        '''
        super().__init__(hyperparams)
        
        self._traj_dir = Path(f'results/{experiment_name}/trajectories')
        self.noise_pct  = float(hyperparams.get('noise_pct', 0.0)) # measurement noise in (x,y)

        
        random.seed(hyperparams.get('seed', 42))
        np.random.seed(hyperparams.get('seed', 42))
        
        
        
        approach = hyperparams.get('split_approach', 'trajectory')
        if approach == 'trajectory':

            train_ic = hyperparams.get('initial_conditions_train', None)
            val_ic   = hyperparams.get('initial_conditions_val',None)
            test_ic  = hyperparams.get('initial_conditions_test',None)
            if not train_ic:
                train_ic = [
                    (
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds)
                    )
                    for _ in range(self.num_trajectories)
                ]

            if not val_ic:
                val_ic = [
                    (
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
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
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds),
                        random.uniform(-self.rand_start_bounds, self.rand_start_bounds)
                    )
                    for _ in range(self.num_trajectories)
                ]
                
                    
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
        for initial_state in initials:
            sol = solve_ivp(self.double_pendulum, self.t_span, initial_state,
                            t_eval=self.t_eval, method='RK45')

            positions = self.calculate_positions(sol.y)                 # already a torch.Tensor
            trajectory = (positions                                  # shape: (4, len(t_eval))
                          .to(self.device, dtype=torch.float)        # move & cast in one go
                          .T.unsqueeze(0))                           # (1, len, 4)
            if self.noise_pct > 0.0:
                scale = trajectory.std(dim=1, keepdim=True)  # (1,1,4)
                noise = torch.randn_like(trajectory) * (self.noise_pct * scale)
                trajectory = trajectory + noise

            out.append(trajectory)
        return out

        

    def double_pendulum(self, t, state):
        g = 9.81
        L1, L2 = 1.0, 1.0  # lengths of pendulums
        m1, m2 = 1.0, 1.0  # masses of pendulums

        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float32)
        theta1, theta2, omega1, omega2 = state
        theta1 = torch.Tensor(theta1)
        delta = theta2 - theta1
        den1 = (m1 + m2) * L1 - m2 * L1 * torch.cos(delta) ** 2
        den2 = (L2 / L1) * den1

        dtheta1_dt = omega1
        dtheta2_dt = omega2

        domega1_dt = (
            m2 * L1 * omega1 ** 2 * torch.sin(delta) * torch.cos(delta)
            + m2 * g * torch.sin(theta2) * torch.cos(delta)
            + m2 * L2 * omega2 ** 2 * torch.sin(delta)
            - (m1 + m2) * g * torch.sin(theta1)
        ) / den1

        domega2_dt = (
            -m2 * L2 * omega2 ** 2 * torch.sin(delta) * torch.cos(delta)
            + (m1 + m2) * g * torch.sin(theta1) * torch.cos(delta)
            - (m1 + m2) * L1 * omega1 ** 2 * torch.sin(delta)
            - (m1 + m2) * g * torch.sin(theta2)
        ) / den2

        # Convert back to a list of floats
        return [dtheta1_dt.item(), dtheta2_dt.item(), domega1_dt.item(), domega2_dt.item()]

    def calculate_positions(self, states):
        L1, L2 = 1.0, 1.0  # lengths of pendulums
        theta1, theta2 = states[0], states[1]
        theta1 = torch.Tensor(theta1)
        theta2 = torch.Tensor(theta2)
        x1 = L1 * torch.sin(theta1)
        y1 = -L1 * torch.cos(theta1)
        x2 = x1 + L2 * torch.sin(theta2)
        y2 = y1 - L2 * torch.cos(theta2)

        return torch.stack([x1, y1, x2, y2], dim=0)
    
    