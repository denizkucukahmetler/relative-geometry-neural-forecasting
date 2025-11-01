from typing import TypedDict, List, Tuple


import numpy as np
import torch
from copy import deepcopy
import os
import numpy as np
from src.data_handlers.dynamical_dataset import DynamicalDataset
from src.data_handlers.tensor_normalizer import TensorNormalizer

class BaseDataHandlerModelParams(TypedDict):
    input_before: int
    input_after: int
    prediction_length: int

class BaseDataHandler:
    def __init__(self, hyperparams): 
        self.hyperparams = hyperparams
        # self.trajectories: list[torch.Tensor] = [] commented out
        self.num_trajectories = hyperparams['num_trajectories']
        self.trajectory_length = hyperparams['trajectory_length']
        self.dt = hyperparams['dt']
        self.t_span = (0, self.dt * self.trajectory_length)
        self.t_eval = np.arange(self.t_span[0], self.t_span[1], self.dt)
        self.device = hyperparams['device']
        self.rand_start_bounds = hyperparams['rand_start_bounds']
        self.save_dir = hyperparams['save_dir']
        

        
        self.global_scaler: TensorNormalizer | None = None
        # placeholders for “custom” splits—if set by a subclass
        self.train_trajectories: list[torch.Tensor] | None = None
        self.val_trajectories:   list[torch.Tensor] | None = None
        self.test_trajectories:  list[torch.Tensor] | None = None
        

        

    def get_datasets(self, model_hyperparams_list: list[BaseDataHandlerModelParams]) -> list[DynamicalDataset]:
        if self.train_trajectories is None \
        or self.val_trajectories   is None \
        or self.test_trajectories  is None:
            raise RuntimeError(
                "DataHandler subclass must set train_/val_/test_trajectories in __init__"
            )

        train_trajectories = self.train_trajectories
        val_trajectories   = self.val_trajectories
        test_trajectories  = self.test_trajectories
        

        if self.global_scaler is None:
            
            try:
                
                if train_trajectories[0].dim() == 3 and train_trajectories[0].shape[0] == 1:
                    # Squeeze the first dimension if it's 1 (batch dim)
                    all_train_data = torch.cat([t.squeeze(0) for t in train_trajectories], dim=0)
                elif train_trajectories[0].dim() == 2:
                    # Assumes trajectories are already [Time, Features]
                    all_train_data = torch.cat(train_trajectories, dim=0)
                else:
                     raise ValueError(f"Unexpected trajectory shape: {train_trajectories[0].shape}")

                self.global_scaler = TensorNormalizer(standardize=True) # Or False for min-max
                self.global_scaler.fit(all_train_data.to(self.device)) # Fit on the correct device
                
                
            except Exception as e:
                print(f"Error fitting global scaler: {e}. Proceeding without global normalization.")

        # Compute the indices of the trajectories which are possible for all models based on their parameters
        max_input_before = max(d['input_before'] for d in model_hyperparams_list)

        max_input_after = max(d['input_after'] for d in model_hyperparams_list)
        max_output = max(d['prediction_length'] for d in model_hyperparams_list)

       
        # Iterate through all hyperparam dicts and create one dataset per each one
        datasets = []
        for hyperparams in model_hyperparams_list:
            input_before = hyperparams.get('input_before', 0)
            input_after = hyperparams.get('input_after', 0)
            prediction_length = hyperparams['prediction_length']
            
            split_datasets = []
            window_len = input_before + max(input_after, prediction_length)
            for split_data in [train_trajectories, val_trajectories, test_trajectories]:

                # Assuming trajectories have shape [Batch (optional), Time, Features]
                # Get num_timesteps from the shape

                num_timesteps = split_data[0].shape[1] if split_data[0].dim() == 3 else split_data[0].shape[0]
                indices = self.__get_valid_indices(num_timesteps, max_input_before, max_input_after, max_output) 
                enc_inputs, dec_inputs, dec_targets, timesteps, scalers_list = self.__create_sequences(
                                                                                              split_data,
                                                                                              input_before,
                                                                                              input_after,
                                                                                              prediction_length,
                                                                                              indices,
                                                                                              normalize=True) # Explicitly request normalization

                dataset = DynamicalDataset(enc_inputs, dec_inputs, dec_targets, timesteps, scalers_list) # Pass the list of scalers
                split_datasets.append(dataset)
            
            datasets.append(split_datasets)
            

        return datasets

    

    def get_trajectory(self, index):
        """
        Retrieves a specific trajectory from the stored trajectories.

        Args:
            index (int): The index of the trajectory to retrieve.

        Returns:
            torch.Tensor: The requested trajectory tensor.
        """
        if index >= len(self.trajectories):
            raise ValueError(f'Trajectory {index} does not exist. Only have {len(self.trajectories)} trajectories.')
        return self.trajectories[index]

    def __get_valid_indices(self, num_timesteps, before_index, after_index, output_length):
        # The valid indices are those where the window does not go out of bounds
        valid_indices = [
            index for index in range(before_index, num_timesteps - output_length - after_index)
            # 1 instead of 0 bc NODEs don't need points before, which is why index 0 would be valid
            # However, the decoder_inputs use index - 1 (which would result in -1); Starting with 1 here avoids that
            if 1 <= index - before_index and index + output_length + after_index < num_timesteps
        ]
        return valid_indices

    # In src/data_handlers/base_data_handler.py, replace the existing __create_sequence method

    def __create_sequence(self, trajectory: torch.Tensor, index: int, input_before: int, input_after: int,
                          target_length: int,
                          normalize: bool = True) -> \
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, TensorNormalizer | None]:
        """
        Cuts a single sequence from the trajectory... (rest of docstring)
        """
        # Encoder inputs: Up to `index - 1` (encodes point `index - 1`)
        # Ensure slicing respects potential batch dimension (index 0) if trajectories have it
        if trajectory.dim() == 3: # Shape [Batch, Time, Features]
            enc_inputs = deepcopy(trajectory[:, index - input_before: index + input_after, :])
            enc_inputs = enc_inputs.squeeze(0) # Remove batch dim for sequence processing
            # Decoder inputs: Start at `index - 1` and predict forward
            dec_at_t = deepcopy(trajectory[:, index - 1: index + target_length, :])
            # Decoder inputs (for teacher forcing): All but the last point from dec_at_t
            dec_inputs = deepcopy(dec_at_t[:, :-1, :])
            dec_inputs = dec_inputs.squeeze(0) # Remove batch dim
            # Decoder targets: All but the first point from dec_at_t
            dec_targets = deepcopy(dec_at_t[:, 1:, :])
            dec_targets = dec_targets.squeeze(0) # Remove batch dim
        elif trajectory.dim() == 2: # Shape [Time, Features]
            enc_inputs = deepcopy(trajectory[index - input_before: index + input_after, :])
            # Decoder inputs: Start at `index - 1` and predict forward
            dec_at_t = deepcopy(trajectory[index - 1: index + target_length, :])
            # Decoder inputs (for teacher forcing): All but the last point from dec_at_t
            dec_inputs = deepcopy(dec_at_t[:-1, :])
            # Decoder targets: All but the first point from dec_at_t
            dec_targets = deepcopy(dec_at_t[1:, :])
        else:
            raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}")


        scaler_to_return = None # Initialize

        # --- Apply Global Normalization ---
        if normalize and self.global_scaler and self.global_scaler.center is not None: # Check if normalize=True and scaler is fitted
            # --- Use transform, not fit_transform ---
            enc_inputs = self.global_scaler.transform(enc_inputs) # No need for deepcopy here if transform doesn't modify inplace
            dec_inputs = self.global_scaler.transform(dec_inputs)
            dec_targets = self.global_scaler.transform(dec_targets)
            scaler_to_return = self.global_scaler # Use the single global scaler
        elif normalize:
             # If normalize is True, but scaler isn't ready (e.g., fitting failed or no data), create a dummy
             print("Warning: Normalization requested but global scaler not available. Using dummy scaler.")
             scaler_to_return = self._create_dummy_scaler(enc_inputs.shape[-1], enc_inputs.device)
        else:
            # If normalize is False, also provide a dummy scaler for downstream consistency
            scaler_to_return = self._create_dummy_scaler(enc_inputs.shape[-1], enc_inputs.device)


        timesteps = torch.arange(0.0, target_length * self.dt, self.dt, dtype=torch.float32, device=self.device) # Use self.device
        return enc_inputs, dec_inputs, dec_targets, timesteps, scaler_to_return




    def __create_sequences(self, trajectories, input_before: int, input_after: int,
                           output: int, indices: list[int] = None, normalize: bool = True) -> tuple[ # Add normalize param
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]: # Change return type hint for scalers
        """
        Calls create_sequence for every index in indices... (rest of docstring)
        """
        enc_inputs, dec_inputs, dec_targets, timesteps, scalers = [], [], [], [], []

        if indices is None:
             # Handle case where indices are not provided (needs implementation if required)
             raise NotImplementedError("Default index generation not implemented yet.")

        for idx, trajectory in enumerate(trajectories):
            for index in indices:
                seq_enc_inputs, seq_dec_inputs, seq_dec_targets, seq_timesteps, seq_scaler = self.__create_sequence(
                    trajectory, index, input_before, input_after, output, normalize=normalize) # Pass normalize flag
                enc_inputs.append(seq_enc_inputs)
                dec_inputs.append(seq_dec_inputs)
                dec_targets.append(seq_dec_targets)
                timesteps.append(seq_timesteps)
                scalers.append(seq_scaler) # Append the returned scaler (global or dummy)

        enc_inputs = torch.stack(enc_inputs).to(self.device)
        dec_inputs = torch.stack(dec_inputs).to(self.device)
        dec_targets = torch.stack(dec_targets).to(self.device)
        timesteps = torch.stack(timesteps).to(self.device) # Ensure timesteps are on correct device too

        scalers = np.array(scalers, dtype=object) # Store as object array

        return enc_inputs, dec_inputs, dec_targets, timesteps, scalers
    


    def save_trajectories_to_file(self, directory: str, traj_list):


        os.makedirs(directory, exist_ok=True) # Dont raise error if dir already exists

        # If any file exists in the directory, skip saving
        if any(os.path.isfile(os.path.join(directory, f)) for f in os.listdir(directory)):
            print(f"Directory '{directory}' already contains files; skipping save.")
            return

        # Save each tensor to a CSV file
        for idx, traj in enumerate(traj_list):
            file_path = os.path.join(directory, f"trajectory_{idx + 1}.csv")
            


            # Transpose the tensor so it has 3 columns and n rows
            traj = traj.squeeze(0).permute(1, 0)
            traj = traj.detach().cpu()
            rows, cols = traj.shape
            header = ",".join([f"Column{i+1}" for i in range(rows)])
            np.savetxt(file_path, traj.T, delimiter=",", header=header, comments="")
    
        print(f"CSV files have been saved in the '{directory}' directory.")
        
    