# src/model/ESN.py

import pickle
import torch
from torch import nn
import numpy as np
from typing import TypedDict, Type
from src.utils.pyESN import ESN
from src.model.model_interface import BaseModel
from src.model.model_params import ModelParams
import os
import random
import matplotlib.pyplot as plt

class ESNHyperParams(ModelParams):
    n_inputs: int
    n_outputs: int
    n_reservoir: int
    spectral_radius: float
    sparsity: float
    input_scaling: float | None
    noise: float
    teacher_forcing: bool
    num_trajectories: int        

class ESNModel(BaseModel):
    def __init__(self, hyperparams: ESNHyperParams) -> None:
        super().__init__(hyperparams)

        self.n_inputs = hyperparams["n_inputs"]
        self.n_outputs = hyperparams["n_outputs"]
        self.n_reservoir = hyperparams["n_reservoir"]
        self.spectral_radius = hyperparams["spectral_radius"]
        self.sparsity = hyperparams["sparsity"]
        self.input_scaling = hyperparams["input_scaling"]
        self.noise = hyperparams["noise"]
        self.teacher_forcing = hyperparams["teacher_forcing"]
        self.model_class = hyperparams["model_class"]
        self.model_name = hyperparams["model_name"]
        self.input_before = hyperparams["input_before"]
        self.input_after = hyperparams["input_after"]
        self.train_params = hyperparams["train_params"]
        self.prediction_length = hyperparams["prediction_length"]
        self.random_state = hyperparams["random_state"]
        # ────────────────────────────────────────────────────────────
        # Retrieve the new hyperparameter (default to 1 if missing)
        self.num_trajectories = hyperparams.get("num_trajectories", 1)
        # ────────────────────────────────────────────────────────────

        self.leaking_rate     = hyperparams.get("leaking_rate", 0.4)
        self.ridge_lambda     = hyperparams.get("ridge_lambda", 1e-4)
        self.feedback_scaling = hyperparams.get("feedback_scaling", 0.3)
        

        self.esn = ESN(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_reservoir=self.n_reservoir,
            spectral_radius=self.spectral_radius,
            sparsity=self.sparsity,
            input_scaling=self.input_scaling,
            noise=self.noise,
            teacher_forcing=self.teacher_forcing,
            random_state=self.random_state,
            silent=False,
            # NEW:
            leaking_rate=self.leaking_rate,
            ridge_lambda=self.ridge_lambda,
            feedback_scaling=self.feedback_scaling,
        )

    def test_esn_on_tesor_data(self, enc_input):
        
        if isinstance(enc_input, torch.Tensor):
            seq = enc_input.squeeze(0).cpu().numpy()       
        else:
            seq = enc_input

        self.esn.teacher_forcing = True                     
        _ = self.esn.predict(seq, continuation=False)      

        # grab final warm-up states
        state      = self.esn.laststate.copy()
        last_out   = self.esn.lastoutput.copy()             
        last_in    = last_out.copy()                       

        horizon = self.prediction_length
        preds   = np.zeros((horizon, self.n_outputs))
        

        for k in range(horizon):
            last_in  = self.feedback_scaling * last_out
            state    = self.esn._update(state, last_in, last_out)

            vec      = np.concatenate([state, last_in, [1.0]])   
            last_out = self.esn.out_activation(self.esn.W_out @ vec)

            preds[k] = last_out

        preds = self.esn._unscale_teacher(preds)
        return torch.from_numpy(preds).unsqueeze(0)

    def plot(self, predictions, ground_truths, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        all_indices = list(range(1, len(predictions)))
        sampled_indices = random.sample(all_indices, min(5, len(all_indices)))

        for idx in sampled_indices:
            prediction = predictions[idx]
            ground_truth = ground_truths[idx]
            preceding = ground_truths[idx - 1]

            num_dims = predictions[0].shape[-1]
            fig, axs = plt.subplots(1, num_dims, figsize=(5 * num_dims, 5))
            for dim in range(num_dims):
                ax = axs[dim] if num_dims > 1 else axs
                ax.plot(range(len(preceding)), preceding[:, dim], color='green', label='Preceding')
                ax.plot(range(len(preceding), len(preceding) + len(ground_truth)), ground_truth[:, dim], color='blue', label='Actual')
                ax.plot(range(len(preceding), len(preceding) + len(prediction)), prediction[:, dim], color='red', linestyle='dashed', label='Predicted')
                ax.set_title(f'Sequence {idx}, Dim {dim + 1}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'Dimension {dim + 1}')
                ax.legend()

            plt.tight_layout()
            file_name = os.path.join(save_dir, f'sequence_{idx}.png')
            fig.savefig(file_name)
            plt.close(fig)

        print(f"Plots saved to directory: {save_dir}")

    def embed(self, input):
        input = input.squeeze(0)  
        input = input.detach().cpu().numpy()
        embeddings = self.esn.collect_reservoir_states(input)
        stacked = torch.stack([torch.from_numpy(state) for state in embeddings])
        return stacked.mean(dim=0, keepdim=True)  
    


    def save_to_file(self, filepath: str) -> None:
        self.esn.reservoir_states = []
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
