from __future__ import annotations

import os
import random
from time import time
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from src.model.model_interface import BaseModel
from src.data_handlers.data_utils import sequence_collate_fn
from src.relative_latent_spaces import evaluate_models
from src.encoder.NoneModelEncoder import NoneModelEncoder
from src.decoder.NoneModelDecoder import NoneModelDecoder
from src.model.NoneModel import NoneModel

plt.switch_backend("Agg")

__all__ = [
    "TrainerHyperparams",
    "Trainer",
    "find_data_range",
]

class TrainerHyperparams(TypedDict):
    lr: float
    device: str
    batch_size: int
    epochs: int
    model_type: str
    lr_decay: bool
    teacher_force_prob: float | None
    directory: str


def find_data_range(dataloader: DataLoader) -> tuple[float, float]:
    """Return (min, max) across *dec_targets* in *dataloader*."""

    min_v, max_v = float("inf"), float("-inf")
    for *_ , dec_targets, _, _ in ((a, b, c, d, e) for a, b, c, d, e in dataloader):
        min_v = min(min_v, float(dec_targets.min()))
        max_v = max(max_v, float(dec_targets.max()))
    return min_v, max_v


class Trainer:
    """Train/validate/test utilities for dynamical‑model experiments."""

    def __init__(self, hyperparams: TrainerHyperparams):
        self.hyperparams = hyperparams
        self.model: BaseModel | None = None

    def train_model(
        self,
        model: BaseModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        hyperparams: dict,
        data_handler_params: dict,
        tune: bool = False,
    ) -> float | None:
        """Train *model*; return best val‑loss when *tune* is *True*."""

        self.model = model.to(self.hyperparams["device"])
        save_dir = (
            os.path.dirname(hyperparams["directory"]) if tune else hyperparams["directory"]
        )
        os.makedirs(save_dir, exist_ok=True)

        if self.hyperparams["model_type"] == "NoneModel":
            return None
        if self.hyperparams["model_type"] == "ESN":
            self._fit_esn(train_dataset)
            return None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparams["batch_size"],
            shuffle=True,
            collate_fn=sequence_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.hyperparams["batch_size"],
            shuffle=False,
            collate_fn=sequence_collate_fn,
        )

        optimizer = optim.Adam(model.parameters(), lr=self.hyperparams["lr"])
        scheduler = ExponentialLR(optimizer, gamma=0.95) if self.hyperparams["lr_decay"] else None
        criterion = nn.MSELoss()

        sim_val = self._latent_similarity(model, test_dataset, save_dir)
        best_val = float("inf")

        log_fp = os.path.join(save_dir, "loss_log.csv")
        with open(log_fp, "w", encoding="utf-8") as lf:
            lf.write("Epoch,Val_Loss,Train_Loss,Similarity\n")
            lf.write(f"0,NA,NA,{sim_val:.6f}\n")
            lf.flush()

            for epoch in range(1,self.hyperparams["epochs"] + 1):
                t0 = time()
                tr_loss = self._run_epoch(model, train_loader, optimizer, criterion, epoch)
                val_loss = self._evaluate(model, val_loader, criterion)

                if scheduler:
                    scheduler.step()
                    lr = scheduler.get_last_lr()[0]
                else:
                    lr = self.hyperparams["lr"]

                sim_val = self._latent_similarity(model, test_dataset, save_dir)

                print(
                    f"Epoch {epoch}/{self.hyperparams['epochs']}  "
                    f"train={tr_loss:.4f}  val={val_loss:.4f}  sim={sim_val:.4f}  "
                    f"lr={lr:.2e}  ({time() - t0:.1f}s)"
                )
                lf.write(f"{epoch},{val_loss},{tr_loss},{sim_val}\n")
                lf.flush()

                if val_loss < best_val:
                    model.save_to_file(os.path.join(save_dir, "best_current_model.pth"))
                    best_val = val_loss

                if epoch >= 20 and self._stable(log_fp, threshold=5):
                    print("Val‑loss change <5% for 3 epochs — early stop.")
                    break

        return best_val if tune else None

    
    def test_model(self, model: BaseModel, test_dataset) -> None:
        self.model = model
        device = self.hyperparams['device']
        self.model.to(device)

        if self.hyperparams['model_type'] == 'NoneModel': #or self.hyperparams['model_type'] == 'ESN':
            return

        test_dataloader = DataLoader(test_dataset, batch_size=self.hyperparams['batch_size'], shuffle=True,
                                     collate_fn=sequence_collate_fn)
        criterion = nn.MSELoss()  # Assuming regression output for continuous values
        
        if self.hyperparams['model_type'] != 'ESN':
            model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for enc_inputs, dec_inputs, dec_targets, timestamp, scalar in iter(test_dataloader):
                enc_inputs = enc_inputs.to(device)
                dec_targets = dec_targets.to(device)
                dec_inputs = dec_inputs.to(device)
                
                if self.hyperparams['model_type'] in ['MLP']:  # need to reshape
                    # Flatten dec_targets
                    batch_size, prediction_length, num_dims = dec_targets.shape
                    dec_targets = dec_targets.view(batch_size, prediction_length * num_dims)
                    dec_targets = dec_targets.to(device)
                    outputs = model(enc_inputs, dec_inputs).to(device)
                elif self.hyperparams['model_type'] in ['RNN', 'Transformer', 'NODE', "Koopman"]:
                    outputs = model(enc_inputs, dec_inputs).to(device)
                elif self.hyperparams['model_type'] == 'ESN':
                    outputs = model.test_esn_on_tesor_data(enc_inputs).to(device)
                else:
                    print('Test loss can not be calculated for this Model or is done elsewhere (ESN).')
                    return None

                loss = criterion(outputs, dec_targets)
                test_loss += loss.item()

            test_loss_save = test_loss / len(test_dataloader)
            print(f'Test Loss: {test_loss_save:.4f}')


    def plot(self, model: BaseModel, test_dataset, num_samples: int, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        if self.hyperparams['model_type'] == 'NoneModel' or self.hyperparams['model_type'] == 'ESN':
            return
        # Move model to device
        device = self.hyperparams['device']
        model.to(device)
        model.eval()


        # Randomly sample indices from the dataset
        all_indices = list(range(len(test_dataset)))
        sampled_indices = random.sample(all_indices, num_samples)

        # Get the model's predictions
        enc_inputs_scaled = []
        dec_targets_scaled = []

        # Sampled sequences loop
        with torch.no_grad():
            for idx in sampled_indices:
                # Access the batch for the current index
                enc_inputs, dec_inputs, dec_targets, timestamp, scaler = test_dataset[idx]

                # Move inputs and targets to the same device as model
                enc_inputs = enc_inputs.unsqueeze(0).to(device)  # Add batch dimension
                dec_targets = dec_targets.unsqueeze(0).to(device)
                dec_inputs = dec_inputs.unsqueeze(0).to(device)

                # Get model prediction
                if self.hyperparams['model_type'] in ['MLP']:  # Reshape MLP output
                    predicted_flat = model(enc_inputs, dec_inputs.to(device))  # Shape: [1, prediction_length * num_dims]
                    predicted = predicted_flat.view(-1, enc_inputs.shape[2]).to(device)  # Shape: [prediction_length, num_dims]
                elif self.hyperparams['model_type'] in ['RNN', 'Transformer', 'NODE', "Koopman"]:
                    predicted = model(enc_inputs, dec_inputs).to(device)  # Shape: [1, prediction_length, num_dims]
                    predicted = predicted.squeeze(0).to(device)  # Shape: [prediction_length, num_dims]
                else: print('Dont know this model, cant generate predictions or is done elsewehere (ESN).')

                # Reverse normalization
                enc_input_scaled = scaler.inverse_transform(enc_inputs.squeeze(0)).to(device)  # Shape: [input_before, num_dims]
                dec_target_scaled = scaler.inverse_transform(
                    dec_targets.squeeze(0)).to(device)  # Shape: [prediction_length, num_dims]
                predicted_scaled = scaler.inverse_transform(predicted).to(device)  # Shape: [prediction_length, num_dims]

                enc_inputs_scaled.append(enc_input_scaled)
                dec_targets_scaled.append(dec_target_scaled)

                # Plotting for each sequence (batch)
                num_dims = enc_inputs.shape[2]  # Number of dimensions
                fig, axs = plt.subplots(1, num_dims, figsize=(5 * num_dims, 5))

                enc_input_scaled = enc_input_scaled.detach().cpu().numpy()
                dec_target_scaled = dec_target_scaled.detach().cpu().numpy()
                predicted_scaled = predicted_scaled.detach().cpu().numpy()

                for dim in range(num_dims):
                    ax = axs[dim] if num_dims > 1 else axs
                    preceding = enc_input_scaled[:, dim]  # Preceding sequence (scaled back)
                    actual = dec_target_scaled[:, dim]  # Actual target sequence
                    predicted = predicted_scaled[:, dim]  # Predicted sequence

                    ax.plot(range(len(preceding)), preceding, color='green', label='Preceding Sequence')
                    ax.plot(range(len(preceding), len(preceding) + len(actual)), actual, color='blue', label='Actual')
                    ax.plot(range(len(preceding), len(preceding) + len(predicted)), predicted, color='red',
                            linestyle='dashed', label='Predicted')
                    ax.set_title(f'Sequence {idx}, Dim {dim + 1}')
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel(f'Dimension {dim + 1}')
                    ax.legend()

                plt.tight_layout()

                # Save each subplot as an individual file
                file_name = os.path.join(save_dir, f'sequence_{idx}.png')
                fig.savefig(file_name)
                plt.close(fig)  # Close the figure to avoid memory leaks

        print(f"Plots saved to directory: {save_dir}")


    def _fit_esn(self, train_dataset: Dataset) -> None:
        loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=sequence_collate_fn)
        ins, outs = [], []
        for _, _, targets, *_ in loader:
            seq = targets.squeeze(0).cpu().numpy()
            ins.append(seq[:-1])
            outs.append(seq[1:])
        assert self.model is not None
        self.model.esn.teacher_forcing = True
        self.model.esn.fit(inputs=ins, outputs=outs)

    def _run_epoch(
        self,
        model: BaseModel,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
    ) -> float:
        device = self.hyperparams["device"]
        model.train()
        tf_prob = self._teacher_force_prob(epoch)
        total = 0.0
        for enc_in, dec_in, dec_tgt, *_ in loader:
            enc_in, dec_in, dec_tgt = (t.to(device) for t in (enc_in, dec_in, dec_tgt))
            optimizer.zero_grad()
            if self.hyperparams["model_type"] == "MLP":
                dec_tgt = dec_tgt.view(dec_tgt.size(0), -1)
            if (
                self.hyperparams["model_type"] == "RNN"
                and "oldrnn" in model.hyperparams.get("model_name", "")
                and tf_prob is not None
            ):
                preds = model(enc_in, dec_in, tf_prob)
                preds = preds.to(device) 
            else:
                preds = model(enc_in, dec_in)
            loss = criterion(preds, dec_tgt)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    def _evaluate(self, model: BaseModel, loader: DataLoader, criterion: nn.Module) -> float:
        device = self.hyperparams["device"]
        total = 0.0
        with torch.no_grad():
            for enc_in, dec_in, dec_tgt, *_ in loader:
                enc_in, dec_in, dec_tgt = (t.to(device) for t in (enc_in, dec_in, dec_tgt))
                if self.hyperparams["model_type"] == "MLP":
                    dec_tgt = dec_tgt.view(dec_tgt.size(0), -1)
                preds = model(enc_in, dec_in)
                preds = preds.to(device) 
                total += criterion(preds, dec_tgt).item()
        return total / len(loader)

    def _latent_similarity(self, model: BaseModel, dataset: Dataset, out_dir: str) -> float:
        device = self.hyperparams["device"]
        reference = NoneModel(
            {
                "model_name": "none",
                "encoder_class": NoneModelEncoder,
                "decoder_class": NoneModelDecoder,
                "encoder_hyperparams": {},
                "decoder_hyperparams": {},
                "train_params": {},
            }
        ).to(device)
        evaluate_models(
            [reference, model],
            [dataset, dataset],
            file_path=os.path.join(out_dir, "sim_tmp_"),
            hyperparams={"device": device, "num_samples": 100, "num_anchors": 80},
        )
        csv = os.path.join(out_dir, "sim_tmp_similarities.csv")
        return pd.read_csv(csv, index_col=0).iloc[0, 1]

    def _teacher_force_prob(self, epoch: int) -> float | None:
        base = self.hyperparams.get("teacher_force_prob")
        return None if base is None else max(0.0, base - 0.1 * epoch)

    @staticmethod
    def _stable(csv_path: str, threshold: float) -> bool:
        """Return True when mean % change of last 3 val-losses < threshold.
        If the log file does not exist yet (early epochs), keep training."""
        if not os.path.exists(csv_path):
            return False
        losses = (
            pd.read_csv(csv_path)["Val_Loss"]
            .tail(3)
            .pct_change()
            .dropna()
            .abs()
            * 100
        )
        return losses.mean() < threshold

