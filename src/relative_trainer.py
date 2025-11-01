from __future__ import annotations

import os
import random
from time import time
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint

import matplotlib.pyplot as plt

from src.model.model_interface import BaseModel
from src.data_handlers.data_utils import sequence_collate_fn
from src.relative_latent_spaces import evaluate_models
from src.encoder.NoneModelEncoder import NoneModelEncoder
from src.decoder.NoneModelDecoder import NoneModelDecoder
from src.model.NoneModel import NoneModel

plt.switch_backend("Agg")

__all__ = [
    "RelativeTrainerHyperparams",
    "RelativeTrainer",
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

class RelativeTrainer:
    """Train/validate/test utilities for dynamical‑model experiments."""

    def __init__(self, hyperparams: TrainerHyperparams):
        self.hyperparams = hyperparams
        self.model: BaseModel | None = None
        self._anchor_indices = None
        self._anchor_inputs = None  
    def print_dataset_ranges(self, *_, **__) -> None:  
        return
    
    def _align_for_loss(self, preds: torch.Tensor, target: torch.Tensor):
        def to_BTD(x):
            if x.dim() == 3:
                return x.permute(1, 0, 2).contiguous() if x.shape[0] < x.shape[1] else x
            return x

        if preds.dim() == 3 and target.dim() == 3:
            if preds.shape[0] == target.shape[1] and preds.shape[1] == target.shape[0]:
                preds = preds.permute(1, 0, 2).contiguous()
            elif target.shape[0] == preds.shape[1] and target.shape[1] == preds.shape[0]:
                target = target.permute(1, 0, 2).contiguous()
        else:
            preds = to_BTD(preds)
            target = to_BTD(target)

        if preds.dim() == 3 and target.dim() == 2:
            preds = preds[:, -1, :]  # [B,D]
        elif preds.dim() == 2 and target.dim() == 3:
            target = target[:, -1, :]  # [B,D]

        if preds.dim() == 3 and target.dim() == 3:
            T = min(preds.shape[1], target.shape[1])
            preds = preds[:, :T, :]
            target = target[:, :T, :]

        if preds.shape != target.shape:
            raise RuntimeError(f"Shape mismatch after align: preds {tuple(preds.shape)} vs target {tuple(target.shape)}")

        return preds, target

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
        print("Calling relative trainer for training on relative latent spaces for stitching.")
        self.model = model.to(self.hyperparams["device"])
        save_dir = (
            os.path.dirname(hyperparams["directory"]) if tune else hyperparams["directory"]
        )
        os.makedirs(save_dir, exist_ok=True)

        if self.hyperparams["model_type"] in ["NoneModel", "ESN"]:
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
        

        best_val = float("inf")

        log_fp = os.path.join(save_dir, "loss_log.csv")
        with open(log_fp, "w", encoding="utf-8") as lf:
            lf.write("Epoch,Val_Loss,Train_Loss\n")
            lf.write(f"0,NA,NA\n")
            lf.flush()

            for epoch in range(1, self.hyperparams["epochs"] + 1):
                t0 = time()
                tr_loss = self._run_epoch(model, train_loader, optimizer, criterion, epoch)
                val_loss = self._evaluate(model, val_loader, criterion)

                if scheduler:
                    scheduler.step()
                    lr = scheduler.get_last_lr()[0]
                else:
                    lr = self.hyperparams["lr"]

                print(
                        f"Epoch {epoch}/{self.hyperparams['epochs']}  "
                        f"train={tr_loss:.4f}  val={val_loss:.4f}"
                        f"lr={lr:.2e}  ({time() - t0:.1f}s)"
                )
                lf.write(f"{epoch},{val_loss},{tr_loss}\n")
                lf.flush()

                if val_loss < best_val:
                    model.save_to_file(os.path.join(save_dir, "best_current_model.pth"))
                    best_val = val_loss

                if epoch >= 20 and self._stable(log_fp, threshold=5):
                    print("Val‑loss change <5% for 3 epochs — early stop.")
                    break

            return best_val if tune else None
    
    def test_model(self, model: BaseModel, test_dataset) -> None:
        device = self.hyperparams["device"]
        total = 0.0
        model.eval()
        model_name = model.hyperparams.get("model_name", "")
        encoder = model.encoder
        decoder = model.decoder
        criterion = nn.MSELoss()

        loader = DataLoader(
            test_dataset,
            batch_size=self.hyperparams["batch_size"],
            shuffle=False,
            collate_fn=sequence_collate_fn,
        )

        with torch.no_grad():
            for enc_input, dec_input, dec_target, *_ in loader:
                enc_input, dec_input, dec_target = (t.to(device) for t in (enc_input, dec_input, dec_target))
                if "mlp" in model_name:
                    dec_target = dec_target.view(dec_target.size(0), -1)
                    enc_output = encoder(enc_input)
                    anchor_latents = self._compute_anchor_latents(encoder, device)
                    rel_space = self._get_relative_space(enc_output, anchor_latents)
                    if rel_space is None: continue
                    if "node" in model_name:
                        rel_space = self.pass_to_ode(model, rel_space, device)
                    if "koopman" in model_name:
                        rel_space = self.pass_to_koopman(model, rel_space).to(device)
                    preds = decoder(rel_space)
                    if "koopman" in model_name: preds = preds.view(preds.size(0),-1) #flatten again
                elif "transformer" in model_name:
                    enc_output = encoder(enc_input)
                    anchor_latents = self._compute_anchor_latents(encoder, device)
                    rel_space = self._get_relative_space(enc_output, anchor_latents)
                    if rel_space is None: continue
                    if "node" in model_name:
                        rel_space = self.pass_to_ode(model, rel_space, device)
                    if "koopman" in model_name:
                        rel_space = self.pass_to_koopman(model, rel_space).to(device)
                    preds = decoder(dec_input, rel_space)
                elif "oldrnn" in model_name:
                    enc_output, hidden = encoder(enc_input)
                    anchor_latents = self._compute_anchor_latents(encoder, device)
                    rel_space = self._get_relative_space(enc_output, anchor_latents)
                    if rel_space is None: continue
                    preds = decoder(dec_input, hidden, rel_space, None)
                    preds = preds.squeeze(-1)
                else: #autoregrnn
                    enc_output, hidden = encoder(enc_input)
                    anchor_latents = self._compute_anchor_latents(encoder, device)
                    rel_space = self._get_relative_space(enc_output, anchor_latents)
                    if rel_space is None: continue
                    if "node" in model_name:
                        rel_space = self.pass_to_ode(model, rel_space, device) 
                        preds, _ = decoder(rel_space, dec_input, hidden, stitching=True)
                    if "koopman" in model_name:
                        rel_space = self.pass_to_koopman(model, rel_space).to(device)
                        preds, _ = decoder(rel_space, dec_input, hidden, stitching=True)
                    if not "koopman" in model_name and not "node" in model_name:
                        preds, _ = decoder(rel_space, dec_input, hidden, teacher_force_prob=0)
                preds = preds.to(device)
                preds, dec_target = self._align_for_loss(preds, dec_target)
                total += criterion(preds, dec_target).item()
            print(f"Test loss is: {total / len(loader)}")
        return None
    
        
    def _init_fixed_anchors(self, train_dataset, model_name: str, K: int, save_dir:str):
        import random
        assert K <= len(train_dataset), "num anchors > dataset size"
        seed = 42
        rng = random.Random(seed)
        self._anchor_indices = rng.sample(range(len(train_dataset)), K)

        
        items = [train_dataset[i] for i in self._anchor_indices]
        enc_inputs, *_ = sequence_collate_fn(items)    
        self._anchor_inputs = enc_inputs              
        np.save(save_dir + "/anchor_indices.npy", np.array(self._anchor_indices, dtype=np.int64))


    def _compute_anchor_latents(self, encoder, device):
        with torch.no_grad():
            out = encoder(self._anchor_inputs.to(device))
        if isinstance(out, tuple): #RNN
            out = out[0]
        return out.detach()  # no grad through anchors
    
    def _z(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-8)

    def _get_relative_space(self, enc_outputs: torch.Tensor, anchor_latents: torch.Tensor):
        if enc_outputs.ndim == 3:
            time_dim = enc_outputs.shape[1] 
            enc_outputs = enc_outputs[:, -1, :] 
            anchor_latents = anchor_latents[:, -1, :] 
            rel_space_2d = F.cosine_similarity(enc_outputs.unsqueeze(1), anchor_latents.unsqueeze(0), dim=-1)
            rel_space_3d = rel_space_2d.unsqueeze(1).expand(-1, time_dim, -1) 
            return rel_space_3d.transpose(1,2) 
        return F.cosine_similarity(enc_outputs.unsqueeze(1), anchor_latents.unsqueeze(0), dim=-1)

    def pass_to_ode(self, model, rel_space, device):
        model = model.to(device)
        dt = self.model.hyperparams['data_handler_params']['dt']
        t = torch.arange(0, dt * self.model.hyperparams['prediction_length'], dt).to(device)
        ode_func = model.ode_func
        if rel_space.dim() == 3:
            rel_space = rel_space[:, :, -1] 
        odeout = odeint(ode_func, rel_space, t, method='rk45', rtol=1e-4, atol=1e-6)
        odeout = odeout.permute(1, 0, 2).contiguous()
        return odeout

    
    def pass_to_koopman(self, model, rel_space):
        K = self.model.K
        if rel_space.dim() == 3: rel_space = rel_space[:, :, -1] 
        encoded_pred = [rel_space]
        for _ in range(model.hyperparams['prediction_length']-1):
            rel_space = torch.matmul(rel_space, K.T)
            encoded_pred.append(rel_space)
        rel_space = torch.stack(encoded_pred, dim=1) 
        return rel_space


    def _run_epoch(self, model, loader, optimizer, criterion, epoch):
        device = self.hyperparams["device"]
        model.train()
        model_name = model.hyperparams.get("model_name", "")
        encoder = model.encoder
        decoder = model.decoder
        tf_prob = self._teacher_force_prob(epoch)
        total = 0.0
        for enc_input, dec_input, dec_target, *_ in loader:
            enc_input, dec_input, dec_target = (t.to(device) for t in (enc_input, dec_input, dec_target)) #move all tensors to the same device
            optimizer.zero_grad() 
            if "mlp" in model_name:
                dec_target = dec_target.view(dec_target.size(0), -1)                                        
                enc_output = encoder(enc_input)
                anchor_latents = self._compute_anchor_latents(encoder, device)
                rel_space = self._get_relative_space(enc_output, anchor_latents)                                      
                if rel_space is None: continue
                if "node" in model_name:
                    rel_space = self.pass_to_ode(model, rel_space, device)
                if "koopman" in model_name:
                    rel_space = self.pass_to_koopman(model, rel_space).to(device)
                preds = decoder(rel_space)
                if "koopman" in model_name: preds = preds.view(preds.size(0),-1) #flatten again
            elif "transformer" in model_name:
                enc_output = encoder(enc_input)
                anchor_latents = self._compute_anchor_latents(encoder, device)
                rel_space = self._get_relative_space(enc_output, anchor_latents)
                if rel_space is None: continue
                if "node" in model_name:
                    rel_space = self.pass_to_ode(model, rel_space, device)
                if "koopman" in model_name:
                    rel_space = self.pass_to_koopman(model, rel_space).to(device)
                preds = decoder(dec_input, rel_space)
            elif "oldrnn" in model_name and tf_prob is not None:
                enc_output, hidden = encoder(enc_input)
                anchor_latents = self._compute_anchor_latents(encoder, device)
                rel_space = self._get_relative_space(enc_output, anchor_latents)
                if rel_space is None: continue
                if "node" in model_name:
                    rel_space = self.pass_to_ode(model, rel_space, device)
                preds = decoder(dec_input, hidden, rel_space, tf_prob)
                preds = preds.squeeze(-1)
            elif "autoreg" in model_name:
                enc_output, hidden = encoder(enc_input)
                anchor_latents = self._compute_anchor_latents(encoder, device)
                rel_space = self._get_relative_space(enc_output, anchor_latents)
                if rel_space is None: continue
                if "node" in model_name:
                    rel_space = self.pass_to_ode(model, rel_space, device) 
                    preds, _ = decoder(rel_space, dec_input, hidden, stitching=True)
                if "koopman" in model_name:
                    rel_space = self.pass_to_koopman(model, rel_space).to(device)
                    preds, _ = decoder(rel_space, dec_input, hidden, stitching=True)
                if not "koopman" in model_name and not "node" in model_name:
                    preds, _ = decoder(rel_space, dec_input, hidden, teacher_force_prob=0)
            elif "none" in model_name:
                continue
            else:
                raise TypeError("not implemented model")
                    
            preds = preds.to(device)
            preds, dec_target = self._align_for_loss(preds, dec_target)
            loss = criterion(preds, dec_target)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    def _evaluate(self, model: BaseModel, loader: DataLoader, criterion: nn.Module) -> float:
        device = self.hyperparams["device"]
        total = 0.0
        model.eval()
        model_name = model.hyperparams.get("model_name", "")
        encoder = model.encoder
        decoder = model.decoder
        with torch.no_grad():
            for enc_input, dec_input, dec_target, *_ in loader:
                enc_input, dec_input, dec_target = (t.to(device) for t in (enc_input, dec_input, dec_target))
                if "mlp" in model_name:
                    dec_target = dec_target.view(dec_target.size(0), -1)
                    enc_output = encoder(enc_input)
                    anchor_latents = self._compute_anchor_latents(encoder, device)
                    rel_space = self._get_relative_space(enc_output, anchor_latents)
                    if rel_space is None: continue
                    if "node" in model_name:
                        rel_space = self.pass_to_ode(model, rel_space, device)
                    if "koopman" in model_name:
                        rel_space = self.pass_to_koopman(model, rel_space).to(device)
                    preds = decoder(rel_space)
                    if "koopman" in model_name: preds = preds.view(preds.size(0),-1)
                elif "transformer" in model_name:
                    enc_output = encoder(enc_input)
                    anchor_latents = self._compute_anchor_latents(encoder, device)
                    rel_space = self._get_relative_space(enc_output, anchor_latents)
                    if rel_space is None: continue
                    if "node" in model_name:
                        rel_space = self.pass_to_ode(model, rel_space, device)
                    if "koopman" in model_name:
                        rel_space = self.pass_to_koopman(model, rel_space).to(device)
                    preds = decoder(dec_input, rel_space)
                elif "oldrnn" in model_name:
                    enc_output, hidden = encoder(enc_input)
                    anchor_latents = self._compute_anchor_latents(encoder, device)
                    rel_space = self._get_relative_space(enc_output, anchor_latents)
                    if rel_space is None: continue
                    preds = decoder(dec_input, hidden, rel_space, None)
                    preds = preds.squeeze(-1)
                else: #autoregrnn
                    enc_output, hidden = encoder(enc_input)
                    anchor_latents = self._compute_anchor_latents(encoder, device)
                    rel_space = self._get_relative_space(enc_output, anchor_latents)
                    if rel_space is None: continue
                    if "node" in model_name:
                        rel_space = self.pass_to_ode(model, rel_space, device)
                        preds, _ = decoder(rel_space, dec_input, hidden, stitching=True)
                    if "koopman" in model_name:
                        rel_space = self.pass_to_koopman(model, rel_space).to(device)
                        preds, _ = decoder(rel_space, dec_input, hidden, stitching=True)
                    if not "koopman" in model_name and not "node" in model_name:
                        preds, _ = decoder(rel_space, dec_input, hidden, teacher_force_prob=0)
                preds = preds.to(device)
                preds, dec_target = self._align_for_loss(preds, dec_target)
                total += criterion(preds, dec_target).item()
        return total / len(loader)
    
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
