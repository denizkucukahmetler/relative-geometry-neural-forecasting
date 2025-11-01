import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from torchdiffeq import odeint
from typing import TypedDict
#data
from src.config_manager import ConfigManager
from src.data_handlers.data_utils import sequence_collate_fn


class RelativeStitcherHyperparams(TypedDict):
    device: str
    experiment_name: str
    dataset_name: str


class RelativeStitcher:
    """Stitch models trained on relative latent spaces across architectures."""

    def __init__(self, hyperparams: RelativeStitcherHyperparams):
        self.hyperparams = hyperparams
        self.experiment_name = hyperparams["experiment_name"]
        self.dataset_name = hyperparams["dataset_name"]
        self.device = hyperparams["device"]
        #self.encoder_model_name = hyperparams["encoder_model_name"]

    # --- Helper methods ---
    def get_encoder_model(self, experiment_name:str, encoder_model_name:str):
        """Get the specified Encoders Model, from the experiments folder."""
        exp_path = Path("results") / experiment_name
        encoder_path = exp_path / encoder_model_name / "model.pth"
        encoder_model = torch.load(encoder_path,weights_only=False)
        return encoder_model

    def get_decoder_models(self, experiment_name:str, encoder_model_name:str) -> list:
        '''Get all models (for the decoder) from the experiments folder'''
        decoder_models = []
        exp_path = Path("results") / experiment_name
        for model_pth in sorted(exp_path.glob("*/model.pth")):  # only one level deep
            try:
                m = torch.load(model_pth, map_location="cpu",weights_only=False)
                decoder_models.append(m)
            except Exception as e:
                print(f"Skipping {model_pth}: {e}")
        return decoder_models

    def load_dataset(self, batch_size:int, encoder_model:torch.nn) -> DataLoader:
        """Loads the dataset of the encoder model."""
        config_manager = ConfigManager(dataset_name=self.dataset_name, device=self.device)
        model_name = encoder_model.hyperparams["model_name"]
        config = config_manager.get_model_config_by_model_name(model_name[:-1])
        (train_data, val_data, test_data) = config['data_handler_params']['data_handler_class'](
            config['data_handler_params'], self.experiment_name, True
        ).get_datasets([config])[0]

        test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=sequence_collate_fn)
        return test_loader, train_data, test_data
    
    def load_global_anchor_inputs(self, experiment_name: str, device: torch.device):
        """Load the saved global anchors from training to reuse them now."""
        p = Path("results") / experiment_name / "anchor_inputs.pt"
        if not p.exists():
            raise FileNotFoundError(f"Global anchors not found at {p}. "
                                    "Train (or precreate) anchors and save anchor_inputs.pt.")
        # keep on CPU; move to device right before encoding
        return torch.load(p, map_location="cpu",weights_only=False)
    
    def to_transformer_memory(self, rel2d: torch.Tensor, decoder) -> torch.Tensor:
        """Transformer needs a time dimension, which MLP does not give. Expand 2D to 3D tensor."""
        # rel2d: [B, A]; make [B, A, d_model] where d_model is the decoder's model dim
        d_model = decoder.input_linear.out_features  # == 144 in your config
        return rel2d.unsqueeze(-1).expand(-1, -1, d_model)
    
    def to_rnn_memory(self, rel2d: torch.Tensor, seq_len: int) -> torch.Tensor:
        """[B, A] -> [B, A, S] where S = input_before used during RNN training."""
        return rel2d.unsqueeze(-1).expand(-1, -1, seq_len)
    
    def to_rnnautoreg_context(self, rel2d: torch.Tensor, decoder) -> torch.Tensor:
        """Adjust tensor (e.g., of MLP) to RNNAutoReg context"""
        # rel2d: [B, A] -> [B, A, H] where H = decoder.gru.hidden_size
        H = decoder.gru.hidden_size
        return rel2d.unsqueeze(-1).expand(-1, -1, H)
    
    def init_rnn_hidden(self,decoder, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a zero init hidden with the right shape [L, B, H] from decoder.gru."""
        gru = decoder.gru  # RNNDecoder has self.gru in your code
        return torch.zeros(gru.num_layers, batch_size, gru.hidden_size, device=device)
    
    def pass_to_ode(self,model, rel_space, device):
        """Pass the encoder output to the ODE solver, for the NODE models."""
        model = model.to(device)
        dt = model.hyperparams['data_handler_params']['dt']
        t = torch.arange(0, dt * model.hyperparams['prediction_length'], dt).to(device)
        ode_func = model.ode_func
        if rel_space.dim() == 3: rel_space = rel_space[:, :, -1] #take last time step
        odeout = odeint(
            ode_func,
            rel_space,
            t,
            method='rk4',  # or try 'rk4', 'adams', 'bdf'
            rtol=1e-4,
            atol=1e-6
        )
        if not("mlp" in model.hyperparams.get("model_name","")):
            odeout = odeout.permute(1, 0, 2).contiguous()
        return odeout
    
    def pass_to_koopman(self, model, rel_space):
        """Pass the encoder output to the Koopman propagator, for the Koopman models."""
        K = model.K.to(self.device)
        if rel_space.dim() == 3: rel_space = rel_space[:, :, -1] #take last time step
        encoded_pred = [rel_space]
        for _ in range(model.hyperparams['prediction_length']-1):
            rel_space = torch.matmul(rel_space, K.T)
            encoded_pred.append(rel_space)
        rel_space = torch.stack(encoded_pred, dim=1)  # Shape: (batch, num_steps+1, latent_dim)
        return rel_space
    
    def mlp_encoder_eval(self, enc_input, encoder, anchor_latents, decoder_model, decoder, dec_input, dec_target) -> torch.tensor:
        """Stitching across models for mlp encoder."""
        decoder_model = decoder_model.to(self.device)
        enc_output = encoder(enc_input)
        rel_space = F.cosine_similarity(enc_output.unsqueeze(1),
                                                    anchor_latents.unsqueeze(0), dim=-1).to(self.device)
        if "node" in decoder_model.hyperparams["model_name"]:
            rel_space = self.pass_to_ode(decoder_model, rel_space, self.device)
        if "koopman" in decoder_model.hyperparams["model_name"]:
            rel_space = self.pass_to_koopman(decoder_model, rel_space)
        if "mlp" in decoder_model.hyperparams["model_name"]:
            preds = decoder(rel_space)
            if "koopman" in decoder_model.hyperparams["model_name"]:
                preds = preds.view(preds.size(0),-1)
            dec_target = dec_target.view(dec_target.size(0), -1)
        elif "transformer" in decoder_model.hyperparams["model_name"]:
            if rel_space.dim() == 2:
                rel_space = self.to_transformer_memory(rel_space, decoder)
            preds = decoder(dec_input, rel_space)
        else:  # RNN family
            if "oldrnn" in decoder_model.hyperparams["model_name"]:
                input_before = decoder_model.hyperparams["input_before"]
                hidden = self.to_rnn_memory(rel_space, input_before)
                init_hidden = self.init_rnn_hidden(decoder, enc_input.size(0), self.device)
                preds = decoder(dec_input, init_hidden, hidden, None).squeeze(-1)
            elif "rnnautoreg" in decoder_model.hyperparams["model_name"]:
                init_hidden = self.init_rnn_hidden(decoder, enc_input.size(0), self.device)
                if "koopman" in decoder_model.hyperparams["model_name"] or "node" in decoder_model.hyperparams["model_name"]:
                    preds, _ = decoder(rel_space, dec_input, init_hidden, stitching=True)
                else:
                    hidden = self.to_rnnautoreg_context(rel_space, decoder)
                    preds, _ = decoder(hidden, dec_input, init_hidden)
            else:
                return None, None
        preds = preds.to(self.device)
        return preds, dec_target
    
    def rnn_encoder_eval(self, enc_input, encoder, anchor_latents, decoder_model, decoder, dec_input, dec_target):
        """Stitching across models for RNN/RNNAutoReg Encoder"""
        decoder_model = decoder_model.to(self.device)
        enc_output, hidden = encoder(enc_input)
        rel_space = F.cosine_similarity(enc_output.unsqueeze(1),
                                                    anchor_latents.unsqueeze(0), dim=-1)
        if "mlp" in decoder_model.hyperparams["model_name"]:
            rel_space = rel_space[:, :, 0]
        if "koopman" in decoder_model.hyperparams["model_name"]:
            rel_space = self.pass_to_koopman(decoder_model, rel_space)
        if "node" in decoder_model.hyperparams["model_name"]:
            rel_space = self.pass_to_ode(decoder_model, rel_space, self.device)

        if "mlp" in decoder_model.hyperparams["model_name"]:
            preds = decoder(rel_space)
            if "koopman" in decoder_model.hyperparams["model_name"]:
                preds = preds.view(preds.size(0), -1) 
            dec_target = dec_target.view(dec_target.size(0), -1)
        elif "transformer" in decoder_model.hyperparams["model_name"]:
            preds = decoder(dec_input, rel_space)
        else: 
            if "oldrnn" in decoder_model.hyperparams["model_name"]:
                preds = decoder(dec_input, hidden, rel_space, None).squeeze(-1)
            elif "rnnautoreg" in decoder_model.hyperparams["model_name"]:
                if "koopman" in decoder_model.hyperparams["model_name"] or "node" in decoder_model.hyperparams["model_name"]:
                    preds, _ = decoder(rel_space, dec_input, hidden, stitching=True)
                else:
                    preds, _ = decoder(rel_space, dec_input, hidden)
            else:
                return None, None
        preds = preds.to(self.device)
        return preds, dec_target
    
    def transformer_encoder_eval(self,enc_input, encoder, anchor_latents, decoder_model, decoder, dec_input, dec_target):
        """Stiching across models for an Transformer Encoder"""
        decoder_model = decoder_model.to(self.device)
        enc_output = encoder(enc_input)
        rel_space = F.cosine_similarity(enc_output.unsqueeze(1),
                                                    anchor_latents.unsqueeze(0), dim=-1)
        if "koopman" in decoder_model.hyperparams["model_name"]:
            rel_space = self.pass_to_koopman(decoder_model, rel_space)
        if "mlp" in decoder_model.hyperparams["model_name"] and not ("koopman" in decoder_model.hyperparams["model_name"]):
            rel_space = rel_space[:, :, 0]
        if "node" in decoder_model.hyperparams["model_name"]:
            rel_space = self.pass_to_ode(decoder_model, rel_space, self.device)
        if "mlp" in decoder_model.hyperparams["model_name"]:
            preds = decoder(rel_space)
            if "koopman" in decoder_model.hyperparams["model_name"]:
                preds = preds.view(preds.size(0), -1)    
            dec_target = dec_target.view(dec_target.size(0), -1)
        elif "transformer" in decoder_model.hyperparams["model_name"]:
            preds = decoder(dec_input, rel_space)
        else:  # RNN family
            if "oldrnn" in decoder_model.hyperparams["model_name"]:
                init_hidden = self.init_rnn_hidden(decoder, enc_input.size(0), self.device)
                preds = decoder(dec_input, init_hidden, rel_space, None).squeeze(-1)
            elif "rnnautoreg" in decoder_model.hyperparams["model_name"]:
                init_hidden = self.init_rnn_hidden(decoder, enc_input.size(0), self.device)
                if "koopman" in decoder_model.hyperparams["model_name"] or "node" in decoder_model.hyperparams["model_name"]:
                    preds, _ = decoder(rel_space, dec_input, init_hidden, stitching=True)
                else:
                    preds, _ = decoder(rel_space, dec_input, init_hidden)
            else:
                return None, None
        preds = preds.to(self.device)
        return preds, dec_target
    
    def stitch(self, encoder_model_name:str, batch_size:int):
        """pipeline for stitching"""
        # load encoder model
        encoder_model = self.get_encoder_model(self.experiment_name, encoder_model_name)
        encoder_model = encoder_model.to(self.device).eval()
        # load all decoder models (exclude the encoder itself)
        decoder_models = [m for m in self.get_decoder_models(self.experiment_name, encoder_model_name)]

        # dataset (for test loader only)
        test_loader, train_data, test_data = self.load_dataset(batch_size, encoder_model)

        # >>> USE SAVED GLOBAL ANCHORS, NOT RESAMPLED <<<
        anchor_inputs = self.load_global_anchor_inputs(self.experiment_name, self.device)  # CPU tensor
        encoder = encoder_model.encoder.to(self.device).eval()
        with torch.no_grad():
            anchor_latents = encoder(anchor_inputs.to(self.device))
            if isinstance(anchor_latents, tuple):
                anchor_latents = anchor_latents[0].detach() #RNN case

        results = []
        criterion = nn.MSELoss()
        for decoder_model in decoder_models:
            print(f"Stitching Decoder: {decoder_model.hyperparams['model_name']}.")
            decoder = decoder_model.decoder.to(self.device).eval()
            total_loss = 0.0
            with torch.no_grad():
                for enc_input, dec_input, dec_target, *_ in test_loader:
                    enc_input, dec_input, dec_target = (t.to(self.device) for t in (enc_input, dec_input, dec_target))
                    if "mlp" in encoder_model.hyperparams['model_name']:
                        preds, dec_target = self.mlp_encoder_eval(enc_input, encoder, anchor_latents, decoder_model, decoder, dec_input, dec_target)
                    elif "rnn" in encoder_model.hyperparams['model_name']:
                        preds, dec_target = self.rnn_encoder_eval(enc_input, encoder, anchor_latents, decoder_model, decoder, dec_input, dec_target)
                    elif "transformer" in encoder_model.hyperparams['model_name']:
                        preds, dec_target = self.transformer_encoder_eval(enc_input, encoder, anchor_latents, decoder_model, decoder, dec_input, dec_target)
                    if preds is None: continue #unknkown model or none model
                    total_loss += criterion(preds, dec_target).item()

            total_loss /= max(1, len(test_loader))
            print("loss (mse):", total_loss)
            results.append({
                'encoder': encoder_model.hyperparams["model_name"],
                'decoder': decoder_model.hyperparams["model_name"],
                'mse_loss': total_loss
            })
        return results
    
    def save_results(self, results:list,encoder_model_name:str)-> None:
        """save the stitching results in a csv file"""
        exp_path = Path("results") / self.experiment_name / "stitching/"
        exp_path.mkdir(parents=True, exist_ok=True)
        out_csv = exp_path / f"{encoder_model_name}_stitching_results.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")