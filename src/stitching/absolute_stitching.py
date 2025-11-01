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


class AbsoluteStitcherHyperparams(TypedDict):
    device: str
    experiment_name: str
    dataset_name: str


class AbsoluteStitcher:
    """Stitch models trained on absolute latent spaces across architectures."""

    def __init__(self, hyperparams: AbsoluteStitcherHyperparams):
        self.hyperparams = hyperparams
        self.experiment_name = hyperparams["experiment_name"]
        self.dataset_name = hyperparams["dataset_name"]
        self.device = hyperparams["device"]

    def get_encoder_model(self, experiment_name:str, encoder_model_name:str):
        """Get the specified Encoders Model, from the experiments folder."""
        exp_path = Path("results") / experiment_name
        encoder_path = exp_path / encoder_model_name / "model.pth"
        encoder_model = torch.load(encoder_path, map_location="cuda:0", weights_only=False)
        return encoder_model

    def get_decoder_models(self, experiment_name:str, encoder_model_name:str) -> list:
        '''Get all models (for the decoder) from the experiments folder'''
        decoder_models = []
        exp_path = Path("results") / experiment_name
        for model_pth in sorted(exp_path.glob("*/model.pth")):  
            try:
                m = torch.load(model_pth, map_location="cpu", weights_only=False)
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
    
    def to_transformer_memory(self, rel2d: torch.Tensor, decoder) -> torch.Tensor:
        """Transformer needs a time dimension, which MLP does not give. Expand 2D to 3D tensor."""
        d_model = decoder.input_linear.out_features  
        return rel2d.unsqueeze(-1).expand(-1, -1, d_model)
    
    def to_rnn_memory(self, rel2d: torch.Tensor, seq_len: int) -> torch.Tensor:
        """[B, A] -> [B, A, S] where S = input_before used during RNN training."""
        return rel2d.unsqueeze(-1).expand(-1, -1, seq_len)
    
    def to_rnnautoreg_context(self, rel2d: torch.Tensor, decoder) -> torch.Tensor:
        """Adjust tensor (e.g., of MLP) to RNNAutoReg context"""
        H = decoder.gru.hidden_size
        return rel2d.unsqueeze(-1).expand(-1, -1, H)
    
    def init_rnn_hidden(self,decoder, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a zero init hidden with the right shape [L, B, H] from decoder.gru."""
        gru = decoder.gru  
        return torch.zeros(gru.num_layers, batch_size, gru.hidden_size, device=device)
    
    def pass_to_ode(self,model, rel_space, device):
        """Pass the encoder output to the ODE solver, for the NODE models."""
        model = model.to(device)
        dt = model.hyperparams['data_handler_params']['dt']
        t = torch.arange(0, dt * model.hyperparams['prediction_length'], dt).to(device)
        ode_func = model.ode_func
        if rel_space.dim() == 3: rel_space = rel_space[:, -1,:] 
        odeout = odeint(
            ode_func,
            rel_space,
            t,
            method='rk45'
            rtol=1e-4,
            atol=1e-6
        )
        if not("mlp" in model.hyperparams.get("model_name","")):
            odeout = odeout.permute(1, 0, 2).contiguous()
        return odeout
    
    def pass_to_koopman(self, model, rel_space):
        """Pass the encoder output to the Koopman propagator, for the Koopman models."""
        K = model.K.to(self.device)
        if rel_space.dim() == 3: rel_space = rel_space[:, -1,:] 
        encoded_pred = [rel_space]
        for _ in range(model.hyperparams['prediction_length']-1):
            rel_space = torch.matmul(rel_space, K.T)
            encoded_pred.append(rel_space)
        rel_space = torch.stack(encoded_pred, dim=1)  
        return rel_space
    
    def mlp_encoder_eval(self, enc_input, encoder, decoder_model, decoder, dec_input, dec_target) -> torch.tensor:
        """Stitching across models for mlp encoder."""
        decoder_model = decoder_model.to(self.device)
        decoder_name = decoder_model.hyperparams["model_name"]
        abs_space = encoder(enc_input)
        if "node" in decoder_name:
            abs_space = self.pass_to_ode(decoder_model, abs_space, self.device)
            if "mlp" in decoder_name:
                abs_space = abs_space.transpose(0,1)
        if "koopman" in decoder_name:
            abs_space = self.pass_to_koopman(decoder_model, abs_space)
        if "mlp" in decoder_name:
            preds = decoder(abs_space)
            if "koopman" in decoder_name or "node" in decoder_name:
                preds = preds.view(preds.size(0),-1)
            dec_target = dec_target.view(dec_target.size(0), -1)
        

        elif "transformer" in decoder_model.hyperparams["model_name"]:
            if abs_space.dim() == 2:
                abs_space = self.to_transformer_memory(abs_space, decoder)
            preds = decoder(dec_input, abs_space)

        else:  
            if "oldrnn" in decoder_model.hyperparams["model_name"]:
                input_before = decoder_model.hyperparams["input_before"]
                hidden = self.to_rnn_memory(abs_space, input_before)
                init_hidden = self.init_rnn_hidden(decoder, enc_input.size(0), self.device)
                preds = decoder(dec_input, init_hidden, hidden, None).squeeze(-1)
            elif "rnnautoreg" in decoder_model.hyperparams["model_name"]:
                init_hidden = self.init_rnn_hidden(decoder, enc_input.size(0), self.device)
                if "koopman" in decoder_model.hyperparams["model_name"] or "node" in decoder_model.hyperparams["model_name"]:
                    preds, _ = decoder(abs_space, dec_input, init_hidden, stitching=False)
                else:
                    hidden = self.to_rnnautoreg_context(abs_space, decoder)
                    preds, _ = decoder(hidden, dec_input, init_hidden)
            else:
                return None, None
        preds = preds.to(self.device)
        return preds, dec_target
    
    def rnn_encoder_eval(self, enc_input, encoder, decoder_model, decoder, dec_input, dec_target):
        """Stitching across models for RNN/RNNAutoReg Encoder"""
        decoder_model = decoder_model.to(self.device)
        abs_space, hidden = encoder(enc_input)
        decoder_name = decoder_model.hyperparams["model_name"]

        if "mlp" in decoder_name:
            abs_space = abs_space[:,-1,:]
        if "koopman" in decoder_name:
            abs_space = self.pass_to_koopman(decoder_model, abs_space)
        if "node" in decoder_name:
            abs_space = self.pass_to_ode(decoder_model, abs_space, self.device)

        if "mlp" in decoder_name:
            preds = decoder(abs_space)
            if "koopman" in decoder_name:
                preds = preds.view(preds.size(0), -1) 
            if "node" in decoder_name:
                preds = preds.transpose(0,1) 
                preds = preds.reshape(preds.shape[0], -1) 
            dec_target = dec_target.view(dec_target.size(0), -1)
        elif "transformer" in decoder_name:
            preds = decoder(dec_input, abs_space)
        else:  
            if "oldrnn" in decoder_name:
                preds = decoder(dec_input, hidden, abs_space, None).squeeze(-1)
            elif "rnnautoreg" in decoder_name:
                if "koopman" in decoder_name or "node" in decoder_name:
                    preds, _ = decoder(abs_space, dec_input, hidden, stitching=False)
                else:
                    preds, _ = decoder(abs_space, dec_input, hidden)
            else:
                return None, None
        preds = preds.to(self.device)
        return preds, dec_target
    
    def transformer_encoder_eval(self,enc_input, encoder, decoder_model, decoder, dec_input, dec_target):
        """Stiching across models for an Transformer Encoder"""
        decoder_model = decoder_model.to(self.device)
        abs_space = encoder(enc_input)
        decoder_name = decoder_model.hyperparams["model_name"]

        if "koopman" in decoder_name:
            abs_space = self.pass_to_koopman(decoder_model, abs_space)
        if "mlp" in decoder_name and not ("koopman" in decoder_name):
            abs_space = abs_space[:, -1,:] 
        if "node" in decoder_name:
            abs_space = self.pass_to_ode(decoder_model, abs_space, self.device)
        if "mlp" in decoder_name:
            preds = decoder(abs_space)
            if "koopman" in decoder_name:
                preds = preds.view(preds.size(0), -1)
            if "node" in decoder_name:
                preds = preds.transpose(0,1) 
                preds = preds.reshape(preds.shape[0], -1)
            dec_target = dec_target.view(dec_target.size(0), -1)
        elif "transformer" in decoder_name:
            preds = decoder(dec_input, abs_space)
        else:  
            if "oldrnn" in decoder_name:
                init_hidden = self.init_rnn_hidden(decoder, enc_input.size(0), self.device)
                preds = decoder(dec_input, init_hidden, abs_space, None).squeeze(-1)
            elif "rnnautoreg" in decoder_name:
                init_hidden = self.init_rnn_hidden(decoder, enc_input.size(0), self.device)
                if "koopman" in decoder_name or "node" in decoder_name:
                    preds, _ = decoder(abs_space, dec_input, init_hidden, stitching=False)
                else:
                    preds, _ = decoder(abs_space, dec_input, init_hidden)
            else:
                return None, None
        preds = preds.to(self.device)
        return preds, dec_target
    
    def stitch(self, encoder_model_name:str, batch_size:int):
        """pipeline for stitching"""
        
        encoder_model = self.get_encoder_model(self.experiment_name, encoder_model_name)
        encoder_model = encoder_model.to(self.device).eval()
        decoder_models = [m for m in self.get_decoder_models(self.experiment_name, encoder_model_name)]

        test_loader, train_data, test_data = self.load_dataset(batch_size, encoder_model)

        encoder = encoder_model.encoder.to(self.device).eval()
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
                        preds, dec_target = self.mlp_encoder_eval(enc_input, encoder, decoder_model, decoder, dec_input, dec_target)
                    elif "rnn" in encoder_model.hyperparams['model_name']:
                        preds, dec_target = self.rnn_encoder_eval(enc_input, encoder, decoder_model, decoder, dec_input, dec_target)
                    elif "transformer" in encoder_model.hyperparams['model_name']:
                        preds, dec_target = self.transformer_encoder_eval(enc_input, encoder, decoder_model, decoder, dec_input, dec_target)
                    if preds is None: continue 
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