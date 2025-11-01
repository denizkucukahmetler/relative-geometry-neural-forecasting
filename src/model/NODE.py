from typing import TypedDict

import torch
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint


from torch import nn

from src.decoder.RNNAutoRegDecoder import RNNAutoRegDecoder
from src.encoder.RNNAutoRegEncoder import RNNAutoRegEncoder
from src.model.model_params import ModelParams
from src.decoder.MLPDecoder import MLPDecoder
from src.decoder.RNNDecoder import RNNDecoder
from src.decoder.TransformerDecoder import TransformerDecoder
from src.encoder.MLPEncoder import MLPEncoder
from src.encoder.RNNEncoder import RNNEncoder
from src.encoder.TransformerEncoder import TransformerEncoder
from src.model.model_interface import BaseModel


class ODEFunctionParams(TypedDict):
    in_out_dim: int
    hidden_dim: int


class ODEFunction(nn.Module):
    def __init__(self, hyperparams: ODEFunctionParams):
        super().__init__()
        self.hyperparams = hyperparams
        self.net = nn.Sequential(
            nn.Linear(self.hyperparams['in_out_dim'], self.hyperparams['hidden_dim']),
            nn.Sigmoid(),
            nn.Linear(self.hyperparams['hidden_dim'], self.hyperparams['hidden_dim']),
            nn.Sigmoid(),
            nn.Linear(self.hyperparams['hidden_dim'], self.hyperparams['hidden_dim']),
            nn.Sigmoid(),
            nn.Linear(self.hyperparams['hidden_dim'], self.hyperparams['in_out_dim'])
        )

    def forward(self, t, y):

        return self.net(y)


class NODEParams(ModelParams):
    ode_params: ODEFunctionParams


class NODE(BaseModel):
    def __init__(self, hyperparams: NODEParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams

        self.encoder = self.hyperparams['encoder_class'](self.hyperparams['encoder_hyperparams'])
        self.ode_func = ODEFunction(self.hyperparams['ode_params'])
        self.decoder = self.hyperparams['decoder_class'](self.hyperparams['decoder_hyperparams'])
        self.device = self.hyperparams['data_handler_params']['device']

    def forward(self, enc_inputs, dec_inputs):  # (B, T, C)
        dt = self.hyperparams['data_handler_params']['dt']
        t = torch.arange(0, dt * self.hyperparams['prediction_length'], dt).to(self.device)
        hidden = None
        if self.hyperparams['encoder_class'] == MLPEncoder:
            encoded = self.encoder(enc_inputs)
        elif self.hyperparams['encoder_class'] == RNNEncoder:
            encoded, hidden = self.encoder(enc_inputs)
            encoded = encoded[:, -1, :]  # Take last token
        elif self.hyperparams['encoder_class'] == RNNAutoRegEncoder:
            encoded, hidden = self.encoder(enc_inputs)
            encoded = encoded[:, -1, :]  # Take last token
        elif self.hyperparams['encoder_class'] == TransformerEncoder:
            encoded = self.encoder(enc_inputs)[:, -1, :]  # Take last token
        else:
            raise Exception(f'Unknown encoder type: {self.hyperparams["encoder_class"]}')
        
        odeouts = odeint(
            self.ode_func,
            encoded,
            t,
            method='rk4',  # or try 'rk4', 'adams', 'bdf'
            rtol=1e-4,
            atol=1e-6
        ).permute(1, 0, 2)
        
        

        if self.hyperparams['decoder_class'] == MLPDecoder:
            decoded = self.decoder(odeouts)
        elif self.hyperparams['decoder_class'] == RNNDecoder:
            decoded, hidden = self.decoder(odeouts, hidden)
        elif self.hyperparams['decoder_class'] == RNNAutoRegDecoder:
            decoded, hidden = self.decoder(odeouts, dec_inputs, hidden)
        elif self.hyperparams['decoder_class'] == TransformerDecoder:
            decoded = self.decoder(dec_inputs, odeouts)
        else:
            raise Exception(f'Unknown decoder type: {self.hyperparams["decoder_class"]}')

        return decoded
