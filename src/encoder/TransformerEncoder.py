import torch.nn as nn
from src.model.model_interface import BaseEncoder, EncoderParams
from src.utils.PositionalEncoding import PositionalEncoding
import torch.nn.init as init



class TransformerEncoderParams(EncoderParams):
    input_dim: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float


class TransformerEncoder(BaseEncoder):
    def __init__(self, hyperparams: TransformerEncoderParams):
        super().__init__(hyperparams)
        
        self.hyperparams = hyperparams
        
        input_dim = self.hyperparams['input_dim']
        d_model = self.hyperparams['d_model']
        n_heads = self.hyperparams['n_heads']
        n_layers = self.hyperparams['n_layers']
        dropout = self.hyperparams['dropout']

        self.input_linear = nn.Linear(input_dim, d_model)  # Project input to d_model
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.reset_parameters()


    def forward(self, src):
        src = self.input_linear(src)  
        src = self.positional_encoding(src) 
        output = self.encoder(src) 
        return output

    
    def reset_parameters(self):
        init.xavier_uniform_(self.input_linear.weight)
        if self.input_linear.bias is not None:
            init.zeros_(self.input_linear.bias)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
