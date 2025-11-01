from torch import nn
from src.model.model_interface import EncoderParams, BaseEncoder
import torch.nn.init as init

class MLPEncoderParams(EncoderParams):
    in_dim: int
    hidden_dim: int
    out_dim: int


class MLPEncoder(BaseEncoder):
    def __init__(self, hyperparams: MLPEncoderParams) -> None:
        super().__init__(hyperparams)

        self.hyperparams = hyperparams

        self.net = nn.Sequential(
            nn.Linear(self.hyperparams['in_dim'], self.hyperparams['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.hyperparams['hidden_dim'], self.hyperparams['out_dim'])
        )
        self.reset_parameters()  # Apply Xavier initialization

    def forward(self, x):
        batch_size, input_before, num_dims = x.shape
        x = x.view(batch_size, input_before * num_dims)        
        return self.net(x)
    
    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

