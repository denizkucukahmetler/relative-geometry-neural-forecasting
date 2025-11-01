from torch import nn
import torch.nn.init as init
from src.model.model_interface import BaseDecoder, DecoderParams


class KoopmanDecoderParams(DecoderParams):
    in_dim: int
    hidden_dim: int
    out_dim: int

class KoopmanDecoder(BaseDecoder):
    def __init__(self, hyperparams: KoopmanDecoderParams):
        super().__init__(hyperparams)
        
        self.hyperparams = hyperparams
        
        self.net = nn.Sequential(
            nn.Linear(self.hyperparams['in_dim'], self.hyperparams['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.hyperparams['hidden_dim'], self.hyperparams['out_dim'])
        )
        self.reset_parameters()


    def forward(self, y):
        return self.net(y)

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)