from torch import nn
import torch.nn.init as init
from src.model.model_interface import BaseDecoder, DecoderParams


class MLPDecoderParams(DecoderParams):
    in_dim: int
    hidden_dim: int
    out_dim: int
    num_anchors: int | None = None
    rel_norm: str | None = "batch"

class MLPDecoder(BaseDecoder):
    def __init__(self, hyperparams: MLPDecoderParams):
        super().__init__(hyperparams)
        
        self.hyperparams = hyperparams

        #max
        if self.hyperparams.get('rel_norm', '') == "batch" and self.hyperparams.get('num_anchors') is not None:
            self.rel_norm = nn.BatchNorm1d(self.hyperparams['num_anchors'], affine=True, track_running_stats=True)
        else:
            self.rel_norm = nn.Identity()
        
        self.net = nn.Sequential(
            nn.Linear(self.hyperparams['in_dim'], self.hyperparams['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.hyperparams['hidden_dim'], self.hyperparams['out_dim'])
        )
        self.reset_parameters()
    def forward(self, y):
        y = self.rel_norm(y) #max
        return self.net(y)

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)