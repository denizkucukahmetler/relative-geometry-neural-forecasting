from torch import nn
import torch.nn.init as init
from src.model.model_interface import EncoderParams, BaseEncoder


class RNNAutoRegEncoderParams(EncoderParams):
    input_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float


class RNNAutoRegEncoder(BaseEncoder):
    def __init__(self, hyperparams: RNNAutoRegEncoderParams) -> None:
        super().__init__(hyperparams)

        self.hyperparams = hyperparams
        self.gru = nn.GRU(input_size=self.hyperparams['input_dim'], hidden_size=self.hyperparams['hidden_dim'],
                          num_layers=self.hyperparams['num_layers'], batch_first=True,
                          dropout=self.hyperparams['dropout'])
        self.reset_parameters()
        self.hidden_states = []

    def forward(self, x):
        return self.gru(x)  

    def reset_parameters(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                init.zeros_(param.data)