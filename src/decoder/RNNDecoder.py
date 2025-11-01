from torch import nn
import torch.nn.init as init
from src.model.model_interface import BaseDecoder, DecoderParams


class RNNDecoderParams(DecoderParams):
    input_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    output_dim: int


class RNNDecoder(BaseDecoder):
    def __init__(self, hyperparams: RNNDecoderParams) -> None:
        super().__init__(hyperparams)

        self.hyperparams = hyperparams
        self.gru = nn.GRU(input_size=self.hyperparams['input_dim'], hidden_size=self.hyperparams['hidden_dim'],
                          num_layers=self.hyperparams['num_layers'], batch_first=True,
                          dropout=self.hyperparams['dropout'])
        self.fc = nn.Linear(in_features=self.hyperparams['hidden_dim'], out_features=self.hyperparams['output_dim'])
        self.reset_parameters()


    def forward(self, y, hidden):
        output, hidden = self.gru(y, hidden)
        output = self.fc(output)
        return output, hidden

    def reset_parameters(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                init.zeros_(param.data)
        init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)