import torch # Make sure torch is imported
import torch.nn as nn
import torch.nn.init as init
from src.model.model_interface import BaseDecoder, DecoderParams
from src.utils.PositionalEncoding import PositionalEncoding 

class TransformerDecoderParams(DecoderParams):
    input_dim: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float


class TransformerDecoder(BaseDecoder):
    def __init__(self, hyperparams: TransformerDecoderParams):
        super().__init__(hyperparams)

        self.hyperparams = hyperparams

        input_dim = self.hyperparams['input_dim']
        d_model = self.hyperparams['d_model']
        n_heads = self.hyperparams['n_heads']
        n_layers = self.hyperparams['n_layers']
        n_dropout = self.hyperparams['dropout']

        self.input_linear = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        # Ensure batch_first=True is consistent with your data (B, S, F)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=n_dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_linear = nn.Linear(d_model, input_dim)
        self.reset_parameters()

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a square mask for preventing attention to subsequent positions."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt_seq_len = tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        tgt_processed = self.input_linear(tgt)       # [batch_size, tgt_seq_len, d_model]
        tgt_processed = self.positional_encoding(tgt_processed) # Add positional encoding
        output = self.decoder(tgt_processed, memory, tgt_mask=tgt_mask)
        
        output = self.output_linear(output)          # Project back to input dimensions
        return output

    def reset_parameters(self):
        init.xavier_uniform_(self.input_linear.weight)
        if self.input_linear.bias is not None:
            init.zeros_(self.input_linear.bias)
        init.xavier_uniform_(self.output_linear.weight)
        if self.output_linear.bias is not None:
            init.zeros_(self.output_linear.bias)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)