import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from src.model.model_interface import BaseDecoder, DecoderParams
from src.model.seq2seq import RnnModel
import torch.nn.init as init

def layer_init(layer, w_scale=1.0):
    nn.init.kaiming_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0.)
    return layer


class OLDRNNDecoderParams(DecoderParams):
    hidden_size: int
    dec_feature_size: int
    dropout: float
    dec_target_size: int
    target_size: int
    dist_size: int
    target_indices: list
    num_gru_layers: int
    num_anchors: int | None = None
    rel_norm: str | None = "batch"

class RnnDecoder(BaseDecoder):
    def __init__(self, hyperparams: OLDRNNDecoderParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams
        self.hidden_size = self.hyperparams['hidden_size']
        self.dec_feature_size = self.hyperparams['dec_feature_size']
        self.dropout =self.hyperparams['dropout']
        self.dec_target_size = self.hyperparams['dec_target_size']
        self.target_size = self.hyperparams['dec_feature_size']
        self.dist_size = self.hyperparams['dist_size']
        self.target_indices = self.hyperparams['target_indices']
        self.num_gru_layers = self.hyperparams['num_gru_layers']

        self.gru = nn.GRU(self.dec_feature_size, self.hidden_size, self.num_gru_layers, batch_first=True, dropout=self.dropout)
        self.out = layer_init(nn.Linear(self.hidden_size + self.dec_feature_size, self.dec_target_size * self.dist_size))
        self.reset_parameters()

        if self.hyperparams.get('rel_norm', '') == "batch" and self.hyperparams.get('num_anchors') is not None:
            self.rel_norm = nn.BatchNorm1d(self.hyperparams['num_anchors'], affine=True, track_running_stats=True)
        else:
            self.rel_norm = nn.Identity()

    def _apply_rel_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BatchNorm1d across the 'anchors' dimension if present.
        Supports [B, A, L] directly or [B, L, A] via transpose."""
        if isinstance(self.rel_norm, nn.BatchNorm1d) and isinstance(x, torch.Tensor) and x.dim() == 3:
            a = self.hyperparams.get('num_anchors', None)
            if a is None:
                return x
            if x.size(1) == a:  # [B, A, L]
                return self.rel_norm(x)
            if x.size(2) == a:  # [B, L, A] -> transpose to [B, A, L]
                return self.rel_norm(x.transpose(1, 2)).transpose(1, 2)
        return x


    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        # inputs: (batch size, 1, num dec features)
        # hidden: (num gru layers, batch size, hidden size)
        
        output, hidden = self.gru(inputs, hidden)
        output = self.out(torch.cat((output, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size, self.dist_size)
        
        # output: (batch size, 1, num targets, num dist params)
        # hidden: (num gru layers, batch size, hidden size)
        return output, hidden

    def forward(self, inputs, hidden, enc_outputs, teacher_force_prob=None):
        # inputs: (batch size, output seq length, num dec features)
        # hidden: (num gru layers, batch size, hidden dim), ie the last hidden state
        # enc_outputs: (batch size, input seq len, hidden size)
        inputs = self._apply_rel_norm(inputs)
        enc_outputs = self._apply_rel_norm(enc_outputs)
        batch_size, dec_output_seq_length, _ = inputs.shape
        
        # Store decoder outputs
        # outputs: (batch size, output seq len, num targets, num dist params)
        outputs = torch.zeros(batch_size, dec_output_seq_length, self.target_size, self.dist_size, dtype=torch.float)

        # curr_input: (batch size, 1, num dec features)
        curr_input = inputs[:, 0:1, :]
        
        for t in range(dec_output_seq_length):
            # dec_output: (batch size, 1, num targets, num dist params)
            # hidden: (num gru layers, batch size, hidden size)
            dec_output, hidden = self.run_single_recurrent_step(curr_input, hidden, enc_outputs)
            # Save prediction
            outputs[:, t:t+1, :, :] = dec_output
            # dec_output: (batch size, 1, num targets)
            dec_output = RnnModel.sample_from_output(dec_output)
            
            # If teacher forcing, use target from this timestep as next input o.w. use prediction
            teacher_force = random.random() < teacher_force_prob if teacher_force_prob is not None else False
            
            curr_input = inputs[:, t:t+1, :].clone()
            if not teacher_force:
                curr_input[:, :, self.target_indices] = dec_output
        return outputs
    
    def reset_parameters(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                init.zeros_(param.data)
        init.kaiming_uniform_(self.out.weight.data)
        if self.out.bias is not None:
            init.constant_(self.out.bias.data, 0.)