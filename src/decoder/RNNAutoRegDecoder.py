from torch import nn
import torch
import torch.nn.init as init
from src.model.model_interface import BaseDecoder, DecoderParams


class RNNAutoRegDecoderParams(DecoderParams):
    input_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    output_dim: int
    teacher_force: float


class RNNAutoRegDecoder(BaseDecoder):
    def __init__(self, hyperparams: RNNAutoRegDecoderParams) -> None:
        super().__init__(hyperparams)

        self.hyperparams = hyperparams
        self.teacher_force_prob = float(self.hyperparams.get('teacher_force', 0.0))
        
        self.fc_input = nn.Linear(self.hyperparams['input_dim'], self.hyperparams['hidden_dim']) # Transform dec_inputs to match GRU input dimension
        self.gru = nn.GRU(input_size=self.hyperparams['hidden_dim'], 
                          hidden_size=self.hyperparams['hidden_dim'],
                          num_layers=self.hyperparams['num_layers'], 
                          batch_first=True,
                          dropout=self.hyperparams['dropout'])
        self.fc = nn.Linear(self.hyperparams['hidden_dim'], self.hyperparams['output_dim'])
        self.dropout = nn.Dropout(self.hyperparams['dropout'])
        self.reset_parameters()


    def forward(self, enc_outputs, dec_inputs, hidden, teacher_force_prob=None): 
        if teacher_force_prob is None:
            teacher_force_prob = self.teacher_force_prob  
            
        _, seq_length, _ = dec_inputs.shape
        dec_inputs = self.fc_input(dec_inputs) # Transform dec_inputs to match GRU input dimension
                      
        outputs = []
        output = enc_outputs[:, -1, :].unsqueeze(1)  # Start with last token
        
        for t in range(seq_length):
            output, hidden = self.gru(output, hidden)  # Step through GRU
            output = self.fc(output)
            output = self.dropout(output)
            outputs.append(output)
            
            # Teacher forcing or autoregression, only if training mode
            if self.training:
                teacher_force = torch.rand(1).item() < teacher_force_prob
                if teacher_force:
                    output = dec_inputs[:, t, :].unsqueeze(1)  # Use ground truth
                else:
                    output = self.fc_input(output.squeeze(1)).unsqueeze(1)  # Use model prediction
            else:
                output = self.fc_input(output.squeeze(1)).unsqueeze(1)  # Use model prediction
            
        outputs = torch.cat(outputs,dim=1)
        
        return outputs, hidden
    
    def reset_parameters(self):
        init.xavier_uniform_(self.fc_input.weight)
        if self.fc_input.bias is not None:
            init.zeros_(self.fc_input.bias)
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