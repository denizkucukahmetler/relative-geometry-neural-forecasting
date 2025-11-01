from typing import Type, Any
import torch
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



class KoopmanParams(ModelParams):
    pass

class Koopman(BaseModel):
    def __init__(self, hyperparams: KoopmanParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams
        self.encoder = self.hyperparams['encoder_class'](self.hyperparams['encoder_hyperparams'])
        self.decoder = self.hyperparams['decoder_class'](self.hyperparams['decoder_hyperparams'])
        self.device = self.hyperparams['data_handler_params']['device']
        
        if self.hyperparams['train_params']['stitching']:
            if self.hyperparams['encoder_class'] == MLPEncoder:
                latent_dim = self.hyperparams['train_params']['stitching_anchor_nr']
            elif self.hyperparams['encoder_class'] in [RNNEncoder, RNNAutoRegEncoder]:
                latent_dim = self.hyperparams['train_params']['stitching_anchor_nr']
            elif self.hyperparams['encoder_class'] == TransformerEncoder:
                latent_dim = self.hyperparams['train_params']['stitching_anchor_nr']
        else:
            if self.hyperparams['encoder_class'] == MLPEncoder:
                latent_dim = self.hyperparams['encoder_hyperparams']['out_dim']
            elif self.hyperparams['encoder_class'] in [RNNEncoder, RNNAutoRegEncoder]:
                latent_dim = self.hyperparams['encoder_hyperparams']['hidden_dim']
            elif self.hyperparams['encoder_class'] == TransformerEncoder:
                latent_dim = self.hyperparams['encoder_hyperparams']['d_model']
            else:
                raise Exception(f'Unknown encoder type: {self.hyperparams["encoder_class"]}')

        self.K = torch.nn.Parameter(torch.eye(latent_dim))  
            

    def forward(self, enc_inputs, dec_inputs):
        """Propagate Koopman state for multiple steps."""
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



        encoded_pred = [encoded]
        for _ in range(self.hyperparams['prediction_length']-1):
            encoded = torch.matmul(encoded, self.K.T)
            encoded_pred.append(encoded)

        koopman_encoded = torch.stack(encoded_pred, dim=1)  # Shape: (batch, num_steps+1, latent_dim)
        if self.hyperparams['decoder_class'] == MLPDecoder:
            decoded = self.decoder(koopman_encoded)
        elif self.hyperparams['decoder_class'] == RNNDecoder:
            decoded, hidden = self.decoder(koopman_encoded, hidden)
        elif self.hyperparams['decoder_class'] == RNNAutoRegDecoder:
            decoded, hidden = self.decoder(koopman_encoded, dec_inputs, hidden)
        elif self.hyperparams['decoder_class'] == TransformerDecoder:
            decoded = self.decoder(dec_inputs, koopman_encoded)
        else:
            raise Exception(f'Unknown decoder type: {self.hyperparams["decoder_class"]}')

        return decoded

