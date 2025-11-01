import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.model_interface import BaseModel
from src.model.model_params import ModelParams

class RNNParams(ModelParams):
    pass

class RnnModel(BaseModel):
    def __init__(self, hyperparams: RNNParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams

        self.encoder = self.hyperparams['encoder_class'](self.hyperparams['encoder_hyperparams'])
        self.decoder = self.hyperparams['decoder_class'](self.hyperparams['decoder_hyperparams'])
        self.latent_representations = []  

    @staticmethod
    def get_dist_params(output):
        mu = output[:, :, :, 0]
        sigma = F.softplus(output[:, :, :, 1])
        return mu, sigma

    @staticmethod
    def sample_from_output(output):
        if output.shape[-1] > 1:  
            mu, sigma = RnnModel.get_dist_params(output)
            return torch.normal(mu, sigma)
        return output.squeeze(-1)

    def forward(self, enc_inputs, dec_inputs, teacher_force_prob=None):
        enc_outputs, hidden = self.encoder(enc_inputs)

        self.latent_representations = hidden 

        outputs = self.decoder(dec_inputs, hidden, enc_outputs, teacher_force_prob)
        outputs = outputs.squeeze(-1)
        return outputs