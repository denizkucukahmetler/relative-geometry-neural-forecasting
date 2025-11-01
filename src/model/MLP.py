import torch
from src.model.model_params import ModelParams
from src.model.model_interface import BaseModel
from typing import TypedDict
from src.encoder.MLPEncoder import MLPEncoder
from src.decoder.MLPDecoder import MLPDecoder

class MLPParams(ModelParams):
    pass

class MLP(BaseModel):
    def __init__(self, hyperparams: MLPParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams

        self.encoder = self.hyperparams['encoder_class'](self.hyperparams['encoder_hyperparams'])
        self.decoder = self.hyperparams['decoder_class'](self.hyperparams['decoder_hyperparams'])

    def forward(self, x, dec_inputs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
