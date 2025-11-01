import torch
from src.model.model_params import ModelParams
from src.model.model_interface import BaseModel
from typing import TypedDict
from src.encoder.NoneModelEncoder import *
from src.decoder.NoneModelDecoder import *

class NoneModelParams(ModelParams):
    pass

class NoneModel(BaseModel):
    def __init__(self, hyperparams: NoneModelParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams

        self.encoder = self.hyperparams['encoder_class'](self.hyperparams['encoder_hyperparams'])
        self.decoder = self.hyperparams['decoder_class'](self.hyperparams['decoder_hyperparams'])

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
