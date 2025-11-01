from torch import nn

from src.model.model_interface import BaseDecoder, DecoderParams


class NoneModelDecoderParams(DecoderParams):
    pass

class NoneModelDecoder(BaseDecoder):
    def __init__(self, hyperparams: NoneModelDecoderParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams
        
    def forward(self, y):
        return y
