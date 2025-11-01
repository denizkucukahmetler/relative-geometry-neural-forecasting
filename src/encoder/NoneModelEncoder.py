from torch import nn
from src.model.model_interface import EncoderParams, BaseEncoder


class NoneModelEncoderParams(EncoderParams):
    pass


class NoneModelEncoder(BaseEncoder):
    def __init__(self, hyperparams: NoneModelEncoderParams) -> None:
        super().__init__(hyperparams)
        self.hyperparams = hyperparams

    def forward(self, x):
        return x
