from src.model.model_params import ModelParams
from src.model.model_interface import BaseModel


class RNNParams(ModelParams):
    pass


class RNN(BaseModel):
    def __init__(self, hyperparams: RNNParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams

        self.encoder = self.hyperparams['encoder_class'](self.hyperparams['encoder_hyperparams'])
        self.decoder = self.hyperparams['decoder_class'](self.hyperparams['decoder_hyperparams'])

    def forward(self, x):
        encoded, hidden = self.encoder(x)
        decoded, hidden = self.decoder(encoded, hidden)
        return decoded
