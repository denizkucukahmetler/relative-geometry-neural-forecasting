from src.model.model_params import ModelParams
from src.model.model_interface import BaseModel


class RNNAutoRegParams(ModelParams):
    pass


class RNNAutoReg(BaseModel):
    def __init__(self, hyperparams: RNNAutoRegParams):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams

        self.encoder = self.hyperparams['encoder_class'](self.hyperparams['encoder_hyperparams'])
        self.decoder = self.hyperparams['decoder_class'](self.hyperparams['decoder_hyperparams'])

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, hidden = self.encoder(enc_inputs)
        self.latent_representations = self.encoder.hidden_states  # Store latent representations
        outputs, hidden = self.decoder(enc_outputs, dec_inputs, hidden)
        return outputs
