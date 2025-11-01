from src.model.model_params import ModelParams
from src.encoder.TransformerEncoder import TransformerEncoder
from src.decoder.TransformerDecoder import TransformerDecoder
from src.model.model_interface import *


class TransformerParams(ModelParams):
    pass

class TransformerModel(BaseModel):
    def __init__(self, hyperparams: TransformerParams) -> None:
        super().__init__(hyperparams)
        
        self.hyperparams = hyperparams
        self.encoder = self.hyperparams['encoder_class'](self.hyperparams['encoder_hyperparams'])
        self.decoder = self.hyperparams['decoder_class'](self.hyperparams['decoder_hyperparams'])

    def forward(self, enc_inputs: torch.Tensor, dec_inputs: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(enc_inputs)
        output = self.decoder(dec_inputs, memory) 
        return output
