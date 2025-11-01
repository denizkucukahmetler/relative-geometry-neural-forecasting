from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams
from src.data_handlers.base_data_handler import BaseDataHandlerModelParams
from src.model.model_interface import BaseDecoder, BaseEncoder, DecoderParams, EncoderParams, BaseModel

from typing import Type


class ModelParams(BaseDataHandlerModelParams):
    encoder_class: Type[BaseEncoder]
    encoder_hyperparams: EncoderParams
    decoder_class: Type[BaseDecoder]
    decoder_hyperparams: DecoderParams

    data_handler_params: BaseDataHandlerParams

    model_class: Type[BaseModel]
    model_name: str
