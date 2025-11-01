import os
from typing import TypedDict, Type
from typing_extensions import Self
import torch
from torch import nn

#from ModelParams import ModelParams 
# Cannot be imported because of circular import


class EncoderParams(TypedDict):
    pass


class DecoderParams(TypedDict):
    pass


# All Encoders inherit from this class
class BaseEncoder(nn.Module):

    def __init__(self, hyperparams: EncoderParams) -> None:
        
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def save_to_file(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self, filepath)

    @classmethod
    def load_from_file(cls, filepath: str):
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pkl":
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            print(f"Pickle‐loaded model from {filepath}")
            return model

        safe_list = [NoneModel]
        with ts.safe_globals(safe_list):
            model = torch.load(filepath, map_location="cpu", weights_only=False)
        print(f"Torch‐loaded model from {filepath} with safe globals {safe_list}")
        return model


# All Decoders inherit from this class
class BaseDecoder(nn.Module):
    def __init__(self, hyperparams: DecoderParams) -> None:
        # Save all hyperparams, e.g., self.hyperparams = hyperparams
        # Initialize layers etc.
        super().__init__()

    # See forward of BaseModel
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BaseModel(nn.Module):
    
    def __init__(self, hyperparams) -> None:
        super().__init__()
        from src.model.model_params import ModelParams

        self.hyperparams = hyperparams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        pass

    def save_to_file(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self, filepath)

    @classmethod
    def load_from_file(cls, filepath: str) -> Self:
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pkl":
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            print(f"Pickle‐loaded model from {filepath}")
            return model
        else:
            model = torch.load(filepath, weights_only=False)
            print(f"Torch‐loaded model from {filepath}")
            return model

    def save_encoder_to_file(self, filepath: str) -> None:
        self.encoder.save_to_file(filepath)

    @classmethod
    def load_encoder_from_file(cls, encoder_class: Type[BaseEncoder], filepath: str) -> BaseEncoder:
        return encoder_class.load_from_file(filepath)

    def required_nr_points_after_prediction(self) -> int:
        return self.hyperparams['input_before']
