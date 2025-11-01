
from typing import TypedDict, List, Tuple


from src.data_handlers.base_data_handler import BaseDataHandler


class BaseDataHandlerParams(TypedDict):
    num_trajectories: int
    trajectory_length: int
    dt: float
    rand_start_bounds: float
    device: str
    data_type: str
    dat_handler_class: BaseDataHandler
    save_dir: str 
    initial_conditions_train: List[Tuple[float,float]]
    initial_conditions_val:   List[Tuple[float,float]]
    initial_conditions_test:  List[Tuple[float,float]]
    val_epsilon: float  