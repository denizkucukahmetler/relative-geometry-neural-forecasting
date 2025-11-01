from src.config_manager import ConfigManager
from src.training_manager import TrainingManager
from src.experiment_manager import ExperimentManager
import torch
device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


dataset_name = "pod"
experiment_name = f"all_{dataset_name}_bests" 
models = [['none',1],['mlp',1]]
config_manager = ConfigManager(dataset_name, device)
training_manager = TrainingManager(device)

data_handler_params = config_manager.get_current_dataset_config()

anchors = 80
points_to_embed = 1000
window = 5         # e.g. time window of 5
stride = 1         # slide by 1
start = 0          # begin at first sample

experiment_manager = ExperimentManager(config_manager, device)

models, rel_lat, abs_lat = experiment_manager.evaluate_latent_spaces_time(
    experiment_name,
    points_to_embed,
    anchors,
    window,
    stride,
    start,
    random_anchors=True,
)
