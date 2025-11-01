from src.config_manager import ConfigManager
from src.training_manager import TrainingManager
from src.experiment_manager import ExperimentManager
import torch
device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(device)

experiment_name = "absolute_sitching_exp" # TODO change exp name
dataset_name = "lorenz"
models = [['mlp', 5], ['nodemlp', 5], ['koopmanmlp', 5], ['transformer', 5], ['nodetransformer', 5], ['koopmantransformer', 5], ['none',1]]

#training
config_manager = ConfigManager(dataset_name, device)
training_manager = TrainingManager(device)
data_handler_params = config_manager.get_current_dataset_config()
training_manager.train_multiple_models(experiment_name, config_manager, models)

#cosine
anchors = 80
points_to_embed = 1000

experiment_manager = ExperimentManager(config_manager, device)
models, latent_spaces, absolute_latent_spaces = experiment_manager.evalute_latent_spaces_exp(experiment_name, points_to_embed, anchors)

#stitching
from src.stitching.relative_stitching import RelativeStitcher
from src.stitching.absolute_stitching import AbsoluteStitcher
model_names = []
for model in models:
    model_name = model.hyperparams["model_name"]
    if "none" in model_name:
        continue
    model_names.append(model_name)
print(model_names)

def stitching(model_names:list, relative:bool, batch_size:int)->None:
    """Calls the Relative or Absolute stitcher and evaluates each encoder decoder stitching pair."""
    if relative is True:
        stitcher = RelativeStitcher({"device": device, "experiment_name": experiment_name, "dataset_name": dataset_name}) #relative stiching
    else:
        stitcher = AbsoluteStitcher({"device": device, "experiment_name": experiment_name, "dataset_name": dataset_name}) #absolute stitching
    
    #loop through every encoder and stitch all decoders
    for encoder_name in model_names:
        print(f"Stiching for Encoder: {encoder_name}")
        print("-------")
        results = stitcher.stitch(encoder_model_name=encoder_name, batch_size=batch_size)
        stitcher.save_results(results,encoder_model_name=encoder_name) #save in stitching/

stitching(model_names, False, 128) # TODO pass relative=True when relative stitching, False if absolute stitching!

# Cosine similarity heatmap
from src.stitching.utils import cosine_heatmap
cosine_heatmap(dir_path=f"results/{experiment_name}/")

