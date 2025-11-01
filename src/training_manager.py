import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import matplotlib.pyplot as plt


from src.model.model_interface import BaseModel
from src.data_handlers.base_data_handler import BaseDataHandler
from src.data_handlers.lorenz_data_handler import LorenzDataHandler
from src.data_handlers.spiral_data_handler import SpiralDataHandler
from src.data_handlers.logistic_map_data_handler import LogisticMapDataHandler 
from src.data_handlers.POD_data_handler import PODDataHandler

from src.data_handlers.dynamical_dataset import DynamicalDataset
from src.data_handlers.data_utils import sequence_collate_fn

from src.model.MLP import MLP
from src.model.Transformer import TransformerModel
from src.model.RNN import RNN
from src.model.ESN import ESNModel, ESNHyperParams
from src.model.NODE import NODE, NODEParams, ODEFunctionParams

from typing import TypedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import json
import random, numpy as np

from src.trainer import Trainer, TrainerHyperparams
from src.relative_trainer import RelativeTrainer

from src.decoder.MLPDecoder import MLPDecoder, MLPDecoderParams
from src.encoder.MLPEncoder import MLPEncoder, MLPEncoderParams
from src.model.MLP import MLPParams, MLP

from src.decoder.TransformerDecoder import TransformerDecoder, TransformerDecoderParams
from src.encoder.TransformerEncoder import TransformerEncoder, TransformerEncoderParams
from src.model.Transformer import TransformerModel, TransformerParams

from src.decoder.RNNDecoder import RNNDecoder, RNNDecoderParams
from src.encoder.RNNEncoder import RNNEncoder, RNNEncoderParams
from src.model.RNN import RNN, RNNParams

from src.decoder.RNNAutoRegDecoder import RNNAutoRegDecoder, RNNAutoRegDecoderParams
from src.encoder.RNNAutoRegEncoder import RNNAutoRegEncoder, RNNAutoRegEncoderParams
from src.model.RNNAutoReg import RNNAutoReg, RNNAutoRegParams

from src.config_manager import ConfigManager

def set_global_seed(seed: int):
    seed = seed % (2**32)  # Ensure the seed is within the valid range
    random.seed(seed);  np.random.seed(seed);  torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def sequential_split(dataset, train_pct, val_pct, gap):
    """
    Split a dataset into non-overlapping sequential subsets with specified percentages and a gap between them.
    Args:
        dataset: Dataset to be split.
        train_pct: Percentage of the dataset to be used for training.
        val_pct: Percentage of the dataset to be used for validation.
        gap: Number of data points to skip between each split.
    Returns:
        A list of Subset datasets.
    """
    total_gap = 2 * gap  # Adjust this if number of splits changes
    effective_dataset_length = len(dataset) - total_gap
    
    train_size = int(train_pct * effective_dataset_length)
    val_size = int(val_pct * effective_dataset_length)
    test_size = effective_dataset_length - train_size - val_size
    
    assert train_size + val_size + test_size + total_gap <= len(dataset), "Configured sizes and gaps exceed dataset length"
    
    splits = []
    idx = 0
    lengths = [train_size, val_size, test_size]
    for length in lengths:
        if idx + length > len(dataset):
            raise ValueError("Split index exceeds dataset length at split length: {}".format(length))
        splits.append(Subset(dataset, list(range(idx, idx + length))))
        idx += length + gap  # Adjust index for the next split, adding the gap
    
    return splits




class TrainingManager:
    
    def __init__(self, device):
        self.device = device
        
   
    def train_multiple_models(self, experiment_name, config_manager: ConfigManager, models):
        data_handler_params = config_manager.get_current_dataset_config()
        copy_for_esn = data_handler_params
        data_handler_params['device'] = self.device

        data_handler = data_handler_params['data_handler_class'](data_handler_params, experiment_name, True)

        file_path = f'results/{experiment_name}/data_handler.json'
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_handler_save = {'dataset_name': data_handler_params['dataset_name']}
        with open(file_path, 'w') as json_file:
            json.dump(data_handler_save, json_file, indent=4)
        


        
        model_params = [[config_manager.get_model_config_by_model_name(m[0]), m[1]] for m in models]

        updated_model_params = []

        for model, n in model_params:
            for i in range(1, n + 1):
                updated_model = model.copy()  # Create a copy to avoid modifying the original
                updated_model["model_name"] = f"{model['model_name']}{i}"  # Append the number
                updated_model_params.append(updated_model)

        model_params = updated_model_params



        trainer_params = [model_param['train_params'] for model_param in model_params]
        from pathlib import Path

        directories = [Path(f"results/{experiment_name}/{model_param['model_name']}/") for model_param in model_params] 

        for directory in directories:
            if directory.is_dir():
                raise RuntimeError(f"A model called  {directory.as_posix()} already exists! Please change the model name")

        datasets = data_handler.get_datasets(model_params)

        
        ### Split datasets
        
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for dataset in datasets:

            train_dataset, val_dataset, test_dataset = dataset[0], dataset[1], dataset[2]
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)

        models = []
        for m in model_params:
            model_class = m["model_class"]
            model = model_class(m)
            models.append(model)

        trainers = []
        for t in trainer_params:
            if t['stitching']:
                trainer = RelativeTrainer(t)
            else:
                trainer = Trainer(t)
            trainers.append(trainer)

        # Training
        base_seed = torch.initial_seed()
        last_model_type = None
        repeat_count = 0
        if trainer_params[0]['stitching']:
            K = trainer_params[0]['stitching_anchor_nr']
            rng = random.Random(42)
            indices = rng.sample(range(len(train_datasets[0])), K)
            items = [train_datasets[0][i] for i in indices]
            anchor_inputs, *_ = sequence_collate_fn(items)  # raw encoder inputs on CPU
            os.makedirs(f"results/{experiment_name}", exist_ok=True)
            np.save(f"results/{experiment_name}/anchor_indices.npy", np.array(indices, dtype=np.int64))
            torch.save(anchor_inputs.cpu(), f"results/{experiment_name}/anchor_inputs.pt")

        for (model, trainer, train_dataset, val_dataset, test_dataset) in zip(models, trainers, train_datasets, val_datasets, test_datasets):
            model_type = ''.join([c for c in model.hyperparams['model_name'] if not c.isdigit()])  # strip number suffix
            torch.cuda.empty_cache() #TODO added

            
            if model_type == last_model_type:
                repeat_count += 1
                set_global_seed(int(base_seed) + repeat_count)
            else:
                repeat_count = 0
                set_global_seed(int(base_seed))

            last_model_type = model_type

            dataset_name = data_handler_params['dataset_name']
            model_name = model.hyperparams['model_name']
            
            if isinstance(trainer, RelativeTrainer):
                trainer._anchor_inputs = anchor_inputs  # reuse across models
                trainer._anchor_indices = indices

            trainer.train_model(model, train_dataset, val_dataset, test_dataset, {'directory':  f'results/{experiment_name}/{model_name}/'}, copy_for_esn)
            trainer.test_model(model, test_dataset)
            
            best_path = f"results/{experiment_name}/{model_name}/best_current_model.pth"
            if os.path.isfile(best_path):
                try:
                    model.load_from_file(best_path)
                    print(f"Loaded best‐val model from {best_path}")
                except Exception as e:
                    print(f"Failed to load best model at {best_path}: {e}\nUsing last‐epoch weights.")
            else:
                print(f"No best‐val checkpoint found at {best_path}. Using last‐epoch weights.")


            print(model_name)
            trainer.plot(model, test_dataset, 5, f'results/{experiment_name}/{model_name}/plots')

            if "esn" in model_name:
                model.save_to_file(f'results/{experiment_name}/{model_name}/model.pkl')
            else:
                model.save_to_file( f'results/{experiment_name}/{model_name}/model.pth') 