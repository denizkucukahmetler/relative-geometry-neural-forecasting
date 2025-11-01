import numpy as np
import torch
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import TypedDict
#import the data handlers for each dataset
from src.model.NoneModel import NoneModel
from src.data_handlers.BaseDataHandlerParams import BaseDataHandlerParams
from src.data_handlers.base_data_handler import BaseDataHandler
from src.data_handlers.lorenz_data_handler import LorenzDataHandler
from src.data_handlers.double_pendulum_data_handler import DoublePendulumDataHandler
from src.data_handlers.logistic_map_data_handler import LogisticMapDataHandler
from src.data_handlers.POD_data_handler import PODDataHandler
from src.data_handlers.spiral_data_handler import SpiralDataHandler
from src.data_handlers.dynamical_dataset import DynamicalDataset
from src.data_handlers.data_utils import sequence_collate_fn
from src.model.MLP import MLPParams, MLP
from src.model.Transformer import TransformerModel, TransformerParams
from src.model.RNNAutoReg import RNNAutoReg, RNNAutoRegParams
from src.model.seq2seq import RnnModel
from src.model.ESN import ESNModel, ESNHyperParams
from src.benchmark.benchmarks import *
from src.model.NODE import NODE
from src.model.Koopman import Koopman

from torch.utils.data import Dataset, DataLoader, random_split
from src.data_handlers.data_utils import sequence_collate_fn

from src.model.MLP import MLP
from src.model.Transformer import TransformerModel
from src.model.RNN import RNN
from src.model.ESN import ESN

from typing import TypedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os


def benchmark_models(models, datasets, filepath, device):
    all_benchmarks = {}  # Dictionary to hold all benchmarks

    for i, model in enumerate(models):
        dataset = datasets[i] #get dataset
        if isinstance(model, NoneModel):
            continue
        elif isinstance(model, ESNModel):
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=sequence_collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=sequence_collate_fn)
            model.eval() #set model to eval since we only want to compute predictions
            model.to(device)

        # Get the model's predictions
        predictions = []
        ground_truths = []
        with torch.no_grad():
            for enc_inputs, dec_inputs, dec_targets, timestamp, scaler in iter(dataloader):
                enc_inputs = enc_inputs.to(device)
                dec_inputs = dec_inputs.to(device)
                if isinstance(model, MLP): # Reshape MLP output
                    predicted_flat = model(enc_inputs, dec_inputs).to(device)  # Shape: [batch, sequence_length * num_dims]
                    predicted = predicted_flat.view(-1, enc_inputs.shape[2])  # Shape: [batch,prediction_length, num_dims]
                elif isinstance(model, RNNAutoReg) or isinstance(model,RnnModel):
                    predicted = model(enc_inputs, dec_inputs).to(device)  # Shape: [1, prediction_length, num_dims]
                    predicted = predicted.reshape(-1, predicted.shape[2]) #shape: (batch*sequence,dim)
                elif isinstance(model, TransformerModel):
                    predicted = model(enc_inputs, dec_inputs).to(device) #shape: (batch,sequence,dim)
                    predicted = predicted.reshape(-1, predicted.shape[2]) #shape: (batch*sequence,dim)
                elif isinstance(model, ESNModel):
                    enc_inputs  = enc_inputs.to(device)

                    predicted   = model.test_esn_on_tesor_data(enc_inputs).to(device)  # [1, Lp, d]
                    predicted   = predicted.squeeze(0)                                # [Lp, d]

                    dec_targets = dec_targets.squeeze(0)[1:]                          # [Lt, d]

                    seq_len = min(predicted.shape[0], dec_targets.shape[0])
                    predicted   = predicted[:seq_len]
                    dec_targets = dec_targets[:seq_len]

                    # put batch dim back because the common code expects it
                    dec_targets = dec_targets.unsqueeze(0)                            # [1, seq_len, d]

                elif isinstance(model, NODE):
                    predicted = model(enc_inputs.to(device), dec_inputs.to(device)).to(device) #shape: (batch,sequence,dim)
                    predicted = predicted.reshape(-1, predicted.shape[2]) #shape (batch*sequence,dim)
                elif isinstance(model, Koopman):
                    predicted = model(enc_inputs.to(device), dec_inputs.to(device)).to(device) #shape: (batch,sequence,dim)
                    predicted = predicted.reshape(-1, predicted.shape[2]) #shape (batch*sequence,dim)
    
                dec_targets = dec_targets.reshape(-1, dec_targets.shape[2])
                dec_targets = dec_targets.to(device)
    
                predictions.append(predicted)
                ground_truths.append(dec_targets)

        all_predictions = torch.cat(predictions)
        all_ground_truth = torch.cat(ground_truths)

        newfilepath = filepath + f"/{model.hyperparams['model_name']}" 

        os.makedirs(filepath, exist_ok=True)  # Create directory if it doesn't exist

        all_predictions = all_predictions.detach().cpu()
        all_ground_truth = all_ground_truth.detach().cpu()
        benchmarks = benchmark(all_predictions,all_ground_truth, newfilepath)
        all_benchmarks[model.hyperparams['model_name']] = benchmarks 
        print(f"{model.hyperparams['model_name']}: \n {benchmarks}") 

    save_all_benchmarks(all_benchmarks, filepath)
