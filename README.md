# Relative-geometry-neural-forecasting
This repository contains the implementation of the paper "Relative Geometry of Neural Forecasting Models:  Linking Accuracy and Alignment in Learned Dynamics". 

We investigated latent spaces of various encoder-decoder architectures to explore what embeddings are learned when models are trained on dynamic (and chaotic) systems like the Lorenz Attractor. To compare these latent spaces, we used **relative embeddings** as proposed by Moschella et al. (2023).

## Installation
```pip install -r requirements.txt```

## Quickstart guide
You can perform experiments using two key files:
- ```main.ipynb```
- ```src/config_manager.py```

In the main notebook, you can:
1) Choose models and datasets
2) Train models
3) Compute relative latent spaces
4) Visualize (absolute and relative) latent spaces as well as temporal alignment with the true system
5) Benchmark model performance

Hyperparameters for models, training, and datasets are set through the ```config_manager```. 

### Choose Models and datasets
Start by setting an experiment name. All results and trained models will be saved in:
```results/experiment_name```

Select a dataset to train on. Available datasets:
- ```lorenz (Lorenz-63)```
- ```double_pendulum (Double Pendulum)```
- ```random_skew (Random Skew Product)```
- ```spiral (Limit Cycle)``` 
- ```pod (POD-wake)```
- ```logistic_map (Logistic Map)```
- ```hopf (Hopf normal form)``` <br>

Then, specify which models to train using a list of [model_name, count (seeds)] pairs:

```
experiment_name = "experiment_test"                     #set the experiment name
dataset_name = "lorenz"                                 #choose Lorenz Dataset
models = [['mlp', 2],['transformer', 1], ['none', 1]]   #train 2 MLPs, 1 transformer & 1 None model
```

Supported models:
- ```mlp (MLP)```
- ```transformer (Transformer)```
- ```old_rnn (RNN)```
- ```nodemlp (Neural ODE-MLP)```
- ```nodernn (Neural ODE-RNN)```
- ```nodetransformer (Neural ODE-Transformer)```
- ```koopmanmlp (Koopman-MLP)```
- ```koopmanrnn (Koopman-RNN)```
- ```koopmantransformer (Koopman-Transformer)```
- ```esn (ESN)```
- ```none (Ground Truth/True System)```


### Training
Load model, dataset, and training hyperparameters using config_manager. Training is managed via ```src/training_manager.py```, which calls ```src/trainer.py```:
```config_manager = ConfigManager(dataset_name, device)             #loading the config manager
training_manager = TrainingManager(device)                          #loading the training manager
data_handler_params = config_manager.get_current_dataset_config()   #loading parameters for the datahandler
#training
training_manager.train_multiple_models(experiment_name, config_manager, models)
```
### Computing relative latent spaces
Once training is complete, all models are saved in ```results/experiment_name```.

Use ```src/experiment_manager.py``` to run experiments. Choose:
- Number of anchors
- Number of points to embed <br>

A similarity matrix (cosine similarity between relative latent spaces) is saved as ```results/experiment_name/similarities_{similarity_metric}.csv```.
```
anchors = 80            #choose number of anchors
points_to_embed = 500   #choose number of point to embed
experiment_manager = ExperimentManager(config_manager, device) #set up experiment manager
models, latent_spaces = experiment_manager.evalute_latent_spaces_exp(experiment_name, points_to_embed, anchors) #call pipeline
```


### Visualize relative latent spaces
Visualize relative latent spaces using:
- A reducer: ```pca``` or ```umap```
- A list of model directories to compare
```
vis_latent1 = ('pca', ['mlp1', 'mlp2'])             #use pca to compare mlp1 and mlp2
vis_latent2 = ('umap', ['transformer1', 'none1'])   #use umap to compare transformer with ground truth
vis_latent = [vis_latent1, vis_latent2]             #visualize both
experiment_manager.visualize_relative_latent_spaces(latent_spaces, vis_latent, experiment_name) #call pipeline
```

### Benchmark the models
We evaluate models using multiple performance metrics beyond just loss:
- Probability Density Function (PDF)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE) 
- Mean Absolute Error (MAE)

```
experiment_manager.benchmark_models_exp(experiment_name) #call the benchmarking pipeline
```


# Reproducing Experiments 

Details for reproducing the results of the paper **"Relative Geometry of Neural Forecasting Models: Linking Accuracy and Alignment in Learned Dynamics"**. 


## Metadata

| File | Description |
|------|-------------|
| `hyperparameters/best_model_parameters.csv` | Parameters used for the final models reported in the paper. |
| `hyperparameters/compiled_results.csv` | Aggregated results from the hyperparameter search for each model–dataset combination. |
| `src/config_manager.py` | Dataset parameters and training defaults. |


## Experiments

All scripts are in the `experiments/` directory. Unless otherwise noted, they use the parameters in `hyperparameters/best_model_parameters.csv` and the dataset settings in `src/config_manager.py`.


### Benchmarking Models:
We train 11 models on 7 datasets using 5 random seeds to measure alignment versus performance during training.

Run training with:

```
experiments/train_best_models.py
```
During training, representational similarity scores (RSS) and performance metrics are saved to:
loss_log.csv

To benchmark on the test set and visualize absolute and relative latent spaces, run:
```
experiments/test_best_models.py
```
where the benchmarks.json includes the benchmark performances (MSE, RMSE, MAE) are recorded. The cross-modal similarity is recorded in similarities_cosine.csv, similarities_top1.csv and similarities_rank.csv

### Noise Experiment:
Evaluate robustness of relative geometry under varying observational noise:
```
experiments/noise_exp.py
```

### Input Length (L) Experiment:
Assess how input sequence length affects geometry and forecasting performance:
```
experiments/varying_input_length_exp.py
```

### Temporal Alignment Experiment:
Examine temporal (mis)alignment between learned and ground-truth trajectories:
```
experiments/temporal_alignment.py
```

### Stitching Experiment:
Study **absolute** and **relative stitching** of partial trajectories into global representations.

**Configuration files:**
- Absolute stitching:* `config_manager_abs.py`
- Relative stitching:* `config_manager_rel.py`

Run:
```
python experiments/stitching_exp.py
```



