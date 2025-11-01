from src.config_manager import ConfigManager
from src.training_manager import TrainingManager
from src.experiment_manager import ExperimentManager
import torch
device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(device)

dataset_names = ["lorenz","spiral","pod","hopf","random_skew","double_pendulum","logistic_map"]

for dataset in dataset_names:

    
    experiment_name = f"all_{dataset}_bests" 
    dataset_name = dataset

    models = [['none',1],['mlp',1],['nodemlp',1],['koopmanmlp',1],['transformer',1],['nodetransformer',1],['koopmantransformer',1],['oldrnn',1],['rnnautoreg',1],['nodernnautoreg',1],['koopmanrnnautoreg',1]]

    config_manager = ConfigManager(dataset_name, device)
    training_manager = TrainingManager(device)

    data_handler_params = config_manager.get_current_dataset_config()

    anchors = 80
    points_to_embed = 1000

    experiment_manager = ExperimentManager(config_manager, device)

    models, latent_spaces, absolute_latent_spaces = experiment_manager.evalute_latent_spaces_exp(experiment_name, points_to_embed, anchors, random_anchors=True)
    experiment_manager.benchmark_models_exp(experiment_name)

    #choose which models latent spaces to visualize
    #specify the individual plots like this: vis_latent = (reducer, [model1, model2]), for reducers [pca, umap] are available

    vis_latent1 = ('pca',['mlp1','transformer1','none1','rnnautoreg1','nodernnautoreg1','nodetransformer1','koopmantransformer1','koopmanrnnautoreg1','esn1','oldrnn1','nodemlp1','koopmanmlp1'])

    vis_latent = [vis_latent1]

    experiment_manager.visualize_relative_latent_spaces(models, latent_spaces, vis_latent, experiment_name, fit_individually=False, n_components = 3)


    #choose which models latent spaces to visualize
    #specify the individual plots like this: vis_latent = (reducetar, [model1, model2]), for reducers [pca, umap] are available



    vis_latent1 = ('pca',['mlp1','transformer1','none1','rnnautoreg1','nodernnautoreg1','nodetransformer1','koopmantransformer1','koopmanrnnautoreg1','esn1','oldrnn1','nodemlp1','koopmanmlp1'])

    vis_latent = [vis_latent1]

                                                                #!                                                   #!                                       #!
    experiment_manager.visualize_relative_latent_spaces(models, absolute_latent_spaces, vis_latent, experiment_name, fit_individually=True, n_components = 3, fix_global_limits = True)