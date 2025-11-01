from torch import device
from src.model.NODE import NODE, NODEParams
from src.decoder.NoneModelDecoder import *
from src.encoder.NoneModelEncoder import *
from copy import deepcopy


from src.model.NoneModel import NoneModel
from src.model.NoneModel import NoneModelParams
from src.data_handlers.double_pendulum_data_handler import DoublePendulumDataHandler
from src.data_handlers.lorenz_data_handler import LorenzDataHandler
from src.data_handlers.spiral_data_handler import SpiralDataHandler
from src.data_handlers.logistic_map_data_handler import LogisticMapDataHandler
from src.data_handlers.POD_data_handler import PODDataHandler
from src.data_handlers.hopf_data_handler import HopfDataHandler
from src.data_handlers.random_systems_data_handler import RandomSkewDataHandler
from src.decoder.MLPDecoder import MLPDecoder, MLPDecoderParams
from src.decoder.KoopmanDecoder import KoopmanDecoder, KoopmanDecoderParams
from src.decoder.RNNAutoRegDecoder import RNNAutoRegDecoder, RNNAutoRegDecoderParams
from src.decoder.old_rnn import RnnDecoder, OLDRNNDecoderParams #RNNOLD
from src.decoder.TransformerDecoder import TransformerDecoder, TransformerDecoderParams
from src.encoder.MLPEncoder import MLPEncoder, MLPEncoderParams
from src.encoder.KoopmanEncoder import KoopmanEncoder, KoopmanEncoderParams
from src.encoder.RNNAutoRegEncoder import RNNAutoRegEncoder, RNNAutoRegEncoderParams
from src.encoder.TransformerEncoder import TransformerEncoder, TransformerEncoderParams
from src.encoder.RNNEncoder import RNNEncoder, RNNEncoderParams
from src.model.ESN import ESNHyperParams, ESNModel
from src.model.MLP import MLP, MLPParams
from src.model.Koopman import Koopman, KoopmanParams
from src.model.RNNAutoReg import RNNAutoReg, RNNAutoRegParams
from src.model.seq2seq import RnnModel, RNNParams #RNNOLD
from src.model.Transformer import TransformerModel, TransformerParams
from src.trainer import TrainerHyperparams
import numpy as np
from collections.abc import Mapping


class ConfigManager:
    def __init__(self, dataset_name, device):
        
        self.dataset_configs = [
            {
                'num_trajectories': 10,
                'trajectory_length': 500, #This will be ignored for the ESN, see points parameter in ESNHyperparams
                'dt': 0.01,
                'rand_start_bounds': 20,
                'device': device,
                'data_handler_class': LorenzDataHandler,
                'dataset_name': 'lorenz',
                'save_dir': "./",  
                'in_dim': 3,
                'noise_pct': 0,
                'split_approach': 'trajectory',
            },
            {
                'num_trajectories': 10,
                'trajectory_length': 500,
                'dt': 0.01,
                'rand_start_bounds': 20,
                'device': device,
                'data_handler_class': SpiralDataHandler,
                'dataset_name': 'spiral',
                'save_dir': "./" ,
                'in_dim': 2,
                'seed': 42,
                'initial_conditions_train': None,#[(1,0), (2,0), (3,0), (4,0)],
                'noise_pct': 0,
                'split_approach': 'trajectory',
            },
            {
                'num_trajectories': 10,
                'trajectory_length': 500,
                'dt': 0.01,
                'rand_start_bounds': np.pi,
                'device': device,
                'data_handler_class': DoublePendulumDataHandler,
                'dataset_name': 'double_pendulum',
                'seed': 42,
                'save_dir': "./",
                'in_dim': 4,
                'noise_pct': 0,
                'split_approach': 'trajectory',
            },
            {
                'num_trajectories': 10,
                'trajectory_length': 500,
                'dt': 0.1,  
                'rand_start_bounds': 0,
                'device': device,
                'data_handler_class': LogisticMapDataHandler,
                'dataset_name': 'logistic_map',
                'save_dir': "./",
                'noise_pct': 0,
                'in_dim': 1,
            },
            {
                'num_trajectories': 10,
                'trajectory_length': 500,
                'dt': 0.2,  
                'rand_start_bounds': 0,  
                'device': device,
                'data_handler_class': PODDataHandler,
                'dataset_name': 'pod',
                'save_dir': "./",
                'in_dim': 3,
                'data_files': ['src/data_handlers/POD_data/PODcoefficients.mat','src/data_handlers/POD_data/PODcoefficients_run1.mat']
            },
            {
                'num_trajectories': 10, 
                'trajectory_length': 500, 
                'dt': 0.01, 
                'rand_start_bounds': 2, # Adjust as needed, e.g., 2 like in PySINDy example
                'device': device,
                'data_handler_class': HopfDataHandler, # Use the new handler
                'dataset_name': 'hopf',
                'save_dir': './', # Optional: specify directory to save raw trajectories
                'in_dim': 2, # Hopf is 2-dimensional
                # --- Hopf specific parameters ---
                'mu': 0.0,     
                'omega': 1.0,    # Default value, adjust as needed
                'A': 1.0,       # Set to 1.0 for PySINDy example, 0.0 for stable focus
                'noise_pct': 0,
            },
            {
            'num_trajectories': 10,          # “draw 100 random systems”
            'trajectory_length': 500,
            'dt': 0.01,
                
            'rand_start_bounds': 1,           # unused but kept for consistency
            'device': device,
            'data_handler_class': RandomSkewDataHandler,
            'dataset_name': 'random_skew',
            'save_dir': 'results',            # used by handler when saving CSVs
            'in_dim': 6,                      # MUST equal output_dim below
            # handler-specific knobs (optional):
            'output_dim': 6,                  # 6 (driver+response) or 3 (driver only)
            'try_hard': 12,                   # attempts per trajectory
            'warmup_frac': 0.1,               # burn-in fraction
            'enforce_chaos': False,            # run 0–1 test
            'zero_one_min': 0.5,              # threshold for 0–1 chaos test
            'split_approach': 'trajectory',   # or 'time'
            'seed': 42,
            'noise_pct': 0,
        }



        ]
        self.current_dataset_config = self.get_dataset_config_by_dataset_name(dataset_name)



        self.model_configs = [



            NoneModelParams({
                'model_name': 'none',
                'encoder_class': NoneModelEncoder,
                'encoder_hyperparams': NoneModelEncoderParams({}),
                'decoder_class': NoneModelDecoder,
                'decoder_hyperparams': NoneModelDecoderParams({}),
                'data_handler_params': self.current_dataset_config,
                'model_class': NoneModel,
                "input_before": 1,
                "input_after": 0,
                "prediction_length": 1,
                'train_params': TrainerHyperparams({
                    'lr': 0.01,
                    'device': device,
                    'batch_size': 64,
                    'epochs': 0,
                    'model_type': 'NoneModel',
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
            }),  

            
            
            MLPParams({
                'model_name': 'mlp',
                'encoder_class': MLPEncoder,
                'encoder_hyperparams': MLPEncoderParams({
                    'in_dim': 144* self.current_dataset_config['in_dim'], #sequence length * #dimensions #generate in_dim from data
                    'hidden_dim': 256,
                    'out_dim': 256,
                }),
                'decoder_class': MLPDecoder,
                'decoder_hyperparams': MLPDecoderParams({
                    'in_dim': 256,
                    'hidden_dim': 150,
                    'out_dim': 50* self.current_dataset_config['in_dim'], #prediction_length * #dimensions
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': MLP,
                "input_before": 144, # dont forget to change this in the encoder hyperparams
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'lr': 0.001,
                    'device': device,
                    'batch_size': 64,
                    'epochs': 200,
                    'model_type': 'MLP',
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
            }),  

            



            TransformerParams({
                'model_name': 'transformer',
                'encoder_class': TransformerEncoder,
                'encoder_hyperparams': TransformerEncoderParams({
                    'input_dim':self.current_dataset_config['in_dim'],
                    'd_model': 128,
                    'n_heads': 8,
                    'n_layers': 2,
                    'dropout': 0.1
                }),
                'decoder_class': TransformerDecoder,
                'decoder_hyperparams': TransformerDecoderParams({
                    'input_dim':self.current_dataset_config['in_dim'],
                    'd_model': 128,
                    'n_heads': 8,
                    'n_layers': 2,
                    'dropout': 0.1
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': TransformerModel,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'lr': 0.001,
                    'device': device,
                    'batch_size': 64,
                    'epochs': 200,
                    'model_type': 'Transformer',
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
            }),


            
            RNNAutoRegParams({
                'model_name': 'rnnautoreg',
                'encoder_class': RNNAutoRegEncoder,
                'encoder_hyperparams': RNNAutoRegEncoderParams({
                    'input_dim': self.current_dataset_config['in_dim'],
                    'hidden_dim': 128,
                    'num_layers': 3,
                    'dropout': 0.12
                }),
                'decoder_class': RNNAutoRegDecoder,
                'decoder_hyperparams': RNNAutoRegDecoderParams({
                    'input_dim': self.current_dataset_config['in_dim'],
                    'hidden_dim': 128,
                    'num_layers': 3,
                    'output_dim': self.current_dataset_config['in_dim'],
                    'dropout': 0.12
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': RNNAutoReg,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'lr': 0.006,
                    'device': device,
                    'batch_size': 64,
                    'epochs': 200,
                    'model_type': 'RNN',
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
            }),
            
            RNNParams({
                'model_name': 'oldrnn',
                'encoder_class': RNNEncoder,
                'encoder_hyperparams': RNNEncoderParams({
                    'input_dim': self.current_dataset_config['in_dim'],
                    'hidden_dim': 128,
                    'num_layers': 3,
                    'dropout': 0.12
                }),
                'decoder_class': RnnDecoder,
                'decoder_hyperparams': OLDRNNDecoderParams({
                    'hidden_size': 128,
                    'dec_feature_size': self.current_dataset_config['in_dim'],#'dec_feature_size': 3, #shape[-1] of dec_inputs
                    'dropout': 0.12,
                    'dec_target_size': self.current_dataset_config['in_dim'], #'dec_target_size': 3, #shape[-1] of dec_tragets
                    'target_size': self.current_dataset_config['in_dim'],  #'target_size': 3, #same as dec_target_size
                    'dist_size': 1,
                    'target_indices': list(range(self.current_dataset_config['in_dim'])), #[0,1,2], #dimensions, e.g. for lorenz 3 -> [0,1,2]
                    'num_gru_layers': 3
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': RnnModel,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'lr': 0.001,
                    'device': device,
                    'batch_size': 32,
                    'epochs': 200,
                    'model_type': 'RNN',
                    'lr_decay': False,
                    'teacher_force_prob': 0.5 ,#None => no teacher forcing, float e.g. 0.5 for 50% teacher forceing probabilty; linear decrase of 0.1=10% each epoch,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
            }),



            
            
            ESNHyperParams({
                'model_name': 'esn',
                "n_inputs": self.current_dataset_config['in_dim'],
                "n_reservoir": 32,
                "spectral_radius": 1.1,
                "sparsity": 0.0,
                "input_scaling": None,
                "leaking_rate":1,
                "ridge_lambda": 1e-2,
                "feedback_scaling": 1,
                "noise": 0.0,
                "teacher_forcing": True,
                "random_state": None, #42,
                "n_outputs": self.current_dataset_config['in_dim'],
                'model_class': ESNModel,
                "input_before": 100,
                "input_after": 0,
                "prediction_length": 50,
                "num_trajectories": 5,
                'train_params': TrainerHyperparams({
                    'device': device,
                    "points": 5000, 
                    "normalization": True,
                    'model_type': 'ESN',
                    #not needed
                    'lr': 0.001,
                    'batch_size': 1,
                    'epochs': 1,
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                })
            }),

            
            NODEParams({
                'model_name': 'nodemlp',
                'encoder_class': MLPEncoder,
                'encoder_hyperparams': MLPEncoderParams({
                    'in_dim': 144* self.current_dataset_config['in_dim'], #sequence length * #dimensions #generate in_dim from data
                    'hidden_dim': 256,
                    'out_dim': 90,
                }),
                'decoder_class': MLPDecoder,
                'decoder_hyperparams': MLPDecoderParams({
                    'in_dim': 90,
                    'hidden_dim': 150,
                    'out_dim': self.current_dataset_config['in_dim'], #prediction_length * #dimensions
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': NODE,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'model_type': 'NODE',
                    'device': device,
                    'lr': 0.001,
                    'batch_size': 256,
                    'epochs': 50,
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
                'ode_params': {'in_out_dim': 90,
                    'hidden_dim': 128
                }
            }),



            NODEParams({
                'model_name': 'nodernnautoreg',
                'encoder_class': RNNAutoRegEncoder,
                'encoder_hyperparams': RNNAutoRegEncoderParams({
                    'input_dim': self.current_dataset_config['in_dim'],
                    'hidden_dim': 128,
                    'num_layers': 4,
                    'dropout': 0.15000000000000002
                }),
                'decoder_class': RNNAutoRegDecoder,
                'decoder_hyperparams': RNNAutoRegDecoderParams({
                    'input_dim': self.current_dataset_config['in_dim'],
                    'hidden_dim': 128,
                    'num_layers': 4,
                    'output_dim': self.current_dataset_config['in_dim'],
                    'dropout':0.15000000000000002,
                    'teacher_force': 0.5
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': NODE,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'lr': 0.0022694851726357654,
                    'device': device,
                    'batch_size': 128,
                    'epochs': 200,
                    'model_type': 'NODE',
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
                'ode_params': {'in_out_dim': 128,
                    'hidden_dim': 64
                }
            }),



            NODEParams({
                'model_name': 'nodetransformer',
                'encoder_class': TransformerEncoder,
                'encoder_hyperparams': TransformerEncoderParams({
                    'input_dim':self.current_dataset_config['in_dim'],
                    'd_model': 512,
                    'n_heads': 8,
                    'n_layers': 4,
                    'dropout': 0.1
                }),
                'decoder_class': TransformerDecoder,
                'decoder_hyperparams': TransformerDecoderParams({
                    'input_dim':self.current_dataset_config['in_dim'],
                    'd_model': 512,
                    'n_heads': 8,
                    'n_layers': 4,
                    'dropout': 0.5
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': NODE,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'lr': 0.0001,
                    'device': device,
                    'batch_size': 256,
                    'epochs': 200,
                    'model_type': 'NODE',
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
                'ode_params': {'in_out_dim': 512,
                    'hidden_dim': 256
                }
            }),

            
            
            
            KoopmanParams({
                'model_name': 'koopmanmlp',
                'encoder_class': MLPEncoder,
                'encoder_hyperparams': MLPEncoderParams({
                    'in_dim':  144* self.current_dataset_config['in_dim'], #sequence length * #dimensions #generate in_dim from data
                    'hidden_dim': 256,
                    'out_dim': 90,
                }),
                'decoder_class': MLPDecoder,
                'decoder_hyperparams': MLPDecoderParams({
                    'in_dim': 90,
                    'hidden_dim': 150,
                    'out_dim': self.current_dataset_config['in_dim'], #prediction_length * #dimensions
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': Koopman,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'model_type': 'Koopman',
                    'device': device,
                    'lr': 0.001,
                    'batch_size': 256,
                    'epochs': 50,
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
                'ode_params': {'in_out_dim': 90,
                    'hidden_dim': 128
                }
            }),


            KoopmanParams({
                'model_name': 'koopmanrnnautoreg',
                'encoder_class': RNNAutoRegEncoder,
                'encoder_hyperparams': RNNAutoRegEncoderParams({
                    'input_dim': self.current_dataset_config['in_dim'],
                    'hidden_dim': 128,
                    'num_layers': 4,
                    'dropout': 0.15000000000000002
                }),
                'decoder_class': RNNAutoRegDecoder,
                'decoder_hyperparams': RNNAutoRegDecoderParams({
                    'input_dim': self.current_dataset_config['in_dim'],
                    'hidden_dim': 128,
                    'num_layers': 4,
                    'output_dim': self.current_dataset_config['in_dim'],
                    'dropout':0.15000000000000002,
                    'teacher_force': 0.5
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': Koopman,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'lr': 0.0022694851726357654,
                    'device': device,
                    'batch_size': 128,
                    'epochs': 200,
                    'model_type': 'Koopman',
                    'lr_decay': True,
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
                'ode_params': {'in_out_dim': 128,
                    'hidden_dim': 64
                }
            }),



            KoopmanParams({
                'model_name': 'koopmantransformer',
                'encoder_class': TransformerEncoder,
                'encoder_hyperparams': TransformerEncoderParams({
                    'input_dim':self.current_dataset_config['in_dim'],
                    'd_model': 448,
                    'n_heads': 4,
                    'n_layers': 5,
                    'dropout': 0.1
                }),
                'decoder_class': TransformerDecoder,
                'decoder_hyperparams': TransformerDecoderParams({
                    'input_dim':self.current_dataset_config['in_dim'],
                    'd_model': 448,
                    'n_heads': 4,
                    'n_layers': 5,
                    'dropout': 0.1
                }),
                'data_handler_params': self.current_dataset_config,
                'model_class': Koopman,
                "input_before": 144,
                "input_after": 0,
                "prediction_length": 50,
                'train_params': TrainerHyperparams({
                    'lr': 0.0001,
                    'device': device,
                    'batch_size': 64,
                    'epochs': 200,
                    'model_type': 'Koopman',
                    'lr_decay': True, 
                    'stitching': False,
                    'stitching_anchor_nr': 32,
                }),
                'ode_params': {'in_out_dim': 448,
                    'hidden_dim': 256
                }
            }),
            

        ]

        # L ablations for context length
        Ls = [5,10,15, 32, 64, 96, 144]

        for L in Ls:
            self.model_configs.extend([
                # --- MLP(L) ---
                MLPParams({
                    'model_name': f'mlp{L}',
                    'encoder_class': MLPEncoder,
                    'encoder_hyperparams': MLPEncoderParams({
                        'in_dim': L * self.current_dataset_config['in_dim'],
                        'hidden_dim': 256,
                        'out_dim': 256,
                    }),
                    'decoder_class': MLPDecoder,
                    'decoder_hyperparams': MLPDecoderParams({
                        'in_dim': 256,
                        'hidden_dim': 150,
                        'out_dim': 50 * self.current_dataset_config['in_dim'],
                    }),
                    'data_handler_params': self.current_dataset_config,
                    'model_class': MLP,
                    "input_before": L,
                    "input_after": 0,
                    "prediction_length": 50,
                    'train_params': TrainerHyperparams({
                        'lr': 0.001,
                        'device': device,
                        'batch_size': 64,
                        'epochs': 200,
                        'model_type': 'MLP',
                        'lr_decay': True,
                        'stitching': False,
                        'stitching_anchor_nr': 32,
                    }),
                }),

                # --- Transformer(L) ---
                TransformerParams({
                    'model_name': f'transformer{L}',
                    'encoder_class': TransformerEncoder,
                    'encoder_hyperparams': TransformerEncoderParams({
                        'input_dim': self.current_dataset_config['in_dim'],
                        'd_model': 128,
                        'n_heads': 8,
                        'n_layers': 2,
                        'dropout': 0.1
                    }),
                    'decoder_class': TransformerDecoder,
                    'decoder_hyperparams': TransformerDecoderParams({
                        'input_dim': self.current_dataset_config['in_dim'],
                        'd_model': 128,
                        'n_heads': 8,
                        'n_layers': 2,
                        'dropout': 0.1
                    }),
                    'data_handler_params': self.current_dataset_config,
                    'model_class': TransformerModel,
                    "input_before": L,
                    "input_after": 0,
                    "prediction_length": 50,
                    'train_params': TrainerHyperparams({
                        'lr': 0.001,
                        'device': device,
                        'batch_size': 64,
                        'epochs': 200,
                        'model_type': 'Transformer',
                        'lr_decay': True,
                        'stitching': False,
                        'stitching_anchor_nr': 32,
                    }),
                }),

                # --- RNN AutoReg(L) ---
                RNNAutoRegParams({
                    'model_name': f'rnnautoreg{L}',
                    'encoder_class': RNNAutoRegEncoder,
                    'encoder_hyperparams': RNNAutoRegEncoderParams({
                        'input_dim': self.current_dataset_config['in_dim'],
                        'hidden_dim': 128,
                        'num_layers': 3,
                        'dropout': 0.12
                    }),
                    'decoder_class': RNNAutoRegDecoder,
                    'decoder_hyperparams': RNNAutoRegDecoderParams({
                        'input_dim': self.current_dataset_config['in_dim'],
                        'hidden_dim': 128,
                        'num_layers': 3,
                        'output_dim': self.current_dataset_config['in_dim'],
                        'dropout': 0.12
                    }),
                    'data_handler_params': self.current_dataset_config,
                    'model_class': RNNAutoReg,
                    "input_before": L,
                    "input_after": 0,
                    "prediction_length": 50,
                    'train_params': TrainerHyperparams({
                        'lr': 0.006,
                        'device': device,
                        'batch_size': 64,
                        'epochs': 200,
                        'model_type': 'RNN',
                        'lr_decay': True,   
                        'stitching': False,
                        'stitching_anchor_nr': 32,
                    }),
                }),
            ])




    def get_dataset_config_by_dataset_name(self, name):
        # Search through the list of configs and return the one with the matching "name"
        for config in self.dataset_configs:
            if config["dataset_name"] == name:
                return config
        return None  # Return None if no config is found with the given name
    
    def get_current_dataset_config(self):
        return self.current_dataset_config
    
   
    
    def get_model_config_by_model_name(self, name):
        # Search through the list of configs and return the one with the matching "name"
        for config in self.model_configs:
            if config["model_name"] == name:
                return config
        return None  # Return None if no config is found with the given name