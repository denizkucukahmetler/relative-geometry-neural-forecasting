#!/usr/bin/env python3
# Train ALL requested models per dataset using best params from CSV (no argparse).

import csv, json, re, torch, random, numpy as np
from pathlib import Path
from src.config_manager import ConfigManager
from src.training_manager import TrainingManager

CSV_PATH = Path("best_model_parameters.csv")
MODELS = [
    'mlp5',
    'mlp10',
    'mlp15',
    'mlp32',
    'mlp64',
    'mlp96',
    'mlp144',# base model with L=144 in original config
    'mlp192',
]

# Transformer ablations
MODELS += [
    'transformer5',
    'transformer10'
    'transformer15',
    'transformer32',
    'transformer64',
    'transformer96',
    'transformer144',
    'transformer192',
]

# RNN AutoReg ablations
MODELS += [
    'rnnautoreg5',
    'rnnautoreg10',
    'rnnautoreg15',
    'rnnautoreg32',
    'rnnautoreg64',
    'rnnautoreg96',
    'rnnautoreg144',
    'rnnautoreg192',
    'rnnautoreg'
]



DATASETS =["lorenz","double_pendulum","spiral", "logistic_map","pod","hopf","random_skew"] 
SEEDS    = range(1,2)
TAG      = "input_length"

def pick(d,*ks,default=None):
    for k in ks:
        if k in d: return d[k]
    return default

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# load best params {(model,dataset)->dict}, normalize model like "mlp1"->"mlp"
best = {}
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        m  = re.sub(r"\d+$", "", row["model_name"].strip().lower())
        ds = row["dataset_name"].strip()
        p  = json.loads((row["model_parameters"] or "").replace("'", '"'))
        best[(m, ds)] = p

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = TrainingManager(dev)

for seed in SEEDS:
    
    seed_all(seed)
    for ds in DATASETS:
        cfg = ConfigManager(ds, dev.type)
        MODEL_LIST, seeds = [], []
    

        for model in MODELS:
            base = re.sub(r"\d+$", "", model)  # "mlp144" -> "mlp"

            # use the BASE for best-params lookup
            if (base, ds) not in best:
                continue
            p = best[(base, ds)]

            seeds.append(int(pick(p, "seed", default=0) or 0))

            # keep using the original model name for config retrieval
            mc = cfg.get_model_config_by_model_name(model)

            if base == "mlp":
                hdE=int(pick(p,"hdE","encoder_hidden_dim",default=128))
                hdD=int(pick(p,"hdD","decoder_hidden_dim",default=128))
                L  =int(pick(p,"L","num_layers",default=1))
                do =float(pick(p,"do","dropout",default=0.0))
                lr =float(pick(p,"lr","learning_rate",default=1e-3))
                bs =int(pick(p,"bs","batch_size",default=64))
                mc["encoder_hyperparams"]["hidden_dim"]=hdE
                if "num_layers" in mc["encoder_hyperparams"]: mc["encoder_hyperparams"]["num_layers"]=L
                mc["decoder_hyperparams"]["hidden_dim"]=hdD
                if "dropout" in mc["decoder_hyperparams"]: mc["decoder_hyperparams"]["dropout"]=do
                mc["train_params"]["lr"]=lr; mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "oldrnn": #CHECKED
                hid=int(pick(p,"hid","hidden_dim","hd",default=128))
                nL =int(pick(p,"L","num_layers",default=2))
                do =float(pick(p,"do","dropout",default=0.1))
                tf =float(pick(p,"tf","teacher_force_prob",default=0.0))
                lr =float(pick(p,"lr","learning_rate",default=1e-3))
                bs =int(pick(p,"bs","batch_size",default=64))
                mc["encoder_hyperparams"]["hidden_dim"]=hid
                mc["encoder_hyperparams"]["num_layers"]=nL
                mc["encoder_hyperparams"]["dropout"]=do
                mc["decoder_hyperparams"]["hidden_size"]=hid
                mc["decoder_hyperparams"]["num_gru_layers"]=nL
                mc["decoder_hyperparams"]["dropout"]=do
                mc["train_params"]["lr"]=lr
                mc["train_params"]["batch_size"]=bs
                mc["train_params"]["teacher_force_prob"]=tf
                mc["experiment_tag"]=TAG

            elif base == "transformer": #CHECKED
                dm =int(pick(p,"dm","d_model",default=256))
                nh =int(pick(p,"nh","n_heads",default=4))
                nL =int(pick(p,"L","n_layers","nl",default=2))
                do =float(pick(p,"do","dropout",default=0.1))
                lr =float(pick(p,"lr","learning_rate",default=1e-3))
                bs =int(pick(p,"bs","batch_size",default=64))
                for side in ("encoder_hyperparams","decoder_hyperparams"):
                    mc[side]["d_model"]=dm; mc[side]["n_heads"]=nh; mc[side]["n_layers"]=nL; mc[side]["dropout"]=do
                mc["train_params"]["lr"]=lr; mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "rnnautoreg":  #CHECKED
                hid=int(pick(p,"hd","hid","hidden_dim",default=128))
                nL =int(pick(p,"L","layers","num_layers",default=2))
                do =float(pick(p,"do","dropout",default=0.1))
                tf =float(pick(p,"tf","teacher_force","teacher_force_prob",default=0.2))
                lr =float(pick(p,"lr","learning_rate",default=2e-3))
                bs =int(pick(p,"bs","batch_size",default=64))
                mc["encoder_hyperparams"]["hidden_dim"]=hid
                mc["encoder_hyperparams"]["num_layers"]=nL
                mc["encoder_hyperparams"]["dropout"]=do
                mc["decoder_hyperparams"]["hidden_dim"]=hid
                mc["decoder_hyperparams"]["num_layers"]=nL
                mc["decoder_hyperparams"]["dropout"]=do
                mc["decoder_hyperparams"]["teacher_force"]=tf
                mc["train_params"]["lr"]=lr
                mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "koopmanmlp": #CHECKED
                encH=int(pick(p,"enc_hidden","encoder_hidden_dim","eh",default=256))
                lat =int(pick(p,"latent_dim","latent","out_dim","encoder_out_dim","eo","lat",default=90))
                decH=int(pick(p,"dec_hidden","decoder_hidden_dim","dh",default=256))
                odeH=int(pick(p,"ode_hidden","koopman_hidden","koopman_hidden_dim","oh",default=128))
                lr  =float(pick(p,"lr","learning_rate",default=5e-4))
                bs  =int(pick(p,"bs","batch_size",default=64))
                mc["encoder_hyperparams"]["hidden_dim"]=encH
                mc["encoder_hyperparams"]["out_dim"]=lat
                mc["decoder_hyperparams"]["hidden_dim"]=decH
                mc["decoder_hyperparams"]["in_dim"]=lat
                mc["ode_params"]["in_out_dim"]=lat
                mc["ode_params"]["hidden_dim"]=odeH
                mc["train_params"]["lr"]=lr; mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "koopmantransformer": #CHECKED
                dm  =int(pick(p,"dm","d_model",default=256))
                nL  =int(pick(p,"L","n_layers","nl",default=2))
                ode =int(pick(p,"ode","ode_hd","oh",default=128))
                do  =float(pick(p,"do","dropout",default=0.1))
                #lr  =float(pick(p,"lr","learning_rate",default=1e-3))
                bs  =int(pick(p,"bs","batch_size",default=64))
                for side in ("encoder_hyperparams","decoder_hyperparams"):
                    mc[side]["d_model"]=dm; mc[side]["n_layers"]=nL; mc[side]["dropout"]=do
                mc["ode_params"]["hidden_dim"]=ode
                mc["ode_params"]["in_out_dim"]=dm
                #mc["train_params"]["lr"]=lr; 
                mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "koopmanrnnautoreg":  #CHECKED
                hd  =int(pick(p,"hd","hid","hidden_dim",default=128))
                nL  =int(pick(p,"L","layers","num_layers",default=2))
                ode =int(pick(p,"ode","ode_hd","oh",default=128))
                lat =int(pick(p,"lat","latent","latent_dim","out_dim","encoder_out_dim","eo",default=64))
                lr  =float(pick(p,"lr","learning_rate",default=1e-3))
                bs  =int(pick(p,"bs","batch_size",default=64))
                mc["encoder_hyperparams"]["hidden_dim"]=hd
                mc["encoder_hyperparams"]["num_layers"]=nL
                mc["decoder_hyperparams"]["hidden_dim"]=hd
                mc["decoder_hyperparams"]["num_layers"]=nL
                mc["ode_params"]["hidden_dim"]=ode
                mc["ode_params"]["in_out_dim"]=lat
                mc["train_params"]["lr"]=lr; mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "nodetransformer": #CHECKED
                dm  =int(pick(p,"dm","d_model",default=256))
                nL  =int(pick(p,"L","n_layers","nl",default=2))
                ode =int(pick(p,"ode","ode_hd","oh",default=128))
                do  =float(pick(p,"do","dropout",default=0.1))
                lr  =float(pick(p,"lr","learning_rate",default=1e-3))
                bs  =int(pick(p,"bs","batch_size",default=64))
                for side in ("encoder_hyperparams","decoder_hyperparams"):
                    mc[side]["d_model"]=dm; mc[side]["n_layers"]=nL; mc[side]["dropout"]=do
                mc["ode_params"]["hidden_dim"]=ode
                mc["ode_params"]["in_out_dim"]=dm
                mc["train_params"]["lr"]=lr; mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "nodernnautoreg":  #CHECKED
                hd  =int(pick(p,"hd","hid","hidden_dim",default=128))
                nL  =int(pick(p,"L","layers","num_layers",default=2))
                ode =int(pick(p,"ode","ode_hd","oh",default=128))
                tf  =float(pick(p,"tf","teacher_force","teacher_force_prob",default=0.0))
                lr  =float(pick(p,"lr","learning_rate",default=1e-3))
                bs  =int(pick(p,"bs","batch_size",default=64))
                mc["encoder_hyperparams"]["hidden_dim"]=hd
                mc["encoder_hyperparams"]["num_layers"]=nL
                mc["decoder_hyperparams"]["hidden_dim"]=hd
                mc["decoder_hyperparams"]["num_layers"]=nL
                mc["decoder_hyperparams"]["teacher_force"]=tf
                mc["ode_params"]["in_out_dim"]=hd
                mc["ode_params"]["hidden_dim"]=ode
                mc["train_params"]["lr"]=lr; mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "nodemlp":  #CHECKED
                eh =int(pick(p,"eh","enc_hidden","encoder_hidden_dim",default=128))
                eo =int(pick(p,"eo","enc_out","latent","latent_dim","out_dim","encoder_out_dim",default=64))
                oh =int(pick(p,"oh","ode_hidden","ode","ode_hd",default=128))
                lr =float(pick(p,"lr","learning_rate",default=1e-3))
                bs =int(pick(p,"bs","batch_size",default=64))
                mc["encoder_hyperparams"]["hidden_dim"]=eh
                mc["encoder_hyperparams"]["out_dim"]=eo
                mc["decoder_hyperparams"]["in_dim"]=eo
                mc["decoder_hyperparams"]["hidden_dim"]=eh
                mc["ode_params"]["in_out_dim"]=eo
                mc["ode_params"]["hidden_dim"]=oh
                mc["train_params"]["lr"]=lr; mc["train_params"]["batch_size"]=bs
                mc["experiment_tag"]=TAG

            elif base == "esn":
                nr =int(pick(p,"nr","n_reservoir",default=1000))
                sr =float(pick(p,"sr","spectral_radius",default=1.0))
                sp =float(pick(p,"sp","sparsity",default=0.1))
                ns =float(pick(p,"ns","noise",default=0.0))
                ins=float(pick(p,"is","input_scaling","input_scale",default=0.1))
                ra =float(pick(p,"ra","ridge_alpha","ridge",default=1e-3))
                mc["n_reservoir"]=nr
                mc["spectral_radius"]=sr
                mc["sparsity"]=sp
                mc["noise"]=ns
                mc["input_scaling"]=ins
                mc["ridge_alpha"]=ra
                mc["experiment_tag"]=TAG

            else:
                print("NONE MODEL or MODEL UNKNOWN:",model)
                

            MODEL_LIST.append([model, 5])

        if not MODEL_LIST:
            continue

        s = next((x for x in seeds if x), 0)
        if s: seed_all(s)

        exp = f"all_{ds}_{TAG}"
        print("▶", exp, "models:", [m for m,_ in MODEL_LIST])
        trainer.train_multiple_models(exp, cfg, MODEL_LIST)
