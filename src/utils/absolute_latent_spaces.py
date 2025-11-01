import torch.nn.functional as F
import csv

#------------Helper------------
def compute_cosine(absolute_spaces:list)->float:
    """Computes the average cosine similarity between
    the absolute spaces given in the list."""
    sims = []
    n = len(absolute_spaces)
    for i in range(n):
        for j in range(i,n):
            if i == j:
                continue
            sim = F.cosine_similarity(absolute_spaces[i], absolute_spaces[j]).mean()
            sims.append(sim.item())
    return float(sum(sims) / len(sims)) #avg

def save_csv_to_exp(similarities:dict, experiment_name:str):
    """Write results to a csv file."""
    file_path = f"results/{experiment_name}/absolute_similarities.csv"
    with open(file_path, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(["model", "mean_cos"]) #header
        for key, value in similarities.items():
            writer.writerow([key, value])
    print(f"Saved results in {file_path}")

#------------compute absolute space similarity across seeds------------
def cosine_absolute_spaces(models:list, absolute_spaces:list, experiment_name:str):
    """Clusters given experiment models in their familiy and then
    computes the alignment across seeds within family.
    Outputs the mean cosine similarity for cluster."""

    model_cluster = {
        "mlp": [],
        "transformer": [],
        "oldrnn": [],
        "rnnautoreg": [],
        "nodemlp": [],
        "nodernnautoreg": [],
        "nodetransformer": [],
        "koopmanmlp": [],
        "koopmanrnnautoreg": [],
        "koopmantransformer": [],
    }
    for model, space in zip(models, absolute_spaces):
        #cluster the spaces by model
        model_name = model.hyperparams.get("model_name", "")
        for key in model_cluster.keys():
            if model_name.startswith(key):
                model_cluster[key].append(space)
   
    sim_dict = {}
    for key in model_cluster:
        spaces = model_cluster.get(key, [])
        if len(spaces) != 0:
            mean_sim = compute_cosine(spaces)
            sim_dict[key] = mean_sim
    
    save_csv_to_exp(sim_dict, experiment_name)
    
