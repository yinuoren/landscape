import json
import os
import shutil
import numpy as np
import deepdish as dd
import torch
import torch.nn as nn
from utils import load
from utils import optimize
from utils import flags
from utils import landscape
from tqdm import tqdm
from glob import glob
from sklearn.decomposition import PCA
import argparse

# load parameters
def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help='Directory to save checkpoints and features (default: "results")',
    )
    argparser.add_argument(
        "--experiment",
        type=str,
        default="",
        help='name used to save results (default: "")',
    )
    argparser.add_argument(
        "--expid", 
        type=str, 
        default="", 
        help='name used to save results (default: "")'
    )
    argparser.add_argument(
        "--save-name",
        type=str,
        default="pca_info",
        help="the name of the file to save PCA information"
    )
    args = argparser.parse_args()
    return args

## parameters
args = get_args()
save_path = args.save_dir + "/" + args.experiment + "/" + str(args.expid)
json_file_name = save_path + "/hyperparameters.json"
with open(json_file_name, "r") as f:
    ARGS = json.load(f)
    
# print(ARGS)
model_class = ARGS['model_class']
model_name = ARGS['model']
dataset = ARGS['dataset']
data_subset_classes = ARGS['data_subset_classes']

# ckpt_exp_list = [str(i) for i in range(600, 1001, 25)]
total_steps = ARGS['data_subset_classes'][0] * ARGS['data_subset_ndata'][0] * ARGS['epochs'] // ARGS['train_batch_size']
ckpt_step_list = [str(i) for i in range(0, total_steps, ARGS['save_freq'])]
pca_n_components = 5

save_name = args.save_name

# functions
def param_to_vec(model):
    P = []
    pcount = 0
    info = {}
    for name, param in model.named_parameters():
        p = param.data.cpu().numpy()
        shape = p.shape
        size = np.prod(shape)
        P.append(p)
        info[name] = [pcount, pcount+size, shape]
        pcount += size
    P = np.concatenate([np.reshape(p, (-1)) for p in P], axis=0)
    return P, info


def vec_to_param(model, pdir, info, device):
    """
    pdir is put into param of model
    """
    for name, param in model.named_parameters():
        temp = info[name]
        p = np.reshape(pdir[temp[0]:temp[1]], temp[2])
        param.data = torch.tensor(p).to(device)
    return model


## collect parameters from checkpoints
# build the model
input_shape, num_classes = load.dimension(dataset, data_subset_classes)
device = load.device("0")
print("Creating {}-{} models...".format(model_class, model_name))
model = load.model(model_name, model_class)(
    input_shape=input_shape,
    num_classes=num_classes,
)
model = model.to(device)
# load checkpoints
print("Loading all parameters...")
Params = []
    
def load_parameters(save_path, model, device):
    ckpt_path = f"{save_path}/ckpt"
    ckpt_names = glob(f"{ckpt_path}/*.tar")
    
    # Extract step numbers from checkpoint filenames
    ckpt_steps = [int(name.split(".tar")[0].split("step")[1]) for name in ckpt_names]
    
    # Iterate through sorted checkpoints based on step numbers
    for ckpt_name, _ in tqdm(sorted(list(zip(ckpt_names, ckpt_steps)), key=lambda x: x[1])):
        state_dict = torch.load(ckpt_name, map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
        p, info = param_to_vec(model)
        Params.append(p)
    return Params, info
    
Params, info = load_parameters(save_path, model, device)

# for j in ckpt_step_list:
#     ckpt_name = save_path + "/ckpt/step" + j + ".tar"
#     state_dict = torch.load(ckpt_name, map_location=device)
#     model.load_state_dict(state_dict["model_state_dict"])
#     p, info = param_to_vec(model)
#     Params.append(p)

def load_parameters_subpath(save_path, model, device):
    sub_paths = sorted(glob(f"{save_path}/sub/*"), key=lambda x: int(x.split('/')[-1]))

    for sub_path in tqdm(sub_paths, desc="Processing subpaths"):
        # Get all checkpoints for the current subpath
        ckpt_names = glob(f"{sub_path}/ckpt/*.tar")
        ckpt_steps = [int(name.split(".tar")[0].split("step")[1]) for name in ckpt_names]
        
        for ckpt_name, _ in tqdm(sorted(list(zip(ckpt_names, ckpt_steps)), key=lambda x: x[1]), desc="Loading checkpoints", leave=False):
            state_dict = torch.load(ckpt_name, map_location=device)
            model.load_state_dict(state_dict["model_state_dict"])
            p, info = param_to_vec(model)
            Params.append(p)
    return Params, info

Params, info = load_parameters_subpath(save_path, model, device)

# for j in range(args.subpath_start, args.subpath_end, args.subpath_save_freq):
#     json_file_name = save_path + "/sub/" + str(j) + "/hyperparameters.json"
#     with open(json_file_name, "r") as f:
#         ARGS_sub = json.load(f)
        
#     total_steps_sub = ARGS_sub['data_subset_classes'][0] * ARGS_sub['data_subset_ndata'][0] * ARGS_sub['epochs'] // ARGS_sub['train_batch_size']
#     ckpt_step_list_sub = [str(i) for i in range(0, total_steps_sub, ARGS['save_freq'])]
#     for k in ckpt_step_list_sub:
#         ckpt_name = save_path + "/sub/" + str(j) + "/ckpt/step" + str(k) + ".tar"
#         state_dict = torch.load(ckpt_name, map_location=device)
#         model.load_state_dict(state_dict["model_state_dict"])
#         p, info = param_to_vec(model)
#         Params.append(p)
    
    
Params = np.concatenate([np.reshape(p, (1,-1)) for p in Params], axis=0)
print("Loading finished, getting ({}, {}) parameters".format(Params.shape[0], Params.shape[1]))

## perform PCA, compute and save coordinates for checkpoints
print("PCA...")
pca = PCA(n_components=pca_n_components)
pca.fit(Params)
print("%d components are computed, "%pca_n_components,
      "variance explained: ", pca.explained_variance_ratio_)
pc = pca.components_
coords = pca.transform(Params)

####
# regress 3rd coordinate using quadratic functions of the 1st coordinate
Y = coords[:,2]
X = np.ones((len(Y), 3))
X[:,0] = coords[:,0]**2
X[:,1] = coords[:,0]
quad_reg_coeffs = np.linalg.lstsq(X,Y)[0]
####

np.savez(save_path+"/"+save_name, pc=pc, coords=coords, center=pca.mean_, reg_coeff=quad_reg_coeffs, info=info)

# print the range of coordinates
print("the range of the 1st principle component: ({}, {})".format(np.min(coords[:,0]), np.max(coords[:,0])))
print("the range of the 2nd principle component: ({}, {})".format(np.min(coords[:,1]), np.max(coords[:,1])))
