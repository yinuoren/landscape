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
from sklearn.decomposition import PCA

## parameters
model_class = "mini"
model_name = "vgg-small"
dataset = "cifar10"
data_subset_classes = [10]
ckpt_prefix = "results/cifar10/small_data/03_sub"
ckpt_exp_list = [str(i) for i in range(600, 1001, 25)]
ckpt_step_list = [str(i) for i in range(0, 40000, 4000)]
pca_n_components = 5
save_path = "results/cifar10/small_data/03_sub"
save_name = "pca_info_reg_3rd"


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
for i in tqdm(ckpt_exp_list):
    for j in ckpt_step_list:
        ckpt_name = ckpt_prefix + "/" + i + "/ckpt/step" + j + ".tar"
        state_dict = torch.load(ckpt_name, map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
        p, info = param_to_vec(model)
        Params.append(p)
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

np.savez(save_path+"/"+save_name, pc=pc, coords=coords, center=pca.mean_,
         reg_coeff=quad_reg_coeff, info=info)

# print the range of coordinates
print("the range of the 1st principle component: ({}, {})".format(np.min(coords[:,0]), np.max(coords[:,0])))
print("the range of the 2nd principle component: ({}, {})".format(np.min(coords[:,1]), np.max(coords[:,1])))
