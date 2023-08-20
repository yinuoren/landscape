import json
import os
import shutil
import numpy as np
import deepdish as dd
import torch
import torch.nn as nn
import argparse
from utils import load
from utils import optimize
from utils import flags
from utils import landscape
from tqdm import tqdm
from sklearn.decomposition import PCA

## parameters
# fixed parameters
model_class = "mini"
model_name = "vgg-small"
dataset = "cifar10"
data_dir = "data"
data_subset_classes = [10]
data_subset_ndata = [500]
data_subset_random = True
data_subset_seed = 87654
save_path = "results/cifar10/small_data/03_sub"
save_folder = "pca_landscape_reg_3rd"
load_name = "pca_info_reg_3rd"

# load parameters
def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--coord-range",
        type=float,
        nargs="*",
        default=[],
        help="the range of coordinates to evaluate the model, [l1,r1,l2,r2]"
    )
    argparser.add_argument(
        "--n",
        type=int,
        nargs="*",
        default=[],
        help="the number of equispaced points to evaluate in the ranges, [n1,n2]. Including both boundaries"
    )
    argparser.add_argument(
        "--exp-id",
        type=int,
        default=0,
        help="the id of the current experiment"
    )
    
    args = argparser.parse_args()
    return args
args = get_args()
l1, r1, l2, r2 = args.coord_range
n1, n2 = args.n

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


## compute the training and testing error on grids given by pca
# build the model
input_shape, num_classes = load.dimension(dataset, data_subset_classes)
device = load.device("0")
print("Creating {}-{} models...".format(model_class, model_name))
model = load.model(model_name, model_class)(
    input_shape=input_shape,
    num_classes=num_classes,
)
model = model.to(device)
loss = load.loss("mse")

# build the dataloader
train_loader = load.dataloader(
    dataset=dataset,
    batch_size=125,
    train=True,
    workers=4,
    datadir=data_dir,
    tpu=None,
    subset=True,
    subset_classes=data_subset_classes,
    subset_ndata=data_subset_ndata,
    subset_random=True,
    subset_seed=data_subset_seed,
)
test_loader = load.dataloader(
    dataset=dataset,
    batch_size=1000,
    train=False,
    test_with_train=True,
    workers=4,
    datadir=data_dir,
    tpu=None,
    subset=True,
    subset_classes=data_subset_classes,
    subset_ndata=data_subset_ndata,
)

X1 = np.linspace(l1, r1, n1)
X2 = np.linspace(l2, r2, n2)
pca_info = np.load(save_path+"/"+load_name+".npz", allow_pickle=True)
pc = pca_info["pc"]
info = pca_info["info"].item()
pc1 = np.reshape(pc[0], (-1))
pc2 = np.reshape(pc[1], (-1))
pc3 = np.reshape(pc[2], (-1))
center = pca_info["center"]
reg_coeff = pca_info["reg_coeff"]
a, b, c = reg_coeff[0], reg_coeff[1], reg_coeff[2]

train_loss, train_acc = np.zeros((n1,n2)), np.zeros((n1,n2))
test_loss, test_acc = np.zeros((n1,n2)), np.zeros((n1,n2))
for i in range(n1):
    print(i)
    for j in tqdm(range(n2)):
        p = center + X1[i]*pc1 + X2[j]*pc2 + (a*X1[i]**2+b*X1[i]+c)*pc3
        model = vec_to_param(model, p, info, device)
        trloss, tracc, _ = optimize.eval(model, loss, train_loader, device, verbose=0, epoch=0)
        teloss, teacc, _ = optimize.eval(model, loss, test_loader, device, verbose=0, epoch=0)
        train_loss[i,j] = trloss
        train_acc[i,j] = tracc
        test_loss[i,j] = teloss
        test_acc[i,j] = teacc
# save results
np.savez(save_path+"/"save_folder+"/pca_landscape_"+str(args.exp_id), X1=X1, X2=X2,
         train_loss=train_loss, train_acc=train_acc, test_loss=test_loss,
         test_acc=test_acc)


