import numpy as np
import torch
from tqdm import tqdm
#import matplotlib.pyplot as plt

ckpt_folder = "results/cifar100/resnet18/sgdm/01/ckpt"

def get_param(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    param_dict = state_dict["model_state_dict"]
    P = []
    for k, v in param_dict.items():
        if ("weight" in k) or ("bias" in k):
            P.append(np.reshape(v.numpy(), (-1)))
    P = np.concatenate(P)
    return P

D = 11220132   # number of params for resnet18
n = 391  # number of iterations per epoch
N = 181  # number of checkpoints

Param = np.zeros((N, D))
for i in tqdm(range(N)):
    P = get_param(ckpt_folder+"/step"+str(n*i)+".tar")
    Param[i] = P

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

#pca = PCA(n_components=5)
#pca.fit(Param[70:119])
#print(pca.explained_variance_ratio_)
#Comp = pca.components_
#Pred = pca.transform(Param)
model = Isomap(n_components=2)
P_transformed = model.fit_transform(X=Param)

np.savez("results/param_isomap_sgdm.npz", p=P_transformed)

