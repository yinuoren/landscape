import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
import torch_optimizer as custom_optim
import torch.nn.functional as F
from utils import load
from utils import optimize
from tqdm import tqdm
from sklearn.decomposition import PCA


def landscape(args, device, train_loader, test_loader, save_path):
    if args.data_subset:
        input_shape, num_classes = load.dimension(args.dataset, args.data_subset_classes)
    else:
        input_shape, num_classes = load.dimension(args.dataset)
    if len(args.dir_types)==1:
        model = get_direction(args.dir_types[0], 0, args, device, train_loader, input_shape, num_classes)
        if args.ckpt_center:
            replace_param(model, args.ckpt_center, device, args.model, args.model_class, input_shape, num_classes)
        landscape1d(model=model,
                    loss=args.loss,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    interval_l=args.interval[0],
                    interval_r=args.interval[1],
                    N=args.N,
                    save_path=save_path,
                    seed=args.seed
        )
    elif len(args.dir_types)==2:
        model0 = get_direction(args.dir_types[0], 0, args, device, train_loader, input_shape, num_classes)
        model1 = get_direction(args.dir_types[1], 1, args, device, train_loader, input_shape, num_classes)
        if args.ckpt_center:
            replace_param(model0, args.ckpt_center, device, args.model, args.model_class, input_shape, num_classes)
        landscape2d(model0=model0,
                    model1=model1,
                    loss=args.loss,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    interval_l_0=args.interval[0],
                    interval_r_0=args.interval[1],
                    interval_l_1=args.interval_1[0],
                    interval_r_1=args.interval_1[1],
                    N_0=args.N,
                    N_1=args.N_1,
                    save_path=save_path,
                    seed=args.seed
        )
    else:
        raise ValueError("direction type not implemented!")


def get_direction(dir_type, ndir, args, device, dataloader, input_shape, num_classes):
    if dir_type == "interpolate":
        if ndir == 0:
            model = direction_interpolate(device=device,
                                          ckpt_start=args.model_ckpt_start,
                                          ckpt_end=args.model_ckpt_end,
                                          model_name=args.model,
                                          model_class=args.model_class,
                                          input_shape=input_shape,
                                          num_classes=num_classes
                    )
        elif ndir == 1:
            model = direction_interpolate(device=device,
                                          ckpt_start=args.model_ckpt_start_1,
                                          ckpt_end=args.model_ckpt_end_1,
                                          model_name=args.model,
                                          model_class=args.model_class,
                                          input_shape=input_shape,
                                          num_classes=num_classes
                    )
    elif dir_type == "grad":
        if ndir == 0:
            model = direction_grad(device=device,
                                   ckpt=args.model_ckpt_start,
                                   dataloader=dataloader,
                                   norm=args.normalization,
                                   model_name=args.model,
                                   model_class=args.model_class,
                                   input_shape=input_shape,
                                   num_classes=num_classes,
                                   loss=args.loss
                    )
        elif ndir == 1:
            model = direction_grad(device=device,
                                   ckpt=args.model_ckpt_start_1,
                                   dataloader=dataloader,
                                   norm=args.normalization_1,
                                   model_name=args.model,
                                   model_class=args.model_class,
                                   input_shape=input_shape,
                                   num_classes=num_classes,
                                   loss=args.loss
                    )
    elif dir_type == "random":
        if ndir == 0:
            model = direction_random(device=device,
                                     ckpt=args.model_ckpt_start,
                                     model_name=args.model,
                                     model_class=args.model_class,
                                     input_shape=input_shape,
                                     num_classes=num_classes
                    )
        elif ndir == 1:
            model = direction_random(device=device,
                                     ckpt=args.model_ckpt_start_1,
                                     model_name=args.model,
                                     model_class=args.model_class,
                                     input_shape=input_shape,
                                     num_classes=num_classes
                    )
    elif dir_type == "pca":
        if ndir == 0:
            model = direction_pca(device=device,
                                  ckpt_center=args.model_ckpt_start,
                                  ckpt_prefix=args.pca_ckpt_prefix,
                                  ckpt_list=args.pca_ckpt_list,
                                  n_pc=args.n_pc,
                                  model_name=args.model,
                                  model_class=args.model_class,
                                  input_shape=input_shape,
                                  num_classes=num_classes
                    )
        elif ndir == 1:
            model = direction_pca(device=device,
                                  ckpt_center=args.model_ckpt_start_1,
                                  ckpt_prefix=args.pca_ckpt_prefix_1,
                                  ckpt_list=args.pca_ckpt_list_1,
                                  n_pc=args.n_pc_1,
                                  model_name=args.model,
                                  model_class=args.model_class,
                                  input_shape=input_shape,
                                  num_classes=num_classes
                    )
    else:
        raise ValueError("the direction type is not implemented!")
    return model


def direction_interpolate(device, ckpt_start, ckpt_end, model_name, model_class, input_shape, num_classes):
    # initialize
    print("Computing interpolation direction...")
    # load model
    print("Creating {}-{} models...".format(model_class, model_name))
    print("Loading starting checkpoint: " + ckpt_start)
    model0 = create_and_load_model(model_name, model_class, input_shape, num_classes, device, ckpt_start)
    print("Loading ending checkpoint: " + ckpt_end)
    model1 = create_and_load_model(model_name, model_class, input_shape, num_classes, device, ckpt_end)
    # compute direction
    for p0, p1 in zip(model0.parameters(), model1.parameters()):
        p0.grad = p1.data - p0.data
    return model0
    

def direction_grad(device, ckpt, dataloader, norm, model_name, model_class, loss, input_shape, num_classes):
    # initialize
    print("Computing gradient direction...")
    # load model
    print("Creating {}-{} models...".format(model_class, model_name))
    print("Loading checkpoint: " + ckpt)
    model = create_and_load_model(model_name, model_class, input_shape, num_classes, device, ckpt)
    # build loss
    loss = load.loss(loss)
    # compute gradient
    model.train()
    total_samples = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        train_loss = loss(output, target) * data.size(0)
        train_loss.backward()
        total_samples += data.size(0)
    if norm:
        gnorm = 0
        for p in model.parameters():
            gnorm += torch.sum(p.grad*p.grad)
        gnorm = torch.sqrt(gnorm)
    else:
        gnorm = total_samples
    for p in model.parameters():
        p.grad /= gnorm
    return model
    

def direction_random(device, ckpt, model_name, model_class, input_shape, num_classes):
    # initialize
    print("Computing random direction...")
    # load model
    print("Creating {}-{} models...".format(model_class, model_name))
    print("Loading checkpoint: " + ckpt)
    model = create_and_load_model(model_name, model_class, input_shape, num_classes, device, ckpt)
    # generate random direction
    gnorm = 0
    for p in model.parameters():
        p.grad = torch.randn_like(p.data)
        gnorm += torch.sum(p.grad*p.grad)
    gnorm = torch.sqrt(gnorm)
    for p in model.parameters():
        p.grad /= gnorm
    return model
    

def direction_pca(device, ckpt_center, ckpt_prefix, ckpt_list, n_pc, model_name, model_class, input_shape, num_classes):
    # initialize
    print("Computing PCA direction...")
    # load center model
    print("Creating {}-{} models...".format(model_class, model_name))
    print("Loading checkpoint: " + ckpt_center)
    model = create_and_load_model(model_name, model_class, input_shape, num_classes, device, ckpt_center)
    # collect parameter vectors
    print("Loading parameter list...")
    model_temp = create_and_load_model(model_name, model_class, input_shape, num_classes, device, ckpt_center)
    start, end, step = ckpt_list[0], ckpt_list[1], ckpt_list[2]
    Param = []
    for i in tqdm(range(start, end+1, step)):
        ckpt_name = ckpt_prefix + str(i) + ".tar"
        state_dict = torch.load(ckpt_name, map_location=device)
        model_temp.load_state_dict(state_dict["model_state_dict"])
        p, info = param_to_vec(model_temp)
        Param.append(p)
    Param = np.concatenate([np.reshape(p, (1,-1)) for p in Param], axis=0)
    # PCA
    print("PCA...")
    pca = PCA(n_components=n_pc)
    pca.fit(Param)
    print("pca explained variance ratio: %g" % np.sum(pca.explained_variance_ratio_))
    pdir = np.reshape(pca.components_[n_pc-1], (-1))
    # vectors to parameters, add to grad
    model = vec_to_param(model, pdir, info, device)
    return model


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
    pdir is put into grad of model
    """
    for name, param in model.named_parameters():
        temp = info[name]
        p = np.reshape(pdir[temp[0]:temp[1]], temp[2])
        param.grad = torch.tensor(p).to(device)
    return model

#=================================================================
def landscape1d(model, loss, device, train_loader, test_loader, interval_l, interval_r, N, save_path, seed=0):
    # initailize
    print("Evaluating landscape...")
    #build loss
    loss = load.loss(loss)
    #build grid points
    x = np.linspace(interval_l, interval_r, N)
    #evaluate landscape
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    for p in model.parameters():
        p.data = p.data + x[0]*p.grad
    for i in tqdm(range(N)):
        torch.manual_seed(seed)
        trloss, tracc, _ = optimize.eval(model, loss, train_loader, device, verbose=0, epoch=0, diag=True)
        torch.manual_seed(seed)
        teloss, teacc, _ = optimize.eval(model, loss, test_loader, device, verbose=0, epoch=0, diag=True)
        train_loss.append(trloss)
        train_acc.append(tracc)
        test_loss.append(teloss)
        test_acc.append(teacc)
        if i != N-1:
            for p in model.parameters():
                p.data = p.data + (x[i+1]-x[i]) * p.grad
    # save results
    save_fcn(save_path, train_loss, train_acc, test_loss, test_acc, x)
    print("Complete!")


def landscape2d(model0, model1, loss, device, train_loader, test_loader, interval_l_0, interval_r_0, interval_l_1, interval_r_1, N_0, N_1, save_path, seed=0):
    # initailize
    print("Evaluating landscape...")
    #build loss
    loss = load.loss(loss)
    #build grid points
    x = np.linspace(interval_l_0, interval_r_0, N_0)
    y = np.linspace(interval_l_1, interval_r_1, N_1)
    #evaluate landscape
    train_loss, train_acc = np.zeros((N_0, N_1)), np.zeros((N_0, N_1))
    test_loss, test_acc = np.zeros((N_0, N_1)), np.zeros((N_0, N_1))
    for p0, p1 in zip(model0.parameters(), model1.parameters()):
        p1.data = p0.data.clone()
    for i in tqdm(range(N_0)):
        for j in tqdm(range(N_1)):
            for p0, p1 in zip(model0.parameters(), model1.parameters()):
                p0.data = p1.data + x[i]*p0.grad + y[i]*p1.grad
            torch.manual_seed(seed)
            trloss, tracc, _ = optimize.eval(model0, loss, train_loader, device, verbose=0, epoch=0, diag=True)
            torch.manual_seed(seed)
            teloss, teacc, _ = optimize.eval(model0, loss, test_loader, device, verbose=0, epoch=0, diag=True)
            train_loss[i,j] = trloss
            train_acc[i,j] = tracc
            test_loss[i,j] = teloss
            test_acc[i,j] = teacc
    # save results
    save_fcn(save_path, train_loss, train_acc, test_loss, test_acc, x, y)
    print("Complete!")


#=================================================================
def replace_param(model, ckpt_center, device, model_name, model_class, input_shape, num_classes):
    """
    Replace the parameters of the input model using ckpt_center
    """
    print("Replacing model parameter to checkpoint: %s" % ckpt_center)
    model1 = create_and_load_model(model_name, model_class, input_shape, num_classes, device, ckpt_center)
    for p0, p1 in zip(model.parameters(), model1.parameters()):
        p0.data = p1.data.clone()
    return model

def create_and_load_model(model_name, model_class, input_shape, num_classes, device, load_path):
    model = load.model(model_name, model_class)(
        input_shape=input_shape,
        num_classes=num_classes,
    )
    model = model.to(device)
    pretrained_dict = torch.load(load_path)
    model.load_state_dict(pretrained_dict["model_state_dict"])
    return model
    
def save_fcn(save_path, train_loss, train_acc, test_loss, test_acc, x, y=None):
    print("Saving results to " + save_path)
    if y:
        np.savez(save_path+"/landscape.npz", train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc, x=x, y=y)
    else:
        np.savez(save_path+"/landscape.npz", train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc, x=x)


