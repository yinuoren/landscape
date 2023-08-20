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


def main(ARGS):
    print_fn = print

    ## Set Save Path ##
    if not ARGS.landscape_save_path:
        print_fn("WARNING: no save path provided, using experiment path for save path. May cause error if experiment path is not provided.")
        save_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}"
    else:
        save_path = ARGS.landscape_save_path
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print_fn("The save path alraedy exists, results are directly saved in.")

    ## Save Args ##
    filename = save_path + "/hyperparameters.json"
    with open(filename, "w") as f:
        json.dump(ARGS.__dict__, f, sort_keys=True, indent=4)

    ## Random Seed and Device ##
    torch.manual_seed(ARGS.seed)
    device = load.device(ARGS.gpu, tpu=ARGS.tpu)

    ## Data ##
    print_fn("Loading {} dataset.".format(ARGS.dataset))
            
    train_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.train_batch_size,
        train=True,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
        tpu=ARGS.tpu,
        subset=ARGS.data_subset,
        subset_classes=ARGS.data_subset_classes,
        subset_ndata=ARGS.data_subset_ndata,
    )
    test_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.test_batch_size,
        train=False,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
        tpu=ARGS.tpu,
        subset=ARGS.data_subset,
        subset_classes=ARGS.data_subset_classes,
        subset_ndata=ARGS.data_subset_ndata,
    )
    
    landscape.landscape(args=ARGS,
                        device=device,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        save_path=save_path
    )
    

if __name__ == "__main__":
    parser = flags.landscape()
    ARGS = parser.parse_args()
    main(ARGS)
