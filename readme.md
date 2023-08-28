# Landscape

This code is used to visualize the landscape of several widely-used neural network architectures in image classification.

## Usage

### Training

- Train one model

> python3 train.py --save-dir results/cifar10 --experiment small_data --expid 02 --overwrite --model vgg-small --model-class mini --data-dir data --dataset cifar10  --workers 4 --train-batch-size 125 --test-batch-size 250 --loss mse --optimizer momentum --momentum 0.9 --epochs 3000 --lr 0.1 --seed 7 --save-freq 400 --eval-mid-epoch --wd 0.0001 --data-subset --data-subset-classes 10 --data-subset-ndata 500 --test-with-train --data-subset-random --data-subset-seed 23479

- Train multiple models

> sh submit_array.sh

### PCA

> python3 param_pca.py --save-dir results/cifar10 --experiment small_data --expid 02 --save-name pca_info 

### Compute the landscape

> #SBATCH --array=11-20
>
> python3 landscape_pca.py --save-dir results/cifar10 --experiment small_data --expid 01 --load-name pca_info --save-name pca_landscape --coord-range -20 21 -15 15 --n 42 31 --exp-id \${SLURM_ARRAY_TASK_ID}