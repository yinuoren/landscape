import argparse


def str_list(x):
    return x.split(",")


def default():
    parser = argparse.ArgumentParser(description="AdaP")
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
        help='name used to save results (default: "")',
    )
    parser.add_argument(
        "--expid", type=str, default="", help='name used to save results (default: "")'
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help='Directory to save checkpoints and features (default: "results")',
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="GPU device to use. Must be a single int or "
        "a comma separated list with no spaces (default: 0)"
    )
    parser.add_argument(
        "--tpu", type=str, default=None, help="Name of the TPU device to use",
    )
    parser.add_argument(
        "--overwrite", dest="overwrite", action="store_true", default=False
    )
    return parser


def model_flags(parser):
    model_args = parser.add_argument_group("model")
    model_args.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=[
            "logistic",
            "fc",
            "fc-bn",
            "conv",
            "vgg-small",
            "vgg-small-bn",
            "vgg-big",
            "vgg-big-bn",
            "fnn2",
            "fnn2-bn",
            "fnn4",
            "fnn4-bn",
            "fnn6",
            "fnn6-bn",
            "resnet-small",
            "resnet-big",
            "vgg11",
            "vgg11-bn",
            "vgg13",
            "vgg13-bn",
            "vgg16",
            "vgg16-bn",
            "vgg19",
            "vgg19-bn",
            "resnet18",
            "resnet20",
            "resnet32",
            "resnet34",
            "resnet44",
            "resnet50",
            "resnet56",
            "resnet101",
            "resnet110",
            "resnet110",
            "resnet152",
            "resnet1202",
            "wide-resnet18",
            "wide-resnet20",
            "wide-resnet32",
            "wide-resnet34",
            "wide-resnet44",
            "wide-resnet50",
            "wide-resnet56",
            "wide-resnet101",
            "wide-resnet110",
            "wide-resnet110",
            "wide-resnet152",
            "wide-resnet1202",
            "alexnet",
            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201",
            "googlenet",
        ],
        help="model architecture (default: logistic)",
    )
    model_args.add_argument(
        "--model-class",
        type=str,
        default="default",
        choices=["default", "mini", "tinyimagenet", "imagenet"],
        help="model class (default: default)",
    )
    model_args.add_argument(
        "--pretrained", action="store_true", default=False,
        help="load pretrained weights (default: False)",
    )
    model_args.add_argument(
        "--model-dir",
        type=str,
        default="pretrained_models",
        help="Directory for pretrained models. "
             "Save pretrained models to use here. "
             "Downloaded models will be stored here.",
    )
    model_args.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="Path to a checkpoint to restore a model from.",
    )
    return parser


def data_flags(parser):
    data_args = parser.add_argument_group("data")
    data_args.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store the datasets to be downloaded",
    )
    data_args.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "cifar100", "tiny-imagenet", "imagenet"],
        help="dataset (default: mnist)",
    )
    data_args.add_argument(
        "--workers",
        type=int,
        default="4",
        help="number of data loading workers (default: 4)",
    )
    data_args.add_argument(
        "--train-load-size",
        type=int,
        default=None,
        help="input bload size for training (default: None)",
    )
    data_args.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64), per core in TPU setting",
    )
    data_args.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="input batch size for testing (default: 256), per core in TPU setting",
    )
    data_args.add_argument(
        "--test-with-train",
        action="store_true",
        default=False,
        help="Use the train set for test, usually used when trained with data subset"
    )
    data_args.add_argument(
        "--data-subset", action="store_true", default=False, help="whether taking subset of the data"
    )
    data_args.add_argument(
        "--data-subset-classes",
        type=int,
        nargs="*",
        default=[],
        help="the classes to take if a subset of the dataset is used. If one number n is given, then the first n classes are taken. If a list is given, then classes with indices in the list are taken. Cannot be empty."
    )
    data_args.add_argument(
        "--data-subset-ndata",
        type=int,
        nargs="*",
        default=[],
        help="the number of training data in each class if a subset of the dataset is used. If one number n is given, then n data per class. If a list is given, the number of data for each class are taken according to the list in order. Empty means taking all."
    )
    data_args.add_argument(
        "--data-subset-random",
        action="store_true",
        default=False,
        help="Taking subset of data randomly. If False, the subset is taken in order from the original training set."
    )
    data_args.add_argument(
        "--data-subset-seed",
        type=int,
        default=None,
        help="The random seed used to subset the data"
    )
    data_args.add_argument(
        "--label-noise-p",
        type=float,
        default=0,
        help="the probability of random multinomial label noise"
    )
    data_args.add_argument(
        "--label-noise-seed",
        type=int,
        default=None,
        help="the random seed used in generating label noise"
    )
    return parser


def train():
    parser = default()
    parser = model_flags(parser)
    parser = data_flags(parser)
    train_args = parser.add_argument_group("train")
    train_args.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["mse", "ce",],
        help="loss funcion (default: ce)",
    )
    train_args.add_argument(
        "--binomial-loss-weights", action="store_true", default=False,
        help="adding binomial random variables as weights of single losses",
    )
    train_args.add_argument(
        "--binomial-loss-p",
        type=float,
        default=0,
        help="p for binomial weights, values in [0,0.5)",
    )
    train_args.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adap", "ladap", "adap_dev", "adap3", "adap3w",
                 "adamw", "adamp", "lamb", "lars",
                 "sgd", "momentum", "adam",  "rms"],
        help="optimizer (default: sgd)",
    )
    train_args.add_argument(
        "--epochs", type=int, default=0, help="number of epochs to train (default: 0)",
    )
    train_args.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    train_args.add_argument(
        "--lr-drops",
        type=int,
        nargs="*",
        default=[],
        help="list of learning rate drops (default: [])",
    )
    train_args.add_argument(
        "--lr-drop-rate",
        type=float,
        default=0.1,
        help="multiplicative factor of learning rate drop (default: 0.1)",
    )
    train_args.add_argument(
        "--wd", type=float, default=0.0, help="weight decay (default: 0.0)"
    )
    train_args.add_argument(
        "--momentum", type=float, default=0.9, help="momentum parameter (default: 0.9)"
    )
    train_args.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 parameter (default: 0.9)"
    )
    train_args.add_argument(
        "--beta2", type=float, default=0.999, help="beta 2 parameter (default: 0.999)"
    )
    train_args.add_argument(
        "--eps", type=float, default=1e-8, help="epsilon parameter - (default: 1e-8)"
    )
    train_args.add_argument(
        "--dampening",
        type=float,
        default=0.0,
        help="dampening parameter (default: 0.0)",
    )
    train_args.add_argument(
        "--decouple-wd", action="store_true", default=False,
        help="decouple weight decay from gradient (default: False)",
    )
    train_args.add_argument(
        "--nesterov", action="store_true", default=False,
        help="nesterov momentum (default: False)",
    )
    train_args.add_argument(
        "--seed", type=int, default=1, help="random seed (default: 1)"
    )
    train_args.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print statistics during training and testing. "
             "Use -vv for higher verbosity.",
    )
    # Save flags
    train_args.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Frequency (in batches) to save model checkpoints at",
    )
    train_args.add_argument(
        "--lean-ckpt", action="store_true", default=False,
        help="Make checkpoints lean: i.e. only save metric_dict",
    )
    train_args.add_argument(
        "--ckpt-per-epoch", action="store_true", default=False,
        help="Save checkpoint after each epoch, default False"
    )
    train_args.add_argument(
        "--ckpt-step-list",
        type=int,
        nargs="*",
        default=[],
        help="List of to save checkpoints, the format is [-1, start, end, step, -1, start, end, step, ..., all others], including ends"
    )
    train_args.add_argument(
        "--eval-mid-epoch", action="store_true", default=False,
        help="Include train and test loss in lean ckpts mid epoch",
    )
    return parser
    
    
def landscape():
    parser = train()
    land_args = parser.add_argument_group("landscape")
    land_args.add_argument(
        "--dir-types",
        type=str,
        nargs="*",
        default=["interpolate"],
        help="diagnose type, a list of interpolation, grad, random, pca. Max length 2",
    )
    land_args.add_argument(
        "--model-ckpt-start",
        type=str,
        default=None,
        help="the path of checkpoints used as starting parameter vector"
    )
    land_args.add_argument(
        "--model-ckpt-end",
        type=str,
        default=None,
        help="the path of checkpoints used as ending parameter vector for diagnose type interpolation"
    )
    land_args.add_argument(
        "--ckpt-center",
        type=str,
        default=None,
        help="the origin point of landscape evaluation, if different from model-ckpt-start."
    )
    land_args.add_argument(
        "--interval",
        type=float,
        nargs="*",
        default=[-1.0, 1.0],
        help="The endpoints for the interval of loss landscape visualization"
    )
    land_args.add_argument(
        "--normalization", action="store_true", default=False,
        help="Whether the direction is normalized, used for diagnose type grad"
    )
    land_args.add_argument(
        "--N",
        type=int,
        default=101,
        help="The number of equispaced points to evaluate loss"
    )
    land_args.add_argument(
        "--landscape-save-path",
        type=str,
        default=None,
        help="The path to save the diagnose results. If Noen then the results are saved to the experiment folder."
    )
    land_args.add_argument(
        "--pca-ckpt-prefix",
        type=str,
        default=None,
        help="the prefix of checkpoints for pca direction."
    )
    land_args.add_argument(
        "--pca-ckpt-list",
        type=int,
        nargs="*",
        default=None,
        help="the checkpoint list for pca direction, format: [start, end, step], include end"
    )
    land_args.add_argument(
        "--n-pc",
        type=int,
        default=1,
        help="the index of principal direction to use, default is the first one (1)"
    )
    ####
    # the following flags are for the second direction
    ####
    land_args.add_argument(
        "--model-ckpt-start-1",
        type=str,
        default=None,
        help="the path of checkpoints used as starting parameter vector, the second direction"
    )
    land_args.add_argument(
        "--model-ckpt-end-1",
        type=str,
        default=None,
        help="the path of checkpoints used as ending parameter vector for diagnose type interpolation, the second direction"
    )
    land_args.add_argument(
        "--interval-1",
        type=float,
        nargs="*",
        default=[-1.0, 1.0],
        help="The endpoints for the interval of loss landscape visualization for the second direction"
    )
    land_args.add_argument(
        "--normalization-1", action="store_true", default=False,
        help="Whether the direction is normalized, used for diagnose type grad. For the second direction"
    )
    land_args.add_argument(
        "--N-1",
        type=int,
        default=101,
        help="The number of equispaced points to evaluate loss for the second direction"
    )
    land_args.add_argument(
        "--pca-ckpt-prefix-1",
        type=str,
        default=None,
        help="the prefix of checkpoints for pca direction. For the second direction"
    )
    land_args.add_argument(
        "--pca-ckpt-list-1",
        type=int,
        nargs="*",
        default=None,
        help="the checkpoint list for pca direction, format: [start, end, step], include end. For the second direction"
    )
    land_args.add_argument(
        "--n-pc-1",
        type=int,
        default=1,
        help="the index of principal direction to use, default is the first one (1). For the second direction"
    )
    return parser

