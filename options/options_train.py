import sys
import argparse
import torch
from util.util_print import str_warning
from datasets import get_dataset
from models import get_model


def add_general_arguments(parser):
    # Parameters that will NOT be overwritten when resuming
    unique_params = {'gpu', 'resume', 'epoch', 'workers', 'batch_size', 'save_net', 'epoch_batches', 'logdir'}

    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu to use')
    parser.add_argument('--manual_seed', type=int, default=None,
                        help='manual seed for randomness')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume training by loading checkpoint.pt or best.pt. Use 0 for training from scratch, -1 for last and -2 for previous best. Use positive number for a specific epoch. \
                            Most options will be overwritten to resume training with exactly same environment')
    parser.add_argument(
        '--suffix', default='', type=str,
        help="Suffix for `logdir` that will be formatted with `opt`, e.g., '{classes}_lr{lr}'"
    )
    parser.add_argument('--epoch', type=int, default=0,
                        help='number of epochs to train')

    # Dataset IO
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset to use')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--classes', default='chair', type=str,
                        help='class to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--epoch_batches', default=None, type=int, help='number of batches used per epoch')
    parser.add_argument('--eval_batches', default=None,
                        type=int, help='max number of batches used for evaluation per epoch')
    parser.add_argument('--eval_at_start', action='store_true',
                        help='run evaluation before starting to train')
    parser.add_argument('--log_time', action='store_true', help='adding time log')

    # Network name
    parser.add_argument('--net', type=str, required=True,
                        help='network type to use')

    # Optimizer
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--adam_beta1', type=float, default=0.5,
                        help='beta1 of adam')
    parser.add_argument('--adam_beta2', type=float, default=0.9,
                        help='beta2 of adam')
    parser.add_argument('--sgd_momentum', type=float, default=0.9,
                        help="momentum factor of SGD")
    parser.add_argument('--sgd_dampening', type=float, default=0,
                        help="dampening for momentum of SGD")
    parser.add_argument('--wdecay', type=float, default=0.0,
                        help='weight decay')

    # Logging and visualization
    parser.add_argument('--logdir', type=str, default=None,
                        help='Root directory for logging. Actual dir is [logdir]/[net_classes_dataset]/[expr_id]')
    parser.add_argument('--log_batch', action='store_true',
                        help='Log batch loss')
    parser.add_argument('--expr_id', type=int, default=0,
                        help='Experiment index. non-positive ones are overwritten by default. Use 0 for code test. ')
    parser.add_argument('--save_net', type=int, default=1,
                        help='Period of saving network weights')
    parser.add_argument('--save_net_opt', action='store_true',
                        help='Save optimizer state in regular network saving')
    parser.add_argument('--vis_every_vali', default=1, type=int,
                        help="Visualize every N epochs during validation")
    parser.add_argument('--vis_every_train', default=1, type=int,
                        help="Visualize every N epochs during training")
    parser.add_argument('--vis_batches_vali', type=int, default=10,
                        help="# batches to visualize during validation")
    parser.add_argument('--vis_batches_train', type=int, default=10,
                        help="# batches to visualize during training")
    parser.add_argument('--tensorboard', action='store_true',
                        help='Use tensorboard for logging. If enabled, the output log will be at [logdir]/[tensorboard]/[net_classes_dataset]/[expr_id]')
    parser.add_argument('--vis_workers', default=4, type=int, help="# workers for the visualizer")
    parser.add_argument('--vis_param_f', default=None, type=str,
                        help="Parameter file read by the visualizer on every batch; defaults to 'visualize/config.json'")

    return parser, unique_params


def overwrite(opt, opt_f_old, unique_params):
    opt_dict = vars(opt)
    opt_dict_old = torch.load(opt_f_old)
    for k, v in opt_dict_old.items():
        if k in opt_dict:
            if (k not in unique_params) and (opt_dict[k] != v):
                print(str_warning, "Overwriting %s for resuming training: %s -> %s"
                      % (k, str(opt_dict[k]), str(v)))
                opt_dict[k] = v
        else:
            print(str_warning, "Ignoring %s, an old option that no longer exists" % k)
    opt = argparse.Namespace(**opt_dict)
    return opt


def parse(add_additional_arguments=None):
    parser = argparse.ArgumentParser()
    parser, unique_params = add_general_arguments(parser)
    if add_additional_arguments is not None:
        parser, unique_params_additional = add_additional_arguments(parser)
        unique_params = unique_params.union(unique_params_additional)
    opt_general, _ = parser.parse_known_args()
    dataset_name, net_name = opt_general.dataset, opt_general.net
    del opt_general

    # Add parsers depending on dataset and models
    parser, unique_params_dataset = get_dataset(dataset_name).add_arguments(parser)
    parser, unique_params_model = get_model(net_name).add_arguments(parser)

    # Manually add '-h' after adding all parser arguments
    if '--printhelp' in sys.argv:
        sys.argv.append('-h')

    opt = parser.parse_args()
    unique_params = unique_params.union(unique_params_dataset)
    unique_params = unique_params.union(unique_params_model)
    return opt, unique_params
