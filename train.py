import sys
import os
import time
import pandas as pd
import torch
from options import options_train
import datasets
import models
from loggers import loggers
from util.util_print import str_error, str_stage, str_verbose, str_warning
from util import util_loadlib as loadlib


###################################################

print(str_stage, "Parsing arguments")
opt, unique_opt_params = options_train.parse()
# Get all parse done, including subparsers
print(opt)

###################################################

print(str_stage, "Setting device")
if opt.gpu == '-1':
    device = torch.device('cpu')
else:
    loadlib.set_gpu(opt.gpu)
    device = torch.device('cuda')
if opt.manual_seed is not None:
    loadlib.set_manual_seed(opt.manual_seed)

###################################################

print(str_stage, "Setting up logging directory")
exprdir = '{}_{}_{}'.format(opt.net, opt.dataset, opt.lr)
exprdir += ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
logdir = os.path.join(opt.logdir, exprdir, str(opt.expr_id))

if opt.resume == 0:
    if os.path.isdir(logdir):
        if opt.expr_id <= 0:
            print(
                str_warning, (
                    "Will remove Experiment %d at\n\t%s\n"
                    "Do you want to continue? (y/n)"
                ) % (opt.expr_id, logdir)
            )
            need_input = True
            while need_input:
                response = input().lower()
                if response in ('y', 'n'):
                    need_input = False
            if response == 'n':
                print(str_stage, "User decides to quit")
                sys.exit()
            os.system('rm -rf ' + logdir)
        else:
            raise ValueError(str_error +
                             " Refuse to remove positive expr_id")
    os.system('mkdir -p ' + logdir)
else:
    assert os.path.isdir(logdir)
    opt_f_old = os.path.join(logdir, 'opt.pt')
    opt = options_train.overwrite(opt, opt_f_old, unique_opt_params)

# Save opt
torch.save(vars(opt), os.path.join(logdir, 'opt.pt'))
with open(os.path.join(logdir, 'opt.txt'), 'w') as fout:
    for k, v in vars(opt).items():
        fout.write('%20s\t%-20s\n' % (k, v))

opt.full_logdir = logdir
print(str_verbose, "Logging directory set to: %s" % logdir)

###################################################

print(str_stage, "Setting up loggers")
if opt.resume != 0 and os.path.isfile(os.path.join(logdir, 'best.pt')):
    try:
        prev_best_data = torch.load(os.path.join(logdir, 'best.pt'))
        prev_best = prev_best_data['loss_eval']
        del prev_best_data
    except KeyError:
        prev_best = None
else:
    prev_best = None
best_model_logger = loggers.ModelSaveLogger(
    os.path.join(logdir, 'best.pt'),
    period=1,
    save_optimizer=True,
    save_best=True,
    prev_best=prev_best
)
logger_list = [
    loggers.TerminateOnNaN(),
    loggers.ProgbarLogger(allow_unused_fields='all'),
    loggers.CsvLogger(
        os.path.join(logdir, 'epoch_loss.csv'),
        allow_unused_fields='all'
    ),
    loggers.ModelSaveLogger(
        os.path.join(logdir, 'nets', '{epoch:04d}.pt'),
        period=opt.save_net,
        save_optimizer=opt.save_net_opt
    ),
    loggers.ModelSaveLogger(
        os.path.join(logdir, 'checkpoint.pt'),
        period=1,
        save_optimizer=True
    ),
    best_model_logger,
]
if opt.log_batch:
    logger_list.append(
        loggers.BatchCsvLogger(
            os.path.join(logdir, 'batch_loss.csv'),
            allow_unused_fields='all'
        )
    )
if opt.tensorboard:
    tf_logdir = os.path.join(
        opt.logdir, 'tensorboard', exprdir, str(opt.expr_id))
    if os.path.isdir(tf_logdir) and opt.resume == 0:
        os.system('rm -r ' + tf_logdir)  # remove previous tensorboard log if overwriting
    if not os.path.isdir(os.path.join(logdir, 'tensorboard')):
        os.symlink(tf_logdir, os.path.join(logdir, 'tensorboard'))
    logger_list.append(
        loggers.TensorBoardLogger(
            tf_logdir,
            allow_unused_fields='all'
        )
    )
logger = loggers.ComposeLogger(logger_list)

###################################################

print(str_stage, "Setting up models")
Model = models.get_model(opt.net)
model = Model(opt, logger)
model.to(device)
print(model)
print("# model parameters: {:,d}".format(model.num_parameters()))

initial_epoch = 1
if opt.resume != 0:
    if opt.resume == -1:
        net_filename = os.path.join(logdir, 'checkpoint.pt')
    elif opt.resume == -2:
        net_filename = os.path.join(logdir, 'best.pt')
    else:
        net_filename = os.path.join(
            logdir, 'nets', '{epoch:04d}.pt').format(epoch=opt.resume)
    if not os.path.isfile(net_filename):
        print(str_warning, ("Network file not found for opt.resume=%d. "
                            "Starting from scratch") % opt.resume)
    else:
        additional_values = model.load_state_dict(net_filename, load_optimizer='auto')
        try:
            initial_epoch += additional_values['epoch']
        except KeyError as err:
            # Old saved model does not have epoch as additional values
            epoch_loss_csv = os.path.join(logdir, 'epoch_loss.csv')
            if opt.resume == -1:
                try:
                    initial_epoch += pd.read_csv(epoch_loss_csv)['epoch'].max()
                except pd.errors.ParserError:
                    with open(epoch_loss_csv, 'r') as f:
                        lines = f.readlines()
                    initial_epoch += max([int(l.split(',')[0]) for l in lines[1:]])
            else:
                initial_epoch += opt.resume

###################################################

print(str_stage, "Setting up data loaders")
start_time = time.time()
dataset = datasets.get_dataset(opt.dataset)
dataset_train = dataset(opt, mode='train', model=model)
dataset_vali = dataset(opt, mode='vali', model=model)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True
)
dataloader_vali = torch.utils.data.DataLoader(
    dataset_vali,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True,
    shuffle=False
)
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# training points: " + str(len(dataset_train)))
print(str_verbose, "# training batches per epoch: " + str(len(dataloader_train)))
print(str_verbose, "# test batches: " + str(len(dataloader_vali)))

###################################################

if opt.epoch > 0:
    print(str_stage, "Training")
    model.train_epoch(
        dataloader_train,
        dataloader_eval=dataloader_vali,
        max_batches_per_train=opt.epoch_batches,
        epochs=opt.epoch,
        initial_epoch=initial_epoch,
        max_batches_per_eval=opt.eval_batches,
        eval_at_start=opt.eval_at_start
    )
