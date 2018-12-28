import os
import time
from shutil import rmtree
from tqdm import tqdm
import torch
from options import options_test
import datasets
import models
from util.util_print import str_error, str_stage, str_verbose
import util.util_loadlib as loadlib
from loggers import loggers


print("Testing Pipeline")

###################################################

print(str_stage, "Parsing arguments")
opt = options_test.parse()
opt.full_logdir = None
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

print(str_stage, "Setting up output directory")
output_dir = opt.output_dir
output_dir += ('_' + opt.suffix.format(**vars(opt))) \
    if opt.suffix != '' else ''
opt.output_dir = output_dir

if os.path.isdir(output_dir):
    if opt.overwrite:
        rmtree(output_dir)
    else:
        raise ValueError(str_error +
                         " %s already exists, but no overwrite flag"
                         % output_dir)
os.makedirs(output_dir)

###################################################

print(str_stage, "Setting up loggers")
logger_list = [
    loggers.TerminateOnNaN(),
]
logger = loggers.ComposeLogger(logger_list)

###################################################

print(str_stage, "Setting up models")
Model = models.get_model(opt.net, test=True)
model = Model(opt, logger)
model.to(device)
model.eval()
print(model)
print("# model parameters: {:,d}".format(model.num_parameters()))

###################################################

print(str_stage, "Setting up data loaders")
start_time = time.time()
Dataset = datasets.get_dataset('test')
dataset = Dataset(opt, model=model)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=False,
    shuffle=False
)
n_batches = len(dataloader)
dataiter = iter(dataloader)
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# test points: " + str(len(dataset)))
print(str_verbose, "# test batches: " + str(n_batches))

###################################################

print(str_stage, "Testing")
for i in tqdm(range(n_batches)):
    batch = next(dataiter)
    model.test_on_batch(i, batch)
