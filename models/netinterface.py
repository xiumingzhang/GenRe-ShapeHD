# A general training procedure
import time
import torch.optim as optim
from torch.nn import init
import torch
from torch import FloatTensor, tensor
from loggers.loggers import _LogCumulator
from util.util_print import str_warning
from visualize.visualizer import Visualizer


def _get_num_samples(dataloader):
    # import torch.utils.data.sampler as samplers
    batch_sampler = dataloader.batch_sampler
    if batch_sampler.drop_last:
        return len(batch_sampler.sampler) // batch_sampler.batch_size * \
            batch_sampler.batch_size
    return len(batch_sampler.sampler)


class NetInterface(object):
    """ base class of all Model Interface
    all derived classes should overwrite __init__ and _train_on_batch(),
    and defines variables '_moveable_vars', '_nets', '_metrics' and '_optimizers'.

    Requirements for derived class:
    Variables:
        '_moveable_vars' for cuda();
        '_nets' and '_optimizers' for saving;
        '_metrics' for logging
    batch_log: all _train_on_batch() and _eval_on_batch() should return a batch_log for logging.
               All values in the batch_log are considerred as sample-wise mean.
    This log must have the key:
        'loss': the standard loss for choosing best performing eval model
        'size': the batch size for the batch
    """

    def init_weight(self, net=None, init_type='kaiming', init_param=0.02):
        """
        This is borrowed from Junyan
        """
        def init_func(m, init_type=init_type):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orth':
                    init.orthogonal_(m.weight.data, gain=init_param)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm') != -1:
                init.normal_(m.weight.data, 1.0, init_param)
                init.constant_(m.bias.data, 0.0)
        if net is not None:
            net.apply(init_func)
        else:
            self.net.apply(init_func)

    @classmethod
    def add_arguments(cls, parser):
        unique_params = set()
        return parser, unique_params

    def __init__(self, opt, logger):
        self._internal_logger = _LogCumulator()
        logger.add_logger(self._internal_logger)
        assert opt.optim == 'adam', "All projects here used adam"
        self.adam = optim.Adam
        self._logger = logger
        self.opt = opt
        self.full_logdir = opt.full_logdir
        self.grad_hook_gen, self.grad_stats = self.dict_grad_hook_factory(
            add_func=lambda x: {
                'mean': x.mean(),
                'std': x.std()
            }
        )
        self._nets = []
        self._moveable_vars = []
        self._optimizers = []
        self.visualizer = Visualizer(
            n_workers=getattr(opt, 'vis_workers', 4),
            param_f=getattr(opt, 'vis_param_f', None),
        ) # getattr used for backward compatibility
        self.batches_to_vis = {}
        self.input_names = []
        self._input = lambda: None
        self.gt_names = []
        self._gt = lambda: None
        self.aux_names = []
        self._aux = lambda: None  # auxiliary tensors that need moving to GPU
        # Use cases include the one for D in WGAN

    def init_vars(self, add_path=True):
        """
        Also add stuff to movable_vars
        """
        for net in self._nets:
            self._moveable_vars.append(net)
        for name in self.input_names:
            setattr(self._input, name, FloatTensor())
            self._moveable_vars.append('_input.' + name)
            if add_path:
                setattr(self._input, name + '_path', None)
        for name in self.gt_names:
            setattr(self._gt, name, FloatTensor())
            self._moveable_vars.append('_gt.' + name)
            if add_path:
                setattr(self._gt, name + '_path', None)
        for name in self.aux_names:
            if name == 'one':
                setattr(self._aux, name, tensor(1).float())
            elif name == 'neg_one':
                setattr(self._aux, name, tensor(-1).float())
            else:
                setattr(self._aux, name, FloatTensor([]))
            self._moveable_vars.append('_aux.' + name)

    def load_batch(self, batch, include_gt=True):
        for name in self.input_names:
            if name in batch.keys():
                var = getattr(self._input, name)
                if var.shape != batch[name].shape:
                    var.resize_(batch[name].shape)
                var.copy_(batch[name])
                setattr(self._input, name, var)
            if name + '_path' in batch.keys():
                setattr(self._input, name + '_path', batch[name + '_path'])
        if include_gt:
            for name in self.gt_names:
                if name in batch.keys():
                    var = getattr(self._gt, name)
                    if var.shape != batch[name].shape:
                        var.resize_(batch[name].shape)
                    var.copy_(batch[name])
                    setattr(self._gt, name, var)
                if name + '_path' in batch.keys():
                    setattr(self._gt, name + '_path', batch[name + '_path'])

    def _train_on_batch(self, epoch, batch_ind, dataloader_out):
        """ function that trains the model over one batch and return batch_log (including size and loss) """
        raise NotImplementedError

    def _vali_on_batch(self, epoch, batch_ind, dataloader_out):
        """ function that trains the model over one batch and return batch_log (including size and loss) """
        raise NotImplementedError

    def test_on_batch(self, batch_ind, dataloader_out):
        raise NotImplementedError

    def train_epoch(
            self,
            dataloader,
            *,
            dataloader_eval=None,
            max_batches_per_train=None,
            max_batches_per_eval=None,
            epochs=1,
            initial_epoch=1,
            verbose=1,
            reset_dataset=None,
            eval_at_start=False):
        """
        Train the model with given dataloader and run evaluation with the given dataloader_eval
        max_batches_per_train: limit the number of batches for each epoch
        reset_dataset: if the dataset needs to be reset for each training epoch,
            define a reset() method in the dataset and provide it as this argument.
            reset() will be called at the beginning of each epoch, and the procedure will
            check for dataset size change and change accordingly.
        """
        logger = self._logger
        # set logger params and number of batches in an epoch
        steps_per_epoch = len(dataloader)
        samples_per_epoch = _get_num_samples(dataloader)
        if max_batches_per_train is not None:
            steps_per_epoch = min(max_batches_per_train, steps_per_epoch)
            samples_per_epoch = min(samples_per_epoch, steps_per_epoch * dataloader.batch_sampler.batch_size)
        if dataloader_eval is not None:
            steps_per_eval = len(dataloader_eval)
            samples_per_eval = _get_num_samples(dataloader_eval)
            if max_batches_per_eval is not None:
                steps_per_eval = min(steps_per_eval, max_batches_per_eval)
                samples_per_eval = min(samples_per_eval, steps_per_eval * dataloader.batch_sampler.batch_size)
        else:
            steps_per_eval = 0
            samples_per_eval = 0
        logger.set_params({
            'epochs': epochs + initial_epoch - 1,
            'steps': steps_per_epoch,
            'steps_eval': steps_per_eval,
            'samples': samples_per_epoch,
            'samples_eval': samples_per_eval,
            'verbose': 1,
            'metrics': self._metrics,
        })
        logger.set_model(self)
        logger.on_train_begin()
        dataset_size = 0            # monitor if dataset size change due to reset_dataset.reset(). update steps_per_epoch if needed

        # define train closure
        def _train(epoch):
            nonlocal dataset_size
            nonlocal steps_per_epoch
            nonlocal samples_per_epoch
            nonlocal steps_per_eval
            nonlocal samples_per_eval
            self.train()
            logger.train()
            if reset_dataset is not None:
                reset_dataset.reset()
                # reset steps if necessary
                if dataset_size != len(reset_dataset):
                    steps_per_epoch = len(dataloader)
                    samples_per_epoch = _get_num_samples(dataloader)
                    if max_batches_per_train is not None:
                        steps_per_epoch = min(max_batches_per_train, steps_per_epoch)
                        samples_per_epoch = min(samples_per_epoch, steps_per_epoch * dataloader.batch_sampler.batch_size)
                    if dataloader_eval is not None:
                        steps_per_eval = len(dataloader_eval)
                        samples_per_eval = _get_num_samples(dataloader_eval)
                        if max_batches_per_eval is not None:
                            steps_per_eval = min(steps_per_eval, max_batches_per_eval)
                            samples_per_eval = min(samples_per_eval, steps_per_eval * dataloader.batch_sampler.batch_size)
                    logger.set_params({
                        'epochs': epochs + initial_epoch - 1,
                        'steps': steps_per_epoch,
                        'steps_eval': steps_per_eval,
                        'samples': samples_per_epoch,
                        'samples_eval': samples_per_eval,
                        'verbose': 1,
                        'metrics': self._metrics,
                    })
                    dataset_size = len(reset_dataset)

            dataiter = iter(dataloader)
            logger.on_epoch_begin(epoch)
            for i in range(steps_per_epoch):
                start_time = time.time()
                data = next(dataiter)
                data_time = time.time() - start_time
                logger.on_batch_begin(i)
                batch_log = self._train_on_batch(epoch, i, data)
                batch_log['batch'] = i
                batch_log['epoch'] = epoch
                batch_log['data_time'] = data_time
                batch_log['batch_time'] = time.time() - start_time
                logger.on_batch_end(i, batch_log)
            epoch_log = self._internal_logger.get_epoch_log()
            logger.on_epoch_end(epoch, epoch_log)

        # define eval closure
        def _eval(epoch):
            self.eval()
            logger.eval()
            dataiter = iter(dataloader_eval)
            logger.on_epoch_begin(epoch)
            for i in range(steps_per_eval):
                start_time = time.time()
                data = next(dataiter)
                data_time = time.time() - start_time
                logger.on_batch_begin(i)
                batch_log = self._vali_on_batch(epoch, i, data)
                batch_log['batch'] = i
                batch_log['epoch'] = epoch
                batch_log['data_time'] = data_time
                batch_log['batch_time'] = time.time() - start_time
                logger.on_batch_end(i, batch_log)
            epoch_log = self._internal_logger.get_epoch_log()
            logger.on_epoch_end(epoch, epoch_log)

        # run actual training
        if eval_at_start:
            if dataloader_eval is None:
                raise ValueError('eval_at_beginning is set to True but no eval data is given.')
            _eval(initial_epoch - 1)
        for epoch in range(initial_epoch, initial_epoch + epochs):
            _train(epoch)
            if dataloader_eval is not None:
                _eval(epoch)

        logger.on_train_end()

    @staticmethod
    def circular_grad_hook_factory(num_to_keep, add_func=lambda x: x):
        class CircularList(object):
            def __init__(self, num):
                self.vals = [None] * num
                self.ncyc = 0
                self.c = 0
                self.n = num

            def append(self, value):
                self.vals[self.c] = value
                self.c += 1
                if self.c == self.n:
                    self.c = 0
                    self.ncyc += 1

            def full(self):
                return self.c == 0

            def __iter__(self):
                return iter(self.vals)

            def __len__(self):
                return self.n

            def __getitem__(self, index):
                assert index < self.n
                return self.vals[index]

            def __repr__(self):
                return str(self.vals)

        saved_grads = CircularList(num_to_keep)

        def grad_hook(grad):
            saved_tensor = add_func(grad)
            saved_grads.append(saved_tensor)
        return grad_hook, saved_grads

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook

        return hook_gen, saved_dict

    def predict(self, batch, net='net', load_gt=True, no_grad=False):
        net = getattr(self, net)
        self.load_batch(batch, include_gt=load_gt)
        if no_grad:
            with torch.no_grad():
                pred = net(self._input)  # a structure
                # How to extract data and then forward them
                # should be dealt with in model file
        else:
            pred = net(self._input)
        return pred

    def train(self):
        for m in self._nets:
            m.train()

    def eval(self):
        for m in self._nets:
            m.eval()

    def num_parameters(self, return_list=False):
        nparams_list = list()
        for net in self._nets:
            parameters = list(net.parameters())
            nparams_list.append(sum([x.numel() for x in parameters]))
        if return_list:
            return nparams_list
        return sum(nparams_list)

    def cuda(self):
        for v in self._moveable_vars:
            if isinstance(v, str):
                if '.' in v:
                    var_type, var_name = v.split('.')
                    var = getattr(getattr(self, var_type), var_name)
                    setattr(getattr(self, var_type), var_name, var.cuda())
                else:
                    setattr(self, v, getattr(self, v).cuda())
            else:
                v.cuda()

    def cpu(self):
        for v in self._moveable_vars:
            if isinstance(v, str):
                if '.' in v:
                    var_type, var_name = v.split('.')
                    var = getattr(getattr(self, var_type), var_name)
                    setattr(getattr(self, var_type), var_name, var.cpu())
                else:
                    setattr(self, v, getattr(self, v).cpu())
            else:
                v.cpu()

    def to(self, device):
        for v in self._moveable_vars:
            if isinstance(v, str):
                if '.' in v:
                    var_type, var_name = v.split('.')
                    var = getattr(getattr(self, var_type), var_name)
                    setattr(getattr(self, var_type), var_name, var.to(device))
                else:
                    setattr(self, v, getattr(self, v).to(device))
            else:
                v.to(device)

    def save_state_dict(self, filepath, *, save_optimizer=False, additional_values={}):
        state_dicts = dict()
        state_dicts['nets'] = [net.state_dict() for net in self._nets]
        if save_optimizer:
            state_dicts['optimizers'] = [optimizer.state_dict() for optimizer in self._optimizers]
        for k, v in additional_values.items():
            state_dicts[k] = v
        torch.save(state_dicts, filepath)

    def load_state_dict(self, filepath, *, load_optimizer='auto'):
        state_dicts = torch.load(filepath)

        if load_optimizer == 'auto':
            load_optimizer = ('optimizers' in state_dicts)
            if not load_optimizer:
                print(str_warning, 'Model loaded without optimizer states. ')

        assert len(self._nets) == len(state_dicts['nets'])
        for i in range(len(self._nets)):
            self._nets[i].load_state_dict(state_dicts['nets'][i])

        if load_optimizer:
            assert len(self._optimizers) == len(state_dicts['optimizers'])
            for i in range(len(self._optimizers)):
                optimizer = self._optimizers[i]
                state = state_dicts['optimizers'][i]

                # load optimizer state without overwriting training hyper-parameters, e.g. lr
                optimizer_load_state_dict(optimizer, state, keep_training_params=True)

        additional_values = {k: v for k, v in state_dicts.items() if k not in ('optimizers', 'nets')}
        return additional_values


def optimizer_load_state_dict(optimizer, state, keep_training_params=False):
    if keep_training_params:
        oldstate = optimizer.state_dict()
        assert len(oldstate['param_groups']) == len(state['param_groups'])
        # use oldstate to override this state
        for oldg, g in zip(oldstate['param_groups'], state['param_groups']):
            for k, v in oldg.items():
                if k != 'params':   # parameter id not overwriten
                    g[k] = v
    optimizer.load_state_dict(state)


def top_n_err(output, label, nlist):
    output = output.detach().cpu().numpy()
    label = label.detach().cpu().numpy().reshape(-1, 1)
    idx_sort = output.argsort()
    errlist = list()
    for n in nlist:
        errlist.append(1 - (idx_sort[:, -n:] - label == 0).any(1).mean())
    return errlist


def parse_optimizer_specific_params(optimizer_name, opt):
    optim_params = dict()
    if optimizer_name == 'adam':
        optim_params['betas'] = (opt.adam_beta1, opt.adam_beta2)
    elif optimizer_name == 'sgd':
        optim_params['momentum'] = opt.sgd_momentum
        optim_params['dampening'] = opt.sgd_dampening
        optim_params['weight_decay'] = opt.sgd_wdecay
    return optim_params


def data_parallel_decorator(ModuleClass):
    """
    A decorator for forward function to use multiGPU training
    while maintaining network layout (and thus saved parameter
    names).

    Note: the way pytorch replicate a module only guarantees
    that pytorch internal parameters, buffers and modules are
    properly copied and set over devices. Any other class attributes
    are copied in a shallow copy manner - any list, dictionary, etc.
    that uses class attributes may not be set properly, and
    will point to the original copy instead of the copied one
    on the new GPU.
    """
    from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
    from torch.nn.parallel.replicate import replicate
    from torch.nn.parallel.parallel_apply import parallel_apply
    assert not hasattr(ModuleClass, '_forward_worker'), 'data_parallel_decorator cannot be used on a class twice'
    ModuleClass._forward_worker = ModuleClass.forward

    def wrapped(self, *inputs, **module_kwargs):
        if (not hasattr(self, '_is_replica')) and inputs[0].is_cuda:
            device_count = torch.cuda.device_count()
            if inputs[0].shape[0] % device_count != 0:
                import os
                cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else ''
                raise ValueError('batch size (%d) must be divisible by the number of GPUs (%d) used\n CUDA_VISIBLE_DEVICES: %s' % (inputs[0].shape[0], device_count, cuda_visible_devices))
            if device_count > 1:
                # modified from pytorch (torch.nn.parallel.DataParallel)
                device_ids = list(range(device_count))
                output_device = device_ids[0]
                inputs, kwargs = scatter_kwargs(inputs, module_kwargs, device_ids)
                replicas = replicate(self, device_ids[:len(inputs)])

                # add a _is_replica flag to avoid infinite loop
                # from recursively calling parallel_apply
                for replica in replicas:
                    replica._is_replica = True
                outputs = parallel_apply(replicas, inputs, kwargs)
                return gather(outputs, output_device)

        return self._forward_worker(*inputs, **module_kwargs)
    ModuleClass.forward = wrapped
    return ModuleClass


def print_grad_stats(grad):
    grad_ = grad.detach()
    print('\nmin, max, mean, std: %e, %e, %e, %e' % (grad_.min().item(), grad_.max().item(), grad_.mean().item(), grad_.std().item()))
