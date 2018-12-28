# Common loggers used during training

# design follows Callback from Keras
import os
import csv
from collections import OrderedDict, defaultdict
import numpy as np
from util.util_print import str_error, str_warning
from .Progbar import Progbar


class BaseLogger(object):
    """ base class for all logger.
    Each logger should expect an batch (batch index) and batch log
    for batch end, an epoch (epoch index) and epoch log for
    epoch end. no logs are given at batch/epoch begin, only the index.

    Note: epoch_log will be used for all loggers, and should not be modified
    in any logger's on_epoch_end() """

    def __init__(self):
        raise NotImplementedError

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, epoch_log):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch, batch_log):
        pass

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _set_unused_metric_mode(self, mode='none'):
        if mode in ('all', 'always', 'both'):
            mode = 'all'
        elif mode in ('none', 'neither', 'never'):
            mode = 'none'
        assert mode in ('none', 'train', 'test', 'all')
        self._allow_unused_metric_training = False
        self._allow_unused_metric_testing = False
        if mode in ('train', 'all'):
            self._allow_unused_metric_training = True
        if mode in ('test', 'all'):
            self._allow_unused_metric_testing = True

    def _allow_unused(self):
        return self._allow_unused_metric_training if self.training else self._allow_unused_metric_testing


class _LogCumulator(BaseLogger):
    """ cumulate the batch_log and generate an epoch_log
    Note that this logger is used for generating epoch_log,
    and thus does not take epoch_log as input"""

    def __init__(self):
        pass

    def on_epoch_begin(self, epoch):
        self.log_values = defaultdict(list)
        self.sizes = list()
        self.epoch_log = None

    def on_batch_end(self, batch, batch_log):
        for k, v in batch_log.items():
            self.log_values[k].append(v)
        self.sizes.append(batch_log['size'])

    def get_epoch_log(self):
        epoch_log = dict()
        for k in self.log_values:
            epoch_log[k] = (np.array(self.log_values[k]) *
                            np.array(self.sizes)).sum() / np.array(self.sizes).sum()
        return epoch_log


class ProgbarLogger(BaseLogger):
    """ display a progbar """

    def __init__(self, count_mode='samples', allow_unused_fields='none'):
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))
        self._set_unused_metric_mode(allow_unused_fields)

    def on_train_begin(self):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch):
        if self.verbose:
            if self.training:
                desc = 'Epoch %d/%d' % (epoch, self.epochs)
                print(desc)
                if self.use_steps:
                    target = self.params['steps']
                else:
                    target = self.params['samples']
                self.target = target
                self.progbar = Progbar(target=self.target,
                                       verbose=self.verbose)
            else:
                print('Eval %d/%d' % (epoch, self.epochs))
                if self.use_steps:
                    target = self.params['steps_eval']
                else:
                    target = self.params['samples_eval']
                self.target = target
                self.progbar = Progbar(target=self.target,
                                       verbose=self.verbose)

        self.seen = 0

    def on_batch_begin(self, batch):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, batch_log):
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_log['size']

        for k in self.params['metrics']:
            if self._allow_unused() and (k not in batch_log):
                continue
            self.log_values.append((k, batch_log[k]))

        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, epoch_log):
        # Note: epoch_log not used
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)


class CsvLogger(BaseLogger):
    """ loss logger to csv files """

    def __init__(self, filename, allow_unused_fields='none'):
        self.sep = ','
        self.filename = filename
        self._set_unused_metric_mode(allow_unused_fields)

    def on_train_begin(self):
        if not os.path.isfile(self.filename):
            newfile = True
        else:
            newfile = False
        if not os.path.isdir(os.path.dirname(self.filename)):
            os.system('mkdir -p ' + os.path.dirname(self.filename))
        self.metrics = self.params['metrics']

        self.csv_file = open(self.filename, 'a+')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=[
                                     'epoch', 'mode'] + self.metrics)
        if newfile:
            self.writer.writeheader()
            self.csv_file.flush()

    def on_epoch_end(self, epoch, epoch_log):
        row_dict = OrderedDict(
            {'epoch': epoch, 'mode': 'train' if self.training else ' eval'})
        for k in self.metrics:
            if self._allow_unused() and (k not in epoch_log):
                continue
            row_dict[k] = epoch_log[k]
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self):
        self.csv_file.close()
        self.writer = None


class BatchCsvLogger(BaseLogger):
    """ loss logger to csv files """

    def __init__(self, filename, allow_unused_fields='none'):
        self.sep = ','
        self.filename = filename
        self._set_unused_metric_mode(allow_unused_fields)

    def on_train_begin(self):
        if not os.path.isfile(self.filename):
            newfile = True
        else:
            newfile = False
        if not os.path.isdir(os.path.dirname(self.filename)):
            os.system('mkdir -p ' + os.path.dirname(self.filename))
        self.metrics = self.params['metrics']

        self.csv_file = open(self.filename, 'a+')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=[
                                     'epoch', 'mode'] + self.metrics)
        if newfile:
            self.writer.writeheader()
            self.csv_file.flush()

    def on_batch_end(self, batch, batch_log=None):
        row_dict = OrderedDict(
            {'epoch': batch_log['epoch'], 'mode': 'train' if self.training else ' eval'})
        for k in self.metrics:
            if self._allow_unused() and (k not in batch_log):
                continue
            row_dict[k] = batch_log[k]
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self):
        self.csv_file.close()
        self.writer = None


class ModelSaveLogger(BaseLogger):
    """
    A logger that saves model periodically.
    The logger can be configured to save the model with the best eval score.
    """

    def __init__(self, filepath, period=1, save_optimizer=False, save_best=False, prev_best=None):
        self.filepath = filepath
        self.period = period
        self.save_optimizer = save_optimizer
        self.save_best = save_best
        self.loss_name = 'loss'
        self.current_best_eval = prev_best
        self.current_best_epoch = None

        # search for previous best
        if self.save_best and prev_best is None:
            # try:
            #     # parse epoch_loss. overwrite previous best if fail
            #     if os.path.isfile(filepath):
            #         prev_loss = pd.read_csv(os.path.join(os.path.dirname(filepath), 'epoch_loss.csv'))
            #         prev_eval_loss = prev_loss[prev_loss['mode'] == 'eval']
            #         if prev_eval_loss.size == 0:
            #             raise ValueError('loaded epoch loss file has no eval loss')
            #         self.current_best_eval = prev_eval_loss[self.loss_name].min()
            #         self.current_best_epoch = prev_eval_loss[prev_eval_loss[self.loss_name] == self.current_best_eval]['epoch'].iloc[0]
            # except: # (IOError, pd.errors.ParserError, KeyError):
            print(
                str_warning, 'Previous best eval loss not given. Best validation model WILL be overwritten.')

    def on_train_begin(self):
        if not os.path.isdir(self.filepath):
            os.system('mkdir -p ' + os.path.dirname(self.filepath))
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, epoch_log):
        # avoid saving twice (once after training, once after eval)
        if self.training:
            if self.save_best:  # save_best mode is not used right after training
                return
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period:
                filepath = self.filepath.format(epoch=epoch)
                self.model.save_state_dict(
                    filepath, save_optimizer=self.save_optimizer, additional_values={'epoch': epoch})
                self.epochs_since_last_save = 0
        else:
            if self.save_best:
                if self.loss_name not in epoch_log:
                    print(
                        str_warning, 'Loss name %s not found in batch_log. "Best model saving" is turned off"' % self.loss_name)
                else:
                    current_eval = epoch_log['loss']
                    if self.current_best_eval is None or current_eval < self.current_best_eval:
                        self.current_best_eval = current_eval
                        self.current_best_epoch = epoch
                        filepath = self.filepath.format(epoch=epoch)
                        self.model.save_state_dict(filepath, save_optimizer=self.save_optimizer, additional_values={
                                                   'epoch': epoch, 'loss_eval': self.current_best_eval})


class TerminateOnNaN(BaseLogger):
    def __init__(self):
        self._training = True

    def on_batch_begin(self, batch):
        if not self._training:
            raise ValueError(str_error, 'inf/nan found')

    def on_batch_end(self, batch, batch_log):
        if batch_log:
            for k, v in batch_log.items():
                if np.isnan(v): # or np.isinf(v):
                    self._training = False
                    break


class TensorBoardLogger(BaseLogger):
    def __init__(self, filepath, allow_unused_fields='none'):
        try:
            import tensorflow as tf
            self.tf = tf
        except Exception as err:
            print(str_warning, "TensorBoard logger disabled due to an error while importing tensorflow: \n%s" % str(err))
            self.tf = None
        self.filepath = filepath
        self._set_unused_metric_mode(allow_unused_fields)

    def on_train_begin(self):
        if not self.tf:
            return
        if not os.path.isdir((self.filepath)):
            os.system('mkdir -p ' + (self.filepath))
        self.metrics = self.params['metrics']
        self.writer_train = None
        self.writer_test = None

    def on_epoch_end(self, epoch, epoch_log):
        if not self.tf:
            return
        else:
            tf = self.tf
        if self.training:
            if not self.writer_train:
                self.writer_train = tf.summary.FileWriter(os.path.join(self.filepath, 'train'))
            writer = self.writer_train
        else:
            if not self.writer_test:
                self.writer_test = tf.summary.FileWriter(os.path.join(self.filepath, 'eval'))
            writer = self.writer_test

        row_dict = dict()
        for k in self.metrics:
            if self._allow_unused() and (k not in epoch_log):
                continue
            row_dict[k] = epoch_log[k]

        summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in row_dict.items()])
        writer.add_summary(summary, epoch)
        writer.flush()

    def on_train_end(self):
        if not self.tf:
            return
        if self.writer_train:
            self.writer_train.flush()
            self.writer_train = None
        if self.writer_test:
            self.writer_test.flush()
            self.writer_test = None


class ComposeLogger(BaseLogger):
    """ loss logger to csv files """

    def __init__(self, loggers):
        self.loggers = loggers
        self.params = None
        self.model = None
        self._in_training = False

    def add_logger(self, logger):
        assert not self._in_training, str_error + \
            ' Unsafe to add logger during training'
        self.loggers.append(logger)

    def on_train_begin(self):
        self._in_training = True
        for logger in self.loggers:
            logger.on_train_begin()

    def on_train_end(self):
        self._in_training = False
        for logger in self.loggers:
            logger.on_train_end()

    def on_epoch_begin(self, epoch):
        for logger in self.loggers:
            logger.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, epoch_log):
        for logger in self.loggers:
            logger.on_epoch_end(epoch, epoch_log)

    def on_batch_begin(self, batch):
        for logger in self.loggers:
            logger.on_batch_begin(batch)

    def on_batch_end(self, batch, batch_log):
        for logger in self.loggers:
            logger.on_batch_end(batch, batch_log)

    def set_params(self, params):
        self.params = params
        for logger in self.loggers:
            logger.set_params(params)

    def set_model(self, model):
        self.model = model
        for logger in self.loggers:
            logger.set_model(model)

    def train(self):
        self.training = True
        for logger in self.loggers:
            logger.train()

    def eval(self):
        self.training = False
        for logger in self.loggers:
            logger.eval()


################################################
# Test BatchLogger, CsvLogger and ProgbarLogger
if __name__ == '__main__':
    test_logdir = './test_logger_dir'
    if os.path.isdir(test_logdir):
        os.system('rm -r ' + test_logdir)
    internal_logger = _LogCumulator()
    logger = ComposeLogger([internal_logger, ProgbarLogger(), BatchCsvLogger(
        test_logdir + '/batch_loss.csv'), CsvLogger(test_logdir + '/epoch_loss.csv')])
    logger.set_params({
        'epochs': 5,
        'steps': 20,
        'steps_eval': 5,
        'samples': 100,
        'samples_eval': 25,
        'verbose': 1,
        'metrics': ['loss']
    })
    logger.on_train_begin()
    for epoch in range(5):
        logger.train()
        logger.on_epoch_begin(epoch)
        for i in range(logger.params['steps']):
            logger.on_batch_begin(i)
            batch_log = {'batch': i, 'epoch': epoch, 'loss': np.random.rand(
                1)[0], 'size': np.random.randint(9) + 1}
            logger.on_batch_end(i, batch_log)
        epoch_log = internal_logger.get_epoch_log()
        logger.on_epoch_end(epoch, epoch_log)

        logger.eval()
        logger.on_epoch_begin(epoch)
        for i in range(logger.params['steps_eval']):
            logger.on_batch_begin(i)
            batch_log = {'batch': i, 'epoch': epoch,
                         'loss': np.random.rand(1)[0], 'size': 5}
            logger.on_batch_end(i, batch_log)
        epoch_log = internal_logger.get_epoch_log()
        logger.on_epoch_end(epoch, epoch_log)
    logger.on_train_end()
