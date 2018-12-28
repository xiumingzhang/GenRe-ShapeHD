import os
import numpy as np
import torch
import collections
import time
# from torch._six import string_classes
try:
    from .util_print import str_warning
except ImportError:
    str_warning = '[Warning]'

def default_collate(batches):
    """ Merge batches of data into a larger batch. Input must be a list of batch outputs (list/dict).
    The each element of the batch outputs must be a numpy array, torch tensor or list """
    assert isinstance(batches, collections.Sequence)
    if isinstance(batches[0], collections.Mapping):
        return {key: _collate_list([d[key] for d in batches]) for key in batches[0]}
    elif isinstance(batches[0], collections.Sequence):
        transposed = zip(*batches)
        return [_collate_list(samples) for samples in transposed]


def _collate_list(list_of_data):
    # list_of_data = [elem.data if type(elem) is torch.autograd.Variable else elem for elem in list_of_data]   # remove torch tensors
    # list_of_data = [elem.cpu().numpy() if torch.is_tensor(elem) else elem for elem in list_of_data]   # remove torch tensors
    if type(list_of_data[0]).__module__ == 'numpy':
        return np.concatenate(list_of_data)
    elif isinstance(list_of_data[0], int):
        return list_of_data
    elif isinstance(list_of_data[0], float):
        return list_of_data
    elif isinstance(list_of_data[0], str):
        return list_of_data
    elif isinstance(list_of_data[0], collections.Sequence):
        return [x for subitem in list_of_data for x in subitem]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batches[0]))))


def default_subset(batch, indl, indh):
    # import pdb; pdb.set_trace()
    if isinstance(batch, collections.Sequence):
        return [elem[indl:indh] for elem in batch]
    elif isinstance(batch, collections.Mapping):
        return {key: batch[key][indl:indh] for key in batch}

    raise TypeError(("batch must contain dicts or lists; found {}"
                     .format(type(batch))))


def default_len(batch):
    if isinstance(batch, collections.Mapping):
        lens = {len(batch[key]) for key in batch}
    elif isinstance(batch, collections.Sequence):
        lens = {_item_len(elem) for elem in batch}
    else:
        raise TypeError(("batch must contain dicts or lists; found {}"
                         .format(type(elem))))

    assert len(lens) == 1, 'items of a batch output should have the same length. Got lengths: ' + str(lens)
    return next(iter(lens))     # get the only element from a set


def _item_len(elem):
    if isinstance(elem, int) or isinstance(elem, float) or isinstance(elem, str):
        return 1
    elif type(elem).__module__ == 'numpy':
        return elem.shape[0]
    elif isinstance(elem, collections.Sequence):
        return len(elem)
    else:
        raise TypeError(("batch must contain dicts or lists; found {}"
                         .format(type(elem))))


def default_save(savepath, batch):
    # print(batch)
    if type(batch).__module__ == 'numpy':
        np.savez_compressed(savepath, batch)
    elif isinstance(batch, collections.Sequence):
        np.savez_compressed(savepath, *batch)
    elif isinstance(batch, collections.Mapping):
        np.savez_compressed(savepath, **batch)
    else:
        raise TypeError(("batch must contain numpy arrays, dicts or lists; found {}"
                        .format(type(batch))))


def default_clean(batch):
    if isinstance(batch, str) or isinstance(batch, int) or isinstance(batch, float) or type(batch).__module__ == 'numpy':
        return batch
    if isinstance(batch, collections.Mapping):
        return {k: default_clean(v) for k, v in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [default_clean(v) for v in batch]
    elif type(batch) is torch.autograd.Variable: # class removed in pytorch 0.4.0
        return batch.detach().cpu().numpy()
    elif torch.is_tensor(batch):
        return batch.detach().cpu().numpy()
    else:
        raise TypeError(("batch elements must be int, float, str, torch tensor/variable, numpy arrays, dicts or lists; found {}"
                        .format(type(batch))))


class BatchSave(object):
    """
    This class is a general IO class aiming to offer saving flexibility.
    Note: data is only copied when saving, not at the time of adding (lazy copy).
          Do not mutate list/numpy arrays after adding. (torch tensors are fine as they're copied to numpy arrays at time of adding)
    """

    def __init__(self, savepath, filesize, *, collate_fn=default_collate, subset_fn=default_subset, len_fn=default_len, clean_fn=default_clean, verbose=False):
        """
        savepath: filename pattern for saved packed file. Use {ind} or formats like {ind:04d} for saved file index.
        filesize: number of data points per file. NOT size of file in bytes.
        collate_fn: Function to merge different batches of data.
        subset_fn: Function to get a subset of each added batch data.
        len_fn: Function to get length of each added batch data.
        clean_fn: Clean up input by changing data type. By default it maps all torch tensors to CPU numpy arrays.
        """
        self.savepath = savepath
        self.collate_fn = collate_fn
        self.subset_fn = subset_fn
        self.len_fn = len_fn
        self.clean_fn = clean_fn
        if os.path.isdir(os.path.dirname(savepath)):
            print(str_warning, 'Saving into an existing directory: %s' % os.path.dirname(savepath))
        else:
            os.system('mkdir -p %s' % os.path.dirname(savepath))
        self._saveind = 0
        self._buffer = list()
        self._buffer_size = 0
        self.filesize = filesize
        self.closed = False
        self.verbose = verbose

    def add_data(self, batch):
        """ Accept a batch of data and put it into a buffer. Save when buffer size is over file size.
        The batch must be a list/dict of list/np.array/torch.Tensor (each element must be iterable since it's defined on a batch)
        """
        assert not self.closed
        times = [time.time()]
        batch = self.clean_fn(batch)
        times.append(time.time())
        self._buffer_size += self.len_fn(batch)
        times.append(time.time())
        self._buffer.append(batch)
        times.append(time.time())
        while self._buffer_size >= self.filesize:
            buffer_data = self.collate_fn(self._buffer)
            times.append(time.time())
            data_to_save = self.subset_fn(buffer_data, 0, self.filesize)
            times.append(time.time())
            default_save(self.savepath.format(ind=self._saveind), data_to_save)
            times.append(time.time())
            self._buffer = [self.subset_fn(buffer_data, self.filesize, self._buffer_size)]
            times.append(time.time())
            self._buffer_size -= self.filesize
            self._saveind += 1
        times.append(time.time())
        if self.verbose:
            print(*[times[i+1] - times[i] for i in range(len(times)-1)])

    def close(self):
        if self._buffer_size > 0:
            buffer_data = self.collate_fn(self._buffer)
            default_save(self.savepath.format(ind=self._saveind), buffer_data)
            self._saveind += 1
        self.closed = True

    def get_fileind(self):
        return self._saveind

    def get_buffer_size(self):
        return self._buffer_size


#################################################
# Test
if __name__ == '__main__':
    fout = BatchSave('./test_{ind}.npz', 3)
    for i in range(10):
        fout.add_data([[i]])
    fout.close()

    for i in range(fout._saveind):
        data = (np.load('./test_{ind}.npz'.format(ind=i)))
        for k, v in data.items():
            print(k, v)

    fout = BatchSave('./test_{ind}.npz', 3)
    for i in range(10):
        fout.add_data([np.array([i])])
    fout.close()

    for i in range(fout._saveind):
        data = (np.load('./test_{ind}.npz'.format(ind=i)))
        for k, v in data.items():
            print(k, v)

    fout = BatchSave('./test_{ind}.npz', 3)
    for i in range(10):
        fout.add_data([torch.FloatTensor([i])])
    fout.close()

    for i in range(fout._saveind):
        data = (np.load('./test_{ind}.npz'.format(ind=i)))
        for k, v in data.items():
            print(k, v)

    fout = BatchSave('./test_{ind}.npz', 3)
    for i in range(10):
        fout.add_data([torch.FloatTensor([i, i, i, i])])
    fout.close()

    for i in range(fout._saveind):
        data = (np.load('./test_{ind}.npz'.format(ind=i)))
        for k, v in data.items():
            print(k, v)

    fout = BatchSave('./test_{ind}.npz', 3)
    for i in range(10):
        fout.add_data([[0, 0], np.array([i, i])])
    fout.close()

    for i in range(fout._saveind):
        data = (np.load('./test_{ind}.npz'.format(ind=i)))
        for k, v in data.items():
            print(k, v)

    fout = BatchSave('./test_{ind}.npz', 3)
    for i in range(10):
        fout.add_data({'a': [0, 0], 'b': np.array([i, i])})
    fout.close()

    for i in range(fout._saveind):
        data = (np.load('./test_{ind}.npz'.format(ind=i)))
        for k, v in data.items():
            print(k, v)

    fout = BatchSave('./test_{ind}.npz', 1)
    for i in range(10):
        fout.add_data({'a': [0, 0], 'b': np.array([i, i])})
    fout.close()

    for i in range(fout._saveind):
        data = (np.load('./test_{ind}.npz'.format(ind=i)))
        for k, v in data.items():
            print(k, v)

    fout = BatchSave('./test_{ind}.npz', 21)
    for i in range(10):
        fout.add_data({'a': [0, 0], 'b': np.array([i, i])})
    fout.close()

    for i in range(fout._saveind):
        data = (np.load('./test_{ind}.npz'.format(ind=i)))
        for k, v in data.items():
            print(k, v)

    os.system('rm ./test_*.npz')
