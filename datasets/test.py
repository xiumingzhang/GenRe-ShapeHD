from glob import glob
import numpy as np
import torch.utils.data as data
import util.util_img


class Dataset(data.Dataset):
    @classmethod
    def add_arguments(cls, parser):
        return parser, set()

    def __init__(self, opt, model):
        # Get required keys and preprocessing from the model
        required = model.requires
        self.preproc = model.preprocess_wrapper
        # Wrapper usually crops and resizes the input image (so that it's just
        # like our renders) before sending it to the actual preprocessing

        # Associate each data type required by the model with input paths
        type2filename = {}
        for k in required:
            type2filename[k] = getattr(opt, 'input_' + k)

        # Generate a sorted filelist for each data type
        type2files = {}
        for k, v in type2filename.items():
            type2files[k] = sorted(glob(v))
        ns = [len(x) for x in type2files.values()]
        assert len(set(ns)) == 1, \
            ("Filelists for different types must be of the same length "
             "(1-to-1 correspondance)")
        self.length = ns[0]

        samples = []
        for i in range(self.length):
            sample = {}
            for k, v in type2files.items():
                sample[k + '_path'] = v[i]
            samples.append(sample)
        self.samples = samples

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        sample = self.samples[i]

        # Actually loading the item
        sample_loaded = {}
        for k, v in sample.items():
            sample_loaded[k] = v  # as-is
            if k == 'rgb_path':
                im = util.util_img.imread_wrapper(
                    v, util.util_img.IMREAD_COLOR, output_channel_order='RGB')
                # Normalize to [0, 1] floats
                im = im.astype(float) / float(np.iinfo(im.dtype).max)
                sample_loaded['rgb'] = im
            elif k == 'mask_path':
                im = util.util_img.imread_wrapper(
                    v, util.util_img.IMREAD_GRAYSCALE)
                # Normalize to [0, 1] floats
                im = im.astype(float) / float(np.iinfo(im.dtype).max)
                sample_loaded['silhou'] = im
            else:
                raise NotImplementedError(v)

        # Preprocessing specified by the model
        sample_loaded = self.preproc(sample_loaded)
        # Convert all types to float32 for faster copying
        self.convert_to_float32(sample_loaded)
        return sample_loaded

    @staticmethod
    def convert_to_float32(sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)
