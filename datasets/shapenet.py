from os.path import join
import random
import numpy as np
from scipy.io import loadmat
import torch.utils.data as data
import util.util_img


class Dataset(data.Dataset):
    data_root = './downloads/data/shapenet'
    list_root = join(data_root, 'status')
    status_and_suffix = {
        'rgb': {
            'status': 'rgb.txt',
            'suffix': '_rgb.png',
        },
        'depth': {
            'status': 'depth.txt',
            'suffix': '_depth.png',
        },
        'depth_minmax': {
            'status': 'depth_minmax.txt',
            'suffix': '.npy',
        },
        'silhou': {
            'status': 'silhou.txt',
            'suffix': '_silhouette.png',
        },
        'normal': {
            'status': 'normal.txt',
            'suffix': '_normal.png'
        },
        'voxel': {
            'status': 'vox_rot.txt',
            'suffix': '_gt_rotvox_samescale_128.npz'
        },
        'spherical': {
            'status': 'spherical.txt',
            'suffix': '_spherical.npz'
        },
        'voxel_canon': {
            'status': 'vox_canon.txt',
            'suffix': '_voxel_normalized_128.mat'
        },
    }
    class_aliases = {
        'drc': '03001627+02691156+02958343',
        'chair': '03001627',
        'table': '04379243',
        'sofa': '04256520',
        'couch': '04256520',
        'cabinet': '03337140',
        'bed': '02818832',
        'plane': '02691156',
        'car': '02958343',
        'bench': '02828884',
        'monitor': '03211117',
        'lamp': '03636649',
        'speaker': '03691459',
        'firearm': '03948459+04090263',
        'cellphone': '02992529+04401088',
        'watercraft': '04530566',
        'hat': '02954340',
        'pot': '03991062',
        'rocket': '04099429',
        'train': '04468005',
        'bus': '02924116',
        'pistol': '03948459',
        'faucet': '03325088',
        'helmet': '03513137',
        'clock': '03046257',
        'phone': '04401088',
        'display': '03211117',
        'vessel': '04530566',
        'rifle': '04090263',
        'small': '03001627+04379243+02933112+04256520+02958343+03636649+02691156+04530566',
        'all-but-table': '02691156+02747177+02773838+02801938+02808440+02818832+02828884+02843684+02871439+02876657+02880940+02924116+02933112+02942699+02946921+02954340+02958343+02992529+03001627+03046257+03085013+03207941+03211117+03261776+03325088+03337140+03467517+03513137+03593526+03624134+03636649+03642806+03691459+03710193+03759954+03761084+03790512+03797390+03928116+03938244+03948459+03991062+04004475+04074963+04090263+04099429+04225987+04256520+04330267+04401088+04460130+04468005+04530566+04554684',
        'all-but-chair': '02691156+02747177+02773838+02801938+02808440+02818832+02828884+02843684+02871439+02876657+02880940+02924116+02933112+02942699+02946921+02954340+02958343+02992529+03046257+03085013+03207941+03211117+03261776+03325088+03337140+03467517+03513137+03593526+03624134+03636649+03642806+03691459+03710193+03759954+03761084+03790512+03797390+03928116+03938244+03948459+03991062+04004475+04074963+04090263+04099429+04225987+04256520+04330267+04379243+04401088+04460130+04468005+04530566+04554684',
        'all': '02691156+02747177+02773838+02801938+02808440+02818832+02828884+02843684+02871439+02876657+02880940+02924116+02933112+02942699+02946921+02954340+02958343+02992529+03001627+03046257+03085013+03207941+03211117+03261776+03325088+03337140+03467517+03513137+03593526+03624134+03636649+03642806+03691459+03710193+03759954+03761084+03790512+03797390+03928116+03938244+03948459+03991062+04004475+04074963+04090263+04099429+04225987+04256520+04330267+04379243+04401088+04460130+04468005+04530566+04554684',
    }
    class_list = class_aliases['all'].split('+')

    @classmethod
    def add_arguments(cls, parser):
        return parser, set()

    @classmethod
    def read_bool_status(cls, status_file):
        with open(join(cls.list_root, status_file)) as f:
            lines = f.read()
        return [x == 'True' for x in lines.split('\n')[:-1]]

    def __init__(self, opt, mode='train', model=None):
        assert mode in ('train', 'vali')
        self.mode = mode
        if model is None:
            required = ['rgb']
            self.preproc = None
        else:
            required = model.requires
            self.preproc = model.preprocess

        # Parse classes
        classes = []  # alias to real for locating data
        class_str = ''  # real to alias for logging
        for c in opt.classes.split('+'):
            class_str += c + '+'
            if c in self.class_aliases:  # nickname given
                classes += self.class_aliases[c].split('+')
            else:
                classes = c.split('+')
        class_str = class_str[:-1]  # removes the final +
        classes = sorted(list(set(classes)))

        # Load items and train-test split
        with open(join(self.list_root, 'items_all.txt')) as f:
            lines = f.read()
        item_list = lines.split('\n')[:-1]
        is_train = self.read_bool_status('is_train.txt')
        assert len(item_list) == len(is_train)

        # Load status the network requires
        has = {}
        for data_type in required:
            assert data_type in self.status_and_suffix.keys(), \
                "%s required, but unspecified in status_and_suffix" % data_type
            has[data_type] = self.read_bool_status(
                self.status_and_suffix[data_type]['status']
            )
            assert len(has[data_type]) == len(item_list)

        # Pack paths into a dict
        samples = []
        for i, item in enumerate(item_list):
            class_id, _ = item.split('/')[:2]
            item_in_split = ((self.mode == 'train') == is_train[i])
            if item_in_split and class_id in classes:
                # Look up subclass_id for this item
                sample_dict = {'item': join(self.data_root, item)}
                # As long as a type is required, it appears as a key
                # If it doens't exist, its value will be None
                for data_type in required:
                    suffix = self.status_and_suffix[data_type]['suffix']
                    k = data_type + '_path'
                    if data_type == 'voxel_canon':
                        # All different views share the same canonical voxel
                        sample_dict[k] = join(self.data_root, item.split('_view')[0] + suffix) \
                            if has[data_type][i] else None
                    else:
                        sample_dict[k] = join(self.data_root, item + suffix) \
                            if has[data_type][i] else None
                if None not in sample_dict.values():
                    # All that are required exist
                    samples.append(sample_dict)

        # If validation, dataloader shuffle will be off, so need to DETERMINISTICALLY
        # shuffle here to have a bit of every class
        if self.mode == 'vali':
            if opt.manual_seed:
                seed = opt.manual_seed
            else:
                seed = 0
            random.Random(seed).shuffle(samples)
        self.samples = samples

    def __getitem__(self, i):
        sample_loaded = {}
        for k, v in self.samples[i].items():
            sample_loaded[k] = v  # as-is
            if k.endswith('_path'):
                if v.endswith('.png'):
                    im = util.util_img.imread_wrapper(
                        v, util.util_img.IMREAD_UNCHANGED,
                        output_channel_order='RGB')
                    # Normalize to [0, 1] floats
                    im = im.astype(float) / float(np.iinfo(im.dtype).max)
                    sample_loaded[k[:-5]] = im
                elif v.endswith('.npy'):
                    # Right now .npy must be depth_minmax
                    sample_loaded['depth_minmax'] = np.load(v)
                elif v.endswith('_128.npz'):
                    sample_loaded['voxel'] = np.load(v)['voxel'][None, ...]
                elif v.endswith('_spherical.npz'):
                    spherical_data = np.load(v)
                    sample_loaded['spherical_object'] = spherical_data['obj_spherical'][None, ...]
                    sample_loaded['spherical_depth'] = spherical_data['depth_spherical'][None, ...]
                elif v.endswith('.mat'):
                    # Right now .mat must be voxel_canon
                    sample_loaded['voxel_canon'] = loadmat(v)['voxel'][None, ...]
                else:
                    raise NotImplementedError(v)
            # Three identical channels for grayscale images
        if self.preproc is not None:
            sample_loaded = self.preproc(sample_loaded, mode=self.mode)
        # convert all types to float32 for better copy speed
        self.convert_to_float32(sample_loaded)
        return sample_loaded

    @staticmethod
    def convert_to_float32(sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def get_classes(self):
        return self._class_str
