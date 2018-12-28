from os.path import join, dirname
from os import makedirs
from shutil import copyfile
from multiprocessing import Pool
import atexit
import json
import numpy as np
from skimage import measure
from util.util_img import imwrite_wrapper


class Visualizer():
    """
    Unified Visulization Worker
    """
    paths = [
        'rgb_path',
        'silhou_path',
        'depth_path',
        'normal_path',
    ]
    imgs = [
        'rgb',
        'pred_depth',
        'pred_silhou',
        'pred_normal',
    ]
    voxels = [
        'pred_voxel_noft',
        'pred_voxel',
        'gen_voxel',
    ]  # will go through sigmoid
    txts = [
        'gt_depth_minmax',
        'pred_depth_minmax',
        'disc',
        'scores'
    ]
    sphmaps = [
        'pred_spherical_full',
        'pred_spherical_partial',
        'gt_spherical_full',
    ]
    voxels_gt = [
        'pred_proj_depth',
        'gt_voxel',
        'pred_proj_sph_full',
    ]

    def __init__(self, n_workers=4, param_f=None):
        if n_workers == 0:
            pool = None
        elif n_workers > 0:
            pool = Pool(n_workers)
        else:
            raise ValueError(n_workers)
        self.pool = pool
        if param_f:
            self.param_f = param_f
        else:
            self.param_f = join(dirname(__file__), 'config.json')

        def cleanup():
            if pool:
                pool.close()
                pool.join()
        atexit.register(cleanup)

    def visualize(self, pack, batch_idx, outdir):
        if self.pool:
            self.pool.apply_async(
                self._visualize,
                [pack, batch_idx, self.param_f, outdir],
                error_callback=self._error_callback
            )
        else:
            self._visualize(pack, batch_idx, self.param_f, outdir)

    @classmethod
    def _visualize(cls, pack, batch_idx, param_f, outdir):
        makedirs(outdir, exist_ok=True)

        # Dynamically read parameters from disk
        #param_dict = cls._read_params(param_f)
        voxel_isosurf_th = 0.25  # param_dict['voxel']['isosurf_thres']

        batch_size = cls._get_batch_size(pack)
        instance_cnt = batch_idx * batch_size
        counter = 0
        for k in cls.paths:
            prefix = '{:04d}_%02d_' % counter + k.split('_')[0] + '.png'
            cls._cp_img(pack.get(k), join(outdir, prefix), instance_cnt)
            counter += 1
        for k in cls.imgs:
            prefix = '{:04d}_%02d_' % counter + k + '.png'
            cls._vis_img(pack.get(k), join(outdir, prefix), instance_cnt)
            counter += 1
        for k in cls.voxels_gt:
            prefix = '{:04d}_%02d_' % counter + k + '.obj'
            cls._vis_voxel(pack.get(k), join(outdir, prefix), instance_cnt,
                           voxel_isosurf_th, False)
            counter += 1
        for k in cls.voxels:
            prefix = '{:04d}_%02d_' % counter + k + '.obj'
            cls._vis_voxel(pack.get(k), join(outdir, prefix), instance_cnt,
                           voxel_isosurf_th)
            counter += 1
        for k in cls.txts:
            prefix = '{:04d}_%02d_' % counter + k + '.txt'
            cls._vis_txt(pack.get(k), join(outdir, prefix), instance_cnt)
            counter += 1
        for k in cls.sphmaps:
            prefix = '{:04d}_%02d_' % counter + k + '.png'
            cls._vis_sph(pack.get(k), join(outdir, prefix), instance_cnt)
            counter += 1

    @staticmethod
    def _read_params(param_f):
        with open(param_f, 'r') as h:
            param_dict = json.load(h)
        return param_dict

    @staticmethod
    def _get_batch_size(pack):
        batch_size = None
        for v in pack.values():
            if hasattr(v, 'shape'):
                if batch_size is None or batch_size == 0:
                    batch_size = v.shape[0]
                else:
                    assert batch_size == v.shape[0]
        return batch_size

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _to_obj_str(verts, faces):
        text = ""
        for p in verts:
            text += "v "
            for x in p:
                text += "{} ".format(x)
            text += "\n"
        for f in faces:
            text += "f "
            for x in f:
                text += "{} ".format(x + 1)
            text += "\n"
        return text

    @classmethod
    def _save_iso_obj(cls, df, path, th, shift=True):
        if th < np.min(df):
            df[0, 0, 0] = th - 1
        if th > np.max(df):
            df[-1, -1, -1] = th + 1
        spacing = (1 / 128, 1 / 128, 1 / 128)
        verts, faces, _, _ = measure.marching_cubes_lewiner(
            df, th, spacing=spacing)
        if shift:
            verts -= np.array([0.5, 0.5, 0.5])
        obj_str = cls._to_obj_str(verts, faces)
        with open(path, 'w') as f:
            f.write(obj_str)

    @staticmethod
    def _vis_img(img, output_pattern, counter=0):
        if img is not None and not isinstance(img, str):
            assert img.shape[0] != 0
            img = np.clip(img * 255, 0, 255).astype(int)
            img = np.transpose(img, (0, 2, 3, 1))
            bsize = img.shape[0]
            for batch_id in range(bsize):
                im = img[batch_id, :, :, :]
                imwrite_wrapper(output_pattern.format(counter + batch_id), im)

    @staticmethod
    def _vis_sph(img, output_pattern, counter=0):
        if img is not None and not isinstance(img, str):
            assert img.shape[0] != 0
            img = np.transpose(img, (0, 2, 3, 1))
            bsize = img.shape[0]
            for batch_id in range(bsize):
                im = img[batch_id, :, :, 0]
                im = im / im.max()
                im = np.clip(im * 255, 0, 255).astype(int)
                imwrite_wrapper(output_pattern.format(counter + batch_id), im)

    @staticmethod
    def _cp_img(paths, output_pattern, counter=0):
        if paths is not None:
            for batch_id, path in enumerate(paths):
                copyfile(path, output_pattern.format(counter + batch_id))

    @classmethod
    def _vis_voxel(cls, voxels, output_pattern, counter=0, th=0.5, use_sigmoid=True):
        if voxels is not None:
            assert voxels.shape[0] != 0
            for batch_id, voxel in enumerate(voxels):
                if voxel.ndim == 4:
                    voxel = voxel[0, ...]
                voxel = cls._sigmoid(voxel) if use_sigmoid else voxel
                cls._save_iso_obj(voxel, output_pattern.format(counter + batch_id), th=th)

    @staticmethod
    def _vis_txt(txts, output_pattern, counter=0):
        if txts is not None:
            for batch_id, txt in enumerate(txts):
                with open(output_pattern.format(counter + batch_id), 'w') as h:
                    h.write("%s\n" % txt)

    @staticmethod
    def _error_callback(e):
        print(str(e))
