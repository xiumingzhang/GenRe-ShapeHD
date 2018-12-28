import numpy as np
from scipy.misc import imresize
from numba import jit


@jit
def calc_ptnum(triangle, density):
    pt_num_tr = np.zeros(len(triangle)).astype(int)
    pt_num_total = 0
    for tr_id, tr in enumerate(triangle):
        a = np.linalg.norm(np.cross(tr[1] - tr[0], tr[2] - tr[0])) / 2
        ptnum = max(int(a * density), 1)
        pt_num_tr[tr_id] = ptnum
        pt_num_total += ptnum
    return pt_num_tr, pt_num_total


class Camera():
    # camera coordinates: y up, z forward, x right.
    # consistent with blender definitions.
    # res = [w,h]
    def __init__(self):
        self.position = np.array([1.6, 0, 0])
        self.rx = np.array([0, 1, 0])
        self.ry = np.array([0, 0, 1])
        self.rz = np.array([1, 0, 0])
        self.res = [800, 600]
        self.focal_length = 0.05
        # set the diagnal to be 35mm film's diagnal
        self.set_diagal((0.036**2 + 0.024**2)**0.5)

    def rotate(self, rot_mat):
        self.rx = rot_mat[:, 0]
        self.ry = rot_mat[:, 1]
        self.rz = rot_mat[:, 2]

    def move_cam(self, new_pos):
        self.position = new_pos

    def set_pose(self, inward, up):
        self.rx = np.cross(up, inward)
        self.ry = np.array(up)
        self.rz = np.array(inward)
        self.rx /= np.linalg.norm(self.rx)
        self.ry /= np.linalg.norm(self.ry)
        self.rz /= np.linalg.norm(self.rz)

    def set_diagal(self, diag):
        h_relative = self.res[1] / self.res[0]
        self.sensor_width = np.sqrt(diag**2 / (1 + h_relative**2))

    def lookat(self, orig, target, up):
        self.position = np.array(orig)
        target = np.array(target)
        inward = self.position - target
        right = np.cross(up, inward)
        up = np.cross(inward, right)
        self.set_pose(inward, up)

    def set_cam_from_mitsuba(self, path):
        camparam = util.cam_from_mitsuba(path)
        self.lookat(orig=camparam['origin'],
                    up=camparam['up'], target=camparam['target'])
        self.res = [camparam['width'], camparam['height']]
        self.focal_length = 0.05
        # set the diagnal to be 35mm film's diagnal
        self.set_diagal((0.036**2 + 0.024**2)**0.5)

    def project_point(self, pt):
        # project global point to image coordinates in pixels (float not
        # integer).
        res = self.res
        rel = np.array(pt) - self.position
        depth = -np.dot(rel, self.rz)
        if rel.ndim != 1:
            depth = depth.reshape([np.size(depth, axis=0), 1])
        rel_plane = rel * self.focal_length / depth
        rel_width = np.dot(rel_plane, self.rx)
        rel_height = np.dot(rel_plane, self.ry)
        topleft = np.array([-self.sensor_width / 2,
                            self.sensor_width * (res[1] / res[0]) / 2])
        pix_size = self.sensor_width / res[0]
        topleft += np.array([pix_size / 2, -pix_size / 2])
        im_pix_x = (topleft[1] - rel_height) / pix_size
        im_pix_y = (rel_width - topleft[0]) / pix_size
        return im_pix_x, im_pix_y

    def project_depth(self, pt, depth_type='ray'):
        if depth_type == 'ray':
            if np.array(pt).ndim == 1:
                return np.linalg.norm(pt - self.position)
            return np.linalg.norm(pt - self.position, axis=1)
        else:
            return np.dot(pt - self.position, -self.rz)

    def pack(self):
        params = []
        params += self.res
        params += [self.sensor_width]
        params += self.position.tolist()
        params += self.rx.tolist()
        params += self.ry.tolist()
        params += self.rz.tolist()
        params += [self.focal_length]
        return params


class tsdf_renderer:
    def __init__(self):
        self.camera = Camera()
        self.depth = []

    def load_depth_map_npy(self, path):
        self.depth = np.load(path)

    def back_project_ptcloud(self, upsample=1.0, depth_type='ray'):
        if not self.check_valid():
            return
        mask = np.where(self.depth < 0, 0, 1)
        depth = imresize(self.depth, upsample, mode='F', interp='bilinear')
        up_mask = imresize(mask, upsample, mode='F', interp='bilinear')
        up_mask = np.where(up_mask < 1, 0, 1)
        ind = np.where(up_mask == 0)
        depth[ind] = -1
        # res = self.camera.res
        res = np.array([0, 0])
        res[0] = np.shape(depth)[1]  # width
        res[1] = np.shape(depth)[0]  # height
        self.check_depth = np.zeros([res[1], res[0]], dtype=np.float32) - 1
        pt_pos = np.where(up_mask == 1)
        ptnum = len(pt_pos[0])
        ptcld = np.zeros([ptnum, 3])
        half_width = self.camera.sensor_width / 2
        half_height = half_width * res[1] / res[0]
        pix_size = self.camera.sensor_width / res[0]
        top_left = self.camera.position \
            - self.camera.focal_length * self.camera.rz\
            - half_width * self.camera.rx\
            + half_height * self.camera.ry

        for x in range(ptnum):
            height_id = pt_pos[0][x]
            width_id = pt_pos[1][x]
            pix_depth = depth[height_id, width_id]
            pix_coord = - (height_id + 0.5) * pix_size * self.camera.ry\
                + (width_id + 0.5) * pix_size * self.camera.rx\
                + top_left
            pix_rel = pix_coord - self.camera.position
            if depth_type == 'plane':
                ptcld_pos = (pix_rel)\
                    * (pix_depth / self.camera.focal_length) \
                    + self.camera.position
                back_project_depth = -np.dot(pix_rel, self.camera.rz)
            else:
                ptcld_pos = (pix_rel / np.linalg.norm(pix_rel))\
                    * (pix_depth) + self.camera.position
                back_project_depth = np.linalg.norm(
                    ptcld_pos - self.camera.position)
            ptcld[x, :] = ptcld_pos
            self.check_depth[height_id, width_id] = back_project_depth
        self.ptcld = ptcld
        self.pt_pos = pt_pos

    def check_valid(self, warning=True):
        if self.depth == []:
            print('No depth map available!')
            return False
        shape = np.shape(self.depth)
        if warning and (shape[0] != self.camera.res[1] or shape[1] != self.camera.res[0]):
            print('depth map and camera resolution mismatch!')
            print('camera: {}'.format(self.camera.res))
            print('depth:  {}'.format(shape))
            return True
        return True
