from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging

log = logging.getLogger('trimesh')
log.setLevel(40)

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s.ply' % sub_name))

    return meshs

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

        self.B_MIN = np.array([-1, -1, -1])
        self.B_MAX = np.array([1, 1, 1])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0, 360, opt.rot_step))
        self.locations = [[[1.0, 1.0, 0.4]], [[0.0, 0.0, 0.4]], [[0.0, 0.0, 1.0]]]
        # ambient: dominant light source
        self.ambient_colors = [[[0.3, 0.3, 0.3]], [[0.45, 0.45, 0.45]], [[0.6, 0.6, 0.6]]]
        # diffuse: minor light source
        self.diffuse_colors = [[[0.7, 0.7, 0.7]], [[0.6, 0.6, 0.6]], [[0.4, 0.4, 0.4]]]

        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self):
        all_subjects = os.listdir(self.RENDER)
        var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) \
            * len(self.yaw_list) \
            * len(self.locations) 
            # * len(self.ambient_colors) \
            # * len(self.diffuse_colors)

    def get_render(self, subject, yid, lid, aid, did):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        yaw = self.yaw_list[yid]
        location = self.locations[lid]
        ambient_color = self.ambient_colors[aid]
        diffuse_color = self.diffuse_colors[did]

        param_path = os.path.join(self.PARAM, subject, f"{location}_{ambient_color}_{diffuse_color}_{yaw}.npy")
        render_path = os.path.join(self.RENDER, subject, f"{location}_{ambient_color}_{diffuse_color}_{yaw}.png")
        mask_path = os.path.join(self.MASK, subject, f"{location}_{ambient_color}_{diffuse_color}_{yaw}.png")

        # loading calibration data
        param = np.load(param_path, allow_pickle=True)
        # pixel unit / world unit
        # ortho_ratio = param.item().get('ortho_ratio')
        # world unit / model unit
        scale = param.item().get('scale')
        # camera center world coordinate
        center = param.item().get('center')
        # model rotation
        R = param.item().get('R')

        translate = -np.matmul(R, center).reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        # Match camera space to image pixel space
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = scale
        scale_intrinsic[1, 1] = -scale
        scale_intrinsic[2, 2] = scale
        # Match image pixel space to image uv space
        uv_intrinsic = np.identity(4)
        # uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
        # uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
        # uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
        # Transform under image pixel space
        trans_intrinsic = np.identity(4)

        mask = Image.open(mask_path).convert('L')
        render = Image.open(render_path).convert('RGB')

        intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        extrinsic = torch.Tensor(extrinsic).float()

        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        mask_list.append(mask)

        render = self.to_tensor(render)
        render = mask.expand_as(render) * render

        render_list.append(render)
        calib_list.append(calib)
        extrinsic_list.append(extrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0)
        }

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        mesh = self.mesh_dic[subject]
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)

        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]

        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        # save_samples_truncted_prob('out.ply', samples.T, labels.T)
        # exit()

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()

        del mesh

        return {
            'samples': samples,
            'labels': labels
        }


    def get_color_sampling(self, subject):
        mesh = self.mesh_dic[subject]

        surface_points, faces = mesh.sample(self.num_sample_color, return_index=True)
        surface_normal = mesh.face_normals[faces]
        surface_colors = mesh.visual.face_colors[faces][..., :3]/255.
        vertex_colors = mesh.visual.vertex_colors[:, :3]/255.

        if self.opt.patch_sample == True:
            patch_len = self.opt.patch_len
            patch_size = self.opt.patch_size

            patch = np.mgrid[:patch_size, :patch_size, :patch_size]
            patch = patch.transpose(1, 2, 3, 0)   # [patch_len, patch_len, patch_len, 3]
            patch = 2. * (patch / patch_size) - 1.0
            patch = patch * patch_len

            patch = np.tile(patch[None], [self.num_sample_color, 1, 1, 1, 1])
            samples = patch + surface_points[:, None, None, None, :]
            samples = samples.reshape(-1, patch_size**3, 3)  # [N, patch_len**3, 3]

            # nearnest neighbor or texture mapping?
            nn_index = mesh.kdtree.query(samples)[1]
            rgbs_color = vertex_colors[nn_index]
            rgbs_color = 2.0 * torch.Tensor(rgbs_color).float() - 1.0

            normal = torch.Tensor(surface_normal).float()[:, None, :]
            normal = normal.repeat(1, patch_size**3, 1)
            # samples = torch.Tensor(samples).float() \
            #         + torch.normal(mean=torch.zeros((1, 1, 3)), std=self.opt.sigma).expand_as(normal) * normal
            samples = torch.Tensor(samples).float()

            samples = samples.permute(2, 1, 0)
            rgbs_color = rgbs_color.permute(2, 1, 0)

        else:
            # Samples are around the true surface with an offset
            normal = torch.Tensor(surface_normal).float()
            samples = torch.Tensor(surface_points).float() \
                    + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal
            # Normalized to [-1, 1]
            rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

            samples = samples.T
            rgbs_color = rgbs_color.T

        return {
            'color_samples': samples,
            'rgbs': rgbs_color
        }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        tmp = tmp // len(self.yaw_list)
        lid = tmp % len(self.locations)
        aid = lid
        did = lid
        # tmp = tmp // len(self.locations)
        # aid = tmp % len(self.ambient_colors)
        # did = tmp // len(self.ambient_colors)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': f"{subject}",
            'mesh_path': os.path.join(self.OBJ, f"{subject}/{subject}.ply"),
            'sid': sid,
            'yid': yid,
            'lid': lid,
            'aid': aid,
            'did': did,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(
            subject, yid=yid, lid=lid, aid=aid, did=did)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)

        # img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        # rot = render_data['calib'][0,:3, :3]
        # trans = render_data['calib'][0,:3, 3:4]
        # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        # for p in pts:
        #     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        if self.num_sample_color:
            color_data = self.get_color_sampling(subject)
            res.update(color_data)
        return res
        # except Exception as e:
        #     print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)


if __name__ == "__main__":
    from lib.options import BaseOptions
    opt = BaseOptions().parse()

    dataset = TrainDataset(opt)
    data = dataset[0]
