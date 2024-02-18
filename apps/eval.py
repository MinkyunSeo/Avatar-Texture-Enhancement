import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import torch
import trimesh
import math
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from pytorch3d.loss import chamfer_distance

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *
from lib.data import *
from lib.metrics import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

import csv

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s/recon' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image_path, mask_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        # Mask
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            exp_name = opt.name
            sample_name = data['name']
            save_path = f'{opt.results_path}/{exp_name}/recon/{exp_name}_{sample_name}.obj'
            if self.netC:
                verts, faces, color = gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                verts, faces, color = gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)
            return verts, faces, color

if __name__ == '__main__':

    evaluator = Evaluator(opt)

    test_dataset = TrainDataset(opt, phase='test')
    print(f"Testing on {len(test_dataset)} meshes")
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    # Open a CSV file in write mode
    csv_file_path = os.path.join(opt.results_path, opt.name, 'output.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(['Test Index', 'Subject Index', 'Yaw Index', 'Augmentation Index', 'PATD'])

        for test_idx, test_data in enumerate(test_data_loader):
            test_data = {k: val[0] for k, val in test_data.items()}
            verts_pred, faces_pred, colors_pred = evaluator.eval(test_data, True)
            verts_pred = torch.tensor(verts_pred.copy(), dtype=torch.float32)
            faces_pred = torch.tensor(faces_pred.copy(), dtype=torch.int32)
            colors_pred = torch.tensor(colors_pred.copy(), dtype=torch.float32) * 255.

            mesh_gt = trimesh.load(test_data['mesh_path'])
            verts_gt = torch.tensor(mesh_gt.vertices, dtype=torch.float32)
            faces_gt = torch.tensor(mesh_gt.faces, dtype=torch.int32)
            colors_gt = torch.tensor(mesh_gt.visual.vertex_colors, dtype=torch.float32)[..., :3]

            PATD_value = PATD(verts_pred, colors_pred, verts_gt, colors_gt)
            CFD_value = chamfer_distance(verts_pred, verts_gt)

            # Write a row for each test case
            csv_writer.writerow([test_idx, test_data['name'], f'{PATD_value:.4f}', f'{CFD_value:.4f}'])
            print(f"{test_idx} | {test_data['name']} | {test_data['yid']} | {test_data['lid']} | PATD: {PATD_value:.4f} | CFD: {CFD_value:.4f}")
