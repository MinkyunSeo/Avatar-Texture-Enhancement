import os
import sys
import math
from glob import glob
from random import randint

import tyro
import torch
import trimesh
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, Transform3d
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer import PointLights

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import get_renderer, get_normal_renderer


def make_rotate(rx, ry, rz):
    sinX = torch.sin(torch.deg2rad(torch.tensor(rx)))
    sinY = torch.sin(torch.deg2rad(torch.tensor(ry)))
    sinZ = torch.sin(torch.deg2rad(torch.tensor(rz)))

    cosX = torch.cos(torch.deg2rad(torch.tensor(rx)))
    cosY = torch.cos(torch.deg2rad(torch.tensor(ry)))
    cosZ = torch.cos(torch.deg2rad(torch.tensor(rz)))

    Rx = torch.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = torch.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = torch.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = torch.matmul(torch.matmul(Rz,Ry),Rx)
    return R


def render_data(
        dataroot: str,
        resolution: int = 512,
        scale: float = 0.9,
        rot_step: int = 8,
        aa_factor: int = 6,
        gpu_id: int = 0
    ) -> None:

    device = torch.device(f"cuda:{gpu_id}")

    mesh_dir = os.path.join(dataroot, 'meshes')
    mesh_paths = sorted(glob(os.path.join(mesh_dir, '**', '*.ply'), recursive=True))
    print(f"Number of meshes to render: {len(mesh_paths)}")

    # # Use nested loops to create a list of coordinate tuples
    # coordinates = [(xi, yi, zi) for xi in x for yi in y for zi in z]
    # # Convert the list to a NumPy array
    # locations = np.array(coordinates).reshape(-1, 1, 3)

    # locations = [[[0.0, 0.0, 0.0]],
    #              [[0.0, 0.0, 0.4]], [[0.0, 0.0, 1.0]],
    #              [[1.0, 1.0, 0.4]], [[-1.0, -1.0, 0.4]]]
    
    locations = [[[1.0, 1.0, 0.4]], [[0.0, 0.0, 0.4]], [[0.0, 0.0, 1.0]]]
    # ambient: dominant light source
    ambient_colors = [[[0.3, 0.3, 0.3]], [[0.45, 0.45, 0.45]], [[0.6, 0.6, 0.6]]]
    # diffuse: minor light source
    diffuse_colors = [[[0.7, 0.7, 0.7]], [[0.6, 0.6, 0.6]], [[0.4, 0.4, 0.4]]]

    for mesh_path in mesh_paths:
        subject_name = os.path.basename(mesh_path).split('.')[0]
        print(f"Rendering mesh {subject_name}")

        os.makedirs(os.path.join(dataroot, 'GEO', 'OBJ', subject_name),exist_ok=True)
        os.makedirs(os.path.join(dataroot, 'PARAM', subject_name),exist_ok=True)
        os.makedirs(os.path.join(dataroot, 'RENDER', subject_name),exist_ok=True)
        os.makedirs(os.path.join(dataroot, 'MASK', subject_name),exist_ok=True)

        # copy obj file
        cmd = 'cp %s %s' % (mesh_path, os.path.join(dataroot, 'GEO', 'OBJ', subject_name))
        print(cmd)
        os.system(cmd)

        if not os.path.exists(os.path.join(dataroot, 'val.txt')):
            f = open(os.path.join(dataroot, 'val.txt'), 'w')
            f.close()

        # Load mesh
        mesh = trimesh.load(mesh_path)
        verts, faces, vert_rgbas = map(
            lambda x: torch.tensor(x, dtype=torch.float32, device=device),
            (mesh.vertices, mesh.faces, mesh.visual.vertex_colors))
        
        for i in range(3): 
                # prev_loc, prev_amb, prev_diff = None, None, None

                # location = locations[randint(0, len(locations)-1)]
                # ambient_color = ambient_colors[randint(0, len(ambient_colors)-1)]
                # diffuse_color = diffuse_colors[randint(0, len(diffuse_colors)-1)]
                location = locations[i]
                ambient_color = ambient_colors[i]
                diffuse_color = diffuse_colors[i]

                # while (location == prev_loc) or (ambient_color == prev_amb) or (diffuse_color == prev_diff):
                #     location = locations[randint(0, len(locations)-1)]
                #     ambient_color = ambient_colors[randint(0, len(ambient_colors)-1)]
                #     diffuse_color = diffuse_colors[randint(0, len(diffuse_colors)-1)]

                # prev_loc, prev_amb, prev_diff = location, ambient_color, diffuse_color

                for y in tqdm(range(0, 360, rot_step)):
                    # Transform mesh into world coordinate
                    ## Translate mesh into the origin
                    vmin = verts.min(0).values
                    vmax = verts.max(0).values
                    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2

                    vmed = torch.median(verts, 0).values
                    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
                    transl = -vmed[None]

                    assert up_axis == 1
                    rot_mat = make_rotate(0, y, 0).to(device)

                    y_scale = (2.0 * scale)/(vmax[up_axis] - vmin[up_axis])
                    scale_mat = torch.eye(3, device=device)
                    scale_mat *= y_scale

                    model_mat = Transform3d.compose(
                        Translate(transl, device=device),
                        Rotate(rot_mat, device=device),
                        Rotate(scale_mat, device=device),
                    )

                    verts_world = model_mat.transform_points(verts)

                    # TODO: rgba?
                    mesh_py3d = Meshes(
                        verts=verts_world[None],
                        faces=faces[None],
                        textures=TexturesVertex(vert_rgbas[None, :, :3])
                    ).to(device)

                    lights = PointLights(device=device, location=location, ambient_color=ambient_color, diffuse_color=diffuse_color)
                    renderer = get_renderer(resolution, aa_factor, device, lights)

                    img_path = os.path.join(dataroot, 'RENDER', subject_name, f"{location}_{ambient_color}_{diffuse_color}_{y}.png")
                    msk_path = os.path.join(dataroot, 'MASK', subject_name, f"{location}_{ambient_color}_{diffuse_color}_{y}.png")
                    param_path = os.path.join(dataroot, 'PARAM', subject_name, f"{location}_{ambient_color}_{diffuse_color}_{y}.npy")

                    # Save Image
                    images = renderer(mesh_py3d)
                    images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
                    images = F.avg_pool2d(images, kernel_size=aa_factor, stride=aa_factor)
                    images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC

                    image_pil = Image.fromarray(images[0][..., :3].cpu().numpy().astype(np.uint8))
                    image_pil = image_pil.convert('RGB')
                    image_pil.save(img_path)

                    mask = (images > 0).float()*255
                    mask_pil = Image.fromarray(mask[0][..., 3].cpu().numpy().astype(np.uint8))
                    mask_pil = mask_pil.convert('L')
                    mask_pil.save(msk_path)

                    # Save param
                    param_dict = {'ortho_ratio': None, 'scale': float(y_scale), 'center': vmed.cpu().numpy(), 'R': rot_mat.inverse().cpu().numpy()}
                    np.save(param_path,
                            param_dict)

            # for location in locations:
            #     for ambient_color in ambient_colors:
            #         for diffuse_color in diffuse_colors:

            #             lights = PointLights(device=device, location=location, ambient_color=ambient_color, diffuse_color=diffuse_color)
            #             renderer = get_renderer(resolution, aa_factor, device, lights)

            #             img_path = os.path.join(dataroot, 'RENDER', subject_name, f"{location}_{ambient_color}_{diffuse_color}_{y}.png")
            #             msk_path = os.path.join(dataroot, 'MASK', subject_name, f"{location}_{ambient_color}_{diffuse_color}_{y}.png")
            #             param_path = os.path.join(dataroot, 'PARAM', subject_name, f"{location}_{ambient_color}_{diffuse_color}_{y}.npy")

            #             # Save Image
            #             images = renderer(mesh_py3d)
            #             images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
            #             images = F.avg_pool2d(images, kernel_size=aa_factor, stride=aa_factor)
            #             images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC

            #             image_pil = Image.fromarray(images[0][..., :3].cpu().numpy().astype(np.uint8))
            #             image_pil = image_pil.convert('RGB')
            #             image_pil.save(img_path)

            #             mask = (images > 0).float()*255
            #             mask_pil = Image.fromarray(mask[0][..., 3].cpu().numpy().astype(np.uint8))
            #             mask_pil = mask_pil.convert('L')
            #             mask_pil.save(msk_path)

            #             # Save param
            #             param_dict = {'ortho_ratio': None, 'scale': float(y_scale), 'center': vmed.cpu().numpy(), 'R': rot_mat.inverse().cpu().numpy()}
            #             np.save(param_path,
            #                     param_dict)

if __name__ == '__main__':
    tyro.cli(render_data)
