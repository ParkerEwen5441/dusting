import os
import cv2
import sys
import torch
import trimesh
import numpy as np
from PIL import Image
from pathlib import Path
import lovely_tensors as lt
import matplotlib.pyplot as plt
from typing import NamedTuple, Optional
from dataclasses import dataclass, field

# Import submodules
sys.path.append('../submodules/dust3r')
sys.path.append('../submodules/gaussian-splatting')

from dust3r.inference import inference, load_model                  #
from dust3r.utils.image import load_images                          #
from dust3r.utils.device import to_numpy                            # DUST3R
from dust3r.image_pairs import make_pairs                           #
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode      #

from scene.gaussian_model import BasicPointCloud                    #
from scene.colmap_loader import rotmat2qvec                         # Gaussian-Splatting
from utils.graphics_utils import focal2fov, fov2focal               # 
from scene.dataset_readers import storePly                          #

lt.monkey_patch()

@dataclass
class UtilParams:
    def __init__(self):
        # General Parameters
        self.device: str = "cuda:0"
        self.save_dir = Path('../data/scenes/turtle')
        self.image_dir: str = "../data/images/turtle_imgs/"

        # DUST3R Parameters
        self.lr: float = 0.01
        self.niter: int = 300
        self.batch_size: int = 1
        self.schedule: str = "cosine"
        self.model_path: str = "../submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

        # Gaussian Splatting Parameters
        self.min_conf_thr: int = 20

        # Read images in image_dir
        self.image_files()

        # Make save_dir for outputs
        self.init_filestructure()
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.depth_max = 2

    def image_files(self):
        Path.ls = lambda x: list(x.iterdir())
        image_data_dir = Path(self.image_dir)
        image_files = [str(x) for x in image_data_dir.ls() if x.suffix in ['.png', '.jpg']]
        self.image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    def init_filestructure(self):
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.images_path = self.save_dir / 'images'
        self.depths_path = self.save_dir / 'depths'
        self.sparse_path = self.save_dir / 'sparse/0'
        
        self.images_path.mkdir(exist_ok=True, parents=True)
        self.depths_path.mkdir(exist_ok=True, parents=True)
        self.sparse_path.mkdir(exist_ok=True, parents=True)


class DUST3R():
    def __init__(self, data : UtilParams):
        """
        Initializes the DUST3R model.
        
        :param      data:  data for the model
        :type       data:  UtilParams
        """
        model = load_model(data.model_path, data.device)
        images = load_images(data.image_files, size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, data.device, batch_size=data.batch_size)
        scene = global_aligner(output, device=data.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=data.niter, schedule=data.schedule, lr=data.lr)

        self.intrinsics = scene.get_intrinsics().detach().cpu().numpy()
        self.world2cam = self.inv(scene.get_im_poses().detach()).cpu().numpy()
        self.principal_points = scene.get_principal_points().detach().cpu().numpy()
        self.focals = scene.get_focals().detach().cpu().numpy()
        self.imgs = np.array(scene.imgs)
        self.pts3d = [i.detach() for i in scene.get_pts3d()]
        self.depth_maps = [i.detach() for i in scene.get_depthmaps()]

        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(data.min_conf_thr)))
        self.masks = to_numpy(scene.get_masks())        

    def save(self, data : UtilParams):
        # Saving images and optionally depth maps
        for i, (depth, image) in enumerate(zip(self.depth_maps, self.imgs)):
            image_save_path = data.images_path / f"{i}.png"
            rgb_image = cv2.cvtColor(image*255, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(image_save_path), rgb_image)

            depth_save_path = data.depths_path / f"{i}.png"
            depth_normalized = depth.cpu().numpy() / data.depth_max
            Image.fromarray((depth_normalized * 255).astype(np.uint8), mode='L').save(depth_save_path)
        
        # Save cameras.txt
        cameras_file = data.sparse_path / 'cameras.txt'
        with open(cameras_file, 'w') as cameras_file:
            cameras_file.write("# Camera list with one line of data per camera:\n")
            cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            for i, (focal, pp) in enumerate(zip(self.focals, self.principal_points)):
                cameras_file.write(f"{i} PINHOLE {self.imgs.shape[2]} {self.imgs.shape[1]} {focal[0]} {focal[0]} {pp[0]} {pp[1]}\n")

        # Save images.txt
        images_file = data.sparse_path / 'images.txt'
        # Generate images.txt content
        with open(images_file, 'w') as images_file:
            images_file.write("# Image list with two lines of data per image:\n")
            images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for i in range(self.world2cam.shape[0]):
                # Convert rotation matrix to quaternion
                rotation_matrix = self.world2cam[i, :3, :3]
                qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
                tx, ty, tz = self.world2cam[i, :3, 3]
                images_file.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {i}.png\n")
                images_file.write("\n") # Placeholder for points, assuming no points are associated with images here

        pc = self.get_pc(self.imgs, self.pts3d, self.masks)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

        # Define a default normal, e.g., [0, 1, 0]
        default_normal = [0, 1, 0]

        # Prepare vertices, colors, and normals for saving
        vertices = pc.vertices
        colors = pc.colors
        normals = np.tile(default_normal, (vertices.shape[0], 1))

        # Construct the header of the PLY file
        header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

        # Write the PLY file
        save_path = data.sparse_path / 'points3D.ply'
        with open(save_path, 'w') as ply_file:
            ply_file.write(header)
            for vertex, color, normal in zip(vertices, colors, normals):
                ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                    vertex[0], vertex[1], vertex[2],
                    int(color[0]), int(color[1]), int(color[2]),
                    normal[0], normal[1], normal[2]
                ))

    def inv(self, mat):
        """
        Inverts torch or numpy matrix.
        
        :param      mat:  The matrix
        :type       mat:  torch.Tensor or numpy.ndarray
        """
        if isinstance(mat, torch.Tensor):
            return torch.linalg.inv(mat)
        if isinstance(mat, np.ndarray):
            return np.linalg.inv(mat)
        raise ValueError(f'bad matrix type = {type(mat)}')

    def get_pc(self, imgs, pts3d, mask):
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)

        pts = np.concatenate([p for p in pts3d])
        col = np.concatenate([p for p in imgs])
        
        # pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        # col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        
        pts = pts.reshape(-1, 3)[::3]
        col = col.reshape(-1, 3)[::3]
        
        #mock normals:
        normals = np.tile([0, 1, 0], (pts.shape[0], 1))
        
        pct = trimesh.PointCloud(pts, colors=col)
        pct.vertices_normal = normals  # Manually add normals to the point cloud
        
        return pct

if __name__ == "__main__":
    data = UtilParams()
    duster = DUST3R(data)
    duster.run()
    duster.save(data)
