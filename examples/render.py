import sys
sys.path.append('../submodules/gaussian-splatting')

from os import makedirs
from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix
import torchvision
import subprocess
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm.auto import tqdm
from utils.image_utils import psnr
import lovely_tensors as lt

lt.monkey_patch()

from dataclasses import dataclass, field

@dataclass
class ModelParams:
    def __init__(self):
        self.sh_degree: int = 3
        self.source_path: str = "../data/scenes/turtle"
        self.model_path: str = self.get_output_folder()
        self.images: str = "images"
        self.resolution: int = -1
        self.white_background: bool = True
        self.data_device: str = "cuda"
        self.eval: bool = False

    def post_init(self):
        self.source_path = os.path.abspath(self.source_path)

    def get_output_folder(self):
        all_subdirs = [os.path.abspath(os.path.join('../output', d)) for d in os.listdir(os.path.abspath('../output/')) 
                            if os.path.isdir(os.path.abspath(os.path.abspath(os.path.join('../output', d))))]
        return max(all_subdirs, key=os.path.getmtime)

@dataclass
class PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

@dataclass
class OptimizationParams:
    iterations = 7001 #30_000
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01
    lambda_dssim = 0.2
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002
    random_background = False


@torch.no_grad()
def render_path(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_resize_method='crop'):
    """
    render_resize_method: crop, pad
    """
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    iteration = scene.loaded_iter

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    model_path = dataset.model_path
    name = "render"

    views = scene.getRenderCameras()

    # print(len(views))
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if render_resize_method == 'crop':
            image_size = 512
        elif render_resize_method == 'pad':
            image_size = max(view.image_width, view.image_height)
        else:
            raise NotImplementedError
        view.original_image = torch.zeros((3, image_size, image_size), device=view.original_image.device)
        focal_length_x = fov2focal(view.FoVx, view.image_width)
        focal_length_y = fov2focal(view.FoVy, view.image_height)
        view.image_width = image_size
        view.image_height = image_size
        view.FoVx = focal2fov(focal_length_x, image_size)
        view.FoVy = focal2fov(focal_length_y, image_size)
        view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda().float()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)

        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    # Use ffmpeg to output video
    renders_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4")
    # Use ffmpeg to output video
    subprocess.run(["ffmpeg", "-y", 
                "-framerate", "24",
                "-i", os.path.join(render_path, "%05d.png"), 
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", 
                "-pix_fmt", "yuv420p",
                "-crf", "23", 
                # "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
                renders_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

dataset = ModelParams()
opt = OptimizationParams()
pipe = PipelineParams()
render_path(dataset, 7000, pipe, render_resize_method='crop')

