import os
import sys
import uuid
import torch
from random import randint
from tqdm.auto import tqdm
import lovely_tensors as lt
from dataclasses import dataclass, field

# Import submodules
sys.path.append('../submodules/normalized_splatting')

from utils.image_utils import psnr                                          #
from scene import Scene, GaussianModel                                      #
from utils.loss_utils import l1_loss, ssim                                  # Gaussian Splatting
from utils.general_utils import safe_state                                  #
from gaussian_renderer import render, network_gui                           #
from train import training                                                  #

lt.monkey_patch()

@dataclass
class ModelParams:
    eval: bool = False
    sh_degree: int = 3
    resolution: int = -1
    model_path: str = "../"
    images: str = "images"
    depths = "depths"
    data_device: str = "cuda"
    white_background: bool = True
    source_path: str = "../data/scenes/turtle"
    resolution: int = -1
    filter_radius_3d: float = 0.0
    opacity_scale: float = 1.0
    normalize_gaussians: bool = False
    filter_radius_2d: float = 0.3
    filter_by_loss: bool = False

    def post_init(self):
        self.source_path = os.path.abspath(self.source_path)

@dataclass
class PipelineParams:
    debug: bool = False
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False

@dataclass
class OptimizationParams:
    iterations = 7001 #30_000
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    opacity_scale_lr = 0.005
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01
    lambda_dssim = 0.2
    lambda_depth = 1.0
    scaling_max_steps = 10_000
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002
    random_background = False
    min_opacity = 0.005
    lambda_filter = 1e-3
    five_sigma_radius_limit_m = None
    lambda_sigma_limit = 1
    min_eigvalue = None
    prune_min_eigv = False

@dataclass
class TrainingArgs:
    ip: str = "127.0.0.1"
    port: int = 6009
    debug_from: int = -1
    detect_anomaly: bool = False
    test_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    quiet: bool = False
    checkpoint_iterations: list[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    start_checkpoint: str = None
    pose_trans_noise: float = 0.0
    DS: bool = True
    BA: bool = False
    localization: bool = False
    single_frame_id: int = None
    
dataset = ModelParams()
opt = OptimizationParams()
pipe = PipelineParams()
args = TrainingArgs()

if __name__ == "__main__":
    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    print("\nTraining complete.")