import sys
sys.path.append('../submodules/normalized_splatting')

import os
from tqdm.auto import tqdm
from dataclasses import dataclass

from render import render_sets
from utils.general_utils import safe_state


@dataclass
class ModelParams:
    def __init__(self):
        self.eval: bool = False
        self.sh_degree: int = 3
        self.resolution: int = -1
        self.model_path: str = self.get_output_folder()
        self.images: str = "images"
        self.depths = "depths"
        self.data_device: str = "cuda"
        self.white_background: bool = True
        self.source_path: str = "../data/scenes/turtle"
        self.resolution: int = -1
        self.filter_radius_3d: float = 0.0
        self.opacity_scale: float = 1.0
        self.normalize_gaussians: bool = False
        self.filter_radius_2d: float = 0.3
        self.filter_by_loss: bool = False


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

class Arguments:
    iteration: int = 7000
    skip_train: bool = False
    skip_test: bool = True
    quiet: bool = False

if __name__ == "__main__":
    args = Arguments()
    model = ModelParams()
    pipeline = PipelineParams()
    print("Rendering " + model.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model, args.iteration, pipeline, args.skip_train, args.skip_test)