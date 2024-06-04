import os
from pathlib import Path
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
    debug: bool = False
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False

@dataclass
class OptimizationParams:
    iterations = 7000 #30_000
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

@dataclass
class TrainingArgs:
    ip: str = "0.0.0.0"
    port: int = 6007
    debug_from: int = -1
    detect_anomaly: bool = False
    test_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    quiet: bool = False
    checkpoint_iterations: list[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    start_checkpoint: str = None

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

def get_params():
    return ModelParams(), PipelineParams(), OptimizationParams(), TrainingArgs(), UtilParams()