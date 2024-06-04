import sys
from params import get_params
from run_dust3r import DUST3R
from render import render_path

# Import submodules
sys.path.append('../submodules/dust3r')
sys.path.append('../submodules/teaser')
sys.path.append('../submodules/gaussian-splatting')

from train import training

# Compute parameters
model, pipe, opt, args, util = get_params()

# Run DUST3R and save results
duster = DUST3R(util)
duster.save(util)

# Train 3DGS and save results
training(model, opt, pipe, args.test_iterations, 
		  args.save_iterations, args.checkpoint_iterations, 
		  args.start_checkpoint, args.debug_from)

# Render 3DGS results
render_path(model, opt.iterations, pipe, render_resize_method='crop')

