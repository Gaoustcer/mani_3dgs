#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from copy import deepcopy
import numpy as np

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from torchvision.utils import save_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import Camera
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def init_render(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # datasets {'sh_degree': 3, 
    # 'source_path': '/lustre/S/gaohaihan/embodiedai/gaussian-splatting/datasets/bicycle', 
    # 'model_path': '', 'images': 'images', 
    # 'resolution': -1, 'white_background': False, 
    # 'data_device': 'cuda', 'eval': False}
    # opt training opt

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    # gaussians active_sh_degree 0
    # max_sh_degree 3
    # _xyz [N,3] N number of gaussians
    # _features_dc [N,1,3]
    # _features_reset [N,15,3]
    scene = Scene(dataset, gaussians) # init gaussians with point cloud
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    first_iter += 1
    viewpoint_cameras = scene.getTrainCameras().copy()
    for camera in tqdm(viewpoint_cameras):
        gaussians.optimizer.zero_grad()
        render_pkg = render(camera,gaussians,pipe,torch.tensor([1,1,1]).cuda().float())
        render_image = render_pkg['render']
        gt_image = camera.original_image.cuda()
        loss = l1_loss(render_image,gt_image)
        loss.backward()
        xyz_grad = gaussians._xyz.grad
        idx = (torch.norm(xyz_grad,dim = -1) != 0)
        points = gaussians._xyz[idx]
        colors = gaussians._features_dc[idx]
        points = points.squeeze().detach().cpu().numpy()
        colors = colors.squeeze().detach().cpu().numpy()
        import open3d as o3d

        pcd = o3d.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        from utils.sh_utils import SH2RGB
        # print("colors",colors.shape)
        colors = SH2RGB(colors)
        print("colors",colors.shape)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        image_name = camera.image_name.split(".")[0]
        o3d.io.write_point_cloud(os.path.join(args.model_path,"{}.pcd".format(image_name)),pcd)
        save_image(render_image,os.path.join(args.model_path,"{}.png".format(image_name)))

    # exit()
    # training_report(tb_writer, first_iter, [1], scene, render, (pipe, background))
    # from tqdm import tqdm
   

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            args.model_path = os.path.join("./output/",unique_str)
        else:
            unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, testing_iterations, scene : Scene, renderFunc, renderArgs):
    # if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(16)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    novel_R = deepcopy(viewpoint.R)
                    novel_T = deepcopy(viewpoint.T)
                    novel_R = np.random.normal(novel_R,0.1)
                    novel_T = np.random.normal(novel_T,0.1)
                    novel_view = Camera(
                        colmap_id=viewpoint.colmap_id,
                        R = novel_R,
                        T = novel_T,
                        FoVx = viewpoint.FoVx,
                        FoVy = viewpoint.FoVy,
                        image = viewpoint.original_image,
                        gt_alpha_mask=None,
                        image_name = viewpoint.image_name,
                        uid = viewpoint.uid
                    )
                    distort_image = torch.clamp(renderFunc(novel_view, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/novel_view".format(viewpoint.image_name), distort_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # l1_test += l1_loss(image, gt_image).mean().double()
                    # psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--pcd-path",type = str,default = "point_cloud.pcd")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    init_render(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
