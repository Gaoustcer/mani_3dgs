# Scene Synthesis Pipeline(Environmen Setup,Data Preprocess and training of 3D Gaussian Primitives)
## Environment Setup
This project is forked from [original 3dgs](https://github.com/graphdeco-inria/gaussian-splatting) with pre-compiled binary package `diff_gaussian_rasterization` and `simple_knn'. We also implement the code to convert multiple RGBD images from calibrated cameras into point_cloud. To config the project, you can create the environment from `environment.yml`
## Data Prepare
A standard data format is
```txt
<root-to-dataset>
    - boundray-mask
        - image_0.npy
        - image_1.npy
        ...
    - images
        - image_0.png
        - image_1.png
        ...
    - depths
        - image_0.npy
        - image_1.npy
        ...
    - normals
        - image_0.npy
        - image_1.npy
        ...
```
In our model, we assume the multi-view image is captured by a camera fixed on a hand which can rotate and translate in 3D space. We assume the pose of camera from hand view is fixed and we could calculate the c2w matrix with hand2base matrix. In this case, we also provide a `hand2base.txt` file to store hand2base rotation and translation, which is store in `<path_hand2base>/hand2base.txt`
To generate pcd, simply run
```python
python point_cloud_generate.py --root-path <root-to-dataset> --base-path <path_hand2base>
```
This will generate two subdir under \<root-to-dataset\>, `pcds` store the point_cloud of each view and `mask_image` store mask area for each image. This will also generate `point_cloud.pcd` under this path which integrates point-cloud under different views.
### downsample 3dgs
We do not use point cloud alignment technique to match the point cloud. Therefore, the merged point cloud can be quite large(about 3 millions), which will increase the time amd memory consumption in 3dgs training. To overcome this issue, we design a down sampling procedure which randomly several points from the overall point clouds. Experiment results reveal that the 3dgs reconstruction quality can be pretty well even under quite low sampling rate(1%). Usage of `down_sampling_pcd.py`
```shell
python down_sampling_pcd.py --pcd-root-path <path-to-store-point_cloud.pcd> --down-sample-ratio <float ratio between 0 and 1>
```
## Training 3DGS
```shell
python train.py -s <root-to-dataset>
``` 
This will use the integrated point-cloud. If you want to use a single-view point cloud, just `export OAR_JOB_ID=<relative-path-of-single-view-pcd-from-root-to-dataset>`
To visualize the result, you can install [SIBR](https://sibr.gitlabpages.inria.fr/) and follow the instructions to run it.
We also add depth as a supervise signal to help the 3dgs converge.Experiment result indicates depth will improve the converage speed of loss and improve the reconstruction result as well. Here is an example
```shell
    python train_depth.py -s datasets/scene_0001 -m logs/scene_0001 --pcd-path point_cloud_0.1.pcd --port 12357
```
`pcd_path` actually appoint the pcd file you want to load as 3dgs for initlization, which can be any down-sampling pcd file you obtain in the former stage