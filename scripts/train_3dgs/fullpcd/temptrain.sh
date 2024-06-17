file=scene_0001
port=$RANDOM
CUDA_VISIBLE_DEVICES=3 python train_depth.py \
    -s ./real_data/$file \
    -m logs/fullpcd/depth_withmask/$file \
    --port $port --pcd-path pcd/point_cloud.pcd \
    --load-mask --test_iterations 1 1000 7000 12000 30000 