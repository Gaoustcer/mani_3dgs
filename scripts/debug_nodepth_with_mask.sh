python -m pdb train.py -s ./real_data/scene_0001\
    -m logs/depth_mask/debug \
    --port 40000 \
    --pcd-path point_cloud.pcd \
    --test_iterations 1 1000 7000 5000
