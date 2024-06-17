python activate_pcd.py -s real_data/scene_0001 -m logs/init_render/full --port 12375 --pcd-path point_cloud.pcd --white_background 
# exit
port=12375
# ratios=(0.001 0.01 0.1)

# for ratio in ${ratios[*]};do
#     ((port=$port+2000))
#     python activate_pcd.py -s real_data/scene_0001 -m logs/init_render/$ratio --pcd-path point_cloud_$ratio.pcd --port $port --white_background &
# done
# wait