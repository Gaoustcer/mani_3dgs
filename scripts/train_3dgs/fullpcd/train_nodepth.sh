port=$RANDOM
DEVICEID=0
DEVICENUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ratios=(0.01 0.02 0.04 0.1 0.2 0.3 0.5)
file=scene_0001
CUDA_VISIBLE_DEVICES=$DEVICEID python train.py -s ./real_data/$file\
    -m ./logs/fullpcd/nodepth/$file\
    --port $port\
    --test_iterations 1 3000 7000 10000 15000 20000 25000 30000\
    --pcd-path pcd/point_cloud.pcd &
((DEVICEID=$DEVICEID+1))
((DEVICEID=$DEVICEID%$DEVICENUM))
((port=$port+1000))
CUDA_VISIBLE_DEVICES=$DEVICEID python train_depth.py -s ./real_data/$file \
    -m logs/fullpcd/depth_nomask/$file \
    --port $port --pcd-path pcd/point_cloud.pcd \
    --test_iterations 1 1000 7000 12000 30000 &

((DEVICEID=$DEVICEID+1))
((DEVICEID=$DEVICEID%$DEVICENUM))
((port=$port+1000))
CUDA_VISIBLE_DEVICES=$DEVICEID python train_depth.py \
    -s ./real_data/$file \
    -m logs/fullpcd/depth_withmask/$file \
    --port $port --pcd-path point_cloud.pcd \
    --load-mask --test_iterations 1 1000 7000 12000 30000 &
wait
# for file in $(ls ./real_data);
# do

#     if [[ $file == scene_0001* ]];then
#         for ratio in ${ratios[*]};do
#             ((port=$port+1000)) 
#             CUDA_VISIBLE_DEVICES=$DEVICEID python train.py -s ./real_data/$file\
#                 -m ./logs/nodepth/$file/$ratio\
#                 --port $port\
#                 --test_iterations 1 3000 7000 10000 15000 20000 25000 30000\
#                 --pcd-path point_cloud_$ratio.pcd &
#             ((DEVICEID=$DEVICEID+1))
#             ((DEVICEID=$DEVICEID%$DEVICENUM))
#         done
#         # echo $file
#     fi
# done
# wait
# echo "training finish"
# python train.py -s ./real_data/scene_0001 --port 12375


