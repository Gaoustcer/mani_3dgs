port=10000
DEVICEID=0
DEVICENUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ratios=(0.01 0.04 0.1)
for file in $(ls ./real_data);
do
    if [[ $file == scene_0001* ]];then
        # ((port=$port+5000))
        # CUDA_VISIBLE_DEVICES=$DEVICEID python train_depth.py -s ./real_data/$file -m logs/depth_mask_novel_view/${file}_nodownsample --port $port --pcd-path point_cloud.pcd --load-mask --test_iterations 1 1000 7000 12000 30000  &
        # ((DEVICEID=$DEVICEID+1))
        # ((DEVICEID=$DEVICEID%$DEVICENUM))
        for ratio in ${ratios[*]};do
            ((port=$port+1000))
            CUDA_VISIBLE_DEVICES=$DEVICEID python train_depth.py -s ./real_data/$file -m logs/shdegree0/depth_withmask/$file_$ratio --port $port --pcd-path point_cloud_$ratio.pcd --load-mask --test_iterations 1 1000 7000 12000 30000 --separate-cameras --sh_degree 0&
            ((DEVICEID=$DEVICEID+1))
            ((DEVICEID=$DEVICEID%$DEVICENUM))
        done
        # echo $file
    fi
done
wait
echo "training finish"
# python train.py -s ./real_data/scene_0001 --port 12375


