ratios=(0.001 0.01 0.05 0.1 0.3)
# DEVICE=0
port=12768
DEVICEID=0
DEVICENUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
SwitchDevice(){
    ((DEVICEID=$DEVICEID+1))
    ((DEVICEID=$DEVICEID%$DEVICENUM))
    ((port=$port+4000))
    ((port=$port%50000))
    
    echo "switch device"
}
for ratio in ${ratios[*]};do
    for i in {0..9};do
        # SwitchDevice
        CUDA_VISIBLE_DEVICES=$DEVICEID python train_depth.py -s ./sim_data/close_door/episode0 -m logs/sim/episode$i/$ratio --port $port --pcd-path point_cloud_$ratio.pcd &
        SwitchDevice
    done
    wait
done
for i in {0..9};do
    CUDA_VISIBLE_DEVICES=$DEVICEID python train_depth.py -s ./sim_data/close_door/episode0 -m logs/sim/episode$i/full --port $port --pcd-path point_cloud.pcd  &
    SwitchDevice
done
wait