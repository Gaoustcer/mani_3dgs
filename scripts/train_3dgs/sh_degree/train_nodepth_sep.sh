port=$RANDOM
((port=$port%10000))
DEVICEID=0
DEVICENUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ratios=(0.01 0.02 0.04 0.1 0.2 0.3 0.5)
for file in $(ls ./real_data);
do

    if [[ $file == scene_0001* ]];then
        for ratio in ${ratios[*]};do
            ((port=$port+1000)) 
            ((port=$port%30000))
            CUDA_VISIBLE_DEVICES=$DEVICEID python train.py -s ./real_data/$file\
                -m ./logs/shdegree0/nodepth/$file/$ratio\
                --port $port\
                --test_iterations 1 3000 7000 10000 15000 20000 25000 30000\
                --pcd-path point_cloud_$ratio.pcd \
                --separate-cameras --sh_degree 0 &
            ((DEVICEID=$DEVICEID+1))
            ((DEVICEID=$DEVICEID%$DEVICENUM))
        done
        # echo $file
    fi
done
wait
echo "training finish"
# python train.py -s ./real_data/scene_0001 --port 12375


