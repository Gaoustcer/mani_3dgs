port=1000
DEVICEID=0
for file in $(ls ./real_data);
do

    if [[ $file == scene* ]];then
        ((port=$port+1000))
        export OAR_JOB_ID=singleview/$file
        # echo $port
        echo $DEVICEID
        echo $OAR_JOB_ID
        CUDA_VISIBLE_DEVICES=$DEVICEID python train.py -s ./real_data/$file --port $port --pcd-path pcds/images_0001.pcd &
        ((DEVICEID=$DEVICEID+1))
        ((DEVICEID=$DEVICEID%2))
        # echo $file
    fi
done
wait
echo "training finish"
# python train.py -s ./real_data/scene_0001 --port 12375


