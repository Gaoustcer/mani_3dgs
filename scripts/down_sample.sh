ratiolist=(0.01 0.02 0.04 0.1 0.2 0.3 0.5)
for file in $(ls real_data);
do
    if [[ $file == scene* ]];then
        for ratio in ${ratiolist[*]};do
            python down_sampling_pcd.py --pcd-root-path real_data/$file \
                --down-sample-ratio $ratio &
        done
    fi
done
wait