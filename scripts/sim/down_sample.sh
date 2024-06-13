ratios=(0.001 0.01 0.05 0.1 0.3)
for ratio in ${ratios[*]};
do
    for i in {0..9};do
        python down_sampling_pcd.py --pcd-root-path ./sim_data/close_door/episode$i/ --down-sample-ratio $ratio &
    done
done
wait