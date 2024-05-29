# python point_cloud_generate.py --root-path real_data/scene_0001/ \
    # --base-path real_data/camera/0229/
for file in $(ls real_data);do
    if [[ $file == scene* ]];then
        # echo $file
        python point_cloud_generate.py --root-path real_data/$file/ \
            --base-path real_data/camera/0123/
    fi
done