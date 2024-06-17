names=(imp_opa sep_views sh_degree normal)
for name in ${names[*]};do
    for file in $(ls scripts/train_3dgs/$name/);do
        bash ./scripts/train_3dgs/$name/$file &
    done
done
wait