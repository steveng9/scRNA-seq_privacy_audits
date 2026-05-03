#!/bin/bash

sizes=(2 5 10 20 50 100 200)
#sizes=(10 20 50)
datasets=(ok cg aida)

for size in "${sizes[@]}"; do
    for dataset in "${datasets[@]}"; do

        #if [[ "$size" -eq 5 && "$dataset" == "ok" ]]; then
        #    continue
        #fi

        yaml="/home/golobs/data/${dataset}/exp_cfgs/${size}d_000.yaml"

        printf "\n\n\n\n\n\n\n\n\n\n\n"
        echo "Running with: $yaml"
        python ../baseline_mias.py T "$yaml"

        echo "-----------------------------------"
    done
done
