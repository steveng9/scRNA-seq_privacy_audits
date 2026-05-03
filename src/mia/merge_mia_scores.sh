#!/bin/bash

sizes=(2 5 10 20 50 100 200)
datasets=(ok cg aida)
trials=(1 2 3 4 5 6)

for size in "${sizes[@]}"; do
    for dataset in "${datasets[@]}"; do
        for trial_num in "${trials[@]}"; do

            yaml="/home/golobs/data/${dataset}/exp_cfgs/${size}d_000.yaml"

            printf "\n\n\n\n\n\n\n\n\n\n\n"
            echo "Running with: $yaml"
            python rescore_mias.py T "$yaml" "$trial_num"

            echo "-----------------------------------"
        done
    done
done
