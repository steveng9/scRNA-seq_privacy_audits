#!/bin/bash

sizes=(50)
#sizes=(2 5 10 20 50)
#sizes=(100 200)
datasets=(aida aida aida aida aida)

for size in "${sizes[@]}"; do
    for dataset in "${datasets[@]}"; do

        yaml="/home/golobs/data/${dataset}/exp_cfgs/${size}d_000.yaml"

        printf "\n\n\n\n\n\n\n\n\n\n\n"
        echo "Running with: $yaml"
        python ../simplified_quality_evaluation.py T "$yaml" P

        echo "-----------------------------------"
    done
done
