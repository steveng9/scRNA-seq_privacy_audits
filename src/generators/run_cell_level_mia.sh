#!/bin/bash

sizes=(1 3 10 30)
variants=(wb bb)
datasets=(ok cg aida)

for size in "${sizes[@]}"; do
    for variant in "${variants[@]}"; do
        for dataset in "${datasets[@]}"; do

            yaml="/home/golobs/data/${dataset}/${size}k_${variant}.yaml"

            echo "\n\n\n\n\n\n\n\n\n\nRunning with: $yaml"
            python ../mia_experiments_main.py T "$yaml" P

            echo "-----------------------------------"
        done
    done
done
