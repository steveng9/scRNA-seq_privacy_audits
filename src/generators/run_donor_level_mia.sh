#!/bin/bash

sizes=(5 10 20 50)
variants=(wb bb)
datasets=(ok cg aida)

for size in "${sizes[@]}"; do
    for variant in "${variants[@]}"; do
        for dataset in "${datasets[@]}"; do

            yaml="/home/golobs/data/${dataset}/${size}d_${variant}.yaml"

            printf "\n\n\n\n\n\n\n\n"
            echo "Running with: $yaml"
            python ../mia_experiments_main.py T "$yaml" P

            echo "-----------------------------------"
        done
    done
done
