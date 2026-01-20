#!/bin/bash

#sizes=(200 100 50 20 10 5 2)
sizes=(2 5 10 20 50 100 200)
#sizes=(100 200)
variants=(001 101 001 101)
datasets=(ok)

for size in "${sizes[@]}"; do
    for variant in "${variants[@]}"; do
        for dataset in "${datasets[@]}"; do

            #if [[ "$size" -eq 5 && "$dataset" == "ok" && "$variant" == "wb" ]]; then
            #    continue
            #fi

            yaml="/home/golobs/data/${dataset}/exp_cfgs/${size}d_${variant}.yaml"

            printf "\n\n\n\n\n\n\n\n\n\n\n"
            echo "Running with: $yaml"
            python ../mia_experiments_main.py T "$yaml" P

            echo "-----------------------------------"
        done
    done
done
