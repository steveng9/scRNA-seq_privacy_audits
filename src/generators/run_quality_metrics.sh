#!/bin/bash

sizes=(100)
datasets=(ok)

for size in "${sizes[@]}"; do
    for dataset in "${datasets[@]}"; do

        realdata="/home/golobs/data/${dataset}/${size}d/datasets/train.h5ad"
        synthdata="/home/golobs/data/${dataset}/${size}d/datasets/synthetic.h5ad"
        outfile="/home/golobs/data/${dataset}/${size}d/results/quality.csv"
        printf "\n\n\n\n\n\n\n\n"
        echo "Running with: $realdata"
        python ../simplified_quality_evaluation.py --real "$realdata" --synthetic "$synthdata" --out "$outfile"

        echo "-----------------------------------"
    done
done
