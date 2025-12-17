#!/usr/bin/env bash
set -euo pipefail

TEMPLATE="../scRNA-seq_privacy_audits/example_cfg_file_for_mamamia_experiments.yaml"

Ds=(ok cg aida)
Xs=(2 5 10 20 50 100 200)
# Generate C = 000 .. 111
Cs=()
for i in {0..1}; do
  for j in {0..1}; do
    for k in {0..1}; do
      Cs+=("${i}${j}${k}")
    done
  done
done

for D in "${Ds[@]}"; do
  mkdir -p "$D"
  for X in "${Xs[@]}"; do
    for C in "${Cs[@]}"; do
      outfile="${D}/${X}d_${C}.yaml"
      cp "$TEMPLATE" "$outfile"
      # Decode C bits
      wb_bit="${C:0:1}"
      hvg_bit="${C:1:1}"
      aux_bit="${C:2:1}"
      [[ "$wb_bit" == "0" ]] && white_box=true || white_box=false
      [[ "$hvg_bit" == "0" ]] && use_wb_hvgs=true || use_wb_hvgs=false
      [[ "$aux_bit" == "0" ]] && use_aux=true || use_aux=false
      # Apply replacements
      sed -i '' \
        -e 's/^  mahalanobis: false$/  mahalanobis: true/' \
        -e "s/^dataset_name: ok$/dataset_name: ${D}/" \
        -e 's/^min_aux_donors: 10$/min_aux_donors: 5/' \
        -e "s/^  num_donors: 100$/  num_donors: ${X}/" \
        -e "s/^  white_box: false$/  white_box: ${white_box}/" \
        -e "s/^  use_wb_hvgs: true$/  use_wb_hvgs: ${use_wb_hvgs}/" \
        -e "s/^  use_aux: true$/  use_aux: ${use_aux}/" \
        -e '/determine_new_hvgs: false/d' \
        "$outfile"
    done
  done
done

echo "Done."
