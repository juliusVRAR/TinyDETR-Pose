#!/bin/bash
# run this with sh on an amperecontrol
# Slurm reports land here
slurm_out="$HOME/lw-detr6d/slurm_${task}_${model}"
# Model config 
model='tiny'
task='train'
# Loss config
coef_kpt=10.0
coef_trans_xy=1.0
coef_trans_z=1.0 
coef_adds=1.0
coef_rot_values="1.0 1.5 2 2.25 2.5 3.5 5"
# Check if path exists (file or directory)
if [ -e $slurm_out ]; then
    echo "Path $slurm_out exists"
else 
    mkdir -p $slurm_out
fi

for coef_rot in $coef_rot_values; do
    job_name="${task}_${model}_r_${coef_rot}"
    sbatch --job-name=$job_name \
            --gpus-per-node=1 \
            --cpus-per-task=24 \
            --mem=64G \
            --output=$slurm_out/%j-%x.out \
            --error=$slurm_out/%j-%x.err \
            --export=COEF_ROT=$coef_rot,COEF_KPT=$coef_kpt,COEF_TRANS_XY=$coef_trans_xy,COEF_TRANS_Z=$coef_trans_z,COEF_ADDS=$coef_adds,MODEL=$model,TASK=$task \
            run_on_cluster.sh
done