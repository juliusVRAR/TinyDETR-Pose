#!/bin/bash
# Notification
mail=julius.kuehn@igd.fraunhofer.de
# run this with sh on an amperecontrol
# Hardware 
gpus=4
cpus=120
ram=500G
# Model config 
model='tiny'
task='train'
job_name="${task}_${model}_r_${coef_rot}_test"
# Slurm reports land here
slurm_out="$HOME/lw-detr6d/slurm_${task}_${model}"
# Detection Loss config
coef_clas=2.0
coef_bbox=2.0
coef_giou=1.0
# Pose Loss config
coef_kpt=2.0
coef_trans_xy=20.0
coef_trans_z=2.0 
coef_adds=15.0
coef_rot=2.0

# Check if path exists (file or directory)
if [ -e $slurm_out ]; then
    echo "Path $slurm_out exists"
else 
    mkdir -p $slurm_out
fi

sbatch --job-name=$job_name \
        --gpus-per-node=$gpus \
        --cpus-per-task=$cpus \
        --mem=$ram \
        --output=$slurm_out/%j-%x.out \
        --error=$slurm_out/%j-%x.err \
        --export=COEF_ROT=$coef_rot,COEF_KPT=$coef_kpt,COEF_TRANS_XY=$coef_trans_xy,COEF_TRANS_Z=$coef_trans_z,COEF_ADDS=$coef_adds,COEF_CLAS=$coef_clas,COEF_BBOX=$coef_bbox,COEF_GIOU=$coef_giou,MODEL=$model,TASK=$task,SLURM_OUT=$slurm_out,JOB_NAME=$job_name \
        --mail-user=$mail \
        --mail-type=BEGIN,END,FAIL\
        run_on_cluster.sh

###################################
# Example for multiple runs
# Use this and a for loop to deploy multiple configs
# coef_rot_values="1 2 2.25"
# for coef_rot in $coef_rot_values; do  
#     job_name="${task}_${model}_r_${coef_rot}"
#     sbatch --job-name=$job_name \
#            --gpus-per-node=$gpus \
            # --cpus-per-task=$cpus \
            # --mem=$ram \
            # --output=$slurm_out/%j-%x.out \
            # --error=$slurm_out/%j-%x.err \
            # --export=COEF_ROT=$coef_rot,COEF_KPT=$coef_kpt,COEF_TRANS_XY=$coef_trans_xy,COEF_TRANS_Z=$coef_trans_z,COEF_ADDS=$coef_adds,COEF_CLAS=$coef_clas,COEF_BBOX=$coef_bbox,COEF_GIOU=$coef_giou,MODEL=$model,TASK=$task \
            # run_on_cluster.sh
#     cp "$0" "$slurm_out"
#     cp "$0" "$slurm_out/config_${job_name}.sh"
# done
####################################