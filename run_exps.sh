#!/bin/bash
# Notification 
mail=julius.kuehn@igd.fraunhofer.de
# run this with sh on an amperecontrol
# Hardware 
node=ampere2
cpus=220
ram=900G
gpus=8
qos="normal" # idle, normal, priority
###################################
# Model options: tiny, small, medium, large, xlarge
model='tiny'
task='train' # train, eval, resume
# Slurm reports land here
slurm_out="$HOME/lw-detr6d/slurm_${task}_${model}"
# Training config
batch_size_per_gpu=4
lr=1e-4
lr_encoder=1.5e-4
lr_pose_heads=$lr
lr_drop=55
# Detection Loss config (standard values from LWDETR paper)
coef_clas=1.0
coef_bbox=5.0
coef_giou=2.0
# Pose Loss config
coef_adds=0.0
coef_kpt=2.5
coef_rot=1.0
coef_trans_xy=0.0
coef_trans_z=15.0
rot_rep=6d
warm_up_epochs=1
cad_models=/models/
matcher_type=6d_rot_trans # hungarian, 6d, 6d_rot_trans
reduce_det_loss_epochs=6767
set_cost_class=2.0 # default is 2.0
set_cost_bbox=5.0 # default is 5.0
set_cost_giou=1.0 # default is 2.0, set to 1.0 when 6d_rot_trans is active!
set_cost_kpt=5.0 # only active when matcher_type is 6d or 6d_rot_trans
set_cost_rot=0.0 # only active when matcher_type is 6d_rot_trans
set_cost_trans=5.0 # only active when matcher_type is 6d_rot_trans
matcher_symmetry_stride=30
cad_tag=$(basename "${cad_models%/}")
job_name="${task}_${model}_${rot_rep}_z${coef_trans_z}_r${coef_rot}_adds${coef_adds}_wu${warm_up_epochs}_m${matcher_type}_mrot${set_cost_rot}_ms${matcher_symmetry_stride}_rd${reduce_det_loss_epochs}_std"
# Check if path exists (file or directory)
if [ -e $slurm_out ]; then
    echo "Path $slurm_out exists"
else 
    mkdir -p $slurm_out
fi
# To chosse specific Amperenode --nodelist=ampere[1-5]
sbatch --job-name=$job_name \
        --gpus-per-node=$gpus \
        --cpus-per-task=$cpus \
        --mem=$ram \
        --output=$slurm_out/%j-%x.out \
        --error=$slurm_out/%j-%x.err \
        --export=BATCH_SIZE=$batch_size_per_gpu,LR=$lr,LR_ENCODER=$lr_encoder,LR_POSE_HEADS=$lr_pose_heads,LR_DROP=$lr_drop,COEF_ROT=$coef_rot,COEF_KPT=$coef_kpt,COEF_TRANS_XY=$coef_trans_xy,COEF_TRANS_Z=$coef_trans_z,COEF_ADDS=$coef_adds,COEF_CLAS=$coef_clas,COEF_BBOX=$coef_bbox,COEF_GIOU=$coef_giou,ROT_REP=$rot_rep,WARM_UP_EPOCHS=$warm_up_epochs,CAD_MODELS=$cad_models,MATCHER_TYPE=$matcher_type,REDUCE_DET_LOSS_EPOCHS=$reduce_det_loss_epochs,SET_COST_CLASS=$set_cost_class,SET_COST_BBOX=$set_cost_bbox,SET_COST_GIOU=$set_cost_giou,SET_COST_ROT=$set_cost_rot,SET_COST_TRANS=$set_cost_trans,SET_COST_KPT=$set_cost_kpt,MATCHER_SYMMETRY_STRIDE=$matcher_symmetry_stride,MODEL=$model,TASK=$task,SLURM_OUT=$slurm_out,JOB_NAME=$job_name \
        --mail-user=$mail \
        --mail-type=BEGIN,END,FAIL,PREEMPT,REQUEUE,ALL \
        --qos=$qos \
        --nodelist=$node \
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