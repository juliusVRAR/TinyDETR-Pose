#!/bin/bash
#SBATCH --job-name=slurm_test
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=SLURM_OUT=not_called_from_exp
#SBATCH --export=JOB_NAME=not_called_from_exp
# Task options 'train' or 'eval'
#SBATCH --export=TASK=train
# Model options: tiny, small, medium, large, xlarge
#SBATCH --export=MODEL=tiny


# Copy config for reproducability
REPORT_OUT="${SLURM_OUT}/${SLURM_JOB_ID}_${JOB_NAME}"
mkdir -p $REPORT_OUT


gio_mount_nas()
{
    mkdir -p ~/dbus
    XDG_RUNTIME_DIR=~/dbus
    export DBUS_SESSION_BUS_ADDRESS=`dbus-daemon --fork --print-address --session`
    gio mount smb://pc3163/nobackup < ~/smbcreds
}

DSNAME=lw_detr6d_data

DATAPATH=/opt/cache/$USER
# Check if path exists (file or directory)
if [ -e $DATAPATH ]; then
    echo "Path $DATAPATH exists"
else 
    mkdir -p $DATAPATH
fi

DATA=$DATAPATH/$DSNAME

PATH_TO_BOP=$DATA/bop_datasets
PATH_TO_WEIGHTS=$DATA/weights/lw-detr
WORKSPACE=$PWD

# Download dataset if not present
if test ! -d $DATA ; then
    gio_mount_nas
    gio copy --progress smb://pc3163/nobackup/cache/jkuehn/$DSNAME.zip $DATAPATH
    cd $DATAPATH
    unzip -q $DSNAME.zip 
fi

# Configure model
if [ $MODEL == 'tiny' ]; then
    echo "Using tiny model settings"
    IDX='0'
elif [ $MODEL == 'small' ]; then
    echo "Using small model settings"
    IDX='1'
elif [ $MODEL == 'medium' ]; then
    echo "Using base model settings"
    IDX='2'
elif [ $MODEL == 'large' ]; then
    echo "Using base model settings"
    IDX='3'
elif [ $MODEL == 'xlarge' ]; then
    echo "Using base model settings"
    IDX='4'
else
    echo "Unknown model $MODEL, exiting"
    exit 1
fi
MODEL="${IDX}_${MODEL}"

# Configure task
if [ $TASK == 'train' ]; then
    echo "Train task selected"
    IDX='0'
elif [ $TASK == 'eval' ]; then
    echo "Eval task selected"
    IDX='1'
else
    echo "Unknown task $TASK, exiting"
    exit 1
fi
TASK="${IDX}_${TASK}"

# Copy exp and train config for reproducability
cp "run_exps.sh" "$REPORT_OUT/${SLURM_JOB_ID}_${JOB_NAME}_exp.sh" 
cp "scripts/pose/${MODEL}/${TASK}.sh" "$REPORT_OUT/${SLURM_JOB_ID}_${JOB_NAME}_config.sh" 


IMAGE_NAME=lw-detr6d
## MAIN
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
rootless-docker run --gpus all --shm-size=256g \
    -v $WORKSPACE:/workspace/LWDETR\
    -v $PATH_TO_BOP:/workspace/LWDETR/data/datasets/bop \
    -v $PATH_TO_WEIGHTS:/workspace/LWDETR/data/weights \
    pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME\
    bash -c "python /workspace/LWDETR/models/ops/setup.py build install && \
                /workspace/LWDETR/scripts/pose/$MODEL/$TASK.sh $NUM_GPUS $COEF_KPT $COEF_TRANS_XY $COEF_TRANS_Z $COEF_ROT $COEF_ADDS $COEF_CLAS $COEF_BBOX $COEF_GIOU $SLURM_JOB_ID $JOB_NAME $ROT_REP $WARM_UP_EPOCHS"
                                    
        
## OUTPUT
# zip & copy the data back
# cd $OUTPATH/../
# zip -q -r $DSNAME.zip $DSNAME
# if test ! -z "$SLURM_JOB_ID"; then
#     XDG_RUNTIME_DIR=~/dbus
#     export DBUS_SESSION_BUS_ADDRESS=`dbus-daemon --fork --print-address --session`
#     gio mount smb://pc3163/nobackup < ~/smbcreds
#     gio copy --progress $DSNAME.zip smb://pc3163/nobackup/cache/$USER
# fi
