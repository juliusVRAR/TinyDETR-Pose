#!/bin/bash
#SBATCH --job-name=test_training
#SBATCH --gpus-per-node=2
#SBATCH --export=NUM_GPU=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

DSNAME=lw_detr6d_data

DATAPATH=/opt/cache/$USER
OUTPATH=$DATAPATH/$DSNAME

PATH_TO_BOP=$OUTPATH/bop_datasets
PATH_TO_WEIGHTS=$OUTPATH/weights/lw-detr
WORKSPACE=$PWD

gio_mount_nas()
{
    mkdir -p ~/dbus
    XDG_RUNTIME_DIR=~/dbus
    export DBUS_SESSION_BUS_ADDRESS=`dbus-daemon --fork --print-address --session`
    gio mount smb://pc3163/nobackup < ~/smbcreds
}


if test ! -d $OUTPATH ; then
    gio_mount_nas
    gio copy --progress smb://pc3163/nobackup/cache/jkuehn/$DSNAME.zip $DATAPATH
    cd $DATAPATH
    unzip -q $DSNAME.zip 
fi

IMAGE_NAME=lw-detr6d
## MAIN
rootless-docker run --gpus all --shm-size=256g \
    -v $WORKSPACE:/workspace/LWDETR\
    -v $PATH_TO_BOP:/workspace/LWDETR/data/datasets/bop \
    -v $PATH_TO_WEIGHTS:/workspace/LWDETR/data/weights \
    pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME\
    bash -c "python /workspace/LWDETR/models/ops/setup.py build install && \
                /workspace/LWDETR/scripts/pose/0_tiny/0_train.sh /workspace/LWDETR/data/datasets/bop $NUM_GPU"
                                    
        

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
