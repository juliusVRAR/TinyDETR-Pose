#!/bin/bash
# Start docker container in WSL to debug locally
WORKSPACE=$PWD
# PATH_TO_COCO="/mnt/e/datasets/coco2017"
PATH_TO_BOP="/mnt/e/datasets/bop_datasets/"
PATH_TO_WEIGHTS="/mnt/e/weights/lw-detr/"
NAME=LW-DETR6D-TRT
IMAGE_NAME=lw-detr6d_trt
  # -e DISPLAY=$DISPLAY \
  # -v /tmp/.X11-unix:/tmp/.X11-unix \
    # -v $PATH_TO_COCO:/workspace/LWDETR/data/datasets/coco \

docker run --gpus all -it -d --ipc=host \
  --name $NAME \
  -v $WORKSPACE:/workspace/LWDETR\
  -v $PATH_TO_BOP:/workspace/LWDETR/data/datasets/bop/ycbv/ \
  -v $PATH_TO_WEIGHTS:/workspace/LWDETR/data/weights \
  pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME

