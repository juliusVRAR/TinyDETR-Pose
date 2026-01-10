#!/bin/bash
IMAGE_NAME=lw-detr6d_trt
docker login pc3163.igd.fraunhofer.de:4567
docker build -f Dockerfile.trt -t pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME .
#docker push pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME  


