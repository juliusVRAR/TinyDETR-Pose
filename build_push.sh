#!/bin/bash
IMAGE_NAME=lwdetr6d-duc
docker build -f Dockerfile.6d -t pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME .
docker push pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME  


