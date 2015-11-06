#!/bin/sh

IMAGE_NAME="cuda_g2g"
CONTAINER_NAME="${IMAGE_NAME}_00"

if [ -z $GPU ]; then
    echo "No GPU selected"
    exit 0
fi

docker rm -f $CONTAINER_NAME 2> /dev/null

docker build -t $IMAGE_NAME --rm=true "g2g"
../nvidia-docker run --hostname="node00" --name=${CONTAINER_NAME} ${IMAGE_NAME}
docker rm $CONTAINER_NAME
