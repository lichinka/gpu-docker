#! /bin/sh

IMAGE_NAME="cuda_sample"
CONTAINER_NAME="${IMAGE_NAME}_container"

if [ -z $GPU ]; then
    echo "No GPU selected"
    exit 0
fi

docker rm -f $CONTAINER_NAME 2> /dev/null

for sample in $(ls -d */); do
    docker build -t $IMAGE_NAME --rm=true "$sample"
    ../nvidia-docker run -it --name=$CONTAINER_NAME $IMAGE_NAME
    docker rm $CONTAINER_NAME
    docker rmi $IMAGE_NAME
done
