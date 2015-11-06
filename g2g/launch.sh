GPU=1 ../nvidia-docker run --rm=true --volume=${HOME}/nvidia-docker/samples/g2g/ssh:/root/.ssh --hostname=node01 --name=cuda_g2g_01 -it cuda_g2g bash
