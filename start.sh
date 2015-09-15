#!/bin/bash

#
# get the driver version
#
DRIVER_VERSION="$( nvidia-smi | grep Driver | awk '{ print $3; }' )"
if [ -z "${DRIVER_VERSION}" ]; then
    echo "No CUDA driver could be detected"
    exit 1
else
    echo "CUDA driver ${DRIVER_VERSION} detected"
fi

#
# get the driver files
#
CONT_DRIVER_FILES=""
HOST_DRIVER_FILES="$( find / -name "libcuda.so.${DRIVER_VERSION}" 2>/dev/null )"

for f in ${HOST_DRIVER_FILES}; do
    if [ -n "$( file ${f} | grep -o 'ELF 64' )" ]; then
        HOST_DIR_NAME="$( dirname ${f} )"
        CONT_DRIVER_FILES="${CONT_DRIVER_FILES} -v ${f}:/opt/cuda/lib64/$( basename ${f} )"
    fi
done
CONT_DRIVER_FILES="${CONT_DRIVER_FILES} -v ${HOST_DIR_NAME}/libcuda.so:/opt/cuda/lib64/libcuda.so"
CONT_DRIVER_FILES="${CONT_DRIVER_FILES} -v ${HOST_DIR_NAME}/libcuda.so.1:/opt/cuda/lib64/libcuda.so.1"

docker run --rm                                                             \
           -it                                                              \
           --privileged                                                     \
           ${CONT_DRIVER_FILES}                                             \
           cudagpu:latest
