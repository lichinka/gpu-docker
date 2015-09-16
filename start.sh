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
HOST_DRIVER_FILES="$( ldconfig -p | grep 'libcuda.so' | grep -v 'lib32' | awk '{ print $4; }' )"

for f in ${HOST_DRIVER_FILES}; do
    CONT_DRIVER_FILES="${CONT_DRIVER_FILES} -v ${f}:/opt/cuda/lib64/$( basename ${f} )"
done
HOST_DIR_NAME="$( dirname ${f} )"
CONT_DRIVER_FILES="${CONT_DRIVER_FILES} -v ${HOST_DIR_NAME}/libcuda.so.${DRIVER_VERSION}:/opt/cuda/lib64/libcuda.so.${DRIVER_VERSION}"

echo "${CONT_DRIVER_FILES}"

docker run --rm                                                             \
           -it                                                              \
           --privileged                                                     \
           ${CONT_DRIVER_FILES}                                             \
           cuda65:latest
