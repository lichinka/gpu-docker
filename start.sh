#!/bin/bash


TOOLKIT_VERSION="$1"

if [ -z "${TOOLKIT_VERSION}" ]; then
    echo "$0 <version tag>"
    echo
    echo "where <version tag> should be one of:"
    echo "	55	for CUDA 5.5"
    echo "	65	for CUDA 6.5"
    echo "	70	for CUDA 7.0"
    exit 1
else
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
    HOST_DRIVER_FILES="$( ldconfig -p | grep 'libcuda.so' | grep 'lib64' | awk '{ print $4; }' )"
    if [ -z "${HOST_DRIVER_FILES}" ]; then
        HOST_DRIVER_FILES="$( ldconfig -p | grep 'libcuda.so' | grep -v 'lib32' | awk '{ print $4; }' )"
    fi

    for f in ${HOST_DRIVER_FILES}; do
        CONT_DRIVER_FILES="${CONT_DRIVER_FILES} -v ${f}:/opt/cuda/lib64/$( basename ${f} )"
    done
    HOST_DIR_NAME="$( dirname ${f} )"
    CONT_DRIVER_FILES="${CONT_DRIVER_FILES} -v ${HOST_DIR_NAME}/libcuda.so.${DRIVER_VERSION}:/opt/cuda/lib64/libcuda.so.${DRIVER_VERSION}"

    docker run --rm                                                             \
               -it                                                              \
               --privileged                                                     \
               ${CONT_DRIVER_FILES}                                             \
               cuda${TOOLKIT_VERSION}:latest
fi

