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
    echo "Ready to build a CUDA ${TOOLKIT_VERSION} image."
    echo "Press any key to continue or Ctrl+C to exit ..."
    read -n 1

    docker build --rm -t cuda${TOOLKIT_VERSION} --file=Dockerfile.cuda${TOOLKIT_VERSION} .
fi
