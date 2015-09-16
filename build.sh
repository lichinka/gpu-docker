#!/bin/bash

echo "The container image will try to autodetect the NVIDIA driver"
echo "Press any key to continue or Ctrl+C to exit ..."
read -n 1

docker build --rm -t cuda65 --file=Dockerfile.cuda65 .
