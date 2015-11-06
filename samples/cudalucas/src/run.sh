#!/usr/bin/env bash

for i in 64 128 256 512; do
    ./CUDALucas $(( ${i} * 1024 ))
done
