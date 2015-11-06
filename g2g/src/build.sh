#!/bin/sh

# 
# compilation script prepared for Daint and Opcode
#
SRC=.

case $( hostname ) in 
    *daint* | *santis*)
        MODS="PrgEnv-gnu craype-accel-nvidia35 cudatoolkit cray-libsci_acc"
        CC=cc
        CXX=CC
        CLSDK=/opt/nvidia/cudatoolkit
        CLLIB=/opt/cray/nvidia/default
        ;;
    *opcode*)
        MODS="mvapich2/2.0-gcc-opcode2-4.7.2"
        CC=mpicc
        CXX=g++
        CLSDK=/apps/opcode/CUDA-5.5
        CLLIB=${CLSDK}
        ;;
    *)
        echo "Don't know how to compile here. Trying anyway ..."
        MODS=""
        CC=mpicc
        CXX=mpic++
        CLSDK=/usr/local/cuda
        CLLIB=${CLSDK}
        ;;
esac

echo -n "Checking modules on $( hostname ) ... "
for m in ${MODS}; do
    if [ -z "$( echo ${LOADEDMODULES} | grep ${m} )" ]; then
        echo -e "Please issue:\n\tmodule load ${m}"
        exit 1
    fi
done
echo "ok"

echo "Building on $( hostname ) ..."
$CXX $SRC/01_device_query.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 01_device_query
$CXX $SRC/10_mpi.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 10_mpi
$CC  -DPINNED $SRC/osu_bw_cl.c -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o osu_bw_cl
$CC  -DPINNED $SRC/osu_bw_cuda.c -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o osu_bw_cuda

