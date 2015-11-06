#!/usr/bin/env bash

#
# all tests fail if:
#
#   export MPICH_RDMA_ENABLED_CUDA=1
#
# is set (error -33: CL_INVALID_DEVICE) on Cray systems with cray and gnu 
# program environments. Failure occurs at context-creation time.
# Moreover, setting this variable:
#
#   export MPICH_G2G_PIPELINE=16
#
# is ignored by the MPICH-cray implementation if the above RDMA is not set.
#
NPROC=2

case $( hostname ) in
    *daint* | *santis*)
        EXEC="aprun -N1 -n${NPROC}"
        ;;
    *opcode*)
        export CUDA_VISIBLE_DEVICES="0,4"
        EXEC="mpiexec.hydra -n${NPROC}"
        ;;
    *)
        echo "Don't know how to execute here. Exiting."
        exit 1;
        ;;
esac

#
# query for OpenCL devices
#
if [ -n "$( ${EXEC} ./01_device_query | grep CUDA )" ]; then
    echo "PASSED : Found CUDA devices with OpenCL support"
else
    echo "FAILED : No OpenCL accellerators found"
fi

#
# timed-transfer tests
#
for i in $( seq 13 15 ); do
    NUM="$( echo "2^${i}" | bc -l )"
    echo "# Transfering ${NUM} doubles ..."
    ${EXEC} ./10_mpi 0 gpu 0 ${NUM}
done

#
# bandwidth test
#
export MV2_USE_CUDA=1
for streams in $( seq 0 1 ); do
    for ipc in $( seq 0 1 ); do
        for smp in $( seq 0 1 ); do
            export MV2_CUDA_NONBLOCKING_STREAMS=${streams}
            export MV2_CUDA_IPC=${ipc}
            export MV2_CUDA_SMP_IPC=${smp}
            env | grep MV2 | sed -e 's/^M/# M/g'
            #
            # OpenCL bandwidth test
            #
            ${EXEC} ./osu_bw_cl
            #
            # CUDA bandwidth test
            #
            ${EXEC} ./osu_bw_cuda D D
            echo "##########"
        done
    done
done

