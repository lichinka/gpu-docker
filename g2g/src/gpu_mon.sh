#nvidia-smi > ${SCRATCH}/nvidia.$$.log &
nvidia-smi -q --display=MEMORY --loop=1 > ${SCRATCH}/smi.$$.log &
