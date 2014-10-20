#!/bin/sh
nvcc entryPoint.cu common/simpleTime.cu csr_src/csr-utils.cu csr_src/MTCPU.cu csr_src/VWC.cu cusha_src/cusha-CW.cu cusha_src/cusha-GS.cu cusha_src/cusha-utils.cu -arch=sm_35 -O3 -rdc=true -o cusha
