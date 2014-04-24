#CuSha

CuSha is a CUDA-based vertex-centric graph processing framework that uses G-Shards and Concatenated Windows (CW) representations to store graphs inside the GPU global memory. G-Shards and CW consume more space compared to Virtual Warp-Centric (VWC) method that uses Compressed Sparse Row (CSR) format but on the other hand provide better performance due to GPU-friednly representations. For completeness, provided package also includes VWC processing method for GPU and CPU. CPU implementation utilizes Pthreads.


##Requirements

First of all, provided package, as it is, contains multiple source files. As a result, you will need CUDA 5.0 or higher to compile separately and link, alongside a CUDA-enabled device with Compute Capability (CC) 2.0 or higher. For more info, please have a look at [these slides](http://on-demand.gputechconf.com/gtc-express/2012/presentations/gpu-object-linking.pdf).    
For processing with both G-Shards and CW representations, I have used  *__syncthreads _or(int)* function, which is not supported in architectures before Fermi. Other than that, atomic operations require minimum CCs. Please refer to [CUDA programming guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions) for more info. CuSha is forward-compatible with warp sizes rather than 32.    
Provided VWC on GPU requires that physical warp size to be 32.   
VWC on CPU requires *NVCC* to be compiled although with small modifications it can be a pure C/C++ piece of code.   
This package is tested under **Ubuntu 12.04**. It wouldn't be hard to port it to other Operating Systems.

 
##Future

With the introduction of new CUDA devices that translate shared memory atomics into one instruction, and also the increase in shared memory per SM quota, I anticipate CuSha will be executed even faster on Maxwell GPUs.
