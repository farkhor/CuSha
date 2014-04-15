CuSha
=====

CuSha is a CUDA-based vertex-centric graph processing framework that uses G-Shards and CW representation formats.


Requirements
-----
 Cuda Compute Capability of your device has to be 2.0 or above.
 Warp size has to be 32.
 
 
Future
-----
With the introduction of new CUDA devices that translate shared memory atomics into one instruction, and also the increase in shared memory per SM quota, I anticipate CuSha will be executed even faster on Maxwell GPUs.  
