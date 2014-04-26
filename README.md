#![alt text](http://farkhor.github.io/CuSha/images/CuSha-logo-small.png "CuSha")

CuSha is a CUDA-based vertex-centric graph processing framework that uses G-Shards and Concatenated Windows (CW) representations to store graphs inside the GPU global memory. G-Shards and CW consume more space compared to Compressed Sparse Row (CSR) format but on the other hand provide better performance due to GPU-friednly representations. For completeness, provided package also includes Virtual Warp-Centric (VWC) processing method for GPU and a multi-threaded CPU implementation, both using CSR representation.        
We prepared a paper about CuSha that's accepted in [HPDC'14](http://www.hpdc.org/2014/) conference:    

    F. Khorasani, K. Vora, R. Gupta, and L.N. Bhuyan    
    CuSha: Vertex-Centric Graph Processing on GPUs,    
    23rd International ACM Symposium on High Performance Parallel and Distributed Computing,    
    13 pages, Vancouver, Canada, June 2014.


##Requirements

Provided package, as it is, contains multiple source files. As a result, you will need CUDA 5.0 or higher to compile separately and link, alongside a CUDA-enabled device with Compute Capability (CC) 2.0 or higher. For a detailed explanation about requirements, please check out [CuSha page](http://farkhor.github.io/CuSha/).

 
##Usage

For elaborated usage instructions, please have a look at  [CuSha page](http://farkhor.github.io/CuSha/).
