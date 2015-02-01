#include <stdexcept>
#include <cmath>

#include "find_block_size.cuh"
#include "../common/user_specified_structures.h"
#include "../common/cuda_error_check.cuh"

// This function does NOT guarantee the best block size. But it tries to come up with the best.
// Be aware block sizes rather than what this function chooses might end up showing better performance.
// Any suggestions to improve this function will be appreciated.
blockSize_N_pair find_proper_block_size(
		const int suggestedBlockSize,
		const uint nEdges,
		const uint nVertices ) {

	// Getting current device properties to properly select block size and N.
	int currentDevice;
	CUDAErrorCheck( cudaGetDevice( &currentDevice ) );
	cudaDeviceProp deviceProp;
	CUDAErrorCheck ( cudaGetDeviceProperties( &deviceProp, currentDevice ) );
	int maxVerticesPerSM = deviceProp.sharedMemPerBlock / sizeof(Vertex);

	int MaxBlockPerSM;	// Maximum number of resident blocks per multiprocessor. Not queryable (is it a word??) by CUDA runtime.
	#if __CUDA_ARCH__ < 300
		MaxBlockPerSM = 8;
	#endif
	#if __CUDA_ARCH__ >= 300 & __CUDA_ARCH__ < 500
		MaxBlockPerSM = 16;
	#endif
	#if __CUDA_ARCH__ >= 500
		MaxBlockPerSM = 32;
	#endif

	// If suggested block size is 0 (user hasn't entered anything), we ignore it.
	blockSize_N_pair BS_N;
	if( suggestedBlockSize == 0 ) {
		int approximated_N = (int)std::sqrt((deviceProp.warpSize * std::pow(nVertices,2))/nEdges);	// Please refer to paper for explanation.
		//fprintf( stdout, "Approximated N: %d\n", approximated_N);
		for( int b_per_SM = 2; b_per_SM<=MaxBlockPerSM; ++b_per_SM ) {
			blockSize_N_pair temp_pair;
			temp_pair.blockSize = deviceProp.maxThreadsPerMultiProcessor/b_per_SM;
			if ( deviceProp.maxThreadsPerMultiProcessor % (temp_pair.blockSize * b_per_SM) != 0 )
				continue;
			if( temp_pair.blockSize > deviceProp.maxThreadsPerBlock)
				continue;
			temp_pair.N = maxVerticesPerSM / b_per_SM;
			if( temp_pair.N > approximated_N )
				BS_N = temp_pair;
		}

	}
	else {
		// The behavior is undefined if user-specified block size is not a power of two. Usual block sizes are 1024, 512, 256, and 128.
		if( suggestedBlockSize > deviceProp.maxThreadsPerBlock )
			throw std::runtime_error( "Specified block size is invalid." );
		BS_N.blockSize = suggestedBlockSize;
		BS_N.N = ( maxVerticesPerSM * suggestedBlockSize ) / deviceProp.maxThreadsPerMultiProcessor;
	}

	return BS_N;
}
