#ifndef FIND_BLOCK_SIZE_CUH
#define FIND_BLOCK_SIZE_CUH

struct blockSize_N_pair{
	uint blockSize;
	uint N;	// The maximum number of vertices inside a shard.
};

// This function does NOT guarantee the best block size. But it tries to come up with the best.
// Be aware block sizes rather than what this function chooses might end up showing better performance.
// Any suggestions to improve this function will be appreciated.
blockSize_N_pair find_proper_block_size(
		const int suggestedBlockSize,
		const uint nEdges,
		const uint nVertices );


#endif	//	FIND_BLOCK_SIZE_CUH
