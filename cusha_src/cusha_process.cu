#include "cusha_process.cuh"
#include "../common/user_specified_device_functions.cuh"

template <GraphProcessingMethod procesingMethod>
__global__ void CuSha_GPU_kernel(
		const uint nShards,
		const uint N, // Maximum number of vertices assigned to a shard.
		const uint* SrcIndex,
		const uint* DestIndex,
		Vertex* SrcValue,
		Vertex* VertexValues,
		const Edge* EdgeValue,
		const Vertex_static* SrcValue_static,
		int* finishedProcessing,
		const uint* shardSizesScan,
		const uint* concatenatedWindowsSizesScan,	// For CW.
		const uint* windowSizesScansVertical,	// For GS.
		const uint* Mapper = NULL ) {	// Used only when processing method is CW.

	extern __shared__ Vertex localVertices[];

	// Shard index is determined by blockIdx.x.
	uint shardOffset = blockIdx.x * N;
	uint shardStartingAddress = shardSizesScan[ blockIdx.x ];
	uint shardEndingAddress = shardSizesScan[ blockIdx.x + 1 ];
	Vertex* shardVertexValues = VertexValues + shardOffset;

	/* 1st stage */
	// Initialize block vertices residing in shared memory.
	for( 	uint vertexID = threadIdx.x;
			vertexID < N;
			vertexID += blockDim.x ) {
		init_compute_CuSha( localVertices + vertexID, shardVertexValues + vertexID );
	}

	/* 2nd stage */
	// Consecutive entries of shard are processed by consecutive threads.
	__syncthreads();
	for( 	uint EntryAddress = shardStartingAddress + threadIdx.x;
			EntryAddress < shardEndingAddress;
			EntryAddress += blockDim.x ) {

		compute_CuSha(	SrcValue[ EntryAddress ],
						SrcValue_static + EntryAddress,
						EdgeValue + EntryAddress,
						localVertices + ( DestIndex[ EntryAddress ] - shardOffset ) );

	}


	/* 3rd stage */
	// Check if any update has happened.
	__syncthreads();
	int flag = false;
	for( 	uint vertexID = threadIdx.x;
			vertexID < N;
			vertexID += blockDim.x	) {

		if( update_condition_CuSha( localVertices + vertexID, shardVertexValues + vertexID ) ) {
			flag = true;
			shardVertexValues[ vertexID ] = localVertices[ vertexID ];
		}

	}


	/* 4th stage */
	// If any vertex has been updated during processing, update shard's corresponding windows.
	if( __syncthreads_or( flag ) ) {	// Requires (CC>=2.0).

		if( procesingMethod == CW ) {

			uint shardCWStartingAddress = concatenatedWindowsSizesScan[ blockIdx.x ];
			uint shardCWEndingAddress = concatenatedWindowsSizesScan[ blockIdx.x + 1 ];
			for( 	uint EntryAddress = shardCWStartingAddress + threadIdx.x;
					EntryAddress < shardCWEndingAddress;
					EntryAddress += blockDim.x	) {

					SrcValue[ Mapper[ EntryAddress ] ] = localVertices[ SrcIndex[ EntryAddress ] - shardOffset ];

			}

		}
		else {	//GS

			// Each warp in the block updates one specific window in one specific shard.
			for( 	uint targetShardIndex = threadIdx.x / warpSize;	//	threadIdx.x >> 5
					targetShardIndex < nShards;
					targetShardIndex += ( blockDim.x / warpSize ) ) {	//	blockDim.x >> 5

				uint targetWindowStartingAddress = windowSizesScansVertical[ targetShardIndex * nShards + blockIdx.x ];
				uint targetWindowEndingAddress = windowSizesScansVertical[ targetShardIndex * nShards + blockIdx.x + 1 ];
				// Threads of warp update window entries in parallel.
				for( 	uint windowEntry = targetWindowStartingAddress + ( threadIdx.x & ( warpSize - 1 ) );
						windowEntry < targetWindowEndingAddress;
						windowEntry += warpSize ) {

					SrcValue[ windowEntry ] = localVertices[ SrcIndex[ windowEntry ] - shardOffset ];

				}

			}

		}

		// Signal the host to launch a new kernel.
		if( threadIdx.x == 0 )
			(*finishedProcessing) = 1;

	}

}

void cusha_process(
		const GraphProcessingMethod procesingMethod,
		const uint blockSize,
		const uint N,
		const uint nShards,
		const uint nVertices,
		Vertex* vertexValue,
		const uint* concatenatedWindowsSizesScan,
		const uint* windowSizesScansVertical,
		const uint* shardSizesScans,
		int* finished,
		Vertex* srcValue,
		const uint* dstIndex,
		Edge* edgeValue,
		Vertex_static* vertexValueStatic,
		const uint* srcIndex,
		const uint* mapper ) {


	if( procesingMethod == CW ) {
		CuSha_GPU_kernel < CW >
			<<< nShards, blockSize, ( N * sizeof(Vertex) ) >>> (
					nShards,
					N,
					srcIndex,
					dstIndex,
					srcValue,
					vertexValue,
					edgeValue,
					vertexValueStatic,
					finished,
					shardSizesScans,
					concatenatedWindowsSizesScan,
					windowSizesScansVertical,
					mapper );
	}
	else {	// Processing method is GS.
		CuSha_GPU_kernel < GS >
			<<< nShards, blockSize, ( N * sizeof(Vertex) ) >>> (
					nShards,
					N,
					srcIndex,
					dstIndex,
					srcValue,
					vertexValue,
					edgeValue,
					vertexValueStatic,
					finished,
					shardSizesScans,
					concatenatedWindowsSizesScan,
					windowSizesScansVertical );
	}


}
