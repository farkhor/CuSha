#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../common/simpleTime.h"
#include "../common/CUDAErrorCheck.h"
#include "cusha-utils.cuh"
#include "cusha-UserFunctions.cuh"

// GPU kernel performing vertex-centric computation having G-Shards representation.
__global__ void CuSha_GS_GPU_kernel(	const uint num_of_shards,
										const uint N, // Maximum number of vertices assigned to a shard.
										const uint* SrcIndex,
										const uint* DestIndex,
										Vertex* SrcValue,
										Vertex* VertexValues,
										Edge* EdgeValue,
										Vertex_static* SrcValue_static,
										const uint* win_offsets,
										const uint* shard_sizes,
										int* finished_processing	) {

	extern __shared__ Vertex local_Vertices[];

	// Shard index is determined by blockIdx.x.
	uint shard_offset = blockIdx.x * N;

	// Get the stating and ending regions of the long arrays of entries the block has to work on as a shard.
	// Alternatives can be used. It eliminates the need for "shard_sizes" array.
	// I decided to keep it like this for simplicity. Plus I hope the compiler stores the "shard_sizes" array into the constant memory when it can fit.
	uint shard_starting_address = shard_sizes [ blockIdx.x ];	// 	Alternative: uint shard_starting_address = win_offsets[shard_index*num_of_shards];
	uint shard_ending_address = shard_sizes [ blockIdx.x + 1 ];	//	Alternative: uint shard_ending_address = win_offsets[shard_index*num_of_shards+num_of_shards];

	Vertex* shard_VertexValues = VertexValues + shard_offset;


	/* 1st stage */
	// Initialize block vertices residing in shared memory.
	for ( 	uint vertexID = threadIdx.x;
			vertexID < N;
			vertexID += blockDim.x	) {
		init_compute_CuSha ( local_Vertices + vertexID, shard_VertexValues + vertexID );
	}


	/* 2nd stage */
	// Consecutive entries of shard are processed by consecutive threads.
	__syncthreads();
	for ( 	uint EntryAddress = shard_starting_address + threadIdx.x;
			EntryAddress < shard_ending_address;
			EntryAddress += blockDim.x	) {

		compute_CuSha (	SrcValue + EntryAddress,
						SrcValue_static + EntryAddress,
						EdgeValue + EntryAddress,
						local_Vertices + ( DestIndex[ EntryAddress ] - shard_offset ) );

	}


	/* 3rd stage */
	// Check if any update has happened.
	__syncthreads();
	int flag = false;
	for ( 	uint vertexID = threadIdx.x;
			vertexID < N;
			vertexID += blockDim.x ) {

		if( update_condition_CuSha ( local_Vertices + vertexID, shard_VertexValues + vertexID ) ) {
			flag = true;
			shard_VertexValues [ vertexID ] = local_Vertices [ vertexID ];
		}


	}


	/* 4th stage */
	// If any vertex has been updated during processing, update shard's corresponding windows.
	if( __syncthreads_or( flag ) ) {	// Requires (CC>=2.0).

		//uint warpMask = 31 - __clz(warpSize);	// Requires (CC>=2.0).

		// Each warp in the block updates one specific window in one specific shard.
		for( 	uint target_shard_index = threadIdx.x / warpSize;	//	Alternative for "/ warpSize": ">> warpMask"
				target_shard_index < num_of_shards;
				target_shard_index += ( blockDim.x / warpSize )	) {	//	Alternative for "/ warpSize": ">> warpMask"

			uint target_window_starting_address = win_offsets [ target_shard_index * num_of_shards + blockIdx.x ];
			uint target_window_ending_address = win_offsets [ target_shard_index * num_of_shards + blockIdx.x + 1 ];

			// Threads of warp update window entries in parallel.
			for( 	uint window_entry = target_window_starting_address + ( threadIdx.x & ( warpSize - 1 ) );
					window_entry < target_window_ending_address;
					window_entry += warpSize	) {

				SrcValue [ window_entry ] = local_Vertices [ SrcIndex [ window_entry ] - shard_offset ];

			}

		}

		if( threadIdx.x == 0 )
			*finished_processing = JOB_NOT_FINISHED_YET;

	}

}

bool processGraphGS(	GSGraph* Host_Graph,
						const blockSize_N_pair blockSize_and_N) {

	// Variables collecting timing info.
	float H2D_copy_time, processing_time, D2H_copy_time;

	// Setting kernel launch parameters. Each shard: one block.
	/*
	 * BTW, I tried using enough blocks to occupy all SMs, and then reusing threads to eliminate the overhead associated with scheduling new blocks.
	 * But this made me use another __syncthreads() at the end of each iteration; that did lead to increased computation time.
	 */
	dim3 gridDim( Host_Graph->num_of_shards, 1, 1);
	dim3 blockDim( blockSize_and_N.blockSize, 1, 1 );
	const unsigned int requiredSharedMem = blockSize_and_N.N * sizeof(Vertex);

	// G-Shards representation required arrays.
	unsigned int *dev_SrcIndex, *dev_DestIndex;
	Vertex *dev_SrcValue, *dev_VertexValues;
	Edge *dev_EdgeValue;
	Vertex_static *dev_SrcValue_static;
	unsigned int *dev_win_offsets, *dev_shard_sizes;

	// Host and device flags indicating if the processing is finished.
	int finished;
	int* dev_finished;
	CUDAErrorCheck ( cudaMalloc((void**)&dev_finished, sizeof(int)) );

	// Creation and memory allocation for the graph in device global memory.
	CUDAErrorCheck ( cudaMalloc( (void**)&dev_SrcIndex, Host_Graph->num_of_edges * sizeof(unsigned int) ) );
	CUDAErrorCheck ( cudaMalloc( (void**)&dev_DestIndex, Host_Graph->num_of_edges * sizeof(unsigned int) ) );
	CUDAErrorCheck ( cudaMalloc( (void**)&dev_SrcValue, Host_Graph->num_of_edges * sizeof(Vertex) ) );
	CUDAErrorCheck ( cudaMalloc( (void**)&dev_VertexValues, Host_Graph->num_of_shards * blockSize_and_N.N * sizeof(Vertex) ) ); // Instead of "Host_Graph->num_of_vertices * sizeof(Vertex) ) );" for padding.
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaMalloc( (void**)&dev_EdgeValue, Host_Graph->num_of_edges * sizeof(Edge) ) );
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaMalloc( (void**)&dev_SrcValue_static, Host_Graph->num_of_edges * sizeof(Vertex_static) ) );
	CUDAErrorCheck ( cudaMalloc( (void**)&dev_win_offsets, ( Host_Graph->num_of_shards * Host_Graph->num_of_shards + 1 )* sizeof(unsigned int) ) );
	CUDAErrorCheck ( cudaMalloc( (void**)&dev_shard_sizes, ( Host_Graph->num_of_shards + 1 )* sizeof(unsigned int) ) );

	// Copy the graph from the host to the device.
	// Use asynchronous copies to avoid unnecessary synchronization with the host for each and every copy.
	// Synchronize when copy operations are all queued.
	fprintf ( stdout, "Copying the graph from the host to the device ...\n" );
	setTime();
	// As you can see for CuSha-GS, CuSha-CW, and VWC we haven't included allocation or pre-processing times.
	// It makes it unfair to compare them with each other and especially with the MT-CPU version if the user only looks for one-time graph processing.

	CUDAErrorCheck ( cudaMemcpyAsync( dev_VertexValues, Host_Graph->VertexValues, (Host_Graph->num_of_vertices)*sizeof(Vertex), cudaMemcpyHostToDevice ) );
	CUDAErrorCheck ( cudaMemcpyAsync( dev_win_offsets, Host_Graph->windowOffsets, ( Host_Graph->num_of_shards * Host_Graph->num_of_shards + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	CUDAErrorCheck ( cudaMemcpyAsync( dev_shard_sizes, Host_Graph->shard_sizes, ( Host_Graph->num_of_shards + 1 )*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	CUDAErrorCheck ( cudaMemcpyAsync( dev_SrcIndex, Host_Graph->incoming_index, (Host_Graph->num_of_edges)*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	CUDAErrorCheck ( cudaMemcpyAsync( dev_DestIndex, Host_Graph->outgoing_index, (Host_Graph->num_of_edges)*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	CUDAErrorCheck ( cudaMemcpyAsync( dev_SrcValue, Host_Graph->SrcValue, (Host_Graph->num_of_edges)*sizeof(Vertex), cudaMemcpyHostToDevice ) );
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaMemcpyAsync( dev_EdgeValue, Host_Graph->EdgeValue, (Host_Graph->num_of_edges)*sizeof(Edge), cudaMemcpyHostToDevice ) );
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaMemcpyAsync( dev_SrcValue_static, Host_Graph->SrcValue_static, (Host_Graph->num_of_edges)*sizeof(Vertex_static), cudaMemcpyHostToDevice ) );
	CUDAErrorCheck ( cudaDeviceSynchronize() );

	H2D_copy_time = getTime();
	fprintf( stdout, "Copying the graph from the host to the device finished in : %f (ms)\n", H2D_copy_time );

	// Iteratively process the graph on the device.
	fprintf( stdout, "Processing graph using G-Shards representation ...\n" );
	unsigned int counter = 0;
	setTime();
	do {
		finished = JOB_FINISHED;
		CUDAErrorCheck ( cudaMemcpyAsync ( dev_finished, &finished, sizeof(int), cudaMemcpyHostToDevice ) );
		CuSha_GS_GPU_kernel <<< gridDim, blockDim, requiredSharedMem >>> (	Host_Graph->num_of_shards,
																			blockSize_and_N.N,
																			dev_SrcIndex,
																			dev_DestIndex,
																			dev_SrcValue,
																			dev_VertexValues,
																			dev_EdgeValue,
																			dev_SrcValue_static,
																			dev_win_offsets,
																			dev_shard_sizes,
																			dev_finished);
		CUDAErrorCheck ( cudaPeekAtLastError() );
		// Implicit synchronization point with the host using synchronous memory copy.
		CUDAErrorCheck ( cudaMemcpy ( &finished, dev_finished, sizeof(int), cudaMemcpyDeviceToHost ) );
		counter++;
	} while ( finished == JOB_NOT_FINISHED_YET );
	processing_time = getTime();
	fprintf( stdout, "Processing finished in : %f (ms)\n", processing_time);
	fprintf( stdout, "Performed %u iterations in total.\n", counter);

	// Copy resulted vertex values back from the device to the host.
	fprintf( stdout, "Copying final vertex values from the device to the host ...\n" );
	setTime();
	CUDAErrorCheck ( cudaMemcpy( Host_Graph->VertexValues, dev_VertexValues, (Host_Graph->num_of_vertices)*sizeof(Vertex), cudaMemcpyDeviceToHost));
	D2H_copy_time = getTime();
	fprintf( stdout, "Copying final vertex values back from the device to the host finished in : %f (ms)\n", D2H_copy_time);
	fprintf( stdout, "Total Execution time was : %f (ms)\n", H2D_copy_time+processing_time+D2H_copy_time );

	// Free up allocated device memories.
	CUDAErrorCheck( cudaFree( dev_finished ) );
	CUDAErrorCheck( cudaFree( dev_SrcIndex ) );
	CUDAErrorCheck( cudaFree( dev_DestIndex ) );
	CUDAErrorCheck( cudaFree( dev_SrcValue ) );
	CUDAErrorCheck( cudaFree( dev_VertexValues ) );
	if(sizeof(Edge)>0) CUDAErrorCheck( cudaFree( dev_EdgeValue ) );
	if(sizeof(Vertex_static)>0)	CUDAErrorCheck( cudaFree( dev_SrcValue_static ) );
	CUDAErrorCheck( cudaFree( dev_win_offsets ) );
	CUDAErrorCheck( cudaFree( dev_shard_sizes ) );

	return(EXIT_SUCCESS);
}

void ExecuteCuShaGS(	FILE* inputFile,
						const int suggestedBlockSize,
						FILE* outputFile,
						const int arbparam) {

	GSGraph HostGraph;
	HostGraph.num_of_edges = 0;
	HostGraph.num_of_vertices = 0;
	HostGraph.num_of_shards = 0;

	primitiveVertex* primitiveVertices = (primitiveVertex*) malloc ( sizeof(primitiveVertex));
	assert( primitiveVertices );
	init_primitiveVertex ( primitiveVertices );

	fprintf( stdout, "Populating the graph ...\n" );
	populatePrimitiveVerticesForCuSha( &(HostGraph.num_of_edges), &(HostGraph.num_of_vertices), &primitiveVertices, inputFile, arbparam);

	if(HostGraph.num_of_edges == 0) {
		fprintf( stderr, "No edge could be read from the file. Make sure provided formatting matches defined specification.\n");
		exit(EXIT_FAILURE);
	}
	else {
		fprintf( stdout, "Graph is populated with %u vertices and %u edges.\n", HostGraph.num_of_vertices, HostGraph.num_of_edges );
	}
	fprintf( stdout, "Organizing the graph to process it using G-Shards representation ...\n");

	blockSize_N_pair blockSize_N = findProperBlockSize ( suggestedBlockSize, HostGraph.num_of_edges, HostGraph.num_of_vertices, sizeof(Vertex) );
	if( blockSize_N.blockSize == 0 ) {
		fprintf( stderr, "Block size couldn't be specified.\n");
		exit(EXIT_FAILURE);
	}
	else {
		fprintf( stdout, "The size of each kernel thread block is set to %u and %u vertices are assigned to each shard.\n", blockSize_N.blockSize, blockSize_N.N );
	}

	HostGraph.num_of_shards = ceil( (double)HostGraph.num_of_vertices / blockSize_N.N );
	fprintf( stdout, "Graph is divided into %u shard(s).\n", HostGraph.num_of_shards );
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.VertexValues), (HostGraph.num_of_vertices) * sizeof(Vertex), cudaHostAllocDefault ) );
	for( int i = 0; i < HostGraph.num_of_vertices; ++i )
		HostGraph.VertexValues[i] = primitiveVertices[i].VertexValue;
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.windowOffsets), (HostGraph.num_of_shards*HostGraph.num_of_shards+1) * sizeof(unsigned int), cudaHostAllocDefault ) );
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.shard_sizes), (HostGraph.num_of_shards+1) * sizeof(unsigned int), cudaHostAllocDefault ) );
	memset(HostGraph.windowOffsets, 0, (HostGraph.num_of_shards*HostGraph.num_of_shards+1) * sizeof(unsigned int));
	memset(HostGraph.shard_sizes, 0, (HostGraph.num_of_shards+1) * sizeof(unsigned int));

	GSShard* temp_shards = new GSShard[HostGraph.num_of_shards];

	populate_shards_from_primitive_vertices( 	HostGraph.num_of_shards,
												HostGraph.num_of_vertices,
												HostGraph.windowOffsets,
												HostGraph.shard_sizes,
												blockSize_N.N,
												temp_shards,
												primitiveVertices);

	// Free up primitive vertices.
	for( int i = 0; i < HostGraph.num_of_vertices; ++i )
		delete_primitiveVertex( primitiveVertices+i );
	free( primitiveVertices );

	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.incoming_index), (HostGraph.num_of_edges) * sizeof(unsigned int), cudaHostAllocDefault ) );
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.outgoing_index), (HostGraph.num_of_edges) * sizeof(unsigned int), cudaHostAllocDefault ) );
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.SrcValue), (HostGraph.num_of_edges) * sizeof(Vertex), cudaHostAllocDefault ) );
	if( sizeof(Edge) > 0 ) CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.EdgeValue), (HostGraph.num_of_edges) * sizeof(Edge), cudaHostAllocDefault ) );
	if( sizeof(Vertex_static) ) CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.SrcValue_static), (HostGraph.num_of_edges) * sizeof(Vertex_static), cudaHostAllocDefault ) );

	for( int i = 0; i < HostGraph.num_of_shards; ++i ) {
		unsigned int shard_size = HostGraph.shard_sizes[i+1]-HostGraph.shard_sizes[i];
		memcpy( &(HostGraph.incoming_index[HostGraph.shard_sizes[i]]), temp_shards[i].incoming_index, shard_size * sizeof(unsigned int));
		memcpy( &(HostGraph.outgoing_index[HostGraph.shard_sizes[i]]), temp_shards[i].outgoing_index, shard_size * sizeof(unsigned int));
		memcpy( &(HostGraph.SrcValue[HostGraph.shard_sizes[i]]), temp_shards[i].SrcValue, shard_size * sizeof(Vertex));
		if( sizeof(Vertex_static) > 0 ) memcpy( &(HostGraph.SrcValue_static[HostGraph.shard_sizes[i]]), temp_shards[i].SrcValue_static, shard_size * sizeof(Vertex_static));
		if( sizeof(Edge) > 0 ) memcpy( &(HostGraph.EdgeValue[HostGraph.shard_sizes[i]]), temp_shards[i].EdgeValue, shard_size * sizeof(Edge));
	}
	delete[] temp_shards;

	fprintf( stdout, "Organizing the graph finished.\n");

	if ( processGraphGS(&HostGraph, blockSize_N) == EXIT_FAILURE) {
		fprintf(stderr, "An error happened. Exiting.\n");
		exit(EXIT_FAILURE);
	}

	// Print final values into the file using the user-specified function.
	for ( int i = 0; i < HostGraph.num_of_vertices; ++i )
		printVertexOutputCuSha(i, HostGraph.VertexValues[i], outputFile);

	// Clean up allocated memory regions before going back.
	CUDAErrorCheck ( cudaFreeHost(HostGraph.VertexValues) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.windowOffsets) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.shard_sizes) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.incoming_index) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.outgoing_index) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.SrcValue) );
	if( sizeof(Vertex_static) > 0 ) CUDAErrorCheck ( cudaFreeHost(HostGraph.SrcValue_static) );
	if( sizeof(Edge) > 0 ) CUDAErrorCheck ( cudaFreeHost(HostGraph.EdgeValue) );

}
