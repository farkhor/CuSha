#include <stdio.h>
#include <assert.h>
#include "cusha-AuxStructures.h"
#include "cusha-UserFunctions.cuh"
#include "../common/CUDAErrorCheck.h"

// This function initializes primitive vertex data.
void init_primitiveVertex( primitiveVertex* prim_vertex ) {
	prim_vertex->num_of_nbrs = 0;
	prim_vertex->nbrs = (unsigned int*) malloc ( sizeof(unsigned int) );
	assert(prim_vertex->nbrs);
	prim_vertex->EdgeValue = (Edge*) malloc ( sizeof(Edge) );
	if(sizeof(Edge) > 0)
		assert(prim_vertex->EdgeValue);
}

// This function deletes primitive vertex data.
void delete_primitiveVertex( primitiveVertex* prim_vertex ) {
	free ( prim_vertex->nbrs );
	if(sizeof(Edge) > 0)
		free ( prim_vertex->EdgeValue );
}

// This function reads input file line by line, and stores them in form of primitiveVertex.
void populatePrimitiveVerticesForCuSha(	unsigned int* num_edges,	// Input CSR graph number of edges address that need to be initialized by this function.
										unsigned int* num_vertices,	// Input CSR graph number of vertices address that need to be initialized by this function.
										primitiveVertex** primitiveVertices,	// PrimitiveVertex* pointer that will be filled in.
										FILE* inFile,	// Input file.
										const int inParam	){	// Integer command-line input parameter.

	void* helper_pointer;

	const unsigned int MAX_LINE_SIZE = 256;
	const unsigned int MAX_NUMBER_OF_ADDITIONAL_INPUT_ARGUMENTS = 61;
	char theLine[MAX_LINE_SIZE];
	char*	pch;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	int first_index, second_index;

	unsigned int Additionalargc=0;
	char** Additionalargv;
	helper_pointer = malloc(MAX_NUMBER_OF_ADDITIONAL_INPUT_ARGUMENTS*sizeof(char*));
	assert( helper_pointer );
	Additionalargv = (char**) helper_pointer;

	unsigned int num_of_vertices = 0, num_of_edges = 0, theMax;

	// Read till the end of file
	while ( fgets( theLine, MAX_LINE_SIZE, inFile ) ) {
		if ( theLine[0] < '0' || theLine[0] > '9')	// Skipping any line blank or starting with a character rather than a number.
			continue;

		// At least two elements are required to have incoming and outgoing indices for the edge. Otherwise skip the line.
		pch = strtok (theLine, delim);
		if( pch != NULL )
			first_index = atoi ( pch );
		else
			continue;
		pch = strtok (NULL, delim);
		if( pch != NULL )
			second_index = atoi ( pch );
		else
			continue;

		theMax = max (first_index, second_index);
		if ( theMax >= num_of_vertices ) {
			helper_pointer = realloc ( *primitiveVertices, (theMax+1)*sizeof(primitiveVertex) );
			assert( helper_pointer );
			(*primitiveVertices) = (primitiveVertex*) helper_pointer;
			for ( unsigned int i = num_of_vertices; i <= theMax; ++i )
				init_primitiveVertex ( &((*primitiveVertices)[i]) );
			num_of_vertices = theMax + 1;
		}

		//reallocation for every 32 neighbors. Nothing special about 32. Tradeoff between lots of reallocations (that slow down the program) and using lots of memory.
		if ( (*primitiveVertices)[first_index].num_of_nbrs % 32 == 0) {
			helper_pointer = realloc ( (*primitiveVertices)[first_index].nbrs, ((*primitiveVertices)[first_index].num_of_nbrs + 32)*sizeof(unsigned int) );
			assert( helper_pointer );
			(*primitiveVertices)[first_index].nbrs = (unsigned int*) helper_pointer;
			helper_pointer = realloc ( (*primitiveVertices)[first_index].EdgeValue, ((*primitiveVertices)[first_index].num_of_nbrs + 32)*sizeof(Edge) );
			if(sizeof(Edge) > 0)
				assert( helper_pointer );
			(*primitiveVertices)[first_index].EdgeValue = (Edge*) helper_pointer;
		}
		(*primitiveVertices)[first_index].nbrs [ (*primitiveVertices)[first_index].num_of_nbrs ] = second_index;
		(*primitiveVertices)[first_index].num_of_nbrs++;

		Additionalargc=0;
		Additionalargv[Additionalargc] = strtok (NULL, delim);
		while( Additionalargv[Additionalargc] != NULL ){
			Additionalargc++;
			Additionalargv[Additionalargc] = strtok (NULL, delim);
		}
		completeEntryCuSha(	Additionalargc,
							Additionalargv,
							first_index,
							second_index,
							((*primitiveVertices)[first_index].EdgeValue)+((*primitiveVertices)[first_index].num_of_nbrs),
							&((*primitiveVertices)[first_index].VertexValue),
							&((*primitiveVertices)[first_index].VertexValue_static),
							&((*primitiveVertices)[second_index].VertexValue),
							&((*primitiveVertices)[second_index].VertexValue_static),
							inParam	);

		num_of_edges++;

	}

	(*num_vertices) = num_of_vertices;
	(*num_edges) = num_of_edges;

	free( Additionalargv );

}

// This function does NOT guarantee the best block size. But it tries to come up with the best.
// Be aware block sizes rather than what this function chooses might end up showing better performance.
// Any suggestions to improve this function will be appreciated.
blockSize_N_pair findProperBlockSize (	const int suggestedBlockSize,
										const unsigned num_of_edges,
										const unsigned num_of_vertices,
										const unsigned int Vertex_size ) {

	#if __CUDA_ARCH__ >= 300
		if ( Vertex_size > 4 ) // Less possible bank conflict when the vertex is big.
			CUDAErrorCheck( cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte ) );
	#endif

	/*
	#if __CUDA_ARCH__ >= 200
		CUDAErrorCheck(  cudaDeviceSetCacheConfig( cudaFuncCachePreferShared ) );
	#endif
	*/

	// Getting current device properties to properly select block size and N.
	int currentDevice;
	CUDAErrorCheck ( cudaGetDevice( &currentDevice ) );
	cudaDeviceProp deviceProp;
	CUDAErrorCheck ( cudaGetDeviceProperties(&deviceProp, currentDevice) );
	int max_vertices_in_SM = deviceProp.sharedMemPerBlock / Vertex_size;

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

	// If suggested block size is 1 (user hasn't entered anything), we ignore it.
	blockSize_N_pair BS_N;
	if( suggestedBlockSize == 1 ) {
		//int difference = 1<<29;	// Very large number.
		int approximated_N = (int)sqrt((deviceProp.warpSize * pow(num_of_vertices,2))/num_of_edges);	// Please refer to paper for explanation.
		//fprintf( stdout, "Approximated N: %d\n", approximated_N);
		for( int b_per_SM = 2; b_per_SM<=MaxBlockPerSM; b_per_SM++ ) {
			blockSize_N_pair temp_pair;
			temp_pair.blockSize = deviceProp.maxThreadsPerMultiProcessor/b_per_SM;
			if ( deviceProp.maxThreadsPerMultiProcessor % (temp_pair.blockSize * b_per_SM) != 0 )
				continue;
			if( temp_pair.blockSize > deviceProp.maxThreadsPerBlock)
				continue;
			temp_pair.N = max_vertices_in_SM / b_per_SM;
			//if( abs(approximated_N-(int)test_pair.N) < difference) {
			if( temp_pair.N > approximated_N ) {
				//difference = abs(approximated_N - (int)test_pair.N);
				BS_N = temp_pair;
			}
		}

	}
	else {
		// The behavior is undefined if user-specified block size is not a power of two. Usual block sizes are 1024, 512, 256, and 128.
		assert( suggestedBlockSize <= deviceProp.maxThreadsPerBlock );
		BS_N.blockSize = suggestedBlockSize;
		BS_N.N = (max_vertices_in_SM * BS_N.blockSize) / deviceProp.maxThreadsPerMultiProcessor;
	}

	return BS_N;
}

// Below function fills up temporary shards from primitive vertices.
// Temporary shards will be used to fill up graph elements.
void populate_shards_from_primitive_vertices(	const unsigned int num_of_shards,
												const unsigned int num_of_vertices,
												unsigned int* window_offsets,
												unsigned int* shard_sizes,
												const unsigned int N,
												GSShard* temp_shards,
												const primitiveVertex* primitiveVertices	) {


	void* helper_pointer;

	// For all the neighbors of all vertices
	for( unsigned int vertex_count = 0; vertex_count < num_of_vertices; ++vertex_count ) {

		for( unsigned int outgoing_nbr_count = 0; outgoing_nbr_count<primitiveVertices[vertex_count].num_of_nbrs; ++outgoing_nbr_count ) {

			unsigned int target_shard = primitiveVertices[vertex_count].nbrs[outgoing_nbr_count] / N;
			assert(target_shard < num_of_shards);

			if (shard_sizes[target_shard] % 1024 == 0) {	// Nothing special about 1024. Trade-off between lots of allocations and temporary memory usage.
				helper_pointer = realloc(temp_shards[target_shard].incoming_index, (shard_sizes[target_shard]+1024)*sizeof(unsigned int) );
				assert(helper_pointer);
				temp_shards[target_shard].incoming_index = (unsigned int*) helper_pointer;
				helper_pointer = realloc(temp_shards[target_shard].outgoing_index, (shard_sizes[target_shard]+1024)*sizeof(unsigned int) );
				assert(helper_pointer);
				temp_shards[target_shard].outgoing_index = (unsigned int*) helper_pointer;
				helper_pointer = realloc(temp_shards[target_shard].SrcValue, (shard_sizes[target_shard]+1024)*sizeof(Vertex) );
				assert(helper_pointer);
				temp_shards[target_shard].SrcValue = (Vertex*) helper_pointer;
				if( sizeof(Edge) > 0) {
					helper_pointer = realloc(temp_shards[target_shard].EdgeValue, (shard_sizes[target_shard]+1024)*sizeof(Edge) );
					assert(helper_pointer);
					temp_shards[target_shard].EdgeValue = (Edge*) helper_pointer;
				}
				if( sizeof(Vertex_static) > 0) {
					helper_pointer = realloc(temp_shards[target_shard].SrcValue_static, (shard_sizes[target_shard]+1024)*sizeof(Vertex_static) );
					assert(helper_pointer);
					temp_shards[target_shard].SrcValue_static = (Vertex_static*) helper_pointer;
				}
			}

			temp_shards[target_shard].incoming_index[ shard_sizes[target_shard] ] = vertex_count;
			temp_shards[target_shard].outgoing_index[ shard_sizes[target_shard] ] = primitiveVertices[vertex_count].nbrs[outgoing_nbr_count];
			temp_shards[target_shard].SrcValue[ shard_sizes[target_shard] ] = primitiveVertices[vertex_count].VertexValue;
			if( sizeof(Vertex_static) > 0 )
				temp_shards[target_shard].SrcValue_static[ shard_sizes[target_shard] ] = primitiveVertices[vertex_count].VertexValue_static;
			if( sizeof(Edge) > 0 )
				temp_shards[target_shard].EdgeValue[ shard_sizes[target_shard] ] = primitiveVertices[vertex_count].EdgeValue[outgoing_nbr_count];

			shard_sizes[target_shard]++;

			window_offsets[ num_of_shards*target_shard + (vertex_count/N) +1]++;

		}
	}

	// We keep prefix sum of shard sizes in shard_sizes array. It makes it easier for GPU to access proper addresses.
	for(unsigned int target_shard = 1; target_shard<num_of_shards; ++target_shard ) {
		shard_sizes[target_shard] += shard_sizes[target_shard-1];
	}
	for(unsigned int target_shard = num_of_shards; target_shard>=1; --target_shard ) {
		shard_sizes[target_shard] = shard_sizes[target_shard-1];
	}
	shard_sizes[0] = 0;

	// We keep prefix sum of window offsets.
	window_offsets[0] = 0;
	for(unsigned int window = 1; window<(num_of_shards*num_of_shards+1); ++window ) {
		window_offsets[window] += window_offsets[window-1];
	}


}
