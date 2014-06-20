#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../common/simpleTime.h"
#include "../common/CUDAErrorCheck.h"
#include "csr-utils.cuh"
#include "csr-UserFunctions.cuh"

#define JOB_FINISHED 1
#define JOB_NOT_FINISHED_YET 0

#define PHYSICAL_WARP_SIZE 32
#define COMPILE_TIME_DEFINED_BLOCK_SIZE 256

// Virtual Warp-Centric (VWC) manner of processing graph using Compressed Sparse Row (CSR) representation format.
// One can use shuffle instructions to implement this kernel. I used shared memory for data exchange between threads.
template < uint VWSize, uint VWMask >
__global__ void VWC_CSR_GPU_kernel(	const uint num_of_vertices,
									const uint* vertices_indices,
									const uint* edges_indices,
									Vertex* VertexValue,
									Edge* EdgeValue,
									Vertex_static* VertexValue_static,
									int* dev_finished) {

	__shared__ Vertex final_vertex_values[COMPILE_TIME_DEFINED_BLOCK_SIZE >> VWMask];
	__shared__ Vertex thread_outcome[COMPILE_TIME_DEFINED_BLOCK_SIZE];
	volatile __shared__ uint edges_starting_address[COMPILE_TIME_DEFINED_BLOCK_SIZE >> VWMask];
	volatile __shared__ uint ngbrs_size[COMPILE_TIME_DEFINED_BLOCK_SIZE >> VWMask];
	Vertex previous_vertex_value;

	// You might gain some performance if you limit maximum number of registers per thread with -maxrregcount flag. For example, specifying 32 for the Kepler architecture.
	const uint warp_in_block_offset = threadIdx.x >> VWMask;
	const uint VLane_id = threadIdx.x & (VWSize-1);
	const uint t_id = threadIdx.x + blockIdx.x * blockDim.x;
	const uint VW_id = t_id >> VWMask;
	if ( VW_id >= num_of_vertices )
		return;

	previous_vertex_value = VertexValue[VW_id];
	// Only one virtual lane in the virtual warp does vertex initialization.
	if ( VLane_id == 0 ) {
		edges_starting_address [ warp_in_block_offset ] = vertices_indices [ VW_id ];
		ngbrs_size [ warp_in_block_offset ] = vertices_indices [ VW_id + 1 ] - edges_starting_address [ warp_in_block_offset ] ;
		init_compute( final_vertex_values + warp_in_block_offset, &previous_vertex_value );
	}

	for ( uint index = VLane_id; index < ngbrs_size[ warp_in_block_offset ]; index += VWSize ) {

		uint target_edge = edges_starting_address[ warp_in_block_offset ] + index;
		uint target_vertex = edges_indices [ target_edge ];
		compute_local ( 	VertexValue + target_vertex,
							VertexValue_static + target_vertex,
							EdgeValue + target_edge,
							thread_outcome + threadIdx.x,
							&previous_vertex_value );

		// Parallel Reduction. Totally unrolled.
		if ( VWSize == 32 )
			if( VLane_id < 16 )
				if ( (index + 16) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce ( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 16 );
		if ( VWSize >= 16 )
			if( VLane_id < 8 )
				if ( (index + 8) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce ( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 8 );
		if ( VWSize >= 8 )
			if( VLane_id < 4 )
				if ( (index + 4) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce ( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 4 );
		if ( VWSize >= 4 )
			if( VLane_id < 2 )
				if ( (index + 2) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce ( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 2 );
		if ( VWSize >= 2 )
			if( VLane_id < 1 ) {
				if ( (index + 1) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce ( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 1 );
				compute_reduce ( final_vertex_values + warp_in_block_offset, thread_outcome + threadIdx.x );	//	Virtual lane 0 saves the final value of current iteration.
			}

	}

	if ( VLane_id == 0 )
		if ( update_condition ( final_vertex_values + warp_in_block_offset, &previous_vertex_value  ) ) {
			(*dev_finished) = JOB_NOT_FINISHED_YET;
			VertexValue [ VW_id ] = (Vertex) (final_vertex_values [ warp_in_block_offset ]);
		}

}


bool processGraphVWC(	CSRGraph* hostGraph,
						const uint VirtualWarpSize ) {

	// Variables collecting statistics info.
	float H2D_copy_time, processing_time, D2H_copy_time;

	// Getting current device properties to fully occupy SM available threads.
	int currentDevice;
	CUDAErrorCheck ( cudaGetDevice( &currentDevice ) );
	cudaDeviceProp deviceProp;
	CUDAErrorCheck ( cudaGetDeviceProperties(&deviceProp, currentDevice) );

	dim3 blockDim( COMPILE_TIME_DEFINED_BLOCK_SIZE, 1, 1 );

	/*
	// Occupying all SMs with (constant) maximum number of threads to eliminate new block scheduling overhead.
	dim3 gridDim( (deviceProp.multiProcessorCount*deviceProp.maxThreadsPerMultiProcessor)/blockDim.x, 1, 1);
	 */

	dim3 gridDim( ceil(((double)hostGraph->num_of_vertices)/(COMPILE_TIME_DEFINED_BLOCK_SIZE/VirtualWarpSize)), 1, 1 );

	// Host and device flags indicating if the processing is finished.
	int finished;
	int* dev_finished;
	CUDAErrorCheck ( cudaMalloc((void**)&dev_finished, sizeof(int)) );

	// Creation and memory allocation for the graph in device global memory.
	CSRGraph dev_Graph;
	dev_Graph.num_of_edges = hostGraph->num_of_edges;
	dev_Graph.num_of_vertices = hostGraph->num_of_vertices;
	CUDAErrorCheck ( cudaMalloc((void**)&(dev_Graph.vertices_indices), ( dev_Graph.num_of_vertices + 1 ) * sizeof(unsigned int)) );
	CUDAErrorCheck ( cudaMalloc((void**)&(dev_Graph.edges_indices), ( dev_Graph.num_of_edges ) * sizeof(unsigned int)) );
	CUDAErrorCheck ( cudaMalloc((void**)&(dev_Graph.VertexValue), ( dev_Graph.num_of_vertices ) * sizeof(Vertex)) );
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaMalloc((void**)&(dev_Graph.EdgeValue), ( dev_Graph.num_of_edges ) * sizeof(Edge)) );
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaMalloc((void**)&(dev_Graph.VertexValue_static), ( dev_Graph.num_of_vertices ) * sizeof(Vertex_static)) );

	// Copy the graph from the host to the device
	fprintf ( stdout, "Copying the graph from the host to the device ...\n" );
	setTime();

	CUDAErrorCheck ( cudaMemcpyAsync( dev_Graph.vertices_indices, hostGraph->vertices_indices, ( dev_Graph.num_of_vertices + 1 ) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDAErrorCheck ( cudaMemcpyAsync( dev_Graph.edges_indices, hostGraph->edges_indices, ( dev_Graph.num_of_edges ) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDAErrorCheck ( cudaMemcpyAsync( dev_Graph.VertexValue, hostGraph->VertexValue, ( dev_Graph.num_of_vertices ) * sizeof(Vertex), cudaMemcpyHostToDevice));
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaMemcpyAsync( dev_Graph.EdgeValue, hostGraph->EdgeValue, ( dev_Graph.num_of_edges ) * sizeof(Edge), cudaMemcpyHostToDevice));
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaMemcpyAsync( dev_Graph.VertexValue_static, hostGraph->VertexValue_static, ( dev_Graph.num_of_vertices ) * sizeof(Vertex_static), cudaMemcpyHostToDevice));
	CUDAErrorCheck ( cudaDeviceSynchronize() );

	H2D_copy_time = getTime();
	fprintf( stdout, "Copying the graph from the host to the device finished in: %f (ms)\n", H2D_copy_time );

	// Iteratively process the graph on the device.
	fprintf( stdout, "Processing graph in a virtual warp-centric manner using CSR representation ...\n" );
	unsigned int counter = 0;
	setTime();
	do {
		finished = JOB_FINISHED;
		CUDAErrorCheck ( cudaMemcpyAsync ( dev_finished, &finished, sizeof(char), cudaMemcpyHostToDevice ) );
		switch( VirtualWarpSize ) {
		case(32):
			VWC_CSR_GPU_kernel <32,5> <<< gridDim, blockDim >>> (	dev_Graph.num_of_vertices,
																	dev_Graph.vertices_indices,
																	dev_Graph.edges_indices,
																	dev_Graph.VertexValue,
																	dev_Graph.EdgeValue,
																	dev_Graph.VertexValue_static,
																	dev_finished );
			break;
		case(16):
			VWC_CSR_GPU_kernel <16,4> <<< gridDim, blockDim >>> (	dev_Graph.num_of_vertices,
																	dev_Graph.vertices_indices,
																	dev_Graph.edges_indices,
																	dev_Graph.VertexValue,
																	dev_Graph.EdgeValue,
																	dev_Graph.VertexValue_static,
																	dev_finished );
			break;
		case(8):
			VWC_CSR_GPU_kernel <8,3> <<< gridDim, blockDim >>> (	dev_Graph.num_of_vertices,
																	dev_Graph.vertices_indices,
																	dev_Graph.edges_indices,
																	dev_Graph.VertexValue,
																	dev_Graph.EdgeValue,
																	dev_Graph.VertexValue_static,
																	dev_finished );
			break;
		case(4):
			VWC_CSR_GPU_kernel <4,2> <<< gridDim, blockDim >>> (	dev_Graph.num_of_vertices,
																	dev_Graph.vertices_indices,
																	dev_Graph.edges_indices,
																	dev_Graph.VertexValue,
																	dev_Graph.EdgeValue,
																	dev_Graph.VertexValue_static,
																	dev_finished );
			break;
		case(2):
			VWC_CSR_GPU_kernel <2,1> <<< gridDim, blockDim >>> (	dev_Graph.num_of_vertices,
																	dev_Graph.vertices_indices,
																	dev_Graph.edges_indices,
																	dev_Graph.VertexValue,
																	dev_Graph.EdgeValue,
																	dev_Graph.VertexValue_static,
																	dev_finished );
			break;

		}

		CUDAErrorCheck ( cudaPeekAtLastError() );
		CUDAErrorCheck ( cudaMemcpy ( &finished, dev_finished, sizeof(char), cudaMemcpyDeviceToHost ) );
		counter++;
	} while ( finished == JOB_NOT_FINISHED_YET );
	processing_time = getTime();
	fprintf( stdout, "Processing finished in: %f (ms)\n", processing_time);
	fprintf( stdout, "Performed %u iterations in total.\n", counter);

	// Copy resulted vertex values back from the device to the host.
	fprintf( stdout, "Copying final vertex values from the device to the host ...\n" );
	setTime();
	CUDAErrorCheck ( cudaMemcpy( hostGraph->VertexValue, dev_Graph.VertexValue, ( dev_Graph.num_of_vertices ) * sizeof(Vertex), cudaMemcpyDeviceToHost));
	D2H_copy_time = getTime();
	fprintf( stdout, "Copying final vertex values back from the device to the host finished in: %f (ms)\n", D2H_copy_time);
	fprintf( stdout, "Total Execution time was: %f (ms)\n", H2D_copy_time+processing_time+D2H_copy_time );

	// Free up allocated device memory.
	CUDAErrorCheck ( cudaFree( dev_finished ) );
	CUDAErrorCheck ( cudaFree( dev_Graph.vertices_indices ) );
	CUDAErrorCheck ( cudaFree( dev_Graph.edges_indices ) );
	CUDAErrorCheck ( cudaFree( dev_Graph.VertexValue ) );
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaFree( dev_Graph.EdgeValue ) );
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaFree( dev_Graph.VertexValue_static ) );

	return(EXIT_SUCCESS);
}

void ExecuteVWC(	FILE* inputFile,
					const int suggestedVirtualWarpSize,
					FILE* outputFile,
					const int arbparam) {

	CSRGraph HostGraph;
	primitiveVertex* primitiveVertices = (primitiveVertex*) malloc ( sizeof(primitiveVertex));
	assert( primitiveVertices );
	init_primitiveVertexCSR ( primitiveVertices );
	fprintf( stdout, "Populating the graph ...\n" );
	populatePrimitiveVertices( &HostGraph, &primitiveVertices, inputFile, arbparam);

	if(HostGraph.num_of_edges == 0) {
		fprintf( stderr, "No edge could be read from the file. Make sure provided formatting matches defined specification.\n");
		exit(EXIT_FAILURE);
	}
	else {
		fprintf( stdout, "Graph is populated with %u vertices and %u edges.\n", HostGraph.num_of_vertices, HostGraph.num_of_edges );
	}

	// Allocate using page-locked host memory. Graph size is limited by the size of system DRAM.
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.vertices_indices), (HostGraph.num_of_vertices+1) * sizeof(unsigned int), cudaHostAllocDefault ) );
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.edges_indices), HostGraph.num_of_edges * sizeof(unsigned int), cudaHostAllocDefault ) );
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.VertexValue), HostGraph.num_of_vertices * sizeof(Vertex), cudaHostAllocDefault ) );
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.EdgeValue), HostGraph.num_of_edges * sizeof(Edge), cudaHostAllocDefault ) );
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.VertexValue_static), HostGraph.num_of_vertices * sizeof(Vertex_static), cudaHostAllocDefault ) );

	copyPrimitiveVerticesIntoCSRGraph(&HostGraph, &primitiveVertices);

	// Free up primitive vertices.
	for( int i = 0; i < HostGraph.num_of_vertices; ++i )
		delete_primitiveVertexCSR( primitiveVertices+i );
	free( primitiveVertices );

	unsigned int VirtualWarpSize = PHYSICAL_WARP_SIZE;

	if( (suggestedVirtualWarpSize % 2 == 0) && (suggestedVirtualWarpSize != 0) && suggestedVirtualWarpSize <= PHYSICAL_WARP_SIZE ) {
		VirtualWarpSize = suggestedVirtualWarpSize;
		fprintf( stdout, "Chosen virtual warp size: %u\n", VirtualWarpSize);
	}
	else
		fprintf( stdout, "Chosen virtual warp size is not 2, 4, 8, 16, or 32. Default virtual warp size %u is used.\n", VirtualWarpSize);

	if ( processGraphVWC(&HostGraph, VirtualWarpSize) == EXIT_FAILURE) {
		fprintf(stderr, "An error happened. Exiting.\n");
		exit(EXIT_FAILURE);
	}

	for ( int i = 0; i < HostGraph.num_of_vertices; ++i )
		printVertexOutput(i, HostGraph.VertexValue[i], outputFile);

	CUDAErrorCheck ( cudaFreeHost(HostGraph.vertices_indices) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.edges_indices) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.VertexValue) );
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaFreeHost(HostGraph.EdgeValue) );
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaFreeHost(HostGraph.VertexValue_static) );

}
