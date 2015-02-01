
#include "vwc_process.cuh"
#include "../common/user_specified_device_functions.cuh"

// Virtual Warp-Centric (VWC) manner of processing graph using Compressed Sparse Row (CSR) representation format.
template < uint VWSize, uint VWMask >
__global__ void VWC_CSR_GPU_kernel(
		const uint num_of_vertices,
		const uint* edges_indices,
		const uint* vertices_indices,
		Vertex* VertexValue,
		Edge* EdgeValue,
		Vertex_static* VertexValue_static,
		int* dev_finished ) {

	__shared__ Vertex final_vertex_values[ VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE >> VWMask ];
	__shared__ Vertex thread_outcome[ VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE ];
	volatile __shared__ uint edges_starting_address[ VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE >> VWMask ];
	volatile __shared__ uint ngbrs_size[ VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE >> VWMask ];
	Vertex previous_vertex_value;

	// You might gain some performance if you limit maximum number of registers per thread with -maxrregcount flag. For example, specifying 32 for the Kepler architecture.
	const uint warp_in_block_offset = threadIdx.x >> VWMask;
	const uint VLane_id = threadIdx.x & (VWSize-1);
	const uint t_id = threadIdx.x + blockIdx.x * blockDim.x;
	const uint VW_id = t_id >> VWMask;
	if( VW_id >= num_of_vertices )
		return;

	previous_vertex_value = VertexValue[ VW_id ];
	// Only one virtual lane in the virtual warp does vertex initialization.
	if( VLane_id == 0 ) {
		edges_starting_address[ warp_in_block_offset ] = vertices_indices[ VW_id ];
		ngbrs_size[ warp_in_block_offset ] = vertices_indices[ VW_id + 1 ] - edges_starting_address[ warp_in_block_offset ] ;
		init_compute( final_vertex_values + warp_in_block_offset, &previous_vertex_value );
	}

	for( uint index = VLane_id; index < ngbrs_size[ warp_in_block_offset ]; index += VWSize ) {

		uint target_edge = edges_starting_address[ warp_in_block_offset ] + index;
		uint target_vertex = edges_indices[ target_edge ];
		compute_local(
				VertexValue[target_vertex],
				VertexValue_static + target_vertex,
				EdgeValue + target_edge,
				thread_outcome + threadIdx.x,
				final_vertex_values + warp_in_block_offset );

		// Parallel Reduction. Totally unrolled.
		if( VWSize == 32 )
			if( VLane_id < 16 )
				if( (index + 16) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 16 );
		if( VWSize >= 16 )
			if( VLane_id < 8 )
				if( (index + 8) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 8 );
		if( VWSize >= 8 )
			if( VLane_id < 4 )
				if( (index + 4) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 4 );
		if( VWSize >= 4 )
			if( VLane_id < 2 )
				if( (index + 2) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 2 );
		if( VWSize >= 2 )
			if( VLane_id < 1 ) {
				if( (index + 1) < ngbrs_size[ warp_in_block_offset ])
					compute_reduce( thread_outcome + threadIdx.x, thread_outcome + threadIdx.x + 1 );
				compute_reduce( final_vertex_values + warp_in_block_offset, thread_outcome + threadIdx.x );	//	Virtual lane 0 saves the final value of current iteration.
			}

	}

	if( VLane_id == 0 )
		if( update_condition ( final_vertex_values + warp_in_block_offset, &previous_vertex_value  ) ) {
			(*dev_finished) = 1;
			VertexValue[ VW_id ] = final_vertex_values[ warp_in_block_offset ];
		}

}

void vwc_process(
		int vwSize,
		uint gridDimen,
		const uint nVertices,
		const uint* vertexIndices,
		const uint* edgesIndices,
		Vertex* VertexValue,
		Edge* EdgeValue,
		Vertex_static* VertexValueStatic,
		int* finished ) {

	switch( vwSize ) {
		case(32):
			VWC_CSR_GPU_kernel< 32, 5 >
					<<< gridDimen, VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE >>> (
								nVertices,
								vertexIndices,
								edgesIndices,
								VertexValue,
								EdgeValue,
								VertexValueStatic,
								finished );
			break;
		case(16):
			VWC_CSR_GPU_kernel< 16, 4 >
					<<< gridDimen, VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE >>> (
								nVertices,
								vertexIndices,
								edgesIndices,
								VertexValue,
								EdgeValue,
								VertexValueStatic,
								finished );
			break;
		case(8):
			VWC_CSR_GPU_kernel< 8, 3 >
					<<< gridDimen, VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE >>> (
								nVertices,
								vertexIndices,
								edgesIndices,
								VertexValue,
								EdgeValue,
								VertexValueStatic,
								finished );
			break;
		case(4):
			VWC_CSR_GPU_kernel< 4, 2 >
					<<< gridDimen, VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE >>> (
								nVertices,
								vertexIndices,
								edgesIndices,
								VertexValue,
								EdgeValue,
								VertexValueStatic,
								finished );
			break;
		case(2):
			VWC_CSR_GPU_kernel< 2, 1 >
					<<< gridDimen, VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE >>> (
								nVertices,
								vertexIndices,
								edgesIndices,
								VertexValue,
								EdgeValue,
								VertexValueStatic,
								finished );
			break;

	}

}
