#include <cmath>
#include <iostream>
#include <pthread.h>

#include "../common/simpleTime.cuh"
#include "csr_format.cuh"
#include "../common/cuda_error_check.cuh"
#include "../common/cuda_utilities.cuh"
#include "vwc_process.cuh"
#include "../common/user_specified_structures.h"
#include "../common/user_specified_pre_and_post_processing_functions.hpp"


void csr_format::process(
		const int vwsize_or_threads,
		std::vector<initial_vertex>* initGraph,
		const uint nEdges,
		std::ofstream& outputFile,
		bool EdgesOnHost ) {

	const uint nVertices = initGraph->size();

	// Variables collecting timing info.
	float H2D_copy_time = 0, processing_time = 0, D2H_copy_time = 0;

	// Allocate host buffers.
	host_pinned_buffer<Vertex> vertexValue( nVertices );
	host_pinned_buffer<uint> edgesIndices( nVertices + 1 );
	edgesIndices.at(0) = 0;
	host_pinned_buffer<uint> vertexIndices( nEdges );
	host_pinned_buffer<Edge> EdgeValue;
	if( sizeof(Edge) > 1 ) EdgeValue.alloc( nEdges );
	host_pinned_buffer<Vertex_static> VertexValueStatic;
	if( sizeof(Vertex_static) > 1 ) VertexValueStatic.alloc( nVertices );

	// Put vertices into host buffer CSR form.
	for( uint vIdx = 0; vIdx < nVertices; ++vIdx ) {
		initial_vertex& vvv = initGraph->at(vIdx);
		vertexValue[ vIdx ] = vvv.vertexValue;
		if( sizeof(Vertex_static) > 1 ) VertexValueStatic[ vIdx ] = vvv.VertexValueStatic;
		uint nNbrs = vvv.nbrs.size();
		uint edgeIdxOffset = edgesIndices[ vIdx ];
		for( uint nbrIdx = 0; nbrIdx < nNbrs; ++nbrIdx ) {
			neighbor& nbr = vvv.nbrs.at( nbrIdx );
			vertexIndices[ edgeIdxOffset + nbrIdx ] = nbr.srcIndex;
			if( sizeof(Edge) > 1 ) EdgeValue[ edgeIdxOffset + nbrIdx ] = nbr.edgeValue;
		}
		edgesIndices[ vIdx + 1 ] = edgeIdxOffset + nNbrs;
	}

	// Define device buffers.
	device_buffer<Vertex> dev_vertexValue;
	device_buffer<uint> dev_edgesIndices;
	device_buffer<uint> dev_vertexIndices;
	device_buffer<Edge> dev_EdgeValue;
	device_buffer<Vertex_static> dev_VertexValueStatic;
	device_buffer<int> devFinished;

	uint vwcGridDimen = 0;

	vwcGridDimen = std::ceil( static_cast<float>( nVertices ) / ( VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE / vwsize_or_threads ) );

	// Allocate device buffers.
	dev_vertexValue.alloc( nVertices );
	dev_edgesIndices.alloc( nVertices + 1 );
	if( !EdgesOnHost ) dev_vertexIndices.alloc( nEdges );
	if( !EdgesOnHost ) if( sizeof(Edge) > 1 ) dev_EdgeValue.alloc( nEdges );
	if( sizeof(Vertex_static) > 1 ) dev_VertexValueStatic.alloc( nVertices );
	devFinished.alloc( 1 );

	// Copy data to device buffers.
	setTime();
	dev_vertexValue = vertexValue;
	dev_edgesIndices = edgesIndices;
	if( !EdgesOnHost ) dev_vertexIndices = vertexIndices;
	if( !EdgesOnHost ) if( sizeof(Edge) > 1 ) dev_EdgeValue = EdgeValue;
	if( sizeof(Vertex_static) > 1 ) dev_VertexValueStatic = VertexValueStatic;
	CUDAErrorCheck( cudaDeviceSynchronize() );
	H2D_copy_time = getTime();
	std::cout << "Copying data to device took " << H2D_copy_time << " (ms)." << std::endl;

	int finished;

	// Iteratively process the graph.
	unsigned int IterationCounter = 0;
	setTime();
	do {
		finished = 0;

		CUDAErrorCheck( cudaMemcpyAsync( devFinished.get_ptr(), &finished, sizeof(int), cudaMemcpyHostToDevice ) );
		vwc_process(
				vwsize_or_threads,
				vwcGridDimen,
				nVertices,
				( !EdgesOnHost ) ? dev_vertexIndices.get_ptr() : vertexIndices.get_ptr(),
				dev_edgesIndices.get_ptr(),
				dev_vertexValue.get_ptr(),
				( !EdgesOnHost ) ? dev_EdgeValue.get_ptr() : EdgeValue.get_ptr(),
				dev_VertexValueStatic.get_ptr(),
				devFinished.get_ptr() );
		CUDAErrorCheck( cudaPeekAtLastError() );
		CUDAErrorCheck( cudaMemcpyAsync( &finished, devFinished.get_ptr(), sizeof(int), cudaMemcpyDeviceToHost ) );
		CUDAErrorCheck( cudaDeviceSynchronize() );

		++IterationCounter;
	} while( finished == 1 );
	processing_time = getTime();
	std::cout << "Processing finished in " << processing_time << " (ms).\n";
	std::cout << "Performed " << IterationCounter << " iterations in total.\n";

	// Copy resulted vertex values back from the device to the host.
	setTime();
	CUDAErrorCheck( cudaMemcpy( vertexValue.get_ptr(), dev_vertexValue.get_ptr(), vertexValue.sizeInBytes(), cudaMemcpyDeviceToHost ) );
	D2H_copy_time = getTime();
	std::cout << "Copying final vertex values back to the host took " << D2H_copy_time << " (ms).\n";

	//std::cout << "Total Execution time was " << H2D_copy_time + processing_time + D2H_copy_time << " (ms)." << std::endl;
	//std::cout << IterationCounter <<"\t"<< H2D_copy_time <<"\t"<< processing_time <<"\t"<< D2H_copy_time << "\n";

	// Print the output vertex values to the file.
	for( uint vvv = 0; vvv < nVertices; ++vvv )
		print_vertex_output(
				vvv,
				vertexValue[ vvv ],
				outputFile	);

}
