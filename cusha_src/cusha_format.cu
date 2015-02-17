#include <iostream>
#include <cmath>
#include <vector>

#include "cusha_format.cuh"
#include "../common/simpleTime.cuh"
#include "../common/cuda_utilities.cuh"
#include "../common/cuda_error_check.cuh"
#include "../common/user_specified_structures.h"
#include "../common/user_specified_pre_and_post_processing_functions.hpp"
#include "find_block_size.cuh"
#include "cusha_process.cuh"
#include "../common/globals.hpp"
#include "cusha_process.cuh"

struct shard_entry{
	Edge edgeVal;
	uint srcIdx;
	uint dstIdx;
};

void cusha_format::process(
		const GraphProcessingMethod procesingMethod,
		const int bsize,
		std::vector<initial_vertex>* initGraph,
		const uint nEdges,
		std::ofstream& outputFile,
		bool EdgesOnHost ) {

	const uint nVerticesInitially = initGraph->size();

	// Variables collecting timing info.
	float H2D_copy_time = 0, processing_time = 0, D2H_copy_time = 0;

	 // Less possible bank conflict when the vertex is big.
	#if __CUDA_ARCH__ >= 300
		if ( sizeof(Vertex) > 4 )
			CUDAErrorCheck( cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte ) );
	#endif

	// Estimate the proper block size.
	const blockSize_N_pair bsizeNPair = find_proper_block_size( bsize, nEdges, nVerticesInitially );
	const uint nShards = std::ceil( (double)nVerticesInitially / bsizeNPair.N );
	const uint nVertices = nShards * bsizeNPair.N;
	std::cout << "Block size would be " << bsizeNPair.blockSize << ".\n";
	std::cout << "The graph is divided into " << nShards << " shards.\n";
	std::cout << ( ( procesingMethod == GS ) ? "G-Shards" : "Concatenated Windows" ) << " will be the processing method.\n";

	// Allocate host buffers.
	host_pinned_buffer<Vertex> vertexValue( nVertices );
	std::vector<Vertex_static> tmpVertexValueStatic;
	if( sizeof(Vertex_static) > 1 ) tmpVertexValueStatic.resize( nVertices );
	std::vector< std::vector<shard_entry> > graphWindows( nShards * nShards, std::vector<shard_entry>( 0 ) );

	// Collecting graph data into shard form.
	for( uint vIdx = 0; vIdx < nVerticesInitially; ++vIdx ) {
		initial_vertex& vvv = initGraph->at(vIdx);
		vertexValue[ vIdx ] = vvv.vertexValue;
		if( sizeof(Vertex_static) > 1 ) tmpVertexValueStatic[ vIdx ] = vvv.VertexValueStatic;
		uint nNbrs = vvv.nbrs.size();
		for( uint nbrIdx = 0; nbrIdx < nNbrs; ++nbrIdx ) {
			neighbor& nbr = vvv.nbrs.at( nbrIdx );
			shard_entry tmpShardEntry;
			tmpShardEntry.dstIdx = vIdx;
			tmpShardEntry.srcIdx = nbr.srcIndex;
			if( sizeof(Edge) > 1 ) tmpShardEntry.edgeVal = nbr.edgeValue;
			uint belongingShardIdx = ( static_cast<unsigned long long>( tmpShardEntry.dstIdx ) * nShards ) / nVertices;
			uint belongingWindowIdx = ( static_cast<unsigned long long>( tmpShardEntry.srcIdx ) * nShards ) / nVertices;
			graphWindows.at( belongingShardIdx * nShards + belongingWindowIdx ).push_back( tmpShardEntry );
		}
	}
	initGraph->clear();
	// no need to sort inside a window.

	// Define and allocate host buffers.
	host_pinned_buffer<Vertex> SrcValue( nEdges );
	host_pinned_buffer<uint> DstIndex( nEdges );
	host_pinned_buffer<Edge> EdgeValues;
	if( sizeof(Edge) > 1 ) EdgeValues.alloc( nEdges );
	host_pinned_buffer<Vertex_static> VertexValuesStatic;
	if( sizeof(Vertex_static) > 1 ) VertexValuesStatic.alloc( nEdges );
	host_pinned_buffer<uint> SrcIndex( nEdges );
	host_pinned_buffer<uint> Mapper;
	if( procesingMethod == CW ) Mapper.alloc( nEdges );
	host_pinned_buffer<uint> windowSizesScansVertical( nShards * nShards + 1 );
	windowSizesScansVertical.at( 0 ) = 0;
	host_pinned_buffer<uint> shardSizesScans( nShards + 1 );
	shardSizesScans.at( 0 ) = 0;
	host_pinned_buffer<uint> concatenatedWindowsSizesScan( nShards + 1 );
	concatenatedWindowsSizesScan.at( 0 ) = 0;

	// Put collected shard-based graph data into host pinned buffers.
	uint movingIdx = 0;
	uint winMovingIdx = 0;
	for( uint shardIdx = 0; shardIdx < nShards; ++shardIdx ) {
		for( uint winIdx = 0; winIdx < nShards; ++winIdx ) {
			std::vector<shard_entry>& window = graphWindows.at( shardIdx * nShards + winIdx );
			for( uint entryIdx = 0; entryIdx < window.size(); ++entryIdx ) {
				SrcValue[ movingIdx ] = vertexValue[ window.at( entryIdx ).srcIdx ];
				DstIndex[ movingIdx ] = window.at( entryIdx ).dstIdx;
				if( sizeof(Edge) > 1 ) EdgeValues[ movingIdx ] = window.at( entryIdx ).edgeVal;
				if( sizeof(Vertex_static) > 1 ) VertexValuesStatic[ movingIdx ] = tmpVertexValueStatic[ window.at( entryIdx ).srcIdx ];
				if( procesingMethod == GS ) SrcIndex[ movingIdx ] = window.at( entryIdx ).srcIdx;
				++movingIdx;
			}
			windowSizesScansVertical[ winMovingIdx + 1 ] = windowSizesScansVertical[ winMovingIdx ] + window.size();
			++winMovingIdx;
		}
		shardSizesScans[ shardIdx + 1 ] = movingIdx;
	}
	tmpVertexValueStatic.clear();
	movingIdx = 0;
	for( uint winIdx = 0; winIdx < nShards; ++winIdx ) {
		for( uint shardIdx = 0; shardIdx < nShards; ++shardIdx ) {
			std::vector<shard_entry>& window = graphWindows.at( shardIdx * nShards + winIdx );
			uint inWinMovingIdx = 0;
			for( uint entryIdx = 0; entryIdx < window.size(); ++entryIdx ) {
				if( procesingMethod == CW ) {
					SrcIndex[ movingIdx ] = window.at( entryIdx ).srcIdx;
					Mapper[ movingIdx ] = windowSizesScansVertical[ shardIdx * nShards + winIdx ] + inWinMovingIdx;
				}
				++inWinMovingIdx;
				++movingIdx;
			}
		}
		concatenatedWindowsSizesScan[ winIdx + 1 ] = movingIdx;
	}
	graphWindows.clear();

	// Define and allocate device buffers.
	device_buffer<Vertex> dev_vertexValue( nVertices );
	device_buffer<Vertex> dev_SrcValue;
	device_buffer<uint> dev_DstIndex;
	device_buffer<Edge> dev_EdgeValues;
	device_buffer<Vertex_static> dev_VertexValuesStatic;
	device_buffer<uint> dev_SrcIndex;
	device_buffer<uint> dev_Mapper;
	device_buffer<uint> dev_concatenatedWindowsSizesScan;
	if( procesingMethod == CW ) dev_concatenatedWindowsSizesScan.alloc( nShards + 1 );
	device_buffer<uint> dev_windowSizesScansVertical;
	if( procesingMethod == GS ) dev_windowSizesScansVertical.alloc( nShards * nShards + 1 );
	device_buffer<uint> dev_shardSizesScans( nShards + 1 );
	device_buffer<int> dev_Finished( 1 );
	if( !EdgesOnHost ) {
		dev_SrcValue.alloc( nEdges );
		dev_DstIndex.alloc( nEdges );
		if( sizeof(Edge) > 1 ) dev_EdgeValues.alloc( nEdges );
		if( sizeof(Vertex_static) > 1 ) dev_VertexValuesStatic.alloc( nEdges );
		dev_SrcIndex.alloc( nEdges );
		if( procesingMethod == CW ) dev_Mapper.alloc( nEdges );
	}

	// Copy data to device buffers.
	setTime();
	dev_vertexValue = vertexValue;
	if( procesingMethod == CW ) dev_concatenatedWindowsSizesScan = concatenatedWindowsSizesScan;
	if( procesingMethod == GS ) dev_windowSizesScansVertical = windowSizesScansVertical;
	dev_shardSizesScans = shardSizesScans;
	if( !EdgesOnHost ) {
		dev_SrcValue = SrcValue;
		dev_DstIndex = DstIndex;
		if( sizeof(Edge) > 1 ) dev_EdgeValues = EdgeValues;
		if( sizeof(Vertex_static) > 1 ) dev_VertexValuesStatic = VertexValuesStatic;
		dev_SrcIndex = SrcIndex;
		if( procesingMethod == CW ) dev_Mapper = Mapper;
	}
	CUDAErrorCheck( cudaDeviceSynchronize() );
	H2D_copy_time = getTime();
	std::cout << "Copying data to device took " << H2D_copy_time << " (ms).\n";

	// Iteratively process the graph.
	int finished;
	unsigned int IterationCounter = 0;
	setTime();
	do {
		finished = 0;

		CUDAErrorCheck( cudaMemcpyAsync( dev_Finished.get_ptr(), &finished, sizeof(int), cudaMemcpyHostToDevice ) );

		cusha_process(
				procesingMethod,
				bsizeNPair.blockSize,
				bsizeNPair.N,
				nShards,
				nVertices,
				dev_vertexValue.get_ptr(),
				dev_concatenatedWindowsSizesScan.get_ptr(),
				dev_windowSizesScansVertical.get_ptr(),
				dev_shardSizesScans.get_ptr(),
				dev_Finished.get_ptr(),
				( !EdgesOnHost ) ? dev_SrcValue.get_ptr() : SrcValue.get_ptr(),
				( !EdgesOnHost ) ? dev_DstIndex.get_ptr() : DstIndex.get_ptr(),
				( !EdgesOnHost ) ? dev_EdgeValues.get_ptr() : EdgeValues.get_ptr(),
				( !EdgesOnHost ) ? dev_VertexValuesStatic.get_ptr() : VertexValuesStatic.get_ptr(),
				( !EdgesOnHost ) ? dev_SrcIndex.get_ptr() : SrcIndex.get_ptr(),
				( !EdgesOnHost ) ? dev_Mapper.get_ptr() : Mapper.get_ptr() );

		CUDAErrorCheck( cudaPeekAtLastError() );
		CUDAErrorCheck( cudaMemcpyAsync( &finished, dev_Finished.get_ptr(), sizeof(int), cudaMemcpyDeviceToHost ) );
		CUDAErrorCheck( cudaDeviceSynchronize() );

		++IterationCounter;
	} while( finished == 1 );
	processing_time = getTime();
	std::cout << "Processing finished in " << processing_time << " (ms).\n";
	std::cout << "Performed " << IterationCounter << " iterations in total.\n";

	// Copy resulted vertex values back from the device to the host.
	setTime();
	CUDAErrorCheck( cudaMemcpy( vertexValue.get_ptr(), dev_vertexValue.get_ptr(), nVerticesInitially * sizeof(Vertex), cudaMemcpyDeviceToHost ) );
	D2H_copy_time = getTime();
	std::cout << "Copying final vertex values back to the host took " << D2H_copy_time << " (ms).\n";

	//std::cout << "Total Execution time was " << H2D_copy_time + processing_time + D2H_copy_time << " (ms).\n";
	//std::cout << IterationCounter <<"\t"<< H2D_copy_time <<"\t"<< processing_time <<"\t"<< D2H_copy_time << "\n";

	// Print the output vertex values to the file.
	for( uint vvv = 0; vvv < nVerticesInitially; ++vvv )
		print_vertex_output(
				vvv,
				vertexValue[ vvv ],
				outputFile	);

}
