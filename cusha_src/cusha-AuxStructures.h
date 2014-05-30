#ifndef _CUSHA_AUX_STRUCTURES_H
#define _CUSHA_AUX_STRUCTURES_H

#include "cusha-UserStructures.h"

#define JOB_FINISHED 1
#define JOB_NOT_FINISHED_YET 0

// A shard structure.
typedef struct GSShard{

	unsigned int* incoming_index;
	unsigned int* outgoing_index;
	Vertex* SrcValue;
	Edge* EdgeValue;
	Vertex_static* SrcValue_static;

	GSShard(){
		incoming_index = (unsigned int*) malloc ( sizeof(unsigned int) );
		assert(incoming_index);
		outgoing_index = (unsigned int*) malloc ( sizeof(unsigned int) );
		assert(outgoing_index);
		SrcValue = (Vertex*) malloc ( sizeof(Vertex) );
		assert(SrcValue);
		if( sizeof(Edge) > 0 ) {
			EdgeValue = (Edge*) malloc ( sizeof(Edge) );
			assert(EdgeValue);
		}
		if( sizeof(Vertex_static) > 0 ) {
			SrcValue_static = (Vertex_static*) malloc ( sizeof(Vertex_static) );
			assert(SrcValue_static);
		}
	}

	~GSShard(){
		free ( incoming_index );
		free ( outgoing_index );
		free ( SrcValue );
		if( sizeof(Edge) > 0 ) free ( EdgeValue );
		if( sizeof(Vertex_static) > 0 )free ( SrcValue_static );
	}

}GSShard;

// A graph structure for G-Shards representation.
typedef struct GSGraph{

	unsigned int* incoming_index;
	unsigned int* outgoing_index;
	Vertex* SrcValue;
	Edge* EdgeValue;
	Vertex_static* SrcValue_static;
	Vertex* VertexValues;
	unsigned int* shard_sizes;
	unsigned int* windowOffsets;

	unsigned int num_of_vertices;
	unsigned int num_of_edges;
	unsigned int num_of_shards;

}GSGraph;

// A graph structure for Concatenated Windows (CW) representation.
typedef struct CWGraph{

	unsigned int* incoming_index;
	unsigned int* mapper;
	unsigned int* outgoing_index;
	Vertex* SrcValue;
	Edge* EdgeValue;
	Vertex_static* SrcValue_static;
	Vertex* VertexValues;
	unsigned int* shard_sizes;
	unsigned int* shard_concatenated_windows_sizes;

	unsigned int num_of_vertices;
	unsigned int num_of_edges;
	unsigned int num_of_shards;

}CWGraph;

// Below structure helps us collecting vertex/edge data when initially reading from the file.
typedef struct primitiveVertex {

	unsigned int num_of_nbrs;
	unsigned int* nbrs;
	Edge* EdgeValue;
	Vertex VertexValue;
	Vertex_static VertexValue_static;

}primitiveVertex;

// The structure holding the block size for GPU kernel calls and the appropriate number of vertices for it.
// This can help us utilizing all resources and having 100% theoretical achieved occupancy.
typedef struct blockSize_N_pair{

	unsigned int blockSize;
	unsigned int N;
	blockSize_N_pair(){
		blockSize = 0;
		N = 0;
	}

}blockSize_N_pair;

#endif
