#ifndef _CSR_AUX_STRUCTURES_H
#define _CSR_AUX_STRUCTURES_H

#include "csr-UserStructures.h"

// The graph structure that can hold a CSR represented graph.
typedef struct CSRGraph {
	unsigned int num_of_vertices;
	unsigned int num_of_edges;
	unsigned int* edges_indices;
	unsigned int* vertices_indices;
	Vertex_static* VertexValue_static;
	Vertex* VertexValue;
	Edge* EdgeValue;
}CSRGraph;

// Below structure helps us collecting vertex/edge data when initially reading from the file.
typedef struct primitiveVertex {

	unsigned int num_of_nbrs;
	unsigned int* nbrs;
	Edge* EdgeValue;
	Vertex VertexValue;
	Vertex_static VertexValue_static;

	primitiveVertex(){
		num_of_nbrs = 0;
		nbrs = (unsigned int*) malloc ( sizeof(unsigned int) );
		assert(nbrs);
		EdgeValue = (Edge*) malloc ( sizeof(Edge) );
		assert(EdgeValue);
	}

	~primitiveVertex(){
		free(nbrs);
		free(EdgeValue);
	}

}primitiveVertex;

#endif
