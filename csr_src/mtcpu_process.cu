#include <cmath>
#include <iostream>
#include <pthread.h>

#include "mtcpu_process.cuh"
#include "../common/user_specified_device_functions.cuh"


// Each thread executes this function.
void* ACPUThreadJob ( void *arguments ) {

	// Thread gets arguments
	arg_struct* args = (arg_struct*)arguments;
	Vertex* VertexValues = args->vertexValue;
	uint* VertexIndices = args->vertexIndices;
	uint* edgeIndices = args->edgesIndices;

	// Process all the vertices assigned to this thread. Calculate vertex ID interval the thread has to process.
	for( unsigned int vertexID = (args->id)*(unsigned int)std::ceil((double)args->nVertices/args->numOfThreads);
			( vertexID < (((args->id)+1)*(unsigned int)std::ceil((double)args->nVertices/args->numOfThreads)) ) && ( vertexID < args->nVertices );
			++vertexID ) {

		Vertex final_vertex, intermed_outcome;

		// Get the vertex value at the end of previous iteration.
		Vertex previous_vertex_value = VertexValues[ vertexID ];

		// Do init_compute which initializes the vertex value that is going to be resulted at the end of this iteration.
		init_compute( &final_vertex, &previous_vertex_value );

		// Get starting address of the edges belonging to this vertex
		uint edges_starting_address = edgeIndices[ vertexID ];

		// Get the size of neighbors the vertex has.
		uint ngbrs_size = edgeIndices[ vertexID+1 ] - edges_starting_address;

		// For all the neighbors of the vertex.
		for( uint nbr = 0; nbr < ngbrs_size; ++nbr ) {

			// Get the edge address.
			uint target_edge_index = edges_starting_address + nbr;

			// Get the vertex value index at the other end of the edge.
			uint target_vertex_index = VertexIndices[ target_edge_index ];

			// Compute the results using vertex value at the other end of edge, edge value, results from previously processed vertex edges, and vertex value from previous iteration.
			compute_local( VertexValues[ target_vertex_index ],
							(args->VertexValueStatic)+target_vertex_index,
							(args->EdgeValue)+target_edge_index,
							&intermed_outcome,
							&final_vertex );

			// Reduce the resulted value from above computation and the value from reduction of previous iterations.
			compute_reduce( &final_vertex, &intermed_outcome );

		}

		// When all edges are processed, check if the vertex value has updated since the last iteration.
		if( update_condition( &final_vertex, &previous_vertex_value ) ) {

			// Signal that "The Show Must Go On".
			*(args->finished) = 1;

			// Update the newly calculated vertex value.
			VertexValues[ vertexID ] = final_vertex;

		}

	}

	pthread_exit(NULL);
	return NULL;

}


