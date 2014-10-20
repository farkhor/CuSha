#ifndef	_CUSHA_USER_FUNCTIONS_H
#define	_CUSHA_USER_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cusha-UserStructures.h"

/**************************************
 *  INITIALIZATION FUNCTIONS
 **************************************/

// The function that helps to initialize vertex/edge contents before processing the graph.
// It will be called for every line of graph input file.
// Items are separated by space or tab.
inline void completeEntryCuSha (	unsigned int argcount,	// The number of additional items in the line
							char** argvector,	// char* pointer for which dereferencing its elements provides us with the additional items in the line in form of char*.
							const int src_vertex,	// Source vertex index.
							const int dst_vertex,	// Destination vertex index.
							Edge* edge_address,	// Pointer to the current edge corresponding to the current line.
							Vertex* src_vertex_address,	// Pointer to the source vertex.
							Vertex_static* src_vertex_static_address,	// Pointer to the source Vertex_static.
							Vertex* dst_vertex_address,  // Pointer to the destination vertex.
							Vertex_static* dst_vertex_static_address,	// Pointer to the destination Vertex_static.
							const int arbparam = 0	) {	// Arbitrary integer input in the console.

#ifdef CUSHA_BFS
	if( src_vertex != arbparam )	// For BFS, input parameter specifies the source vertex in traversal.
		src_vertex_address->distance = CUSHA_BFS_INF;	// BFS level infinity for others at the beginning
	else
		src_vertex_address->distance = 0;	// BFS level 0 for source

	// Most of the time for both head and tail, initialization is necessary otherwise you might end up having garbage in garbage out.
	if( dst_vertex != arbparam )
		dst_vertex_address->distance = CUSHA_BFS_INF;
	else
		dst_vertex_address->distance = 0;
#endif

#ifdef CUSHA_SSSP
	if( src_vertex != arbparam )	// For BFS, input parameter specifies the source vertex in traversal.
		src_vertex_address->distance = CUSHA_SSSP_INF;	// BFS level infinity for others at the beginning
	else
		src_vertex_address->distance = 0;	// BFS level 0 for source

	// Most of the time for both head and tail, initialization is necessary otherwise you might end up having garbage in garbage out.
	if( dst_vertex != arbparam )
		dst_vertex_address->distance = CUSHA_SSSP_INF;
	else
		dst_vertex_address->distance = 0;

	if ( argcount > 0 )
		edge_address->weight = atoi(argvector[0]);
	else
		edge_address->weight = 0;
#endif

#ifdef CUSHA_PR
	src_vertex_address->rank = CUSHA_PR_INITIAL_VALUE;
	dst_vertex_address->rank = CUSHA_PR_INITIAL_VALUE;
	if ( argcount > 0 )	// For this PageRank, the third column in input file has to be the number of neighbors the source vertex has.
		src_vertex_static_address->NbrsNum = atoi(argvector[0]);
	else
		src_vertex_static_address->NbrsNum = 1;
#endif

}

// Below function outputs the resulted vertex content.
// It will be performed at the end of processing for each and every vertex.
inline void printVertexOutputCuSha(	const int vertexIndex,
								const Vertex resultVertex,
								FILE* outFile	) {

#ifdef CUSHA_BFS
	fprintf(outFile, "%d:\t%u\n", vertexIndex, resultVertex.distance);
#endif

#ifdef CUSHA_SSSP
	fprintf(outFile, "%d:\t%u\n", vertexIndex, resultVertex.distance);
#endif

#ifdef CUSHA_PR
	fprintf(outFile, "%d:\t%f\n", vertexIndex, resultVertex.rank);
#endif

}


/**************************************
 *  PROCESSING FUNCTIONS
 **************************************/

// At every iteration, you need to initialize shared memory with vertices.
// This function is executed for each and every vertex.
inline __device__ void init_compute_CuSha(	Vertex* local_V,	// Address of the corresponding vertex in shared memory.
									Vertex* V	) {	// Address of the vertex in global memory

#ifdef CUSHA_BFS
	local_V->distance = V->distance;
#endif

#ifdef CUSHA_SSSP
	local_V->distance = V->distance;
#endif

#ifdef CUSHA_PR
	local_V->rank = 0;
#endif

}

// This function is executed for every edge in the shard.
// Since multiple threads may access same shared memory address at the same time, it (usually) has to be implemented with atomics.
inline __device__ void compute_CuSha(	Vertex* SrcV,	// Source vertex in global memory.
								Vertex_static* SrcV_static,	// Source Vertex_static in global memory. Dereferencing this pointer if it's not defined causes run-time error.
								Edge* E,	// Edge content for the entry. Dereferencing this pointer if it's not defined creates run-time error.
								Vertex* local_V	) {	// Current value of the corresponding (destination) vertex in the shared memory.

#ifdef CUSHA_BFS
	if (SrcV->distance != CUSHA_BFS_INF)	// Just to prevent possible unpredicted overflows.
		atomicMin ( &(local_V->distance), SrcV->distance + 1 );
#endif

#ifdef CUSHA_SSSP
	if (SrcV->distance != CUSHA_SSSP_INF)	// Just to prevent possible unpredicted overflows.
		atomicMin ( &(local_V->distance), SrcV->distance + E->weight );
#endif

#ifdef CUSHA_PR
	unsigned int nbrsNum = SrcV_static->NbrsNum;
	if ( nbrsNum != 0 )
		atomicAdd ( &(local_V->rank), SrcV->rank / nbrsNum );
#endif

}

// Below function signals the caller (and consequently the host) if the vertex content should be replaced with the newly calculated value.
// It is executed for every vertex assigned to the shard.
inline __device__ bool update_condition_CuSha(	Vertex* local_V,	// newly calculated vertex content.
										Vertex* V	) {	// Vertex content at the end of previous iteration.

#ifdef CUSHA_BFS
	return ( local_V->distance < V->distance );
#endif

#ifdef CUSHA_SSSP
	return ( local_V->distance < V->distance );
#endif

#ifdef CUSHA_PR
	local_V->rank = (1-CUSHA_PR_DAMPING_FACTOR) + local_V->rank*CUSHA_PR_DAMPING_FACTOR;
	return ( fabs( local_V->rank - V->rank ) > CUSHA_PR_TOLERANCE );
#endif

}

#endif
