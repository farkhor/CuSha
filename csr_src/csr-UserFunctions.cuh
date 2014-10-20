#ifndef	_CSR_USER_FUNCTIONS_H
#define	_CSR_USER_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "csr-UserStructures.h"


/**************************************
 *  INITIALIZING FUNCTIONS
 **************************************/

// The function that helps to initialize vertex/edge contents before processing the graph.
// It will be called for every line of graph input file.
// Items are separated by space or tab.
inline void completeEntry (	unsigned int argcount,	// The number of additional items in the line
						char** argvector,	// char* pointer for which dereferencing its elements provides us with the additional items in the line in form of char*.
						const int src_vertex,	// Source vertex index.
						const int dst_vertex,	// Destination vertex index.
						Edge* edge_address,	// Pointer to the current edge corresponding to the current line.
						Vertex* src_vertex_address,	// Pointer to the source vertex.
						Vertex_static* src_vertex_static_address,	// Pointer to the source Vertex_static.
						Vertex* dst_vertex_address,  // Pointer to the destination vertex.
						Vertex_static* dst_vertex_static_address,	// Pointer to the destination Vertex_static.
						const int arbparam = 0	) {	// Arbitrary integer input in the console.

#ifdef VWC_BFS
	if( src_vertex != arbparam )	// For BFS, input parameter specifies the source vertex in traversal.
		src_vertex_address->distance = VWC_BFS_INF;	// BFS level infinity for others at the beginning
	else
		src_vertex_address->distance = 0;	// BFS level 0 for source

	// Most of the time for both head and tail, initialization is necessary otherwise you might end up having garbage in garbage out.
	if( dst_vertex != arbparam )
		dst_vertex_address->distance = VWC_BFS_INF;
	else
		dst_vertex_address->distance = 0;
#endif

#ifdef VWC_SSSP
	if( src_vertex != arbparam )	// For BFS, input parameter specifies the source vertex in traversal.
		src_vertex_address->distance = VWC_SSSP_INF;	// BFS level infinity for others at the beginning
	else
		src_vertex_address->distance = 0;	// BFS level 0 for source

	// Most of the time for both head and tail, initialization is necessary otherwise you might end up having garbage in garbage out.
	if( dst_vertex != arbparam )
		dst_vertex_address->distance = VWC_SSSP_INF;
	else
		dst_vertex_address->distance = 0;

	if ( argcount > 0 )
		edge_address->weight = atoi(argvector[0]);
	else
		edge_address->weight = 0;
#endif

#ifdef VWC_PR
	src_vertex_address->rank = VWC_PR_INITIAL_VALUE;
	dst_vertex_address->rank = VWC_PR_INITIAL_VALUE;
	if ( argcount > 0 )	// For this PageRank, the third column in input file has to be the number of incoming neighbors the source vertex has.
		src_vertex_static_address->NbrsNum = atoi(argvector[0]);
	else
		src_vertex_static_address->NbrsNum = 1;
#endif

}

// Below function outputs the resulted vertex content.
// It will be performed at the end of processing for each and every vertex.
inline void printVertexOutput(	const int vertexIndex,
						const Vertex resultVertex,
						FILE* outFile	) {

#ifdef VWC_BFS
	fprintf(outFile, "%d:\t%u\n", vertexIndex, resultVertex.distance);
#endif

#ifdef VWC_SSSP
	fprintf(outFile, "%d:\t%u\n", vertexIndex, resultVertex.distance);
#endif

#ifdef VWC_PR
	fprintf(outFile, "%d:\t%f\n", vertexIndex, resultVertex.rank);
#endif

}


/**************************************
 *  PROCESSING FUNCTIONS
 **************************************/

// At every iteration, you need to initialize shared memory with vertices.
// This function is executed for each and every vertex.
inline __device__ __host__ void init_compute (	volatile Vertex* local_V,	// Address of the corresponding vertex in shared memory.
		Vertex* V	) {	// Address of the previous version of the vertex in shared memory.

#ifdef VWC_BFS
	local_V->distance = V->distance;
#endif

#ifdef VWC_SSSP
	local_V->distance = V->distance;
#endif

#ifdef VWC_PR
	local_V->rank = 0;
#endif

}

// In below, each thread computes a result based on edge data and writes to its own specific shared memory.
// This function is executed for each and every edge.
inline __device__ __host__ void compute_local (	Vertex* SrcV,	// Source vertex in global memory.
											Vertex_static* SrcV_static,	// Source Vertex_static in global memory. Dereferencing this pointer if it's not defined causes error.
											Edge* E,	// Edge in global memory. Dereferencing this pointer if it's not defined cause error.
											volatile Vertex* thread_V_in_shared,	// Thread's specific shared memory.
											Vertex* preV	) {	// Value of the corresponding (destination) vertex initialized at the previous step.

#ifdef VWC_BFS
	thread_V_in_shared->distance = SrcV->distance + 1;
#endif

#ifdef VWC_SSSP
	thread_V_in_shared->distance = SrcV->distance + E->weight;
#endif

#ifdef VWC_PR
	unsigned int nbrsNum = SrcV_static->NbrsNum;
	if ( nbrsNum != 0 )
		thread_V_in_shared->rank = SrcV->rank / nbrsNum;
#endif

}

// Reduction function that is performed for every pair of neighbors of a vertex.
inline __device__ __host__ void compute_reduce (	volatile Vertex* thread_V_in_shared,
		volatile Vertex* next_thread_V_in_shared	) {

#ifdef VWC_BFS
	if ( thread_V_in_shared->distance > next_thread_V_in_shared->distance )
		thread_V_in_shared->distance = next_thread_V_in_shared->distance;
#endif

#ifdef VWC_SSSP
	if ( thread_V_in_shared->distance > next_thread_V_in_shared->distance)
		thread_V_in_shared->distance = next_thread_V_in_shared->distance;
#endif

#ifdef VWC_PR
	thread_V_in_shared->rank += next_thread_V_in_shared->rank;
#endif

}

// Below function signals the caller (and consequently the host) if the vertex content should be replaced with the newly calculated value.
// This function is performed by one virtual lane in the virtual warp.
inline __device__ __host__ bool update_condition (	volatile Vertex* computed_V,
		Vertex* previous_V	) {

#ifdef VWC_BFS
	return ( computed_V->distance < previous_V->distance );
#endif

#ifdef VWC_SSSP
	return ( computed_V->distance < previous_V->distance );
#endif

#ifdef VWC_PR
	computed_V->rank = (1-VWC_PR_DAMPING_FACTOR) + computed_V->rank*VWC_PR_DAMPING_FACTOR;	// Or you can replace this expression by fused multiply-add.
	return ( fabs( computed_V->rank - previous_V->rank) > VWC_PR_TOLERANCE );
#endif

}

#endif



