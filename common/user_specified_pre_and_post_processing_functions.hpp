#ifndef	USER_SPECIFIED_PRE_AND_POST_PROCESSING_FUNCTIONS_HPP
#define	USER_SPECIFIED_PRE_AND_POST_PROCESSING_FUNCTIONS_HPP

#include <fstream>
#include <string>

#include "user_specified_structures.h"
#include "user_specified_global_configurations.h"

/**************************************
 *  INITIALIZATION FUNCTION
 **************************************/

// The function that helps to initialize vertex/edge contents before processing the graph.
// It will be called for every line of graph input file.
// Items are separated by space or tab.
inline void completeEntry(
		unsigned int argcount,	// The number of additional items in the line
		char** argvector,	// char* pointer for which dereferencing its elements provides us with the additional items in the line in form of char*.
		const int src_vertex_index,	// Source vertex index.
		const int dst_vertex_index,	// Destination vertex index.
		Edge* edge_address,	// Pointer to the current edge corresponding to the current line.
		Vertex& src_vertex_ref,	// Pointer to the source vertex.
		Vertex_static* src_vertex_static_address,	// Pointer to the source Vertex_static.
		Vertex& dst_vertex_ref,  // Pointer to the destination vertex.
		Vertex_static* dst_vertex_static_address,	// Pointer to the destination Vertex_static.
		const long long arbparam = 0	// Arbitrary integer input in the console.
		) {

#ifdef BFS
	src_vertex_ref.distance = ( src_vertex_index != arbparam ) ? BFS_INF : 0;
	dst_vertex_ref.distance = ( dst_vertex_index != arbparam ) ? BFS_INF : 0;
#endif

#ifdef SSSP
	src_vertex_ref.distance = ( src_vertex_index != arbparam ) ? SSSP_INF : 0;
	dst_vertex_ref.distance = ( dst_vertex_index != arbparam ) ? SSSP_INF : 0;
	edge_address->weight = ( argcount > 0 ) ? atoi(argvector[0]) : 0;
#endif

#ifdef PR
	src_vertex_ref.rank = PR_INITIAL_VALUE;
	dst_vertex_ref.rank = PR_INITIAL_VALUE;
	src_vertex_static_address->NbrsNum = ( argcount > 0 ) ? atoi( argvector[0] ) : 0;
#endif

}


/**************************************
 *  OUTPUT FORMATTING FUNCTION
 **************************************/

// Below function outputs the resulted vertex content.
// It will be performed at the end of processing for each and every vertex.
inline void print_vertex_output(
		const uint vertexIndex,
		const Vertex resultVertex,
		std::ofstream& outFile
		) {

#ifdef BFS
	outFile << vertexIndex << ":\t" << resultVertex.distance << "\n";
#endif

#ifdef SSSP
	outFile << vertexIndex << ":\t" << resultVertex.distance << "\n";
#endif

#ifdef PR
	outFile << vertexIndex << ":\t" << resultVertex.rank << "\n";
#endif

}


#endif	//	USER_SPECIFIED_PRE_AND_POST_PROCESSING_FUNCTIONS_HPP
