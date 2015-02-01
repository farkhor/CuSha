#ifndef	USER_SPECIFIED_STRUCTURES_H
#define	USER_SPECIFIED_STRUCTURES_H

#include "user_specified_global_configurations.h"

/**************************************
 *  STRUCTURES
 **************************************/


// Vertex structure.
struct Vertex{

#ifdef BFS
	unsigned int distance;
#endif

#ifdef SSSP
	unsigned int distance;
#endif

#ifdef PR
	float rank;
#endif

};

// Vertex_static structure. Those properties of the vertex that remain constant during processing should be declared here.
typedef struct Vertex_static{

#ifdef PR
	unsigned int NbrsNum;
#endif

}Vertex_static;

// Edge structure.
struct Edge{

#ifdef SSSP
	unsigned int weight;
#endif

};



#endif	//	USER_SPECIFIED_STRUCTURES_H
