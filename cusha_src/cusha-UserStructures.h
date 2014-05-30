#ifndef	_CUSHAUSERSTRUCTURES_H
#define	_CUSHAUSERSTRUCTURES_H

// Sample algorithm templates. Uncomment whichever (only the one) you want to use.
#define CUSHA_BFS 1
//#define CUSHA_SSSP 1
//#define CUSHA_PR 1

// User's Compile-time constant definitions
#define CUSHA_BFS_INF 1073741824
#define CUSHA_SSSP_INF 1073741824
#define CUSHA_PR_INITIAL_VALUE 0.0f
#define CUSHA_PR_DAMPING_FACTOR 0.85f
#define CUSHA_PR_TOLERANCE 0.005f

/**************************************
 *  STRUCTURES
 **************************************/

// Vertex structure.
typedef struct Vertex{

#ifdef CUSHA_BFS
	unsigned int distance;
#endif
#ifdef CUSHA_SSSP
	unsigned int distance;
#endif
#ifdef CUSHA_PR
	float rank;
#endif

}Vertex;

// Vertex_static structure. Those properties of the vertex that remain constant during processing should be declared here.
typedef struct Vertex_static{
	char dummy[];	// Dummy flexible array. It is useful when the structure is empty. Don't delete.

#ifdef CUSHA_PR
	unsigned int NbrsNum;
#endif

}Vertex_static;

// Edge structure.
typedef struct Edge{
	char dummy[];	// Dummy flexible array. It is useful when the structure is empty. Don't delete.

#ifdef CUSHA_SSSP
	unsigned int weight;
#endif

}Edge;

#endif
