#ifndef _CSR_USER_STRUCTURES_H
#define _CSR_USER_STRUCTURES_H

// Sample algorithm templates. Uncomment whichever (only the one) you want to use.
//#define VWC_BFS 1
//#define VWC_SSSP 1
#define VWC_PR 1

// User's Compile-time constant definitions
#define VWC_BFS_INF 1073741824
#define VWC_SSSP_INF 1073741824
#define VWC_PR_INITIAL_VALUE 0.0f
#define VWC_PR_DAMPING_FACTOR 0.85f
#define VWC_PR_TOLERANCE 0.005f

/**************************************
 *  STRUCTURES
 **************************************/

// Vertex structure.
typedef struct Vertex{

#ifdef VWC_BFS
	unsigned int distance;
#endif
#ifdef VWC_SSSP
	unsigned int distance;
#endif
#ifdef VWC_PR
	float rank;
#endif

}Vertex;

// Vertex_static structure. Those properties of the vertex that remain constant during processing should be declared here.
typedef struct Vertex_static{
	char dummy[];	// Dummy flexible array. It is useful when the structure is empty. Don't delete.

#ifdef VWC_PR
	unsigned int NbrsNum;
#endif

}Vertex_static;

// Edge structure.
typedef struct Edge{
	char dummy[];	// Dummy flexible array. It is useful when the structure is empty. Don't delete.

#ifdef VWC_SSSP
	unsigned int weight;
#endif

}Edge;

#endif
