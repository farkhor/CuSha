#ifndef MTCPU_PROCESS_HPP
#define MTCPU_PROCESS_HPP

#include "../common/user_specified_structures.h"


// Arguments passing to each thread.
struct arg_struct{
	int id;						//Thread ID.
	int numOfThreads;			//Total number of threads.
	int* finished;		//Pointer to finished flag in the main thread.
	int nVertices;
	uint* vertexIndices;
	uint* edgesIndices;
	Vertex* vertexValue;
	Edge* EdgeValue;
	Vertex_static* VertexValueStatic;
};

// Each thread executes this function.
void* ACPUThreadJob ( void *arguments );


#endif	//	MTCPU_PROCESS_HPP
