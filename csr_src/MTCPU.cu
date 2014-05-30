#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <math.h>
#include "../common/simpleTime.h"
#include "../common/CUDAErrorCheck.h"
#include "csr-utils.cuh"
#include "csr-UserFunctions.cuh"

#define JOB_FINISHED 1
#define JOB_NOT_FINISHED_YET 0

// Arguments passing to each thread.
typedef struct arg_struct {
	int id;						//Thread ID.
	int numOfThreads;			//Total number of threads.
	unsigned int* finished;		//Pointer to finished flag in the main thread.
	CSRGraph* Graph;				//Pointer to the graph.
}arg_struct;


// Each thread executes this function.
void* ACPUThreadJob ( void *arguments ) {

	// Thread gets arguments
	arg_struct* args = (arg_struct*)arguments;
	CSRGraph* theGraph = args->Graph;

	// Calculate vertex ID interval the thread has to process.
	unsigned int vertexID = (args->id)*(unsigned int)ceil((double)theGraph->num_of_vertices/args->numOfThreads);
	unsigned int upperLimitVertexID = (((args->id)+1)*(unsigned int)ceil((double)theGraph->num_of_vertices/args->numOfThreads));

	// All the variables we are going to use in the upcoming for loop.
	unsigned int nbr, edges_starting_address, ngbrs_size, target_edge_index, target_vertex_index;
	Vertex final_vertex, previous_vertex_value, intermed_outcome;

	// Process all the vertices assigned to this thread.
	for ( ;vertexID < upperLimitVertexID && vertexID < theGraph->num_of_vertices; ++vertexID ) {

		// Get the vertex value at the end of previous iteration.
		previous_vertex_value = theGraph->VertexValue[vertexID];

		// Do init_compute which initializes the vertex value that is going to be resulted at the end of this iteration.
		init_compute ( &final_vertex, (theGraph->VertexValue)+vertexID );

		// Get starting address of the edges belonging to this vertex
		edges_starting_address = theGraph->vertices_indices[vertexID];

		// Get the size of neighbors the vertex has.
		ngbrs_size = theGraph->vertices_indices[vertexID+1] - edges_starting_address;

		// For all the neighbors of the vertex.
		for ( nbr = 0; nbr < ngbrs_size; ++nbr ) {

			// Get the edge address.
			target_edge_index = edges_starting_address + nbr;

			// Get the vertex value index at the other end of the edge.
			target_vertex_index = theGraph->edges_indices[target_edge_index];

			// Compute the results using vertex value at the other end of edge, edge value, results from previously processed vertex edges, and vertex value from previous iteration.
			compute_local( (theGraph->VertexValue)+target_vertex_index,
							(theGraph->VertexValue_static)+target_vertex_index,
							(theGraph->EdgeValue)+target_edge_index,
							&intermed_outcome,
							&previous_vertex_value );

			// Reduce the resulted value from above computation and the value from reduction of previous iterations.
			compute_reduce( &final_vertex, &intermed_outcome );

		}
		// When all edges are processed, check if the vertex value has updated since the last iteration.
		if ( update_condition( &final_vertex, &previous_vertex_value ) ) {

			// Signal that "The Show Must Go On".
			*(args->finished) = JOB_NOT_FINISHED_YET;

			// Update the newly calculated vertex value.
			theGraph->VertexValue[vertexID] = final_vertex;

		}

	}

	pthread_exit(NULL);
	return NULL;

}

bool processGraphMTCPU(	CSRGraph* theGraph,
						const int num_of_threads ) {

	// Creating threads and needed arguments.
	pthread_t* allThreads = (pthread_t*) malloc( num_of_threads * sizeof(pthread_t) );
	assert( allThreads );	// Catch if the pointer is NULL.

	arg_struct* args = (arg_struct*) malloc( num_of_threads * sizeof(arg_struct) );
	assert( args );	// Catch if pointer is NULL.

	unsigned int counter = 0;	// Count the number of iterations.
	unsigned int finished;	// flag indicating if the iterative processing is finished.
	int i;

	// Iteratively process the graph.
	fprintf( stdout, "Processing graph in a virtual warp-centric manner with CSR representation using %d CPU thread(s) ...\n", num_of_threads );
	// Start the timer to measure how long processing is going to take.
	setTime();

	do {

		// Initialize finished flag.
		finished = JOB_FINISHED;

		// For all CPU threads processing the graph.
		for ( i = 0; i < num_of_threads; ++i ) {

			// Filling up thread's specific argument structure.
			args[i].id = i;
			args[i].Graph = theGraph;
			args[i].numOfThreads = num_of_threads;
			args[i].finished = &finished;
			if (pthread_create(&(allThreads[i]), NULL, &ACPUThreadJob, (void *)&args[i]) != 0) {

				// If there is a problem when creating threads.
				fprintf(stderr, "An error happened during creation of thread(s).\n", i);

			    // Free allocated memory before going back.
				free (allThreads);
				free (args);
			    return(EXIT_FAILURE);
			}
		}

		// Join all created threads.
		for ( i = 0; i < num_of_threads; ++i ) {
			pthread_join(allThreads[i], NULL);
		}

		// Increase iteration counter.
		counter++;

	// Keep going if the processing is not yet finished.
	} while ( finished == JOB_NOT_FINISHED_YET );

	// Printing the processing statistics.
	fprintf( stdout, "Processing finished in: %f (ms)\n", getTime());
	fprintf( stdout, "Performed %u iterations in total.\n", counter);

	// Clean up before going back.
	free (allThreads);
	free (args);
	return(EXIT_SUCCESS);

}


void ExecuteMTCPU(	FILE* inputFile,
					const int num_of_threads,
					FILE* outputFile,
					const int arbparam) {

	CSRGraph HostGraph;
	primitiveVertex* primitiveVertices = (primitiveVertex*) malloc ( sizeof(primitiveVertex));
	assert( primitiveVertices );
	init_primitiveVertexCSR ( primitiveVertices );
	fprintf( stdout, "Populating the graph ...\n" );
	populatePrimitiveVertices( &HostGraph, &primitiveVertices, inputFile, arbparam);

	if(HostGraph.num_of_edges == 0) {
		fprintf( stderr, "No edge could be read from the file. Make sure provided formatting matches defined specification.\n");
		exit(EXIT_FAILURE);
	}
	else {
		fprintf( stdout, "Graph is populated with %u vertices and %u edges.\n", HostGraph.num_of_vertices, HostGraph.num_of_edges );
	}

	// Allocate using page-locked host memory. Graph size is limited by the size of system DRAM.
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.vertices_indices), (HostGraph.num_of_vertices+1) * sizeof(unsigned int), cudaHostAllocDefault ) );
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.edges_indices), HostGraph.num_of_edges * sizeof(unsigned int), cudaHostAllocDefault ) );
	CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.VertexValue), HostGraph.num_of_vertices * sizeof(Vertex), cudaHostAllocDefault ) );
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.EdgeValue), HostGraph.num_of_edges * sizeof(Edge), cudaHostAllocDefault ) );
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaHostAlloc( (void**) &(HostGraph.VertexValue_static), HostGraph.num_of_vertices * sizeof(Vertex_static), cudaHostAllocDefault ) );

	copyPrimitiveVerticesIntoCSRGraph(&HostGraph, &primitiveVertices);

	// Free up primitive vertices.
	for( int i = 0; i < HostGraph.num_of_vertices; ++i )
		delete_primitiveVertexCSR( primitiveVertices+i );
	free( primitiveVertices );

	if ( processGraphMTCPU(&HostGraph, num_of_threads) == EXIT_FAILURE) {
		fprintf(stderr, "Exiting.\n");
		exit(EXIT_FAILURE);
	}

	for ( int i = 0; i < HostGraph.num_of_vertices; ++i )
		printVertexOutput(i, HostGraph.VertexValue[i], outputFile);

	CUDAErrorCheck ( cudaFreeHost(HostGraph.vertices_indices) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.edges_indices) );
	CUDAErrorCheck ( cudaFreeHost(HostGraph.VertexValue) );
	if(sizeof(Edge)>0) CUDAErrorCheck ( cudaFreeHost(HostGraph.EdgeValue) );
	if(sizeof(Vertex_static)>0) CUDAErrorCheck ( cudaFreeHost(HostGraph.VertexValue_static) );

}
