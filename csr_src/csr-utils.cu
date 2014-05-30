#include <stdio.h>
#include <assert.h>
#include "csr-AuxStructures.h"
#include "csr-UserFunctions.cuh"

// This function initializes primitive vertex data.
void init_primitiveVertexCSR(	primitiveVertex* prim_vertex	) {
	prim_vertex->num_of_nbrs = 0;
	prim_vertex->nbrs = (unsigned int*) malloc ( sizeof(unsigned int) );
	assert(prim_vertex->nbrs);
	prim_vertex->EdgeValue = (Edge*) malloc ( sizeof(Edge) );
	if(sizeof(Edge) > 0)
		assert(prim_vertex->EdgeValue);
}

// This function deletes primitive vertex data.
void delete_primitiveVertexCSR( primitiveVertex* prim_vertex ) {
	free ( prim_vertex->nbrs );
	if(sizeof(Edge) > 0)
		free ( prim_vertex->EdgeValue );
}

// This function reads input file line by line, and stores them in form of primitiveVertex.
void populatePrimitiveVertices(	CSRGraph* inGraph,	// Input CSR graph. I only use number_of_vertices and number_of_edges elements of it here.
								primitiveVertex** primitiveVertices,	// PrimitiveVertex* pointer that will be filled in.
								FILE* inFile,	// Input file.
								const int inParam){	// Integer command-line input parameter.

	void* helper_pointer;

	const unsigned int MAX_LINE_SIZE = 256;
	const unsigned int MAX_NUMBER_OF_ADDITIONAL_INPUT_ARGUMENTS = 61;
	char theLine[MAX_LINE_SIZE];
	char*	pch;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	int first_index, second_index;

	unsigned int Additionalargc=0;
	char** Additionalargv;
	helper_pointer = malloc(MAX_NUMBER_OF_ADDITIONAL_INPUT_ARGUMENTS*sizeof(char*));
	assert( helper_pointer );
	Additionalargv = (char**) helper_pointer;

	unsigned int num_of_vertices = 0, num_of_edges = 0, theMax;

	// Read till the end of file
	while ( fgets( theLine, MAX_LINE_SIZE, inFile ) ) {
		if ( theLine[0] < '0' || theLine[0] > '9')	// Skipping any line blank or starting with a character rather than a number.
			continue;

		// At least two elements are required to have incoming and outgoing indices for the edge. Otherwise skip the line.
		pch = strtok (theLine, delim);
		if( pch != NULL )
			second_index = atoi ( pch );	// first_index = atoi ( pch );
		else
			continue;
		pch = strtok (NULL, delim);
		if( pch != NULL )
			first_index = atoi ( pch );	// second_index = atoi ( pch );
		else
			continue;

		theMax = max (first_index, second_index);
		if ( theMax >= num_of_vertices ) {
			helper_pointer = realloc ( *primitiveVertices, (theMax+1)*sizeof(primitiveVertex) );
			assert( helper_pointer );
			(*primitiveVertices) = (primitiveVertex*) helper_pointer;
			for ( unsigned int i = num_of_vertices; i <= theMax; ++i )
				init_primitiveVertexCSR ( &((*primitiveVertices)[i]) );
			num_of_vertices = theMax + 1;
		}

		//reallocation for every 32 neighbors. Nothing special about 32. Tradeoff between lots of reallocations (that slow down the program) and using lots of memory.
		if ( (*primitiveVertices)[first_index].num_of_nbrs % 32 == 0) {
			helper_pointer = realloc ( (*primitiveVertices)[first_index].nbrs, ((*primitiveVertices)[first_index].num_of_nbrs + 32)*sizeof(unsigned int) );
			assert( helper_pointer );
			(*primitiveVertices)[first_index].nbrs = (unsigned int*) helper_pointer;
			helper_pointer = realloc ( (*primitiveVertices)[first_index].EdgeValue, ((*primitiveVertices)[first_index].num_of_nbrs + 32)*sizeof(Edge) );
			if(sizeof(Edge) > 0)
				assert( helper_pointer );
			(*primitiveVertices)[first_index].EdgeValue = (Edge*) helper_pointer;
		}
		(*primitiveVertices)[first_index].nbrs [ (*primitiveVertices)[first_index].num_of_nbrs ] = second_index;
		(*primitiveVertices)[first_index].num_of_nbrs++;

		Additionalargc=0;
		Additionalargv[Additionalargc] = strtok (NULL, delim);
		while( Additionalargv[Additionalargc] != NULL ){
			Additionalargc++;
			Additionalargv[Additionalargc] = strtok (NULL, delim);
		}
		completeEntry(	Additionalargc,
						Additionalargv,
						first_index,
						second_index,
						((*primitiveVertices)[first_index].EdgeValue)+((*primitiveVertices)[first_index].num_of_nbrs),
						&((*primitiveVertices)[first_index].VertexValue),
						&((*primitiveVertices)[first_index].VertexValue_static),
						&((*primitiveVertices)[second_index].VertexValue),
						&((*primitiveVertices)[second_index].VertexValue_static),
						inParam	);

		num_of_edges++;

	}

	inGraph->num_of_vertices = num_of_vertices;
	inGraph->num_of_edges = num_of_edges;

	free( Additionalargv );

}

// Below function fills up the CSR graph from primitive vertices.
void copyPrimitiveVerticesIntoCSRGraph(	CSRGraph* inGraph,
										primitiveVertex** primitiveVertices ) {

	inGraph->vertices_indices[0] = 0;
	for ( int i = 0; i < inGraph->num_of_vertices; ++i ){
		inGraph->VertexValue[i] = (*primitiveVertices)[i].VertexValue;
		if(sizeof(Vertex_static)>0)
			inGraph->VertexValue_static[i] = (*primitiveVertices)[i].VertexValue_static;
		for ( int j = 0; j < (*primitiveVertices)[i].num_of_nbrs; ++j ) {
			inGraph->edges_indices [ inGraph->vertices_indices[i] + j ] = (*primitiveVertices)[i].nbrs[j];
			if(sizeof(Edge)>0)
				inGraph->EdgeValue [ inGraph->vertices_indices[i] + j ] = (*primitiveVertices)[i].EdgeValue[j];
		}
		inGraph->vertices_indices[i+1] = inGraph->vertices_indices[i] + (*primitiveVertices)[i].num_of_nbrs;
	}

}
