#ifndef	_CUSHA_UTILS_H
#define	_CUSHA_UTILS_H

#include "cusha-AuxStructures.h"

void init_primitiveVertex(	primitiveVertex* prim_vertex	);
void delete_primitiveVertex(	primitiveVertex* prim_vertex	);

void populatePrimitiveVerticesForCuSha(	unsigned int* num_edges,
										unsigned int* num_vertices,
										primitiveVertex** primitiveVertices,
										FILE* inFile,
										const int inParam);

blockSize_N_pair findProperBlockSize(	const int suggestedBlockSize,
										const unsigned int num_of_edges,
										const unsigned int num_of_vertices,
										const unsigned int Vertex_size );

void populate_shards_from_primitive_vertices(	const unsigned int num_of_shards,
												const unsigned int num_of_vertices,
												unsigned int* window_offsets,
												unsigned int* shard_sizes,
												const unsigned int N,
												GSShard* temp_shards,
												const primitiveVertex* primitiveVertices);

#endif
