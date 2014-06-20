#ifndef	_CSR_USER_FUNCTIONS_H
#define	_CSR_USER_FUNCTIONS_H

#include "csr-UserStructures.h"

void completeEntry(	unsigned int argcount,
					char** argvector,
					const int head_vertex,
					const int tail_vertex,
					Edge* edge_address,
					Vertex* vertex_address,
					Vertex_static* vertex_static_address,
					Vertex* tail_vertex_address,
					Vertex_static* tail_vertex_static_address,
					const int arbparam	);

void printVertexOutput(	const int vertexIndex,
						const Vertex resultVertex,
						FILE* outFile	);

__device__ __host__ void init_compute(	volatile Vertex* local_V,
		Vertex* V	);

__device__ __host__ void compute_local(	Vertex* SrcV,
										Vertex_static* SrcV_static,
										Edge* E,
										volatile Vertex* local_V,
										Vertex* preV	);

__device__ __host__ void compute_reduce(	volatile Vertex* local_V,
		volatile Vertex* local_V_next	);

__device__ __host__ bool update_condition(	volatile Vertex* local_V,
		Vertex* V	);


#endif
