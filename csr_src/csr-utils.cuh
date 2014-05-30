#ifndef	_CSRUTILS_H
#define	_CSRUTILS_H

#include "csr-AuxStructures.h"

void init_primitiveVertexCSR(	primitiveVertex* prim_vertex	);

void delete_primitiveVertexCSR(	primitiveVertex* prim_vertex	);

void populatePrimitiveVertices(	CSRGraph* inGraph,
								primitiveVertex** primitiveVertices,
								FILE* inFile,
								const int inParam);

void copyPrimitiveVerticesIntoCSRGraph(	CSRGraph* inGraph,
										primitiveVertex** primitiveVertices );

#endif
