#ifndef VWC_PROCESS_CUH
#define VWC_PROCESS_CUH

#include "../common/user_specified_structures.h"

void vwc_process(
		int vwSize,
		uint gridDimen,
		const uint nVertices,
		const uint* vertexIndices,
		const uint* edgesIndices,
		Vertex* VertexValue,
		Edge* EdgeValue,
		Vertex_static* VertexValueStatic,
		int* finished );


#endif	//	VWC_PROCESS_CUH
