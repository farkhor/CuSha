#ifndef CUSHA_PROCESS_CUH
#define CUSHA_PROCESS_CUH

#include "../common/globals.hpp"
#include "../common/user_specified_structures.h"


void cusha_process(
		const GraphProcessingMethod procesingMethod,
		const uint blockSize,
		const uint N,
		const uint nShards,
		const uint nVertices,
		Vertex* vertexValue,
		const uint* concatenatedWindowsSizesScan,
		const uint* windowSizesScansVertical,
		const uint* shardSizesScans,
		int* finished,
		Vertex* srcValue,
		const uint* dstIndex,
		Edge* edgeValue,
		Vertex_static* vertexValueStatic,
		const uint* srcIndex,
		const uint* mapper );

#endif	//	CUSHA_PROCESS_CUH
