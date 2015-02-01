#ifndef CUSHA_FORMAT_CUH
#define CUSHA_FORMAT_CUH

#include <fstream>
#include <vector>

#include "../common/initial_graph.hpp"
#include "../common/globals.hpp"


namespace cusha_format{
	void process(
			const GraphProcessingMethod procesingMethod,
			const int bsize,
			std::vector<initial_vertex>* initGraph,
			const uint nEdges,
			std::ofstream& outputFile,
			bool EdgesOnHost = false );
}

#endif	//	CUSHA_FORMAT_CUH
