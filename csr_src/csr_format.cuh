#ifndef CSR_FORMAT_CUH
#define CSR_FORMAT_CUH

#include <fstream>
#include <vector>

#include "../common/initial_graph.hpp"
#include "../common/globals.hpp"


namespace csr_format{
	void process(
			const int vwsize_or_threads,
			std::vector<initial_vertex>* initGraph,
			const uint nEdges,
			std::ofstream& outputFile,
			bool EdgesOnHost = false );
}

#endif	//	CSR_FORMAT_CUH
