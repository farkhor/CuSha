#include <string>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/globals.hpp"
#include "common/cuda_error_check.cuh"
#include "common/initial_graph.hpp"
#include "common/parse_graph.hpp"
#include "cusha_src/cusha_format.cuh"
#include "csr_src/csr_format.cuh"


// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}


// Execution entry point.
int main( int argc, char** argv )
{

	std::string usage =
		"\tRequired command line arguments:\n\
			-Input file: E.g., --input in.txt\n\
			-Processing method: CW, GS, VWC. E.g., --method CW\n\
		Additional arguments:\n\
			-Output file (default: out.txt). E.g., --output myout.txt.\n\
			-Is the input graph directed (default:yes). To make it undirected: --undirected\n\
			-Device ID (default: 0). E.g., --device 1\n\
			-GPU kernels Block size for CW and GS (default: chosen based on analysis). E.g., --bsize 512.\n\
			-Virtual warp size for VWC (default: 32). E.g., --vwsize 8.\n\
			-User's arbitrary parameter (default: 0). E.g., --arbparam 17.\n";

	try {

		GraphProcessingMethod procesingMethod = UNSPECIFIED;
		std::ifstream inputFile;
		std::ofstream outputFile;
		int selectedDevice = 0;
		int bsize = 0;
		int vwsize = 32;
		int threads = 1;
		long long arbparam = 0;
		bool nonDirectedGraph = false;		// By default, the graph is directed.

		/********************************
		 * GETTING INPUT PARAMETERS.
		 ********************************/

		for( int iii = 1; iii < argc; ++iii )
			if ( !strcmp(argv[iii], "--method") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "CW") )
					procesingMethod = CW;
				if ( !strcmp(argv[iii+1], "GS") )
					procesingMethod = GS;
				if ( !strcmp(argv[iii+1], "VWC") )
					procesingMethod = VWC;
			}
			else if( !strcmp( argv[iii], "--input" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ifstream >( inputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--device" ) && iii != argc-1 /*is not the last one*/)
				selectedDevice = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--bsize" ) && iii != argc-1 /*is not the last one*/)
				bsize = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--vwsize" ) && iii != argc-1 /*is not the last one*/)
				vwsize = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--arbparam" ) && iii != argc-1 /*is not the last one*/)
				arbparam = std::atoll( argv[iii+1] );
			else if( !strcmp(argv[iii], "--undirected"))
				nonDirectedGraph = true;

		if( !inputFile.is_open() || procesingMethod == UNSPECIFIED ) {
			std::cerr << "Usage: " << usage;
			throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
		}
		if( !outputFile.is_open() )
			openFileToAccess< std::ofstream >( outputFile, "out.txt" );
		CUDAErrorCheck( cudaSetDevice( selectedDevice ) );
		std::cout << "Device with ID " << selectedDevice << " is selected to process the graph.\n";
		if( procesingMethod == VWC ) {
			if( vwsize != 2 && vwsize !=4 && vwsize != 8 && vwsize != 16 && vwsize != 32 )
				vwsize = 32;
			std::cout << "Virtual-Warp Centric method will be employed to process the graph with virtual warp size " << vwsize << ".\n";
		}


		/********************************
		 * Read the input graph file.
		 ********************************/

		std::cout << "Collecting the input graph ...\n";
		std::vector<initial_vertex> parsedGraph( 0 );
		uint nEdges = parse_graph::parse(
				inputFile,		// Input file.
				parsedGraph,	// The parsed graph.
				arbparam,
				nonDirectedGraph );		// Arbitrary user-provided parameter.
		std::cout << "Input graph collected with " << parsedGraph.size() << " vertices and " << nEdges << " edges.\n";


		/********************************
		 * Process the graph.
		 ********************************/

		if( procesingMethod == GS || procesingMethod == CW ) {
			cusha_format::process(
					procesingMethod,
					bsize,
					&parsedGraph,
					nEdges,
					outputFile );
		}
		else {
			csr_format::process(
					((procesingMethod==VWC)?vwsize:threads),
					&parsedGraph,
					nEdges,
					outputFile );
		}


		/********************************
		 * It's done here.
		 ********************************/

		CUDAErrorCheck( cudaDeviceReset() );
		std::cout << "Done.\n";
		return( EXIT_SUCCESS );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}

}
