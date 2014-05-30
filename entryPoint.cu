/*
 * Free to use or modification (without any guarantee). Please download paper citation from:
 *	http://farkhor.github.io/CuSha
 *
 * Created by Farzad Khorasani.
 * 	Email: fkhor001@cs.ucr.edu
 * 	Webpage: http://www.cs.ucr.edu/~fkhor001
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common/CUDAErrorCheck.h"

enum GraphProcessingMethod {
	UNSPECIFIED = 0,
	CW,		// Concatenated Windows (CW) method
	GS,		// G-Shards method
	VWC,	// Virtual Warp-Centric method
	MTCPU	// Multi-threaded CPU
};

// Processing functions forward declaration.
void ExecuteCuShaCW(FILE* inputFile, const int suggestedBlockSize, FILE* outputFile, const int arbparam);
void ExecuteCuShaGS(FILE* inputFile, const int suggestedBlockSize, FILE* outputFile, const int arbparam);
void ExecuteVWC(FILE* inputFile, const int suggestedVirtualWarpSize, FILE* outputFile, const int arbparam);
void ExecuteMTCPU(FILE* inputFile, const int numOfThreads, FILE* outputFile, const int arbparam);

// Opening files safely.
bool openFileToAccess ( FILE** input_file, const char* file_name, const char* read_or_write ) {
	*input_file = fopen ( file_name, read_or_write);
	if ( *input_file == NULL ) {
		fprintf ( stderr, "Failed to open %s.\nExiting.", file_name ) ;
		return ( EXIT_FAILURE );
	}
	return ( EXIT_SUCCESS );
}

// Select the device safely.
bool selectDevice ( const int selectedDevice ) {
	int count;
	CUDAErrorCheck( cudaGetDeviceCount(&count));
	if ( count == 0 ) {
		fprintf ( stderr, "No CUDA device could be found.\n" ) ;
		return ( EXIT_FAILURE );
	}
	if ( selectedDevice < count ) {
		CUDAErrorCheck( cudaSetDevice(selectedDevice));
		fprintf(stdout, "Selected device ID: %u\n", selectedDevice);
		return ( EXIT_SUCCESS );
	}
	else {
		fprintf ( stderr, "Selected CUDA device could not be found.\n" ) ;
		return ( EXIT_FAILURE );
	}
}

// Execution entry point.
int main(int argc, char** argv)
{

	const char* usage =
	"\tRequired command line arguments:\n\
		-Input file: E.g., --input in.txt\n\
		-Processing method: CW, GS, VWC, MTCPU. E.g., --method CW\n\
	Additional arguments:\n\
		-Output file (default: out.txt). E.g., --output myout.txt.\n\
		-Device ID (default: 0). E.g., --device 1\n\
		-GPU kernels for Block size for CW and GS (default: chosen based on analysis). E.g., --bsize 512.\n\
		-Virtual warp size for VWC (default: 32). E.g., --vwsize 8.\n\
		-Number of threads for MTCPU (default: 1). E.g., --threads 4.\n\
		-User's arbitrary parameter (default: 0). E.g., --arbparam 17.\n";

	FILE* inputFile = NULL;
	FILE* outputFile = NULL;
	GraphProcessingMethod procesingMethod = UNSPECIFIED;
	int procesingMethodParameter = 0;	//default is zero
	int selectedDevice = 0;	//default is zero
	int outputFileStringIndex = 0;	//default to choose out.txt as the output file.
	int arbparam = 0;	//default is 0.

	// Getting required input parameters.
	for ( int iii = 1; iii < argc; ++iii ) {
		if ( !strcmp(argv[iii], "--input") && iii != argc-1 /*is not the last one*/) {
			// Open the input file. Exist in case of a failure.
			if ( openFileToAccess ( &inputFile, argv[iii+1], "r" ) == EXIT_FAILURE ) {
				exit(EXIT_FAILURE);
			}
			fprintf(stdout, "Input file: %s\n", argv[iii+1]);
		}
		if ( !strcmp(argv[iii], "--method") && iii != argc-1 ) {
			if ( !strcmp(argv[iii+1], "CW") ) {
				procesingMethod = CW;
				procesingMethodParameter = 1;
			}
			if ( !strcmp(argv[iii+1], "GS") ) {
				procesingMethod = GS;
				procesingMethodParameter = 1;
			}
			if ( !strcmp(argv[iii+1], "VWC") ) {
				procesingMethod = VWC;
				procesingMethodParameter = 32;	//its default.
			}
			if ( !strcmp(argv[iii+1], "MTCPU") ) {
				procesingMethod = MTCPU;
				procesingMethodParameter = 1;	//its default.
			}
		}
	}
	// Exit if there's no match satisfying required input parameters.
	if ( inputFile == NULL || procesingMethod == UNSPECIFIED ) {
		fprintf( stderr, "Input parameters don't match or are not satisfied. Usage:\n%s\n", usage );
		exit(EXIT_FAILURE);
	}

	// Getting additional input parameters.
	for ( int iii = 1; iii < argc; ++iii ) {
		if ( !strcmp(argv[iii], "--device") && iii != argc-1 )
			selectedDevice = atoi( argv[iii+1] );
		if ( !strcmp(argv[iii], "--output") && iii != argc-1 )
			outputFileStringIndex = iii+1;
		if ( ( 		( !strcmp(argv[iii], "--bsize") && ( procesingMethod == CW || procesingMethod == GS ) )
				||	( !strcmp(argv[iii], "--vwsize") && procesingMethod == VWC )
				||	( !strcmp(argv[iii], "--threads") && procesingMethod == MTCPU )	)
				&& iii != argc-1 ) {
			procesingMethodParameter = atoi( argv[iii+1] );
			assert( procesingMethodParameter > 0 );
		}
		if ( !strcmp(argv[iii], "--arbparam") && iii != argc-1 )
			arbparam = atoi( argv[iii+1] );
	}

	// Select the device. Exit in case of a failure.
	if ( procesingMethod != MTCPU )
		if ( selectDevice( selectedDevice ) == EXIT_FAILURE )
			exit(EXIT_FAILURE);

	// Open the output file. Exist in case of a failure.
	if ( openFileToAccess( &outputFile, (outputFileStringIndex) ? argv[outputFileStringIndex] : "out.txt", "w" ) == EXIT_FAILURE )
		exit(EXIT_FAILURE);

	// In case if someone is more comfortable with function pointers.
	/*
	void (*FPointer)(FILE**, int, FILE**, int);
	switch(procesingMethod){
	case CW:
		FPointer = &ExecuteCuShaCW;
		break;
	case GS:
		FPointer = &ExecuteCuShaGS;
		break;
	case VWC:
		FPointer = &ExecuteVWC;
		break;
	case MTCPU:
		FPointer = &ExecuteMTCPU;
		break;
	}
	FPointer(&inputFile, procesingMethodParameter, &outputFile );
	 */

	// Select which method to use
	switch(procesingMethod){
		case CW:
			ExecuteCuShaCW( inputFile, procesingMethodParameter, outputFile, arbparam );
			break;
		case GS:
			ExecuteCuShaGS( inputFile, procesingMethodParameter, outputFile, arbparam );
			break;
		case VWC:
			ExecuteVWC( inputFile, procesingMethodParameter, outputFile, arbparam );
			break;
		case MTCPU:
			ExecuteMTCPU( inputFile, procesingMethodParameter, outputFile, arbparam );
			break;
		default:
			fprintf( stderr, "Failed to perform any of available methods.\n" );
			exit(EXIT_FAILURE);
	}

	// Cleaning up before leaving.
	fclose( inputFile );
	fclose( outputFile );
	if ( procesingMethod != MTCPU )
		CUDAErrorCheck( cudaDeviceReset() );
	fprintf( stdout, "Done.\n");
	exit(EXIT_SUCCESS);
}
