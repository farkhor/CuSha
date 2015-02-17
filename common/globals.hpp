#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <stdlib.h>	// for timing
#include <sys/time.h>	// for timing

enum GraphProcessingMethod {
	UNSPECIFIED = 0,
	CW,		// Concatenated Windows (CW) method
	GS,		// G-Shards method
	VWC		// Virtual Warp-Centric method
};

#endif 	//	GLOBALS_HPP
