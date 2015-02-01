#ifndef	USER_SPECIFIED_GLOBAL_CONFIGURATIONS_H
#define	USER_SPECIFIED_GLOBAL_CONFIGURATIONS_H
///////////////////////////////////////////////

#define VWC_COMPILE_TIME_DEFINED_BLOCK_SIZE 256

/*********************************************
 * Sample algorithm templates. Uncomment whichever (only the one) you want to use.
 *********************************************/

//#define BFS
#define SSSP
//#define PR

/*********************************************
 * User's Compile-time constant definitions.
 *********************************************/

#define BFS_INF 1073741824

#define SSSP_INF 1073741824

#define PR_INITIAL_VALUE 0.0f
#define PR_DAMPING_FACTOR 0.85f
#define PR_TOLERANCE 0.005f

///////////////////////////////////////////////
#endif	//	USER_SPECIFIED_GLOBAL_CONFIGURATIONS_H
