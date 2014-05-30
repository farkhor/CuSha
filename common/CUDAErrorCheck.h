#ifndef	_CUDAERRCHCK_H
#define	_CUDAERRCHCK_H

//Error checking mechanism
#define CUDAErrorCheck(err) { CUDAAssert((err), __FILE__, __LINE__); }
inline void CUDAAssert( cudaError_t err, char *file, int line, bool abort=true )
{
   if ( err != cudaSuccess )
   {
      fprintf( stderr, "CUDAAssert: %s %s %d\n", cudaGetErrorString(err), file, line );
      if ( abort )
    	  exit(err);
   }
}

#endif
