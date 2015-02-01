#ifndef CUDA_UTILITIES_CUH
#define CUDA_UTILITIES_CUH

#include <stdexcept>

#include "cuda_error_check.cuh"


template <typename T>
class host_pinned_buffer{
private:
	T* ptr;
	size_t nElems;
	void construct(size_t n){
		CUDAErrorCheck( cudaHostAlloc( (void**)&ptr, n*sizeof(T), cudaHostAllocPortable ) );
		nElems = n;
	}
public:
	host_pinned_buffer(){
		nElems = 0;
		ptr = NULL;
	}
	host_pinned_buffer(size_t n){
		construct(n);
	}
	~host_pinned_buffer(){
		if( nElems!=0 )
			CUDAErrorCheck( cudaFreeHost( ptr ) );
	}
	void alloc(size_t n){
		if( nElems==0 )
			construct(n);
	}
	void free(){
		if( nElems!=0 ) {
			nElems = 0;
			CUDAErrorCheck( cudaFreeHost( ptr ) );
		}
	}
	T& at(size_t index){
		if( index >= nElems )
			throw std::runtime_error( "The referred element does not exist in the buffer." );
		return ptr[index];
	}
	T& operator[](size_t index){
		return this->at(index);
	}
	T* get_ptr(){
		return ptr;
	}
	size_t size(){
		return nElems;
	}
	size_t sizeInBytes(){
		return nElems*sizeof(T);
	}
};

template <typename T>
class device_buffer{
private:
	T* ptr;
	size_t nElems;
	void construct(size_t n) {
		CUDAErrorCheck( cudaMalloc( (void**)&ptr, n*sizeof(T) ) );
		nElems = n;
	}
public:
	device_buffer():
		nElems(0), ptr(NULL)
	{}
	device_buffer(size_t n){
		construct(n);
	}
	~device_buffer(){
		if( nElems!=0 )
			CUDAErrorCheck( cudaFree( ptr ) );
	}
	void alloc(size_t n){
		if( nElems==0 )
			construct(n);
	}
	void free(){
		if( nElems!=0 ) {
			nElems = 0;
			CUDAErrorCheck( cudaFree( ptr ) );
		}
	}
	T* get_ptr(){
		return ptr;
	}
	size_t size(){
		return nElems;
	}
	size_t sizeInBytes(){
		return nElems*sizeof(T);
	}
	device_buffer<T>& operator=( host_pinned_buffer<T>& srcHostBuffer )
	{
	    if( nElems == 0 ) {
	    	construct( srcHostBuffer.size() );
	    	CUDAErrorCheck( cudaMemcpyAsync( ptr, srcHostBuffer.get_ptr(), srcHostBuffer.sizeInBytes(), cudaMemcpyHostToDevice ) );
	    }
	    else {
	    	size_t copySize = ( srcHostBuffer.sizeInBytes() < this->sizeInBytes() ) ? srcHostBuffer.sizeInBytes() : this->sizeInBytes();
	    	CUDAErrorCheck( cudaMemcpyAsync( ptr, srcHostBuffer.get_ptr(), copySize, cudaMemcpyHostToDevice ) );
	    }
	    return *this;
	}
};



#endif	//	CUDA_UTILITIES_CUH
