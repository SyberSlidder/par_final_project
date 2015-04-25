#include "main.h"

__global__ void combinedSGEMM_v1(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
)
{

    // Shared memory declarations 
    // Local memory used to hold partial updates of the result Matrix
    //__shared__ float C_holder[THREADS_PER_BLOCK];
    // Sub block of A of size THREADS_PER_BLOCK_Y by KSub
    __shared__ float A_Holder[THREADS_PER_BLOCK];
    // Sub block of B of size KSub by THREADS_PER_BLOCK_X
    __shared__ float B_Holder[THREADS_PER_BLOCK];
    
    // Some identity information
    // myID is a linear index for shared memory
    int myID  = threadIdx.x + (blockDim.x)*threadIdx.y;      
    
    // Does not change
    int A_row    = (blockIdx.y*blockDim.y) + threadIdx.y;
    int A_column = threadIdx.x;
    int B_row    = threadIdx.y;
    // Does not change
    int B_column = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    // This threads value of C
    float temp[4]  = {0.0f,0.0f,0.0f,0.0f};
    
    // Load in sublocks of A and B into shared memory
    for (int i = 0; i < K; i += blockDim.x) {
        // Each thread loads in one element from A and one Element from B
        A_Holder[myID] = _A[A_row*K+A_column];
        B_Holder[myID] = _B[B_row*N+B_column]; 
    
        __syncthreads();
    
        // Calculate the partial sum
        int idxA = A_row % blockDim.x;
        int idxB = B_column % blockDim.y;
        for (int j = 0; j < blockDim.x; j += 4) {
            temp[0] += A_Holder[idxA+0]*B_Holder[idxB+0];
            temp[1] += A_Holder[idxA+1]*B_Holder[idxB+1];
            temp[2] += A_Holder[idxA+2]*B_Holder[idxB+2];
            temp[3] += A_Holder[idxA+3]*B_Holder[idxB+3];
	    idxA++;
	    idxB++;
        }
    
        __syncthreads();
    
        // Update indexing information
	A_column += blockDim.x;
	B_row    += blockDim.y;
	
    }

    temp[0] += temp[1] + temp[2] + temp[3];
   
    // We have Cij at this point
    _C[(blockIdx.y*blockDim.y*N)+(blockIdx.x*blockDim.x)+threadIdx.x] = temp[0];
 
}
