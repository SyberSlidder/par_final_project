#include "main.h"



__global__ void GPU2(float *a, float *b, float *c, int n)
{// thread code to compute a 1 x 2 sub-matrix of c
// cast a, b, and c into types suitable for I/O
  float4 *a4 = (float4 *) a;
  float2 *b2 = (float2 *) b;
  float2 *c2 = (float2 *) c;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  int nDiv2 = n/2;
  int nDiv4 = n/4;
  int aNext = i*nDiv4;
  int bNext = j;
  float2 temp2;
  temp2.x = temp2.y = 0;
  for (int k = 0; k < nDiv4; k++) {
    float4 aIn = a4[aNext++];
    float2 bIn = b2[bNext];
    
    temp2.x += aIn.x*bIn.x; temp2.y += aIn.x*bIn.y;
    bNext += nDiv2;
    
    bIn = b2[bNext];
    temp2.x += aIn.y*bIn.x; temp2.y += aIn.y*bIn.y;
    bNext += nDiv2;
    bIn = b2[bNext];
    temp2.x += aIn.z*bIn.x; temp2.y += aIn.z*bIn.y;
    bNext += nDiv2;
    bIn = b2[bNext];
    temp2.x += aIn.w*bIn.x; temp2.y += aIn.w*bIn.y;
    bNext += nDiv2;
    
  }
  c2[i*nDiv2+j] = temp2;
}

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

    // Load in the elements from the squared reduced vectors of size Mx1 and 1xN
    float M_row_reduced    = sqSumVecA[A_row];
    float N_column_reduced = sqSumVecB[B_column];
    
    // We have Cij at this point
    temp[0] += temp[1] + temp[2] + temp[3];
    
    //temp[0] = 2*temp[0] + M_row_reduced + N_column_reduced;
    
    // Kernel evaluation
    //temp[0] = exp(temp[0]);
    
    // We have Cij at this point
    _C[(blockIdx.y*blockDim.y*N)+(blockIdx.x*blockDim.x)+threadIdx.x] = temp[0];
 
}

#define MAXWELL_MICROTILE_SIZE 8
#define B_TRANSPOSE true

__global__ void combinedSGEMM_v2(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
) {
  
  __shared__ float4 sharedA1[64];
  __shared__ float4 sharedA2[64];
  __shared__ float4 sharedB1[64];
  __shared__ float4 sharedB2[64];

    // Identification
    int linearThreadID = threadIdx.x + (blockDim.x * threadIdx.y);
    
    // Where do I read from A
    int loadRowA = blockIdx.y * (blockDim.y*8); // Should multiple of 64
    loadRowA += linearThreadID; // Which row of A this thread is responsible for loading
    
    float * a1ReadPtr = _A + (loadRowA * K);
    float * a2ReadPtr = a1ReadPtr + 4;
    
    // Where do I read from B
    int loadRowB;
    float * b1ReadPtr;
    float * b2ReadPtr;
    
    if (B_TRANSPOSE) {
      loadRowB = blockIdx.y * (blockDim.y*8);
      loadRowB += linearThreadID;
      b1ReadPtr = _B + (loadRowB * K);
      b2ReadPtr = b1ReadPtr + 4;
    } else {

      
    }

    // Initialization
    float partialSums[MAXWELL_MICROTILE_SIZE][MAXWELL_MICROTILE_SIZE] = { 
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}
		  };

		  
    float4 A_Holder;
    float4 B_Holder;
    int rowSelect;
    int columnSelect;
    
    // Loop through the K dimension of Matrices A and B
    for (int i = 0; i < (K/8); i++) {

      // Load from A into A1
      sharedA1[linearThreadID] = *((float4 *)a1ReadPtr);
      // Load from B into B1
      sharedB1[linearThreadID] = *((float4 *)b1ReadPtr);
      // Update pointers
      a1ReadPtr += 8;
      b1ReadPtr += 8;  
      
      __syncthreads();
      
      sharedA2[linearThreadID] = *((float4 *)a2ReadPtr);
      sharedB2[linearThreadID] = *((float4 *)b2ReadPtr);

      a2ReadPtr += 8;
      b2ReadPtr += 8;
      
      // Compute from A1/B1

      rowSelect    = 8*threadIdx.y;
      columnSelect = 8*threadIdx.x;
      
      #pragma unroll
      for (int j = 0; j < 8 ; j++) {
	A_Holder = sharedA1[rowSelect];
	rowSelect++;
	#pragma unroll
	for (int k = 0; k < 8; k++) {
	  B_Holder = sharedB1[columnSelect];
	  columnSelect++;
	  partialSums[j][k] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	}
	
      }
    
      rowSelect    = 8*threadIdx.y;
      columnSelect = 8*threadIdx.x;
      
      __syncthreads();
      
      // Compute from A2/B2
      #pragma unroll
      for (int j = 0; j < 8 ; j++) {
	A_Holder = sharedA2[rowSelect];
	rowSelect++;
	#pragma unroll
	for (int k = 0; k < 8; k++) {
	  B_Holder = sharedB2[columnSelect];
	  columnSelect++;
	  partialSums[j][k] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	}
	
      }      
      
    
    }
  
    // Write back C
    int C_row = 0;
    int C_column = 0;
    
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
	_C[C_row*0 + C_column] = partialSums[i][j];
      }
    }
    
}




















