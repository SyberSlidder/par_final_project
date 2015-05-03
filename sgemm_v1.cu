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

      #pragma unroll
      for (int j = 0; j < 8 ; j++) {
	A_Holder = sharedA1[rowSelect];
	rowSelect++;
	columnSelect = 8*threadIdx.x;
	#pragma unroll
	for (int k = 0; k < 8; k++) {
	  B_Holder = sharedB1[columnSelect];
	  columnSelect++;
	  partialSums[j][k] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	}
	
	/*B_Holder = sharedB1[columnSelect+0];
	partialSums[j][0] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB1[columnSelect+1];
	partialSums[j][1] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB1[columnSelect+2];
	partialSums[j][2] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB1[columnSelect+3];
	partialSums[j][3] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB1[columnSelect+4];
	partialSums[j][4] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB1[columnSelect+5];
	partialSums[j][5] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB1[columnSelect+6];
	partialSums[j][6] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;	
	B_Holder = sharedB1[columnSelect+7];
	partialSums[j][7] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;	
	*/
      }
    
      rowSelect    = 8*threadIdx.y;

      __syncthreads();
      
      // Compute from A2/B2
      #pragma unroll
      for (int j = 0; j < 8 ; j++) {
	A_Holder = sharedA2[rowSelect];
	rowSelect++;
	columnSelect = 8*threadIdx.x;
	#pragma unroll
	for (int k = 0; k < 8; k++) {
	  B_Holder = sharedB2[columnSelect];
	  columnSelect++;
	  partialSums[j][k] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	}
	/*
	B_Holder = sharedB2[columnSelect+0];
	partialSums[j][0] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB2[columnSelect+1];
	partialSums[j][1] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB2[columnSelect+2];
	partialSums[j][2] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB2[columnSelect+3];
	partialSums[j][3] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB2[columnSelect+4];
	partialSums[j][4] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB2[columnSelect+5];
	partialSums[j][5] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	B_Holder = sharedB2[columnSelect+6];
	partialSums[j][6] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;	
	B_Holder = sharedB2[columnSelect+7];
	partialSums[j][7] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;	
	*/
      }      
      
    
    }
  
    // Write back C
    /*int C_row = loadRowA;
    int C_column = loadRowB;
    
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
	_C[(C_row+i)*N + C_column+j] = partialSums[i][j];
      }
    }
    */
    
    float4 C_holder[2];
    int C_row    = (blockIdx.y * (blockDim.y*8)) + (8*threadIdx.y);
    int C_column = (blockIdx.x * (blockDim.x*8)) + (8*threadIdx.x);
    
    // By Row
    for (int i = 0; i < 8; i++) {
	C_holder[0].x = partialSums[i][0];
	C_holder[0].y = partialSums[i][1];
	C_holder[0].z = partialSums[i][2];
	C_holder[0].w = partialSums[i][3];
	*((float4 *)(_C + (C_row+i)*N + C_column)) = C_holder[0];
	C_holder[1].x = partialSums[i][4];
	C_holder[1].y = partialSums[i][5];
	C_holder[1].z = partialSums[i][6];
	C_holder[1].w = partialSums[i][7];
	*((float4 *)(_C + (C_row+i)*N + C_column + 4)) = C_holder[1];
    }
    
}

__global__ void combinedSGEMM_v3(
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

		  
    float A_Holder[4];
    float B_Holder[4];
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

      rowSelect    = 8*threadIdx.y + threadIdx.y;

      int threadOffset = threadIdx.x;
      
      #pragma unroll
      for (int j = 0; j < 8 ; j++) {
	threadOffset = threadOffset % 8;
	A_Holder[0] = *(((float *)(sharedA1+rowSelect))+0+threadOffset);
	A_Holder[1] = *(((float *)(sharedA1+rowSelect))+1+threadOffset);
	A_Holder[2] = *(((float *)(sharedA1+rowSelect))+2+threadOffset);
	A_Holder[3] = *(((float *)(sharedA1+rowSelect))+3+threadOffset);

	rowSelect++;
	columnSelect = 8*threadIdx.x;
	#pragma unroll
	for (int k = 0; k < 8; k++) {
	  B_Holder[0] = *(((float *)(sharedB1+columnSelect))+0);
	  B_Holder[1] = *(((float *)(sharedB1+columnSelect))+1);
	  B_Holder[2] = *(((float *)(sharedB1+columnSelect))+2);
	  B_Holder[3] = *(((float *)(sharedB1+columnSelect))+3);  
	  columnSelect++;
	  partialSums[j][k] += A_Holder[0]*B_Holder[0] + A_Holder[1]*B_Holder[1] + A_Holder[2]*B_Holder[2] + A_Holder[3]*B_Holder[3];
	}
	threadOffset++;
      }
    
      rowSelect    = 8*threadIdx.y;

      __syncthreads();
      
      // Compute from A2/B2
      #pragma unroll
      for (int j = 0; j < 8 ; j++) {
	A_Holder[0] = *(((float *)(sharedA2+rowSelect))+0);
	A_Holder[1] = *(((float *)(sharedA2+rowSelect))+2);
	A_Holder[2] = *(((float *)(sharedA2+rowSelect))+3);
	A_Holder[3] = *(((float *)(sharedA2+rowSelect))+4);
	rowSelect++;
	columnSelect = 8*threadIdx.x;
	#pragma unroll
	for (int k = 0; k < 8; k++) {
	  B_Holder[0] = *(((float *)(sharedB2+columnSelect))+0);
	  B_Holder[1] = *(((float *)(sharedB2+columnSelect))+1);
	  B_Holder[2] = *(((float *)(sharedB2+columnSelect))+2);
	  B_Holder[3] = *(((float *)(sharedB2+columnSelect))+3);  
	  columnSelect++;
	  partialSums[j][k] += A_Holder[0]*B_Holder[0] + A_Holder[1]*B_Holder[1] + A_Holder[2]*B_Holder[2] + A_Holder[3]*B_Holder[3];
	}

      }      
      
    
    }
  
    // Write back C
    /*int C_row = loadRowA;
    int C_column = loadRowB;
    
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
	_C[(C_row+i)*N + C_column+j] = partialSums[i][j];
      }
    }
    */
    
    float4 C_holder[2];
    int C_row    = (blockIdx.y * (blockDim.y*8)) + (8*threadIdx.y);
    int C_column = (blockIdx.x * (blockDim.x*8)) + (8*threadIdx.x);
    
    // By Row
    for (int i = 0; i < 8; i++) {
	C_holder[0].x = partialSums[i][0];
	C_holder[0].y = partialSums[i][1];
	C_holder[0].z = partialSums[i][2];
	C_holder[0].w = partialSums[i][3];
	*((float4 *)(_C + (C_row+i)*N + C_column)) = C_holder[0];
	C_holder[1].x = partialSums[i][4];
	C_holder[1].y = partialSums[i][5];
	C_holder[1].z = partialSums[i][6];
	C_holder[1].w = partialSums[i][7];
	*((float4 *)(_C + (C_row+i)*N + C_column + 4)) = C_holder[1];
    }
    
}

__global__ void combinedSGEMM_v4(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
) {
  
  __shared__ float sharedA1[4][64];
  __shared__ float sharedA2[4][64];
  __shared__ float sharedB1[4][64];
  __shared__ float sharedB2[4][64];

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
      //sharedA1[linearThreadID] = *((float4 *)a1ReadPtr);
      A_Holder = *((float4 *)a1ReadPtr); //A_Holder is a float4 type, store its x,y,z,w in vertically in sharedA1
      sharedA1[0][linearThreadID] = A_Holder.x;
      sharedA1[1][linearThreadID] = A_Holder.y;
      sharedA1[2][linearThreadID] = A_Holder.z;
      sharedA1[3][linearThreadID] = A_Holder.w;
      // Load from B into B1
      //sharedB1[linearThreadID] = *((float4 *)b1ReadPtr);
      B_Holder = *((float4 *)b1ReadPtr); //B_Holder is a float4 type, store its x,y,z,w in vertically in sharedB1
      sharedB1[0][linearThreadID] = B_Holder.x;
      sharedB1[1][linearThreadID] = B_Holder.y;
      sharedB1[2][linearThreadID] = B_Holder.z;
      sharedB1[3][linearThreadID] = B_Holder.w;
      // Update pointers
      a1ReadPtr += 8;
      b1ReadPtr += 8;  
      
      __syncthreads();
      
      //sharedA2[linearThreadID] = *((float4 *)a2ReadPtr);
      //sharedB2[linearThreadID] = *((float4 *)b2ReadPtr);
      A_Holder = *((float4 *)a2ReadPtr); //A_Holder is a float4 type, store its x,y,z,w in vertically in sharedA2
      sharedA2[0][linearThreadID] = A_Holder.x;
      sharedA2[1][linearThreadID] = A_Holder.y;
      sharedA2[2][linearThreadID] = A_Holder.z;
      sharedA2[3][linearThreadID] = A_Holder.w;

      B_Holder = *((float4 *)b2ReadPtr); //B_Holder is a float4 type, store its x,y,z,w in vertically in sharedB2
      sharedB2[0][linearThreadID] = B_Holder.x;
      sharedB2[1][linearThreadID] = B_Holder.y;
      sharedB2[2][linearThreadID] = B_Holder.z;
      sharedB2[3][linearThreadID] = B_Holder.w;

      a2ReadPtr += 8;
      b2ReadPtr += 8;
      
      // Compute from A1/B1

      rowSelect    = 8*threadIdx.y;
      int rowOffset = linearThreadID % 32;
      
      #pragma unroll
      for (int j = 0; j < 8 ; j++) {
	//A_Holder = sharedA1[rowSelect];
	A_Holder.x = sharedA1[0][rowSelect];
	A_Holder.y = sharedA1[1][rowSelect];
	A_Holder.z = sharedA1[2][rowSelect];
	A_Holder.w = sharedA1[3][rowSelect];
	rowSelect++;
	rowOffset = (rowOffset + 1) % 32;
	columnSelect = 8*threadIdx.x;
	int columnOffset = linearThreadID % 32;
	#pragma unroll
	for (int k = 0; k < 8; k++) {
	  //B_Holder = sharedB1[columnSelect];
	  B_Holder.x = sharedB1[0][columnSelect];
	  B_Holder.y = sharedB1[1][columnSelect];
	  B_Holder.z = sharedB1[2][columnSelect];
	  B_Holder.w = sharedB1[3][columnSelect];
	  columnSelect++;
	  columnOffset = (columnOffset + 1) % 32;
	  partialSums[j][k] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	}

      }
    
      rowSelect    = 8*threadIdx.y;

      __syncthreads();
      
      // Compute from A2/B2
      #pragma unroll
      for (int j = 0; j < 8 ; j++) {
	//A_Holder = sharedA2[rowSelect];
	A_Holder.x = sharedA2[0][rowSelect];
	A_Holder.y = sharedA2[1][rowSelect];
	A_Holder.z = sharedA2[2][rowSelect];
	A_Holder.w = sharedA2[3][rowSelect];
	rowSelect++;
	columnSelect = 8*threadIdx.x;
	#pragma unroll
	for (int k = 0; k < 8; k++) {
	  //B_Holder = sharedB2[columnSelect];
	  B_Holder.x = sharedB2[0][columnSelect];
	  B_Holder.y = sharedB2[1][columnSelect];
	  B_Holder.z = sharedB2[2][columnSelect];
	  B_Holder.w = sharedB2[3][columnSelect];
	  columnSelect++;
	  partialSums[j][k] += A_Holder.x*B_Holder.x + A_Holder.y*B_Holder.y + A_Holder.z*B_Holder.z + A_Holder.w*B_Holder.w;
	}
	
      }      
    
    }
  
    // Write back C
    /*int C_row = loadRowA;
    int C_column = loadRowB;
    
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
	_C[(C_row+i)*N + C_column+j] = partialSums[i][j];
      }
    }
    */
    
    float4 C_holder[2];
    int C_row    = (blockIdx.y * (blockDim.y*8)) + (8*threadIdx.y);
    int C_column = (blockIdx.x * (blockDim.x*8)) + (8*threadIdx.x);
    
    // Access A and B squared sums
    // Reuse the holders
    
    int sqSumVecA_index = (blockIdx.y*blockDim.y*8);
    int sqSumVecB_index = (blockIdx.x*blockDim.x*8);
    float4 A2_Holder,B2_Holder;
    
    A_Holder.x = sqSumVecA[sqSumVecA_index];
    A_Holder.y = sqSumVecA[sqSumVecA_index+1];
    A_Holder.z = sqSumVecA[sqSumVecA_index+2];
    A_Holder.w = sqSumVecA[sqSumVecA_index+3];
    A2_Holder.x = sqSumVecA[sqSumVecA_index+4];
    A2_Holder.y = sqSumVecA[sqSumVecA_index+5];
    A2_Holder.z = sqSumVecA[sqSumVecA_index+6];
    A2_Holder.w = sqSumVecA[sqSumVecA_index+7];

    B_Holder.x = sqSumVecB[sqSumVecB_index];
    B_Holder.y = sqSumVecB[sqSumVecB_index+1];
    B_Holder.z = sqSumVecB[sqSumVecB_index+2];
    B_Holder.w = sqSumVecB[sqSumVecB_index+3];
    B2_Holder.x = sqSumVecB[sqSumVecB_index+4];
    B2_Holder.y = sqSumVecB[sqSumVecB_index+5];
    B2_Holder.z = sqSumVecB[sqSumVecB_index+6];
    B2_Holder.w = sqSumVecB[sqSumVecB_index+7];

    
    // By Row
    #pragma unroll
    for (int i = 0; i < 8; i++) {
	C_holder[0].x = exp(2.0f*partialSums[i][0] + A_Holder.x + B_Holder.x);
	C_holder[0].y = exp(2.0f*partialSums[i][1]+ A_Holder.y + B_Holder.y);
	C_holder[0].z = exp(2.0f*partialSums[i][2]+ A_Holder.z + B_Holder.z);
	C_holder[0].w = exp(2.0f*partialSums[i][3]+ A_Holder.w + B_Holder.w);
	*((float4 *)(_C + (C_row+i)*N + C_column)) = C_holder[0];
	C_holder[1].x = exp(2.0f*partialSums[i][4] + A2_Holder.x + B2_Holder.x);
	C_holder[1].y = exp(2.0f*partialSums[i][5] + A2_Holder.y + B2_Holder.y);
	C_holder[1].z = exp(2.0f*partialSums[i][6] + A2_Holder.z + B2_Holder.z);
	C_holder[1].w = exp(2.0f*partialSums[i][7] + A2_Holder.w + B2_Holder.w);
	*((float4 *)(_C + (C_row+i)*N + C_column + 4)) = C_holder[1];
    }
    
}


















