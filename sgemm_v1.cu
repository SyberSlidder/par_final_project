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
       float * weight, //N x 1 vector of weight
       float * result, //N x 1 vector of kernel summation result
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
) {
  __shared__ float sharedA1[8][64];
  __shared__ float sharedA2[8][64];
  __shared__ float sharedB1[16][32];
  __shared__ float sharedB2[16][32];

    // Identification
    int linearThreadID = threadIdx.x + (blockDim.x * threadIdx.y);
    
    // Where do I read from A
    int loadRowA = blockIdx.y * (blockDim.y*8); // Should multiple of 64
    loadRowA += linearThreadID; // Which row of A this thread is responsible for loading
    
    float * a1ReadPtr = _A + (loadRowA * K);
    float * a2ReadPtr = _A + (loadRowA * K);
    
    // Where do I read from B
    int loadRowB;
    float * b1ReadPtr;
    float * b2ReadPtr;
    
    if (B_TRANSPOSE) {
      loadRowB = blockIdx.y * (blockDim.y*8);
      loadRowB += linearThreadID;
      b1ReadPtr = _B + (loadRowB * K);
      b2ReadPtr = _B + (loadRowB * K);
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
    int colSelect;
    
      // Load from A into A1
      //sharedA1[linearThreadID] = *((float4 *)a1ReadPtr);
      A_Holder = *((float4 *)a1ReadPtr); //A_Holder is a float4 type, store its x,y,z,w in vertically in sharedA1
      sharedA1[0][linearThreadID] = A_Holder.x;
      sharedA1[1][linearThreadID] = A_Holder.y;
      sharedA1[2][linearThreadID] = A_Holder.z;
      sharedA1[3][linearThreadID] = A_Holder.w;
      a1ReadPtr += 4;
      A_Holder = *((float4 *)a1ReadPtr); //A_Holder is a float4 type, store its x,y,z,w in vertically in sharedA1
      sharedA1[4][linearThreadID] = A_Holder.x;
      sharedA1[5][linearThreadID] = A_Holder.y;
      sharedA1[6][linearThreadID] = A_Holder.z;
      sharedA1[7][linearThreadID] = A_Holder.w;
      a1ReadPtr += 4;

      // Load from B into B1
      //sharedB1[linearThreadID] = *((float4 *)b1ReadPtr);
      B_Holder = *((float4 *)b1ReadPtr); //B_Holder is a float4 type, store its x,y,z,w in vertically in sharedB1
      int col = (linearThreadID/8) * 4 + linearThreadID%4;
      int row = (((linearThreadID/4) & 0x01) == 0) ? 0 : 8;
      sharedB1[row+0][col] = B_Holder.x;
      sharedB1[row+1][col] = B_Holder.y;
      sharedB1[row+2][col]= B_Holder.z;
      sharedB1[row+3][col]= B_Holder.w;
      b1ReadPtr += 4;
      B_Holder = *((float4 *)b1ReadPtr); //B_Holder is a float4 type, store its x,y,z,w in vertically in sharedB1
      sharedB1[row+4][col] = B_Holder.x;
      sharedB1[row+5][col] = B_Holder.y;
      sharedB1[row+6][col] = B_Holder.z;
      sharedB1[row+7][col] = B_Holder.w;
      b1ReadPtr += 4;
      __syncthreads();
    // Loop through the K dimension of Matrices A and B
    for (int i = 0; i < K/8; i++) {
      // Load from A into A2
      //sharedA2[linearThreadID] = *((float4 *)a2ReadPtr);
      //sharedB2[linearThreadID] = *((float4 *)b2ReadPtr);
      a2ReadPtr += 8;
      b2ReadPtr += 8;
      A_Holder = *((float4 *)a2ReadPtr); //A_Holder is a float4 type, store its x,y,z,w in vertically in sharedA2
      sharedA2[0][linearThreadID] = A_Holder.x;
      sharedA2[1][linearThreadID] = A_Holder.y;
      sharedA2[2][linearThreadID] = A_Holder.z;
      sharedA2[3][linearThreadID] = A_Holder.w;
      a2ReadPtr += 4;
      A_Holder = *((float4 *)a2ReadPtr); //A_Holder is a float4 type, store its x,y,z,w in vertically in sharedA2
      sharedA2[4][linearThreadID] = A_Holder.x;
      sharedA2[5][linearThreadID] = A_Holder.y;
      sharedA2[6][linearThreadID] = A_Holder.z;
      sharedA2[7][linearThreadID] = A_Holder.w;
      a2ReadPtr += 4;

      // Load from B into B2
      B_Holder = *((float4 *)b2ReadPtr); //B_Holder is a float4 type, store its x,y,z,w in vertically in sharedB2
      sharedB2[0+row][col] = B_Holder.x;
      sharedB2[1+row][col] = B_Holder.y;
      sharedB2[2+row][col] = B_Holder.z;
      sharedB2[3+row][col] = B_Holder.w;
      b2ReadPtr += 4;
      B_Holder = *((float4 *)b2ReadPtr); //B_Holder is a float4 type, store its x,y,z,w in vertically in sharedB2
      sharedB2[4+row][col] = B_Holder.x;
      sharedB2[5+row][col] = B_Holder.y;
      sharedB2[6+row][col] = B_Holder.z;
      sharedB2[7+row][col] = B_Holder.w;
      b2ReadPtr += 4;
      
      // Compute from A1/B1
      rowSelect    = 8*threadIdx.y;
      colSelect    = 4*threadIdx.x;
      for(int track = 0; track < 8; track++){
 	for(int j= 0; j<8; j++){
      		#pragma unroll
	    for(int k=0; k<4; k++)
		partialSums[j][k] += sharedA1[track][rowSelect+j] * sharedB1[track][colSelect+k];	
		//if(linearThreadID == 1)printf("partialSums[%d][%d] += A1[%d][%d]*B1[%d][%d]\n",j,k,track,rowSelect+j,track,colSelect+k);}
      		#pragma unroll
	    for(int k=4; k<8; k++)
		partialSums[j][k] += sharedA1[track][rowSelect+j] * sharedB1[track+8][colSelect+k-4];	
		//if(linearThreadID == 1)printf("partialSums[%d][%d] += A1[%d][%d]*B1[%d][%d]\n",j,k,track,rowSelect+j,track+8,colSelect+k-4);}
	}
      }
/*
	for(int j=0; j< 8; j++)
	    for(int k=0; k<8; k++)
		printf("i=%d	partialsum[%d][%d] = %f\n",i,j,k,partialSums[j][k]);
*/
      i++;
      __syncthreads();

      if((i+1) < K/8){
      // Load from A into A1
      //sharedA1[linearThreadID] = *((float4 *)a1ReadPtr);
      // Update pointers
      a1ReadPtr += 8;
      b1ReadPtr += 8;  
      A_Holder = *((float4 *)a1ReadPtr); //A_Holder is a float4 type, store its x,y,z,w in vertically in sharedA1
      sharedA1[0][linearThreadID] = A_Holder.x;
      sharedA1[1][linearThreadID] = A_Holder.y;
      sharedA1[2][linearThreadID] = A_Holder.z;
      sharedA1[3][linearThreadID] = A_Holder.w;
      a1ReadPtr += 4;
      A_Holder = *((float4 *)a1ReadPtr); //A_Holder is a float4 type, store its x,y,z,w in vertically in sharedA1
      sharedA1[4][linearThreadID] = A_Holder.x;
      sharedA1[5][linearThreadID] = A_Holder.y;
      sharedA1[6][linearThreadID] = A_Holder.z;
      sharedA1[7][linearThreadID] = A_Holder.w;
      a1ReadPtr += 4;

      // Load from B into B1
      //sharedB1[linearThreadID] = *((float4 *)b1ReadPtr);
      B_Holder = *((float4 *)b1ReadPtr); //B_Holder is a float4 type, store its x,y,z,w in vertically in sharedB1
      sharedB1[0+row][col] = B_Holder.x;
      sharedB1[1+row][col] = B_Holder.y;
      sharedB1[2+row][col] = B_Holder.z;
      sharedB1[3+row][col] = B_Holder.w;
      b1ReadPtr += 4;
      B_Holder = *((float4 *)b1ReadPtr); //B_Holder is a float4 type, store its x,y,z,w in vertically in sharedB1
      sharedB1[4+row][col] = B_Holder.x;
      sharedB1[5+row][col] = B_Holder.y;
      sharedB1[6+row][col] = B_Holder.z;
      sharedB1[7+row][col] = B_Holder.w;
      b1ReadPtr += 4;
      //printf("load next A1 B1\n");
      }
      // Compute from A2/B2
      //rowSelect    = 8*threadIdx.y;
      //colSelect    = 4*threadIdx.x;
      for(int track = 0; track < 8; track++){
	for(int j= 0; j< 8; j++){
      		#pragma unroll
	    for(int k=0; k<4; k++)
		partialSums[j][k] += sharedA2[track][rowSelect+j] * sharedB2[track][colSelect+k];	
      		#pragma unroll
	    for(int k=4; k<8; k++)
		partialSums[j][k] += sharedA2[track][rowSelect+j] * sharedB2[track+8][colSelect+k-4];	
	}
      }

/*	for(int j=0; j< 8; j++)
	    for(int k=0; k<8; k++)
		printf("i=%d	partialsum2[%d][%d] = %f\n",i,j,k,partialSums[j][k]);
*/
      __syncthreads();

    } // End of SGEMM
  
    // Write back C
    int C_row = blockIdx.y * (blockDim.y * 8) + threadIdx.y * 8;
    int C_column = blockIdx.x * (blockDim.x * 8) + threadIdx.x * 8;
    //printf("thread (%d,%d)	C[%d][%d]\n",threadIdx.x, threadIdx.y ,C_row,C_column);
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
	_C[(C_row+i)*N + C_column+j] = partialSums[i][j];
	//printf("C[%d][%d] = %f\n",C_row+i,C_column+j,partialSums[i][j]);
      }
    }
/*    
    float4 C_holder[2];
    C_row    = (blockIdx.y * (blockDim.y*8)) + (8*threadIdx.y);
    C_column = (blockIdx.x * (blockDim.x*8)) + (8*threadIdx.x);
    
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

    float * w0ReadPtr = weight + C_row;
    float * w1ReadPtr = weight + C_row+4;
    float4 w0 = *((float4 *)w0ReadPtr);
    float4 w1 = *((float4 *)w1ReadPtr);
    float v;
    __shared__ float vPartialSums[64][8];
    for(int i=0; i<64;i++)
	for(int j=0; j<8; j++)
       	    vPartialSums[i][j] = 0.0f;
    // By Row
    #pragma unroll
    for (int i = 0; i < 8; i++) {
	v = 0.0f;
	C_holder[0].x = exp(-2.0f*partialSums[i][0] + A_Holder.x + B_Holder.x);
	C_holder[0].y = exp(-2.0f*partialSums[i][1]+ A_Holder.y + B_Holder.y);
	C_holder[0].z = exp(-2.0f*partialSums[i][2]+ A_Holder.z + B_Holder.z);
	C_holder[0].w = exp(-2.0f*partialSums[i][3]+ A_Holder.w + B_Holder.w);
	//*((float4 *)(_C + (C_row+i)*N + C_column)) = C_holder[0];
	C_holder[1].x = exp(-2.0f*partialSums[i][4] + A2_Holder.x + B2_Holder.x);
	C_holder[1].y = exp(-2.0f*partialSums[i][5] + A2_Holder.y + B2_Holder.y);
	C_holder[1].z = exp(-2.0f*partialSums[i][6] + A2_Holder.z + B2_Holder.z);
	C_holder[1].w = exp(-2.0f*partialSums[i][7] + A2_Holder.w + B2_Holder.w);
	//printf("partial Sums = %f, %f, %f, %f 	",partialSums[i][0],partialSums[i][1],partialSums[i][2],partialSums[i][3]);
	//printf("C=%f, %f, %f, %f	",C_holder[0].x,C_holder[0].y,C_holder[0].z,C_holder[0].w);
	//*((float4 *)(_C + (C_row+i)*N + C_column + 4)) = C_holder[1];
	v =  C_holder[0].x * w0.x + C_holder[0].y * w0.y + C_holder[0].z * w0.z + C_holder[0].w * w0.w
		+ C_holder[1].x * w1.x + C_holder[1].y * w1.y + C_holder[1].z * w1.z + C_holder[1].w * w1.w;
	//printf("v=%f\n",v);
        vPartialSums[threadIdx.y*8+i][threadIdx.x] = v;
    }
	__syncthreads();
    
    //Each thread does a row reduction on vPartialSums, save sum in register v
	v = vPartialSums[linearThreadID][0] + vPartialSums[linearThreadID][1] + vPartialSums[linearThreadID][2] + vPartialSums[linearThreadID][3] + vPartialSums[linearThreadID][4] + vPartialSums[linearThreadID][5] + vPartialSums[linearThreadID][6] + vPartialSums[linearThreadID][7];
	//printf("v=%f\n",v);
	//v is only partial sum of result[C_row]
	// solution 1: use atomic
	atomicAdd(result+blockIdx.y*blockDim.y*8+linearThreadID, v);
	// solution 2: store v back to memory resultMatrix[m][n/64], do matrix reduction later to get result[m]
	/*
	*resultMatrix(C_row*N/64+blockIdx.x) = v;
    */
    
    
}


















