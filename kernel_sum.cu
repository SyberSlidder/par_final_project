#include "main.h"


__global__ void KernelSumFromC(
       float * _C,
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       float * _W, // Weight Vector
       float * _V, // Result Vector
       int M, // Number rows of A
       int N // Number of columns of B
) {
  
    float v;
    __shared__ float vPartialSums[64][8];
    for(int i=0; i<64;i++)
	for(int j=0; j<8; j++)
       	    vPartialSums[i][j] = 0.0f;
    
    int linearThreadID = threadIdx.x + (blockDim.x * threadIdx.y);
  
    // Load C in
  
    float * cReadPtr = _C + (blockIdx.y*64*N) + (blockIdx.x*64) + (threadIdx.x*8) + (threadIdx.y*8*N); 
    float4 cHolder[8][2];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      cHolder[i][0] = *((float4 *)(_C));
      cHolder[i][1] = *((float4 *)(_C + 4));
    }
    
    // Load the square sum vectors
    float * sqSumVecA_ptr = (blockIdx.y*blockDim.y*8) + sqSumVecA;
    float * sqSumVecB_ptr = (blockIdx.x*blockDim.x*8) + sqSumVecB;    
    float4 sqSumA_holder[2];
    float4 sqSumB_holder[2];
    
    sqSumA_holder[0] = *((float4 *)(sqSumVecA_ptr));
    sqSumA_holder[1] = *((float4 *)(sqSumVecA_ptr+4));
    sqSumB_holder[0] = *((float4 *)(sqSumVecB_ptr));
    sqSumB_holder[1] = *((float4 *)(sqSumVecB_ptr+4));
    
    // Load the weight vector
    float * wReadPtr = _W + (blockIdx.y * (blockDim.y*8)) + (8*threadIdx.y);
    
    float4 wHolder[2];
    
    wHolder[0] = *((float4 *)(wReadPtr));
    wHolder[1] = *((float4 *)(wReadPtr+4)); 
    
    float rowReductions[8];
    // By Row
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      rowReductions[i] = 0.0f;
      rowReductions[i] += wHolder[0].x * exp(-2.0f*cHolder[i][0].x + sqSumA_holder[0].x + sqSumB_holder[0].x);
      rowReductions[i] += wHolder[0].y * exp(-2.0f*cHolder[i][0].y + sqSumA_holder[0].y + sqSumB_holder[0].y);
      rowReductions[i] += wHolder[0].z * exp(-2.0f*cHolder[i][0].z + sqSumA_holder[0].z + sqSumB_holder[0].z);
      rowReductions[i] += wHolder[0].w * exp(-2.0f*cHolder[i][0].w + sqSumA_holder[0].w + sqSumB_holder[0].w);
      rowReductions[i] += wHolder[1].x * exp(-2.0f*cHolder[i][1].x + sqSumA_holder[1].x + sqSumB_holder[1].x);
      rowReductions[i] += wHolder[1].y * exp(-2.0f*cHolder[i][1].y + sqSumA_holder[1].y + sqSumB_holder[1].y);
      rowReductions[i] += wHolder[1].z * exp(-2.0f*cHolder[i][1].z + sqSumA_holder[1].z + sqSumB_holder[1].z);
      rowReductions[i] += wHolder[1].w * exp(-2.0f*cHolder[i][1].w + sqSumA_holder[1].w + sqSumB_holder[1].w);
      vPartialSums[threadIdx.y*8+i][threadIdx.x] = rowReductions[i];
    }
      
    __syncthreads();
    
    v = vPartialSums[linearThreadID][0] + vPartialSums[linearThreadID][1] + vPartialSums[linearThreadID][2] + vPartialSums[linearThreadID][3] + vPartialSums[linearThreadID][4] + vPartialSums[linearThreadID][5] + vPartialSums[linearThreadID][6] + vPartialSums[linearThreadID][7];
      
    atomicAdd(_V+blockIdx.y*blockDim.y*8+linearThreadID, v);
    
}