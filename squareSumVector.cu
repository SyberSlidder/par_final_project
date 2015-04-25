#include "main.h"

#if __CUDA_ARCH__ >= 300
__device__ inline float warpReduce(float value, int laneID){
    // Use XOR mode to perform butterfly reduction
    #pragma unroll
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor(value, i, 32);
    return value;
}
#else
__device__ inline float warpReduce(float value, int laneID){
    volatile __shared__ float values[1024];
    values[threadIdx.x] = 0.0;
    values[threadIdx.x] = value;
    if(laneID < 16){
        for(int i=16; i>=1; i/=2){
            values[threadIdx.x] += values[threadIdx.x+i];
        }
    }
    
    return values[threadIdx.x];
}
#endif


__global__ void calcSquareSumVector(float *srcMatrix,
                                    float *sqSumVector,
                                    int    M,
                                    int    K){
                                    
    // Shared data
    volatile __shared__ float sdata[32];

    // Calculate thread index and stride
    int laneId = threadIdx.x & 0x1f;
    int icol   = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    int warpId = threadIdx.x/32;

    // Thread-Local sum
    float mySqSum = 0.0;

    // Split rows amongst thread blocks
    for(int row  = blockIdx.y;
            row  < M;
            row += gridDim.y){

        // Strided reduction of squared values across columns
        for(int col  = icol;
                col  < K + blockDim.x;
                col += stride){

            // Square the assignmed matrix cell
            float val = (col >= K) ? 0.0 : srcMatrix[K*row + col];
            float sqVal = val*val;

            // Add to thread-local sum
            mySqSum += sqVal;
        }

        // Warp-level reduction with butterfly shuffles
        float warpSqSum = warpReduce(mySqSum,laneId);

        // Store warp-local square-sum
        if(laneId == 0){
            sdata[warpId] = warpSqSum;
        }
        __syncthreads();

        // Lowest work finishes off work
        if(warpId == 0){
            // Read warp-local square-sums
            mySqSum = sdata[laneId];
            //printf("===%3d %3d %3d %5.2f\n", row, warpId, laneId, mySqSum);

            // Add to block-local square sums
            float blkSqSum = warpReduce(mySqSum,laneId);

            // Store result
            if(laneId == 0){
                sqSumVector[row*gridDim.x+blockIdx.x] = blkSqSum;
            }
        }
    }
}
