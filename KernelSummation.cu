#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <cuda.h>
#include <cublas.h>

#include <inplace/transpose.h>

using namespace std;

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK 8*8
#define BLOCKS 64

// Transpose parameters
#define TILE_DIM    16
#define BLOCK_ROWS  16


// Warp reduction functions
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

// Square-sum reduction of matrix rows kernel
__global__ void calcSquareSumVector(float *srcMatrix,
                                    float *sqSumVector,
                                    int    M,
                                    int    K){
                                    
    // Shared data
    volatile __shared__ float sdata[32];

    // Calculate thread index and stride
    int laneId = threadIdx.x & 0x1f;
    int icol   = threadIdx.x;
    int stride = blockDim.x;
    int warpId = threadIdx.x/32;

    

    // Initialize shared data
    if(warpId == 0)
        sdata[laneId] = 0;
    __syncthreads();

    // Split rows amongst thread blocks
    for(int row  = blockIdx.y;
            row  < M;
            row += gridDim.y){

        // Thread-Local sum
        float mySqSum = 0.0;

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
                sqSumVector[row] = blkSqSum;
            }
        }
    }
}


int main(int argc, char * argv[]) {

    // Matrix dimensions
    int    M, N, K;
    
    // Host pointers
    float *hostA;
    float *hostB;
    float *hostC;
    float *hostSqSumVecA;
    float *hostSqSumVecB;
    
    // Device pointers
    float *devA;
    float *devB;
    float *devC;
    float *devSqSumVecA;
    float *devSqSumVecB;
    
    // Kernel parameters
    dim3 gridSize;
    dim3 blockSize;
    
    // Device parameters
    cudaDeviceProp  deviceProp;
    
    ////////////////////////////////////////////////
    //           MEMORY INITIALIZATION            //
    ////////////////////////////////////////////////

    // Check for proper arguments
    if(argc != 4){
        fprintf(stderr, "USAGE: ./kernelSummation M N K\n");
        exit(EXIT_SUCCESS);
    }
    
    // Get matrix dimensions
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    
    // Check for valid arguments
    if(M <= 0 || N <= 0 || K <= 0){
        fprintf(stderr, "ERROR: One of the dimensions is <=0\n");
        exit(EXIT_SUCCESS);
    }

    // Determine size of memory and check memory limits
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t requiredDeviceMem = 0;
    requiredDeviceMem += M*K*sizeof(float); // Matrix A
    requiredDeviceMem += K*N*sizeof(float); // Matrix B
    requiredDeviceMem += M*N*sizeof(float); // Matrix C
    requiredDeviceMem +=   M*sizeof(float); // Square-Sum Vector A
    requiredDeviceMem +=   N*sizeof(float); // Square-Sum Vector B
    if(requiredDeviceMem > deviceProp.totalGlobalMem){
        fprintf(stderr, "ERROR: Data is too large for device\n");
        exit(EXIT_SUCCESS);
    }

    
    // Allocate host memory
    hostA = (float*)malloc(M*K*sizeof(float));
    hostB = (float*)malloc(K*N*sizeof(float));
    hostC = (float*)malloc(M*N*sizeof(float));

    hostSqSumVecA = (float*)malloc(M*sizeof(float));
    hostSqSumVecB = (float*)malloc(N*sizeof(float));
    
    // Allocate device memory
    cudaMalloc((void**)&devA,  M*K*sizeof(float));
    cudaMalloc((void**)&devB,  K*N*sizeof(float));
    cudaMalloc((void**)&devC,  M*N*sizeof(float));
    
    cudaMalloc((void**)&devSqSumVecA, M*sizeof(float));
    cudaMemset(devSqSumVecA, 0, M*sizeof(float));
    cudaMalloc((void**)&devSqSumVecB, N*sizeof(float));
    cudaMemset(devSqSumVecB, 0, N*sizeof(float));
    
    
    
    ////////////////////////////////////////////////
    //             DATA INITIALIZATION            //
    ////////////////////////////////////////////////
    
    // Initialize data on host
    srand(0);
    for(int m = 0; m < M; m++){
        for(int k = 0; k < K; k++){
            hostA[m*K+k] = (float)rand()/(float)(RAND_MAX/10);
        }
    }
    for(int k = 0; k < K; k++){
        for(int n = 0; n < N; n++){
            hostB[k*N+n] = (float)rand()/(float)(RAND_MAX/10);
        }
    }    
    
    
    ////////////////////////////////////////////////
    //              KERNEL SUMMATION              //
    ////////////////////////////////////////////////
    
    // Transfer host data to device
    cudaMemcpy(devA,hostA,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devB,hostB,K*N*sizeof(float),cudaMemcpyHostToDevice);
    
    // Calculate grid and block size
    gridSize.x = (N + 1)/THREADS_PER_BLOCK_X;
    gridSize.y = (M + 1)/THREADS_PER_BLOCK_Y;
    gridSize.z = 1;
    blockSize.x = THREADS_PER_BLOCK_X;
    blockSize.y = THREADS_PER_BLOCK_Y;
    blockSize.z = 1;
    
    // Square-Sum of A
    gridSize.x  = 1;
    gridSize.y  = min(M,deviceProp.maxGridSize[1]);
    blockSize.x = min(1024,max(32,(K/32)*32));
    blockSize.y = 1;
    calcSquareSumVector<<<gridSize,blockSize>>>(devA,devSqSumVecA,M,K);
    
    // Transpose matrix B
    inplace::transpose(true,devB,K,N);
    
    // Square-Sum of B
    gridSize.x  = 1;
    gridSize.y  = min(N,deviceProp.maxGridSize[1]);
    blockSize.x = min(1024,max(32,(K/32)*32));
    blockSize.y = 1;
    calcSquareSumVector<<<gridSize,blockSize>>>(devB,devSqSumVecB,N,K);
    
    // Transfer result from device to host
    cudaMemcpy(hostSqSumVecA,devSqSumVecA,M*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSqSumVecB,devSqSumVecB,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostC,devC,M*N*sizeof(float),cudaMemcpyDeviceToHost);
    
    
    ////////////////////////////////////////////////
    //            RESULT VERIFICATION             //
    ////////////////////////////////////////////////
    
    // Verify square-sum of A
    for(int m = 0; m < M; m++){
        float sqSum = 0.0;
        for(int k = 0; k < K; k++){
            sqSum += hostA[m*K+k]*hostA[m*K+k];
        }
        if(int(sqSum) != int(hostSqSumVecA[m])){
            fprintf(stderr, " Bad square sum: [m = %d] [refSqSum = %f] [devSqSum = %f]\n", m, sqSum, hostSqSumVecA[m]);
            //exit(EXIT_FAILURE);
        }
        else
            fprintf(stderr, "Good square sum: [m = %d] [refSqSum = %f] [devSqSum = %f]\n", m, sqSum, hostSqSumVecA[m]);
    }


    // Verify square-sum of B
    for(int n = 0; n < N; n++){
        float sqSum = 0.0;
        for(int k = 0; k < K; k++){
            sqSum += hostB[k*N+n]*hostB[k*N+n];
        }
        if(int(sqSum) != int(hostSqSumVecB[n])){
            fprintf(stderr, " Bad square sum: [n = %d] [refSqSum = %f] [devSqSum = %f]\n", n, sqSum, hostSqSumVecB[n]);
            //exit(EXIT_FAILURE);   
	}
        else
            fprintf(stderr, "Good square sum: [n = %d] [refSqSum = %f] [devSqSum = %f]\n", n, sqSum, hostSqSumVecB[n]);
    }
    
    
    
    
    
    ////////////////////////////////////////////////
    //             FREE MEMORY & EXIT             //
    ////////////////////////////////////////////////
    
    // Free host memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostSqSumVecA);
    free(hostSqSumVecB);
    
    // Free device memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFree(devSqSumVecA);
    cudaFree(devSqSumVecB);
    
    return 0;
}
