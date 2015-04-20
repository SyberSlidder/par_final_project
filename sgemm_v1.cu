#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cublas.h>
#include "cycleTimer.h"

using namespace std;

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK 8*8
#define BLOCKS 64

/*
__global__ void calcSquareSumVector(float *srcMatrix,
                                    float *sqSumVector,
                                    int    M,
                                    int    K){
                                    
    // Shared data
    float *sData[10
    
    // Calculate data index for thread
    int rowIndex    = threadIdx.x;
    int columnIndex = blockIdx.x; 
    
    // Strided reduction of squared values
    for(rowIndex = threadIdx.x; rowIndex < K+blockDim.x; rowIndex += blockDim.x){

        // Square the assignmed matrix cell
        float val = (rowIndex > K-1) ? 0.0 : srcMatrix[K*columnIndex + rowIndex];
        float sqVal = val*val;













    }




}
*/

__global__ void combinedSGEMM(
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
    __shared__ float C_holder[THREADS_PER_BLOCK];
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






int main(int argc, char * argv[]) {

    // Matrix dimensions
    int    M, N, K;
    
    // Host pointers
    float *hostA;
    float *hostB;
    float *hostC;
    
    // Device pointers
    float *devA;
    float *devB;
    float *devC;
    float *sqSumVecA;
    float *sqSumVecB;
    
    // Kernel parameters
    dim3 gridSize;
    dim3 blockSize;
    
    
    
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
    
    // Allocate host memory
    hostA = (float*)malloc(M*K*sizeof(float));
    hostB = (float*)malloc(K*N*sizeof(float));
    hostC = (float*)malloc(M*N*sizeof(float));
    
    // Allocate device memory
    cudaMalloc((void**)&devA, M*K*sizeof(float));
    cudaMalloc((void**)&devB, K*N*sizeof(float));
    cudaMalloc((void**)&devC, M*N*sizeof(float));
    
    cudaMalloc((void**)&sqSumVecA, M*sizeof(float));
    cudaMalloc((void**)&sqSumVecB, N*sizeof(float));
    
    
    
    ////////////////////////////////////////////////
    //             DATA INITIALIZATION            //
    ////////////////////////////////////////////////
    
    // Initialize data on host
    srand(0);
    for(int m = 0; m < M; m++){
        for(int k = 0; k < K; k++){
            hostA[m*K+k] = (float)rand()/(float)(RAND_MAX/1000);
        }
    }
    for(int k = 0; k < K; k++){
        for(int n = 0; n < N; n++){
            hostA[k*N*+n] = (float)rand()/(float)(RAND_MAX/1000);
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
    
    
    
    // Launch kernel
    combinedSGEMM<<<gridSize,blockSize>>>(devA,devB,devC,0,0,M,N,K);
    
    // Transfer result from device to host
    cudaMemcpy(hostC,devC,M*N*sizeof(float),cudaMemcpyDeviceToHost);
    
    
    
    ////////////////////////////////////////////////
    //            RESULT VERIFICATION             //
    ////////////////////////////////////////////////
    
    // Verify results
    
    
    
    ////////////////////////////////////////////////
    //             FREE MEMORY & EXIT             //
    ////////////////////////////////////////////////
    
    // Free host memory
    free(hostA);
    free(hostB);
    free(hostC);
    
    // Free device memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    
    return 0;
}