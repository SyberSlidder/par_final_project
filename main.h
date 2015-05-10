#ifndef MAINH

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <cuda.h>
#include <cublas.h>
#include "cycleTimer.h"

#include <inplace/transpose.h>

using namespace std;

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK 8*8
#define BLOCKS 64

#define RANDOM_SEED false
#define RAND_SEED 100

#define MAXWELL_MICROTILE_SIZE 8
#define B_TRANSPOSE true

#define GEMM_ONLY true

// Definitions of each Kernel

__global__ void KernelSumFromC(
       float * _C,
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       float * _W, // Weight Vector
       float * _V, // Result Vector
       int M, // Number rows of A
       int N // Number of columns of B
);

// warpReduce Kernel

void callSquareSumVector(float *srcMatrix,
				    float *sqSumVector,
				    int M,
				    int K,
				    int maxGridSize
);

__global__ void calcSquareSumVector(float *srcMatrix,
                                    float *sqSumVector,
                                    int    M,
                                    int    K
);


// Version 1 of the combined Kernel

__global__ void combinedSGEMM_v1(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
);

__global__ void combinedSGEMM_v2(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
);

__global__ void combinedSGEMM_v3(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
);

__global__ void combinedSGEMM_v4(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       float * _W, // Weight vector
       float * result, // result vector
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
);

__global__ void MaxwellCombinedSGEMM_v1(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
);

__global__ void MaxwellCombinedSGEMM_v2(
       float * _A, // Global pointer to matrix A 
       float * _B, // Global pointer to matrix B
       float * _C, // Global pointer to write out result of A*B
       float * sqSumVecA, // M x 1 matrix derived from A
       float * sqSumVecB, // N x 1 matrix derived from B
       float * _W, // Weight vector
	 int M, // Number rows of A
       int N, // Number of columns of B
       int K  // Columns A, rows B
);


__global__ void GPU2 (float *a, float *b, float *c, int n);

#define MAINH
#endif

