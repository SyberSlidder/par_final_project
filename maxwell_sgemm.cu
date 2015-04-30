#include "main.h"


__global__ void MaxwellCombinedSGEMM_v1(
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

    // Initialization
    float cVal[8][8] = { 
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		  {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}
		  };
	
	float track1[8];
	float track2[8];
		  
	// Identification
	int linearThreadID = threadIdx.x + (blockDim.x * threadIdx.y);
	
	// Loop through the K dimension of Matrices A and B
	// We are operating out of a 64 x 16 chunk of A/B
	for (int i = 0; i < (K/16); i++) {
	    
		// Load from A into A1
		
		// Load from B into B2
		
		// Update pointers
		
		__syncthreads();
		
		// Compute from A1 and B1
		
		// Grab 1 8-element track from A and 1 8-elemnt track from B
		// 8 tracks total from A1, 8 tracks total from B1
		for (int trackNum = 0; trackNum < 8; trackNum++) {
			// Load Track from A1 into track1
			
			// Load Track from A2 into track2
			
			// Compute the outer product of the the tracks
			// 64 FMA from each track 1/2 pair
			// 64 x 8 = 512 FMA operations for loading 64 elements from SM
			#pragma unroll
			for (int trackRow = 0; trackRow < 8; trackRow++) {
				cVal[trackRow][0] += track1[trackRow]*track2[0];
				cVal[trackRow][1] += track1[trackRow]*track2[1];
				cVal[trackRow][2] += track1[trackRow]*track2[2];
				cVal[trackRow][3] += track1[trackRow]*track2[3];
				cVal[trackRow][4] += track1[trackRow]*track2[4];
				cVal[trackRow][5] += track1[trackRow]*track2[5];
				cVal[trackRow][6] += track1[trackRow]*track2[6];
				cVal[trackRow][7] += track1[trackRow]*track2[7];
			}
			
		}
		
		__syncthreads();
		
		// Compute from A2 and B2
		
		
		
	}
	
	
}