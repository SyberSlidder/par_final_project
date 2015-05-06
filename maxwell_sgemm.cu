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

    // Complex Grid
    //__shared__ float smA[16][32];
    //__shared__ float smB[16][32];
  
    __shared__ float smA[16][32];
    __shared__ float smB[16][32];
    
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
	int warpID = linearThreadID / 32;
	
	
	
	// Where to read from in A
	int loadRowA = ((blockDim.y*8) * blockIdx.y) + linearThreadID;
	float * aReadPtr = _A + (loadRowA * K);
	// Where to read from in B
	int loadRowB = (blockDim.y*8) * blockIdx.y;
	float * bReadPtr = _B + (loadRowB * K);
	
	// Loop through the K dimension of Matrices A and B
	// We are operating out of a 64 x 16 chunk of A/B
	for (int i = 0; i < (K/8); i++) {
	    
		// Load from A into SM
		//float4 aHolder1 = *((float4 *)(aReadPtr));
		//float4 aHolder2 = *((float4 *)(aReadPtr + 4));
	  
		//smA[][]
		
		// Load from B into SM
		
		// Update pointers
				
		// Grab 1 8-element track from A and 1 8-elemnt track from B
		// 8 tracks total from A, 8 tracks total from B
		int startIndex = ((linearThreadID & 0x1F) / 16) * 8;
		
		int trackStartID = 0;//(linearThreadID % 8);
		int secondHalf = ((linearThreadID & 0x0F) / 8)*8;
		int trackSelect  = secondHalf + (16 * warpID);
		for (int trackNum = 0; trackNum < 8; trackNum++) {
		
			// Load Track from A into track1
			// Reads down a track
			int columnSelect = trackSelect + trackStartID;
			for (int trackElement = 0; trackElement < 8; trackElement++) {
			    track1[trackElement] = smA[startIndex + trackElement][columnSelect];
			}
			
			for (int trackElement = 0; trackElement < 8; trackElement++) {
			    track2[trackElement] = smB[startIndex + trackElement][columnSelect];
			}
			// Load Track from B into track2
			
			// Compute the outer product of the the tracks
			// 64 FMA from each track 1,2 pair
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
			
			// Move to the next track
			trackStartID = (trackStartID + 1) % 8;
			
		}
		
	}

	// Write back C
	int C_row = loadRowA;
	int C_column = loadRowB;
	
	for (int i = 0; i < 8; i++) {
	  for (int j = 0; j < 8; j++) {
	    _C[(C_row+i)*N + C_column+j] = cVal[i][j];
	  }
	}
	
}

union dblFloat4 {
  float4 wideFloats[2];
  float  elem[8];
};

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
)
{  

    // Complex Grid
    //__shared__ float smA[16][32];
    //__shared__ float smB[16][32];
  
    __shared__ float smA[16][32];
    __shared__ float smB[16][32];
    
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
	int warpID = linearThreadID / 32;
	
	// Where to read from in A
	int loadRowA = ((blockDim.y*8) * blockIdx.y) + linearThreadID;
	float * aReadPtr = _A + (loadRowA * K);
	// Where to read from in B
	int loadRowB = (blockDim.y*8) + (linearThreadID / 8);
	int bColumnOffset = (blockDim.x*blockIdx.x*8) + (8*threadIdx.x);
	float * bReadPtr = _B + (loadRowB * N) + bColumnOffset;
	
	
	// Loop through the K dimension of Matrices A and B
	// We are operating out of a 64 x 16 chunk of A/B
	int columnStart = (linearThreadID >> 3) & 0x03;
	int rowStart = warpID * 8;
	
	dblFloat4 aHolder;
	dblFloat4 bHolder;
	
	for (int i = 0; i < (K/8); i++) {
	    
		// Load from A into register
		aHolder.wideFloats[0] = *((float4 *)(aReadPtr));
		aHolder.wideFloats[1] = *((float4 *)(aReadPtr + 4));
		// Fix the bank and store down a row
		int storeAOffset = linearThreadID % 8;
		for (int trackElement = 0; trackElement < 8; trackElement++) {
		      smA[rowStart + trackElement][linearThreadID % 32] = aHolder.elem[trackElement];
		      //storeAOffset = (storeAOffset + 1 ) % 8;
		}
		
		// Load from B into register
		bHolder.wideFloats[0] = *((float4 *)(bReadPtr));
		bHolder.wideFloats[1] = *((float4 *)(bReadPtr + 4));
		// Fix the bank and store down a row when storing B
		for (int trackElement = 0; trackElement < 8 ; trackElement++) {
			smB[rowStart + trackElement][linearThreadID % 32] = bHolder.elem[trackElement]; 
		}
		
		// Update pointers
		aReadPtr += 8;
		bReadPtr += 8*N;
				
		// Wait for everyone to finish their loads
		__syncthreads();
				
		// Grab 1 8-element track from A and 1 8-elemnt track from B
		// 8 tracks total from A, 8 tracks total from B

		//int offSet = linearThreadID % 8;
		
		for (int trackNum = 0; trackNum < 8; trackNum++) {
		
			// Load Track from A into track1
			// Reads down a track
			
			// Fix row and change banks when reading A from shared memory
			int offSet = linearThreadID % 8;
			for (int trackElement = 0; trackElement < 8; trackElement++) {
			    //track1[trackElement] = smA[rowStart + trackElement][offSet + columnStart];
			    track1[offSet] = smA[rowStart + trackNum][columnStart + offSet];
			    offSet = (offSet + 1) % 8;
			}
			
			for (int trackElement = 0; trackElement < 8; trackElement++) {
			    track2[trackElement] = smB[rowStart + trackElement][offSet + columnStart];
			}
			// Load Track from B into track2
			
			// Compute the outer product of the the tracks
			// 64 FMA from each track 1,2 pair
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
			
			// Move to the next track
			//offSet = (offSet + 1) % 8;
			
		}
		
	}

	// Write back C
	int C_row = loadRowA;
	int C_column = loadRowB;
	
	if (GEMM_ONLY) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				_C[(C_row+i)*N + C_column+j] = cVal[i][j];
			}
		}
	} else {
		// Load 8 elements from the weight vector
		float weights[8];
		float * weightPtr = _W + C_column;
		float4 w1,w2;
		w1 = (*(float4 *)(weightPtr));
		w2 = (*(float4 *)(weightPtr+4)); 		
		
		
		
		
	}
	
}