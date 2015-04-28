#include<stdio.h>
#include "cublas.h"
#include <cuda_runtime.h>
#include "cycleTimer.h"
#define warpSize 32

texture<float4,1> texture_A;
//texture<float,1> texture_B;

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__global__
void tileMul(float *_A, float *_B, float *_C,int m, int n, int k, int bm, int bn, int bk, int KtileWidth){
        //assume A is row-major and B is column-major and C has initial value all zero
	float tileC[8][8];
	float4 a0,a1,b0,b1;
	int loopi,loopj;
	for(loopi=0;loopi<8;loopi++)
		for(loopj=0;loopj<8;loopj++)
			tileC[loopi][loopj]=0;
	int tileNum;
    	//int idMod32 = (id%warpSize);
    	//int idDivide32 = (id/warpSize);
	//int subtileN = threadPerBlock/warpSize;
	int totTileNum = min((KtileWidth+bk-1)/bk,(k+bk-1)/bk);
	float *A = _A + blockIdx.y * bm*k;
	float *B = _B + blockIdx.x * bn*k;
	float *C = _C + blockIdx.y * bm*n + blockIdx.x * bn;
	
//	use share memory
	__shared__ float4 tileA0[128][2]; //bk*bn
	__shared__ float4 tileB0[128][2]; //bk*bn
	__shared__ float4 tileA1[128][2]; //bk*bn
	__shared__ float4 tileB1[128][2]; //bk*bn
	tileNum=0;
	tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1] = reinterpret_cast<float4*> (B)[tileNum*2+(threadIdx.x&1) + (threadIdx.y*8+threadIdx.x/2)*(k/4)];
	//printf("tileB0[%d][%d]=%f %f %f %f\n",(threadIdx.y<<3) + (threadIdx.x>>1),threadIdx.x & 1,tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].x, tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].y, tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].z, tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].w);
        tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1] = reinterpret_cast<float4*> (A)[tileNum*2+(threadIdx.x&1) + (threadIdx.y*8+threadIdx.x/2)*(k/4)];
	//printf("tileA0[%d][%d]=%f %f %f %f\n",(threadIdx.y<<3) + (threadIdx.x>>1), threadIdx.x & 1, tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].x ,  tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].y,  tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].z,  tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].w);
	__syncthreads();

	for(tileNum=0 ; tileNum < totTileNum; tileNum+=2){
		/* tileC(blockx,blocky) += tileA[i] * tileB[i]*/
		if((tileNum+1)<totTileNum){
            	tileB1[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1] = reinterpret_cast<float4*> (B)[(tileNum+1)*2+(threadIdx.x&1) + (threadIdx.y*8+threadIdx.x/2)*(k/4)];
		//printf("tileB1[%d][%d]=%f %f %f %f\n",(threadIdx.y<<3) + (threadIdx.x>>1),threadIdx.x & 1,tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].x, tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].y, tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].z, tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].w);
            	tileA1[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1] = reinterpret_cast<float4*> (A)[(tileNum+1)*2+(threadIdx.x&1) + (threadIdx.y*8+threadIdx.x/2)*(k/4)];
		//printf("tileA1[%d][%d]=%f %f %f %f\n",(threadIdx.y<<3) + (threadIdx.x>>1), threadIdx.x & 1, tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].x ,  tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].y,  tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].z,  tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1].w);
		}
		/*compute A0 B0*/
		for(loopi=0; loopi<8 ; loopi++){
			a0 = tileA0[(threadIdx.y<<3)+loopi][0];
			a1 = tileA0[(threadIdx.y<<3)+loopi][1];
			for(loopj=0; loopj<8 ; loopj++){
				b0 = tileB0[(threadIdx.x<<3)+loopj][0];
				b1 = tileB0[(threadIdx.x<<3)+loopj][1];
				tileC[loopi][loopj] += a0.x*b0.x + a0.y*b0.y + a0.z*b0.z + a0.w*b0.w +
							a1.x*b1.x+ a1.y*b1.y + a1.z*b1.z + a1.w*b1.w;
				if((tileNum+1) == totTileNum){
                                /*writeC back*/
                                C[(threadIdx.y*8+loopj)*n + threadIdx.x*8+loopi]= tileC[loopi][loopj];
				//printf("thread(%d,%d) loopi=%d loopj=%d C[%d][%d]=%f\n",threadIdx.x, threadIdx.y,loopi,loopj, threadIdx.x*8+loopi,threadIdx.y*8+loopj,tileC[loopi][loopj]);
                                }
			}
		}
		__syncthreads();
		/*preload A0,B0*/
		if((tileNum+2) < totTileNum){
		        tileB0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1] = reinterpret_cast<float4*> (B)[(((threadIdx.y<<3) + (threadIdx.x>>1))*k+ (tileNum+2)*bk +(threadIdx.x&1))>>2];
		        tileA0[(threadIdx.y<<3) + (threadIdx.x>>1)][threadIdx.x & 1] = reinterpret_cast<float4*> (A)[(((threadIdx.y<<3) + (threadIdx.x>>1))*k + (tileNum+2)*bk+(threadIdx.x&1))>>2];
        	}	
		/*compute A1 B1*/
                for(loopi=0; loopi<8 ; loopi++){
                        a0 = tileA1[(threadIdx.y<<3)+loopi][0];
                        a1 = tileA1[(threadIdx.y<<3)+loopi][1];
                        for(loopj=0; loopj<8 ; loopj++){
                                b0 = tileB1[(threadIdx.x<<3)+loopj][0];
                                b1 = tileB1[(threadIdx.x<<3)+loopj][1];
                                tileC[loopi][loopj] += a0.x*b0.x + a0.y*b0.y + a0.z*b0.z + a0.w*b0.w + 
                                                        a1.x*b1.x+ a1.y*b1.y + a1.z*b1.z + a1.w*b1.w;
				if((tileNum+2) == totTileNum){
                		/*writeC back*/
                		C[(threadIdx.y*8+loopj)*n + threadIdx.x*8+loopi]= tileC[loopi][loopj];
                		}
                        }
                }
		__syncthreads();
	}
}
int main(int argc, char* argv[]){
	int m,n,k;
	int i,j;
//	m=4096; n=4096; k=4096;
	int microTile = 8;
	sscanf( argv[ 1 ], "%d", &m );
	sscanf( argv[ 2 ], "%d", &n );
	sscanf( argv[ 3 ], "%d", &k );
	//sscanf( argv[ 4 ], "%d", &microTile );
	float *A = (float*)malloc(sizeof(float)*m*k);
	float *B = (float*)malloc(sizeof(float)*k*n);
	float *C = (float*)malloc(sizeof(float)*m*n);
//for cublas
/*	for(j=0; j<k; j++){
		for(i=0; i<m; i++){
			A[j*m+i]=(10*i+j)*0.01; //store A in column major, size m*k
		}
	}
*/
//for tile_mul

	for(i=0; i<m; i++){
		for(j=0; j<k; j++){
			A[i*k+j]=i+j*0.01; //store A in row major, size m*k
		}
	}

	for(j=0; j<n; j++){
		for(i=0; i<k; i++){
			B[j*k+i]=i+j*0.01;//store B in column major, size k*n
                }
        }

        for(i=0; i<m; i++){
                for(j=0; j<n; j++){
                        C[i*n+j]=0; //store A in row major, size m*k
                }
        }
/*
	if(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) != cudaSuccess)
		printf("SharedMemBankSizeEightByte failed.\n");
	cudaSharedMemConfig pConfig;
	cudaDeviceGetSharedMemConfig(&pConfig);
	printf("cudaSharedMemBankSize=%d\n",pConfig);//cudaSharedMemBankSizeDefault = 0
						     //cudaSharedMemBankSizeFourByte = 1
						     //cudaSharedMemBankSizeEightByte = 2
*/
	float* dev_A,*dev_B,*dev_C;
	cudaMalloc((void**)&dev_A,m*k*sizeof(float));	
	cudaMalloc((void**)&dev_B,k*n*sizeof(float));	
	/*
	size_t pitch;//=warpsize
	cudaMallocPitch((void**)&dev_A,&pitch,k*sizeof(float),m);
	cudaMallocPitch((void**)&dev_B,&pitch,k*sizeof(float),n);
	*/
	cudaMalloc((void**)&dev_C,m*n*sizeof(float));	
	cudaMemcpy(dev_A,A,m*k*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B,B,k*n*sizeof(float),cudaMemcpyHostToDevice);
	/*
	cudaMemcpy2D(dev_A,pitch,A,sizeof(float)*k,sizeof(float)*k,m,cudaMemcpyHostToDevice);
	cudaMemcpy2D(dev_B,pitch,B,sizeof(float)*k,sizeof(float)*k,n,cudaMemcpyHostToDevice);
	*/
	cublasInit();
	float alpha = 1.0;
	float beta = 0.0;
	cudaBindTexture(NULL,texture_A,dev_A,m*k*sizeof(float));
	//cudaBindTexture(NULL,texture_B,dev_B,k*n*sizeof(float));
	int bm=128;
	int bn=128;//32*4
	int bk=8;
	//int texCache = 48*1024; //48K
	//int microTile = texCache/sizeof(float)/bn/bk;//12
	int threadPerBlock = 16;
	int N = (n+bn-1)/bn;//gridX
	int M = (m+bm-1)/bm;//gridY
	int KtileWidth = bk*microTile;//12*32/2
	int K = (k+KtileWidth-1)/KtileWidth;//gridZ
	dim3 grid_dim(N,M,K);
	dim3 block_dim(threadPerBlock,threadPerBlock,1);
	int blockNum = M*N*K;
	printf("block number %d*%d*%d= %d\n",N,M,K,blockNum);
	double cpuStartTime = CycleTimer::currentSeconds();
//	tileMul<<<grid_dim,block_dim>>>(dev_A, dev_B, dev_C, m, n, k,bm,bn,bk,KtileWidth);
	cublasSgemm( 'n', 'n', m, n, k, alpha, dev_A, m,dev_B, k, beta,dev_C, m);
	cudaThreadSynchronize();
	double cpuEndTime = CycleTimer::currentSeconds();
	double runtime = 1000.f * (cpuEndTime-cpuStartTime);
	double flop = (double)2*m*n*k;
        printf("Dgemm runtime: %.3f ms, GFLOPS=%.6f\n", runtime,flop/runtime/1000000 );
	cudaMemcpy(C,dev_C,m*n*sizeof(float),cudaMemcpyDeviceToHost);
//	cudaUnbindTexture(texture_A);
//	cudaUnbindTexture(texture_B);
	cublasShutdown();
	printf("cuda blas:\n");
	printf("m=%d,n=%d,k=%d ",m,n,k);
	for(i=0; i<m; i++){
                printf("\n");
                for(j=0; j<n; j++)
                        printf("%f      ",C[i*n+j]);
        }
	return 0;
}

