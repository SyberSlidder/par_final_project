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
void tileMul(float *_A, float *_B, float *_C,int m, int n, int k, int bm, int bn, int bk, int KtileWidth,int threadPerBlock){
        //assume A is row-major and B is column-major and C has initial value all zero
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int blockz = blockIdx.z;
	float value[2][2];
	float _value;
	int l,loopi,loopj;
	int tileNum,currentRow;
	int Comb = 4; //float4
	int half = threadPerBlock/2;
	int idy_half = idy-half;
	int idModbm = idy%(bm/2);
	int idDividebm = idy/(bm/2);
    	//int idMod32 = (id%warpSize);
    	//int idDivide32 = (id/warpSize);
	//int subtileN = threadPerBlock/warpSize;
	int totTileNum = min((KtileWidth+bk-1)/bk,(k+bk-1)/bk);
	float *A = _A + blocky * bm*k;
	float *B = _B + blockx * bn*k;
	//float *C = _C + blocky * bm*n + blockx * bn;
	
//	use share memory
	__shared__ float4 tileA[8][16]; //bk*bn
	__shared__ float4 tileB[8][16]; //bk*bn
	__shared__ float tempC[8][8]; //bm*bn,FIXME:dynamic allocate tempC[bn][bm]

	for(tileNum=0 ; tileNum < totTileNum; tileNum++){
		/* tileC(blockx,blocky) += tileA[i] * tileB[i]*/
		int index = blockz*KtileWidth + tileNum*bk + idx*Comb;
		if(index <= k){
            		if(idy<half) tileB[idy][idx] = reinterpret_cast<float4*> (B)[(idy*k + index)/Comb];
					//printf("id=(%d,%d),x=%f,y=%f,z=%f,w=%f,B[%d][%d],offset=%d\n",idx,idy,tileB[idy][idx].x,tileB[idy][idx].y,tileB[idy][idx].z,tileB[idy][idx].w,idy,idx,(idy*k + index)/Comb);}
            		else tileA[idy_half][idx] = reinterpret_cast<float4*> (A)[((idy_half)*k + index)/Comb];
					//printf("id=(%d,%d),x=%f,y=%f,z=%f,w=%f,A[%d][%d],offset=%d\n",idx,idy_half,tileA[idy_half][idx].x,tileA[idy_half][idx].y,tileA[idy_half][idx].z,tileA[idy_half][idx].w,idy_half,idx,(idy_half*k + index)/Comb);}
        	}
		//for(currentRow=0; currentRow < bm; currentRow++){
			//float elementA = A[k*currentRow + blockz*KtileWidth + tileNum*bk + idMod32];//put elementA in reg	
			//for(loop=0; (loop*subtileN)<bn; loop++){
				if(index <= k){
					float4 a0 = tileA[idDividebm*2][idx];
					float4 a1 = tileA[idDividebm*2+1][idx];
					float4 b0 = tileB[idModbm*2][idx];
					float4 b1 = tileB[idModbm*2+1][idx];
					value[0][0] = a0.x*b0.x + a0.y*b0.y + a0.z*b0.z + a0.w*b0.w;
					value[0][1] = a0.x*b1.x + a0.y*b1.y + a0.z*b1.z + a0.w*b1.w;
					value[1][0] = a1.x*b0.x + a1.y*b0.y + a1.z*b0.z + a1.w*b0.w;
					value[1][1] = a1.x*b1.x + a1.y*b1.y + a1.z*b1.z + a1.w*b1.w;
					//printf("index=%d,value=%f,offset=%d\n",index,value,offset);
				}
				else{	
					 value[0][0] = 0;
					 value[0][1] = 0;
					 value[1][0] = 0;
					 value[1][1] = 0;
				}
			/*if(currentRow == 0) printf("Index=%d.block tileNum=%d. Thread %d->%d.Value %f * %f= %f\n",index,tileNum,id,idMod32,elementA, tex1Dfetch(texture_B, (blocky*bn + loop*subtileN + idDivide32)*k + idMod32),value);
			  if(currentRow == 0){
				float valueb = tex1Dfetch(texture_B, (blocky*bn + loop*subtileN + idDivide32)*k + idMod32);
				printf("valueb[%d][%d]=%f\n",blocky * bn+idDivide32+ loop*subtileN, idMod32 ,valueb);
			  }
			*/
        	    //Shuffle Warp Reduce 
		for(loopi=0;loopi<2;loopi++)
			for(loopj=0;loopj<2;loopj++){
				_value=value[loopi][loopj];	
        	    for (l=warpSize/2; l>=2; l/=2)
        	        _value += __shfl_down(_value, l);
	      	    //if(currentRow == 0) printf("Thread %d final value = %f\n", id, value);
        	    if(idx == 0) {
					//tempC[currentRow][idDivide32+loop*32] +=value;		
					tempC[idDividebm*2+loopi][idModbm*2+loopj] += _value;		
					//printf("C_temp[%d][%d] final value = %f\n",currentRow,idDivide32+loop*subtileN,value);		    	
					if(tileNum == totTileNum-1){
						//_C[ (currentRow+blockx*bm)*n + blocky * bn + idDivide32+loop*32] = tempC[currentRow*bn + idDivide32+loop*32];
						atomicAdd(&_C[ (idDividebm*2+loopi+blocky*bm)*n + blockx*bn + idModbm*2+loopj] , tempC[idDividebm*2+loopi][idModbm*2+loopj]);
						//if(currentRow == 0) printf("C[%d][%d] final value = %f\n",currentRow+blockx*bm,idDivide32+loop*32+blocky*bn,tempC[currentRow*bn + idDivide32+loop*32]);
					}
    			}
			}
			//}
		//}
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
	sscanf( argv[ 4 ], "%d", &microTile );
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
	//float alpha = 1.0;
	//float beta = 0.0;
	cudaBindTexture(NULL,texture_A,dev_A,m*k*sizeof(float));
	//cudaBindTexture(NULL,texture_B,dev_B,k*n*sizeof(float));
	int bm=8;
	int bn=8;//32*4
	int bk=64;
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
	tileMul<<<grid_dim,block_dim>>>(dev_A, dev_B, dev_C, m, n, k,bm,bn,bk,KtileWidth,threadPerBlock);
//	cublasDgemm( 'n', 'n', m, n, k, alpha, dev_A, m, dev_B, k, beta, dev_C, m);
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

