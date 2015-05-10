#include "main.h"

int main(int argc, char * argv[]) {
    
    // Misc
    int kernelVersion;
    double cpuStartTime;
    double cpuEndTime;
    double runtime;
    double flop;
    
    
    // Matrix dimensions
    int    M, N, K;
    
    // Host pointers
    float *hostA;
    float *hostB;
    float *hostC;
    float *hostSqSumVecA;
    float *hostSqSumVecB;
    float *hostW;
    float *hostRes;
    
    // Device pointers
    float *devA;
    float *devB;
    float *devC;
    float *devSqSumVecA;
    float *devSqSumVecB;
    float * devW;
    float * devRes;
    
    // Kernel parameters
    dim3 gridSize;
    dim3 blockSize;
    
    cudaDeviceProp  deviceProp;
    
    ////////////////////////////////////////////////
    //           MEMORY INITIALIZATION            //
    ////////////////////////////////////////////////

    // Check for proper arguments
    if(argc != 5){
        fprintf(stderr, "USAGE: ./kernelSummation M N K KernelVersion\n");
        exit(EXIT_SUCCESS);
    }
    
    // Get matrix dimensions
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    kernelVersion = atoi(argv[4]);
    
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

    // Check for valid arguments
    if(M <= 0 || N <= 0 || K <= 0){
        fprintf(stderr, "ERROR: One of the dimensions is <=0\n");
        exit(EXIT_SUCCESS);
    }
    
    // Allocate host memory
    hostA = (float*)malloc(M*K*sizeof(float));
    hostB = (float*)malloc(K*N*sizeof(float));
    hostC = (float*)malloc(M*N*sizeof(float));

    hostSqSumVecA = (float*)malloc(M*sizeof(float));
    hostSqSumVecB = (float*)malloc(N*sizeof(float));
    hostW = (float*)malloc(N*sizeof(float));
    hostRes = (float*)malloc(M*sizeof(float));
    
    // Allocate device memory
    cudaMalloc((void**)&devA, M*K*sizeof(float));
    cudaMalloc((void**)&devB, K*N*sizeof(float));
    cudaMalloc((void**)&devC, M*N*sizeof(float));
    
    printf("Dev A: %x \n",devA);
    printf("Dev B: %x \n",devB);
    printf("Dev C: %x \n",devC);

    
    cudaMalloc((void**)&devSqSumVecA, M*sizeof(float));
    cudaMemset(devSqSumVecA, 0, M*sizeof(float));
    cudaMalloc((void**)&devSqSumVecB, N*sizeof(float));
    cudaMemset(devSqSumVecB, 0, N*sizeof(float));
    
    cudaMalloc((void**)&devW, N*sizeof(float));
    cudaMemset(devW, 1.1, N*sizeof(float));
    cudaMalloc((void**)&devRes, N*sizeof(float));
    
    ////////////////////////////////////////////////
    //             DATA INITIALIZATION            //
    ////////////////////////////////////////////////
    
    // Initialize data on host
    if (RANDOM_SEED) {
      srand(time(0));
    } else {
      srand(RAND_SEED);
    }
    
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
    
    for(int n=0 ; n < N; n++)
	    hostW[n] =  (float)rand()/(float)(RAND_MAX/10);

    

    
    ////////////////////////////////////////////////
    //              KERNEL SUMMATION              //
    ////////////////////////////////////////////////
    
    // Transfer host data to device
    cudaMemcpy(devA,hostA,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devB,hostB,K*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devW,hostW,N*sizeof(float),cudaMemcpyHostToDevice);
    
    // Calculate grid and block size
    gridSize.x = (N + 1)/THREADS_PER_BLOCK_X;
    gridSize.y = (M + 1)/THREADS_PER_BLOCK_Y;
    gridSize.z = 1;
    blockSize.x = THREADS_PER_BLOCK_X;
    blockSize.y = THREADS_PER_BLOCK_Y;
    blockSize.z = 1;
    dim3 gridSize1;
    dim3 gridSize2;
    // Kernel Selection
    switch (kernelVersion) {
      case 0:
	  cpuStartTime = CycleTimer::currentSeconds();
	  cublasInit();
	  cublasSgemm( 'n', 'n', M, N, K, 1.0, devA, M, devB, K, 0.0, devC, M);
	  cpuEndTime = CycleTimer::currentSeconds();
	  runtime = 1000.f * (cpuEndTime-cpuStartTime);
	  printf("Version %d Runtime: %.5f ms\n",kernelVersion,runtime);
	  cublasShutdown();
	  break;
      case 1:
	  gridSize1.x = 1;
	  gridSize1.y = M;
	  gridSize1.z = 1;
	  
	  gridSize2.x = 8;
	  gridSize2.y = 8;
	  gridSize2.z = 1;
	  
	  cpuStartTime = CycleTimer::currentSeconds();
	  calcSquareSumVector<<<gridSize1,1024>>>(devA,devSqSumVecA,M,K);
	  calcSquareSumVector<<<gridSize1,1024>>>(devB,devSqSumVecB,N,K);
	  combinedSGEMM_v1<<<gridSize,blockSize>>>(devA,devB,devC,devSqSumVecA,devSqSumVecB,M,N,K);
	  cpuEndTime = CycleTimer::currentSeconds(); 
	  runtime = 1000.f * (cpuEndTime-cpuStartTime);
	  printf("Version %d Runtime: %.5f ms\n",kernelVersion,runtime);
	  
	  if (cudaGetLastError() != CUDA_SUCCESS) {
	    printf("Error in the kernel evaluation.\n");
	    exit(-1);
	  }
	  break;
      case 2:
	  gridSize1.x = N/64;
	  gridSize1.y = M/64;
	  gridSize1.z = 1;
	  
	  gridSize2.x = 8;
	  gridSize2.y = 8;
	  gridSize2.z = 1;
	  
	  
	  cpuStartTime = CycleTimer::currentSeconds();
	  //calcSquareSumVector<<<gridSize1,1024>>>(devA,devSqSumVecA,M,K);
	  //calcSquareSumVector<<<gridSize1,1024>>>(devB,devSqSumVecB,N,K);
	  combinedSGEMM_v2<<<gridSize1,gridSize2>>>(devA,devB,devC,devSqSumVecA,devSqSumVecB,M,N,K);
	  cpuEndTime = CycleTimer::currentSeconds(); 
	  runtime = 1000.f * (cpuEndTime-cpuStartTime);
	  printf("Version %d Runtime: %.5f ms\n",kernelVersion,runtime);
	  if (cudaGetLastError() != CUDA_SUCCESS) {
	    printf("Error in the kernel evaluation.\n");
	    exit(-1);
	  }
	break;
      case 3:
	  gridSize1.x = N/64;
	  gridSize1.y = M/64;
	  gridSize1.z = 1;
	  
	  gridSize2.x = 8;
	  gridSize2.y = 8;
	  gridSize2.z = 1;
	  
	  
	  cpuStartTime = CycleTimer::currentSeconds();
	  //calcSquareSumVector<<<gridSize1,1024>>>(devA,devSqSumVecA,M,K);
	  //calcSquareSumVector<<<gridSize1,1024>>>(devB,devSqSumVecB,N,K);
	  combinedSGEMM_v3<<<gridSize1,gridSize2>>>(devA,devB,devC,devSqSumVecA,devSqSumVecB,M,N,K);
	  cpuEndTime = CycleTimer::currentSeconds(); 
	  runtime = 1000.f * (cpuEndTime-cpuStartTime);
	  printf("Version %d Runtime: %.5f ms\n",kernelVersion,runtime);
	  if (cudaGetLastError() != CUDA_SUCCESS) {
	    printf("Error in the kernel evaluation.\n");
	    exit(-1);
	  }
	break;
      case 4:
	  gridSize1.x = N/64;
	  gridSize1.y = M/64;
	  gridSize1.z = 1;
	  
	  gridSize2.x = 8;
	  gridSize2.y = 8;
	  gridSize2.z = 1;

          inplace::transpose(true,devB,K,N);
          callSquareSumVector(devA,devSqSumVecA,M,K,deviceProp.maxGridSize[1]);
          callSquareSumVector(devB,devSqSumVecB,N,K,deviceProp.maxGridSize[1]);

	  combinedSGEMM_v4<<<gridSize1,gridSize2>>>(devA,devB,devC,devSqSumVecA,devSqSumVecB,devW,devRes,M,N,K);

	  if (cudaGetLastError() != CUDA_SUCCESS) {
	    printf("Error in the kernel evaluation. %s \n");
	    exit(-1);
	  }
	  
	break;
      case 5:
	  gridSize1.x = N/64;
	  gridSize1.y = M/64;
	  gridSize1.z = 1;
	  
	  gridSize2.x = 8;
	  gridSize2.y = 8;
	  gridSize2.z = 1;
	  
	  inplace::transpose(true,devB,K,N);
	  callSquareSumVector(devA,devSqSumVecA,M,K,deviceProp.maxGridSize[1]);
	  callSquareSumVector(devB,devSqSumVecB,N,K,deviceProp.maxGridSize[1]);
	  MaxwellCombinedSGEMM_v2<<<gridSize1,gridSize2>>>(devA,devB,devC,devSqSumVecA,devSqSumVecB,devW,M,N,K);

	  if (cudaGetLastError() != CUDA_SUCCESS) {
	    printf("Error in the kernel evaluation.\n");
	    exit(-1);
	  }
	  
	break;
      default:
	cout << "Error - Must choose a proper Kernel." << endl;
	exit(-1);
    }    
    
    // Transfer result from device to host
    cudaMemcpy(hostSqSumVecA,devSqSumVecA,M*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSqSumVecB,devSqSumVecB,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostC,devC,M*N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostRes,devRes,M*N*sizeof(float),cudaMemcpyDeviceToHost);
    
    for (int index = 0; index != 16; index++) {
      printf("C %d: %f\n",index,hostC[index]);
    }
    for (int index = 0; index != 16; index++) {
      printf("res %d: %f\n",index,hostRes[index]);
    }
    
    ////////////////////////////////////////////////
    //            RESULT VERIFICATION             //
    ////////////////////////////////////////////////
    /*
    // Verify square-sums
    for(int m = 0; m < M; m++){
        float sqSum = 0.0;
        for(int k = 0; k < K; k++){
            sqSum += hostA[m*K+k]*hostA[m*K+k];
        }
        if(int(sqSum) != int(hostSqSumVecA[m])) {
            //fprintf(stderr, " Bad square sum: [m = %d] [refSqSum = %f] [devSqSum = %f]\n", m, sqSum, hostSqSumVecA[m]);
	} else {
            //fprintf(stderr, "Good square sum: [m = %d] [refSqSum = %f] [devSqSum = %f]\n", m, sqSum, hostSqSumVecA[m]);
	}
    }    

    for(int n = 0; n < N; n++){
        float sqSum = 0.0;
        for(int k = 0; k < K; k++){
            sqSum += hostB[n*K+k]*hostB[n*K+k];
        }
        if(int(sqSum) != int(hostSqSumVecB[n])) {
           // fprintf(stderr, " Bad square sum: [m = %d] [refSqSum = %f] [devSqSum = %f]\n", n, sqSum, hostSqSumVecB[n]);
	} else {
            //fprintf(stderr, "Good square sum: [m = %d] [refSqSum = %f] [devSqSum = %f]\n", n, sqSum, hostSqSumVecB[n]);
	}
    }    
    */
    
    // Verify Matrix multiplication
    
    
    
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
