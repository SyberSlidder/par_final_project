#pragma once
#include "index.h"
#include "smem.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void smem_row_shuffle(int m, int n, T* d, F s) {
    T* shared_row = shared_memory<T>();
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        row_major_index rm(m, n);
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            shared_row[j] = d[rm(i, j)];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[rm(i, j)] = shared_row[s(j)];
        }
        __syncthreads();
    }        
}

}
}
