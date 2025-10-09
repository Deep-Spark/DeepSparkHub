/*
 Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
 Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
//#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>

#define nstreams 16

// global variables.
cudaStream_t stream[nstreams];
cublasHandle_t handle;

///////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Fprop_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
	    	         int batch,
  	  	         torch::Tensor &seq_len,
                         int heads,
		         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{

    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()) + (strided ? embed : 0)); 	// key
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr())); 				// query
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr())); 	        		// output

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams]: at::cuda::getCurrentCUDAStream());
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   seqlen[i],
                                   seqlen[i],
                                   embed,
                                   static_cast<const void*>(scale ? &alpha : &one),
                                   ptrA,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   CUDA_R_16F,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   enable_stream ? heads : batch*heads,
                                   CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
	ptrB = static_cast<void*>(static_cast<half*>(ptrB) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
	ptrC = static_cast<void*>(static_cast<half*>(ptrC) + heads*seqlen[i]*seqlen[i]);
    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Fprop_(torch::Tensor &A,
                    torch::Tensor &B,
                    torch::Tensor &C,
                    int batch,
                    torch::Tensor &seq_len,
                    int heads,
                    int embed,
		    bool scale,
		    bool strided,
		    bool enable_stream,
		    bool sync)
{

    float one = 1.0, zero = 0.0;

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()) + (strided ? 2*embed : 0));  // value
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));            		// query*key
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()));           		 // output

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams]: at::cuda::getCurrentCUDAStream());
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   embed,
                                   seqlen[i],
                                   seqlen[i],
                                   static_cast<const void*>(&one),
                                   ptrA,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   CUDA_R_16F,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   CUDA_R_16F,
                                   enable_stream ? heads*embed : batch*heads*embed,
                                   embed,
                                   enable_stream ? heads : batch*heads,
                                   CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + heads*seqlen[i]*seqlen[i]);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + seqlen[i]*heads*embed);

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Dgrad1_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
                         int batch,
                         torch::Tensor &seq_len,
                         int heads,
                         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{

    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()));           		// query
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()) + (strided ? embed : 0)); 	// grad_key

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_T,
                                   embed,
                                   seqlen[i],
                                   seqlen[i],
                                   static_cast<const void*>(scale ? &alpha : &one),
                                   ptrA,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   CUDA_R_16F,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   enable_stream ? heads : heads*batch,
                                   CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + heads*seqlen[i]*seqlen[i]);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Dgrad1_(torch::Tensor &A,
                     torch::Tensor &B,
                     torch::Tensor &C,
                     int batch,
                     torch::Tensor &seq_len,
                     int heads,
                     int embed,
		     bool scale,
		     bool strided,
		     bool enable_stream,
		     bool sync)
{

    float one = 1.0, zero = 0.0;

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()) + (strided ? 2*embed : 0));  // value
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()));

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   seqlen[i],
                                   seqlen[i],
                                   embed,
                                   static_cast<const void*>(&one),
                                   ptrA,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   CUDA_R_16F,
				   enable_stream ? heads*embed : batch*heads*embed,
                                   embed,
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   CUDA_R_16F,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   enable_stream ? heads : batch*heads,
                                   CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + seqlen[i]*heads*embed);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + heads*seqlen[i]*seqlen[i]);

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Dgrad2_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
                         int batch,
                         torch::Tensor &seq_len,
                         int heads,
                         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{

    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()) + (strided ? embed : 0));  	// key
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()));          		// grad query

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   embed,
                                   seqlen[i],
                                   seqlen[i],
                                   static_cast<const void*>(scale ? &alpha : &one),
                                   ptrA,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   ptrB,
                                   CUDA_R_16F,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   enable_stream ? heads : batch*heads,
                                   CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + heads*seqlen[i]*seqlen[i]);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Dgrad2_(torch::Tensor &A,
                     torch::Tensor &B,
                     torch::Tensor &C,
                     int batch,
                     torch::Tensor &seq_len,
                     int heads,
                     int embed,
		     bool scale,
		     bool strided,
		     bool enable_stream,
		     bool sync)
{

    float one = 1.0, zero = 0.0;

    int *seqlen = static_cast<int*>(seq_len.data_ptr());

    void *ptrA = static_cast<void*>(static_cast<half*>(A.data_ptr()));
    void *ptrB = static_cast<void*>(static_cast<half*>(B.data_ptr()));
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()) + (strided ? 2*embed : 0));  // grad-value

    for(int i = 0; i < (enable_stream ? batch : 1); i++) {
        cublasSetStream(handle, enable_stream ? stream[i%nstreams] : at::cuda::getCurrentCUDAStream());
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_T,
                                   embed,
                                   seqlen[i],
                                   seqlen[i],
                                   static_cast<const void*>(&one),
                                   ptrA,
                                   CUDA_R_16F,
				   enable_stream ? heads*embed : batch*heads*embed,
                                   embed,
                                   ptrB,
                                   CUDA_R_16F,
                                   seqlen[i],
                                   seqlen[i]*seqlen[i],
                                   static_cast<const void*>(&zero),
                                   ptrC,
                                   CUDA_R_16F,
                                   (enable_stream ? 1 : batch) * (strided ? heads*3*embed : heads*embed),
                                   strided ? 3*embed : embed,
                                   enable_stream ? heads : batch*heads,
                                   CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        ptrA = static_cast<void*>(static_cast<half*>(ptrA) + seqlen[i]*heads*embed);
        ptrB = static_cast<void*>(static_cast<half*>(ptrB) + heads*seqlen[i]*seqlen[i]);
        ptrC = static_cast<void*>(static_cast<half*>(ptrC) + (strided ? seqlen[i]*heads*3*embed : seqlen[i]*heads*embed));

    }
    for(int i = 0; i < (enable_stream ? nstreams : 0); i++) {
        if(sync) cudaStreamSynchronize(stream[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mha_cuda_extension()
{
    // CUDA Stream.
    for(int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // CuBlas Handle.
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("InitMHACUDAExtension", &init_mha_cuda_extension, "InitMHACUDAExtension");
  m.def("FastBmm1Fprop", &FastBmm1Fprop_, "FastBmm1Fprop");
  m.def("FastBmm1Dgrad1", &FastBmm1Dgrad1_, "FastBmm1Dgrad1");
  m.def("FastBmm1Dgrad2", &FastBmm1Dgrad2_, "FastBmm1Dgrad2");
  m.def("FastBmm2Fprop", &FastBmm2Fprop_, "FastBmm2Fprop");
  m.def("FastBmm2Dgrad1", &FastBmm2Dgrad1_, "FastBmm2Dgrad1");
  m.def("FastBmm2Dgrad2", &FastBmm2Dgrad2_, "FastBmm2Dgrad2");
}