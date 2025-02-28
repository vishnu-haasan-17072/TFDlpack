#include <pybind11/pybind11.h>
#include <dlpack/dlpack.h>  // Official DLPack header
#include <cuda_runtime.h>   // CUDA memory management
#include <vector>
#include <memory>
#include <iostream>

namespace py = pybind11;

// Function to create a DLPack tensor on GPU
py::capsule CreateDLPackTensor(int x, int y) {
    // Allocate GPU memory (using CUDA runtime API)
    float* gpu_data = nullptr;
    const int size = x*y; // 2x3 tensor

    // Allocate memory on GPU
    cudaError_t err = cudaMalloc(&gpu_data, size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed");
    }

    std::cout << "Allocated" << std::endl;
    // Initialize the data on the CPU
    float *host_data = new float[size];
    for(int i=0;i<x;i++){
        for(int j=0;j<y;j++){
            host_data[i * y + j] = (i + 1) * 10 + j;;
        }
    }

    std::cout << "Inited" << std::endl;

    // Copy data from host to GPU
    cudaMemcpy(gpu_data, host_data, size * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "MEMCOPIED" << std::endl;

    // Create a DLManagedTensor
    auto* dlpack = new DLManagedTensor;

    // Fill the DLTensor structure
    auto& dl_tensor = dlpack->dl_tensor;

    // Set the data pointer to GPU memory
    dl_tensor.data = gpu_data;

    // Set the device type to CUDA (GPU)
    dl_tensor.device.device_type = kDLCUDA;
    dl_tensor.device.device_id = 0;  // Use first GPU

    // Set dimensions (2x3 tensor)
    dl_tensor.ndim = 2;
    dl_tensor.shape = new int64_t[2]{x, y};

    // Set the strides (nullptr means compact tensor)
    dl_tensor.strides = nullptr;

    // Set the byte offset
    dl_tensor.byte_offset = 0;

    // Set the data type (float32)
    dl_tensor.dtype.code = kDLFloat;  // Use DLPack enum for float
    dl_tensor.dtype.bits = 32;        // 32-bit
    dl_tensor.dtype.lanes = 1;        // Scalar type

    // Set the deleter function
    dlpack->deleter = [](DLManagedTensor* dlmt) {
        // Free the shape array
        delete[] dlmt->dl_tensor.shape;

        // Free the GPU memory
        cudaFree(dlmt->dl_tensor.data);

        // Free the DLManagedTensor
        delete dlmt;
    };

    // Set the manager context (can be used for additional context)
    dlpack->manager_ctx = nullptr;

    // Create a Python capsule to hold and manage the DLManagedTensor
    py::capsule dlpack_capsule(dlpack, "dltensor");

    delete[] host_data;

    return dlpack_capsule;
}

PYBIND11_MODULE(dlpack_cuda_module, m) {
    m.doc() = "Module to create DLPack tensor on CUDA";
    m.def("create_dlpack_tensor", &CreateDLPackTensor, "Create a DLPack tensor on CUDA");
}
