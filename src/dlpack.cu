#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>  // Official DLPack header
#include <cuda_runtime.h>   // CUDA memory management
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>

namespace py = pybind11;

struct DLDataTypeInternal{
    DLDataTypeCode code;
    uint8_t bits;
    uint16_t lanes; 
};

DLDataTypeInternal DLPackGetType(const std::string& type_str) {
    if(type_str == "INT8") {
        DLDataTypeInternal d = {kDLInt, 8, 1};
        return d;
    }
    else if(type_str == "INT16") {
        DLDataTypeInternal d = {kDLInt, 16, 1};
        return d;
    }
    else if(type_str == "INT32") {
        DLDataTypeInternal d = {kDLInt, 32, 1};
        return d;
    }
    else if(type_str == "INT64") {
        DLDataTypeInternal d = {kDLInt, 64, 1};
        return d;
    }
    else if(type_str == "UINT8") {
        DLDataTypeInternal d = {kDLUInt, 8, 1};
        return d;
    }
    else if(type_str == "UINT16") {
        DLDataTypeInternal d = {kDLUInt, 16, 1};
        return d;
    }
    else if(type_str == "UINT32") {
        DLDataTypeInternal d = {kDLUInt, 32, 1};
        return d;
    }
    else if(type_str == "UINT64") {
        DLDataTypeInternal d = {kDLUInt, 64, 1};
        return d;
    }
    else if(type_str == "FLOAT16") {
        DLDataTypeInternal d = {kDLFloat, 16, 1};
        return d;
    }
    else if(type_str == "FLOAT32") {
        DLDataTypeInternal d = {kDLFloat, 32, 1};
        return d;
    }
    else if(type_str == "FLOAT64") {
        DLDataTypeInternal d = {kDLFloat, 64, 1};
        return d;
    }
    else {
        DLDataTypeInternal d = {kDLFloat, 32, 1};
        return d;
    }
}

class TFTensor {
    private:
        DLDeviceType device_type;
        int32_t device_id;
        int64_t* dims;
        DLManagedTensor* dlpack;
        int32_t n_dim;
        int64_t* strides;
        uint64_t byte_offset;
        DLDataTypeInternal dtype;
        int64_t size;

    public:
        TFTensor(int64_t* dims, int64_t n_dim, uint64_t byte_offset, const std::string& type_str) {
            this->n_dim = n_dim;
            this->dims = new int64_t[n_dim];
            for (int i = 0; i < n_dim; i++) {
                this->dims[i] = dims[i];
            }
            this->strides = nullptr;
            this->byte_offset = byte_offset;
            this->dtype = DLPackGetType(type_str);
            int64_t size = 1;
            for (int i = 0; i < n_dim; i++) {
                size *= dims[i];
            }
            this->size = size;
            this->device_type = kDLCUDA;
            this->device_id = 0;
        }

        void copy_to_gpu(float* host_data, bool delete_host) {
            float* gpu_data = nullptr;
            
            cudaError_t err = cudaMalloc(&gpu_data, this->size * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed");
            }

            cudaMemcpy(gpu_data, host_data, this->size * sizeof(float), cudaMemcpyHostToDevice);
            auto* dlpack = new DLManagedTensor;
            auto& dl_tensor = dlpack->dl_tensor;

            dl_tensor.data = gpu_data;
            dl_tensor.device.device_type = this->device_type;
            dl_tensor.device.device_id = this->device_id;
            dl_tensor.ndim = this->n_dim;
            dl_tensor.shape = this->dims;
            dl_tensor.strides = this->strides;
            dl_tensor.byte_offset = this->byte_offset;
            dl_tensor.dtype.code = this->dtype.code;
            dl_tensor.dtype.bits = this->dtype.bits;
            dl_tensor.dtype.lanes = this->dtype.lanes;

            dlpack->deleter = [](DLManagedTensor* dlmt) {
            
                delete[] dlmt->dl_tensor.shape;

                cudaFree(dlmt->dl_tensor.data);

                delete dlmt;
            };

            dlpack->manager_ctx = nullptr;
            this->dlpack = dlpack;

            if(delete_host) {
                free(host_data);
            }
        }

        void to_cpu() {
            float* cpu_data;
            cpu_data = (float*) malloc(this->size * sizeof(float));
            cudaMemcpy(cpu_data, this->dlpack->dl_tensor.data, this->size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(this->dlpack->dl_tensor.data);
            this->dlpack->dl_tensor.data = cpu_data;
            this->dlpack->dl_tensor.device.device_type = kDLCPU;
            this->dlpack->dl_tensor.device.device_id = 0;
            this->device_type = kDLCPU;
            this->device_id = 0;

            dlpack->deleter = [](DLManagedTensor* dlmt) {
                delete[] dlmt->dl_tensor.shape;
                free(dlmt->dl_tensor.data);
                delete dlmt;
            };
        }

        void to_gpu() {
            float* gpu_data;
            cudaError_t err = cudaMalloc(&gpu_data, this->size * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed");
            }

            cudaMemcpy(gpu_data, this->dlpack->dl_tensor.data, this->size * sizeof(float), cudaMemcpyHostToDevice);
            free(this->dlpack->dl_tensor.data);
            this->dlpack->dl_tensor.data = gpu_data;
            this->dlpack->dl_tensor.device.device_type = kDLCUDA;
            this->dlpack->dl_tensor.device.device_id = 0;
            this->device_type = kDLCUDA;
            this->device_id = 0;

            dlpack->deleter = [](DLManagedTensor* dlmt) {
                delete[] dlmt->dl_tensor.shape;
                cudaFree(dlmt->dl_tensor.data);
                delete dlmt;
            };
        }

        void* get_ptr() {
            return this->dlpack->dl_tensor.data;
        }

        py::capsule get_tensor() {
            py::capsule dlpack_capsule(this->dlpack, "dltensor");

            return dlpack_capsule;
        }

        void delete_tensor() {
            this->dlpack->deleter(this->dlpack);
            this->dlpack = nullptr;
        }

        ~TFTensor() {
            this->dlpack->deleter(this->dlpack);
            // free(this->dims);
            // free(this->strides);
        }
};

class TFTensorTemplate {
    std::vector<std::int64_t> dims;
    int64_t n_dim;
    uint64_t byte_offset;
    std::string dtype;

    TFTensorTemplate(const std::vector<std::int64_t>& dims, int64_t n_dim, uint64_t byte_offset, const std::string& dtype);
}

TFTensorTemplate::TFTensorTemplate(const std::vector<std::int64_t>& dims, int64_t n_dim, uint64_t byte_offset, const std::string& dtype) {
    this->dims = dims;
    this->n_dim = n_dim;
    this->byte_offset = byte_offset;
    this->dtype = dtype;
}

class Model {
    public:
        std::unordered_map<std::string, TFTensor*> tensors;
        std::string model_name;
        std::string device;
        std::string file_path;
        std::string provider;
        std::string tensor_api;
        std::time_t start_time;

        Model(const std::string& model_name, const std::string& provider, const std::string& tensor_api, const std::string& file_path);
        ~Model();
        void digest_model_template(const std::vector<TFTensorTemplate*>& tensor_templates);
        void to_cpu();
        void to_gpu();
};

Model::Model(const std::string& model_name, const std::string& provider, const std::string& tensor_api, const std::string& file_path) {
    this->device = "GPU";
    this->model_name = model_name;
    this->file_path = file_path;
    this->provider = provider;
    this->tensor_api = tensor_api;
    this->start_time = std::time(nullptr);
}

Model::~Model() {
    for(auto& tensor : this->tensors) {
        delete tensor.second;
    }
}

void Model::to_cpu() {
    for(auto& tensor :  this->tensors) {
        tensor.second->to_cpu();
    }
    this->device = "CPU";
}

void Model::to_gpu() {
    for(auto& tensor : this->tensors) {
        tensor.second->to_gpu();
    }
    this->device = "GPU";
}

std::unordered_map<std::string, *TFTensor> Model::get_tensors() {
    return this->tensors;
}


TFTensor* get_tf_tensor(std::vector<std::int64_t> dimensions) {
    int64_t* dims = new int64_t[dimensions.size()];
    std::copy(dimensions.begin(), dimensions.end(), dims);
    int64_t n_dim = dimensions.size();
    uint64_t byte_offset = 0;
    TFTensor* tensor = new TFTensor{dims, n_dim, byte_offset, "FLOAT32"};
    int64_t size = 1;
    for (int i = 0; i < n_dim; i++) { 
        size *= dims[i]; 
    }
    float *host_data = new float[size];
    auto x = dimensions[0];
    auto y = dimensions[1];
    for(int i=0;i<x;i++){
        for(int j=0;j<y;j++){
            host_data[i * y + j] = (i + 1) * 10 + j;;
        }
    }

    tensor->copy_to_gpu(host_data, true);    
    return tensor;
}



PYBIND11_MODULE(dlpack_cuda_class, m) {
    m.doc() = "Module to create DLPack tensor on CUDA";
    // m.def("create_dlpack_tensor", &CreateDLPackTensor, "Create a DLPack tensor on CUDA");
    m.def("create_tf_tensor", &get_tf_tensor, "Create TF Tensor");

    py::class_<TFTensor>(m, "TF")
        .def("to_gpu", &TFTensor::to_gpu)
        .def("to_cpu", &TFTensor::to_cpu)
        .def("get_ptr", &TFTensor::get_ptr)
        .def("get_tensor", &TFTensor::get_tensor)
        .def("delete_tensor", &TFTensor::delete_tensor);
}