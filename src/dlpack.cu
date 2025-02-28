#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
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

py::dict load_pickle(const std::string& filename) {
    py::gil_scoped_acquire acquire;
    py::module pickle = py::module::import("pickle");
    py::object open = py::module::import("builtins").attr("open");
    py::object file = open(filename, "rb");
    py::object data = pickle.attr("load")(file);
    file.attr("close")();
    return data.cast<py::dict>();
}

std::unordered_map<std::string, float*> loadWeights(const std::string& filename) {
    std::unordered_map<std::string, float*> weights;
    std::unordered_map<std::string, std::vector<size_t>> shapes;
    
    try {
        // Load the pickle file
        std::cout << "Loading pickle file..." << std::endl;
        py::dict weights_dict = load_pickle(filename);
        
        // Iterate through the dictionary
        for (auto item : weights_dict) {
            std::string name = item.first.cast<std::string>();
            py::dict tensor_info = item.second.cast<py::dict>();
            
            // Get shape
            py::tuple shape_tuple = tensor_info["shape"].cast<py::tuple>();
            std::vector<size_t> shape;
            size_t total_elements = 1;
            
            for (auto dim : shape_tuple) {
                size_t dim_size = dim.cast<size_t>();
                shape.push_back(dim_size);
                total_elements *= dim_size;
            }
            
            // Get numpy array
            py::array_t<float> np_array = tensor_info["data"].cast<py::array_t<float>>();
            
            // Allocate C++ memory and copy data
            float* data = new float[total_elements];
            memcpy(data, np_array.data(), total_elements * sizeof(float));
            
            // Store in maps
            weights[name] = data;
            shapes[name] = shape;
            
            // Print information
            std::cout << "Loaded tensor: " << name << " with shape (";
            for (size_t i = 0; i < shape.size(); i++) {
                std::cout << shape[i];
                if (i < shape.size() - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
        }
        
        std::cout << "Successfully loaded " << weights.size() << " weight arrays" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
    }
    
    return weights;
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
                std::cout << "DUPE DELETER CALLED ON GPU" << std::endl;
            };

            dlpack->manager_ctx = nullptr;
            this->dlpack = dlpack;
            if(delete_host) {
                free(host_data);
            }
        }

        void deleter(DLManagedTensor* dlmt) {
            std::cout << "DELETER CALLED" << std::endl;
            std::cout << "Underlying Tensor : " << dlmt->dl_tensor.data << std::endl;
            delete[] dlmt->dl_tensor.shape;
            if(dlmt->dl_tensor.device.device_type == kDLCUDA){
                cudaFree(dlmt->dl_tensor.data);
            }
            else {
                free(dlmt->dl_tensor.data);
            }
            delete dlmt;
        }

        void to_cpu() {
            float* cpu_data;
            cpu_data = (float*) malloc(this->size * sizeof(float));
            cudaMemcpy(cpu_data, this->dlpack->dl_tensor.data, this->size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(this->dlpack->dl_tensor.data);  // Use cudaFree() for GPU memory
            this->dlpack->dl_tensor.data = cpu_data;
            this->dlpack->dl_tensor.device.device_type = kDLCPU;
            this->dlpack->dl_tensor.device.device_id = 0;
            this->device_type = kDLCPU;
            this->device_id = 0;
            dlpack->deleter = [](DLManagedTensor* dlmt) {
                std::cout << "DUPE DELETER CALLED ON CPU" << std::endl;
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
                std::cout << "DUPE DELETER CALLED ON GPU" << std::endl;
            };
        }

        void* get_ptr() {
            return this->dlpack->dl_tensor.data;
        }

        py::capsule get_tensor() {
            py::capsule dlpack_capsule(this->dlpack, "dltensor");
            return dlpack_capsule;
        }

        DLManagedTensor* get_dlpack() {
            return this->dlpack;
        }

        void delete_tensor() {
            this->dlpack->deleter(this->dlpack);
        }

        ~TFTensor() {
            std::cout << "EXECUTING DESTRUCTOR BECAUSE HAVE OWNERSHIP" << std::endl;
            this->deleter(this->dlpack);
        }
};

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
        void digest_model_template_tf(py::dict config);
        void to_cpu();
        void to_gpu();
        std::unordered_map<std::string, py::capsule> get_tensors();
        py::capsule get_tensor_by_name(const std::string& name);

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

void Model::digest_model_template_tf(py::dict config) {
    auto raw_tensors = loadWeights(this->file_path);
    std::unordered_map<std::string, TFTensor*> tensors;
    for(const auto& tensor : config["Tensors"]) {
        std::string name = tensor["name"].cast<std::string>();
        std::string type_str = tensor["type_str"].cast<std::string>();
        auto byte_offset = tensor["byte_offset"].cast<uint64_t>();
        auto n_dim = tensor["n_dim"].cast<int64_t>();
        auto vec_dims = tensor["dims"].cast<std::vector<int64_t>>();
        int64_t* dims = new int64_t[vec_dims.size()];
        std::copy(vec_dims.begin(), vec_dims.end(), dims);
        TFTensor* tf_tensor = new TFTensor{dims, n_dim, byte_offset, type_str};
        float* array = raw_tensors[name];
        tf_tensor->copy_to_gpu(array, false);
        tensors[name] = tf_tensor;
        std::cout << "Name : " << name << " Tensor : " << tf_tensor << std::endl; 
    }
    this->tensors = tensors;
}

std::unordered_map<std::string, py::capsule> Model::get_tensors() {
    std::unordered_map<std::string, py::capsule> out_map;
    for(auto& tensor : this->tensors) {
        std::cout << "First : " << tensor.first << "Second : " << tensor.second->get_tensor() << std::endl;
        out_map[tensor.first] = tensor.second->get_tensor();
    }
    return out_map;
}

py::capsule Model::get_tensor_by_name(const std::string& name) {
    if (this->tensors.find(name) != this->tensors.end()) {
        return this->tensors[name]->get_tensor();
    }
    throw std::runtime_error("Tensor not found: " + name);
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
        .def("get_ptr", [](TFTensor& self) {
            void* ptr = self.get_ptr();
            return py::capsule(ptr, [](void* p) {});
        })
        .def("get_tensor", &TFTensor::get_tensor)
        .def("delete_tensor", &TFTensor::delete_tensor);

    py::class_<Model>(m, "Model")
        .def(py::init<std::string, std::string, std::string, std::string>(), pybind11::arg("model_name"), pybind11::arg("provider"), pybind11::arg("tensor_api"), pybind11::arg("file_path"))
        .def("to_gpu", &Model::to_gpu)
        .def("to_cpu", &Model::to_cpu)
        .def("digest_model_template_tf", &Model::digest_model_template_tf)
        .def("get_tensors", &Model::get_tensors);
}