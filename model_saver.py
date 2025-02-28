import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pickle
import json
import os

def extract_and_save_weights(output_file="weights.pkl"):
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    variables = model.variables

    weights_dict = {}
    total_size = 0
    configuration = {
        "Name": "USE-4",
        "Provider": "Google",
        "TensorApi": "TF",
        "ModelPath": f"{os.getcwd()}/{output_file}",
        "Tensors": []
    }
    
    for var in variables:
        name = var.name
        tensor = var.numpy().astype(np.float32)  # Ensure float32 format for consistency
        shape = tensor.shape
        total_size += tensor.nbytes
        
        configuration["Tensors"].append({
            'name': name,
            'dims': list(tensor.shape),
            'n_dim': tensor.ndim,
            'byte_offset': 0,
            'type_str': 'FLOAT32'
        })

        # Store in dictionary
        weights_dict[name] = {
            'shape': shape,
            'data': tensor
        }

    with open(output_file, 'wb') as f:
        pickle.dump(weights_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    json_file = f"{output_file.split('.')[0]}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(configuration, f, ensure_ascii=False, indent=4)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved {len(weights_dict)} weight arrays to {output_file} ({file_size_mb:.2f} MB)")
    print(f"Total weights data size: {total_size / (1024 * 1024):.2f} MB")

def load_dummy_weights(output_file):
    weights = {
        "r1": tf.random.uniform((2000, 300)),
        "r2": tf.random.uniform((2500, 350))
    }
    weights_dict = {}
    base_name = '.'.join(output_file.split('.')[:-1])
    os.mkdir(f"{os.getcwd()}/{base_name}")
    configuration = {
        "Name": "Random",
        "Provider": "Google",
        "TensorApi": "TF",
        "ModelPath": f"{os.getcwd()}/{base_name}/{output_file}",
        "Tensors": []
    }

    for name, weight in weights.items():
        tensor = weight.numpy().astype(np.float32)
        shape = tensor.shape

        configuration["Tensors"].append({
            "name": name,
            "dims": list(tensor.shape),
            "n_dim": tensor.ndim,
            "byte_offset": 0,
            "type_str": "FLOAT32"
        })

        weights_dict[name] = {
            "shape": shape,
            "data": tensor
        }

    with open(f"{os.getcwd()}/{base_name}/{output_file}", 'wb') as f:
        pickle.dump(weights_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    json_file = f"{os.getcwd()}/{base_name}/{base_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(configuration, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    # extract_and_save_weights()
    load_dummy_weights("weights.pkl")

 