import json
import tensorflow as tf
import dlpack_cuda_class
import time

with open("weights/weights.json") as f:
    conf = json.load(f)

m = dlpack_cuda_class.Model(conf["Name"], conf["Provider"], conf["TensorApi"], conf["ModelPath"])
m.digest_model_template_tf(conf)
print("Digested")
tensors = {k: tf.experimental.dlpack.from_dlpack(v) for k, v in m.get_tensors().items()}
print(f"{[tensor for tensor in tensors.values()]}")
time.sleep(5)

m.to_cpu()
print("Moved to CPU")
tensors = {k: tf.experimental.dlpack.from_dlpack(v) for k, v in m.get_tensors().items()}
print(f"{[tensor for tensor in tensors.values()]}")
time.sleep(5)

m.to_gpu()
print("Moved to GPU")
tensors = {k: tf.experimental.dlpack.from_dlpack(v) for k, v in m.get_tensors().items()}
print(f"{[tensor for tensor in tensors.values()]}")
time.sleep(5)

m.to_cpu()
print("Moved to CPU")
tensors = {k: tf.experimental.dlpack.from_dlpack(v) for k, v in m.get_tensors().items()}
print(f"{[tensor for tensor in tensors.values()]}")
time.sleep(5)
