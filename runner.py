import tensorflow as tf
import dlpack_cuda_class
import time

dlpack = dlpack_cuda_class.create_tf_tensor([2000, 3000])
tensor = tf.experimental.dlpack.from_dlpack(dlpack.get_tensor())
print(f"INITED ON GPU, pointer : {dlpack.get_ptr()}")
time.sleep(5)
# tensor

print("GONNA MOVE TO CPU")
dlpack.to_cpu()
tensor = tf.experimental.dlpack.from_dlpack(dlpack.get_tensor())
print(f"INITED ON CPU, pointer : {dlpack.get_ptr()}")
time.sleep(5)

print("GONNA MOVE TO GPU")
dlpack.to_gpu()
tensor = tf.experimental.dlpack.from_dlpack(dlpack.get_tensor())
print("DONE")
print(f"INITED ON GPU, pointer : {dlpack.get_ptr()}")
time.sleep(5)

dlpack.to_cpu()
# tensor = tf.experimental.dlpack.from_dlpack(dlpack.get_tensor())
print(f"INITED ON CPU, pointer : {dlpack.get_ptr()}")
time.sleep(5)
