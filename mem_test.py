import tensorflow as tf
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

new = tf.random.uniform([2000, 3000])
print("CREATED")
time.sleep(5)


