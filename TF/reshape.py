import numpy as np
import tensorflow as tf
from porting_module.utils.porting_utils import save_numpy_file, save_pickle_file

# Create input data
input_data = np.random.rand(7680, 128).astype(np.float32)

# TensorFlow reshape
def tf_reshape(x, config):
    return tf.keras.backend.reshape(x, (-1, int(config.signal_len / config.beat_len), 128))

class Config:
    signal_len = 1000
    beat_len = 50

config = Config()

tf_input = tf.constant(input_data)
tf_output = tf_reshape(tf_input, config)

save_numpy_file(input_data, 'temp/input_reshape_tf')
save_numpy_file(tf_output.numpy(), 'temp/output_reshape_tf')

print(tf_output.shape)