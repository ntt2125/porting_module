import tensorflow as tf
import numpy as np
from porting_module.utils.porting_utils import save_numpy_file, save_pickle_file

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def test_conv1d():
    # Initialize input with shape (32, 50, 1)
    input_data = np.random.randn(32, 50, 1).astype(np.float32)

    # Create Conv1D layer
    conv1d = tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')

    # Apply Conv1D to input
    output = conv1d(input_data)

    # Get the weights of the Conv1D layer
    weights = conv1d.get_weights()

    # Save input, weights, and output
    save_numpy_file(input_data, 'temp/input')
    save_pickle_file(weights, 'temp/weights')
    save_numpy_file(output, 'temp/output')

    print("TensorFlow Conv1D shape:", output.shape)
    # print("Input, weights, and output saved.")

if __name__=='__main__':
    test_conv1d()