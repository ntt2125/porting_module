import tensorflow as tf
import numpy as np
from porting_module.utils.porting_utils import save_numpy_file, save_pickle_file 

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
def test_batchnorm():
    # Initialize input with shape (32, 50, 64)
    input_data = np.random.randn(32, 50, 64).astype(np.float32)

    # Create BatchNorm layer
    batchnorm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

    # Apply BatchNorm to input
    output = batchnorm(input_data, training=True)

    # Get the weights of the BatchNorm layer
    weights = batchnorm.get_weights()

    # Save input, weights, and output
    save_numpy_file(input_data, 'temp/input_bn_tf')
    save_pickle_file(weights, 'temp/weights_bn_tf')
    save_numpy_file(output.numpy(), 'temp/output_bn_tf')

    print("TensorFlow BatchNorm shape:", output.shape)
    # print("Input, weights, and output saved.")


if __name__=="__main__":
    test_batchnorm()
    
# Print BatchNorm parameters
# print("gamma (scale):", weights[0])
# print("beta (offset):", weights[1])
# print("moving_mean:", weights[2])
# print("moving_variance:", weights[3])