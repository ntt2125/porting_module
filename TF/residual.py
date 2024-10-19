from IMLE_Net.models.IMLENet import residual_block
from porting_module.utils.porting_utils import save_numpy_file, save_pickle_file
import numpy as np
import tensorflow as tf
import pickle

tf.random.set_seed(42)
np.random.seed(42)

input_shape = (7680, 50, 32)
input_tensor = tf.random.normal(input_shape)

# Process the input through the residual block
output_tensor = residual_block(input_tensor, downsample=False, filters=32, kernel_size=7)

# Save input and output as .npy files
np.save('input_tf.npy', input_tensor.numpy())
np.save('output_tf.npy', output_tensor.numpy())

# Get and save weights
weights = []
for layer in [output_tensor.op.inputs[0].op.inputs[0].op,  # First Conv1D
              output_tensor.op.inputs[0].op.inputs[0].op.inputs[0].op,  # First BatchNorm
              output_tensor.op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[0].op,  # Second Conv1D
              output_tensor.op.inputs[0].op]:  # Second BatchNorm
    if isinstance(layer, tf.keras.layers.Conv1D):
        weights.extend([layer.weights[0].numpy(), layer.weights[1].numpy()])
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        weights.extend([layer.gamma.numpy(), layer.beta.numpy(), 
                        layer.moving_mean.numpy(), layer.moving_variance.numpy()])

with open('weights_tf.pickle', 'wb') as f:
    pickle.dump(weights, f)

print("TensorFlow processing complete. Files saved.")