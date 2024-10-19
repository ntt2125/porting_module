import tensorflow as tf
import numpy as np
from utils.porting_utils import save_numpy_file, save_pickle_file

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
batch_size = 32
sequence_length = 50
input_size = 100
hidden_size = 64

# Initialize input with shape (batch_size, sequence_length, input_size)
input_data = np.random.randn(batch_size, sequence_length, input_size).astype(np.float32)

# Create Bi-LSTM layer
bilstm = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(hidden_size, return_sequences=True)
)

# Apply Bi-LSTM to input
output = bilstm(input_data)

# Get the weights of the Bi-LSTM layer
weights = bilstm.get_weights()

# Save input, weights, and output
save_numpy_file(input_data, 'temp/input_tf_bi_lstm')
save_pickle_file(weights, 'temp/weights_tf_bi_lstm')
save_numpy_file(output.numpy(), 'temp/output_tf_bi_lstm')

# print("TensorFlow Bi-LSTM output shape:", output.shape)
# print("Input, weights, and output saved.")

# # Print some information about the weights
# print("Number of weight arrays:", len(weights))
# print("Shapes of weight arrays:")
# for i, w in enumerate(weights):
#     print(f"  Weight {i}: {w.shape}")