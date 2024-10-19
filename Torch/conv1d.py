import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from porting_module.utils.porting_utils import load_numpy_file, load_pickle_file, compare_numpy_files

# todo: set the same config between two file.

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def test_conv1d():
    # Load input data from TensorFlow
    input_data = load_numpy_file('temp/input')
    input_tensor = torch.from_numpy(input_data).permute(0, 2, 1)  # Change to (batch, channels, length)

    # Load weights from TensorFlow
    tf_weights = load_pickle_file('temp/weights')
    kernel = torch.from_numpy(tf_weights[0].transpose(2, 1, 0))  # Change to (out_channels, in_channels, kernel_size)
    bias = torch.from_numpy(tf_weights[1])

    # Create Conv1D layer
    conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3)

    # Set the weights and bias
    conv1d.weight.data = kernel
    conv1d.bias.data = bias

    # Apply Conv1D to input
    output = conv1d(input_tensor).detach().numpy()
    # output = F.relu(output).detach().numpy()  # Apply ReLU activation

    # Load TensorFlow output for comparison
    tf_output = load_numpy_file('temp/output')
    tf_output_tensor = tf_output.transpose(0, 2, 1) # Change to (batch, channels, length)

    # print("PyTorch Conv1D shape:", output.shape)
    # print("TensorFlow Conv1D shape:", tf_output_tensor.shape)

    # Compare outputs
    compare_numpy_files(output, tf_output_tensor)
    
if __name__=="__main__":
    test_conv1d()