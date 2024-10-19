import torch
import torch.nn as nn
import numpy as np
from porting_module.utils.porting_utils import load_numpy_file, compare_numpy_files, load_pickle_file

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def test_batchnorm():
    # Load input data from TensorFlow
    input_data = load_numpy_file('temp/input_bn_tf')
    input_tensor = torch.from_numpy(input_data)

    # Load weights from TensorFlow
    tf_weights = load_pickle_file('temp/weights_bn_tf')
    gamma, beta, moving_mean, moving_var = tf_weights

    # Create BatchNorm layer
    batchnorm = nn.BatchNorm1d(64, eps=0.001, momentum=0.01)  # Note: PyTorch momentum = 1 - TF momentum

    # Set the weights and bias
    batchnorm.weight.data = torch.from_numpy(gamma)
    batchnorm.bias.data = torch.from_numpy(beta)
    batchnorm.running_mean.data = torch.from_numpy(moving_mean)
    batchnorm.running_var.data = torch.from_numpy(moving_var)

    # Apply BatchNorm to input
    batchnorm.train()  # Set to training mode
    output = batchnorm(input_tensor.transpose(1, 2))  # PyTorch expects (N, C, L) format
    output = output.transpose(1, 2)  # Convert back to (N, L, C) for comparison

    print("PyTorch BatchNorm shape:", output.shape)

    # Load TensorFlow output for comparison
    tf_output = load_numpy_file('temp/output_bn_tf')

    # Compare outputs
    comparison_result = compare_numpy_files(tf_output, output.detach().numpy())
    print("Comparison result:", comparison_result)

if __name__=="__main__":
    test_batchnorm()
# if comparison_result['equal']:
#     print("The outputs are practically identical.")
# else:
#     print("There are differences between the outputs.")
#     print(f"Maximum difference: {comparison_result['max_diff']}")
#     print(f"Mean difference: {comparison_result['mean_diff']}")

# # Print BatchNorm parameters
# print("gamma (weight):", batchnorm.weight.data)
# print("beta (bias):", batchnorm.bias.data)
# print("running_mean:", batchnorm.running_mean.data)
# print("running_var:", batchnorm.running_var.data)