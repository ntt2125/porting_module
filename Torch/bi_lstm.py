import torch
import torch.nn as nn
import numpy as np
from utils.porting_utils import load_numpy_file, load_pickle_file, compare_numpy_files

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
batch_size = 32
sequence_length = 50
input_size = 100
hidden_size = 64

# Load input data from TensorFlow
input_data = load_numpy_file('temp/input_tf_bi_lstm')
input_tensor = torch.from_numpy(input_data)

# Load weights from TensorFlow
tf_weights = load_pickle_file('temp/weights_tf_bi_lstm')

# Create Bi-LSTM layer
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        output, _ = self.bilstm(x)
        return output

bilstm = BiLSTM(input_size, hidden_size)

# Set the weights
def set_weights(model, tf_weights):
    # TensorFlow weights order: [forward_kernel, forward_recurrent_kernel, forward_bias,
    #                            backward_kernel, backward_recurrent_kernel, backward_bias]
    # PyTorch weights order: [weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
    #                         weight_ih_l0_reverse, weight_hh_l0_reverse, bias_ih_l0_reverse, bias_hh_l0_reverse]
    
    # Forward weights
    model.bilstm.weight_ih_l0.data = torch.from_numpy(tf_weights[0].T)
    model.bilstm.weight_hh_l0.data = torch.from_numpy(tf_weights[1].T)
    
    # Forward bias
    forward_bias = tf_weights[2]
    model.bilstm.bias_ih_l0.data = torch.from_numpy(forward_bias[:hidden_size*4])
    model.bilstm.bias_hh_l0.data = torch.zeros_like(model.bilstm.bias_hh_l0) 
    
    # Backward weights
    model.bilstm.weight_ih_l0_reverse.data = torch.from_numpy(tf_weights[3].T)
    model.bilstm.weight_hh_l0_reverse.data = torch.from_numpy(tf_weights[4].T)
    
    # Backward bias
    backward_bias = tf_weights[5]
    model.bilstm.bias_ih_l0_reverse.data = torch.from_numpy(backward_bias[:hidden_size*4])
    model.bilstm.bias_hh_l0_reverse.data = torch.zeros_like(model.bilstm.bias_hh_l0_reverse)

# print("TensorFlow weight shapes:")
# for i, w in enumerate(tf_weights):
#     print(f"  Weight {i}: {w.shape}")

# print("\nPyTorch weight shapes (before setting):")
# for name, param in bilstm.named_parameters():
#     print(f"  {name}: {param.shape}")



# Apply Bi-LSTM to input
# 
bilstm(input_tensor)

set_weights(bilstm, tf_weights)
bilstm.eval()  # Set to evaluation mode
with torch.no_grad():
    output = bilstm(input_tensor)

print("PyTorch Bi-LSTM output shape:", output.shape)

# Load TensorFlow output for comparison
tf_output = load_numpy_file('temp/output_tf_bi_lstm')

# Compare outputs
comparison_result = compare_numpy_files(tf_output, output.detach().numpy())
# print("Comparison result:", comparison_result)

# if comparison_result['equal']:
#     print("The outputs are practically identical.")
# else:
#     print("There are differences between the outputs.")
#     print(f"Maximum difference: {comparison_result['max_diff']}")
#     print(f"Mean difference: {comparison_result['mean_diff']}")

# # Print some information about the weights
# print("Number of parameter tensors:", len(list(bilstm.parameters())))
# print("Shapes of parameter tensors:")
# for name, param in bilstm.named_parameters():
#     print(f"  {name}: {param.shape}")