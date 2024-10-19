import torch
import torch.nn as nn
from ECG_Classification.src.models.components.imlenet import ResidualBlock
import numpy as np
from porting_module.utils.porting_utils import compare_numpy_files, load_numpy_file, load_pickle_file


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block1 = ResidualBlock(32, 64, downsample=True)
        self.block2 = ResidualBlock(64, 64, downsample=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# Load input
input_tensor = torch.from_numpy(load_numpy_file('temp/input_res_tf')).float()
input_tensor = input_tensor.permute(0, 2, 1)  # Adjust for PyTorch channel-first format

# Load TensorFlow weights
tf_weights = load_pickle_file('temp/weights_res_tf')

# Create PyTorch model
model = ResNet()

# Convert and load TensorFlow weights to PyTorch model
def load_tf_weights_to_pytorch(pytorch_model, tf_weights):
    pytorch_state_dict = pytorch_model.state_dict()
    tf_weight_index = 0

    for name, param in pytorch_state_dict.items():
        if 'weight' in name:
            if len(param.shape) == 3:  # Conv1d weight
                pytorch_state_dict[name] = torch.from_numpy(tf_weights[tf_weight_index].transpose(2, 1, 0))
            else:  # BatchNorm weight
                pytorch_state_dict[name] = torch.from_numpy(tf_weights[tf_weight_index])
            tf_weight_index += 1
        elif 'bias' in name:
            pytorch_state_dict[name] = torch.from_numpy(tf_weights[tf_weight_index])
            tf_weight_index += 1
        elif 'running_mean' in name or 'running_var' in name:
            pytorch_state_dict[name] = torch.from_numpy(tf_weights[tf_weight_index])
            tf_weight_index += 1

    pytorch_model.load_state_dict(pytorch_state_dict)

load_tf_weights_to_pytorch(model, tf_weights)

# Process input
with torch.no_grad():
    output_torch = model(input_tensor)

# Load TensorFlow output
output_tf = load_numpy_file('temp/output_res_tf')
output_tf = torch.from_numpy(output_tf).float()
output_tf = output_tf.permute(0, 2, 1)  # Adjust for PyTorch channel-first format

# Compare outputs
diff = torch.abs(output_torch - output_tf)
max_diff = torch.max(diff)
mean_diff = torch.mean(diff)

print(f"Max difference: {max_diff.item()}")
print(f"Mean difference: {mean_diff.item()}")

if max_diff.item() < 1e-5:
    print("The outputs are practically identical.")
else:
    print("There are significant differences between the outputs.")