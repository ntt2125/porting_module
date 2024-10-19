import torch
import numpy as np
from porting_module.utils.porting_utils import compare_numpy_files, load_numpy_file, load_pickle_file

input_data = load_numpy_file('temp/input_reshape_tf')

def torch_reshape(x, batch_size, input_channels, num_beats):
    return x.view(batch_size * input_channels, num_beats, -1)

torch_input = torch.from_numpy(input_data)
batch_size = 32
input_channels = 12
num_beats = 20

torch_output = torch_reshape(torch_input, batch_size, input_channels, num_beats).detach().numpy()

output_tf = load_numpy_file('temp/output_reshape_tf')

compare_numpy_files(output_tf, torch_output)