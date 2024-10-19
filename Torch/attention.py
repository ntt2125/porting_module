import torch
from ECG_Classification.src.models.components.imlenet import Attention
import numpy as np
from porting_module.utils.porting_utils import compare_numpy_files, load_numpy_file, load_pickle_file

def test_attention():
    
    tf_weights = load_pickle_file("temp/weights_att_tf")
    input = load_numpy_file('temp/input_att_tf')
    input_tensor = torch.from_numpy(input)
    
    output_tf = load_numpy_file('temp/output_att_tf')
    
    att_layer = Attention()
    att_layer(input_tensor)

    # Set weights
    att_layer.W.data = torch.from_numpy(tf_weights[0])
    att_layer.b.data = torch.from_numpy(tf_weights[1])
    att_layer.V.data = torch.from_numpy(tf_weights[2])
    
    output, _ = att_layer(input_tensor)
    
    print(output.shape)
    
    compare_numpy_files(output.detach().numpy(), output_tf)
    
if __name__=="__main__":
    test_attention()
