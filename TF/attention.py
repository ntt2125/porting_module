from IMLE_Net.models.IMLENet import attention
from porting_module.utils.porting_utils import save_numpy_file, save_pickle_file
import numpy as np
import tensorflow as tf


tf.random.set_seed(42)
np.random.seed(42)


def test_attention():
    input_data = np.random.randn(7680, 13, 128).astype(np.float32)

    # Create and apply attention layer
    att_layer = attention()
    output, _ = att_layer(input_data)
    
    weights = att_layer.get_weights()
    
    save_numpy_file(input_data, 'temp/input_att_tf')
    save_numpy_file(output, 'temp/output_att_tf')
    save_pickle_file(weights, 'temp/weights_att_tf')

if __name__=="__main__":
    test_attention()