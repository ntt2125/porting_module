import numpy as np
import pickle
import os

def load_numpy_file(file_path):
    """
    Load a numpy file (.npy) and return its contents.

    Args:
    file_path (str): Path to the .npy file

    Returns:
    numpy.ndarray: Contents of the numpy file
    """
    if not os.path.exists(f'porting_module/{file_path}.npy'):
        raise FileNotFoundError(f"File not found: {file_path},npy")
    
    try:
        return np.load(f'porting_module/{file_path}.npy', allow_pickle=True)
    except Exception as e:
        raise Exception(f"Error loading numpy file: {e}")

def save_numpy_file(data, file_path):
    """
    Save data to a numpy file (.npy).

    Args:
    data (numpy.ndarray): Data to be saved
    file_path (str): Path where the file will be saved

    Returns:
    None
    """
    try:
        np.save(f'porting_module/{file_path}.npy', data)
        # print(f"File saved successfully: {file_path}.npy")
    except Exception as e:
        raise Exception(f"Error saving numpy file: {e}")

def load_pickle_file(file_path):
    """
    Load a pickle file and return its contents.

    Args:
    file_path (str): Path to the pickle file

    Returns:
    object: Contents of the pickle file
    """
    if not os.path.exists(f'porting_module/{file_path}.pkl'):
        raise FileNotFoundError(f"File not found: {file_path}.pkl")
    
    try:
        with open(f'porting_module/{file_path}.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading pickle file: {e}")

def save_pickle_file(data, file_path):
    """
    Save data to a pickle file.

    Args:
    data (object): Data to be saved
    file_path (str): Path where the file will be saved

    Returns:
    None
    """
    try:
        with open(f'porting_module/{file_path}.pkl', 'wb') as f:
            pickle.dump(data, f)
        # print(f"File saved successfully: {file_path}.pkl")
    except Exception as e:
        raise Exception(f"Error saving pickle file: {e}")

def compare_numpy_files(array1, array2, rtol=1e-5, atol=1e-8):
    """
    Compare two numpy files and return their differences.

    Args:
    file1_path (str): Path to the first numpy file
    file2_path (str): Path to the second numpy file
    rtol (float): Relative tolerance for comparison (default: 1e-5)
    atol (float): Absolute tolerance for comparison (default: 1e-8)

    Returns:
    dict: A dictionary containing comparison results
    """

    if array1.shape != array2.shape:
        result =  {
            "equal": False,
            "error": f"Shape mismatch: {array1.shape} vs {array2.shape}"
        }
        print(result)

    try:
        np.testing.assert_allclose(array1, array2, rtol=rtol, atol=atol)
        result = {
            "equal": True,
            "max_diff": np.max(np.abs(array1 - array2)),
            "mean_diff": np.mean(np.abs(array1 - array2))
        }
        
        print(result)
    except AssertionError as e:
        result =  {
            # "equal": False,
            # "error": str(e),
            "max_diff": np.max(np.abs(array1 - array2)),
            "mean_diff": np.mean(np.abs(array1 - array2))
        }
        print(result)

# Example usage
if __name__ == "__main__":
    # Example numpy file operations
    example_data = np.array([1, 2, 3, 4, 5])
    save_numpy_file(example_data, "example")
    loaded_data = load_numpy_file("example")
    print("Loaded numpy data:", loaded_data)

    # Example pickle file operations
    example_dict = {"a": 1, "b": 2, "c": 3}
    save_pickle_file(example_dict, "example")
    loaded_dict = load_pickle_file("example")
    print("Loaded pickle data:", loaded_dict)

    # # Example comparison
    # save_numpy_file(np.array([1, 2, 3]), "file1.npy")
    # save_numpy_file(np.array([1, 2, 3.00001]), "file2.npy")
    # comparison_result = compare_numpy_files("file1.npy", "file2.npy")
    # print("Comparison result:", comparison_result)