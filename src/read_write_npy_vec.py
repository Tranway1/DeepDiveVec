import os
import sys
import time

import numpy as np

LOG_NPY_READ = open("npy_read_log.csv", "a")
LOG_NPY_WRITE = open("npy_write_log.csv", "a")


def load_encoded_prompts(filename):
    # Load the encoded prompts from the .npy file
    encoded_prompts = np.load(filename, allow_pickle=True)
    return encoded_prompts


def convert_to_2d_numpy(tensor):
    if tensor.ndim == 3:
        # Find the dimension with the minimal length
        min_dim = np.argmin(tensor.shape)
        tensor = np.squeeze(tensor, axis=min_dim)
    elif tensor.ndim != 2:
        raise ValueError("Tensor must be 2D or 3D")
    return tensor



# Check if at least one file name is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <file1> <file2> ...")
    sys.exit(1)
dir_path = '../embeddings/'
embeddings = sys.argv[1:]

# embeddings = [f for f in os.listdir(dir_path) if f.endswith('embed.npy') or f.endswith('embeddings.npy')]
# dir_path = '/Users/chunwei/research/llm-scheduling/'
# embeddings = ['google-qa-length-sfr-embed.npy']
# exclude_files = ['google-qa-length-sfr-embed.npy']
print("Processing files:", embeddings)

for filename in embeddings:
    print("Processing file:", filename)
    full_path = os.path.join(dir_path, filename)
    s_time = time.time()
    encoded_prompts = load_encoded_prompts(full_path)
    read_time = time.time() - s_time
    LOG_NPY_READ.write(f"{filename}, {read_time}\n")
    print("shape of embeddings: ", encoded_prompts.shape)

    # Save the encoded prompts to a .npy file
    # Add suffix to the filename "duplicated,npy"
    full_path = os.path.join(dir_path, f"{os.path.splitext(filename)[0]}_duplicated.npy")
    s_time_write = time.time()
    np.save(full_path, encoded_prompts)
    write_time = time.time() - s_time_write
    LOG_NPY_WRITE.write(f"{filename}, {write_time}\n")

print("All files processed.")
LOG_NPY_READ.close()
LOG_NPY_WRITE.close()
