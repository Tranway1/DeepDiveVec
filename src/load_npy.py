import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as parquet
import pyarrow.orc as orc

# create log file to record the processing
LOG_FILE = open("batch_processing_value_log.csv", "a")
LOG_READ_FILE = open("batch_read_value_log.csv", "a")
LOG_NPY_SIZE = open("batch_npy_size_log.csv", "a")
def load_encoded_prompts(filename):
    # Load the encoded prompts from the .npy file
    encoded_prompts = np.load(filename, allow_pickle=True)
    print("Loaded encoded prompts from:", filename)
    return encoded_prompts

def convert_to_2d_numpy(tensor):
    if tensor.ndim == 3:
        # Find the dimension with the minimal length
        min_dim = np.argmin(tensor.shape)
        tensor = np.squeeze(tensor, axis=min_dim)
    elif tensor.ndim != 2:
        raise ValueError("Tensor must be 2D or 3D")
    return tensor


def save_data(df, dir_path, base_filename, format, compression=None, compression_level=None):
    filename_suffix = f"{compression}_{compression_level if compression_level is not None else 'N'}.{format}"
    filename = os.path.join(dir_path, f"{base_filename}_{filename_suffix.lower()}")
    print(f"dataframe shape: {df.shape}")
    # skip if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        # read the file back into dataframe and count the runtime, then print out the shape of the dataframe to make sure it matches the original
        start_time = time.time()
        if format == 'csv':
            return
        elif format == 'feather':
            df_read = feather.read_feather(filename)
        elif format == 'parquet':
            df_read = parquet.read_table(filename).to_pandas()
        elif format == 'orc':
            df_read = orc.read_table(filename).to_pandas()
        read_time = time.time() - start_time
        print(f"Read {format} from {filename}")
        print(f"Read time: {read_time}")
        file_size = os.path.getsize(filename)
        LOG_READ_FILE.write(
            f"{filename}, {format}, {compression}, {compression_level}, {file_size}, {read_time}, {df_read.shape}\n")
        return
    s_time = time.time()
    if format == 'csv':
        df.to_csv(filename)
    elif format == 'feather':
        feather.write_feather(df, filename, compression=compression, compression_level=compression_level)
    elif format == 'parquet':
        parquet.write_table(pa.Table.from_pandas(df), filename, compression=compression, compression_level=compression_level)
    elif format == 'orc':
        orc.write_table(pa.Table.from_pandas(df), filename, compression=compression)
    total_time = time.time()-s_time
    file_size = os.path.getsize(filename)
    # LOG_FILE.write(f"{filename}, {format}, {compression}, {compression_level}, {file_size}, {total_time}, {df.shape}\n")
    print(f"Saved {format} to {filename}")





# Check if at least one file name is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <file1> <file2> ...")
    sys.exit(1)
dir_path = '../embeddings/'
embeddings = sys.argv[1:]

# dir_path = '../embeddings/'
# embeddings = [f for f in os.listdir(dir_path) if f.endswith('embed.npy') or f.endswith('embeddings.npy')]
print("Processing files:", embeddings)


for filename in embeddings:
    print("Processing file:", filename)
    full_path = os.path.join(dir_path, filename)
    encoded_prompts = load_encoded_prompts(full_path)
    tensor_2d = convert_to_2d_numpy(encoded_prompts)
    df = pd.DataFrame(tensor_2d)
    base_filename = os.path.splitext(filename)[0]

    # Save to CSV with no compression
    save_data(df, dir_path, base_filename, 'csv', None)

    # Save to Feather with various compressions
    for compression in ['uncompressed', 'zstd', 'GZIP', 'SNAPPY', 'lz4', 'ZLIB']:
        try:
            if compression == 'ZSTD':
                compression_level = 1
                save_data(df, dir_path, base_filename, 'feather', compression, compression_level)
            save_data(df, dir_path, base_filename, 'feather', compression)
        except Exception as e:
            print(f"Error saving Feather with {compression} compression: {e}")


    # Save to Parquet with various compressions
    for compression in ['NONE', 'ZSTD', 'GZIP', 'SNAPPY', 'LZ4', 'ZLIB']:
        try:
            if compression == 'ZSTD':
                compression_level = 1
                save_data(df, dir_path, base_filename, 'parquet', compression, compression_level)
            save_data(df, dir_path, base_filename, 'parquet', compression)
        except Exception as e:
            print(f"Error saving Parquet with {compression} compression: {e}")

    # Save to ORC with various compressions
    for compression in ['UNCOMPRESSED', 'ZSTD', 'GZIP', 'SNAPPY', 'LZ4', 'ZLIB']:
        try:
            if compression == 'ZSTD':
                compression_level = 1
                save_data(df, dir_path, base_filename, 'orc', compression, compression_level)
            save_data(df, dir_path, base_filename, 'orc', compression)
        except Exception as e:
            print(f"Error saving ORC with {compression} compression: {e}")

print("All files processed.")
LOG_FILE.close()
LOG_READ_FILE.close()