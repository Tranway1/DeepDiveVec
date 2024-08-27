import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as parquet
import pyarrow.orc as orc

LOG_FILE = open("processing_vec_log.csv", "w")
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
    filename = os.path.join(dir_path, f"vec_{base_filename}_{filename_suffix.lower()}")
    # skip if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        return
    s_time = time.time()
    if format == 'csv':
        df_from_table = df.to_pandas()
        df_from_table.to_csv(filename)
    elif format == 'feather':
        feather.write_feather(df, filename, compression=compression, compression_level=compression_level)
    elif format == 'parquet':
        parquet.write_table(df, filename, compression=compression, compression_level=compression_level)
    elif format == 'orc':
        orc.write_table(df, filename, compression=compression)
    total_time = time.time()-s_time
    file_size = os.path.getsize(filename)
    LOG_FILE.write(f"{filename}, {format}, {compression}, {compression_level}, {file_size}, {total_time}\n")
    print(f"Saved {format} to {filename}")

dir_path = '/Users/chunwei/research/llm-scheduling/'
embeddings = [f for f in os.listdir(dir_path) if f.endswith('embed.npy') or f.endswith('embeddings.npy')]
print("Processing files:", embeddings)

for filename in embeddings:
    print("Processing file:", filename)
    full_path = os.path.join(dir_path, filename)
    encoded_prompts = load_encoded_prompts(full_path)
    print("shape of embeddings: ", encoded_prompts.shape)

    data_slices = [encoded_prompts[i].tolist() for i in range(encoded_prompts.shape[0])]
    print("shape of data_slices: ", len(data_slices))
    # Create DataFrame
    df = pd.DataFrame({
        'vectors': data_slices
    })

    # Convert to PyArrow Table
    table = pa.Table.from_pandas(df)
    print("table: ", table)

    base_filename = os.path.splitext(filename)[0]

    # # Save to CSV with no compression
    # save_data(table, dir_path, base_filename, 'csv', None)

    # Save to Feather with various compressions
    for compression in ['uncompressed', 'zstd', 'GZIP', 'SNAPPY', 'lz4', 'ZLIB']:
        try:
            if compression == 'ZSTD':
                compression_level = 1
                save_data(table, dir_path, base_filename, 'feather', compression, compression_level)
            save_data(table, dir_path, base_filename, 'feather', compression)
        except Exception as e:
            print(f"Error saving Feather with {compression} compression: {e}")


    # Save to Parquet with various compressions
    for compression in ['NONE', 'ZSTD', 'GZIP', 'SNAPPY', 'LZ4', 'ZLIB']:
        try:
            if compression == 'ZSTD':
                compression_level = 1
                save_data(table, dir_path, base_filename, 'parquet', compression, compression_level)
            save_data(table, dir_path, base_filename, 'parquet', compression)
        except Exception as e:
            print(f"Error saving Parquet with {compression} compression: {e}")

    # Save to ORC with various compressions
    for compression in ['UNCOMPRESSED', 'ZSTD', 'GZIP', 'SNAPPY', 'LZ4', 'ZLIB']:
        try:
            if compression == 'ZSTD':
                compression_level = 1
                save_data(table, dir_path, base_filename, 'orc', compression, compression_level)
            save_data(table, dir_path, base_filename, 'orc', compression)
        except Exception as e:
            print(f"Error saving ORC with {compression} compression: {e}")

print("All files processed.")