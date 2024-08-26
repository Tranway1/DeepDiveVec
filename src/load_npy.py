import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as parquet
import pyarrow.orc as orc

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
    # skip if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        return
    if format == 'csv':
        df.to_csv(filename)
    elif format == 'feather':
        feather.write_feather(df, filename, compression=compression, compression_level=compression_level)
    elif format == 'parquet':
        parquet.write_table(pa.Table.from_pandas(df), filename, compression=compression, compression_level=compression_level)
    elif format == 'orc':
        orc.write_table(pa.Table.from_pandas(df), filename, compression=compression)
    print(f"Saved {format} to {filename}")

dir_path = '/Users/chunwei/research/llm-scheduling/'
embeddings = [f for f in os.listdir(dir_path) if f.endswith('embed.npy')]
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