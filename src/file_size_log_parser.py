import os
import pandas as pd

def get_file_size(filepath):
    """ Returns the file size if file exists, otherwise returns 0. """
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0

def read_log_file(log_path):
    """ Reads the log file and returns a DataFrame. """
    columns = ['filepath', 'format', 'compression', 'level', 'file_size', 'runtime', 'shape']
    df = pd.read_csv(log_path, names=columns)
    return df

def collect_file_stats(log_df, dir_path, setup, stat_type='size'):
    """ Collects file statistics based on the log DataFrame. """
    # Prepare a dictionary to hold the data
    data = {}
    # Define the compression settings and formats to check
    compressions = {
        'csv': ['none'],
        'feather': ['uncompressed', 'zstd', 'gzip', 'snappy', 'lz4', 'zlib'],
        'parquet': ['none', 'zstd', 'gzip', 'snappy', 'lz4', 'zlib'],
        'orc': ['uncompressed', 'zstd', 'gzip', 'snappy', 'lz4', 'zlib']
    }
    # Initialize the DataFrame columns
    columns = ['filename', 'original_npy']
    for fmt in compressions:
        for comp in compressions[fmt]:
            columns.append(f"{comp}_{fmt}")
            if comp == 'zstd':
                columns.append(f"{comp}_1_{fmt}")

    # Process each row in the log DataFrame
    for _, row in log_df.iterrows():
        base_filename = os.path.splitext(os.path.basename(row['filepath']))[0]
        if base_filename.endswith('embed'):
            if base_filename not in data:
                # Get the size of the original .npy file
                npy_filepath = os.path.join(dir_path, f"{base_filename}.npy")
                data[base_filename] = {'filename': base_filename, 'original_npy': get_file_size(npy_filepath)}
            comp = row['compression'].lower() if pd.notna(row['compression']) else 'none'
            fmt = row['format']
            key = f"{comp}_{fmt}"
            if pd.notna(row['level']) and row['level'] == 1:
                key = f"{comp}_1_{fmt}"
            data[base_filename][key] = row['file_size'] if stat_type == 'size' else row['runtime']

    # Create a DataFrame from the collected data
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    return df

def save_stats_to_csv(df, output_path):
    """ Saves the DataFrame to a CSV file. """
    df.to_csv(output_path, index=False)
    print(f"File stats saved to {output_path}")

# Configuration
setup = "vec_"
dir_path = '../embeddings/'
log_path = f'batch_processing_{setup}log.csv'
output_csv_path = f'{dir_path}{setup}file_stats.csv'

# Read log file
log_df = read_log_file(log_path)

# Collect file stats
file_stats_df = collect_file_stats(log_df, dir_path, setup, stat_type='size')  # Change 'size' to 'runtime' if needed

# Save to CSV
save_stats_to_csv(file_stats_df, output_csv_path)