import os
import pandas as pd


def get_file_size(filepath):
    """ Returns the file size if file exists, otherwise returns 0. """
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0


def collect_file_sizes(dir_path, setup):
    # List all files in the directory
    all_files = os.listdir(dir_path)

    # Prepare a dictionary to hold the data
    data = {}

    # Define the compression settings and formats to check
    compressions = {
        'csv': ['none'],  # CSV files are named with 'none' as per your example
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

    # Process each file
    for file in all_files:
        if file.endswith('embed.npy'):
            base_filename = os.path.splitext(file)[0]
            row = {'filename': base_filename}

            # Original .npy file size
            row['original_npy'] = get_file_size(os.path.join(dir_path, file))

            # Check each format and compression
            for fmt in compressions:
                for comp in compressions[fmt]:
                    if fmt == 'csv':
                        # Special handling for CSV to match the naming convention
                        filename = f"{base_filename}_{comp}_n.{fmt}"
                    else:
                        filename = f"{base_filename}_{comp}_n.{fmt}"
                    filename = f"{setup}{filename}"
                    filepath = os.path.join(dir_path, filename)
                    row[f"{comp}_{fmt}"] = get_file_size(filepath)

                    # Special case for zstd with level 1
                    if comp == 'zstd':
                        filename = f"{base_filename}_{comp}_1.{fmt}"
                        filename = f"{setup}{filename}"
                        filepath = os.path.join(dir_path, filename)
                        row[f"{comp}_1_{fmt}"] = get_file_size(filepath)

            # Add the row to the data dictionary
            data[base_filename] = row

    # Create a DataFrame from the collected data
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)

    return df


def save_sizes_to_csv(df, output_path):
    """ Saves the DataFrame to a CSV file. """
    df.to_csv(output_path, index=False)
    print(f"File sizes saved to {output_path}")

setup = "vec_"
# setup = ""
dir_path = '/Users/chunwei/research/llm-scheduling/'
output_csv_path = f'/Users/chunwei/research/llm-scheduling/{setup}file_sizes.csv'

# Collect file sizes
file_sizes_df = collect_file_sizes(dir_path, setup)

# Save to CSV
save_sizes_to_csv(file_sizes_df, output_csv_path)