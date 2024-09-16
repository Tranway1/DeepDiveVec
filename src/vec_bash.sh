#!/bin/bash

# Define the directory path
dir_path="../embeddings/"

# Check if the directory exists
if [ ! -d "$dir_path" ]; then
    echo "Directory $dir_path does not exist."
    exit 1
fi

# Loop through files in the directory
for full_path in "$dir_path"/*; do
    # Extract just the filename from the path
    file=$(basename "$full_path")

    # Check if the file ends with 'embed.npy' or 'embeddings.npy'
    if [[ $file == *embed.npy || $file == *embeddings.npy ]]; then
        echo "Processing $file"
        # Call the Python script with just the file name
#        python load_npy_vec.py "$file"
#        python load_npy.py "$file"
        python read_write_npy_vec.py "$file"
    fi
done