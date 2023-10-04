import os
import numpy as np

# Specify the folder path you want to search
folder_path = 'C:\\Users\\njthakka\\Documents\\Flower'

# Initialize an empty list to store the absolute paths
absolute_paths = []

# Traverse through all files and subdirectories
for root, _, files in os.walk(folder_path):
    for file in files:
        # Get the absolute path of the file
        file_path = os.path.join(root, file)
        
        # Append the absolute path to the list
        absolute_paths.append(file_path)

# Convert the list to a numpy array if needed
absolute_paths_np = np.array(absolute_paths)

# Print or do whatever you need with the paths
print(absolute_paths_np)
absolute_paths_np = np.array(absolute_paths)

# Save the numpy array as a .npy file
output_file = 'train_files_age_pred.npy'
np.save(output_file, absolute_paths_np)

print(f"Absolute paths saved to {output_file}")
