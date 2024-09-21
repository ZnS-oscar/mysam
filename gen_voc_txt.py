import os

# Define the root folder
root_folder = "data_zoo/imagenet1k/"

# Define the output file
output_file = "labels.txt"

# Open the output file in write mode
with open(output_file, "w") as f:
    # Walk through each child folder
    for root, dirs, files in os.walk(root_folder):
        # If there are files in the folder
        # if files:
        #     # Sort and take the first 50 files
        #     files = sorted(files)
        dirs = sorted(dirs)
            # Write the relative paths to the output file
        for dir in dirs:
            if dir.startswith('n'):
            # Get the relative path to the root folder
            # if not file.endswith('.xml'):
            #     continue
            # relative_path = os.path.relpath(os.path.join(root, file), root_folder)
            # f.write(relative_path + "\n")
                f.write(dir+"\n")

print(f"File paths saved to {output_file}")
