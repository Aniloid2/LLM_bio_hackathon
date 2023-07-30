import os
import shutil

# source and destination directories
source_dir = '/home/brianformento/Dataset_BUSI_with_GT/malignant/'
destination_dir = '/home/brianformento/bio_llm/ultrasound/malignant/'
unwanted_word = 'mask'

# iterate over all the files in source directory
for filename in os.listdir(source_dir):
    # Check if the filename ends with .png
    if filename.endswith('.png'):
        # construct full file path
        source = os.path.join(source_dir, filename)
        destination = os.path.join(destination_dir, filename)

        # copy the file if it's not start with the unwanted_name
        if unwanted_word not in filename:
            shutil.copy2(source, destination)
            shutil.copy2(source, 'ultrasound')
    