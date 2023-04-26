import yaml
import AR_functions
import shutil
import os

dir_name = "/home/jacob/endo_calib/ENDO_AR/left_calib_one"
data_1 = AR_functions.import_yaml('8x11_L.yaml')


# Create the destination directory if it doesn't exist
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Copy each file to the destination directory
for file_path in data_1:
    shutil.copy(file_path, dir_name)