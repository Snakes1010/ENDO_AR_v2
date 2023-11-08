import shutil

# Path to the directory you want to compress
directory_to_zip = "/home/jacob/endo_calib/low_cost_proj/8_11_2x/6_29_stereoframesR"

# Output filename
output_filename = "6_29_stereoframes"

# Create a ZipFile
shutil.make_archive(output_filename, 'zip', directory_to_zip)

print("Directory zipped successfully.")