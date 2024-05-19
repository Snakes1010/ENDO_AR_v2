import OpenEXR
import Imath
import numpy as np

def read_exr(file_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get the header information
    header = exr_file.header()
    
    # Get the data window
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Get the channels
    channels = header['channels'].keys()
    
    # Read the pixel data
    channel_data = {}
    for channel in channels:
        channel_str = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        channel_data[channel] = np.frombuffer(channel_str, dtype=np.float32).reshape((height, width))
    
    return width, height, channels, channel_data

# File paths
file_path1 = "/mnt/data/16_1_L.exr"
file_path2 = "/mnt/data/16_7_L.exr"

# Read and print information from the first file
width1, height1, channels1, channel_data1 = read_exr(file_path1)
print(f"File: {file_path1}")
print(f"Dimensions: {width1} x {height1}")
print(f"Channels: {channels1}")

# Read and print information from the second file
width2, height2, channels2, channel_data2 = read_exr(file_path2)
print(f"File: {file_path2}")
print(f"Dimensions: {width2} x {height2}")
print(f"Channels: {channels2}")
