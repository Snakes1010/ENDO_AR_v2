import cv2
import matplotlib.pyplot as plt
import numpy as np

# Open the video file
cap = cv2.VideoCapture('/home/jacob/endo_calib/low_cost_proj/8_11_2x/laser_L.mp4')

# Read the first frame
ret, avg_img = cap.read()

# Check if a frame was properly read
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Convert the average to float data type
avg_img = avg_img.astype('float')

while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Accumulate the frames
    avg_img += frame

# Calculate the average
avg_img /= cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Convert back to 8-bit format for displaying
avg_img = avg_img.astype('uint8')

if len(avg_img.shape) != 3 or avg_img.shape[2] != 3:
    print("image should be in color")
    exit()

b, g, r = cv2.split(avg_img)

# Flatten the 2D image arrays into 1D arrays
r = r.flatten()
g = g.flatten()

# Create a 2D histogram using hist2d
plt.hist2d(r, g, bins=30, cmap='plasma')

# Include a colorbar to show the count scale
plt.colorbar(label='count in bin')

plt.show()


# Show the average image
cv2.imshow('Average Image', avg_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture object
cap.release()
