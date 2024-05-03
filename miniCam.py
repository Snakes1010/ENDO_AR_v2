import os
import cv2

def capture_video(device_index_1=0, device_index_2=1, save_folder_left='', save_folder_right=''):
    cap_1 = cv2.VideoCapture(device_index_1)
    cap_2 = cv2.VideoCapture(device_index_2)
    
    if not cap_1.isOpened() or not cap_2.isOpened():
        print("Error: Could not open video devices.")
        return
    
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder_left, exist_ok=True)
    os.makedirs(save_folder_right, exist_ok=True)
    
    counter = 0
    
    while True:
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()
        if not ret_1 or not ret_2:
            print("Error: Failed to capture frame.")
            break
        
        # Resize frames to have the same height (assuming they already have the same width)
        height = min(frame_1.shape[0], frame_2.shape[0])
        frame_1 = cv2.resize(frame_1, (int(frame_1.shape[1] * height / frame_1.shape[0]), height))
        frame_2 = cv2.resize(frame_2, (int(frame_2.shape[1] * height / frame_2.shape[0]), height))
        
        # Concatenate frames horizontally
        frame_concat = cv2.hconcat([frame_1, frame_2])
        
        cv2.imshow('Video Capture', frame_concat)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            cv2.imwrite(os.path.join(save_folder_left, f'left_{counter}.jpg'), frame_1)
            cv2.imwrite(os.path.join(save_folder_right, f'right_{counter}.jpg'), frame_2)
            print(f"Images saved as left_{counter}.jpg and right_{counter}.jpg")
            counter += 1

    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the device indices of your Magewell capture card inputs
    # You may need to change these depending on the system
    device_index_1 = 0
    device_index_2 = 1
    # Specify the folder where you want to save the images
    save_folder_left = "/media/jacob/Viper4TB/minicam_calibration/april30_left_sq/"
    save_folder_right = "/media/jacob/Viper4TB/minicam_calibration/april30_right_sq/"
    capture_video(device_index_1, device_index_2, save_folder_left, save_folder_right)