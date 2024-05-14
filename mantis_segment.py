import os
import cv2

def on_trackbar(val):
    pass

def capture_video(device_index_1=0, device_index_2=1, save_folder_left='', save_folder_right=''):   
    cap_1 = cv2.VideoCapture(device_index_1)
    cap_2 = cv2.VideoCapture(device_index_2)
    cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap_1.isOpened() or not cap_2.isOpened():
        print("Error: Could not open video devices.")
        return
    
    os.makedirs(save_folder_left, exist_ok=True)
    os.makedirs(save_folder_right, exist_ok=True)
    
    # Create a window for the sliders
    cv2.namedWindow('Settings')
    # Create trackbars for color change
    cv2.createTrackbar('Lower H', 'Settings', 40, 179, on_trackbar)
    cv2.createTrackbar('Lower S', 'Settings', 40, 255, on_trackbar)
    cv2.createTrackbar('Lower V', 'Settings', 40, 255, on_trackbar)
    cv2.createTrackbar('Upper H', 'Settings', 80, 179, on_trackbar)
    cv2.createTrackbar('Upper S', 'Settings', 255, 255, on_trackbar)
    cv2.createTrackbar('Upper V', 'Settings', 255, 255, on_trackbar)
    
    counter = 0
    
    while True:
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()
        if not ret_1 or not ret_2:
            print("Error: Failed to capture frame.")
            break
        
        # Get current positions of the trackbars
        l_h = cv2.getTrackbarPos('Lower H', 'Settings')
        l_s = cv2.getTrackbarPos('Lower S', 'Settings')
        l_v = cv2.getTrackbarPos('Lower V', 'Settings')
        u_h = cv2.getTrackbarPos('Upper H', 'Settings')
        u_s = cv2.getTrackbarPos('Upper S', 'Settings')
        u_v = cv2.getTrackbarPos('Upper V', 'Settings')
        
        lower_green = (l_h, l_s, l_v)
        upper_green = (u_h, u_s, u_v)
        
        hsv_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2HSV)
        hsv_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2HSV)

        mask_1 = cv2.inRange(hsv_1, lower_green, upper_green)
        mask_2 = cv2.inRange(hsv_2, lower_green, upper_green)

        result_1 = cv2.bitwise_and(frame_1, frame_1, mask=~mask_1)
        result_2 = cv2.bitwise_and(frame_2, frame_2, mask=~mask_2)

        frame_concat = cv2.hconcat([result_1, result_2])
        cv2.imshow('Video Capture', frame_concat)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            cv2.imwrite(os.path.join(save_folder_left, f'left_{counter}.jpg'), result_1)
            cv2.imwrite(os.path.join(save_folder_right, f'right_{counter}.jpg'), result_2)
            print(f"Images saved as left_{counter}.jpg and right_{counter}.jpg")
            counter += 1

    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device_index_1 = 0
    device_index_2 = 1
    save_folder_left = "/media/jacob/Viper4TB/mantis_lab_tests/may9_microscopeL_focalplus10x_0001/"
    save_folder_right = "/media/jacob/Viper4TB/mantis_lab_tests/may9_microscopeR_focalplus10x_0001/"
    capture_video(device_index_1, device_index_2, save_folder_left, save_folder_right)

# import os
# import cv2

# def on_trackbar(val):
#     pass

# def capture_video(device_index_1=0, device_index_2=1, save_folder_left='', save_folder_right=''):
#     cap_1 = cv2.VideoCapture(device_index_1)
#     cap_2 = cv2.VideoCapture(device_index_2)
#     cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#     cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
#     if not cap_1.isOpened() or not cap_2.isOpened():
#         print("Error: Could not open video devices.")
#         return
    
#     os.makedirs(save_folder_left, exist_ok=True)
#     os.makedirs(save_folder_right, exist_ok=True)
    
#     cv2.namedWindow('Settings')
#     # Create trackbars for HSV color change
#     cv2.createTrackbar('Lower H', 'Settings', 0, 179, on_trackbar)
#     cv2.createTrackbar('Lower S', 'Settings', 0, 255, on_trackbar)
#     cv2.createTrackbar('Lower V', 'Settings', 0, 255, on_trackbar)
#     cv2.createTrackbar('Upper H', 'Settings', 179, 179, on_trackbar)
#     cv2.createTrackbar('Upper S', 'Settings', 255, 255, on_trackbar)
#     cv2.createTrackbar('Upper V', 'Settings', 255, 255, on_trackbar)
    
#     counter = 0
    
#     while True:
#         ret_1, frame_1 = cap_1.read()
#         ret_2, frame_2 = cap_2.read()
#         if not ret_1 or not ret_2:
#             print("Error: Failed to capture frame.")
#             break
        
#         # Get current positions of the trackbars for HSV
#         l_h = cv2.getTrackbarPos('Lower H', 'Settings')
#         l_s = cv2.getTrackbarPos('Lower S', 'Settings')
#         l_v = cv2.getTrackbarPos('Lower V', 'Settings')
#         u_h = cv2.getTrackbarPos('Upper H', 'Settings')
#         u_s = cv2.getTrackbarPos('Upper S', 'Settings')
#         u_v = cv2.getTrackbarPos('Upper V', 'Settings')
        
#         lower_hsv = (l_h, l_s, l_v)
#         upper_hsv = (u_h, u_s, u_v)
        
#         hsv_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2HSV)
#         hsv_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2HSV)

#         mask_1 = cv2.inRange(hsv_1, lower_hsv, upper_hsv)
#         mask_2 = cv2.inRange(hsv_2, lower_hsv, upper_hsv)

#         # Morphological operations to remove small objects
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         clean_mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE, kernel)
#         clean_mask_2 = cv2.morphologyEx(mask_2, cv2.MORPH_CLOSE, kernel)

#         # Find contours and select the largest one
#         contours_1, _ = cv2.findContours(clean_mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours_2, _ = cv2.findContours(clean_mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours_1:
#             largest_contour_1 = max(contours_1, key=cv2.contourArea)
#             cv2.drawContours(frame_1, [largest_contour_1], -1, (0, 255, 0), 3)
#         if contours_2:
#             largest_contour_2 = max(contours_2, key=cv2.contourArea)
#             cv2.drawContours(frame_2, [largest_contour_2], -1, (0, 255, 0), 3)

#         result_1 = cv2.bitwise_and(frame_1, frame_1, mask=clean_mask_1)
#         result_2 = cv2.bit


# import os
# import cv2


# def capture_video(device_index_1=0, device_index_2=1, save_folder_left='', save_folder_right=''):
#     cap_1 = cv2.VideoCapture(device_index_1)
#     cap_2 = cv2.VideoCapture(device_index_2)
#     cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#     cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
#     if not cap_1.isOpened() or not cap_2.isOpened():
#         print("Error: Could not open video devices.")
#         return
    
#     os.makedirs(save_folder_left, exist_ok=True)
#     os.makedirs(save_folder_right, exist_ok=True)
    
#     counter = 0
    
#     while True:
#         ret_1, frame_1 = cap_1.read()
#         ret_2, frame_2 = cap_2.read()
#         if not ret_1 or not ret_2:
#             print("Error: Failed to capture frame.")
#             break
        
      
#         lower_hsv = (0,0,76)
#         upper_hsv = (69,255,255)
        
#         hsv_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2HSV)
#         hsv_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2HSV)

#         mask_1 = cv2.inRange(hsv_1, lower_hsv, upper_hsv)
#         mask_2 = cv2.inRange(hsv_2, lower_hsv, upper_hsv)

#         # Morphological operations to remove small objects
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 3))
#         clean_mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE, kernel)
#         clean_mask_2 = cv2.morphologyEx(mask_2, cv2.MORPH_CLOSE, kernel)

#         # Find contours and select the largest one
#         contours_1, _ = cv2.findContours(clean_mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours_2, _ = cv2.findContours(clean_mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours_1:
#             largest_contour_1 = max(contours_1, key=cv2.contourArea)
#             cv2.drawContours(frame_1, [largest_contour_1], -1, (0, 255, 0), 3)
#         if contours_2:
#             largest_contour_2 = max(contours_2, key=cv2.contourArea)
#             cv2.drawContours(frame_2, [largest_contour_2], -1, (0, 255, 0), 3)

#         result_1 = cv2.bitwise_and(frame_1, frame_1, mask=clean_mask_1)
#         result_2 = cv2.bitwise_and(frame_2, frame_2, mask=clean_mask_2)

#         frame_concat = cv2.hconcat([result_1, result_2])
#         cv2.imshow('Video Capture', frame_concat)
        
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q'):
#             break
#         elif key & 0xFF == ord('s'):
#             cv2.imwrite(os.path.join(save_folder_left, f'left_{counter}.jpg'), result_1)
#             cv2.imwrite(os.path.join(save_folder_right, f'right_{counter}.jpg'), result_2)
#             print(f"Images saved as left_{counter}.jpg and right_{counter}.jpg")
#             counter += 1

#     cap_1.release()
#     cap_2.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     device_index_1 = 0
#     device_index_2 = 1
#     save_folder_left = "/media/jacob/Viper4TB/mantis_lab_tests/may9_microscopeL_focalplus10x_0001/"
#     save_folder_right = "/media/jacob/Viper4TB/mantis_lab_tests/may9_microscopeR_focalplus10x_0001/"
#     capture_video(device_index_1, device_index_2, save_folder_left, save_folder_right)
